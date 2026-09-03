#![cfg(feature = "unstable_protocol_v2")]

//! Integration tests for steering via the custom `_session/steering` method.
//!
//! The agent advertises `_meta.steering.supported = true` from `initialize` and
//! accepts `_session/steering` requests. When no turn is running, steering
//! starts a fresh prompt turn; the injected text is folded into the turn by the
//! engine's steering inbox.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use acp_agent::{build_agent, AgentConfig, AgentDeps, NextTurnFactory};
use agent_client_protocol::schema::v2::{
    ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest, SessionId, SessionUpdate,
    UpdateSessionNotification,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{on_receive_notification, Client, JsonRpcRequest};
use agent_core::{
    NextTurnService, Result as CoreResult, StopReason as CoreStopReason, ToolRegistry, TurnContext,
    TurnEvent, TurnObservation, TurnObserver,
};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

mod common;

/// Client-side mirror of the agent's steering request, so tests can send the
/// custom `_session/steering` method with the same wire shape.
#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "_session/steering", response = serde_json::Value)]
struct TestSteeringRequest {
    #[serde(rename = "sessionId")]
    session_id: SessionId,
    #[serde(default)]
    prompt: Vec<ContentBlock>,
}

type Updates = Arc<Mutex<Vec<SessionUpdate>>>;

#[derive(Default)]
struct RequestRecorder(Mutex<Vec<TurnObservation>>);

impl TurnObserver for RequestRecorder {
    fn on_turn_request(&self, observation: &TurnObservation) {
        self.0.lock().unwrap().push(observation.clone());
    }
}

struct BlockingTurnService {
    release: Arc<Notify>,
    calls: AtomicUsize,
}

#[async_trait]
impl NextTurnService for BlockingTurnService {
    async fn get_next_turn_streaming(
        &self,
        _ctx: &TurnContext,
    ) -> CoreResult<BoxStream<'static, TurnEvent>> {
        if self.calls.fetch_add(1, Ordering::SeqCst) > 0 {
            return Ok(Box::pin(stream::iter([TurnEvent::TurnFinished {
                stop_reason: CoreStopReason::EndTurn,
            }])));
        }
        let release = self.release.clone();
        Ok(Box::pin(stream::unfold(0_u8, move |step| {
            let release = release.clone();
            async move {
                match step {
                    0 => Some((TurnEvent::TextDelta("started".to_string()), 1)),
                    1 => {
                        release.notified().await;
                        Some((
                            TurnEvent::TurnFinished {
                                stop_reason: CoreStopReason::EndTurn,
                            },
                            2,
                        ))
                    }
                    _ => None,
                }
            }
        })))
    }
}

fn deps_with_script(script: Vec<Vec<TurnEvent>>) -> AgentDeps {
    let factory: NextTurnFactory = Arc::new(move |_selection| {
        let service: Arc<dyn NextTurnService> =
            Arc::new(turn_replay::ReplayTurnService::new(script.clone()));
        service
    });
    AgentDeps {
        next_turn_factory: factory,
        tools: ToolRegistry::new(),
        config: AgentConfig {
            require_submit_result: false,
            ..AgentConfig::default()
        },
        store: acp_agent::default_store(),
        turn_observer: None,
        inspector_base_url: None,
    }
}

#[tokio::test]
async fn initialize_advertises_steering_capability() {
    let deps = deps_with_script(vec![vec![]]);
    let agent = build_agent(deps);

    let initialize_response = Client
        .v2()
        .name("test-client")
        .connect_with(agent, async move |cx| {
            let init = cx
                .send_request(InitializeRequest::new(
                    ProtocolVersion::V2,
                    agent_client_protocol::schema::v2::Implementation::new("test-client", "0.1"),
                ))
                .block_task()
                .await?;
            Ok(serde_json::to_value(init).expect("initialize response should serialize"))
        })
        .await
        .expect("connection completed");

    assert_eq!(
        initialize_response["_meta"]["steering"]["supported"],
        serde_json::json!(true),
    );
    assert!(initialize_response["capabilities"]["session"].is_object());
}

#[tokio::test]
async fn steering_into_active_session_returns_injected() {
    let release = Arc::new(Notify::new());
    let factory: NextTurnFactory = Arc::new({
        let release = release.clone();
        move |_selection| {
            Arc::new(BlockingTurnService {
                release: release.clone(),
                calls: AtomicUsize::new(0),
            })
        }
    });
    let deps = AgentDeps::new(
        factory,
        ToolRegistry::new(),
        AgentConfig {
            require_submit_result: false,
            ..AgentConfig::default()
        },
        acp_agent::default_store(),
    );
    let first_chunk_seen = Arc::new(Notify::new());
    let notify_first = first_chunk_seen.clone();
    let updates: common::Updates = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    let completion = updates.clone();

    let response = tokio::time::timeout(
        Duration::from_secs(5),
        Client
            .v2()
            .name("test-client")
            .on_receive_notification(
                async move |notification: UpdateSessionNotification, _cx| {
                    if matches!(
                        &notification.update,
                        SessionUpdate::AgentMessageChunk(chunk)
                            if matches!(&chunk.content, ContentBlock::Text(text) if text.text == "started")
                    ) {
                        notify_first.notify_one();
                    }
                    collected.lock().unwrap().push(notification.update);
                    Ok(())
                },
                on_receive_notification!(),
            )
            .connect_with(build_agent(deps), async move |cx| {
                cx.send_request(InitializeRequest::new(ProtocolVersion::V2, agent_client_protocol::schema::v2::Implementation::new("test-client", "0.1")))
                    .block_task()
                    .await?;
                let session = cx
                    .send_request(NewSessionRequest::new("/tmp"))
                    .block_task()
                    .await?;
                cx.send_request(PromptRequest::new(
                    session.session_id.clone(),
                    vec![ContentBlock::from("initial")],
                ))
                .block_task()
                .await?;

                first_chunk_seen.notified().await;

                let response = cx
                    .send_request(TestSteeringRequest {
                        session_id: session.session_id,
                        prompt: vec![ContentBlock::from("follow-up")],
                    })
                    .block_task()
                    .await?;
                release.notify_one();
                common::wait_for_idle(&completion).await;
                Ok(response)
            }),
    )
    .await
    .expect("steering test timed out")
    .expect("connection completed");

    assert_eq!(response, serde_json::json!({ "outcome": "injected" }));
}

#[tokio::test]
async fn steering_into_idle_session_starts_new_turn() {
    let script = vec![vec![
        TurnEvent::TextDelta("steered".to_string()),
        TurnEvent::TurnFinished {
            stop_reason: CoreStopReason::EndTurn,
        },
    ]];
    let observer = Arc::new(RequestRecorder::default());
    let mut deps = deps_with_script(script);
    deps.turn_observer = Some(observer.clone());
    let agent = build_agent(deps);

    let updates: Updates = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    let completion = updates.clone();

    let response = Client
        .v2()
        .name("test-client")
        .on_receive_notification(
            async move |notif: UpdateSessionNotification, _cx| {
                collected.lock().unwrap().push(notif.update);
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(
                ProtocolVersion::V2,
                agent_client_protocol::schema::v2::Implementation::new("test-client", "0.1"),
            ))
            .block_task()
            .await?;
            let new_session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            // No prompt yet: steering should start a fresh turn.
            let resp = cx
                .send_request(TestSteeringRequest {
                    session_id: new_session.session_id.clone(),
                    prompt: vec![ContentBlock::from("go")],
                })
                .block_task()
                .await?;
            common::wait_for_idle(&completion).await;
            Ok(resp)
        })
        .await
        .expect("connection completed");

    assert_eq!(response, serde_json::json!({ "outcome": "startedNewTurn" }));

    let chunks: Vec<String> = updates
        .lock()
        .unwrap()
        .iter()
        .filter_map(|update| match update {
            SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                ContentBlock::Text(text) => Some(text.text.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert!(
        chunks.contains(&"steered".to_string()),
        "expected the steered turn to stream its output, got {chunks:?}"
    );
    assert_eq!(observer.0.lock().unwrap()[0].workspace_roots, vec!["/tmp"]);
}
