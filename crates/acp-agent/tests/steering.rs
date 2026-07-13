//! Integration tests for steering via the custom `_session/inject` method.
//!
//! The agent advertises an `inject` meta capability from `initialize` and
//! accepts `_session/inject` requests. When no turn is running, an injection
//! starts a fresh prompt turn; the injected text is folded into the turn by the
//! engine's steering inbox.

use std::sync::{Arc, Mutex};

use acp_agent::{build_agent, AgentConfig, AgentDeps, NextTurnFactory};
use agent_client_protocol::schema::v1::{
    ContentBlock, InitializeRequest, NewSessionRequest, SessionId, SessionNotification,
    SessionUpdate,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{
    on_receive_notification, Client, JsonRpcRequest, MetaCapability, MetaCapabilityExt,
};
use agent_core::{NextTurnService, StopReason as CoreStopReason, ToolRegistry, TurnEvent};
use serde::{Deserialize, Serialize};

/// Client-side mirror of the agent's steering request, so tests can send the
/// custom `_session/inject` method with the same wire shape.
#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "_session/inject", response = serde_json::Value)]
struct TestInjectRequest {
    #[serde(rename = "sessionId")]
    session_id: SessionId,
    #[serde(default)]
    prompt: Vec<ContentBlock>,
}

/// The `inject` capability, for asserting it is advertised in `initialize`.
struct InjectCapability;
impl MetaCapability for InjectCapability {
    fn key(&self) -> &'static str {
        "inject"
    }
}

type Updates = Arc<Mutex<Vec<SessionUpdate>>>;

fn deps_with_script(script: Vec<Vec<TurnEvent>>) -> AgentDeps {
    let factory: NextTurnFactory = Arc::new(move |_selection| {
        let service: Arc<dyn NextTurnService> =
            Arc::new(turn_replay::ReplayTurnService::new(script.clone()));
        service
    });
    AgentDeps {
        next_turn_factory: factory,
        tools: ToolRegistry::new(),
        config: AgentConfig::default(),
    }
}

#[tokio::test]
async fn initialize_advertises_inject_capability() {
    let deps = deps_with_script(vec![vec![]]);
    let agent = build_agent(deps);

    let advertised = Client
        .builder()
        .name("test-client")
        .connect_with(agent, async move |cx| {
            let init = cx
                .send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            Ok(init.has_meta_capability(InjectCapability))
        })
        .await
        .expect("connection completed");

    assert!(advertised, "agent should advertise the inject capability");
}

#[tokio::test]
async fn inject_into_idle_session_starts_new_turn() {
    let script = vec![vec![
        TurnEvent::TextDelta("steered".to_string()),
        TurnEvent::TurnFinished {
            stop_reason: CoreStopReason::EndTurn,
        },
    ]];
    let deps = deps_with_script(script);
    let agent = build_agent(deps);

    let updates: Updates = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();

    let response = Client
        .builder()
        .name("test-client")
        .on_receive_notification(
            async move |notif: SessionNotification, _cx| {
                collected.lock().unwrap().push(notif.update);
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let new_session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            // No prompt yet: injecting should start a fresh turn.
            let resp = cx
                .send_request(TestInjectRequest {
                    session_id: new_session.session_id.clone(),
                    prompt: vec![ContentBlock::from("go")],
                })
                .block_task()
                .await?;
            Ok(resp)
        })
        .await
        .expect("connection completed");

    assert_eq!(response["received"], serde_json::json!(true));
    assert_eq!(response["startedNewTurn"], serde_json::json!(true));

    // Allow the spawned turn to stream its output.
    for _ in 0..50 {
        if !updates.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

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
}
