//! End-to-end test (Step 15): a prompt in `orchestrate` mode runs the
//! orchestration pipeline, accumulating each sub-agent result as one assistant
//! message and returning a stop reason. Driven through an in-process fake client with
//! replay backends (no network).
//!
//! Cancellation of an in-flight orchestrated prompt is handled by the same
//! shared [`agent_core::CancellationToken`] threaded through `run_pipeline`
//! into the orchestrator turn and its sub-agents (the chat-mode cancellation
//! path is covered in `e2e_prompt.rs`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v1::{
    CancelNotification, ContentBlock as AcpContentBlock, InitializeRequest, NewSessionRequest,
    PromptRequest, SessionNotification, SessionUpdate, SetSessionModeRequest, StopReason,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{on_receive_notification, Client, JsonRpcRequest};
use agent_core::{
    ContentBlock as HistoryContentBlock, HistoryEntry, NextTurnService, Result as CoreResult,
    StopReason as CoreStopReason, ToolRegistry, TurnContext, TurnEvent, TurnObservation,
    TurnObserver,
};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use serde::{Deserialize, Serialize};

use acp_agent::{build_agent, default_store, AgentConfig, AgentDeps, NextTurnFactory};

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "_session/steering", response = serde_json::Value)]
struct TestSteeringRequest {
    #[serde(rename = "sessionId")]
    session_id: agent_client_protocol::schema::v1::SessionId,
    #[serde(default)]
    prompt: Vec<AcpContentBlock>,
}

/// A factory whose first invocation yields the orchestrator script and every
/// later invocation yields the sub-agent script. `run_pipeline` calls the
/// factory once for the orchestrator backend, then once per sub-agent tier
/// resolution, so this cleanly separates the two roles for the replay backend.
fn staged_factory() -> NextTurnFactory {
    let calls = Arc::new(AtomicUsize::new(0));
    Arc::new(move |_selection| {
        let n = calls.fetch_add(1, Ordering::SeqCst);
        let script = if n == 0 {
            // Orchestrator: call run_subagent(code, step 1), then finish.
            vec![
                vec![
                    TurnEvent::ToolCallRequested {
                        id: agent_core::ToolCallId::new("o1"),
                        name: "run_subagent".into(),
                        arguments: serde_json::json!({ "mode": "code", "step_index": 1 }),
                    },
                    TurnEvent::TurnFinished {
                        stop_reason: CoreStopReason::ToolUse,
                    },
                ],
                vec![TurnEvent::TurnFinished {
                    stop_reason: CoreStopReason::EndTurn,
                }],
            ]
        } else {
            // Sub-agent: submit immediately, then finish.
            vec![
                vec![
                    TurnEvent::ToolCallRequested {
                        id: agent_core::ToolCallId::new("s1"),
                        name: "submit".into(),
                        arguments: serde_json::json!({ "output": "did the work" }),
                    },
                    TurnEvent::TurnFinished {
                        stop_reason: CoreStopReason::ToolUse,
                    },
                ],
                vec![TurnEvent::TurnFinished {
                    stop_reason: CoreStopReason::EndTurn,
                }],
            ]
        };
        let service: Arc<dyn NextTurnService> =
            Arc::new(turn_replay::ReplayTurnService::new(script));
        service
    })
}

fn orchestrate_deps() -> AgentDeps {
    // Start sessions in orchestrate mode so the prompt runs the pipeline.
    let config = AgentConfig {
        enable_orchestrated: true,
        ..AgentConfig::default()
    };
    AgentDeps {
        next_turn_factory: staged_factory(),
        tools: ToolRegistry::new(),
        config,
        store: default_store(),
        turn_observer: None,
        inspector_base_url: None,
    }
}

#[derive(Default)]
struct WorkspaceObserver(Mutex<Vec<TurnObservation>>);

impl TurnObserver for WorkspaceObserver {
    fn on_turn_request(&self, observation: &TurnObservation) {
        self.0.lock().unwrap().push(observation.clone());
    }
}

#[tokio::test]
async fn orchestrate_mode_prompt_emits_one_subagent_result_message() {
    let observer = Arc::new(WorkspaceObserver::default());
    let mut deps = orchestrate_deps();
    deps.turn_observer = Some(observer.clone());
    let agent = build_agent(deps);

    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();

    let stop = Client
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
            let response = cx
                .send_request(PromptRequest::new(
                    new_session.session_id,
                    vec![AcpContentBlock::from("Build a thing")],
                ))
                .block_task()
                .await?;
            Ok(response.stop_reason)
        })
        .await
        .expect("connection completed");

    assert_eq!(stop, StopReason::EndTurn);
    let observed = observer.0.lock().unwrap();
    assert!(observed
        .iter()
        .any(|request| request.agent_path.contains("orchestrator")));
    assert!(observed
        .iter()
        .any(|request| request.agent_path.contains("code")));
    assert!(observed
        .iter()
        .all(|request| request.workspace_roots == vec!["/tmp"]));

    let updates = updates.lock().unwrap();
    let messages = updates
        .iter()
        .filter_map(|update| match update {
            SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                AcpContentBlock::Text(text) => Some(text.text.as_str()),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(messages, vec!["did the work"]);
    assert!(!updates.iter().any(|update| matches!(
        update,
        SessionUpdate::ToolCall(_) | SessionUpdate::ToolCallUpdate(_)
    )));
}

#[tokio::test]
async fn idle_steering_in_orchestrate_mode_runs_the_orchestration_pipeline() {
    let agent = build_agent(orchestrate_deps());
    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
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
            let session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            cx.send_request(TestSteeringRequest {
                session_id: session.session_id,
                prompt: vec![AcpContentBlock::from("Implement through orchestration")],
            })
            .block_task()
            .await
        })
        .await
        .expect("connection completed");

    assert_eq!(response, serde_json::json!({ "outcome": "startedNewTurn" }));
    for _ in 0..50 {
        if updates.lock().unwrap().iter().any(|update| {
            matches!(
                update,
                SessionUpdate::AgentMessageChunk(chunk)
                    if matches!(&chunk.content, AcpContentBlock::Text(text) if text.text == "did the work")
            )
        }) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(updates.lock().unwrap().iter().any(|update| {
        matches!(
            update,
            SessionUpdate::AgentMessageChunk(chunk)
                if matches!(&chunk.content, AcpContentBlock::Text(text) if text.text == "did the work")
        )
    }));
}

#[derive(Default)]
struct ContextRecorder {
    histories: Mutex<Vec<Vec<HistoryEntry>>>,
}

#[async_trait]
impl NextTurnService for ContextRecorder {
    async fn get_next_turn_streaming(
        &self,
        ctx: &TurnContext,
    ) -> CoreResult<BoxStream<'static, TurnEvent>> {
        self.histories
            .lock()
            .unwrap()
            .push(ctx.history.entries.clone());
        Ok(Box::pin(stream::iter([TurnEvent::TurnFinished {
            stop_reason: CoreStopReason::EndTurn,
        }])))
    }
}

struct PendingService;

#[async_trait]
impl NextTurnService for PendingService {
    async fn get_next_turn_streaming(
        &self,
        _ctx: &TurnContext,
    ) -> CoreResult<BoxStream<'static, TurnEvent>> {
        Ok(Box::pin(stream::pending()))
    }
}

#[tokio::test]
async fn orchestrate_mode_inherits_cancelled_chat_prompt() {
    let recorder = Arc::new(ContextRecorder::default());
    let factory_calls = Arc::new(AtomicUsize::new(0));
    let factory: NextTurnFactory = Arc::new({
        let recorder = recorder.clone();
        move |_selection| {
            if factory_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                Arc::new(PendingService)
            } else {
                recorder.clone()
            }
        }
    });
    let deps = AgentDeps {
        next_turn_factory: factory,
        tools: ToolRegistry::new(),
        config: AgentConfig {
            require_submit_result: false,
            ..AgentConfig::default()
        },
        store: default_store(),
        turn_observer: None,
        inspector_base_url: None,
    };

    Client
        .builder()
        .name("test-client")
        .connect_with(build_agent(deps), async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            let first_prompt = cx
                .send_request(PromptRequest::new(
                    session.session_id.clone(),
                    vec![AcpContentBlock::from("Build a Kotlin RTS")],
                ))
                .block_task();
            let cancel = async {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                cx.send_notification(CancelNotification::new(session.session_id.clone()))
            };
            let (first_response, cancel_result) = tokio::join!(first_prompt, cancel);
            assert_eq!(first_response?.stop_reason, StopReason::Cancelled);
            cancel_result?;
            cx.send_request(SetSessionModeRequest::new(
                session.session_id.clone(),
                "orchestrate",
            ))
            .block_task()
            .await?;
            cx.send_request(PromptRequest::new(
                session.session_id.clone(),
                vec![AcpContentBlock::from("From scratch")],
            ))
            .block_task()
            .await?;
            cx.send_request(SetSessionModeRequest::new(
                session.session_id.clone(),
                "chat",
            ))
            .block_task()
            .await?;
            cx.send_request(PromptRequest::new(
                session.session_id,
                vec![AcpContentBlock::from("Proceed")],
            ))
            .block_task()
            .await?;
            Ok(())
        })
        .await
        .expect("connection completed");

    let histories = recorder.histories.lock().unwrap();
    assert_eq!(histories.len(), 2);
    assert!(histories[0].iter().any(|entry| matches!(
        entry,
        HistoryEntry::User(message)
            if message.content.iter().any(|block| matches!(
                block,
                HistoryContentBlock::Text { text } if text == "Build a Kotlin RTS"
            ))
    )));
    assert!(histories[0].iter().any(|entry| matches!(
        entry,
        HistoryEntry::User(message)
            if message.content.iter().any(|block| matches!(
                block,
                HistoryContentBlock::Text { text } if text == "From scratch"
            ))
    )));
    assert!(histories[1].iter().any(|entry| matches!(
        entry,
        HistoryEntry::User(message)
            if message.content.iter().any(|block| matches!(
                block,
                HistoryContentBlock::Text { text } if text == "From scratch"
            ))
    )));
    assert!(histories[1].iter().any(|entry| matches!(
        entry,
        HistoryEntry::User(message)
            if message.content.iter().any(|block| matches!(
                block,
                HistoryContentBlock::Text { text } if text == "Proceed"
            ))
    )));
}
