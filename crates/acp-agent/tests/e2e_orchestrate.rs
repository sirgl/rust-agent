//! End-to-end test (Step 15): a prompt in `orchestrate` mode runs the
//! orchestration pipeline, streaming sub-agent progress as tool-call updates
//! and returning a stop reason. Driven through an in-process fake client with
//! replay backends (no network).
//!
//! Cancellation of an in-flight orchestrated prompt is handled by the same
//! shared [`agent_core::CancellationToken`] threaded through `run_pipeline`
//! into the orchestrator turn and its sub-agents (the chat-mode cancellation
//! path is covered in `e2e_prompt.rs`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v1::{
    ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest, SessionNotification,
    SessionUpdate, StopReason,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{Client, on_receive_notification};
use agent_core::{NextTurnService, StopReason as CoreStopReason, ToolRegistry, TurnEvent};

use acp_agent::{build_agent, AgentConfig, AgentDeps, NextTurnFactory};

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
    }
}

#[tokio::test]
async fn orchestrate_mode_prompt_streams_subagent_progress() {
    let agent = build_agent(orchestrate_deps());

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
                    vec![ContentBlock::from("Build a thing")],
                ))
                .block_task()
                .await?;
            Ok(response.stop_reason)
        })
        .await
        .expect("connection completed");

    assert_eq!(stop, StopReason::EndTurn);

    // The orchestrator's run_subagent call surfaces as a tool_call update.
    let saw_tool_call = updates
        .lock()
        .unwrap()
        .iter()
        .any(|u| matches!(u, SessionUpdate::ToolCall(_) | SessionUpdate::ToolCallUpdate(_)));
    assert!(
        saw_tool_call,
        "orchestrate-mode prompt should stream sub-agent tool activity"
    );
}
