//! End-to-end test for terminal-tool turn completion (`submit_result`).
//!
//! With `AgentConfig::require_submit_result` enabled, a decision round that
//! produces only text must NOT end the turn: the engine nudges the model and
//! keeps the turn alive until the model dispatches the terminal `submit_result`
//! tool. This test scripts exactly that two-round flow and asserts the turn
//! ends only after `submit_result` is called.

use std::sync::{Arc, Mutex};

use acp_agent::{build_agent, AgentConfig, AgentDeps, NextTurnFactory};
use agent_client_protocol::schema::v1::{
    ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest, SessionNotification,
    SessionUpdate, StopReason,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{on_receive_notification, Client};
use agent_core::{NextTurnService, StopReason as CoreStopReason, TurnEvent};

type Updates = Arc<Mutex<Vec<SessionUpdate>>>;

/// Round 1: plain text only (would previously end the turn). Round 2: call the
/// terminal `submit_result` tool.
fn two_round_script() -> Vec<Vec<TurnEvent>> {
    vec![
        vec![
            TurnEvent::TextDelta("Working on it...".to_string()),
            TurnEvent::TurnFinished {
                stop_reason: CoreStopReason::EndTurn,
            },
        ],
        vec![
            TurnEvent::ToolCallRequested {
                id: agent_core::ToolCallId::new("submit-1"),
                name: "submit_result".to_string(),
                arguments: serde_json::json!({ "summary": "finished the task" }),
            },
            TurnEvent::TurnFinished {
                stop_reason: CoreStopReason::ToolUse,
            },
        ],
    ]
}

#[tokio::test]
async fn turn_ends_only_after_submit_result() {
    let factory: NextTurnFactory = Arc::new(move |_selection| {
        let service: Arc<dyn NextTurnService> =
            Arc::new(turn_replay::ReplayTurnService::new(two_round_script()));
        service
    });
    let deps = AgentDeps {
        next_turn_factory: factory,
        tools: tools_builtin::builtin_registry(),
        config: AgentConfig {
            require_submit_result: true,
            ..AgentConfig::default()
        },
        store: acp_agent::default_store(),
    };
    let agent = build_agent(deps);

    let updates: Updates = Arc::new(Mutex::new(Vec::new()));
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
                    vec![ContentBlock::from("do the task")],
                ))
                .block_task()
                .await?;
            Ok(response.stop_reason)
        })
        .await
        .expect("connection completed");

    // The turn ends only after the second round dispatched `submit_result`.
    assert_eq!(stop, StopReason::EndTurn);

    let snapshot = updates.lock().unwrap().clone();

    // The first round's text was still streamed to the client.
    let chunks: Vec<String> = snapshot
        .iter()
        .filter_map(|update| match update {
            SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                ContentBlock::Text(text) => Some(text.text.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(chunks, vec!["Working on it...".to_string()]);

    // The terminal `submit_result` tool call was dispatched.
    let saw_submit = snapshot.iter().any(|update| match update {
        SessionUpdate::ToolCall(call) => call.title.contains("submit_result"),
        _ => false,
    });
    assert!(saw_submit, "expected a submit_result tool call to be streamed");
}
