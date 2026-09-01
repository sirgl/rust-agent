#![cfg(feature = "unstable_protocol_v2")]

//! End-to-end test for terminal-tool turn completion (`submit_result`).
//!
//! With `AgentConfig::require_submit_result` enabled, a decision round that
//! produces only text must NOT end the turn: the engine nudges the model and
//! keeps the turn alive until the model dispatches the terminal `submit_result`
//! tool. This test scripts exactly that two-round flow and asserts the turn
//! ends only after `submit_result` is called.

use std::sync::{Arc, Mutex};

use acp_agent::{build_agent, AgentConfig, AgentDeps, NextTurnFactory};
use agent_client_protocol::schema::v2::{
    ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest, SessionUpdate, StopReason,
    UpdateSessionNotification,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{on_receive_notification, Client};
use agent_core::{NextTurnService, StopReason as CoreStopReason, TurnEvent};

mod common;

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
        turn_observer: None,
        inspector_base_url: None,
    };
    let agent = build_agent(deps);

    let updates: common::Updates = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    let completion = updates.clone();

    let stop = Client
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
            cx.send_request(PromptRequest::new(
                new_session.session_id,
                vec![ContentBlock::from("do the task")],
            ))
            .block_task()
            .await?;
            Ok(common::wait_for_idle(&completion).await)
        })
        .await
        .expect("connection completed");

    // The turn ends only after the second round dispatched `submit_result`.
    assert_eq!(stop, StopReason::EndTurn);

    let snapshot = updates.lock().unwrap().clone();

    // The first round's text was streamed, and the terminal `submit_result`'s
    // summary is rendered as a plain agent message (not a tool call).
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
    assert_eq!(
        chunks,
        vec![
            "Working on it...".to_string(),
            "finished the task".to_string()
        ]
    );

    // The terminal `submit_result` tool is NOT surfaced as a tool call anymore.
    let saw_submit_call = snapshot.iter().any(|update| match update {
        SessionUpdate::ToolCallUpdate(call) => call
            .title
            .value()
            .is_some_and(|title| title.contains("submit_result")),
        _ => false,
    });
    assert!(
        !saw_submit_call,
        "submit_result should be rendered as a message, not a tool call"
    );
}
