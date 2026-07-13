//! End-to-end tests driving `session/prompt` through an in-process fake client.
//!
//! These tests wire a real [`agent_client_protocol`] `Client` connection
//! directly to the agent connection built by [`acp_agent::build_agent`] (the
//! agent builder implements `ConnectTo<Client>`), exercising the full path:
//! `initialize` -> `session/new` -> `session/prompt` -> streamed
//! `session/update` notifications -> `stopReason`. The decision layer is the
//! deterministic replay backend, so no network is involved.

use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v1::{
    ContentBlock, InitializeRequest, NewSessionRequest, PermissionOptionId, PromptRequest,
    RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SelectedPermissionOutcome, SessionNotification, SessionUpdate, StopReason, ToolCallStatus,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{Client, on_receive_notification, on_receive_request};
use agent_core::{
    NextTurnService, Result as CoreResult, StopReason as CoreStopReason, Tool, ToolContext,
    ToolEvent, ToolRegistry, TurnEvent,
};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use serde_json::json;

use acp_agent::{build_agent, AgentConfig, AgentDeps, NextTurnFactory};

/// Build [`AgentDeps`] whose decision layer replays the given scripted rounds.
fn deps_with_script(script: Vec<Vec<TurnEvent>>, tools: ToolRegistry) -> AgentDeps {
    let factory: NextTurnFactory = Arc::new(move |_selection| {
        let service: Arc<dyn NextTurnService> =
            Arc::new(turn_replay::ReplayTurnService::new(script.clone()));
        service
    });
    AgentDeps {
        next_turn_factory: factory,
        tools,
        config: AgentConfig::default(),
    }
}

/// Collector for `session/update` notifications received by the fake client.
type Updates = Arc<Mutex<Vec<SessionUpdate>>>;

#[tokio::test]
async fn plain_streaming_turn_streams_chunks_and_end_turn() {
    let script = vec![vec![
        TurnEvent::TextDelta("Hello, ".to_string()),
        TurnEvent::TextDelta("world!".to_string()),
        TurnEvent::TurnFinished {
            stop_reason: CoreStopReason::EndTurn,
        },
    ]];
    let deps = deps_with_script(script, ToolRegistry::new());
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
                    vec![ContentBlock::from("hi")],
                ))
                .block_task()
                .await?;
            Ok(response.stop_reason)
        })
        .await
        .expect("connection completed");

    assert_eq!(stop, StopReason::EndTurn);

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
    assert_eq!(chunks, vec!["Hello, ".to_string(), "world!".to_string()]);
}

#[tokio::test]
async fn prompt_for_unknown_session_returns_error() {
    let deps = deps_with_script(vec![vec![]], ToolRegistry::new());
    let agent = build_agent(deps);

    let result: Result<(), String> = Client
        .builder()
        .name("test-client")
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            // Never created a session; prompting must fail.
            let outcome = cx
                .send_request(PromptRequest::new(
                    "does-not-exist",
                    vec![ContentBlock::from("hi")],
                ))
                .block_task()
                .await;
            match outcome {
                Ok(_) => Ok(Err("expected an error for unknown session".to_string())),
                Err(_) => Ok(Ok(())),
            }
        })
        .await
        .expect("connection completed");

    result.expect("unknown session should fail");
}

/// A tool that always succeeds and requires permission.
struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn schema(&self) -> serde_json::Value {
        json!({ "type": "object" })
    }
    fn requires_permission(&self) -> bool {
        true
    }
    async fn call(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> CoreResult<BoxStream<'static, ToolEvent>> {
        Ok(Box::pin(stream::iter(vec![
            ToolEvent::Started,
            ToolEvent::Completed(json!("echoed")),
        ])))
    }
}

/// Build a two-round script: a tool call, then a finishing text round.
fn tool_call_script() -> Vec<Vec<TurnEvent>> {
    vec![
        vec![
            TurnEvent::ToolCallRequested {
                id: agent_core::ToolCallId::new("call-1"),
                name: "echo".to_string(),
                arguments: json!({ "value": "hi" }),
            },
            TurnEvent::TurnFinished {
                stop_reason: CoreStopReason::ToolUse,
            },
        ],
        vec![
            TurnEvent::TextDelta("done".to_string()),
            TurnEvent::TurnFinished {
                stop_reason: CoreStopReason::EndTurn,
            },
        ],
    ]
}

/// Drive a tool-call turn, answering the permission request with `option_id`.
async fn run_tool_turn(option_id: &'static str) -> (StopReason, Vec<SessionUpdate>) {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(EchoTool));
    let deps = deps_with_script(tool_call_script(), tools);
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
        .on_receive_request(
            async move |_req: RequestPermissionRequest, responder, _cx| {
                responder.respond(RequestPermissionResponse::new(
                    RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                        PermissionOptionId::new(option_id),
                    )),
                ))
            },
            on_receive_request!(),
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
                    vec![ContentBlock::from("please echo")],
                ))
                .block_task()
                .await?;
            Ok(response.stop_reason)
        })
        .await
        .expect("connection completed");

    let snapshot = updates.lock().unwrap().clone();
    (stop, snapshot)
}

#[tokio::test]
async fn tool_call_with_granted_permission_completes() {
    let (stop, updates) = run_tool_turn("allow-once").await;
    assert_eq!(stop, StopReason::EndTurn);

    let statuses: Vec<ToolCallStatus> = updates
        .iter()
        .filter_map(|update| match update {
            SessionUpdate::ToolCall(call) => Some(call.status),
            SessionUpdate::ToolCallUpdate(update) => update.fields.status,
            _ => None,
        })
        .collect();
    // Pending (creation) -> InProgress -> Completed.
    assert!(statuses.contains(&ToolCallStatus::InProgress));
    assert!(statuses.contains(&ToolCallStatus::Completed));
    assert!(!statuses.contains(&ToolCallStatus::Failed));
}

#[tokio::test]
async fn tool_call_with_denied_permission_fails_but_turn_continues() {
    let (stop, updates) = run_tool_turn("reject-once").await;
    // The turn still finishes (loop continues after the denied tool call).
    assert_eq!(stop, StopReason::EndTurn);

    let statuses: Vec<ToolCallStatus> = updates
        .iter()
        .filter_map(|update| match update {
            SessionUpdate::ToolCall(call) => Some(call.status),
            SessionUpdate::ToolCallUpdate(update) => update.fields.status,
            _ => None,
        })
        .collect();
    assert!(statuses.contains(&ToolCallStatus::Failed));
    // A denied tool never enters InProgress.
    assert!(!statuses.contains(&ToolCallStatus::Completed));
}
