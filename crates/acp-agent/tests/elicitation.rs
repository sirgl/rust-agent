//! End-to-end tests for agent-initiated elicitation.
//!
//! These tests wire a real [`agent_client_protocol`] `Client` connection to the
//! agent built by [`acp_agent::build_agent`]. The scripted decision layer emits
//! a call to the built-in `elicitation` tool, which triggers an
//! `elicitation/create` request back to the fake client. The client answers
//! with accept / decline / cancel, and we assert the resulting tool-call status.

use std::sync::{Arc, Mutex};

use acp_agent::{build_agent, AgentConfig, AgentDeps, NextTurnFactory};
use agent_client_protocol::schema::v1::{
    ContentBlock, CreateElicitationRequest, CreateElicitationResponse, ElicitationAcceptAction,
    ElicitationAction, InitializeRequest, NewSessionRequest, PromptRequest, SessionNotification,
    SessionUpdate, StopReason, ToolCallStatus,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{on_receive_notification, on_receive_request, Client};
use agent_core::{NextTurnService, StopReason as CoreStopReason, TurnEvent};

/// Collector for `session/update` notifications received by the fake client.
type Updates = Arc<Mutex<Vec<SessionUpdate>>>;

/// Build [`AgentDeps`] whose decision layer replays the given scripted rounds,
/// wiring in the built-in tools (which include `elicitation`).
fn deps_with_script(script: Vec<Vec<TurnEvent>>) -> AgentDeps {
    let factory: NextTurnFactory = Arc::new(move |_selection| {
        let service: Arc<dyn NextTurnService> =
            Arc::new(turn_replay::ReplayTurnService::new(script.clone()));
        service
    });
    AgentDeps {
        next_turn_factory: factory,
        tools: tools_builtin::builtin_registry(),
        config: AgentConfig::default(),
    }
}

/// A two-round script: call the `elicitation` tool, then finish with text.
fn elicitation_script() -> Vec<Vec<TurnEvent>> {
    vec![
        vec![
            TurnEvent::ToolCallRequested {
                id: agent_core::ToolCallId::new("call-1"),
                name: "elicitation".to_string(),
                arguments: serde_json::json!({
                    "message": "What is your name?",
                    "requested_schema": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" }
                        },
                        "required": ["name"]
                    }
                }),
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

/// Drive an elicitation tool-call turn, answering with the given action.
async fn run_elicitation_turn(action: ElicitationAction) -> (StopReason, Vec<SessionUpdate>) {
    let deps = deps_with_script(elicitation_script());
    let agent = build_agent(deps);

    let updates: Updates = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    let response_action = Arc::new(Mutex::new(Some(action)));

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
            async move |req: CreateElicitationRequest, responder, _cx| {
                // The message should be forwarded verbatim from the tool call.
                assert_eq!(req.message, "What is your name?");
                let action = response_action.lock().unwrap().take().expect("one call");
                responder.respond(CreateElicitationResponse::new(action))
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
                    vec![ContentBlock::from("ask me")],
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

fn statuses(updates: &[SessionUpdate]) -> Vec<ToolCallStatus> {
    updates
        .iter()
        .filter_map(|update| match update {
            SessionUpdate::ToolCall(call) => Some(call.status),
            SessionUpdate::ToolCallUpdate(update) => update.fields.status,
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn elicitation_accept_completes_tool_call() {
    let mut content = std::collections::BTreeMap::new();
    content.insert("name".to_string(), "Ada".into());
    let action = ElicitationAction::Accept(ElicitationAcceptAction::new().content(content));

    let (stop, updates) = run_elicitation_turn(action).await;
    assert_eq!(stop, StopReason::EndTurn);

    let statuses = statuses(&updates);
    assert!(statuses.contains(&ToolCallStatus::InProgress));
    assert!(statuses.contains(&ToolCallStatus::Completed));
    assert!(!statuses.contains(&ToolCallStatus::Failed));
}

#[tokio::test]
async fn elicitation_decline_fails_tool_call_but_turn_continues() {
    let (stop, updates) = run_elicitation_turn(ElicitationAction::Decline).await;
    // The turn still finishes: the failed tool call is reported to the model.
    assert_eq!(stop, StopReason::EndTurn);

    let statuses = statuses(&updates);
    assert!(statuses.contains(&ToolCallStatus::Failed));
    assert!(!statuses.contains(&ToolCallStatus::Completed));
}

#[tokio::test]
async fn elicitation_cancel_fails_tool_call_but_turn_continues() {
    let (stop, updates) = run_elicitation_turn(ElicitationAction::Cancel).await;
    assert_eq!(stop, StopReason::EndTurn);

    let statuses = statuses(&updates);
    assert!(statuses.contains(&ToolCallStatus::Failed));
    assert!(!statuses.contains(&ToolCallStatus::Completed));
}
