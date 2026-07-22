//! ACP-boundary regressions for canonical todo plan projection.

use std::sync::{Arc, Mutex};

use acp_agent::{build_agent, default_store, AgentConfig, AgentDeps, NextTurnFactory};
use agent_client_protocol::schema::v1::{
    ContentBlock, InitializeRequest, NewSessionRequest, PlanEntryStatus, PromptRequest,
    SessionNotification, SessionUpdate,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{on_receive_notification, Client};
use agent_core::{NextTurnService, StopReason, ToolCallId, TurnEvent};

fn deps(script: Vec<Vec<TurnEvent>>) -> AgentDeps {
    let factory: NextTurnFactory = Arc::new(move |_selection| {
        Arc::new(turn_replay::ReplayTurnService::new(script.clone())) as Arc<dyn NextTurnService>
    });
    AgentDeps {
        next_turn_factory: factory,
        tools: tools_builtin::builtin_registry(),
        config: AgentConfig {
            require_submit_result: false,
            thought_interval: 0,
            ..AgentConfig::default()
        },
        store: default_store(),
        turn_observer: None,
        inspector_base_url: None,
    }
}

fn tool_round(id: &str, name: &str, arguments: serde_json::Value) -> Vec<TurnEvent> {
    vec![
        TurnEvent::ToolCallRequested {
            id: ToolCallId::new(id),
            name: name.to_string(),
            arguments,
        },
        TurnEvent::TurnFinished {
            stop_reason: StopReason::ToolUse,
        },
    ]
}

#[tokio::test]
async fn todo_mutations_emit_exact_full_acp_plans() {
    let script = vec![
        tool_round(
            "update",
            "update_todo_list",
            serde_json::json!({
                "items": [
                    {"id": "one", "content": "One", "status": "in_progress"},
                    {"id": "two", "content": "Two", "status": "pending"}
                ]
            }),
        ),
        tool_round(
            "complete",
            "mark_todo_completed",
            serde_json::json!({"id": "one"}),
        ),
        vec![TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        }],
    ];
    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();

    Client
        .builder()
        .name("todo-plan-client")
        .on_receive_notification(
            async move |notification: SessionNotification, _cx| {
                collected.lock().unwrap().push(notification.update);
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_with(build_agent(deps(script)), async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            cx.send_request(PromptRequest::new(
                session.session_id,
                vec![ContentBlock::from("work")],
            ))
            .block_task()
            .await?;
            Ok(())
        })
        .await
        .expect("connection completed");

    let plans = updates
        .lock()
        .unwrap()
        .iter()
        .filter_map(|update| match update {
            SessionUpdate::Plan(plan) => Some(plan.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].entries.len(), 2);
    assert_eq!(plans[0].entries[0].content, "One");
    assert_eq!(plans[0].entries[0].status, PlanEntryStatus::InProgress);
    assert_eq!(plans[0].entries[1].content, "Two");
    assert_eq!(plans[0].entries[1].status, PlanEntryStatus::Pending);
    assert_eq!(plans[1].entries.len(), 2);
    assert_eq!(plans[1].entries[0].status, PlanEntryStatus::Completed);
    assert_eq!(plans[1].entries[1].status, PlanEntryStatus::Pending);
}

#[tokio::test]
async fn empty_replacement_emits_one_empty_acp_plan() {
    let script = vec![
        tool_round(
            "clear",
            "update_todo_list",
            serde_json::json!({"items": []}),
        ),
        vec![TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        }],
    ];
    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();

    Client
        .builder()
        .name("todo-clear-client")
        .on_receive_notification(
            async move |notification: SessionNotification, _cx| {
                collected.lock().unwrap().push(notification.update);
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_with(build_agent(deps(script)), async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            cx.send_request(PromptRequest::new(
                session.session_id,
                vec![ContentBlock::from("clear")],
            ))
            .block_task()
            .await?;
            Ok(())
        })
        .await
        .expect("connection completed");

    let plans = updates
        .lock()
        .unwrap()
        .iter()
        .filter_map(|update| match update {
            SessionUpdate::Plan(plan) => Some(plan.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(plans.len(), 1);
    assert!(plans[0].entries.is_empty());
}
