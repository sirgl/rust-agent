//! Replay-backed integration tests for canonical todo transitions in the turn engine.

use agent_core::{
    CancellationToken, EngineOutput, HistoryEntry, KnownTool, NextTurnService, PlanStepStatus,
    Result, SessionState, StopReason, ToolCallId, TurnEngine, TurnEvent, UpdateSink,
};
use async_trait::async_trait;
use std::sync::Arc;

#[derive(Default)]
struct CapturingSink {
    outputs: Vec<EngineOutput>,
}

#[async_trait]
impl UpdateSink for CapturingSink {
    async fn send(&mut self, output: EngineOutput) -> Result<()> {
        self.outputs.push(output);
        Ok(())
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

fn finish_round() -> Vec<TurnEvent> {
    vec![TurnEvent::TurnFinished {
        stop_reason: StopReason::EndTurn,
    }]
}

#[tokio::test]
async fn update_then_complete_mutates_state_emits_full_plans_and_records_typed_history() {
    let service: Arc<dyn NextTurnService> = Arc::new(turn_replay::ReplayTurnService::new(vec![
        tool_round(
            "update-1",
            "update_todo_list",
            serde_json::json!({
                "items": [
                    {"id": "inspect", "content": "Inspect", "status": "in_progress"},
                    {"id": "verify", "content": "Verify", "status": "pending"}
                ]
            }),
        ),
        tool_round(
            "complete-1",
            "mark_todo_completed",
            serde_json::json!({"id": "inspect"}),
        ),
        finish_round(),
    ]));
    let mut session = SessionState::new("todo-session");
    session.push_user_text("do the work");
    let mut sink = CapturingSink::default();
    let engine = TurnEngine::new(service, tools_builtin::builtin_registry());

    let stop = engine
        .run_prompt(&mut session, &mut sink, &CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(stop, StopReason::EndTurn);
    assert_eq!(session.todo_list.items.len(), 2);
    assert_eq!(session.todo_list.items[0].id.as_str(), "inspect");
    assert_eq!(session.todo_list.items[0].status, PlanStepStatus::Completed);
    assert_eq!(session.todo_list.items[1].id.as_str(), "verify");
    assert_eq!(session.todo_list.items[1].status, PlanStepStatus::Pending);

    let plans = sink
        .outputs
        .iter()
        .filter_map(|output| match output {
            EngineOutput::Plan(plan) => Some(plan),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(plans.len(), 2, "one full plan per successful transition");
    assert_eq!(plans[0].steps.len(), 2);
    assert_eq!(plans[0].steps[0].status, PlanStepStatus::InProgress);
    assert_eq!(plans[1].steps.len(), 2);
    assert_eq!(plans[1].steps[0].status, PlanStepStatus::Completed);

    assert!(session.history.entries.iter().any(|entry| matches!(
        entry,
        HistoryEntry::ToolCall(call) if matches!(call.tool, KnownTool::UpdateTodoList(_))
    )));
    assert!(session.history.entries.iter().any(|entry| matches!(
        entry,
        HistoryEntry::ToolCall(call) if matches!(call.tool, KnownTool::MarkTodoCompleted(_))
    )));
    let result = session
        .history
        .entries
        .iter()
        .rev()
        .find_map(|entry| match entry {
            HistoryEntry::ToolResult(result) => Some(&result.content),
            _ => None,
        })
        .expect("typed tool result");
    let result_text = result
        .iter()
        .map(|block| match block {
            agent_core::ContentBlock::Text { text } => text.as_str(),
        })
        .collect::<String>();
    assert!(result_text.contains("\"id\":\"inspect\""));
    assert!(result_text.contains("\"status\":\"completed\""));
}

#[tokio::test]
async fn malformed_or_unknown_updates_do_not_mutate_or_emit_plan() {
    for (name, arguments) in [
        (
            "update_todo_list",
            serde_json::json!({
                "items": [
                    {"id": "same", "content": "One", "status": "pending"},
                    {"id": "same", "content": "Two", "status": "pending"}
                ]
            }),
        ),
        ("mark_todo_completed", serde_json::json!({"id": "missing"})),
    ] {
        let service: Arc<dyn NextTurnService> =
            Arc::new(turn_replay::ReplayTurnService::new(vec![
                tool_round("bad", name, arguments),
                finish_round(),
            ]));
        let mut session = SessionState::new("todo-session");
        let mut sink = CapturingSink::default();
        TurnEngine::new(service, tools_builtin::builtin_registry())
            .run_prompt(&mut session, &mut sink, &CancellationToken::new())
            .await
            .unwrap();

        assert!(session.todo_list.items.is_empty());
        assert!(!sink
            .outputs
            .iter()
            .any(|output| matches!(output, EngineOutput::Plan(_))));
    }
}
