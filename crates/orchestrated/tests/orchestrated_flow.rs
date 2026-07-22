//! End-to-end orchestration flow driven entirely by `turn-replay`.
//!
//! The orchestrator LLM and every sub-agent are replay-scripted, so a full
//! multi-pass `implement -> review(negative) -> re-implement -> review(approved)`
//! flow runs deterministically with no network. We assert the state threaded
//! through [`OrchestratedStepContext`] (stored results, verdicts, attempts).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_core::{
    CancellationToken, EngineOutput, NextTurnService, Result, StopReason, ToolCallId, TurnEvent,
    UpdateSink,
};
use async_trait::async_trait;
use orchestrated::{
    ModelResolver, ModelTier, OrchestratedStepContext, OrchestratorBuilder, PlanProposal,
    PlanStepSpec,
};
use turn_replay::ReplayTurnService;

#[derive(Default)]
struct NullSink;

#[async_trait]
impl UpdateSink for NullSink {
    async fn send(&mut self, _o: EngineOutput) -> Result<()> {
        Ok(())
    }
}

/// A resolver that hands out a fresh pre-scripted sub-agent backend on each
/// `run_subagent` call, in order.
struct QueueResolver {
    backends: Mutex<VecDeque<Arc<dyn NextTurnService>>>,
}

impl ModelResolver for QueueResolver {
    fn resolve(&self, _tier: ModelTier) -> Arc<dyn NextTurnService> {
        self.backends
            .lock()
            .expect("backends lock")
            .pop_front()
            .expect("a scripted sub-agent backend for this run_subagent call")
    }
}

/// Two replay rounds: a single tool call then a finishing round.
fn tool_then_finish(name: &str, args: serde_json::Value) -> Arc<dyn NextTurnService> {
    Arc::new(ReplayTurnService::new(vec![
        vec![
            TurnEvent::ToolCallRequested {
                id: ToolCallId::new("t"),
                name: name.to_string(),
                arguments: args,
            },
            TurnEvent::TurnFinished {
                stop_reason: StopReason::ToolUse,
            },
        ],
        vec![TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        }],
    ]))
}

/// One orchestrator round issuing a `run_subagent` call.
fn orchestrator_round(args: serde_json::Value) -> Vec<TurnEvent> {
    vec![
        TurnEvent::ToolCallRequested {
            id: ToolCallId::new("o"),
            name: "run_subagent".into(),
            arguments: args,
        },
        TurnEvent::TurnFinished {
            stop_reason: StopReason::ToolUse,
        },
    ]
}

#[tokio::test]
async fn full_multi_pass_flow_threads_results_and_verdicts() {
    // The orchestrator: code -> review(neg) -> code(retry) -> review(approved) -> done.
    let orchestrator = Arc::new(ReplayTurnService::new(vec![
        orchestrator_round(serde_json::json!({ "mode": "code", "step_index": 1 })),
        orchestrator_round(serde_json::json!({ "mode": "review", "step_index": 1 })),
        orchestrator_round(
            serde_json::json!({ "mode": "code", "step_index": 1, "instructions": "add tests", "model_tier": "high" }),
        ),
        orchestrator_round(serde_json::json!({ "mode": "review", "step_index": 1 })),
        vec![TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        }],
    ]));

    // One sub-agent backend per run_subagent call, in order.
    let backends = VecDeque::from(vec![
        tool_then_finish("submit", serde_json::json!({ "output": "impl v1" })),
        tool_then_finish(
            "submit_review",
            serde_json::json!({ "approved": false, "reasons": "needs tests" }),
        ),
        tool_then_finish(
            "submit",
            serde_json::json!({ "output": "impl v2 with tests" }),
        ),
        tool_then_finish(
            "submit_review",
            serde_json::json!({ "approved": true, "reasons": "looks good" }),
        ),
    ]);
    let resolver = Arc::new(QueueResolver {
        backends: Mutex::new(backends),
    });

    let context = OrchestratedStepContext::with_proposal(PlanProposal::new(
        "Build a widget",
        vec![PlanStepSpec::new("Implement widget", "with tests")],
    ));

    let orch = OrchestratorBuilder::new(orchestrator, resolver)
        .with_context(context.clone())
        .build();

    let mut session = orch.new_session("Build a widget with tests");
    let cancel = CancellationToken::new();
    let stop = orch
        .engine()
        .run_prompt(&mut session, &mut NullSink, &cancel)
        .await
        .expect("orchestrator run");

    assert_eq!(stop, StopReason::EndTurn);

    // The executor was run twice; the latest result is the retry's output.
    assert_eq!(context.attempts("code", 0), 2);
    let code = context.result("code", 0).expect("code result");
    assert!(code.success);
    assert_eq!(code.output, "impl v2 with tests");

    // The reviewer was run twice; the final stored verdict is positive.
    assert_eq!(context.attempts("review", 0), 2);
    let review = context.result("review", 0).expect("review result");
    assert!(review.success);
    assert!(review.output.contains("looks good"));
}
