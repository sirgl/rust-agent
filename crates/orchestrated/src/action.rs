//! The `run_subagent` tool: the orchestrator's single lever.
//!
//! [`RunSubAgentTool`] captures the shared [`OrchestratedStepContext`], the
//! [`ModeRegistry`], a [`ModelResolver`], and the base tool set. Each call runs
//! the full reference flow: parse the request, validate preconditions, record
//! the attempt, resolve the model tier (honoring a fixed model or escalation),
//! run the sub-agent worker, store the result into the context, and return a
//! concise `display_text` the orchestrator uses to pick its next move.

use std::sync::Arc;

use agent_core::{AgentRunMetadata, Result, Tool, ToolContext, ToolEvent, ToolOutputPresentation};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use tracing::{info_span, warn, Instrument};

use crate::context::{ModelTier, OrchestratedStepContext, StepResult};
use crate::mode::SubAgentRequest;
use crate::registry::ModeRegistry;
use crate::worker::{ModelResolver, OrchestratedSubAgentWorker, SubAgentRunParams};

/// The registry name the orchestrator's single tool is exposed under.
pub const RUN_SUBAGENT_TOOL_NAME: &str = "run_subagent";

/// The `run_subagent` tool driving sub-agents in modes over shared state.
pub struct RunSubAgentTool {
    context: Arc<OrchestratedStepContext>,
    modes: ModeRegistry,
    resolver: Arc<dyn ModelResolver>,
    base_tools: agent_core::ToolRegistry,
}

impl RunSubAgentTool {
    /// Create the tool from the shared context, mode registry, model resolver,
    /// and the base tool set sub-agents derive their capabilities from.
    pub fn new(
        context: Arc<OrchestratedStepContext>,
        modes: ModeRegistry,
        resolver: Arc<dyn ModelResolver>,
        base_tools: agent_core::ToolRegistry,
    ) -> Self {
        Self {
            context,
            modes,
            resolver,
            base_tools,
        }
    }
}

/// Wrap a pre-computed sequence of [`ToolEvent`]s as a stream.
fn events(events: Vec<ToolEvent>) -> BoxStream<'static, ToolEvent> {
    Box::pin(stream::iter(events))
}

/// A graceful failed tool result (never panics the orchestrator loop).
fn failed(message: impl Into<String>) -> BoxStream<'static, ToolEvent> {
    let message = message.into();
    warn!("run_subagent failed: {message}");
    events(vec![ToolEvent::Started, ToolEvent::Failed(message)])
}

/// Parse the `run_subagent` arguments into a [`SubAgentRequest`] (1-based
/// `step_index` is converted to 0-based). Returns an error message on failure.
fn parse_request(args: &serde_json::Value) -> std::result::Result<SubAgentRequest, String> {
    let mode = args
        .get("mode")
        .and_then(|v| v.as_str())
        .ok_or("run_subagent requires a string `mode`")?
        .to_string();

    let step_one_based = args
        .get("step_index")
        .and_then(|v| v.as_u64())
        .ok_or("run_subagent requires an integer `step_index` (1-based)")?;
    if step_one_based == 0 {
        return Err("`step_index` is 1-based and must be >= 1".to_string());
    }
    let step_index = (step_one_based - 1) as usize;

    let instructions = args
        .get("instructions")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let scope = args
        .get("scope")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let model_tier = args
        .get("model_tier")
        .and_then(|v| v.as_str())
        .and_then(ModelTier::from_str_opt);

    Ok(SubAgentRequest {
        mode,
        step_index,
        instructions,
        scope,
        model_tier,
    })
}

#[async_trait]
impl Tool for RunSubAgentTool {
    fn name(&self) -> &str {
        RUN_SUBAGENT_TOOL_NAME
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "The sub-agent mode: code, setup, niche, review, review_plan, or plan."
                },
                "step_index": {
                    "type": "integer",
                    "description": "The 1-based plan step this sub-agent works on."
                },
                "instructions": {
                    "type": "string",
                    "description": "Optional extra instructions for the sub-agent."
                },
                "scope": {
                    "type": "string",
                    "description": "Optional scope narrowing (files/areas)."
                },
                "model_tier": {
                    "type": "string",
                    "description": "Optional model tier: low, medium, or high."
                }
            },
            "required": ["mode", "step_index"]
        })
    }

    fn requires_permission(&self) -> bool {
        false
    }

    fn output_presentation(&self) -> ToolOutputPresentation {
        ToolOutputPresentation::AccumulatedMessage
    }

    async fn call(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        // 1. Parse the request (1-based -> 0-based).
        let req = match parse_request(&args) {
            Ok(req) => req,
            Err(msg) => return Ok(failed(msg)),
        };

        // 2. Resolve the mode (unknown -> graceful failure).
        let mode = match self.modes.from_str(&req.mode) {
            Ok(mode) => mode,
            Err(err) => return Ok(failed(err.to_string())),
        };

        // 3. Non-plan-creation modes require a plan proposal.
        if !mode.is_plan_creation_mode() && self.context.proposal().is_none() {
            return Ok(failed(format!(
                "mode '{}' requires a plan proposal, but none exists yet",
                mode.id()
            )));
        }

        // 4. Check ordering preconditions (e.g. review needs an executor result).
        if let Err(err) = mode.check_preconditions(&self.context, req.step_index) {
            return Ok(failed(err.message));
        }

        // 5. Record the attempt (1-based).
        let attempt = self.context.next_attempt(mode.id(), req.step_index);
        let is_retry = attempt > 1;

        // 6. Resolve the model tier and backend.
        let tier = resolve_tier(mode.as_ref(), &req, is_retry);
        let next_turn = self.resolver.resolve(tier);
        let max_steps = if is_retry && mode.supports_retry() {
            mode.retry_max_steps()
        } else {
            mode.initial_max_steps()
        };

        // 7. Run the sub-agent worker under a tracing span.
        let worker = OrchestratedSubAgentWorker::new(mode.clone());
        let span = info_span!(
            "run_subagent",
            mode = mode.id(),
            step_index = req.step_index,
            attempt,
            tier = ?tier
        );
        // Give the sub-agent a distinct, human-readable agent path segment
        // (e.g. `code#1`) so its captured requests/responses are viewable
        // separately in the LLM inspector, grouped under the parent session.
        let agent_label = format!("{}#{}", mode.id(), req.step_index + 1);
        let outcome = worker
            .run(SubAgentRunParams {
                req: &req,
                ctx: &self.context,
                base_tools: &self.base_tools,
                next_turn,
                max_steps,
                depth: ctx.depth + 1,
                client: ctx.client.clone(),
                workspace_roots: ctx.workspace_roots.clone(),
                cancel: &ctx.cancel,
                session_id: ctx.session_id.clone(),
                agent_path: ctx.agent_path.child(agent_label),
                observer: ctx.observer.clone(),
                parent_trace: ctx.trace.clone(),
                parent_tool_call_id: ctx.tool_call_id.clone(),
                trace_metadata: AgentRunMetadata {
                    kind: Some(format!("{:?}", mode.orchestrator_kind()).to_lowercase()),
                    mode: Some(mode.id().to_string()),
                    step_index: Some(req.step_index),
                    attempt: Some(attempt),
                    model_tier: Some(format!("{tier:?}").to_lowercase()),
                },
            })
            .instrument(span)
            .await?;

        // 8. Store the result so later modes/steps can read it.
        self.context.store_result(StepResult {
            mode_id: mode.id().to_string(),
            step_index: req.step_index,
            success: outcome.success,
            output: outcome.output.clone(),
        });

        // 9. Return the concise display text for the orchestrator.
        let display = mode.display_text(&outcome.output);
        Ok(events(vec![
            ToolEvent::Started,
            ToolEvent::OutputDelta(display.clone()),
            ToolEvent::Completed(serde_json::json!({
                "mode": mode.id(),
                "step_index": req.step_index,
                "attempt": attempt,
                "success": outcome.success,
                "stop_reason": outcome.stop_reason,
                "display_text": display,
            })),
        ]))
    }
}

/// Resolve the model tier for a run: a mode's fixed model wins; then an explicit
/// request tier; then, on a retry with escalation support, one tier up (clamped
/// at [`ModelTier::High`]); otherwise the default tier.
fn resolve_tier(
    mode: &dyn crate::mode::SubAgentMode,
    req: &SubAgentRequest,
    is_retry: bool,
) -> ModelTier {
    if let Some(fixed) = mode.fixed_model() {
        return fixed;
    }
    if let Some(tier) = req.model_tier {
        return tier;
    }
    if is_retry && mode.supports_model_escalation() {
        return ModelTier::default().escalated();
    }
    ModelTier::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{PlanProposal, PlanStepSpec};
    use crate::worker::FnModelResolver;
    use agent_core::{CancellationToken, NextTurnService, StopReason, ToolCallId, TurnEvent};
    use futures::StreamExt;
    use turn_replay::ReplayTurnService;

    fn submit_round(output: &str) -> Vec<Vec<TurnEvent>> {
        vec![
            vec![
                TurnEvent::ToolCallRequested {
                    id: ToolCallId::new("c"),
                    name: "submit".into(),
                    arguments: serde_json::json!({ "output": output }),
                },
                TurnEvent::TurnFinished {
                    stop_reason: StopReason::ToolUse,
                },
            ],
            vec![TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            }],
        ]
    }

    fn review_round(approved: bool) -> Vec<Vec<TurnEvent>> {
        vec![
            vec![
                TurnEvent::ToolCallRequested {
                    id: ToolCallId::new("r"),
                    name: "submit_review".into(),
                    arguments: serde_json::json!({ "approved": approved, "reasons": "ok" }),
                },
                TurnEvent::TurnFinished {
                    stop_reason: StopReason::ToolUse,
                },
            ],
            vec![TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            }],
        ]
    }

    fn resolver_of(rounds: Vec<Vec<TurnEvent>>) -> Arc<dyn ModelResolver> {
        let backend: Arc<dyn NextTurnService> = Arc::new(ReplayTurnService::new(rounds));
        Arc::new(FnModelResolver(move |_tier| backend.clone()))
    }

    fn ctx_with_plan() -> Arc<OrchestratedStepContext> {
        OrchestratedStepContext::with_proposal(PlanProposal::new(
            "Build X",
            vec![PlanStepSpec::new("Implement X", "do it")],
        ))
    }

    fn tool_ctx() -> ToolContext {
        ToolContext::new("orch", CancellationToken::new())
    }

    async fn last_event(stream: BoxStream<'static, ToolEvent>) -> ToolEvent {
        stream.collect::<Vec<_>>().await.pop().unwrap()
    }

    #[tokio::test]
    async fn single_executor_pass_stores_result() {
        let ctx = ctx_with_plan();
        let tool = RunSubAgentTool::new(
            ctx.clone(),
            ModeRegistry::with_defaults(),
            resolver_of(submit_round("did the work")),
            agent_core::ToolRegistry::new(),
        );

        let ev = last_event(
            tool.call(
                serde_json::json!({ "mode": "code", "step_index": 1 }),
                &tool_ctx(),
            )
            .await
            .unwrap(),
        )
        .await;
        assert!(matches!(ev, ToolEvent::Completed(_)));

        let stored = ctx.result("code", 0).expect("result stored");
        assert!(stored.success);
        assert_eq!(stored.output, "did the work");
        assert_eq!(ctx.attempts("code", 0), 1);
    }

    #[tokio::test]
    async fn implement_then_review_approved() {
        let ctx = ctx_with_plan();
        // Executor pass.
        RunSubAgentTool::new(
            ctx.clone(),
            ModeRegistry::with_defaults(),
            resolver_of(submit_round("impl")),
            agent_core::ToolRegistry::new(),
        )
        .call(
            serde_json::json!({ "mode": "code", "step_index": 1 }),
            &tool_ctx(),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

        // Review pass now that an executor result exists.
        RunSubAgentTool::new(
            ctx.clone(),
            ModeRegistry::with_defaults(),
            resolver_of(review_round(true)),
            agent_core::ToolRegistry::new(),
        )
        .call(
            serde_json::json!({ "mode": "review", "step_index": 1 }),
            &tool_ctx(),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

        let verdict = ctx.result("review", 0).expect("verdict stored");
        assert!(verdict.success);
    }

    #[tokio::test]
    async fn review_before_implement_fails_precondition() {
        let ctx = ctx_with_plan();
        let tool = RunSubAgentTool::new(
            ctx.clone(),
            ModeRegistry::with_defaults(),
            resolver_of(review_round(true)),
            agent_core::ToolRegistry::new(),
        );
        let ev = last_event(
            tool.call(
                serde_json::json!({ "mode": "review", "step_index": 1 }),
                &tool_ctx(),
            )
            .await
            .unwrap(),
        )
        .await;
        match ev {
            ToolEvent::Failed(msg) => assert!(msg.contains("no prior executor result")),
            other => panic!("expected failure, got {other:?}"),
        }
        assert!(ctx.result("review", 0).is_none());
    }

    #[tokio::test]
    async fn unknown_mode_and_malformed_args_fail_gracefully() {
        let ctx = ctx_with_plan();
        let tool = RunSubAgentTool::new(
            ctx.clone(),
            ModeRegistry::with_defaults(),
            resolver_of(submit_round("x")),
            agent_core::ToolRegistry::new(),
        );

        // Unknown mode.
        let ev = last_event(
            tool.call(
                serde_json::json!({ "mode": "bogus", "step_index": 1 }),
                &tool_ctx(),
            )
            .await
            .unwrap(),
        )
        .await;
        assert!(matches!(ev, ToolEvent::Failed(_)));

        // Missing step_index.
        let ev = last_event(
            tool.call(serde_json::json!({ "mode": "code" }), &tool_ctx())
                .await
                .unwrap(),
        )
        .await;
        assert!(matches!(ev, ToolEvent::Failed(_)));
    }

    #[test]
    fn escalation_and_fixed_model_resolution() {
        let executor = crate::mode::code_mode(Arc::new(crate::mode::ExecutorBehavior));
        let review = crate::mode::ReviewMode::new(Arc::new(crate::mode::ReviewerBehavior));
        let req = SubAgentRequest {
            mode: "code".into(),
            step_index: 0,
            instructions: None,
            scope: None,
            model_tier: None,
        };

        // First attempt -> default tier.
        assert_eq!(resolve_tier(&executor, &req, false), ModelTier::Medium);
        // Retry with escalation -> one tier up.
        assert_eq!(resolve_tier(&executor, &req, true), ModelTier::High);
        // Reviewer has a fixed model regardless of retry.
        assert_eq!(resolve_tier(&review, &req, false), ModelTier::High);

        // Explicit tier wins over escalation but not over a fixed model.
        let req_low = SubAgentRequest {
            model_tier: Some(ModelTier::Low),
            ..req.clone()
        };
        assert_eq!(resolve_tier(&executor, &req_low, true), ModelTier::Low);
        assert_eq!(resolve_tier(&review, &req_low, false), ModelTier::High);
    }

    #[tokio::test]
    async fn escalation_retry_increments_attempts() {
        let ctx = ctx_with_plan();
        let make = |rounds| {
            RunSubAgentTool::new(
                ctx.clone(),
                ModeRegistry::with_defaults(),
                resolver_of(rounds),
                agent_core::ToolRegistry::new(),
            )
        };
        make(submit_round("first"))
            .call(
                serde_json::json!({ "mode": "code", "step_index": 1 }),
                &tool_ctx(),
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        make(submit_round("second"))
            .call(
                serde_json::json!({ "mode": "code", "step_index": 1, "model_tier": "high" }),
                &tool_ctx(),
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;

        assert_eq!(ctx.attempts("code", 0), 2);
        assert_eq!(ctx.result("code", 0).unwrap().output, "second");
    }
}
