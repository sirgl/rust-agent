//! The shared orchestration entry point and the `orchestrate` tool.
//!
//! Both ways of launching orchestration — a dedicated ACP session mode and the
//! [`OrchestrateTool`] the chat agent can call itself — funnel through a single
//! [`run_pipeline`] helper, so behavior stays identical however it is triggered.
//!
//! `run_pipeline` seeds an [`OrchestratedStepContext`] with a single implicit
//! plan step derived from the goal (so `code`/`review` modes work immediately),
//! builds an [`Orchestrator`] via [`OrchestratorBuilder`], and drives its turn
//! engine into the provided [`UpdateSink`].

use std::sync::Arc;

use agent_core::{
    CancellationToken, EngineOutput, NextTurnService, Result, StopReason, Tool, ToolContext,
    ToolEvent, ToolRegistry, UpdateSink,
};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use tracing::debug;

use crate::context::{OrchestratedStepContext, PlanProposal, PlanStepSpec};
use crate::orchestrator::OrchestratorBuilder;
use crate::worker::{FnModelResolver, ModelResolver};

/// A factory that produces a *fresh* decision backend on each call.
///
/// Fresh instances matter because some backends (e.g. the replay backend) are
/// stateful and must not be shared across turns.
pub type BackendFactory = Arc<dyn Fn() -> Arc<dyn NextTurnService> + Send + Sync>;

/// The registry name under which the pipeline-launching tool is exposed.
pub const ORCHESTRATE_TOOL_NAME: &str = "orchestrate";

/// Build a [`ModelResolver`] that maps *every* [`ModelTier`](crate::ModelTier)
/// to a fresh backend from the given factory.
///
/// This keeps v1 wiring simple (all tiers resolve to the single configured
/// chat backend) while leaving the tier seam open for per-tier models later.
pub fn uniform_resolver(factory: BackendFactory) -> Arc<dyn ModelResolver> {
    Arc::new(FnModelResolver(move |_tier| (factory)()))
}

/// Derive a single implicit [`PlanProposal`] from a free-form goal.
fn implicit_proposal(goal: &str) -> PlanProposal {
    let goal = goal.trim();
    PlanProposal::new(
        goal.to_string(),
        vec![PlanStepSpec::new(goal.to_string(), String::new())],
    )
}

/// Build and drive an orchestrator for `goal`, streaming into `sink`.
///
/// The context is seeded with a single implicit plan step derived from the
/// goal; the orchestrator may still call the `plan` mode to refine it.
pub async fn run_pipeline(
    goal: &str,
    orchestrator_backend: Arc<dyn NextTurnService>,
    resolver: Arc<dyn ModelResolver>,
    base_tools: ToolRegistry,
    sink: &mut dyn UpdateSink,
    cancel: &CancellationToken,
) -> Result<StopReason> {
    let context = OrchestratedStepContext::with_proposal(implicit_proposal(goal));
    debug!(goal = %goal, "run_pipeline: starting orchestration");

    let orchestrator = OrchestratorBuilder::new(orchestrator_backend, resolver)
        .with_base_tools(base_tools)
        .with_context(context)
        .build();

    let mut session = orchestrator.new_session(goal);
    orchestrator
        .engine()
        .run_prompt(&mut session, sink, cancel)
        .await
}

/// A [`Tool`] that launches the orchestration pipeline for a goal.
///
/// Modeled on [`subagents::SubagentTool`](../../subagents): it runs
/// [`run_pipeline`] nested within the current turn using a capturing sink and
/// streams the orchestrator's progress back as ordinary [`ToolEvent`]s.
pub struct OrchestrateTool {
    factory: BackendFactory,
    resolver: Arc<dyn ModelResolver>,
    base_tools: ToolRegistry,
    name: String,
}

impl OrchestrateTool {
    /// Create an `orchestrate` tool.
    ///
    /// `factory` produces a fresh orchestrator decision backend per invocation;
    /// the sub-agent [`ModelResolver`] is typically [`uniform_resolver`] built
    /// from the same factory. `base_tools` is the tool set sub-agents derive
    /// their capabilities from.
    pub fn new(
        factory: BackendFactory,
        resolver: Arc<dyn ModelResolver>,
        base_tools: ToolRegistry,
    ) -> Self {
        Self {
            factory,
            resolver,
            base_tools,
            name: ORCHESTRATE_TOOL_NAME.to_string(),
        }
    }

    /// Override the registry name the tool is exposed under.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

#[async_trait]
impl Tool for OrchestrateTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The goal to accomplish via the orchestration pipeline."
                }
            },
            "required": ["goal"]
        })
    }

    fn requires_permission(&self) -> bool {
        false
    }

    async fn call(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        let Some(goal) = args.get("goal").and_then(|v| v.as_str()) else {
            return Ok(events(vec![
                ToolEvent::Started,
                ToolEvent::Failed("orchestrate requires a string `goal` argument".into()),
            ]));
        };

        let mut sink = CapturingSink::default();
        let outcome = run_pipeline(
            goal,
            (self.factory)(),
            self.resolver.clone(),
            self.base_tools.clone(),
            &mut sink,
            &ctx.cancel,
        )
        .await;

        match outcome {
            Ok(stop_reason) => {
                let text = sink.text;
                let mut out = vec![ToolEvent::Started];
                if !text.is_empty() {
                    out.push(ToolEvent::OutputDelta(text.clone()));
                }
                out.push(ToolEvent::Completed(serde_json::json!({
                    "stop_reason": stop_reason,
                    "output": text,
                })));
                Ok(events(out))
            }
            Err(err) => Ok(events(vec![
                ToolEvent::Started,
                ToolEvent::Failed(format!("orchestration failed: {err}")),
            ])),
        }
    }
}

/// An [`UpdateSink`] that captures assistant-visible text from the pipeline.
#[derive(Default)]
struct CapturingSink {
    text: String,
}

#[async_trait]
impl UpdateSink for CapturingSink {
    async fn send(&mut self, output: EngineOutput) -> Result<()> {
        if let EngineOutput::MessageChunk(chunk) = output {
            self.text.push_str(&chunk);
        }
        Ok(())
    }
}

/// Wrap a pre-computed sequence of [`ToolEvent`]s as a stream.
fn events(events: Vec<ToolEvent>) -> BoxStream<'static, ToolEvent> {
    Box::pin(stream::iter(events))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::{ToolCallId, TurnEvent};
    use futures::StreamExt;
    use turn_replay::ReplayTurnService;

    /// An orchestrator script: call run_subagent(code, step 1), then finish.
    fn orchestrator_script() -> Vec<Vec<TurnEvent>> {
        vec![
            vec![
                TurnEvent::ToolCallRequested {
                    id: ToolCallId::new("o1"),
                    name: "run_subagent".into(),
                    arguments: serde_json::json!({ "mode": "code", "step_index": 1 }),
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

    /// A sub-agent script: submit immediately, then finish.
    fn subagent_script() -> Vec<Vec<TurnEvent>> {
        vec![
            vec![
                TurnEvent::ToolCallRequested {
                    id: ToolCallId::new("s1"),
                    name: "submit".into(),
                    arguments: serde_json::json!({ "output": "implemented goal" }),
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

    fn subagent_resolver() -> Arc<dyn ModelResolver> {
        // Each resolve() yields a fresh replay backend (stateful).
        uniform_resolver(Arc::new(|| {
            Arc::new(ReplayTurnService::new(subagent_script())) as Arc<dyn NextTurnService>
        }))
    }

    #[derive(Default)]
    struct NullSink;
    #[async_trait]
    impl UpdateSink for NullSink {
        async fn send(&mut self, _o: EngineOutput) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn run_pipeline_drives_one_subagent_pass() {
        let orchestrator: Arc<dyn NextTurnService> =
            Arc::new(ReplayTurnService::new(orchestrator_script()));
        let resolver = subagent_resolver();

        // We need to inspect the context: run through the builder path but via
        // run_pipeline's own seeding. Instead assert on the streamed stop reason
        // and rely on parity test for stored results.
        let mut sink = NullSink;
        let cancel = CancellationToken::new();
        let stop = run_pipeline(
            "Build a thing",
            orchestrator,
            resolver,
            ToolRegistry::new(),
            &mut sink,
            &cancel,
        )
        .await
        .unwrap();
        assert_eq!(stop, StopReason::EndTurn);
    }

    #[tokio::test]
    async fn orchestrate_tool_returns_summary() {
        let factory: BackendFactory = Arc::new(|| {
            Arc::new(ReplayTurnService::new(orchestrator_script())) as Arc<dyn NextTurnService>
        });
        let tool = OrchestrateTool::new(factory, subagent_resolver(), ToolRegistry::new());

        assert_eq!(tool.name(), "orchestrate");
        assert!(!tool.requires_permission());

        let ctx = ToolContext::new("sess-main", CancellationToken::new());
        let stream = tool
            .call(serde_json::json!({ "goal": "Build a thing" }), &ctx)
            .await
            .unwrap();
        let evts: Vec<ToolEvent> = stream.collect().await;
        assert!(matches!(evts.first(), Some(ToolEvent::Started)));
        match evts.last() {
            Some(ToolEvent::Completed(v)) => {
                assert_eq!(v["stop_reason"], "end_turn");
            }
            other => panic!("unexpected final event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn orchestrate_tool_missing_goal_fails_gracefully() {
        let factory: BackendFactory = Arc::new(|| {
            Arc::new(ReplayTurnService::single(vec![TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            }])) as Arc<dyn NextTurnService>
        });
        let tool = OrchestrateTool::new(factory, subagent_resolver(), ToolRegistry::new());
        let ctx = ToolContext::new("sess-main", CancellationToken::new());
        let stream = tool
            .call(serde_json::json!({ "not_goal": 1 }), &ctx)
            .await
            .unwrap();
        let evts: Vec<ToolEvent> = stream.collect().await;
        assert!(matches!(evts.last(), Some(ToolEvent::Failed(_))));
    }
}
