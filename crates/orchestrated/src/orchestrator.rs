//! Orchestrator assembly.
//!
//! The orchestrator is *just* a [`TurnEngine`] whose registry contains a single
//! tool — [`RunSubAgentTool`] — driven by its own decision backend and an
//! orchestrator system prompt. It never does the work itself; every action is
//! delegated to a sub-agent via `run_subagent`.
//!
//! [`OrchestratorBuilder`] wires the pieces (orchestrator backend, a
//! [`ModelResolver`] for sub-agents, a [`ModeRegistry`], the base tool set, and
//! the shared [`OrchestratedStepContext`]) into a ready-to-drive [`Orchestrator`].

use std::sync::Arc;

use agent_core::{NextTurnService, SessionState, ToolRegistry, TurnEngine};

use crate::action::RunSubAgentTool;
use crate::context::OrchestratedStepContext;
use crate::registry::ModeRegistry;
use crate::worker::ModelResolver;

/// The default system prompt seeded into the orchestrator session.
pub const ORCHESTRATOR_SYSTEM_PROMPT: &str = "\
You are an orchestrator. You do not perform the work yourself; instead you drive \
an un-fixed pipeline by repeatedly calling the `run_subagent` tool, choosing the \
mode (code/setup/niche to implement, review/review_plan to check, plan to plan) \
and the 1-based plan step for each call. Inspect each sub-agent's result before \
deciding the next call. Re-run an executor with instructions or a higher \
model_tier if a review is negative. Finish when the goal is satisfied.";

/// A fully-assembled orchestrator: a turn engine plus its shared state.
pub struct Orchestrator {
    engine: TurnEngine,
    context: Arc<OrchestratedStepContext>,
    system_prompt: String,
}

impl Orchestrator {
    /// The orchestrator's turn engine (registry contains only `run_subagent`).
    pub fn engine(&self) -> &TurnEngine {
        &self.engine
    }

    /// The shared orchestration state (results, verdicts, attempts, plan).
    pub fn context(&self) -> &Arc<OrchestratedStepContext> {
        &self.context
    }

    /// Build a fresh orchestrator [`SessionState`] seeded with the system prompt
    /// and the goal as the initial user message.
    pub fn new_session(&self, goal: impl Into<String>) -> SessionState {
        let mut session = SessionState::new("orchestrator");
        session.system_prompt = Some(self.system_prompt.clone());
        session.push_user_text(goal.into());
        session
    }
}

/// Builder for an [`Orchestrator`].
pub struct OrchestratorBuilder {
    orchestrator_backend: Arc<dyn NextTurnService>,
    resolver: Arc<dyn ModelResolver>,
    modes: ModeRegistry,
    base_tools: ToolRegistry,
    context: Arc<OrchestratedStepContext>,
    system_prompt: String,
}

impl OrchestratorBuilder {
    /// Start a builder with the orchestrator's decision backend and the
    /// sub-agent model resolver. Defaults: the full [`ModeRegistry`], an empty
    /// base tool set, a fresh context, and the default system prompt.
    pub fn new(
        orchestrator_backend: Arc<dyn NextTurnService>,
        resolver: Arc<dyn ModelResolver>,
    ) -> Self {
        Self {
            orchestrator_backend,
            resolver,
            modes: ModeRegistry::with_defaults(),
            base_tools: ToolRegistry::new(),
            context: OrchestratedStepContext::new(),
            system_prompt: ORCHESTRATOR_SYSTEM_PROMPT.to_string(),
        }
    }

    /// Override the mode registry.
    pub fn with_modes(mut self, modes: ModeRegistry) -> Self {
        self.modes = modes;
        self
    }

    /// Set the base tool set sub-agents derive their capabilities from.
    pub fn with_base_tools(mut self, base_tools: ToolRegistry) -> Self {
        self.base_tools = base_tools;
        self
    }

    /// Provide a shared context (e.g. pre-seeded with a plan proposal).
    pub fn with_context(mut self, context: Arc<OrchestratedStepContext>) -> Self {
        self.context = context;
        self
    }

    /// Override the orchestrator system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// The shared context, so the caller can seed a plan or read results.
    pub fn context(&self) -> Arc<OrchestratedStepContext> {
        self.context.clone()
    }

    /// Assemble the [`Orchestrator`].
    pub fn build(self) -> Orchestrator {
        let run_subagent = Arc::new(RunSubAgentTool::new(
            self.context.clone(),
            self.modes,
            self.resolver,
            self.base_tools,
        ));
        let mut registry = ToolRegistry::new();
        registry.register(run_subagent);

        let engine = TurnEngine::new(self.orchestrator_backend, registry);
        Orchestrator {
            engine,
            context: self.context,
            system_prompt: self.system_prompt,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{PlanProposal, PlanStepSpec};
    use crate::worker::FnModelResolver;
    use agent_core::{CancellationToken, EngineOutput, Result, StopReason, ToolCallId, TurnEvent, UpdateSink};
    use async_trait::async_trait;
    use turn_replay::ReplayTurnService;

    #[derive(Default)]
    struct NullSink;
    #[async_trait]
    impl UpdateSink for NullSink {
        async fn send(&mut self, _o: EngineOutput) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn orchestrator_drives_one_run_subagent_call() {
        // Orchestrator LLM: call run_subagent(code, step 1), then finish.
        let orchestrator = Arc::new(ReplayTurnService::new(vec![
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
        ]));

        // Sub-agent backend: submit immediately.
        let sub_backend: Arc<dyn NextTurnService> = Arc::new(ReplayTurnService::new(vec![
            vec![
                TurnEvent::ToolCallRequested {
                    id: ToolCallId::new("s1"),
                    name: "submit".into(),
                    arguments: serde_json::json!({ "output": "implemented step" }),
                },
                TurnEvent::TurnFinished {
                    stop_reason: StopReason::ToolUse,
                },
            ],
            vec![TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            }],
        ]));
        let resolver = Arc::new(FnModelResolver(move |_t| sub_backend.clone()));

        let context = OrchestratedStepContext::with_proposal(PlanProposal::new(
            "Build X",
            vec![PlanStepSpec::new("Implement X", "do it")],
        ));
        let orch = OrchestratorBuilder::new(orchestrator, resolver)
            .with_context(context.clone())
            .build();

        let mut session = orch.new_session("Please build X");
        let cancel = CancellationToken::new();
        let stop = orch
            .engine()
            .run_prompt(&mut session, &mut NullSink, &cancel)
            .await
            .unwrap();

        assert_eq!(stop, StopReason::EndTurn);
        let result = context.result("code", 0).expect("executor result stored");
        assert!(result.success);
        assert_eq!(result.output, "implemented step");
    }
}
