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

use agent_core::{
    AgentPath, ClientAccess, NextTurnService, SessionState, SharedTurnObserver, ToolDescriptor,
    ToolRegistry, TurnEngine, TurnInbox,
};

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
    session_id: String,
    tools: Vec<ToolDescriptor>,
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
        let mut session = self.empty_session();
        session.push_user_text(goal.into());
        session
    }

    /// Build a fresh orchestrator session without an initial user message.
    ///
    /// Long-lived frontends use this to append every prompt to the same typed
    /// history while retaining the shared orchestration context.
    pub fn empty_session(&self) -> SessionState {
        let mut session = SessionState::new(self.session_id.clone());
        session.system_prompt = Some(self.system_prompt.clone());
        // Advertise the orchestrator's tools (i.e. `run_subagent`) to the
        // decision layer, otherwise the model has no tool schema to call and
        // merely describes the call in text (see orchestrate-mode bug).
        session.available_tools = self.tools.clone();
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
    observer: Option<SharedTurnObserver>,
    agent_path: AgentPath,
    session_id: String,
    inbox: Option<TurnInbox>,
    client: Option<Arc<dyn ClientAccess>>,
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
            observer: None,
            agent_path: AgentPath::root(),
            session_id: "orchestrator".to_string(),
            inbox: None,
            client: None,
        }
    }

    /// Attach a steering inbox drained between orchestrator decision rounds.
    #[must_use]
    pub fn with_inbox(mut self, inbox: TurnInbox) -> Self {
        self.inbox = Some(inbox);
        self
    }

    /// Attach client access inherited by orchestrator tools and nested workers.
    #[must_use]
    pub fn with_client(mut self, client: Arc<dyn ClientAccess>) -> Self {
        self.client = Some(client);
        self
    }

    /// Attach a provider-neutral observer so the orchestrator turn and, via the
    /// propagated [`ToolContext`], its sub-agents are captured by the inspector.
    #[must_use]
    pub fn with_observer(mut self, observer: Option<SharedTurnObserver>) -> Self {
        self.observer = observer;
        self
    }

    /// Set the [`AgentPath`] of the orchestrator agent (sub-agents derive
    /// children of it).
    #[must_use]
    pub fn with_agent_path(mut self, agent_path: AgentPath) -> Self {
        self.agent_path = agent_path;
        self
    }

    /// Set the session id used for the orchestrator's session (and inherited by
    /// its sub-agents), so all of a user session's agents group together.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = session_id.into();
        self
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

        // Snapshot tool descriptors so orchestrator sessions can advertise them
        // to the model (the registry itself is only used to dispatch calls).
        let tools: Vec<ToolDescriptor> = registry
            .names()
            .filter_map(|name| registry.get(name).map(|tool| (name.to_string(), tool)))
            .map(|(name, tool)| ToolDescriptor {
                name,
                schema: tool.schema(),
                requires_permission: tool.requires_permission(),
            })
            .collect();

        let mut engine =
            TurnEngine::new(self.orchestrator_backend, registry).with_agent_path(self.agent_path);
        if let Some(inbox) = self.inbox {
            engine = engine.with_inbox(inbox);
        }
        if let Some(client) = self.client {
            engine = engine.with_client(client);
        }
        if let Some(observer) = self.observer {
            engine = engine.with_observer(observer);
        }
        Orchestrator {
            engine,
            context: self.context,
            system_prompt: self.system_prompt,
            session_id: self.session_id,
            tools,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{PlanProposal, PlanStepSpec};
    use crate::worker::FnModelResolver;
    use agent_core::{
        CancellationToken, EngineOutput, Result, StopReason, ToolCallId, TurnEvent, UpdateSink,
    };
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

    #[tokio::test]
    async fn new_session_advertises_run_subagent_tool() {
        // Regression: the orchestrator session must expose `run_subagent`'s
        // schema to the decision layer, otherwise a real LLM has no tool to call
        // and only describes the call in text (orchestrate-mode did nothing).
        let backend: Arc<dyn NextTurnService> = Arc::new(ReplayTurnService::new(vec![vec![
            TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            },
        ]]));
        let resolver = Arc::new(FnModelResolver(move |_t| backend.clone()));
        let orch =
            OrchestratorBuilder::new(Arc::new(ReplayTurnService::new(vec![])), resolver).build();

        let session = orch.new_session("Build X");
        let ctx = session.turn_context();
        assert!(
            ctx.available_tools.iter().any(|t| t.name == "run_subagent"),
            "orchestrator session must advertise run_subagent to the model"
        );
    }
}
