//! The sub-agent worker: runs a nested [`TurnEngine`] in a chosen mode.
//!
//! Modeled on the `subagents` crate's nested-engine pattern. Given a resolved
//! [`SubAgentMode`], the worker:
//! 1. derives the tool set from the mode's [`tool_capabilities`](SubAgentMode::tool_capabilities)
//!    and injects the mode's submit action (writing into a per-run [`SubmitSlot`]),
//! 2. builds a nested [`SessionState`] seeded with the mode's system prompt, the
//!    issue description, and the pre-chat observations,
//! 3. drives the engine to a [`StopReason`] with a capturing sink,
//! 4. reads the submit slot and computes success via
//!    [`parse_success`](SubAgentMode::parse_success).

use std::sync::Arc;

use agent_core::{
    AgentPath, AgentRunMetadata, ClientAccess, EngineOutput, NextTurnService, Result, SessionState,
    SharedTurnObserver, StopReason, Tool, ToolCallId, ToolRegistry, TraceContext, TurnEngine,
    UpdateSink,
};
use async_trait::async_trait;

use crate::context::{ModelTier, OrchestratedStepContext};
use crate::mode::{SubAgentMode, SubAgentRequest, SubmitKind, SubmitOutcome};
use crate::submit::{new_slot, SubmitReviewTool, SubmitTool};

/// A provider-neutral mapping from a [`ModelTier`] to a decision backend.
///
/// Escalation stays provider-agnostic: the orchestration crate never names a
/// concrete provider; the binary injects a resolver built from its factories.
pub trait ModelResolver: Send + Sync {
    /// Resolve the decision backend for the given tier.
    fn resolve(&self, tier: ModelTier) -> Arc<dyn NextTurnService>;
}

/// A [`ModelResolver`] backed by a closure.
pub struct FnModelResolver<F>(pub F)
where
    F: Fn(ModelTier) -> Arc<dyn NextTurnService> + Send + Sync;

impl<F> ModelResolver for FnModelResolver<F>
where
    F: Fn(ModelTier) -> Arc<dyn NextTurnService> + Send + Sync,
{
    fn resolve(&self, tier: ModelTier) -> Arc<dyn NextTurnService> {
        (self.0)(tier)
    }
}

/// The outcome of running a sub-agent turn.
#[derive(Debug, Clone)]
pub struct SubAgentRunOutcome {
    /// The stop reason the nested turn reached.
    pub stop_reason: StopReason,
    /// The typed submit outcome, if the sub-agent submitted one.
    pub submit: Option<SubmitOutcome>,
    /// Whether the run is considered successful (per the mode's `parse_success`).
    pub success: bool,
    /// The compressed, storable text for this result.
    pub output: String,
}

/// Parameters for a single sub-agent run.
pub struct SubAgentRunParams<'a> {
    /// The parsed request (mode/step/instructions/scope/tier).
    pub req: &'a SubAgentRequest,
    /// The shared orchestration state (read for prompts/observations).
    pub ctx: &'a OrchestratedStepContext,
    /// The base tool registry to derive mode capabilities from.
    pub base_tools: &'a ToolRegistry,
    /// The tier-resolved decision backend for this run.
    pub next_turn: Arc<dyn NextTurnService>,
    /// Max sub-agent iterations for this run.
    pub max_steps: usize,
    /// Nesting depth this sub-agent runs at.
    pub depth: usize,
    /// Optional ACP client access for the sub-agent's tools.
    pub client: Option<Arc<dyn ClientAccess>>,
    /// Ordered workspace roots inherited from the orchestrator turn.
    pub workspace_roots: Vec<String>,
    /// Cancellation token shared with the orchestrator turn.
    pub cancel: &'a agent_core::CancellationToken,
    /// The session id to attribute this sub-agent's captured requests to.
    /// Reusing the parent's id groups every agent of a session together in the
    /// LLM inspector, with the [`AgentPath`] telling them apart.
    pub session_id: String,
    /// The [`AgentPath`] identifying this sub-agent within its session.
    pub agent_path: AgentPath,
    /// Optional observer propagated so the sub-agent's requests/responses are
    /// captured too, attributed to its `agent_path`.
    pub observer: Option<SharedTurnObserver>,
    /// Parent orchestrator round that issued `run_subagent`.
    pub parent_trace: Option<TraceContext>,
    /// Concrete `run_subagent` call that spawned this run.
    pub parent_tool_call_id: Option<ToolCallId>,
    /// Typed mode/step/attempt/tier labels for the debug tree.
    pub trace_metadata: AgentRunMetadata,
}

/// Runs a nested turn for a given [`SubAgentMode`].
pub struct OrchestratedSubAgentWorker {
    mode: Arc<dyn SubAgentMode>,
}

impl OrchestratedSubAgentWorker {
    /// Create a worker for the given mode.
    pub fn new(mode: Arc<dyn SubAgentMode>) -> Self {
        Self { mode }
    }

    /// Run the sub-agent to a stop reason and return its outcome.
    pub async fn run(&self, params: SubAgentRunParams<'_>) -> Result<SubAgentRunOutcome> {
        let SubAgentRunParams {
            req,
            ctx,
            base_tools,
            next_turn,
            max_steps,
            depth,
            client,
            workspace_roots,
            cancel,
            session_id,
            agent_path,
            observer,
            parent_trace,
            parent_tool_call_id,
            trace_metadata,
        } = params;

        let step_index = req.step_index;

        // 1. Derive the tool set and inject the submit action.
        let slot = new_slot();
        let mut tools = self.mode.tool_capabilities(base_tools);
        let submit_tool: Arc<dyn Tool> = match self.mode.submit_kind() {
            SubmitKind::Plain => Arc::new(SubmitTool::new(slot.clone())),
            SubmitKind::Review => Arc::new(SubmitReviewTool::new(slot.clone())),
        };
        tools.register(submit_tool);

        // 2. Build the nested session.
        let step_spec = ctx.proposal().and_then(|p| p.step(step_index).cloned());
        let issue = self.mode.build_issue_description(req, step_spec.as_ref());
        let observations = self.mode.build_pre_chat_observations(ctx, step_index);

        // Reuse the parent's session id (grouping) but tag this sub-agent with a
        // distinct `AgentPath` so its output is viewable separately.
        let mut session = SessionState::new(session_id);
        session.system_prompt = Some(self.mode.system_prompt(ctx, step_index));
        session.available_tools = tools.descriptors();
        session.workspace_roots = workspace_roots;
        let mut prompt = issue;
        if !observations.is_empty() {
            prompt.push_str("\n\nContext:\n");
            prompt.push_str(&observations);
        }
        session.push_user_text(prompt);

        // 3. Drive the engine to a stop reason.
        let mut engine = TurnEngine::new(next_turn, tools)
            .with_max_iterations(max_steps)
            .with_depth(depth)
            .with_agent_path(agent_path)
            .with_trace_metadata(trace_metadata);
        if let (Some(parent), Some(tool_call_id)) = (parent_trace, parent_tool_call_id) {
            engine = engine.with_trace_parent(parent, tool_call_id.0);
        }
        if let Some(client) = client {
            engine = engine.with_client(client);
        }
        if let Some(observer) = observer {
            engine = engine.with_observer(observer);
        }

        let mut sink = CapturingSink::default();
        let stop_reason = engine.run_prompt(&mut session, &mut sink, cancel).await?;

        // 4. Read the submit slot and compute success.
        let submit = slot.lock().expect("submit slot").clone();
        let success = submit
            .as_ref()
            .map(|s| self.mode.parse_success(s))
            .unwrap_or(false);
        let output = match &submit {
            Some(s) => self.mode.compress_result(s),
            None => sink.text.clone(),
        };

        Ok(SubAgentRunOutcome {
            stop_reason,
            submit,
            success,
            output,
        })
    }
}

/// An [`UpdateSink`] that captures assistant-visible text from the nested turn.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{PlanProposal, PlanStepSpec, StepResult};
    use crate::mode::{code_mode, ExecutorBehavior, ReviewMode, ReviewerBehavior};
    use agent_core::{
        CancellationToken, StopReason, ToolCallId, TurnEvent, TurnObservation, TurnObserver,
    };
    use std::sync::Mutex;
    use turn_replay::ReplayTurnService;

    fn ctx_with_step() -> Arc<OrchestratedStepContext> {
        OrchestratedStepContext::with_proposal(PlanProposal::new(
            "Build X",
            vec![PlanStepSpec::new("Implement X", "do it")],
        ))
    }

    #[tokio::test]
    async fn executor_runs_submits_and_reports_success() {
        #[derive(Default)]
        struct RequestRecorder(Mutex<Vec<TurnObservation>>);
        impl TurnObserver for RequestRecorder {
            fn on_turn_request(&self, observation: &TurnObservation) {
                self.0.lock().unwrap().push(observation.clone());
            }
        }
        // The nested turn immediately calls `submit`, then finishes.
        let replay = Arc::new(ReplayTurnService::new(vec![
            vec![
                TurnEvent::TextDelta("working".into()),
                TurnEvent::ToolCallRequested {
                    id: ToolCallId::new("c1"),
                    name: "submit".into(),
                    arguments: serde_json::json!({ "output": "implemented X" }),
                },
                TurnEvent::TurnFinished {
                    stop_reason: StopReason::ToolUse,
                },
            ],
            vec![TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            }],
        ]));

        let mode = Arc::new(code_mode(Arc::new(ExecutorBehavior)));
        let worker = OrchestratedSubAgentWorker::new(mode);
        let ctx = ctx_with_step();
        let base = ToolRegistry::new();
        let cancel = CancellationToken::new();
        let req = SubAgentRequest {
            mode: "code".into(),
            step_index: 0,
            instructions: None,
            scope: None,
            model_tier: None,
        };
        let recorder = Arc::new(RequestRecorder::default());

        let outcome = worker
            .run(SubAgentRunParams {
                req: &req,
                ctx: &ctx,
                base_tools: &base,
                next_turn: replay,
                max_steps: 10,
                depth: 1,
                client: None,
                workspace_roots: vec!["/workspace/project".into(), "/workspace/shared".into()],
                cancel: &cancel,
                session_id: "sess-test".into(),
                agent_path: AgentPath::root().child("code#1"),
                observer: Some(recorder.clone()),
                parent_trace: None,
                parent_tool_call_id: None,
                trace_metadata: AgentRunMetadata::default(),
            })
            .await
            .unwrap();

        assert!(outcome.success);
        assert_eq!(outcome.output, "implemented X");
        assert_eq!(outcome.submit.unwrap().output, "implemented X");
        let requests = recorder.0.lock().unwrap();
        assert!(requests[0].tools.iter().any(|tool| tool.name == "submit"));
        assert_eq!(
            requests[0].workspace_roots,
            vec!["/workspace/project", "/workspace/shared"]
        );
    }

    #[tokio::test]
    async fn reviewer_reports_verdict() {
        let replay = Arc::new(ReplayTurnService::new(vec![
            vec![
                TurnEvent::ToolCallRequested {
                    id: ToolCallId::new("r1"),
                    name: "submit_review".into(),
                    arguments: serde_json::json!({ "approved": true, "reasons": "looks good" }),
                },
                TurnEvent::TurnFinished {
                    stop_reason: StopReason::ToolUse,
                },
            ],
            vec![TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            }],
        ]));

        let ctx = ctx_with_step();
        // Seed a prior executor result so the review precondition is satisfiable
        // (the worker itself does not check preconditions; that is action.rs).
        ctx.store_result(StepResult {
            mode_id: "code".into(),
            step_index: 0,
            success: true,
            output: "did it".into(),
        });

        let mode = Arc::new(ReviewMode::new(Arc::new(ReviewerBehavior)));
        let worker = OrchestratedSubAgentWorker::new(mode);
        let base = ToolRegistry::new();
        let cancel = CancellationToken::new();
        let req = SubAgentRequest {
            mode: "review".into(),
            step_index: 0,
            instructions: None,
            scope: None,
            model_tier: None,
        };

        let outcome = worker
            .run(SubAgentRunParams {
                req: &req,
                ctx: &ctx,
                base_tools: &base,
                next_turn: replay,
                max_steps: 10,
                depth: 1,
                client: None,
                workspace_roots: Vec::new(),
                cancel: &cancel,
                session_id: "sess-test".into(),
                agent_path: AgentPath::root().child("code#1"),
                observer: None,
                parent_trace: None,
                parent_tool_call_id: None,
                trace_metadata: AgentRunMetadata::default(),
            })
            .await
            .unwrap();

        assert!(outcome.success);
        assert!(outcome.output.contains("approved=true"));
        assert!(outcome.output.contains("looks good"));
    }

    #[test]
    fn fn_model_resolver_maps_tier() {
        let backend: Arc<dyn NextTurnService> = Arc::new(ReplayTurnService::single(vec![]));
        let b = backend.clone();
        let resolver = FnModelResolver(move |_tier| b.clone());
        let resolved = resolver.resolve(ModelTier::High);
        assert!(Arc::ptr_eq(&resolved, &backend));
    }
}
