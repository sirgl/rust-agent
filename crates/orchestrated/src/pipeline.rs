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
    AgentPath, CancellationToken, ClientAccess, ContentBlock, EngineOutput, HistoryEntry,
    NextTurnService, Result, SessionState, SharedTurnObserver, StopReason, Tool, ToolContext,
    ToolEvent, ToolRegistry, TurnInbox, UpdateSink, THOUGHT_NUDGE,
};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use tracing::debug;

use crate::context::{OrchestratedStepContext, PlanProposal, PlanStepSpec};
use crate::orchestrator::{Orchestrator, OrchestratorBuilder};
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

/// A long-lived orchestration runtime for one frontend session.
///
/// Unlike [`run_pipeline`], this value retains both the orchestrator's typed
/// conversation and its [`OrchestratedStepContext`] across user prompts.
pub struct PipelineSession {
    orchestrator: Orchestrator,
    session: SessionState,
}

impl PipelineSession {
    /// Build an observed pipeline session seeded from the first user goal.
    #[allow(clippy::too_many_arguments)]
    pub fn new_observed(
        initial_goal: &str,
        orchestrator_backend: Arc<dyn NextTurnService>,
        resolver: Arc<dyn ModelResolver>,
        base_tools: ToolRegistry,
        observer: Option<SharedTurnObserver>,
        agent_path: AgentPath,
        session_id: Option<String>,
    ) -> Self {
        Self::new_observed_inner(
            initial_goal,
            orchestrator_backend,
            resolver,
            base_tools,
            observer,
            agent_path,
            session_id,
            None,
            None,
        )
    }

    /// Build an observed pipeline session with a live steering inbox.
    #[allow(clippy::too_many_arguments)]
    pub fn new_observed_with_inbox(
        initial_goal: &str,
        orchestrator_backend: Arc<dyn NextTurnService>,
        resolver: Arc<dyn ModelResolver>,
        base_tools: ToolRegistry,
        observer: Option<SharedTurnObserver>,
        agent_path: AgentPath,
        session_id: Option<String>,
        inbox: TurnInbox,
        client: Option<Arc<dyn ClientAccess>>,
    ) -> Self {
        Self::new_observed_inner(
            initial_goal,
            orchestrator_backend,
            resolver,
            base_tools,
            observer,
            agent_path,
            session_id,
            Some(inbox),
            client,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_observed_inner(
        initial_goal: &str,
        orchestrator_backend: Arc<dyn NextTurnService>,
        resolver: Arc<dyn ModelResolver>,
        base_tools: ToolRegistry,
        observer: Option<SharedTurnObserver>,
        agent_path: AgentPath,
        session_id: Option<String>,
        inbox: Option<TurnInbox>,
        client: Option<Arc<dyn ClientAccess>>,
    ) -> Self {
        let context = OrchestratedStepContext::with_proposal(implicit_proposal(initial_goal));
        let mut builder = OrchestratorBuilder::new(orchestrator_backend, resolver)
            .with_base_tools(base_tools)
            .with_context(context)
            .with_observer(observer)
            .with_agent_path(agent_path);
        if let Some(session_id) = session_id {
            builder = builder.with_session_id(session_id);
        }
        if let Some(inbox) = inbox {
            builder = builder.with_inbox(inbox);
        }
        if let Some(client) = client {
            builder = builder.with_client(client);
        }
        let orchestrator = builder.build();
        let session = orchestrator.empty_session();
        Self {
            orchestrator,
            session,
        }
    }

    /// Seed conversation memory accumulated by another frontend mode.
    ///
    /// The orchestrator keeps its own system prompt and tool descriptors, but
    /// inherits the user-visible conversation, workspace, and cumulative usage
    /// so switching an existing ACP session into orchestration does not discard
    /// earlier goals or clarifications.
    pub fn inherit_state(&mut self, source: &SessionState) {
        let prior_user_context = source
            .history
            .entries
            .iter()
            .filter_map(|entry| match entry {
                HistoryEntry::User(message) => Some(&message.content),
                _ => None,
            })
            .flatten()
            .filter_map(|block| match block {
                ContentBlock::Text { text } if text != THOUGHT_NUDGE => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !prior_user_context.is_empty() {
            let current_goal = self
                .orchestrator
                .context()
                .proposal()
                .map(|proposal| proposal.goal)
                .unwrap_or_default();
            let effective_goal = format!(
                "Previous user context:\n{prior_user_context}\n\nCurrent request:\n{current_goal}"
            );
            self.orchestrator
                .context()
                .set_proposal(implicit_proposal(&effective_goal));
        }
        self.session.workspace_roots = source.workspace_roots.clone();
        self.session.history = source.history.clone();
        self.session.usage = source.usage;
        self.session.todo_list = source.todo_list.clone();
    }

    /// Append a user prompt and continue the same orchestration run history.
    pub async fn run_prompt(
        &mut self,
        prompt: impl Into<String>,
        sink: &mut dyn UpdateSink,
        cancel: &CancellationToken,
    ) -> Result<StopReason> {
        self.session.push_user_text(prompt.into());
        self.orchestrator
            .engine()
            .run_prompt(&mut self.session, sink, cancel)
            .await
    }

    /// The persistent typed orchestrator state used by the inspector/store.
    pub fn session(&self) -> &SessionState {
        &self.session
    }

    /// Results, attempts and proposal shared by all prompts in this session.
    pub fn context(&self) -> &Arc<OrchestratedStepContext> {
        self.orchestrator.context()
    }
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
    run_pipeline_observed(
        goal,
        orchestrator_backend,
        resolver,
        base_tools,
        sink,
        cancel,
        None,
        AgentPath::root().child("orchestrator"),
        None,
        None,
    )
    .await
}

/// Like [`run_pipeline`], but also attaches an observer and an explicit agent
/// identity so the orchestrator turn and every sub-agent it spawns are captured
/// by the LLM inspector, viewable separately, grouped under `session_id`.
///
/// `observer` is the provider-neutral [`agent_core::TurnObserver`] (typically
/// the inspector); `agent_path` identifies the orchestrator agent (sub-agents
/// derive children of it); `session_id` groups every agent under one session
/// (defaults to the orchestrator's own id when `None`).
#[allow(clippy::too_many_arguments)]
pub async fn run_pipeline_observed(
    goal: &str,
    orchestrator_backend: Arc<dyn NextTurnService>,
    resolver: Arc<dyn ModelResolver>,
    base_tools: ToolRegistry,
    sink: &mut dyn UpdateSink,
    cancel: &CancellationToken,
    observer: Option<SharedTurnObserver>,
    agent_path: AgentPath,
    session_id: Option<String>,
    client: Option<Arc<dyn ClientAccess>>,
) -> Result<StopReason> {
    debug!(goal = %goal, agent = %agent_path, "run_pipeline: starting orchestration");
    let mut pipeline = PipelineSession::new_observed_inner(
        goal,
        orchestrator_backend,
        resolver,
        base_tools,
        observer,
        agent_path,
        session_id,
        None,
        client,
    );
    pipeline.run_prompt(goal, sink, cancel).await
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
        // Propagate the parent's observer and a child agent path so the nested
        // orchestrator and its sub-agents are captured separately in the
        // inspector, grouped under the same session as the calling agent.
        let outcome = run_pipeline_observed(
            goal,
            (self.factory)(),
            self.resolver.clone(),
            self.base_tools.clone(),
            &mut sink,
            &ctx.cancel,
            ctx.observer.clone(),
            ctx.agent_path.child(ORCHESTRATE_TOOL_NAME),
            Some(ctx.session_id.clone()),
            ctx.client.clone(),
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
    async fn pipeline_session_preserves_step_context_across_prompts() {
        let orchestrator: Arc<dyn NextTurnService> = Arc::new(ReplayTurnService::new(vec![
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
            vec![
                TurnEvent::ToolCallRequested {
                    id: ToolCallId::new("o2"),
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
        let mut pipeline = PipelineSession::new_observed(
            "From scratch",
            orchestrator,
            subagent_resolver(),
            ToolRegistry::new(),
            None,
            AgentPath::root().child("orchestrator"),
            Some("session-1".into()),
        );
        let mut prior_state = SessionState::new("session-1");
        prior_state.push_user_text("Build a thing");
        prior_state.push_user_text(THOUGHT_NUDGE);
        prior_state.todo_list = serde_json::from_value(serde_json::json!({
            "items": [{"id": "carry", "content": "Carry state", "status": "in_progress"}]
        }))
        .unwrap();
        pipeline.inherit_state(&prior_state);
        let proposal = pipeline.context().proposal().expect("implicit proposal");
        assert!(proposal.goal.contains("Build a thing"));
        assert!(proposal.goal.contains("From scratch"));
        assert!(!proposal.goal.contains(THOUGHT_NUDGE));
        assert_eq!(pipeline.session().todo_list, prior_state.todo_list);
        let mut sink = NullSink;
        let cancel = CancellationToken::new();

        pipeline
            .run_prompt("From scratch", &mut sink, &cancel)
            .await
            .unwrap();
        pipeline
            .run_prompt("Continue from scratch", &mut sink, &cancel)
            .await
            .unwrap();

        assert_eq!(pipeline.context().attempts("code", 0), 2);
        let user_messages = pipeline
            .session()
            .history
            .entries
            .iter()
            .filter(|entry| matches!(entry, agent_core::HistoryEntry::User(_)))
            .count();
        assert_eq!(user_messages, 4);
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
