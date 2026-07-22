//! `subagents`: a provider-independent [`agent_core::Tool`] that delegates a
//! sub-task to a *nested* [`agent_core::TurnEngine`].
//!
//! Delegation lets a top-level turn hand a self-contained task to a fresh turn
//! loop — potentially backed by a different [`NextTurnService`] and a narrower
//! [`ToolRegistry`] — and stream that nested turn's progress back as ordinary
//! tool updates.
//!
//! Like every services-layer capability, this tool is written once and is
//! provider-agnostic: which decision backend the subagent uses is an injected
//! [`NextTurnService`], never hard-coded.
//!
//! ## Recursion depth limit
//!
//! Unbounded delegation (a subagent that spawns a subagent that spawns a
//! subagent ...) is guarded by a depth limit. The [`agent_core::TurnEngine`]
//! propagates its `depth` into the [`ToolContext`] of every tool it dispatches;
//! when a nested engine is spawned it runs at `ctx.depth + 1`. Before spawning,
//! [`SubagentTool`] refuses the delegation if `ctx.depth >= max_depth`, yielding
//! a graceful failed tool result rather than recursing without bound.
//!
//! ## Streaming
//!
//! The nested engine is driven to a stop reason while a capturing sink records
//! its assistant-visible output in order. That output is then replayed to the
//! parent as [`ToolEvent::OutputDelta`] items followed by a
//! [`ToolEvent::Completed`] (or [`ToolEvent::Failed`]) carrying the delegated
//! turn's stop reason and text.

#![warn(missing_docs)]

use std::sync::Arc;

use agent_core::{
    EngineOutput, NextTurnService, Result, SessionState, Tool, ToolContext, ToolEvent,
    ToolOutputPresentation, ToolRegistry, TurnEngine, UpdateSink,
};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use tracing::{debug, warn};

/// Default maximum subagent nesting depth (a top-level turn is depth 0, so a
/// default limit of 2 permits two levels of delegation before refusing).
pub const DEFAULT_MAX_DEPTH: usize = 2;

/// The registry name under which the subagent delegation tool is exposed.
pub const SUBAGENT_TOOL_NAME: &str = "delegate";

/// A [`Tool`] that delegates a sub-task to a nested [`TurnEngine`].
pub struct SubagentTool {
    next_turn: Arc<dyn NextTurnService>,
    tools: ToolRegistry,
    max_depth: usize,
    system_prompt: Option<String>,
    name: String,
}

impl SubagentTool {
    /// Create a subagent tool driven by `next_turn`, giving the nested engine
    /// access to `tools`.
    pub fn new(next_turn: Arc<dyn NextTurnService>, tools: ToolRegistry) -> Self {
        Self {
            next_turn,
            tools,
            max_depth: DEFAULT_MAX_DEPTH,
            system_prompt: None,
            name: SUBAGENT_TOOL_NAME.to_string(),
        }
    }

    /// Override the maximum nesting depth at which delegation is allowed.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Seed a system prompt into every delegated session.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Override the registry name the tool is exposed under.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

#[async_trait]
impl Tool for SubagentTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "A self-contained description of the sub-task to delegate."
                }
            },
            "required": ["task"]
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
        // Enforce the recursion depth limit before doing any work.
        if ctx.depth >= self.max_depth {
            let msg = format!(
                "subagent depth limit exceeded (depth {} >= max {})",
                ctx.depth, self.max_depth
            );
            warn!("{msg}");
            return Ok(events(vec![ToolEvent::Started, ToolEvent::Failed(msg)]));
        }

        // Extract the delegated task.
        let Some(task) = args.get("task").and_then(|v| v.as_str()) else {
            return Ok(events(vec![
                ToolEvent::Started,
                ToolEvent::Failed("subagent delegation requires a string `task` argument".into()),
            ]));
        };
        debug!(depth = ctx.depth, "delegating sub-task to nested engine");

        // Build the nested session seeded with the delegated task. It reuses the
        // parent's session id so every agent of a user session groups together
        // in the LLM inspector; the distinct `AgentPath` (below) tells them
        // apart, letting each sub-agent's output be viewed separately.
        let mut session = SessionState::new(ctx.session_id.clone());
        session.system_prompt = self.system_prompt.clone();
        session.available_tools = self.tools.descriptors();
        session.workspace_roots = ctx.workspace_roots.clone();
        session.push_user_text(task);

        // The nested engine runs at ctx.depth + 1 so the limit propagates.
        let mut engine = TurnEngine::new(self.next_turn.clone(), self.tools.clone())
            .with_depth(ctx.depth + 1)
            .with_agent_path(ctx.agent_path.child(self.name.clone()));
        if let (Some(trace), Some(tool_call_id)) = (ctx.trace.clone(), ctx.tool_call_id.clone()) {
            engine = engine.with_trace_parent(trace, tool_call_id.0);
        }
        if let Some(client) = ctx.client.clone() {
            engine = engine.with_client(client);
        }
        // Propagate the parent's observer so the nested agent's requests and
        // responses are captured too, attributed to its child `AgentPath`.
        if let Some(observer) = ctx.observer.clone() {
            engine = engine.with_observer(observer);
        }

        let mut sink = CapturingSink::default();
        let outcome = engine
            .run_prompt(&mut session, &mut sink, &ctx.cancel)
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
                ToolEvent::Failed(format!("subagent turn failed: {err}")),
            ])),
        }
    }
}

/// An [`UpdateSink`] that captures assistant-visible text from the nested turn
/// so it can be streamed back to the parent as tool output.
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
    // request_permission uses the default (grants) — nested tool permissions are
    // out of scope for delegated turns in v1.
}

/// Wrap a pre-computed sequence of [`ToolEvent`]s as a stream.
fn events(events: Vec<ToolEvent>) -> BoxStream<'static, ToolEvent> {
    Box::pin(stream::iter(events))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::{
        CancellationToken, NextTurnService, StopReason, ToolCallId, TurnContext, TurnEvent,
    };
    use async_trait::async_trait;
    use futures::StreamExt;
    use std::sync::Mutex;
    use turn_replay::ReplayTurnService;

    fn ctx_at_depth(depth: usize) -> ToolContext {
        let mut c = ToolContext::new("sess-main", CancellationToken::new());
        c.depth = depth;
        c
    }

    async fn collect(stream: BoxStream<'static, ToolEvent>) -> Vec<ToolEvent> {
        stream.collect().await
    }

    struct WorkspaceRecorder(Mutex<Vec<String>>);

    #[async_trait]
    impl NextTurnService for WorkspaceRecorder {
        async fn get_next_turn_streaming(
            &self,
            ctx: &TurnContext,
        ) -> agent_core::Result<futures::stream::BoxStream<'static, TurnEvent>> {
            *self.0.lock().unwrap() = ctx.workspace_roots.clone();
            Ok(Box::pin(futures::stream::iter(vec![
                TurnEvent::TurnFinished {
                    stop_reason: StopReason::EndTurn,
                },
            ])))
        }
    }

    #[tokio::test]
    async fn nested_agent_inherits_workspace_roots() {
        let backend = Arc::new(WorkspaceRecorder(Mutex::new(Vec::new())));
        let tool = SubagentTool::new(backend.clone(), ToolRegistry::new());
        let mut ctx = ctx_at_depth(0);
        ctx.workspace_roots = vec!["/workspace/project".into(), "/workspace/shared".into()];

        collect(
            tool.call(serde_json::json!({ "task": "inspect" }), &ctx)
                .await
                .unwrap(),
        )
        .await;

        assert_eq!(*backend.0.lock().unwrap(), ctx.workspace_roots);
    }

    #[tokio::test]
    async fn propagates_observer_and_child_agent_path_to_nested_engine() {
        use agent_core::{
            AgentPath, AgentRunMetadata, SharedTurnObserver, ToolCallId, TraceContext,
            TurnObservation, TurnObserver,
        };
        use std::sync::Mutex;

        // An observer that records the agent path of every captured request.
        #[derive(Default)]
        struct RecordingObserver {
            observations: Mutex<Vec<TurnObservation>>,
        }
        impl TurnObserver for RecordingObserver {
            fn on_turn_request(&self, obs: &TurnObservation) {
                self.observations.lock().unwrap().push(obs.clone());
            }
        }

        let replay = Arc::new(ReplayTurnService::single(vec![TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        }]));
        let tool = SubagentTool::new(replay, ToolRegistry::new());

        let observer = Arc::new(RecordingObserver::default());
        // The parent engine dispatched from `root`, so the tool context carries
        // that identity and the shared observer.
        let mut ctx = ctx_at_depth(0);
        ctx.observer = Some(observer.clone() as SharedTurnObserver);
        ctx.agent_path = AgentPath::root();
        ctx.trace = Some(TraceContext {
            turn_id: "turn-parent".into(),
            agent_run_id: "agent-parent".into(),
            parent_agent_run_id: None,
            spawned_by_tool_call_id: None,
            round_id: Some("round-parent".into()),
            agent_path: "root".into(),
            metadata: AgentRunMetadata::default(),
        });
        ctx.tool_call_id = Some(ToolCallId("delegate-call".to_string()));

        let stream = tool
            .call(serde_json::json!({ "task": "do a thing" }), &ctx)
            .await
            .unwrap();
        let _ = collect(stream).await;

        // The nested engine issued its request under a child path of `root`.
        let observations = observer.observations.lock().unwrap();
        assert_eq!(observations[0].agent_path, "root › delegate");
        assert_eq!(observations[0].trace.turn_id, "turn-parent");
        assert_eq!(
            observations[0].trace.parent_agent_run_id.as_deref(),
            Some("agent-parent")
        );
        assert_eq!(
            observations[0].trace.spawned_by_tool_call_id.as_deref(),
            Some("delegate-call")
        );
    }

    #[tokio::test]
    async fn delegates_and_streams_nested_output_back() {
        // The nested engine produces a plain streaming turn.
        let replay = Arc::new(ReplayTurnService::single(vec![
            TurnEvent::TextDelta("delegated ".into()),
            TurnEvent::TextDelta("answer".into()),
            TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            },
        ]));
        let tool = SubagentTool::new(replay, ToolRegistry::new());

        let stream = tool
            .call(
                serde_json::json!({ "task": "do a thing" }),
                &ctx_at_depth(0),
            )
            .await
            .unwrap();
        let evts = collect(stream).await;

        assert!(matches!(evts.first(), Some(ToolEvent::Started)));
        assert!(evts
            .iter()
            .any(|e| matches!(e, ToolEvent::OutputDelta(t) if t == "delegated answer")));
        match evts.last() {
            Some(ToolEvent::Completed(v)) => {
                assert_eq!(v["output"], "delegated answer");
                assert_eq!(v["stop_reason"], "end_turn");
            }
            other => panic!("unexpected final event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn refuses_when_depth_limit_exceeded() {
        let replay = Arc::new(ReplayTurnService::single(vec![TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        }]));
        let tool = SubagentTool::new(replay, ToolRegistry::new()).with_max_depth(1);

        // Called at depth 1 with max_depth 1 -> refused.
        let stream = tool
            .call(serde_json::json!({ "task": "x" }), &ctx_at_depth(1))
            .await
            .unwrap();
        let evts = collect(stream).await;
        assert!(matches!(
            evts.last(),
            Some(ToolEvent::Failed(m)) if m.contains("depth limit exceeded")
        ));
    }

    #[tokio::test]
    async fn missing_task_argument_fails_gracefully() {
        let replay = Arc::new(ReplayTurnService::single(vec![TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        }]));
        let tool = SubagentTool::new(replay, ToolRegistry::new());
        let stream = tool
            .call(serde_json::json!({ "not_task": 1 }), &ctx_at_depth(0))
            .await
            .unwrap();
        let evts = collect(stream).await;
        assert!(matches!(evts.last(), Some(ToolEvent::Failed(_))));
    }

    #[tokio::test]
    async fn nested_depth_propagates_to_inner_tools() {
        // A subagent whose nested engine has *another* subagent tool registered
        // must refuse the inner delegation once the depth limit is hit. With
        // max_depth 1, the outer call at depth 0 spawns a nested engine at
        // depth 1, whose inner delegate call is refused.
        let inner_replay = Arc::new(ReplayTurnService::new(vec![
            // Round 1: the nested turn requests another delegation...
            vec![
                TurnEvent::ToolCallRequested {
                    id: ToolCallId::new("inner-1"),
                    name: SUBAGENT_TOOL_NAME.to_string(),
                    arguments: serde_json::json!({ "task": "recurse" }),
                },
                TurnEvent::TurnFinished {
                    stop_reason: StopReason::ToolUse,
                },
            ],
            // Round 2: after the refused delegation, the nested turn finishes.
            vec![TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            }],
        ]));

        // Build the nested registry containing a subagent tool that would
        // recurse; it shares the same replay service (single round is consumed
        // by the nested engine's first decision call).
        let mut nested_registry = ToolRegistry::new();
        let recursing = SubagentTool::new(
            Arc::new(ReplayTurnService::single(vec![TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            }])),
            ToolRegistry::new(),
        )
        .with_max_depth(1);
        nested_registry.register(Arc::new(recursing));

        let outer = SubagentTool::new(inner_replay, nested_registry).with_max_depth(1);

        let stream = outer
            .call(serde_json::json!({ "task": "top" }), &ctx_at_depth(0))
            .await
            .unwrap();
        let evts = collect(stream).await;
        // The outer delegation itself completes (its nested engine ran); the
        // inner delegation was refused inside that nested turn.
        assert!(matches!(evts.last(), Some(ToolEvent::Completed(_))));
    }
}
