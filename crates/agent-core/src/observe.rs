//! Provider-neutral observability seam for the decision layer.
//!
//! The [`TurnObserver`] trait is notified — from inside the [`crate::TurnEngine`]
//! itself, before every call to the [`crate::NextTurnService`] — with a
//! strongly-typed snapshot of *what the agent is about to ask the model*
//! ([`TurnObservation`]): the system prompt, the available tools, and the full
//! typed conversation history.
//!
//! Because this seam lives in the engine rather than in any one backend, it
//! captures every turn regardless of the decision backend in use (Anthropic,
//! the deterministic replay backend, or any future provider). Provider-specific
//! wire bodies (e.g. the compiled Anthropic JSON) are captured separately by
//! provider-level observers; this seam is the readable, typed view.

use std::sync::Arc;

use serde::Serialize;

use crate::agent::AgentPath;
use crate::event::{StopReason, TokenUsage};
use crate::history::Conversation;
use crate::session::{SessionState, ToolDescriptor, TurnContext};
use crate::todo::TodoList;
use crate::trace::{TraceContext, TraceEvent};

/// A lightweight, serializable summary of a tool advertised to the model.
#[derive(Debug, Clone, Serialize)]
pub struct ToolSummary {
    /// The tool name.
    pub name: String,
    /// Whether the tool requires explicit user permission before executing.
    pub requires_permission: bool,
}

impl From<&ToolDescriptor> for ToolSummary {
    fn from(descriptor: &ToolDescriptor) -> Self {
        Self {
            name: descriptor.name.clone(),
            requires_permission: descriptor.requires_permission,
        }
    }
}

/// A provider-neutral, strongly-typed snapshot of a decision-layer request.
///
/// This is the readable, typed counterpart to a provider's raw wire body: it
/// carries the typed conversation history and tool set exactly as the engine
/// holds them, so an observer (e.g. the LLM inspector) can render the request in
/// terms the rest of the workspace already understands.
#[derive(Debug, Clone, Serialize)]
pub struct TurnObservation {
    /// Stable causal identity for this exact decision round.
    pub trace: TraceContext,
    /// The session this request belongs to, when known.
    pub session_id: Option<String>,
    /// The hierarchical identity of the agent issuing this request, rendered as
    /// a display string (e.g. `root › orchestrator › code#1`).
    pub agent_path: String,
    /// The system prompt guiding the assistant, if any.
    pub system_prompt: Option<String>,
    /// The tools advertised to the model for this turn.
    pub tools: Vec<ToolSummary>,
    /// Ordered workspace roots; index `0` is the working directory.
    pub workspace_roots: Vec<String>,
    /// The full typed conversation history sent to the decision layer.
    pub history: Conversation,
    /// The subagent nesting depth of the engine issuing the request (`0` = top).
    pub depth: usize,
}

impl TurnObservation {
    /// Build an observation from an immutable [`TurnContext`] and the issuing
    /// engine's [`AgentPath`] (whose depth is used as the nesting depth).
    #[must_use]
    pub fn from_context(ctx: &TurnContext, agent_path: &AgentPath, trace: TraceContext) -> Self {
        Self {
            trace,
            session_id: ctx.session_id.clone(),
            agent_path: agent_path.to_string(),
            system_prompt: ctx.system_prompt.clone(),
            tools: ctx.available_tools.iter().map(ToolSummary::from).collect(),
            workspace_roots: ctx.workspace_roots.clone(),
            history: ctx.history.clone(),
            depth: agent_path.depth(),
        }
    }
}

/// A single tool call the model asked for during a decision round.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallSummary {
    /// The tool name the model invoked.
    pub name: String,
    /// The raw JSON arguments the model supplied.
    pub arguments: serde_json::Value,
}

/// A provider-neutral, strongly-typed snapshot of what the model *replied* with
/// during a single decision round.
///
/// This is the read side counterpart to [`TurnObservation`]: where the request
/// snapshot captures what the agent asked, this captures what came back — the
/// assistant text, how much internal reasoning was produced, which tools were
/// requested, the reported token usage (including prompt-cache reuse), why the
/// round ended, and how long it took. It lets an observer (e.g. the LLM
/// inspector) show responses, usage/token-reuse and timing for every backend.
#[derive(Debug, Clone, Serialize)]
pub struct TurnResponseObservation {
    /// Stable causal identity shared with the corresponding request.
    pub trace: TraceContext,
    /// The session this response belongs to, when known.
    pub session_id: Option<String>,
    /// The hierarchical identity of the agent that produced this response,
    /// rendered as a display string (e.g. `root › orchestrator › code#1`).
    pub agent_path: String,
    /// The subagent nesting depth of the engine issuing the round (`0` = top).
    pub depth: usize,
    /// The assistant-visible text produced this round.
    pub assistant_text: String,
    /// Number of characters of internal reasoning ("thinking") streamed.
    pub thinking_chars: usize,
    /// The tool calls the model requested this round, in order.
    pub tool_calls: Vec<ToolCallSummary>,
    /// Token usage reported for this round (fresh, output, and cache tokens).
    pub usage: TokenUsage,
    /// Why the round ended, when known (`None` if the stream ended abruptly or
    /// errored).
    pub stop_reason: Option<StopReason>,
    /// An error message, when the round ended because of an error.
    pub error: Option<String>,
    /// Wall-clock duration of the round in milliseconds.
    pub duration_ms: u64,
}

/// A provider-neutral, serializable snapshot of the *persisted state* of a
/// session at a point in time.
///
/// Where [`TurnObservation`]/[`TurnResponseObservation`] capture a single
/// decision round, this captures the durable memory the agent keeps *between*
/// turns: the conversation history, cumulative token usage, workspace roots,
/// system prompt, current mode and available tools. It is the read-side view of
/// what a session would be persisted/restored as (see
/// [`crate::SessionRecord`]), letting an observer (e.g. the LLM inspector) show
/// the live session state — not just individual requests.
#[derive(Debug, Clone, Serialize)]
pub struct SessionStateSnapshot {
    /// The session identifier.
    pub session_id: String,
    /// The current session mode (e.g. `chat`/`orchestrate`), when known.
    pub mode: Option<String>,
    /// Workspace root directories granted to the session.
    pub workspace_roots: Vec<String>,
    /// The system prompt guiding the assistant, if any.
    pub system_prompt: Option<String>,
    /// Cumulative token usage across every decision round in this session.
    pub usage: TokenUsage,
    /// The number of typed history entries retained.
    pub num_entries: usize,
    /// The tools currently available to the session.
    pub tools: Vec<ToolSummary>,
    /// The full typed conversation history retained for the session.
    pub history: Conversation,
    /// Canonical todo state retained for the session.
    pub todo_list: TodoList,
}

impl SessionStateSnapshot {
    /// Build a snapshot from a live [`SessionState`] (mode left unset).
    #[must_use]
    pub fn from_state(state: &SessionState) -> Self {
        Self {
            session_id: state.session_id.clone(),
            mode: None,
            workspace_roots: state.workspace_roots.clone(),
            system_prompt: state.system_prompt.clone(),
            usage: state.usage,
            num_entries: state.history.len(),
            tools: state
                .available_tools
                .iter()
                .map(ToolSummary::from)
                .collect(),
            history: state.history.clone(),
            todo_list: state.todo_list.clone(),
        }
    }

    /// Set the current session mode, returning the updated snapshot.
    #[must_use]
    pub fn with_mode(mut self, mode: impl Into<String>) -> Self {
        self.mode = Some(mode.into());
        self
    }
}

/// Observes the provider-neutral, typed request the engine is about to send to
/// the decision layer, and the typed response it gets back.
///
/// Implementations must be cheap and non-blocking: the callbacks run inline on
/// the turn path and must never panic.
///
/// [`on_turn_request`]: TurnObserver::on_turn_request
pub trait TurnObserver: Send + Sync {
    /// Called once per decision round with the typed request snapshot.
    fn on_turn_request(&self, observation: &TurnObservation);

    /// Called once per decision round when it completes, with the typed
    /// response snapshot (assistant text, tool calls, usage, stop reason).
    ///
    /// Defaults to a no-op so existing observers keep working unchanged.
    fn on_turn_response(&self, _observation: &TurnResponseObservation) {}

    /// Called for append-only execution lifecycle events.
    fn on_trace_event(&self, _event: &TraceEvent) {}

    /// Called when a session's persisted state changes (e.g. after a session is
    /// created, a turn completes, or a session is loaded), with a snapshot of
    /// the current session state.
    ///
    /// Defaults to a no-op so existing observers keep working unchanged.
    fn on_session_state(&self, _snapshot: &SessionStateSnapshot) {}
}

/// A shared, optional [`TurnObserver`] handle carried by the engine.
pub type SharedTurnObserver = Arc<dyn TurnObserver>;
