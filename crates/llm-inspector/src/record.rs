//! The unified record model for a captured LLM request.
//!
//! A single decision round produces up to two correlated records:
//!
//! - a **typed** record ([`CapturedPayload::Typed`]) captured by the engine's
//!   provider-neutral [`agent_core::TurnObserver`] seam — the readable, typed
//!   view of the conversation (works for *every* backend); and
//! - a **provider** record ([`CapturedPayload::Provider`]) captured by a
//!   provider-level observer (e.g. Anthropic's `RequestObserver`) — the exact
//!   wire body sent over the network.
//!
//! Both flow through the same [`crate::Recorder`] pipeline and are shown together
//! in the UI, grouped by session.

use agent_core::{TraceContext, TraceEvent};
use serde::Serialize;

/// A tool advertised to the model, as shown in a typed record.
#[derive(Debug, Clone, Serialize)]
pub struct ToolInfo {
    /// The tool name.
    pub name: String,
    /// Whether the tool requires explicit user permission before executing.
    pub requires_permission: bool,
}

/// The kind-specific payload of a [`CapturedRequest`].
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CapturedPayload {
    /// A provider-neutral, strongly-typed request snapshot from the engine.
    Typed {
        /// The system prompt, if any.
        system_prompt: Option<String>,
        /// The tools advertised to the model.
        tools: Vec<ToolInfo>,
        /// The typed conversation history, serialized as JSON.
        history: serde_json::Value,
        /// Number of history entries.
        num_entries: usize,
        /// Subagent nesting depth of the issuing engine (`0` = top level).
        depth: usize,
    },
    /// A provider-specific compiled wire body (the exact JSON sent).
    Provider {
        /// The provider name (e.g. `"anthropic"`).
        provider: String,
        /// The exact JSON body sent to the provider.
        body: serde_json::Value,
        /// Number of messages in the body (best-effort).
        num_messages: usize,
        /// Number of tools in the body (best-effort).
        num_tools: usize,
        /// Serialized size of the body in bytes.
        size_bytes: usize,
    },
    /// A provider-neutral, typed snapshot of what the model *replied* with for a
    /// single decision round (assistant text, tool calls, usage, stop reason).
    Response {
        /// The assistant-visible text produced this round.
        assistant_text: String,
        /// Number of characters of internal reasoning ("thinking") streamed.
        thinking_chars: usize,
        /// The tool calls the model requested, as `{name, arguments}` JSON.
        tool_calls: serde_json::Value,
        /// Token usage for this round, serialized as JSON.
        usage: serde_json::Value,
        /// The reason the round ended (`"end_turn"`, `"tool_use"`, ...), if known.
        stop_reason: Option<String>,
        /// An error message, when the round ended because of an error.
        error: Option<String>,
        /// Wall-clock duration of the round in milliseconds.
        duration_ms: u64,
    },
    /// An agent log line captured from `tracing`, for extra debugging context.
    Log {
        /// The log level (`"ERROR"`, `"WARN"`, `"INFO"`, `"DEBUG"`, `"TRACE"`).
        level: String,
        /// The event target (module path).
        target: String,
        /// The rendered log message.
        message: String,
        /// Any additional structured fields, as a JSON object.
        fields: serde_json::Value,
    },
    /// An append-only execution lifecycle event used to build the hierarchy.
    Trace {
        /// Typed event; unmatched starts are rendered as interrupted.
        event: TraceEvent,
    },
}

/// A captured snapshot of a session's persisted state.
///
/// Unlike a [`CapturedRequest`] — which is one entry in an append-only stream —
/// the inspector keeps only the *latest* session-state snapshot per session, so
/// the UI can always show the current memory of each live session (history,
/// cumulative usage, tools, mode) rather than a scrolling log of past states.
#[derive(Debug, Clone, Serialize)]
pub struct CapturedSessionState {
    /// The session this state belongs to.
    pub session_id: String,
    /// Unix timestamp in milliseconds when this snapshot was captured.
    pub timestamp_ms: u128,
    /// The current session mode (e.g. `chat`/`orchestrate`), when known.
    pub mode: Option<String>,
    /// Workspace root directories granted to the session.
    pub workspace_roots: Vec<String>,
    /// The system prompt guiding the assistant, if any.
    pub system_prompt: Option<String>,
    /// Cumulative token usage across every decision round, serialized as JSON.
    pub usage: serde_json::Value,
    /// The number of typed history entries retained.
    pub num_entries: usize,
    /// The tools currently available to the session.
    pub tools: Vec<ToolInfo>,
    /// The full typed conversation history, serialized as JSON.
    pub history: serde_json::Value,
    /// The canonical todo list, including stable IDs and statuses.
    pub todo_list: serde_json::Value,
}

/// A single captured LLM request, either typed or provider-specific.
#[derive(Debug, Clone, Serialize)]
pub struct CapturedRequest {
    /// Monotonic id, unique within the process (newest has the largest id).
    pub id: u64,
    /// Strict append order within this inspector process.
    ///
    /// Equal to `id` in v1, but named explicitly so readers never infer event
    /// ordering from wall-clock timestamps.
    pub sequence_no: u64,
    /// The session this request belongs to, or `"(unknown)"` when unattributed.
    pub session_id: String,
    /// The hierarchical identity of the agent this record belongs to (e.g.
    /// `root › orchestrator › code#1`), or `None` when not attributable to a
    /// specific agent (e.g. a raw provider body). Lets the UI show each agent's
    /// output separately within a session.
    pub agent_path: Option<String>,
    /// Stable causal identity. Absent on legacy/unattributed log records.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<TraceContext>,
    /// Unix timestamp in milliseconds when the request was captured.
    pub timestamp_ms: u128,
    /// The model the request targets, when known.
    pub model: Option<String>,
    /// The kind-specific payload.
    #[serde(flatten)]
    pub payload: CapturedPayload,
}

impl CapturedRequest {
    /// The short kind label (`"typed"` or `"provider"`), useful for logs/UI.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self.payload {
            CapturedPayload::Typed { .. } => "typed",
            CapturedPayload::Provider { .. } => "provider",
            CapturedPayload::Response { .. } => "response",
            CapturedPayload::Log { .. } => "log",
            CapturedPayload::Trace { .. } => "trace",
        }
    }
}
