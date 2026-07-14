//! Session state and turn context data models.
//!
//! Conversation history is stored in the strongly-typed [`crate::history`] model
//! ([`Conversation`]), never as raw provider messages. Converting that typed
//! history into a provider-specific request is the job of the
//! [`crate::compiler::LlmCompiler`] seam.

use serde::{Deserialize, Serialize};

use crate::event::{TokenUsage, ToolCallId};
use crate::history::{Conversation, HistoryEntry, KnownTool};

/// The current version of the persisted [`SessionRecord`] format.
pub const SESSION_RECORD_VERSION: u32 = 1;

/// A serializable snapshot of the *core* session state.
///
/// This captures everything needed to reconstruct a session's memory across
/// process restarts: the conversation history, cumulative usage, workspace
/// roots, and system prompt. It intentionally excludes `available_tools`, which
/// is re-derived from the live tool registry at load time (tool schemas depend
/// on the set of MCP servers reconnected during a `session/load`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionRecord {
    /// Version of the persisted record format.
    pub version: u32,
    /// The session identifier.
    pub session_id: String,
    /// Workspace root directories granted to the session.
    pub workspace_roots: Vec<String>,
    /// Strongly-typed conversation history.
    pub history: Conversation,
    /// Optional system prompt guiding the assistant.
    pub system_prompt: Option<String>,
    /// Cumulative token usage across every decision round in this session.
    pub usage: TokenUsage,
}

/// Descriptor for a tool that is available in a session.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDescriptor {
    /// Unique tool name.
    pub name: String,
    /// JSON schema describing the tool's arguments.
    pub schema: serde_json::Value,
    /// Whether the tool requires explicit user permission before executing.
    pub requires_permission: bool,
}

/// Mutable state associated with a single ACP session.
#[derive(Debug, Clone, Default)]
pub struct SessionState {
    /// The session identifier.
    pub session_id: String,
    /// Workspace root directories granted to the session.
    pub workspace_roots: Vec<String>,
    /// Strongly-typed conversation history.
    pub history: Conversation,
    /// Tool descriptors available to the decision layer.
    pub available_tools: Vec<ToolDescriptor>,
    /// Optional system prompt guiding the assistant.
    pub system_prompt: Option<String>,
    /// Cumulative token usage across every decision round in this session.
    pub usage: TokenUsage,
}

impl SessionState {
    /// Create a new empty session with the given id.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            ..Default::default()
        }
    }

    /// Append a raw history entry.
    pub fn push_entry(&mut self, entry: HistoryEntry) {
        self.history.push(entry);
    }

    /// Append a user text message.
    pub fn push_user_text(&mut self, text: impl Into<String>) {
        self.history.push_user_text(text);
    }

    /// Append an assistant text message.
    pub fn push_assistant_text(&mut self, text: impl Into<String>) {
        self.history.push_assistant_text(text);
    }

    /// Append a tool call.
    pub fn push_tool_call(&mut self, id: ToolCallId, tool: KnownTool) {
        self.history.push_tool_call(id, tool);
    }

    /// Append a tool result.
    pub fn push_tool_result(&mut self, id: ToolCallId, success: bool, text: impl Into<String>) {
        self.history.push_tool_result(id, success, text);
    }

    /// Append a previously-ingested/compiled fragment preserved for reuse.
    pub fn push_ingested(&mut self, source: impl Into<String>, payload: serde_json::Value) {
        self.history.push_ingested(source, payload);
    }

    /// Accumulate a token-usage record reported by the decision layer.
    pub fn add_usage(&mut self, usage: TokenUsage) {
        self.usage.add(usage);
    }

    /// Snapshot the *core* (serializable) state into a [`SessionRecord`].
    ///
    /// The `available_tools` field is deliberately not included; it is
    /// re-derived from the live tool registry when the record is loaded.
    pub fn to_record(&self) -> SessionRecord {
        SessionRecord {
            version: SESSION_RECORD_VERSION,
            session_id: self.session_id.clone(),
            workspace_roots: self.workspace_roots.clone(),
            history: self.history.clone(),
            system_prompt: self.system_prompt.clone(),
            usage: self.usage,
        }
    }

    /// Reconstruct a [`SessionState`] from a persisted [`SessionRecord`],
    /// re-deriving `available_tools` from the supplied tool descriptors.
    pub fn from_record(record: SessionRecord, available_tools: Vec<ToolDescriptor>) -> Self {
        Self {
            session_id: record.session_id,
            workspace_roots: record.workspace_roots,
            history: record.history,
            available_tools,
            system_prompt: record.system_prompt,
            usage: record.usage,
        }
    }

    /// Build an immutable [`TurnContext`] snapshot from the current state.
    pub fn turn_context(&self) -> TurnContext {
        TurnContext {
            history: self.history.clone(),
            available_tools: self.available_tools.clone(),
            system_prompt: self.system_prompt.clone(),
        }
    }
}

/// An immutable view of the conversation passed to a [`crate::NextTurnService`].
#[derive(Debug, Clone, Default)]
pub struct TurnContext {
    /// The strongly-typed conversation history so far.
    pub history: Conversation,
    /// Tool schemas available to the decision layer.
    pub available_tools: Vec<ToolDescriptor>,
    /// Optional system prompt.
    pub system_prompt: Option<String>,
}
