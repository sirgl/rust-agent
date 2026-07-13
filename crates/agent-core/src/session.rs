//! Session state and turn context data models.
//!
//! Conversation history is stored in the strongly-typed [`crate::history`] model
//! ([`Conversation`]), never as raw provider messages. Converting that typed
//! history into a provider-specific request is the job of the
//! [`crate::compiler::LlmCompiler`] seam.

use crate::event::ToolCallId;
use crate::history::{Conversation, HistoryEntry, KnownTool};

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
