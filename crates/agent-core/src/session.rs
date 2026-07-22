//! Session state and turn context data models.
//!
//! Conversation history is stored in the strongly-typed [`crate::history`] model
//! ([`Conversation`]), never as raw provider messages. Converting that typed
//! history into a provider-specific request is the job of the
//! [`crate::compiler::LlmCompiler`] seam.

use serde::{Deserialize, Serialize};

use crate::event::{TokenUsage, ToolCallId};
use crate::history::{Conversation, HistoryEntry, KnownTool, ThinkingRecord};
use crate::todo::TodoList;
use crate::trace::TraceContext;

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
    /// Canonical todo state. Missing in legacy records and defaults to empty.
    #[serde(default)]
    pub todo_list: TodoList,
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
    /// Canonical, validated todo state for this session.
    pub todo_list: TodoList,
}

impl SessionState {
    /// Create a new empty session with the given id.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            ..Default::default()
        }
    }

    /// Replace the ordered workspace roots for this session.
    ///
    /// The primary working directory is always stored at index `0`; all
    /// additional roots retain the order supplied by the client.
    pub fn set_workspace<I, S>(&mut self, cwd: impl Into<String>, additional_roots: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.workspace_roots.clear();
        self.workspace_roots.push(cwd.into());
        self.workspace_roots
            .extend(additional_roots.into_iter().map(Into::into));
    }

    /// Return the session's primary working directory, when one is configured.
    #[must_use]
    pub fn working_directory(&self) -> Option<&str> {
        self.workspace_roots.first().map(String::as_str)
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

    /// Append an assistant extended-thinking block.
    pub fn push_thinking(&mut self, record: ThinkingRecord) {
        self.history.push_thinking(record);
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
            todo_list: self.todo_list.clone(),
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
            todo_list: record.todo_list,
        }
    }

    /// Build an immutable [`TurnContext`] snapshot from the current state.
    pub fn turn_context(&self) -> TurnContext {
        TurnContext {
            history: self.history.clone(),
            available_tools: self.available_tools.clone(),
            workspace_roots: self.workspace_roots.clone(),
            system_prompt: self.system_prompt.clone(),
            todo_list: self.todo_list.clone(),
            session_id: Some(self.session_id.clone()),
            trace: None,
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
    /// Ordered workspace roots; index `0` is the working directory.
    pub workspace_roots: Vec<String>,
    /// Optional system prompt.
    pub system_prompt: Option<String>,
    /// Snapshot of the session's canonical todo state.
    pub todo_list: TodoList,
    /// The id of the session this context belongs to, when known.
    ///
    /// Carried so that observability seams (e.g. an LLM-request inspector) can
    /// attribute a decision-layer call to a specific session. It is `None` for
    /// ad-hoc contexts constructed outside a [`SessionState`].
    pub session_id: Option<String>,
    /// Correlation identity assigned by the engine for the current round.
    pub trace: Option<TraceContext>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PlanStepStatus, SessionStateSnapshot, TodoItem, TodoItemId, TodoList};

    fn todo_list() -> TodoList {
        TodoList::new(vec![TodoItem {
            id: TodoItemId::new("verify").unwrap(),
            content: "Verify behavior".to_string(),
            status: PlanStepStatus::InProgress,
        }])
        .unwrap()
    }

    #[test]
    fn todo_state_roundtrips_through_record_and_context() {
        let mut state = SessionState::new("session-todo");
        state.todo_list = todo_list();

        assert_eq!(state.turn_context().todo_list, state.todo_list);
        let restored = SessionState::from_record(state.to_record(), Vec::new());
        assert_eq!(restored.todo_list, state.todo_list);
    }

    #[test]
    fn workspace_helpers_preserve_primary_and_additional_root_order() {
        let mut state = SessionState::new("session-workspace");
        state.set_workspace(
            "/workspace/project",
            ["/workspace/shared", "/workspace/generated"],
        );

        assert_eq!(state.working_directory(), Some("/workspace/project"));
        assert_eq!(
            state.workspace_roots,
            vec![
                "/workspace/project",
                "/workspace/shared",
                "/workspace/generated"
            ]
        );
        assert_eq!(state.turn_context().workspace_roots, state.workspace_roots);
    }

    #[test]
    fn empty_and_legacy_workspace_has_no_working_directory() {
        let state = SessionState::new("session-empty");
        assert_eq!(state.working_directory(), None);

        let restored = SessionState::from_record(state.to_record(), Vec::new());
        assert_eq!(restored.working_directory(), None);
        assert!(restored.turn_context().workspace_roots.is_empty());
    }

    #[test]
    fn legacy_record_without_todo_list_loads_empty() {
        let record: SessionRecord = serde_json::from_value(serde_json::json!({
            "version": SESSION_RECORD_VERSION,
            "session_id": "legacy",
            "workspace_roots": [],
            "history": {"entries": []},
            "system_prompt": null,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0
            }
        }))
        .expect("legacy records must remain loadable");

        assert!(record.todo_list.items.is_empty());
        assert!(SessionState::from_record(record, Vec::new())
            .todo_list
            .items
            .is_empty());
    }

    #[test]
    fn session_snapshot_serializes_todo_ids_content_and_status() {
        let mut state = SessionState::new("session-todo");
        state.todo_list = todo_list();
        let value = serde_json::to_value(SessionStateSnapshot::from_state(&state)).unwrap();

        assert_eq!(value["todo_list"]["items"][0]["id"], "verify");
        assert_eq!(value["todo_list"]["items"][0]["content"], "Verify behavior");
        assert_eq!(value["todo_list"]["items"][0]["status"], "in_progress");
    }
}
