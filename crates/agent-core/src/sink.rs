//! The [`UpdateSink`] trait: maps engine output to ACP `session/update`.

use async_trait::async_trait;

use crate::error::Result;
use crate::event::{PlanUpdate, ToolCallId};

/// The status of a tool call as surfaced to the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallStatus {
    /// Requested but not yet started.
    Pending,
    /// Currently executing.
    InProgress,
    /// Finished successfully.
    Completed,
    /// Finished with an error.
    Failed,
}

/// The provider-agnostic category of a tool.
///
/// Mirrors the ACP `ToolKind` so clients can pick appropriate icons and UI
/// treatment for a tool call. Kept in `agent-core` (rather than reusing the ACP
/// schema type) so the services and engine layers stay protocol-independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolKind {
    /// Reads data (e.g. a file).
    Read,
    /// Edits or writes data (e.g. a file).
    Edit,
    /// Deletes data.
    Delete,
    /// Moves or renames data.
    Move,
    /// Searches for data.
    Search,
    /// Executes a command or program.
    Execute,
    /// Performs internal reasoning.
    Think,
    /// Fetches data from an external source.
    Fetch,
    /// Any other, uncategorized tool.
    #[default]
    Other,
}

/// A file location touched by a tool call.
///
/// Enables "follow-along" features in clients (e.g. highlighting the file a
/// tool is reading or editing). Provider-agnostic mirror of the ACP
/// `ToolCallLocation`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallLocation {
    /// The absolute path being accessed or modified.
    pub path: String,
    /// Optional line number within the file.
    pub line: Option<u32>,
}

/// Text snapshots for one file changed by a tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileDiff {
    /// The absolute path of the changed file.
    pub path: String,
    /// The file content before the change. `None` means that the tool created the file.
    pub old_text: Option<String>,
    /// The file content after the change.
    pub new_text: String,
}

impl ToolCallLocation {
    /// Create a location for `path` with no line number.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            line: None,
        }
    }

    /// Set the line number.
    #[must_use]
    pub fn line(mut self, line: u32) -> Self {
        self.line = Some(line);
        self
    }
}

/// An item emitted by the [`crate::TurnEngine`] for the ACP layer to forward.
///
/// This is kept abstract so tests can use an in-memory sink instead of a real
/// ACP client.
#[derive(Debug, Clone)]
pub enum EngineOutput {
    /// A chunk of assistant-visible message text.
    MessageChunk(String),
    /// A chunk of assistant reasoning ("thinking").
    ThinkingChunk(String),
    /// A plan update.
    Plan(PlanUpdate),
    /// A new tool call was created (status pending).
    ToolCall {
        /// Correlation id.
        id: ToolCallId,
        /// Tool name.
        name: String,
        /// Human-readable title describing what the tool is doing.
        title: String,
        /// The category of tool being invoked.
        kind: ToolKind,
        /// Initial status.
        status: ToolCallStatus,
        /// File locations affected by this tool call (for "follow-along").
        locations: Vec<ToolCallLocation>,
        /// Raw input arguments sent to the tool.
        raw_input: Option<serde_json::Value>,
    },
    /// An update to an existing tool call.
    ToolCallUpdate {
        /// Correlation id.
        id: ToolCallId,
        /// New status.
        status: ToolCallStatus,
        /// Optional human-readable output appended.
        output: Option<String>,
        /// Optional raw output returned by the tool (set on completion).
        raw_output: Option<serde_json::Value>,
        /// File changes produced by this tool update.
        file_diffs: Vec<FileDiff>,
    },
}

/// Receives [`EngineOutput`] items and forwards them to the client.
#[async_trait]
pub trait UpdateSink: Send {
    /// Forward a single engine output item.
    async fn send(&mut self, output: EngineOutput) -> Result<()>;

    /// Request permission for a tool call, returning whether it was granted.
    ///
    /// The default implementation grants permission, which is convenient for
    /// tests and non-interactive sinks.
    async fn request_permission(&mut self, _id: &ToolCallId, _tool_name: &str) -> Result<bool> {
        Ok(true)
    }
}
