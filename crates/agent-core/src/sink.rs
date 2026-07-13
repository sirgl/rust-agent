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
        /// Initial status.
        status: ToolCallStatus,
    },
    /// An update to an existing tool call.
    ToolCallUpdate {
        /// Correlation id.
        id: ToolCallId,
        /// New status.
        status: ToolCallStatus,
        /// Optional human-readable output appended.
        output: Option<String>,
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
