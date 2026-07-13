//! Shared error types for the agent core.

use thiserror::Error;

/// Result alias used throughout the core crate.
pub type Result<T> = std::result::Result<T, AgentError>;

/// Top-level error type for agent operations.
#[derive(Debug, Error)]
pub enum AgentError {
    /// The requested tool was not found in the registry.
    #[error("unknown tool: {0}")]
    UnknownTool(String),

    /// A tool received arguments that could not be parsed or validated.
    #[error("invalid tool arguments: {0}")]
    InvalidArguments(String),

    /// Permission for a tool call was denied by the client.
    #[error("permission denied for tool: {0}")]
    PermissionDenied(String),

    /// The operation was cancelled.
    #[error("operation cancelled")]
    Cancelled,

    /// An error originating from the decision (next-turn) layer.
    #[error("turn error: {0}")]
    Turn(#[from] TurnError),

    /// Serialization / deserialization failure.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// A catch-all error with a descriptive message.
    #[error("{0}")]
    Other(String),
}

/// Errors surfaced by a [`crate::NextTurnService`] while producing turn events.
#[derive(Debug, Error, Clone)]
#[error("{message}")]
pub struct TurnError {
    /// Human-readable description of the failure.
    pub message: String,
}

impl TurnError {
    /// Create a new turn error with the given message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
