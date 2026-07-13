//! The [`ClientAccess`] seam: a provider-independent abstraction over the ACP
//! client capabilities that built-in tools rely on (filesystem and terminal).
//!
//! Services (tools) are written **once** against this trait and are shared by
//! every decision-layer provider; they never depend on a specific provider or
//! on the concrete `agent-client-protocol` client type. The ACP binary supplies
//! a concrete implementation backed by the live connection, while tests supply
//! an in-memory fake. This keeps the services layer testable without a network
//! or a real editor, and keeps the provider-specific concerns confined to the
//! decision layer / dialect path.

use async_trait::async_trait;

use crate::error::Result;

/// The outcome of running a terminal command via the client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalOutcome {
    /// The captured (possibly truncated) command output.
    pub output: String,
    /// Whether the output was truncated due to a byte limit.
    pub truncated: bool,
    /// The process exit code, if it exited normally.
    pub exit_code: Option<u32>,
    /// The signal that terminated the process, if any.
    pub signal: Option<String>,
}

impl TerminalOutcome {
    /// Whether the command is considered successful (exit code `0`, no signal).
    pub fn is_success(&self) -> bool {
        self.signal.is_none() && matches!(self.exit_code, Some(0))
    }
}

/// Provider-independent access to the ACP client's filesystem and terminal
/// capabilities.
///
/// Implemented once by the ACP binary (backed by the live connection) and by
/// test fakes. Built-in tools depend on this trait only, never on a concrete
/// client.
#[async_trait]
pub trait ClientAccess: Send + Sync {
    /// Read the full text content of a file at `path` (absolute).
    async fn read_text_file(&self, path: &str) -> Result<String>;

    /// Write `content` to a file at `path` (absolute), creating or truncating it.
    async fn write_text_file(&self, path: &str, content: &str) -> Result<()>;

    /// Run a terminal command, waiting for it to exit, and return its outcome.
    async fn run_terminal(&self, command: &str, args: &[String]) -> Result<TerminalOutcome>;
}
