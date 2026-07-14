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
use futures::stream::{self, BoxStream};

use crate::error::{AgentError, Result};

/// The outcome of an elicitation request (structured user input) sent to the
/// client.
///
/// Provider-independent mirror of the ACP `elicitation/create` response: the
/// user either accepts and provides content matching the requested schema,
/// declines, or the request is cancelled.
#[derive(Debug, Clone, PartialEq)]
pub enum ElicitationOutcome {
    /// The user accepted and provided content, as a JSON object matching the
    /// requested schema (may be empty).
    Accepted(serde_json::Value),
    /// The user declined to provide the requested input.
    Declined,
    /// The elicitation was cancelled (e.g. the turn was interrupted).
    Cancelled,
}

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

/// A single item in a streaming terminal run.
///
/// A terminal command is surfaced as a stream of zero or more
/// [`TerminalChunk::Output`] deltas (incremental stdout/stderr as it is
/// produced), followed by exactly one terminal [`TerminalChunk::Finished`]
/// carrying the final [`TerminalOutcome`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminalChunk {
    /// An incremental chunk of combined stdout/stderr output.
    Output(String),
    /// The command finished; carries the final outcome.
    Finished(TerminalOutcome),
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

    /// Run a terminal command, streaming its output as it is produced.
    ///
    /// Returns a stream of [`TerminalChunk::Output`] deltas followed by a single
    /// [`TerminalChunk::Finished`] with the final outcome. The default
    /// implementation is a non-streaming fallback built on top of
    /// [`run_terminal`](Self::run_terminal): it runs the command to completion
    /// and yields the whole output as one chunk. Implementations backed by a
    /// live process (e.g. the local machine) should override this to emit
    /// incremental output for a "live log" experience.
    async fn run_terminal_streaming(
        &self,
        command: &str,
        args: &[String],
    ) -> Result<BoxStream<'static, TerminalChunk>> {
        let outcome = self.run_terminal(command, args).await?;
        let mut chunks = Vec::new();
        if !outcome.output.is_empty() {
            chunks.push(TerminalChunk::Output(outcome.output.clone()));
        }
        chunks.push(TerminalChunk::Finished(outcome));
        Ok(Box::pin(stream::iter(chunks)))
    }

    /// Request structured input from the user via a form.
    ///
    /// `message` is a human-readable description of what input is needed, and
    /// `requested_schema` is a JSON Schema (an `"object"` schema) describing the
    /// form fields. Blocks until the user responds, returning an
    /// [`ElicitationOutcome`].
    ///
    /// The default implementation reports the capability as unsupported; the
    /// ACP binary overrides it with a live `elicitation/create` request.
    async fn request_elicitation(
        &self,
        message: &str,
        requested_schema: serde_json::Value,
    ) -> Result<ElicitationOutcome> {
        let _ = (message, requested_schema);
        Err(AgentError::Other(
            "elicitation is not supported by this client".to_string(),
        ))
    }
}
