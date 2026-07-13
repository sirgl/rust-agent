//! [`LocalClientAccess`]: a [`ClientAccess`] implementation backed by the local
//! machine.
//!
//! The CLI front-end has no editor behind it to serve ACP `fs/*` and
//! `terminal/*` requests, so the built-in filesystem/terminal tools are backed
//! directly by `std::fs` and `std::process::Command`. This is the CLI dialect
//! of the client seam, mirroring `acp-agent`'s `AcpClientAccess` but targeting
//! the real machine. The built-in tools themselves are unchanged: they depend
//! only on the [`ClientAccess`] trait.

use std::process::Command;

use agent_core::{AgentError, ClientAccess, Result, TerminalOutcome};
use async_trait::async_trait;
use tracing::debug;

/// Byte limit applied to captured terminal output before truncation, matching
/// the spirit of the ACP client's bounded output capture.
const MAX_OUTPUT_BYTES: usize = 64 * 1024;

/// [`ClientAccess`] implementation that executes against the local filesystem
/// and shell.
#[derive(Debug, Default, Clone)]
pub struct LocalClientAccess;

impl LocalClientAccess {
    /// Create a new local client access handle.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Truncate `output` to at most [`MAX_OUTPUT_BYTES`], returning the (possibly
/// shortened) string and whether truncation occurred.
fn truncate_output(mut output: String) -> (String, bool) {
    if output.len() <= MAX_OUTPUT_BYTES {
        return (output, false);
    }
    // Find a valid char boundary at or below the limit.
    let mut end = MAX_OUTPUT_BYTES;
    while end > 0 && !output.is_char_boundary(end) {
        end -= 1;
    }
    output.truncate(end);
    (output, true)
}

#[async_trait]
impl ClientAccess for LocalClientAccess {
    async fn read_text_file(&self, path: &str) -> Result<String> {
        debug!(path, "local fs read");
        let path = path.to_string();
        tokio::task::spawn_blocking(move || std::fs::read_to_string(&path))
            .await
            .map_err(|err| AgentError::Other(format!("read task join error: {err}")))?
            .map_err(|err| AgentError::Other(format!("read_text_file failed: {err}")))
    }

    async fn write_text_file(&self, path: &str, content: &str) -> Result<()> {
        debug!(path, "local fs write");
        let path = path.to_string();
        let content = content.to_string();
        tokio::task::spawn_blocking(move || std::fs::write(&path, content))
            .await
            .map_err(|err| AgentError::Other(format!("write task join error: {err}")))?
            .map_err(|err| AgentError::Other(format!("write_text_file failed: {err}")))
    }

    async fn run_terminal(&self, command: &str, args: &[String]) -> Result<TerminalOutcome> {
        debug!(command, "local terminal run");
        let command = command.to_string();
        let args = args.to_vec();
        let output = tokio::task::spawn_blocking(move || Command::new(&command).args(&args).output())
            .await
            .map_err(|err| AgentError::Other(format!("terminal task join error: {err}")))?
            .map_err(|err| AgentError::Other(format!("run_terminal failed: {err}")))?;

        let mut combined = String::new();
        combined.push_str(&String::from_utf8_lossy(&output.stdout));
        combined.push_str(&String::from_utf8_lossy(&output.stderr));
        let (combined, truncated) = truncate_output(combined);

        // `code()` returns None when the process was terminated by a signal.
        let exit_code = output.status.code().map(|c| c as u32);
        let signal = signal_of(&output.status);

        Ok(TerminalOutcome {
            output: combined,
            truncated,
            exit_code,
            signal,
        })
    }
}

/// Extract the terminating signal name (Unix only) from an exit status.
#[cfg(unix)]
fn signal_of(status: &std::process::ExitStatus) -> Option<String> {
    use std::os::unix::process::ExitStatusExt;
    status.signal().map(|s| s.to_string())
}

/// Non-Unix platforms do not expose a terminating signal.
#[cfg(not(unix))]
fn signal_of(_status: &std::process::ExitStatus) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn file_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("cli-agent-test-{}.txt", std::process::id()));
        let path_str = path.to_string_lossy().into_owned();

        let client = LocalClientAccess::new();
        client
            .write_text_file(&path_str, "hello local")
            .await
            .expect("write should succeed");
        let content = client
            .read_text_file(&path_str)
            .await
            .expect("read should succeed");
        assert_eq!(content, "hello local");

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn runs_trivial_command() {
        let client = LocalClientAccess::new();
        let outcome = client
            .run_terminal("echo", &["hi from cli".to_string()])
            .await
            .expect("command should run");
        assert!(outcome.is_success());
        assert!(outcome.output.contains("hi from cli"));
        assert!(!outcome.truncated);
    }

    #[tokio::test]
    async fn reports_nonzero_exit() {
        let client = LocalClientAccess::new();
        let outcome = client
            .run_terminal("false", &[])
            .await
            .expect("command should run");
        assert!(!outcome.is_success());
    }
}
