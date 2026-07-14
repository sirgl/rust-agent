//! [`LocalClientAccess`]: a [`ClientAccess`] implementation backed by the local
//! machine.
//!
//! The CLI front-end has no editor behind it to serve ACP `fs/*` and
//! `terminal/*` requests, so the built-in filesystem/terminal tools are backed
//! directly by `std::fs` and `std::process::Command`. This is the CLI dialect
//! of the client seam, mirroring `acp-agent`'s `AcpClientAccess` but targeting
//! the real machine. The built-in tools themselves are unchanged: they depend
//! only on the [`ClientAccess`] trait.

use std::process::{Command, Stdio};

use agent_core::{AgentError, ClientAccess, Result, TerminalChunk, TerminalOutcome};
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::stream::BoxStream;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::Command as TokioCommand;
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
        tokio::task::spawn_blocking(move || {
            if let Some(parent) = std::path::Path::new(&path).parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, content)
        })
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

    async fn run_terminal_streaming(
        &self,
        command: &str,
        args: &[String],
    ) -> Result<BoxStream<'static, TerminalChunk>> {
        debug!(command, "local terminal run (streaming)");
        spawn_streaming_terminal(command.to_string(), args.to_vec())
    }
}

/// Spawn `command` with piped stdout/stderr and return a stream of output
/// chunks followed by a single [`TerminalChunk::Finished`].
///
/// Output is streamed line-by-line (stdout first, then stderr) as it is
/// produced, giving the terminal UI a "live log". The combined output is capped
/// at [`MAX_OUTPUT_BYTES`]; once exceeded, further output is dropped and the
/// final outcome is marked truncated.
fn spawn_streaming_terminal(
    command: String,
    args: Vec<String>,
) -> Result<BoxStream<'static, TerminalChunk>> {
    let mut child = TokioCommand::new(&command)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| AgentError::Other(format!("run_terminal failed: {err}")))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let (tx, rx) = mpsc::unbounded::<TerminalChunk>();

    tokio::spawn(async move {
        let mut combined = String::new();
        let mut truncated = false;

        if let Some(stdout) = stdout {
            stream_lines(stdout, &tx, &mut combined, &mut truncated).await;
        }
        if let Some(stderr) = stderr {
            stream_lines(stderr, &tx, &mut combined, &mut truncated).await;
        }

        let (exit_code, signal) = match child.wait().await {
            Ok(status) => (status.code().map(|c| c as u32), signal_of(&status)),
            Err(_) => (None, None),
        };

        let _ = tx.unbounded_send(TerminalChunk::Finished(TerminalOutcome {
            output: combined,
            truncated,
            exit_code,
            signal,
        }));
    });

    Ok(Box::pin(rx))
}

/// Read `reader` line-by-line, forwarding each line (with a trailing newline)
/// as a [`TerminalChunk::Output`] while accumulating into `combined` under the
/// [`MAX_OUTPUT_BYTES`] cap.
async fn stream_lines<R: AsyncRead + Unpin>(
    reader: R,
    tx: &mpsc::UnboundedSender<TerminalChunk>,
    combined: &mut String,
    truncated: &mut bool,
) {
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        if combined.len() >= MAX_OUTPUT_BYTES {
            *truncated = true;
            continue;
        }
        let mut chunk = line;
        chunk.push('\n');
        let remaining = MAX_OUTPUT_BYTES - combined.len();
        if chunk.len() > remaining {
            let mut end = remaining;
            while end > 0 && !chunk.is_char_boundary(end) {
                end -= 1;
            }
            chunk.truncate(end);
            *truncated = true;
        }
        combined.push_str(&chunk);
        let _ = tx.unbounded_send(TerminalChunk::Output(chunk));
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
