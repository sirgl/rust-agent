//! [`LocalClientAccess`]: a [`ClientAccess`] implementation backed by the local
//! machine.
//!
//! By default the ACP agent performs its filesystem and terminal work directly
//! against the local machine (`std::fs` and `std::process::Command`) instead of
//! delegating to the editor over ACP `fs/*` and `terminal/*` JSON-RPC requests.
//! This keeps file/command behavior identical regardless of which client is
//! connected. The built-in tools are unchanged: they depend only on the
//! [`ClientAccess`] trait.
//!
//! Elicitation (structured interactive user input) has no local equivalent — it
//! is inherently a client capability — so it is still routed to the connected
//! ACP client via `elicitation/create` when a connection is available.

use std::process::{Command, Stdio};

use agent_client_protocol::schema::v1::{
    CreateElicitationRequest, ElicitationAction, ElicitationFormMode, ElicitationSchema,
    ElicitationSessionScope, SessionId,
};
use agent_client_protocol::{Client, ConnectionTo};
use agent_core::{
    AgentError, ClientAccess, ElicitationOutcome, Result, TerminalChunk, TerminalOutcome,
};
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::stream::BoxStream;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::Command as TokioCommand;
use tracing::debug;

/// Byte limit applied to captured terminal output before truncation.
const MAX_OUTPUT_BYTES: usize = 64 * 1024;

/// [`ClientAccess`] implementation that executes against the local filesystem
/// and shell.
///
/// Filesystem and terminal operations run locally. Elicitation, which requires
/// interactive user input, is delegated to the connected ACP client when an
/// `elicitation` connection handle is present.
#[derive(Clone)]
pub struct LocalClientAccess {
    /// Optional ACP connection used only to serve elicitation requests.
    elicitation: Option<(ConnectionTo<Client>, SessionId)>,
}

impl Default for LocalClientAccess {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalClientAccess {
    /// Create a local client access handle with no elicitation support.
    #[must_use]
    pub fn new() -> Self {
        Self { elicitation: None }
    }

    /// Create a local client access handle that routes elicitation requests to
    /// the given ACP connection/session.
    #[must_use]
    pub fn with_elicitation(cx: ConnectionTo<Client>, session_id: SessionId) -> Self {
        Self {
            elicitation: Some((cx, session_id)),
        }
    }
}

/// Truncate `output` to at most [`MAX_OUTPUT_BYTES`], returning the (possibly
/// shortened) string and whether truncation occurred.
fn truncate_output(mut output: String) -> (String, bool) {
    if output.len() <= MAX_OUTPUT_BYTES {
        return (output, false);
    }
    let mut end = MAX_OUTPUT_BYTES;
    while end > 0 && !output.is_char_boundary(end) {
        end -= 1;
    }
    output.truncate(end);
    (output, true)
}

/// Spawn `command` with piped stdout/stderr and return a stream of output
/// chunks followed by a single [`TerminalChunk::Finished`].
///
/// Output is streamed line-by-line (stdout first, then stderr) as it is
/// produced, giving the client a "live log". The combined output is capped at
/// [`MAX_OUTPUT_BYTES`]; once exceeded, further output is dropped and the final
/// outcome is marked truncated.
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

    async fn request_elicitation(
        &self,
        message: &str,
        requested_schema: serde_json::Value,
    ) -> Result<ElicitationOutcome> {
        let Some((cx, session_id)) = &self.elicitation else {
            return Err(AgentError::Other(
                "elicitation is not supported without a client connection".to_string(),
            ));
        };
        debug!(message, "elicitation/create");
        let schema: ElicitationSchema = serde_json::from_value(requested_schema)
            .map_err(|err| AgentError::Other(format!("invalid elicitation schema: {err}")))?;
        let scope = ElicitationSessionScope::new(session_id.clone());
        let mode = ElicitationFormMode::new(scope, schema);
        let request = CreateElicitationRequest::new(mode, message);

        let response = cx
            .send_request(request)
            .block_task()
            .await
            .map_err(|err| AgentError::Other(format!("elicitation/create failed: {err}")))?;

        match response.action {
            ElicitationAction::Accept(accept) => {
                let content = accept.content.unwrap_or_default();
                let value = serde_json::to_value(content).map_err(|err| {
                    AgentError::Other(format!("invalid elicitation content: {err}"))
                })?;
                Ok(ElicitationOutcome::Accepted(value))
            }
            ElicitationAction::Decline => Ok(ElicitationOutcome::Declined),
            ElicitationAction::Cancel => Ok(ElicitationOutcome::Cancelled),
            other => Err(AgentError::Other(format!(
                "unsupported elicitation action: {other:?}"
            ))),
        }
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
        let path = dir.join(format!("acp-agent-test-{}.txt", std::process::id()));
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
            .run_terminal("echo", &["hi from acp".to_string()])
            .await
            .expect("command should run");
        assert!(outcome.is_success());
        assert!(outcome.output.contains("hi from acp"));
        assert!(!outcome.truncated);
    }

    #[tokio::test]
    async fn streaming_emits_output_then_finished() {
        use futures::StreamExt;

        let client = LocalClientAccess::new();
        let stream = client
            .run_terminal_streaming(
                "printf",
                &["line1\\nline2\\nline3\\n".to_string()],
            )
            .await
            .expect("command should start");
        let chunks: Vec<TerminalChunk> = stream.collect().await;

        // At least one output chunk plus a final Finished.
        assert!(chunks.len() >= 2, "expected output + finished, got {chunks:?}");
        let outputs: Vec<&String> = chunks
            .iter()
            .filter_map(|c| match c {
                TerminalChunk::Output(text) => Some(text),
                TerminalChunk::Finished(_) => None,
            })
            .collect();
        assert!(!outputs.is_empty(), "expected streamed output chunks");
        let combined: String = outputs.iter().map(|s| s.as_str()).collect();
        assert!(combined.contains("line1"));
        assert!(combined.contains("line3"));

        match chunks.last() {
            Some(TerminalChunk::Finished(outcome)) => {
                assert!(outcome.is_success());
                assert!(outcome.output.contains("line2"));
            }
            other => panic!("expected Finished last, got {other:?}"),
        }
    }
}
