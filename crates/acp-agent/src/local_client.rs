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

use std::path::{Path, PathBuf};
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
    /// Immutable working directory captured for this ACP session.
    cwd: PathBuf,
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
        let cwd = std::env::current_dir().expect("process working directory must be available");
        Self {
            cwd,
            elicitation: None,
        }
    }

    /// Create a local client bound to an explicit, validated working directory.
    pub fn with_cwd(cwd: impl Into<PathBuf>) -> Result<Self> {
        Ok(Self {
            cwd: validate_cwd(cwd.into())?,
            elicitation: None,
        })
    }

    /// Create a local client access handle that routes elicitation requests to
    /// the given ACP connection/session.
    #[must_use]
    pub fn with_elicitation(cx: ConnectionTo<Client>, session_id: SessionId) -> Self {
        let cwd = std::env::current_dir().expect("process working directory must be available");
        Self {
            cwd,
            elicitation: Some((cx, session_id)),
        }
    }

    /// Create an elicitation-capable client bound to a validated session cwd.
    pub fn with_elicitation_and_cwd(
        cx: ConnectionTo<Client>,
        session_id: SessionId,
        cwd: impl Into<PathBuf>,
    ) -> Result<Self> {
        Ok(Self {
            cwd: validate_cwd(cwd.into())?,
            elicitation: Some((cx, session_id)),
        })
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        }
    }
}

fn validate_cwd(cwd: PathBuf) -> Result<PathBuf> {
    let requested = cwd.display().to_string();
    let metadata = std::fs::metadata(&cwd).map_err(|err| {
        AgentError::Other(format!("invalid working directory `{requested}`: {err}"))
    })?;
    if !metadata.is_dir() {
        return Err(AgentError::Other(format!(
            "invalid working directory `{requested}`: path is not a directory"
        )));
    }
    std::fs::canonicalize(&cwd)
        .map_err(|err| AgentError::Other(format!("invalid working directory `{requested}`: {err}")))
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
    cwd: PathBuf,
) -> Result<BoxStream<'static, TerminalChunk>> {
    let mut child = TokioCommand::new(&command)
        .args(&args)
        .current_dir(&cwd)
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
        let path = self.resolve_path(path);
        debug!(path = %path.display(), cwd = %self.cwd.display(), "local fs read");
        tokio::task::spawn_blocking(move || std::fs::read_to_string(&path))
            .await
            .map_err(|err| AgentError::Other(format!("read task join error: {err}")))?
            .map_err(|err| AgentError::Other(format!("read_text_file failed: {err}")))
    }

    async fn write_text_file(&self, path: &str, content: &str) -> Result<()> {
        let path = self.resolve_path(path);
        debug!(path = %path.display(), cwd = %self.cwd.display(), "local fs write");
        let content = content.to_string();
        tokio::task::spawn_blocking(move || {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, content)
        })
        .await
        .map_err(|err| AgentError::Other(format!("write task join error: {err}")))?
        .map_err(|err| AgentError::Other(format!("write_text_file failed: {err}")))
    }

    async fn run_terminal(&self, command: &str, args: &[String]) -> Result<TerminalOutcome> {
        debug!(command, cwd = %self.cwd.display(), "local terminal run");
        let command = command.to_string();
        let args = args.to_vec();
        let cwd = self.cwd.clone();
        let output = tokio::task::spawn_blocking(move || {
            Command::new(&command).args(&args).current_dir(cwd).output()
        })
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
        debug!(command, cwd = %self.cwd.display(), "local terminal run (streaming)");
        spawn_streaming_terminal(command.to_string(), args.to_vec(), self.cwd.clone())
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

    fn test_dir(label: &str) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("rust-acp-agent-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[tokio::test]
    async fn relative_files_and_shell_commands_use_captured_cwd() {
        let root = test_dir("captured-cwd");
        let process_cwd = std::env::current_dir().unwrap();
        let client = LocalClientAccess::with_cwd(&root).unwrap();

        client
            .write_text_file("nested/value.txt", "from workspace")
            .await
            .unwrap();
        assert_eq!(
            client.read_text_file("nested/value.txt").await.unwrap(),
            "from workspace"
        );

        let outcome = client
            .run_shell_command("printf 'alpha\\nbeta\\n' | tail -n 1 > shell.txt; pwd")
            .await
            .unwrap();
        assert!(outcome.is_success());
        assert_eq!(
            outcome.output.trim(),
            std::fs::canonicalize(&root).unwrap().to_string_lossy()
        );
        assert_eq!(client.read_text_file("shell.txt").await.unwrap(), "beta\n");
        assert_eq!(std::env::current_dir().unwrap(), process_cwd);
    }

    #[tokio::test]
    async fn absolute_paths_are_not_rebased() {
        let root = test_dir("absolute-root");
        let outside = test_dir("absolute-outside").join("value.txt");
        std::fs::write(&outside, "outside").unwrap();
        let client = LocalClientAccess::with_cwd(root).unwrap();

        assert_eq!(
            client
                .read_text_file(outside.to_str().unwrap())
                .await
                .unwrap(),
            "outside"
        );
    }

    #[test]
    fn invalid_cwd_error_names_requested_path() {
        let missing =
            std::env::temp_dir().join(format!("rust-acp-agent-missing-cwd-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&missing);

        let error = LocalClientAccess::with_cwd(&missing)
            .err()
            .expect("missing cwd must fail validation");
        assert!(error
            .to_string()
            .contains(&missing.to_string_lossy().to_string()));
    }

    #[tokio::test]
    async fn concurrent_clients_keep_workspaces_isolated() {
        let first = test_dir("isolation-first");
        let second = test_dir("isolation-second");
        std::fs::write(first.join("same.txt"), "first").unwrap();
        std::fs::write(second.join("same.txt"), "second").unwrap();
        let first_client = LocalClientAccess::with_cwd(&first).unwrap();
        let second_client = LocalClientAccess::with_cwd(&second).unwrap();

        let (first_read, second_read, first_shell, second_shell) = tokio::join!(
            first_client.read_text_file("same.txt"),
            second_client.read_text_file("same.txt"),
            first_client.run_shell_command("cat same.txt"),
            second_client.run_shell_command("cat same.txt")
        );

        assert_eq!(first_read.unwrap(), "first");
        assert_eq!(second_read.unwrap(), "second");
        assert_eq!(first_shell.unwrap().output, "first");
        assert_eq!(second_shell.unwrap().output, "second");
    }

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
            .run_terminal_streaming("printf", &["line1\\nline2\\nline3\\n".to_string()])
            .await
            .expect("command should start");
        let chunks: Vec<TerminalChunk> = stream.collect().await;

        // At least one output chunk plus a final Finished.
        assert!(
            chunks.len() >= 2,
            "expected output + finished, got {chunks:?}"
        );
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
