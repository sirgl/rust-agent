//! [`TerminalUpdateSink`]: renders [`EngineOutput`] to a terminal writer.
//!
//! This is the CLI presentation seam. It streams assistant message and thinking
//! chunks to the writer as they arrive (flushing so the user sees output
//! incrementally) and renders tool-call lifecycle updates as concise status
//! lines. It also implements permission prompting: when interactive, it asks the
//! user `[y/N]` on stdin before a gated tool runs; when not interactive (tests),
//! permission is auto-granted.
//!
//! The sink is generic over any [`std::io::Write`] so tests can assert against an
//! in-memory buffer while the binary uses stdout.

use std::io::{BufRead, Write};

use agent_core::{AgentError, EngineOutput, Result, ToolCallId, ToolCallStatus, UpdateSink};
use async_trait::async_trait;

/// Renders engine output to a [`Write`] target (stdout for the binary, an
/// in-memory buffer for tests).
pub struct TerminalUpdateSink<W: Write + Send> {
    writer: W,
    /// Whether we are mid-line on a streamed message (so status lines can insert
    /// a newline before printing).
    mid_line: bool,
    /// When `true`, [`UpdateSink::request_permission`] prompts the user on stdin
    /// with `[y/N]`; when `false` (the default, and what tests use) permission is
    /// auto-granted.
    interactive: bool,
}

impl<W: Write + Send> TerminalUpdateSink<W> {
    /// Create a sink writing to `writer`.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            mid_line: false,
            interactive: false,
        }
    }

    /// Enable interactive `[y/N]` permission prompts read from stdin.
    #[must_use]
    pub fn interactive(mut self, interactive: bool) -> Self {
        self.interactive = interactive;
        self
    }

    /// Consume the sink and return the underlying writer (useful in tests).
    pub fn into_inner(self) -> W {
        self.writer
    }

    /// Write raw text and flush, mapping IO errors into the core error type.
    fn write_flush(&mut self, text: &str) -> Result<()> {
        self.writer
            .write_all(text.as_bytes())
            .and_then(|()| self.writer.flush())
            .map_err(|err| AgentError::Other(format!("terminal write failed: {err}")))
    }

    /// Ensure any streamed, unterminated line is closed before a status line.
    fn ensure_newline(&mut self) -> Result<()> {
        if self.mid_line {
            self.mid_line = false;
            self.write_flush("\n")?;
        }
        Ok(())
    }
}

/// Human-readable label for a tool-call status.
fn status_label(status: ToolCallStatus) -> &'static str {
    match status {
        ToolCallStatus::Pending => "pending",
        ToolCallStatus::InProgress => "in_progress",
        ToolCallStatus::Completed => "completed",
        ToolCallStatus::Failed => "failed",
    }
}

#[async_trait]
impl<W: Write + Send> UpdateSink for TerminalUpdateSink<W> {
    async fn send(&mut self, output: EngineOutput) -> Result<()> {
        match output {
            EngineOutput::MessageChunk(text) => {
                self.mid_line = !text.ends_with('\n');
                self.write_flush(&text)
            }
            EngineOutput::ThinkingChunk(text) => {
                self.ensure_newline()?;
                self.write_flush(&format!("\u{1b}[2m[thinking] {text}\u{1b}[0m\n"))
            }
            EngineOutput::Plan(plan) => {
                self.ensure_newline()?;
                let mut rendered = String::from("[plan]\n");
                for step in &plan.steps {
                    rendered.push_str(&format!("  - {}\n", step.content));
                }
                self.write_flush(&rendered)
            }
            EngineOutput::ToolCall { id, name, status } => {
                self.ensure_newline()?;
                self.write_flush(&format!(
                    "[tool {}] {} ({})\n",
                    id.0,
                    name,
                    status_label(status)
                ))
            }
            EngineOutput::ToolCallUpdate { id, status, output } => {
                self.ensure_newline()?;
                let mut line = format!("[tool {}] -> {}", id.0, status_label(status));
                if let Some(out) = output {
                    line.push_str(&format!(": {out}"));
                }
                line.push('\n');
                self.write_flush(&line)
            }
        }
    }

    async fn request_permission(&mut self, id: &ToolCallId, tool_name: &str) -> Result<bool> {
        if !self.interactive {
            return Ok(true);
        }
        self.ensure_newline()?;
        self.write_flush(&format!(
            "[permission] allow tool '{}' (call {})? [y/N] ",
            tool_name, id.0
        ))?;
        // Read one line from stdin on a blocking thread so the async executor is
        // not stalled while waiting for the user.
        let line = tokio::task::spawn_blocking(|| {
            let mut buf = String::new();
            let stdin = std::io::stdin();
            let mut lock = stdin.lock();
            match lock.read_line(&mut buf) {
                Ok(0) => None,      // EOF: treat as deny
                Ok(_) => Some(buf),
                Err(_) => None,
            }
        })
        .await
        .map_err(|err| AgentError::Other(format!("permission read failed: {err}")))?;
        let granted = line.map(|l| parse_yes_no(&l)).unwrap_or(false);
        self.write_flush(if granted { "granted\n" } else { "denied\n" })?;
        Ok(granted)
    }
}

/// Interpret a line of user input as an affirmative (`y`/`yes`) answer.
fn parse_yes_no(line: &str) -> bool {
    matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::{PlanStep, PlanStepStatus, PlanUpdate, ToolCallId};

    fn rendered(outputs: Vec<EngineOutput>) -> String {
        futures::executor::block_on(async {
            let mut sink = TerminalUpdateSink::new(Vec::<u8>::new());
            for output in outputs {
                sink.send(output).await.expect("send should succeed");
            }
            String::from_utf8(sink.into_inner()).expect("utf8")
        })
    }

    #[test]
    fn streams_message_chunks_verbatim() {
        let out = rendered(vec![
            EngineOutput::MessageChunk("Hello, ".to_string()),
            EngineOutput::MessageChunk("world".to_string()),
        ]);
        assert_eq!(out, "Hello, world");
    }

    #[test]
    fn thinking_is_annotated() {
        let out = rendered(vec![EngineOutput::ThinkingChunk("pondering".to_string())]);
        assert!(out.contains("[thinking] pondering"));
    }

    #[test]
    fn tool_lifecycle_renders_status_lines() {
        let id = ToolCallId("call-1".to_string());
        let out = rendered(vec![
            EngineOutput::MessageChunk("mid".to_string()),
            EngineOutput::ToolCall {
                id: id.clone(),
                name: "fs_read".to_string(),
                status: ToolCallStatus::Pending,
            },
            EngineOutput::ToolCallUpdate {
                id: id.clone(),
                status: ToolCallStatus::Completed,
                output: Some("ok".to_string()),
            },
        ]);
        // The mid-line message must be closed before the status line.
        assert!(out.contains("mid\n[tool call-1] fs_read (pending)"));
        assert!(out.contains("[tool call-1] -> completed: ok"));
    }

    #[test]
    fn parse_yes_no_accepts_affirmatives() {
        assert!(parse_yes_no("y\n"));
        assert!(parse_yes_no("  YES \n"));
        assert!(!parse_yes_no("n\n"));
        assert!(!parse_yes_no("\n"));
        assert!(!parse_yes_no("nope"));
    }

    #[test]
    fn plan_lists_entries() {
        let plan = PlanUpdate {
            steps: vec![
                PlanStep {
                    content: "step one".to_string(),
                    status: PlanStepStatus::Pending,
                },
                PlanStep {
                    content: "step two".to_string(),
                    status: PlanStepStatus::Completed,
                },
            ],
        };
        let out = rendered(vec![EngineOutput::Plan(plan)]);
        assert!(out.contains("[plan]"));
        assert!(out.contains("- step one"));
        assert!(out.contains("- step two"));
    }
}
