//! The [`TurnEngine`]: drives a full turn loop.
//!
//! The engine consumes a stream of [`TurnEvent`]s from a [`NextTurnService`],
//! forwards streaming output to an [`UpdateSink`], dispatches tool calls through
//! a [`ToolRegistry`], appends the typed results to the session
//! [`Conversation`](crate::history::Conversation), and repeats until a
//! [`StopReason`] is reached.
//!
//! Cancellation is cooperative: a shared [`CancellationToken`] is observed both
//! while consuming the decision stream and while dispatching tools (via
//! [`ToolContext`]). When cancellation is observed the loop stops promptly,
//! returns [`StopReason::Cancelled`], and emits no further updates.

use std::sync::{Arc, Mutex as StdMutex};

use futures::StreamExt;
use tracing::{debug, warn};

use crate::cancel::CancellationToken;
use crate::client::ClientAccess;
use crate::error::{AgentError, Result};
use crate::event::{StopReason, ToolCallId, TurnEvent};
use crate::history::KnownTool;
use crate::service::NextTurnService;
use crate::session::SessionState;
use crate::sink::{EngineOutput, ToolCallStatus, ToolKind, UpdateSink};
use crate::tool::{ToolContext, ToolEvent};
use crate::tool::ToolRegistry;

/// The message injected as a user turn when the backend would end the turn
/// without dispatching a terminal tool (see
/// [`TurnEngine::require_terminal_tool`]). It reminds the model that only a
/// terminal tool (`submit_result`) hands control back to the user.
pub const TERMINAL_TOOL_NUDGE: &str = "\
You stopped without ending the turn. The turn only ends — and control only \
returns to the user — when you call the `submit_result` tool. If the task is \
fully complete, call `submit_result` now with a concise summary. Otherwise, keep \
working and call the appropriate tools.";

/// The message injected periodically (see [`TurnEngine::with_thought_interval`])
/// to prompt the model to briefly share what it is planning. Kept intentionally
/// short so the streamed reply stays to one or two lines.
pub const THOUGHT_NUDGE: &str = "\
In one or two short lines, tell the user what you are doing or planning right \
now. Keep it super short, then continue working.";

/// A shared queue of user messages injected into a running turn ("steering").
///
/// Messages pushed here by the ACP layer (via `_session/inject`) are drained by
/// the [`TurnEngine`] between decision rounds and appended to the conversation
/// as user messages, so they take effect on the next decision call within the
/// same turn.
pub type TurnInbox = Arc<StdMutex<Vec<String>>>;

/// Drives the agent turn loop.
///
/// A `TurnEngine` couples a [`NextTurnService`] (the decision layer) with a
/// [`ToolRegistry`] (the services layer). It is deliberately provider-agnostic:
/// swapping the `NextTurnService` implementation (e.g. Anthropic vs. replay) is
/// all that is required to change behavior.
pub struct TurnEngine {
    next_turn: Arc<dyn NextTurnService>,
    tools: ToolRegistry,
    /// Optional upper bound on decision/tool iterations within a single prompt.
    /// `None` (the default) means the turn loop is unbounded and only ends on a
    /// real stop reason (terminal tool, backend end-of-turn, cancellation).
    max_iterations: Option<usize>,
    client: Option<Arc<dyn ClientAccess>>,
    depth: usize,
    inbox: Option<TurnInbox>,
    require_terminal_tool: bool,
    /// Optional cadence (in decision rounds) at which a [`THOUGHT_NUDGE`] is
    /// injected so the model briefly narrates its plan. `None` disables it.
    thought_interval: Option<usize>,
}

impl TurnEngine {
    /// Create a new engine from a decision service and a tool registry.
    pub fn new(next_turn: Arc<dyn NextTurnService>, tools: ToolRegistry) -> Self {
        Self {
            next_turn,
            tools,
            max_iterations: None,
            client: None,
            depth: 0,
            inbox: None,
            require_terminal_tool: false,
            thought_interval: None,
        }
    }

    /// Cap the maximum number of loop iterations.
    ///
    /// By default the turn loop is unbounded; call this to impose a hard cap
    /// (e.g. to bound a subagent's work budget).
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = Some(max_iterations);
        self
    }

    /// Attach a [`ClientAccess`] implementation so dispatched tools can reach
    /// the ACP client's filesystem and terminal capabilities.
    pub fn with_client(mut self, client: Arc<dyn ClientAccess>) -> Self {
        self.client = Some(client);
        self
    }

    /// Attach a steering [`TurnInbox`] whose queued user messages are drained
    /// between decision rounds and appended to the conversation.
    pub fn with_inbox(mut self, inbox: TurnInbox) -> Self {
        self.inbox = Some(inbox);
        self
    }

    /// Require an explicit *terminal tool* to end the turn.
    ///
    /// When enabled, a decision round that finishes without dispatching a tool
    /// whose [`Tool::ends_turn`] returns `true` does **not** end the turn:
    /// instead a [`TERMINAL_TOOL_NUDGE`] user message is appended and the loop
    /// continues, so the model is pushed to keep working and eventually call a
    /// terminal tool. The turn ends only when some terminal tool is dispatched
    /// (or on cancellation / the iteration guard). This makes hand-back to the
    /// user an explicit, deliberate action rather than an implicit consequence
    /// of the model emitting plain text.
    ///
    /// Which tools are terminal is decided by the tools themselves (via
    /// [`Tool::ends_turn`]), so any number of terminal tools may be registered
    /// without changing the engine.
    pub fn require_terminal_tool(mut self) -> Self {
        self.require_terminal_tool = true;
        self
    }

    /// Periodically prompt the model to briefly share what it is planning.
    ///
    /// When set to `Some(n)` (with `n > 0`), a [`THOUGHT_NUDGE`] user message is
    /// injected before the first decision round (right after the user's prompt)
    /// and then again every `n` rounds, so a long-running turn keeps the user
    /// informed. `None` (the default) disables the behavior.
    pub fn with_thought_interval(mut self, interval: usize) -> Self {
        self.thought_interval = (interval > 0).then_some(interval);
        self
    }

    /// Drain any queued steering messages into the session as user messages.
    /// Returns `true` if at least one message was drained.
    fn drain_inbox(&self, session: &mut SessionState) -> bool {
        let Some(inbox) = &self.inbox else {
            return false;
        };
        let drained: Vec<String> = {
            let mut queue = inbox.lock().expect("inbox mutex poisoned");
            std::mem::take(&mut *queue)
        };
        let had_any = !drained.is_empty();
        for text in drained {
            debug!("draining injected steering message into turn");
            session.push_user_text(text);
        }
        had_any
    }

    /// Whether the steering inbox currently has pending messages.
    fn has_pending_inbox(&self) -> bool {
        self.inbox
            .as_ref()
            .is_some_and(|inbox| !inbox.lock().expect("inbox mutex poisoned").is_empty())
    }

    /// Set the subagent nesting depth this engine runs at.
    ///
    /// A top-level turn runs at depth `0`. A subagent tool that spawns a nested
    /// engine sets the nested engine's depth to `ctx.depth + 1` so the value
    /// propagates through the [`ToolContext`] of every tool the nested engine
    /// dispatches, letting the subagent tool enforce a recursion depth limit.
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Run the turn loop until a [`StopReason`] is reached.
    ///
    /// Streaming output is forwarded to `sink` as it arrives. Tool calls are
    /// dispatched through the registry, their typed records appended to the
    /// session history, and their results fed back for the next decision call.
    /// Both the decision stream and tool dispatch observe `cancel`; if it fires,
    /// the loop returns [`StopReason::Cancelled`] and emits no further updates.
    pub async fn run_prompt(
        &self,
        session: &mut SessionState,
        sink: &mut dyn UpdateSink,
        cancel: &CancellationToken,
    ) -> Result<StopReason> {
        let mut iteration = 0usize;
        loop {
            if let Some(max) = self.max_iterations {
                if iteration >= max {
                    warn!("turn loop exceeded max iterations ({max})");
                    return Ok(StopReason::EndTurn);
                }
            }
            if cancel.is_cancelled() {
                debug!("cancellation observed before iteration {iteration}");
                return Ok(StopReason::Cancelled);
            }

            // Fold in any steering messages injected since the previous round so
            // they become part of the context for this decision call.
            self.drain_inbox(session);

            // Periodically ask the model to briefly narrate its plan: on the
            // first round (right after the user prompt) and every `interval`
            // rounds thereafter.
            if let Some(interval) = self.thought_interval {
                if iteration.is_multiple_of(interval) {
                    session.push_user_text(THOUGHT_NUDGE.to_string());
                }
            }

            let ctx = session.turn_context();
            let mut stream = self.next_turn.get_next_turn_streaming(&ctx).await?;

            // Accumulates assistant-visible text within this decision stream so we
            // can persist it as a single history entry when the stream yields a
            // tool call or finishes.
            let mut assistant_text = String::new();
            // Whether this stream produced any tool calls (so we know to loop).
            let mut had_tool_call = false;

            loop {
                // Observe cancellation while awaiting the next decision event.
                let event = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        debug!("cancellation observed while streaming decision events");
                        return Ok(StopReason::Cancelled);
                    }
                    next = stream.next() => next,
                };

                let Some(event) = event else {
                    // Stream ended without an explicit TurnFinished; treat the
                    // accumulated text as a completed turn.
                    warn!("decision stream ended without TurnFinished");
                    flush_assistant_text(session, &mut assistant_text);
                    return Ok(StopReason::EndTurn);
                };

                match event {
                    TurnEvent::TextDelta(delta) => {
                        assistant_text.push_str(&delta);
                        sink.send(EngineOutput::MessageChunk(delta)).await?;
                    }
                    TurnEvent::Thinking(delta) => {
                        sink.send(EngineOutput::ThinkingChunk(delta)).await?;
                    }
                    TurnEvent::ThinkingBlock(record) => {
                        // The live text was already streamed via `Thinking`;
                        // here we only persist the complete, signed block so it
                        // can be replayed back to the provider on the next turn
                        // (required when thinking is combined with tool use).
                        //
                        // It is stored before any assistant text/tool call of
                        // this turn (thinking blocks close first), so the
                        // Anthropic dialect can place it first in the merged
                        // assistant message, as the API requires.
                        session.push_thinking(record);
                    }
                    TurnEvent::Plan(plan) => {
                        sink.send(EngineOutput::Plan(plan)).await?;
                    }
                    TurnEvent::Usage(usage) => {
                        session.add_usage(usage);
                    }
                    TurnEvent::ToolCallRequested {
                        id,
                        name,
                        arguments,
                    } => {
                        had_tool_call = true;
                        // Persist any assistant text emitted before this tool call.
                        flush_assistant_text(session, &mut assistant_text);
                        // A tool that declares itself terminal ends the turn once
                        // it has been dispatched (only when enforcement is on).
                        let is_terminal = self.require_terminal_tool
                            && self.tools.get(&name).is_some_and(|t| t.ends_turn());
                        self.dispatch_tool(session, sink, cancel, id, name, arguments)
                            .await?;
                        if cancel.is_cancelled() {
                            return Ok(StopReason::Cancelled);
                        }
                        if is_terminal {
                            debug!("terminal tool dispatched; ending turn");
                            return Ok(StopReason::EndTurn);
                        }
                    }
                    TurnEvent::TurnFinished { stop_reason } => {
                        flush_assistant_text(session, &mut assistant_text);
                        match stop_reason {
                            // The turn paused for tool execution; loop for another
                            // decision round only if a tool was actually run.
                            StopReason::ToolUse if had_tool_call => break,
                            // The backend would end the turn, but a steering
                            // message arrived while it was running: keep the turn
                            // alive and loop so the injected input is answered
                            // within the same prompt turn.
                            _ if self.has_pending_inbox() => break,
                            // A terminal tool is required to end the turn, but
                            // the backend finished without calling it. Nudge the
                            // model and keep the turn alive so it explicitly
                            // hands control back via a terminal tool.
                            _ if self.require_terminal_tool => {
                                debug!("turn finished without terminal tool; nudging to continue");
                                session.push_user_text(TERMINAL_TOOL_NUDGE.to_string());
                                break;
                            }
                            other => return Ok(other),
                        }
                    }
                    TurnEvent::Error(err) => {
                        flush_assistant_text(session, &mut assistant_text);
                        return Err(AgentError::Turn(err));
                    }
                }
            }

            iteration += 1;
        }
    }

    /// Dispatch a single tool call: record it, request permission if required,
    /// execute it while streaming updates, and append the typed result.
    async fn dispatch_tool(
        &self,
        session: &mut SessionState,
        sink: &mut dyn UpdateSink,
        cancel: &CancellationToken,
        id: ToolCallId,
        name: String,
        arguments: serde_json::Value,
    ) -> Result<()> {
        // Record the (typed) tool call in history regardless of outcome.
        let known = KnownTool::classify(&name, arguments.clone());
        session.push_tool_call(id.clone(), known);

        // Look up the tool up front so we can surface rich metadata (title,
        // kind, affected file locations) on the initial pending update.
        let tool = self.tools.get(&name);
        let (title, kind, locations) = match &tool {
            Some(tool) => (
                tool.title(&arguments).unwrap_or_else(|| name.clone()),
                tool.kind(),
                tool.locations(&arguments),
            ),
            None => (name.clone(), ToolKind::Other, Vec::new()),
        };

        sink.send(EngineOutput::ToolCall {
            id: id.clone(),
            name: name.clone(),
            title,
            kind,
            status: ToolCallStatus::Pending,
            locations,
            raw_input: Some(arguments.clone()),
        })
        .await?;

        let Some(tool) = tool else {
            let msg = format!("unknown tool: {name}");
            warn!("{msg}");
            sink.send(EngineOutput::ToolCallUpdate {
                id: id.clone(),
                status: ToolCallStatus::Failed,
                output: Some(msg.clone()),
                raw_output: None,
            })
            .await?;
            session.push_tool_result(id, false, msg);
            return Ok(());
        };

        // Enforce permission before any execution begins.
        if tool.requires_permission() {
            let granted = sink.request_permission(&id, &name).await?;
            if !granted {
                let msg = format!("permission denied for tool: {name}");
                debug!("{msg}");
                sink.send(EngineOutput::ToolCallUpdate {
                    id: id.clone(),
                    status: ToolCallStatus::Failed,
                    output: Some(msg.clone()),
                    raw_output: None,
                })
                .await?;
                session.push_tool_result(id, false, msg);
                return Ok(());
            }
        }

        sink.send(EngineOutput::ToolCallUpdate {
            id: id.clone(),
            status: ToolCallStatus::InProgress,
            output: None,
            raw_output: None,
        })
        .await?;

        let tool_ctx = ToolContext {
            session_id: session.session_id.clone(),
            cancel: cancel.clone(),
            depth: self.depth,
            client: self.client.clone(),
        };

        // Starting the tool may itself fail (e.g. argument validation).
        let mut events = match tool.call(arguments, &tool_ctx).await {
            Ok(events) => events,
            Err(err) => {
                let msg = err.to_string();
                warn!("tool {name} failed to start: {msg}");
                sink.send(EngineOutput::ToolCallUpdate {
                    id: id.clone(),
                    status: ToolCallStatus::Failed,
                    output: Some(msg.clone()),
                    raw_output: None,
                })
                .await?;
                session.push_tool_result(id, false, msg);
                return Ok(());
            }
        };

        let mut collected = String::new();
        let mut success = true;
        let mut final_message: Option<String> = None;
        let mut final_value: Option<serde_json::Value> = None;

        loop {
            let event = tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    debug!("cancellation observed while running tool {name}");
                    return Ok(());
                }
                next = events.next() => next,
            };

            let Some(event) = event else { break };

            match event {
                ToolEvent::Started => {}
                ToolEvent::OutputDelta(delta) => {
                    collected.push_str(&delta);
                    sink.send(EngineOutput::ToolCallUpdate {
                        id: id.clone(),
                        status: ToolCallStatus::InProgress,
                        output: Some(delta),
                        raw_output: None,
                    })
                    .await?;
                }
                ToolEvent::Completed(value) => {
                    let rendered = render_value(&value);
                    if !rendered.is_empty() {
                        collected.push_str(&rendered);
                    }
                    final_message = Some(rendered);
                    if !value.is_null() {
                        final_value = Some(value);
                    }
                }
                ToolEvent::Failed(message) => {
                    success = false;
                    final_message = Some(message);
                }
            }
        }

        let status = if success {
            ToolCallStatus::Completed
        } else {
            ToolCallStatus::Failed
        };
        sink.send(EngineOutput::ToolCallUpdate {
            id: id.clone(),
            status,
            output: final_message.clone(),
            raw_output: final_value,
        })
        .await?;

        let result_text = if let Some(msg) = final_message {
            if collected.is_empty() {
                msg
            } else {
                collected
            }
        } else {
            collected
        };
        // Very large tool outputs (e.g. reading a big file, a chatty terminal
        // command, or a verbose MCP tool) are offloaded to a file and only a
        // truncated preview is fed back to the model, so they don't blow up the
        // context / token budget. The full output stays available on disk for
        // the agent to `fs_read` on demand.
        let result_text = self
            .maybe_offload_large_output(session, &id, &name, result_text)
            .await;
        session.push_tool_result(id, success, result_text);
        Ok(())
    }

    /// If `text` exceeds [`MAX_TOOL_RESULT_CHARS`], write the full output to a
    /// file (via the client's filesystem access) and return a truncated preview
    /// annotated with the byte count and the file path; otherwise return `text`
    /// unchanged.
    ///
    /// This applies uniformly to every tool dispatched through the engine —
    /// built-in, MCP, and subagents — because they all funnel their results
    /// here.
    async fn maybe_offload_large_output(
        &self,
        session: &SessionState,
        id: &ToolCallId,
        name: &str,
        text: String,
    ) -> String {
        if text.chars().count() <= MAX_TOOL_RESULT_CHARS {
            return text;
        }

        let total_chars = text.chars().count();
        let preview = truncate_preview(&text);

        // Try to persist the full output so the agent can read it back in full.
        let saved_path = match &self.client {
            Some(client) => {
                let path = offload_path(&session.session_id, id);
                match client.write_text_file(&path, &text).await {
                    Ok(()) => Some(path),
                    Err(err) => {
                        warn!("failed to offload large output for tool {name}: {err}");
                        None
                    }
                }
            }
            None => None,
        };

        match saved_path {
            Some(path) => format!(
                "{preview}\n\n[Output truncated: showing {shown} of {total} characters. \
                 The full output was saved to {path}. Use the fs_read tool to view it.]",
                shown = preview.chars().count(),
                total = total_chars,
            ),
            None => format!(
                "{preview}\n\n[Output truncated: showing {shown} of {total} characters. \
                 The full output was too large to keep in context and could not be saved to a file.]",
                shown = preview.chars().count(),
                total = total_chars,
            ),
        }
    }
}

/// Maximum number of characters of a tool result kept inline in the
/// conversation before it is offloaded to a file (see
/// [`TurnEngine::maybe_offload_large_output`]).
pub const MAX_TOOL_RESULT_CHARS: usize = 8000;

/// Number of leading characters kept in a truncated preview.
const PREVIEW_HEAD_CHARS: usize = 4000;
/// Number of trailing characters kept in a truncated preview.
const PREVIEW_TAIL_CHARS: usize = 2000;

/// Build a head+tail preview of an over-long tool output.
fn truncate_preview(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let head: String = chars.iter().take(PREVIEW_HEAD_CHARS).collect();
    let tail: String = chars
        .iter()
        .skip(chars.len().saturating_sub(PREVIEW_TAIL_CHARS))
        .collect();
    format!("{head}\n...\n{tail}")
}

/// Compute a stable, unique file path for an offloaded tool output.
fn offload_path(session_id: &str, id: &ToolCallId) -> String {
    let sanitize = |s: &str| -> String {
        s.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect()
    };
    let dir = std::env::temp_dir().join("acp-agent-tool-outputs");
    let file = format!("{}-{}.txt", sanitize(session_id), sanitize(id.as_str()));
    dir.join(file).to_string_lossy().into_owned()
}

/// Persist accumulated assistant text as a single history entry, clearing the
/// buffer. No-op when the buffer is empty.
fn flush_assistant_text(session: &mut SessionState, buffer: &mut String) {
    if !buffer.is_empty() {
        session.push_assistant_text(std::mem::take(buffer));
    }
}

/// Render a JSON tool result into a compact human-readable string. Strings are
/// used verbatim; everything else is serialized as JSON.
fn render_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}
