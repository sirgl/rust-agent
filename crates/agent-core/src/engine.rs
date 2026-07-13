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

use std::sync::Arc;

use futures::StreamExt;
use tracing::{debug, warn};

use crate::cancel::CancellationToken;
use crate::client::ClientAccess;
use crate::error::{AgentError, Result};
use crate::event::{StopReason, ToolCallId, TurnEvent};
use crate::history::KnownTool;
use crate::service::NextTurnService;
use crate::session::SessionState;
use crate::sink::{EngineOutput, ToolCallStatus, UpdateSink};
use crate::tool::{ToolContext, ToolEvent};
use crate::tool::ToolRegistry;

/// Default upper bound on decision/tool iterations within a single prompt to
/// guard against a misbehaving backend looping forever.
pub const DEFAULT_MAX_ITERATIONS: usize = 128;

/// Drives the agent turn loop.
///
/// A `TurnEngine` couples a [`NextTurnService`] (the decision layer) with a
/// [`ToolRegistry`] (the services layer). It is deliberately provider-agnostic:
/// swapping the `NextTurnService` implementation (e.g. Anthropic vs. replay) is
/// all that is required to change behavior.
pub struct TurnEngine {
    next_turn: Arc<dyn NextTurnService>,
    tools: ToolRegistry,
    max_iterations: usize,
    client: Option<Arc<dyn ClientAccess>>,
    depth: usize,
}

impl TurnEngine {
    /// Create a new engine from a decision service and a tool registry.
    pub fn new(next_turn: Arc<dyn NextTurnService>, tools: ToolRegistry) -> Self {
        Self {
            next_turn,
            tools,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            client: None,
            depth: 0,
        }
    }

    /// Override the maximum number of loop iterations.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Attach a [`ClientAccess`] implementation so dispatched tools can reach
    /// the ACP client's filesystem and terminal capabilities.
    pub fn with_client(mut self, client: Arc<dyn ClientAccess>) -> Self {
        self.client = Some(client);
        self
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
        for iteration in 0..self.max_iterations {
            if cancel.is_cancelled() {
                debug!("cancellation observed before iteration {iteration}");
                return Ok(StopReason::Cancelled);
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
                    TurnEvent::Plan(plan) => {
                        sink.send(EngineOutput::Plan(plan)).await?;
                    }
                    TurnEvent::ToolCallRequested {
                        id,
                        name,
                        arguments,
                    } => {
                        had_tool_call = true;
                        // Persist any assistant text emitted before this tool call.
                        flush_assistant_text(session, &mut assistant_text);
                        self.dispatch_tool(session, sink, cancel, id, name, arguments)
                            .await?;
                        if cancel.is_cancelled() {
                            return Ok(StopReason::Cancelled);
                        }
                    }
                    TurnEvent::TurnFinished { stop_reason } => {
                        flush_assistant_text(session, &mut assistant_text);
                        match stop_reason {
                            // The turn paused for tool execution; loop for another
                            // decision round only if a tool was actually run.
                            StopReason::ToolUse if had_tool_call => break,
                            other => return Ok(other),
                        }
                    }
                    TurnEvent::Error(err) => {
                        flush_assistant_text(session, &mut assistant_text);
                        return Err(AgentError::Turn(err));
                    }
                }
            }
        }

        warn!("turn loop exceeded max iterations ({})", self.max_iterations);
        Ok(StopReason::EndTurn)
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

        sink.send(EngineOutput::ToolCall {
            id: id.clone(),
            name: name.clone(),
            status: ToolCallStatus::Pending,
        })
        .await?;

        let Some(tool) = self.tools.get(&name) else {
            let msg = format!("unknown tool: {name}");
            warn!("{msg}");
            sink.send(EngineOutput::ToolCallUpdate {
                id: id.clone(),
                status: ToolCallStatus::Failed,
                output: Some(msg.clone()),
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
                })
                .await?;
                session.push_tool_result(id, false, msg);
                return Ok(());
            }
        };

        let mut collected = String::new();
        let mut success = true;
        let mut final_message: Option<String> = None;

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
                    })
                    .await?;
                }
                ToolEvent::Completed(value) => {
                    let rendered = render_value(&value);
                    if !rendered.is_empty() {
                        collected.push_str(&rendered);
                    }
                    final_message = Some(rendered);
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
        session.push_tool_result(id, success, result_text);
        Ok(())
    }
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
