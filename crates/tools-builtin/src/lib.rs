//! `tools-builtin`: filesystem and terminal [`agent_core::Tool`]s implemented via
//! the provider-independent [`agent_core::ClientAccess`] seam.
//!
//! These tools are part of the **services layer**: they are written once and
//! shared by every decision-layer provider (Anthropic, replay, scripted, ...).
//! They never depend on a specific provider or on the concrete
//! `agent-client-protocol` client — only on the [`agent_core::ClientAccess`]
//! trait, which the ACP binary implements against the live connection and tests
//! implement in memory.
//!
//! The built-in tools provided are:
//! - [`FsReadTool`] (`fs_read`) — read a text file. Non-destructive, so it does
//!   **not** require permission.
//! - [`FsWriteTool`] (`fs_write`) — write a text file. Destructive, so it
//!   **requires permission** (enforced by the engine before `Tool::call`).
//! - [`TerminalRunTool`] (`terminal_run`) — run a shell command. Destructive, so
//!   it **requires permission**.
//! - [`UpdateTodoListTool`] (`update_todo_list`) — atomically replace the
//!   session's canonical todo list.
//! - [`MarkTodoCompletedTool`] (`mark_todo_completed`) — complete one item by
//!   stable id without changing its order or content.
//! - [`SubmitResultTool`] (`submit_result`) — end the current turn and hand
//!   control back to the user. Non-destructive; recognized by the engine as the
//!   terminal tool (see [`agent_core::TurnEngine::with_terminal_tool`]).
//!
//! Each tool validates its JSON arguments up front (returning
//! [`agent_core::AgentError::InvalidArguments`] on malformed input, which the
//! engine surfaces as a failed tool call) and then streams
//! [`ToolEvent`]s: [`ToolEvent::Started`], zero or more
//! [`ToolEvent::OutputDelta`], and a terminal [`ToolEvent::Completed`] or
//! [`ToolEvent::Failed`].

#![warn(missing_docs)]

use std::sync::Arc;

use agent_core::{
    AgentError, ClientAccess, ElicitationOutcome, MarkTodoCompletedToolCall, PlanStepStatus,
    Result, TerminalChunk, Tool, ToolCallLocation, ToolContext, ToolEvent, ToolKind, ToolRegistry,
    UpdateTodoListToolCall,
};
use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use serde::Deserialize;
use serde_json::json;
use tracing::debug;

/// Register all built-in filesystem and terminal tools into `registry`.
pub fn register_builtin_tools(registry: &mut ToolRegistry) {
    registry.register(Arc::new(FsReadTool));
    registry.register(Arc::new(FsWriteTool));
    registry.register(Arc::new(TerminalRunTool));
    registry.register(Arc::new(ElicitationTool));
    registry.register(Arc::new(UpdateTodoListTool));
    registry.register(Arc::new(MarkTodoCompletedTool));
    registry.register(Arc::new(SubmitResultTool));
}

/// The registry name of the built-in turn-terminating tool.
pub const SUBMIT_RESULT_TOOL_NAME: &str = "submit_result";

/// Build a fresh registry pre-populated with the built-in tools.
pub fn builtin_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    register_builtin_tools(&mut registry);
    registry
}

/// Turn a stream of pre-computed [`ToolEvent`]s into the boxed stream tools
/// return. Work is performed eagerly inside `call`; the resulting events are
/// then replayed to the engine, which forwards status transitions to the sink.
fn events(events: Vec<ToolEvent>) -> BoxStream<'static, ToolEvent> {
    Box::pin(stream::iter(events))
}

fn todo_updated_events(todo_list: agent_core::TodoList) -> Result<BoxStream<'static, ToolEvent>> {
    let result = serde_json::to_value(&todo_list)
        .map_err(|err| AgentError::Other(format!("failed to serialize todo list: {err}")))?;
    Ok(events(vec![
        ToolEvent::Started,
        ToolEvent::TodoListUpdated(todo_list),
        ToolEvent::Completed(result),
    ]))
}

/// Deserialize tool arguments, mapping failures to a structured
/// [`AgentError::InvalidArguments`] so the engine can report a failed tool call
/// instead of panicking.
fn parse_args<T: for<'de> Deserialize<'de>>(name: &str, args: serde_json::Value) -> Result<T> {
    serde_json::from_value(args)
        .map_err(|err| AgentError::InvalidArguments(format!("{name}: {err}")))
}

/// Arguments for [`FsReadTool`].
#[derive(Debug, Deserialize)]
struct FsReadArgs {
    /// Absolute path of the file to read.
    path: String,
}

/// Arguments for [`FsWriteTool`].
#[derive(Debug, Deserialize)]
struct FsWriteArgs {
    /// Absolute path of the file to write.
    path: String,
    /// The exact text content to write.
    content: String,
}

/// Arguments for [`ElicitationTool`].
#[derive(Debug, Deserialize)]
struct ElicitationArgs {
    /// Human-readable message describing what input is needed.
    message: String,
    /// JSON Schema (an `"object"` schema) describing the requested form fields.
    requested_schema: serde_json::Value,
}

/// Arguments for [`SubmitResultTool`].
#[derive(Debug, Deserialize)]
struct SubmitResultArgs {
    /// A concise, user-facing summary of the work performed this turn.
    summary: String,
}

/// Arguments for [`TerminalRunTool`].
#[derive(Debug, Deserialize)]
struct TerminalRunArgs {
    /// The command to execute.
    command: String,
    /// Optional command-line arguments.
    #[serde(default)]
    args: Vec<String>,
}

/// The built-in `submit_result` tool: ends the current turn and hands control
/// back to the user.
///
/// This tool performs no side effects; it exists purely as the explicit,
/// typed signal that the turn is complete. The turn engine, when configured
/// with [`agent_core::TurnEngine::with_terminal_tool`], recognizes a call to
/// this tool as the sole way to end a turn, so the agent keeps working until
/// it deliberately submits its result rather than stopping on plain text.
pub struct SubmitResultTool;

#[async_trait]
impl Tool for SubmitResultTool {
    fn name(&self) -> &str {
        SUBMIT_RESULT_TOOL_NAME
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A concise, user-facing summary of what was done this turn. \
        Keep in mind the user has already seen every plain-text message you streamed earlier this \
        turn, so this summary should read as a smooth continuation of that text, not repeat it. \
        If you already fully answered in plain text just before submitting, prefer an empty summary \
        so no redundant closing message is shown."
                }
            },
            "required": ["summary"]
        })
    }

    fn requires_permission(&self) -> bool {
        // Ending the turn is not a destructive action.
        false
    }

    fn ends_turn(&self) -> bool {
        // This is the terminal tool: dispatching it ends the turn.
        true
    }

    async fn call(
        &self,
        args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        let args: SubmitResultArgs = parse_args(self.name(), args)?;
        debug!("submit_result: ending turn");
        Ok(events(vec![
            ToolEvent::Started,
            ToolEvent::Completed(serde_json::json!({
                "submitted": true,
                "summary": args.summary,
            })),
        ]))
    }
}

/// The built-in `fs_read` tool: read a text file via the client.
pub struct FsReadTool;

#[async_trait]
impl Tool for FsReadTool {
    fn name(&self) -> &str {
        "fs_read"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path of the file to read."
                }
            },
            "required": ["path"]
        })
    }

    fn requires_permission(&self) -> bool {
        // Reading is non-destructive.
        false
    }

    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }

    fn title(&self, args: &serde_json::Value) -> Option<String> {
        serde_json::from_value::<FsReadArgs>(args.clone())
            .ok()
            .map(|a| format!("Read {}", a.path))
    }

    fn locations(&self, args: &serde_json::Value) -> Vec<ToolCallLocation> {
        serde_json::from_value::<FsReadArgs>(args.clone())
            .ok()
            .map(|a| vec![ToolCallLocation::new(a.path)])
            .unwrap_or_default()
    }

    async fn call(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        let args: FsReadArgs = parse_args(self.name(), args)?;
        let client: &Arc<dyn ClientAccess> = ctx.client()?;
        debug!(path = %args.path, "fs_read");
        match client.read_text_file(&args.path).await {
            Ok(content) => Ok(events(vec![
                ToolEvent::Started,
                ToolEvent::OutputDelta(content.clone()),
                ToolEvent::Completed(serde_json::json!({ "content": content })),
            ])),
            Err(err) => Ok(events(vec![
                ToolEvent::Started,
                ToolEvent::Failed(format!("failed to read {}: {err}", args.path)),
            ])),
        }
    }
}

/// The built-in `fs_write` tool: write a text file via the client.
pub struct FsWriteTool;

#[async_trait]
impl Tool for FsWriteTool {
    fn name(&self) -> &str {
        "fs_write"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path of the file to write."
                },
                "content": {
                    "type": "string",
                    "description": "The exact text content to write."
                }
            },
            "required": ["path", "content"]
        })
    }

    fn requires_permission(&self) -> bool {
        // Writing to disk is destructive.
        true
    }

    fn kind(&self) -> ToolKind {
        ToolKind::Edit
    }

    fn title(&self, args: &serde_json::Value) -> Option<String> {
        serde_json::from_value::<FsWriteArgs>(args.clone())
            .ok()
            .map(|a| format!("Write {}", a.path))
    }

    fn locations(&self, args: &serde_json::Value) -> Vec<ToolCallLocation> {
        serde_json::from_value::<FsWriteArgs>(args.clone())
            .ok()
            .map(|a| vec![ToolCallLocation::new(a.path)])
            .unwrap_or_default()
    }

    async fn call(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        let args: FsWriteArgs = parse_args(self.name(), args)?;
        let client: &Arc<dyn ClientAccess> = ctx.client()?;
        debug!(path = %args.path, bytes = args.content.len(), "fs_write");
        match client.write_text_file(&args.path, &args.content).await {
            Ok(()) => {
                let msg = format!("wrote {} bytes to {}", args.content.len(), args.path);
                Ok(events(vec![
                    ToolEvent::Started,
                    ToolEvent::Completed(serde_json::json!({
                        "path": args.path,
                        "bytes_written": args.content.len(),
                        "message": msg,
                    })),
                ]))
            }
            Err(err) => Ok(events(vec![
                ToolEvent::Started,
                ToolEvent::Failed(format!("failed to write {}: {err}", args.path)),
            ])),
        }
    }
}

/// The built-in `terminal_run` tool: run a command via the client.
pub struct TerminalRunTool;

#[async_trait]
impl Tool for TerminalRunTool {
    fn name(&self) -> &str {
        "terminal_run"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "A complete shell command to execute. Supports flags, quoting, pipes, redirects, and command chaining. Relative paths resolve from the session working directory."
                }
            },
            "required": ["command"]
        })
    }

    fn requires_permission(&self) -> bool {
        // Running commands is destructive.
        true
    }

    fn kind(&self) -> ToolKind {
        ToolKind::Execute
    }

    fn title(&self, args: &serde_json::Value) -> Option<String> {
        serde_json::from_value::<TerminalRunArgs>(args.clone())
            .ok()
            .map(|a| {
                if a.args.is_empty() {
                    format!("Run {}", a.command)
                } else {
                    format!("Run {} {}", a.command, a.args.join(" "))
                }
            })
    }

    async fn call(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        let args: TerminalRunArgs = parse_args(self.name(), args)?;
        let client: &Arc<dyn ClientAccess> = ctx.client()?;
        debug!(command = %args.command, "terminal_run");
        // Stream output incrementally: each chunk of stdout/stderr becomes an
        // `OutputDelta`, giving the client a "live log" as the command runs, and
        // the final `Finished` chunk maps to `Completed`/`Failed`.
        let term_stream = match if args.args.is_empty() {
            client.run_shell_command_streaming(&args.command).await
        } else {
            // Backward compatibility for typed history or callers using the
            // legacy executable-plus-args payload. New schemas expose only the
            // full shell-command form above.
            client
                .run_terminal_streaming(&args.command, &args.args)
                .await
        } {
            Ok(stream) => stream,
            Err(err) => {
                return Ok(events(vec![
                    ToolEvent::Started,
                    ToolEvent::Failed(format!("failed to run `{}`: {err}", args.command)),
                ]));
            }
        };

        let command = args.command.clone();
        let mapped = term_stream.map(move |chunk| match chunk {
            TerminalChunk::Output(text) => ToolEvent::OutputDelta(text),
            TerminalChunk::Finished(outcome) => {
                let payload = json!({
                    "output": outcome.output,
                    "truncated": outcome.truncated,
                    "exit_code": outcome.exit_code,
                    "signal": outcome.signal,
                });
                if outcome.is_success() {
                    ToolEvent::Completed(payload)
                } else {
                    let detail = match (outcome.exit_code, &outcome.signal) {
                        (_, Some(sig)) => format!("terminated by signal {sig}"),
                        (Some(code), None) => format!("exited with code {code}"),
                        (None, None) => "exited abnormally".to_string(),
                    };
                    ToolEvent::Failed(format!("command `{command}` {detail}"))
                }
            }
        });

        let started = stream::once(async { ToolEvent::Started });
        Ok(Box::pin(started.chain(mapped)))
    }
}

/// The built-in `elicitation` tool: request structured input from the user.
///
/// Exposes the [`ClientAccess::request_elicitation`] capability to the decision
/// layer so a model can ask the user for structured input (e.g. an API key or a
/// configuration choice) mid-turn. The provided `requested_schema` is a JSON
/// Schema describing the form fields; the client renders it and returns the
/// user's response.
pub struct ElicitationTool;

/// Replace the complete canonical todo list for the current session.
pub struct UpdateTodoListTool;

#[async_trait]
impl Tool for UpdateTodoListTool {
    fn name(&self) -> &str {
        "update_todo_list"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "items": {
                    "type": "array",
                    "description": "The complete replacement todo list. Omitted prior items are removed; an empty array clears the list.",
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "id": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Stable unique identifier chosen by the caller."
                            },
                            "content": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Non-empty description of the work item."
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"]
                            }
                        },
                        "required": ["id", "content", "status"]
                    }
                }
            },
            "required": ["items"]
        })
    }

    fn requires_permission(&self) -> bool {
        false
    }

    fn kind(&self) -> ToolKind {
        ToolKind::Think
    }

    fn title(&self, args: &serde_json::Value) -> Option<String> {
        let count = args.get("items")?.as_array()?.len();
        Some(format!("Update todo list ({count} items)"))
    }

    async fn call(
        &self,
        args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        let args: UpdateTodoListToolCall = parse_args(self.name(), args)?;
        todo_updated_events(args.0)
    }
}

/// Mark one canonical todo item completed by its stable identifier.
pub struct MarkTodoCompletedTool;

#[async_trait]
impl Tool for MarkTodoCompletedTool {
    fn name(&self) -> &str {
        "mark_todo_completed"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "id": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Stable identifier of the todo item to mark completed."
                }
            },
            "required": ["id"]
        })
    }

    fn requires_permission(&self) -> bool {
        false
    }

    fn kind(&self) -> ToolKind {
        ToolKind::Think
    }

    fn title(&self, args: &serde_json::Value) -> Option<String> {
        Some(format!("Complete todo item {}", args.get("id")?.as_str()?))
    }

    async fn call(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        let args: MarkTodoCompletedToolCall = parse_args(self.name(), args)?;
        let mut todo_list = ctx.todo_list.clone();
        let Some(item) = todo_list.items.iter_mut().find(|item| item.id == args.id) else {
            return Err(AgentError::InvalidArguments(format!(
                "{}: unknown todo item id '{}'",
                self.name(),
                args.id
            )));
        };
        item.status = PlanStepStatus::Completed;
        todo_updated_events(todo_list)
    }
}

#[async_trait]
impl Tool for ElicitationTool {
    fn name(&self) -> &str {
        "elicitation"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Human-readable message describing what input is needed from the user."
                },
                "requested_schema": {
                    "type": "object",
                    "description": "A JSON Schema (an object schema with a `properties` map) describing the form fields to request from the user."
                }
            },
            "required": ["message", "requested_schema"]
        })
    }

    fn requires_permission(&self) -> bool {
        // Asking the user for input is interactive, not destructive.
        false
    }

    async fn call(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        let args: ElicitationArgs = parse_args(self.name(), args)?;
        let client: &Arc<dyn ClientAccess> = ctx.client()?;
        debug!(message = %args.message, "elicitation");
        match client
            .request_elicitation(&args.message, args.requested_schema)
            .await
        {
            Ok(ElicitationOutcome::Accepted(content)) => Ok(events(vec![
                ToolEvent::Started,
                ToolEvent::Completed(serde_json::json!({
                    "accepted": true,
                    "content": content,
                })),
            ])),
            Ok(ElicitationOutcome::Declined) => Ok(events(vec![
                ToolEvent::Started,
                ToolEvent::Failed("user declined to provide the requested input".to_string()),
            ])),
            Ok(ElicitationOutcome::Cancelled) => Ok(events(vec![
                ToolEvent::Started,
                ToolEvent::Failed("elicitation was cancelled".to_string()),
            ])),
            Err(err) => Ok(events(vec![
                ToolEvent::Started,
                ToolEvent::Failed(format!("elicitation failed: {err}")),
            ])),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use agent_core::{CancellationToken, TerminalOutcome};
    use futures::StreamExt;

    /// An in-memory fake [`ClientAccess`] recording writes and serving canned
    /// reads / terminal outcomes.
    #[derive(Default)]
    struct FakeClient {
        read_result: Mutex<Option<Result<String>>>,
        writes: Mutex<Vec<(String, String)>>,
        terminal_result: Mutex<Option<Result<TerminalOutcome>>>,
        elicitation_result: Mutex<Option<Result<ElicitationOutcome>>>,
    }

    #[async_trait]
    impl ClientAccess for FakeClient {
        async fn read_text_file(&self, path: &str) -> Result<String> {
            match self.read_result.lock().unwrap().take() {
                Some(r) => r,
                None => Ok(format!("contents of {path}")),
            }
        }

        async fn write_text_file(&self, path: &str, content: &str) -> Result<()> {
            self.writes
                .lock()
                .unwrap()
                .push((path.to_string(), content.to_string()));
            Ok(())
        }

        async fn run_terminal(&self, _command: &str, _args: &[String]) -> Result<TerminalOutcome> {
            match self.terminal_result.lock().unwrap().take() {
                Some(r) => r,
                None => Ok(TerminalOutcome {
                    output: "ok".into(),
                    truncated: false,
                    exit_code: Some(0),
                    signal: None,
                }),
            }
        }

        async fn request_elicitation(
            &self,
            _message: &str,
            _requested_schema: serde_json::Value,
        ) -> Result<ElicitationOutcome> {
            match self.elicitation_result.lock().unwrap().take() {
                Some(r) => r,
                None => Ok(ElicitationOutcome::Accepted(serde_json::json!({}))),
            }
        }
    }

    fn ctx_with(client: Arc<dyn ClientAccess>) -> ToolContext {
        ToolContext::new("sess-test", CancellationToken::new()).with_client(client)
    }

    async fn collect(stream: BoxStream<'static, ToolEvent>) -> Vec<ToolEvent> {
        stream.collect().await
    }

    #[tokio::test]
    async fn fs_read_streams_content() {
        let client = Arc::new(FakeClient::default());
        let ctx = ctx_with(client);
        let tool = FsReadTool;
        assert!(!tool.requires_permission());
        let stream = tool
            .call(serde_json::json!({ "path": "/tmp/a.txt" }), &ctx)
            .await
            .unwrap();
        let evts = collect(stream).await;
        assert!(matches!(evts[0], ToolEvent::Started));
        assert!(matches!(evts.last(), Some(ToolEvent::Completed(_))));
    }

    #[tokio::test]
    async fn fs_write_records_write_and_requires_permission() {
        let client = Arc::new(FakeClient::default());
        let ctx = ctx_with(client.clone());
        let tool = FsWriteTool;
        assert!(tool.requires_permission());
        let stream = tool
            .call(
                serde_json::json!({ "path": "/tmp/out.txt", "content": "hello" }),
                &ctx,
            )
            .await
            .unwrap();
        let evts = collect(stream).await;
        assert!(matches!(evts.last(), Some(ToolEvent::Completed(_))));
        let writes = client.writes.lock().unwrap();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, "/tmp/out.txt");
        assert_eq!(writes[0].1, "hello");
    }

    #[tokio::test]
    async fn terminal_run_failure_maps_to_failed_event() {
        let client = Arc::new(FakeClient::default());
        *client.terminal_result.lock().unwrap() = Some(Ok(TerminalOutcome {
            output: "boom".into(),
            truncated: false,
            exit_code: Some(2),
            signal: None,
        }));
        let ctx = ctx_with(client);
        let tool = TerminalRunTool;
        let stream = tool
            .call(serde_json::json!({ "command": "false" }), &ctx)
            .await
            .unwrap();
        let evts = collect(stream).await;
        assert!(matches!(evts.last(), Some(ToolEvent::Failed(_))));
    }

    #[tokio::test]
    async fn malformed_arguments_error_before_execution() {
        let client = Arc::new(FakeClient::default());
        let ctx = ctx_with(client);
        let tool = FsWriteTool;
        // Missing required `content`.
        let result = tool
            .call(serde_json::json!({ "path": "/tmp/x" }), &ctx)
            .await;
        assert!(matches!(
            result.err(),
            Some(AgentError::InvalidArguments(_))
        ));
    }

    #[tokio::test]
    async fn missing_client_yields_error() {
        let ctx = ToolContext::new("sess-test", CancellationToken::new());
        let tool = FsReadTool;
        let result = tool
            .call(serde_json::json!({ "path": "/tmp/a" }), &ctx)
            .await;
        assert!(matches!(result.err(), Some(AgentError::Other(_))));
    }

    #[tokio::test]
    async fn elicitation_accept_completes_with_content() {
        let client = Arc::new(FakeClient::default());
        *client.elicitation_result.lock().unwrap() = Some(Ok(ElicitationOutcome::Accepted(
            serde_json::json!({ "name": "Ada" }),
        )));
        let ctx = ctx_with(client);
        let tool = ElicitationTool;
        assert!(!tool.requires_permission());
        let stream = tool
            .call(
                serde_json::json!({
                    "message": "What is your name?",
                    "requested_schema": { "type": "object", "properties": {} }
                }),
                &ctx,
            )
            .await
            .unwrap();
        let evts = collect(stream).await;
        match evts.last() {
            Some(ToolEvent::Completed(payload)) => {
                assert_eq!(payload["accepted"], serde_json::json!(true));
                assert_eq!(payload["content"]["name"], serde_json::json!("Ada"));
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn elicitation_decline_maps_to_failed_event() {
        let client = Arc::new(FakeClient::default());
        *client.elicitation_result.lock().unwrap() = Some(Ok(ElicitationOutcome::Declined));
        let ctx = ctx_with(client);
        let tool = ElicitationTool;
        let stream = tool
            .call(
                serde_json::json!({
                    "message": "Provide a key",
                    "requested_schema": { "type": "object" }
                }),
                &ctx,
            )
            .await
            .unwrap();
        let evts = collect(stream).await;
        assert!(matches!(evts.last(), Some(ToolEvent::Failed(_))));
    }

    #[tokio::test]
    async fn elicitation_cancel_maps_to_failed_event() {
        let client = Arc::new(FakeClient::default());
        *client.elicitation_result.lock().unwrap() = Some(Ok(ElicitationOutcome::Cancelled));
        let ctx = ctx_with(client);
        let tool = ElicitationTool;
        let stream = tool
            .call(
                serde_json::json!({
                    "message": "Provide a key",
                    "requested_schema": { "type": "object" }
                }),
                &ctx,
            )
            .await
            .unwrap();
        let evts = collect(stream).await;
        assert!(matches!(evts.last(), Some(ToolEvent::Failed(_))));
    }

    #[tokio::test]
    async fn submit_result_completes_and_needs_no_client() {
        // No client attached: submit_result must not require one.
        let ctx = ToolContext::new("sess-test", CancellationToken::new());
        let tool = SubmitResultTool;
        assert!(!tool.requires_permission());
        assert!(tool.ends_turn());
        let stream = tool
            .call(serde_json::json!({ "summary": "all done" }), &ctx)
            .await
            .unwrap();
        let evts = collect(stream).await;
        match evts.last() {
            Some(ToolEvent::Completed(payload)) => {
                assert_eq!(payload["submitted"], serde_json::json!(true));
                assert_eq!(payload["summary"], serde_json::json!("all done"));
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn submit_result_requires_summary() {
        let ctx = ToolContext::new("sess-test", CancellationToken::new());
        let tool = SubmitResultTool;
        let result = tool.call(serde_json::json!({}), &ctx).await;
        assert!(matches!(
            result.err(),
            Some(AgentError::InvalidArguments(_))
        ));
    }

    #[tokio::test]
    async fn update_todo_list_replaces_and_clears_the_full_list() {
        let ctx = ToolContext::new("sess-test", CancellationToken::new());
        let tool = UpdateTodoListTool;
        let stream = tool
            .call(
                serde_json::json!({
                    "items": [
                        {"id": "b", "content": "Second", "status": "in_progress"},
                        {"id": "a", "content": "First", "status": "pending"}
                    ]
                }),
                &ctx,
            )
            .await
            .unwrap();
        let evts = collect(stream).await;
        let ToolEvent::TodoListUpdated(list) = &evts[1] else {
            panic!("expected todo update, got {:?}", evts[1]);
        };
        assert_eq!(list.items[0].id.as_str(), "b");
        assert_eq!(list.items[1].id.as_str(), "a");
        assert!(matches!(evts[2], ToolEvent::Completed(_)));

        let cleared = collect(
            tool.call(serde_json::json!({"items": []}), &ctx)
                .await
                .unwrap(),
        )
        .await;
        let ToolEvent::TodoListUpdated(list) = &cleared[1] else {
            panic!("expected todo update");
        };
        assert!(list.items.is_empty());
    }

    #[tokio::test]
    async fn update_todo_list_rejects_duplicate_or_empty_items() {
        let ctx = ToolContext::new("sess-test", CancellationToken::new());
        let tool = UpdateTodoListTool;
        for invalid in [
            serde_json::json!({
                "items": [
                    {"id": "a", "content": "One", "status": "pending"},
                    {"id": "a", "content": "Two", "status": "pending"}
                ]
            }),
            serde_json::json!({
                "items": [{"id": "a", "content": " ", "status": "pending"}]
            }),
        ] {
            assert!(matches!(
                tool.call(invalid, &ctx).await,
                Err(AgentError::InvalidArguments(_))
            ));
        }
    }

    #[tokio::test]
    async fn mark_todo_completed_preserves_order_and_is_idempotent() {
        let mut ctx = ToolContext::new("sess-test", CancellationToken::new());
        ctx.todo_list = serde_json::from_value(serde_json::json!({
            "items": [
                {"id": "a", "content": "First", "status": "pending"},
                {"id": "b", "content": "Second", "status": "in_progress"}
            ]
        }))
        .unwrap();
        let tool = MarkTodoCompletedTool;

        for _ in 0..2 {
            let evts = collect(
                tool.call(serde_json::json!({"id": "b"}), &ctx)
                    .await
                    .unwrap(),
            )
            .await;
            let ToolEvent::TodoListUpdated(list) = &evts[1] else {
                panic!("expected todo update");
            };
            assert_eq!(list.items[0].id.as_str(), "a");
            assert_eq!(list.items[0].status, PlanStepStatus::Pending);
            assert_eq!(list.items[1].id.as_str(), "b");
            assert_eq!(list.items[1].content, "Second");
            assert_eq!(list.items[1].status, PlanStepStatus::Completed);
            ctx.todo_list = list.clone();
        }
    }

    #[tokio::test]
    async fn mark_todo_completed_rejects_unknown_id_without_mutation() {
        let mut ctx = ToolContext::new("sess-test", CancellationToken::new());
        ctx.todo_list = serde_json::from_value(serde_json::json!({
            "items": [{"id": "known", "content": "Known", "status": "pending"}]
        }))
        .unwrap();
        let before = ctx.todo_list.clone();
        let result = MarkTodoCompletedTool
            .call(serde_json::json!({"id": "missing"}), &ctx)
            .await;

        assert!(matches!(result, Err(AgentError::InvalidArguments(_))));
        assert_eq!(ctx.todo_list, before);
    }

    #[test]
    fn registry_registers_all_builtins() {
        let registry = builtin_registry();
        assert!(registry.contains("fs_read"));
        assert!(registry.contains("fs_write"));
        assert!(registry.contains("terminal_run"));
        assert!(registry.contains("elicitation"));
        assert!(registry.contains("update_todo_list"));
        assert!(registry.contains("mark_todo_completed"));
        assert!(registry.contains("submit_result"));
    }

    #[test]
    fn tools_expose_rich_metadata() {
        let read_args = serde_json::json!({ "path": "/tmp/a.txt" });
        assert_eq!(FsReadTool.kind(), ToolKind::Read);
        assert_eq!(
            FsReadTool.title(&read_args).as_deref(),
            Some("Read /tmp/a.txt")
        );
        let read_locs = FsReadTool.locations(&read_args);
        assert_eq!(read_locs.len(), 1);
        assert_eq!(read_locs[0].path, "/tmp/a.txt");

        let write_args = serde_json::json!({ "path": "/tmp/b.txt", "content": "x" });
        assert_eq!(FsWriteTool.kind(), ToolKind::Edit);
        assert_eq!(
            FsWriteTool.title(&write_args).as_deref(),
            Some("Write /tmp/b.txt")
        );
        assert_eq!(FsWriteTool.locations(&write_args).len(), 1);

        let run_args = serde_json::json!({ "command": "ls", "args": ["-la"] });
        assert_eq!(TerminalRunTool.kind(), ToolKind::Execute);
        assert_eq!(
            TerminalRunTool.title(&run_args).as_deref(),
            Some("Run ls -la")
        );
        assert!(TerminalRunTool.locations(&run_args).is_empty());

        let todo_args = serde_json::json!({"items": []});
        assert_eq!(UpdateTodoListTool.kind(), ToolKind::Think);
        assert_eq!(
            UpdateTodoListTool.title(&todo_args).as_deref(),
            Some("Update todo list (0 items)")
        );
        assert_eq!(MarkTodoCompletedTool.kind(), ToolKind::Think);
    }

    #[test]
    fn terminal_run_schema_exposes_one_full_shell_command() {
        let schema = TerminalRunTool.schema();
        assert!(schema["properties"].get("args").is_none());
        assert!(schema["properties"]["command"]["description"]
            .as_str()
            .unwrap()
            .contains("pipes"));
    }

    /// A fake client that streams several output chunks before finishing.
    #[derive(Default)]
    struct StreamingFakeClient;

    #[async_trait]
    impl ClientAccess for StreamingFakeClient {
        async fn read_text_file(&self, _path: &str) -> Result<String> {
            Ok(String::new())
        }
        async fn write_text_file(&self, _path: &str, _content: &str) -> Result<()> {
            Ok(())
        }
        async fn run_terminal(&self, _command: &str, _args: &[String]) -> Result<TerminalOutcome> {
            unreachable!("streaming path should be used")
        }
        async fn run_terminal_streaming(
            &self,
            _command: &str,
            _args: &[String],
        ) -> Result<BoxStream<'static, TerminalChunk>> {
            let chunks = vec![
                TerminalChunk::Output("line1\n".into()),
                TerminalChunk::Output("line2\n".into()),
                TerminalChunk::Finished(TerminalOutcome {
                    output: "line1\nline2\n".into(),
                    truncated: false,
                    exit_code: Some(0),
                    signal: None,
                }),
            ];
            Ok(Box::pin(stream::iter(chunks)))
        }
    }

    #[tokio::test]
    async fn terminal_run_streams_incremental_output() {
        let client = Arc::new(StreamingFakeClient);
        let ctx = ctx_with(client);
        let tool = TerminalRunTool;
        let stream = tool
            .call(serde_json::json!({ "command": "echo" }), &ctx)
            .await
            .unwrap();
        let evts = collect(stream).await;

        assert!(matches!(evts[0], ToolEvent::Started));
        let deltas: Vec<&String> = evts
            .iter()
            .filter_map(|e| match e {
                ToolEvent::OutputDelta(text) => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(deltas.len(), 2, "expected two incremental output chunks");
        assert_eq!(deltas[0], "line1\n");
        assert_eq!(deltas[1], "line2\n");
        assert!(matches!(evts.last(), Some(ToolEvent::Completed(_))));
    }
}
