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
//! Three tools are provided:
//! - [`FsReadTool`] (`fs_read`) — read a text file. Non-destructive, so it does
//!   **not** require permission.
//! - [`FsWriteTool`] (`fs_write`) — write a text file. Destructive, so it
//!   **requires permission** (enforced by the engine before `Tool::call`).
//! - [`TerminalRunTool`] (`terminal_run`) — run a shell command. Destructive, so
//!   it **requires permission**.
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
    AgentError, ClientAccess, ElicitationOutcome, Result, Tool, ToolContext, ToolEvent,
    ToolRegistry,
};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use serde::Deserialize;
use tracing::debug;

/// Register all built-in filesystem and terminal tools into `registry`.
pub fn register_builtin_tools(registry: &mut ToolRegistry) {
    registry.register(Arc::new(FsReadTool));
    registry.register(Arc::new(FsWriteTool));
    registry.register(Arc::new(TerminalRunTool));
    registry.register(Arc::new(ElicitationTool));
}

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

/// Arguments for [`TerminalRunTool`].
#[derive(Debug, Deserialize)]
struct TerminalRunArgs {
    /// The command to execute.
    command: String,
    /// Optional command-line arguments.
    #[serde(default)]
    args: Vec<String>,
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
                    "description": "The command to execute."
                },
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional command-line arguments."
                }
            },
            "required": ["command"]
        })
    }

    fn requires_permission(&self) -> bool {
        // Running commands is destructive.
        true
    }

    async fn call(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        let args: TerminalRunArgs = parse_args(self.name(), args)?;
        let client: &Arc<dyn ClientAccess> = ctx.client()?;
        debug!(command = %args.command, "terminal_run");
        match client.run_terminal(&args.command, &args.args).await {
            Ok(outcome) => {
                let mut stream_events = vec![ToolEvent::Started];
                if !outcome.output.is_empty() {
                    stream_events.push(ToolEvent::OutputDelta(outcome.output.clone()));
                }
                let payload = serde_json::json!({
                    "output": outcome.output,
                    "truncated": outcome.truncated,
                    "exit_code": outcome.exit_code,
                    "signal": outcome.signal,
                });
                if outcome.is_success() {
                    stream_events.push(ToolEvent::Completed(payload));
                } else {
                    let detail = match (outcome.exit_code, &outcome.signal) {
                        (_, Some(sig)) => format!("terminated by signal {sig}"),
                        (Some(code), None) => format!("exited with code {code}"),
                        (None, None) => "exited abnormally".to_string(),
                    };
                    stream_events.push(ToolEvent::Failed(format!(
                        "command `{}` {detail}: {}",
                        args.command, outcome.output
                    )));
                }
                Ok(events(stream_events))
            }
            Err(err) => Ok(events(vec![
                ToolEvent::Started,
                ToolEvent::Failed(format!("failed to run `{}`: {err}", args.command)),
            ])),
        }
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
        let result = tool.call(serde_json::json!({ "path": "/tmp/x" }), &ctx).await;
        assert!(matches!(
            result.err(),
            Some(AgentError::InvalidArguments(_))
        ));
    }

    #[tokio::test]
    async fn missing_client_yields_error() {
        let ctx = ToolContext::new("sess-test", CancellationToken::new());
        let tool = FsReadTool;
        let result = tool.call(serde_json::json!({ "path": "/tmp/a" }), &ctx).await;
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

    #[test]
    fn registry_registers_all_builtins() {
        let registry = builtin_registry();
        assert!(registry.contains("fs_read"));
        assert!(registry.contains("fs_write"));
        assert!(registry.contains("terminal_run"));
        assert!(registry.contains("elicitation"));
    }
}
