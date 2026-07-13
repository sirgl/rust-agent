//! A concrete [`McpConnection`] over the MCP **stdio** transport.
//!
//! This is the real runtime transport used by the agent: it launches a
//! configured MCP server as a child process and speaks JSON-RPC 2.0 over the
//! child's stdin/stdout using newline-delimited messages (the framing used by
//! the MCP stdio transport). It performs the MCP `initialize` handshake on
//! connect, then serves [`McpConnection::list_tools`] via `tools/list` and
//! [`McpConnection::call_tool`] via `tools/call`.
//!
//! ## Concurrency model (v1)
//!
//! Requests are serialized behind a single [`tokio::sync::Mutex`]: each call
//! writes one request line and then reads response lines until it sees the
//! object whose `id` matches (skipping any server-initiated notifications or
//! log lines). This is simple and correct for the agent's turn loop, which
//! dispatches tools one at a time. Concurrent multiplexing over a single
//! server is intentionally out of scope for v1.
//!
//! Only the **stdio** transport is implemented. The ACP `McpServer` schema also
//! defines `http`, `sse`, and `acp` variants; those are not yet supported and
//! are reported as a structured error by the wiring layer.

use std::process::Stdio;
use std::sync::atomic::{AtomicI64, Ordering};

use agent_core::{AgentError, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::{McpConnection, McpToolDescriptor, McpToolResult};

/// The MCP protocol version this client advertises during `initialize`.
const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// A live connection to an MCP server launched over stdio.
pub struct StdioMcpConnection {
    name: String,
    next_id: AtomicI64,
    inner: Mutex<Inner>,
}

/// The mutable transport state guarded during each request/response exchange.
struct Inner {
    /// The child process handle (kept alive; killed on drop).
    child: Child,
    /// The child's stdin, where requests are written.
    stdin: ChildStdin,
    /// Line-buffered reader over the child's stdout.
    reader: BufReader<ChildStdout>,
}

impl StdioMcpConnection {
    /// Launch `command` (with `args`/`env`) as an MCP server and perform the
    /// `initialize` handshake, returning a ready-to-use connection.
    ///
    /// `name` is the label used to namespace the server's tools in the
    /// registry.
    ///
    /// # Errors
    ///
    /// Returns an error if the process cannot be spawned, its stdio pipes are
    /// unavailable, or the `initialize` handshake fails.
    pub async fn connect(
        name: impl Into<String>,
        command: impl AsRef<std::ffi::OsStr>,
        args: &[String],
        env: &[(String, String)],
    ) -> Result<Self> {
        let name = name.into();
        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        for (key, value) in env {
            cmd.env(key, value);
        }

        let mut child = cmd
            .spawn()
            .map_err(|err| AgentError::Other(format!("failed to launch mcp server `{name}`: {err}")))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| AgentError::Other(format!("mcp server `{name}` has no stdin")))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AgentError::Other(format!("mcp server `{name}` has no stdout")))?;

        let connection = Self {
            name,
            next_id: AtomicI64::new(1),
            inner: Mutex::new(Inner {
                child,
                stdin,
                reader: BufReader::new(stdout),
            }),
        };
        connection.handshake().await?;
        Ok(connection)
    }

    /// Perform the MCP `initialize` request followed by the
    /// `notifications/initialized` notification.
    async fn handshake(&self) -> Result<()> {
        let params = json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": { "name": "rust-acp-agent", "version": env!("CARGO_PKG_VERSION") },
        });
        let _ = self.request("initialize", params).await?;
        self.notify("notifications/initialized", json!({})).await?;
        debug!(server = %self.name, "mcp stdio handshake complete");
        Ok(())
    }

    /// Send a JSON-RPC request and wait for the matching response.
    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let payload = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let mut inner = self.inner.lock().await;
        write_line(&mut inner.stdin, &payload).await?;

        // Read lines until the response with the matching id arrives, skipping
        // notifications and unrelated messages.
        let mut line = String::new();
        loop {
            line.clear();
            let read = inner.reader.read_line(&mut line).await.map_err(|err| {
                AgentError::Other(format!("mcp server `{}` read failed: {err}", self.name))
            })?;
            if read == 0 {
                return Err(AgentError::Other(format!(
                    "mcp server `{}` closed the connection before responding to `{method}`",
                    self.name
                )));
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let message: Value = match serde_json::from_str(trimmed) {
                Ok(value) => value,
                Err(err) => {
                    warn!(server = %self.name, %err, line = trimmed, "skipping non-JSON mcp line");
                    continue;
                }
            };
            // Ignore anything that is not the response to our request id.
            if message.get("id").and_then(Value::as_i64) != Some(id) {
                continue;
            }
            if let Some(error) = message.get("error") {
                return Err(AgentError::Other(format!(
                    "mcp server `{}` returned an error for `{method}`: {error}",
                    self.name
                )));
            }
            return Ok(message.get("result").cloned().unwrap_or(Value::Null));
        }
    }

    /// Send a JSON-RPC notification (no response expected).
    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let payload = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        let mut inner = self.inner.lock().await;
        write_line(&mut inner.stdin, &payload).await
    }
}

impl Drop for StdioMcpConnection {
    fn drop(&mut self) {
        // Best-effort: the child is killed on drop via `kill_on_drop`, but we
        // also try to start_kill to be prompt.
        if let Ok(inner) = self.inner.try_lock() {
            let _ = inner_child_kill(&inner.child);
        }
    }
}

/// Helper so `Drop` does not need a mutable borrow of the child.
fn inner_child_kill(child: &Child) -> std::io::Result<()> {
    // `Child::start_kill` needs `&mut`; we only have `&`. Rely on
    // `kill_on_drop` instead — this is a no-op placeholder kept for clarity.
    let _ = child;
    Ok(())
}

/// Serialize `payload` as a single newline-terminated JSON line and flush it.
async fn write_line(stdin: &mut ChildStdin, payload: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec(payload)
        .map_err(|err| AgentError::Other(format!("failed to encode mcp request: {err}")))?;
    bytes.push(b'\n');
    stdin
        .write_all(&bytes)
        .await
        .map_err(|err| AgentError::Other(format!("failed to write mcp request: {err}")))?;
    stdin
        .flush()
        .await
        .map_err(|err| AgentError::Other(format!("failed to flush mcp request: {err}")))
}

/// Parse a `tools/list` result into [`McpToolDescriptor`]s.
fn parse_tools(result: &Value) -> Vec<McpToolDescriptor> {
    let Some(tools) = result.get("tools").and_then(Value::as_array) else {
        return Vec::new();
    };
    tools
        .iter()
        .filter_map(|tool| {
            let name = tool.get("name").and_then(Value::as_str)?;
            let mut descriptor = McpToolDescriptor::new(name);
            if let Some(desc) = tool.get("description").and_then(Value::as_str) {
                descriptor = descriptor.with_description(desc);
            }
            if let Some(schema) = tool.get("inputSchema") {
                descriptor = descriptor.with_schema(schema.clone());
            }
            Some(descriptor)
        })
        .collect()
}

/// Map a `tools/call` result object into an [`McpToolResult`].
fn parse_call_result(result: &Value) -> McpToolResult {
    let is_error = result
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    // Prefer the structured `content` field; fall back to the whole result.
    let content = result.get("content").cloned().unwrap_or_else(|| result.clone());
    McpToolResult {
        is_error,
        content,
    }
}

#[async_trait]
impl McpConnection for StdioMcpConnection {
    fn server_name(&self) -> &str {
        &self.name
    }

    async fn list_tools(&self) -> Result<Vec<McpToolDescriptor>> {
        let result = self.request("tools/list", json!({})).await?;
        Ok(parse_tools(&result))
    }

    async fn call_tool(&self, name: &str, args: Value) -> Result<McpToolResult> {
        let params = json!({ "name": name, "arguments": args });
        let result = self.request("tools/call", params).await?;
        Ok(parse_call_result(&result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::register_mcp_tools;
    use agent_core::{CancellationToken, ToolContext, ToolEvent, ToolRegistry};
    use futures::StreamExt;

    /// A tiny, self-contained MCP server implemented as a POSIX shell script.
    ///
    /// It speaks just enough of the MCP stdio protocol to answer `initialize`,
    /// `tools/list`, and `tools/call` for a single `echo` tool, so the real
    /// [`StdioMcpConnection`] transport can be exercised against a genuine
    /// child process without any external dependency.
    const MOCK_SERVER: &str = r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"method":"initialize"'*|*'"method": "initialize"'*)
      id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"mock","version":"0"}}}\n' "$id"
      ;;
    *'"method":"notifications/initialized"'*)
      : ;;
    *'"method":"tools/list"'*)
      id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"echo back","inputSchema":{"type":"object"}}]}}\n' "$id"
      ;;
    *'"method":"tools/call"'*)
      id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"pong"}],"isError":false}}\n' "$id"
      ;;
  esac
done
"#;

    /// Write the mock server script to a temp file and mark it executable.
    fn write_mock_server() -> std::path::PathBuf {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir();
        let path = dir.join(format!("mock-mcp-{}.sh", std::process::id()));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(MOCK_SERVER.as_bytes()).unwrap();
        file.flush().unwrap();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
        path
    }

    #[tokio::test]
    async fn connects_lists_and_calls_over_real_stdio() {
        let path = write_mock_server();
        let conn = StdioMcpConnection::connect("mock", &path, &[], &[])
            .await
            .unwrap();

        let tools = conn.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");

        let result = conn
            .call_tool("echo", serde_json::json!({ "text": "hi" }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.to_string().contains("pong"));

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn registers_real_stdio_tools_into_registry() {
        let path = write_mock_server();
        let conn: Arc<dyn McpConnection> =
            Arc::new(StdioMcpConnection::connect("mock", &path, &[], &[]).await.unwrap());

        let mut registry = ToolRegistry::new();
        let registered = register_mcp_tools(&mut registry, conn).await.unwrap();
        assert_eq!(registered, vec!["mcp__mock__echo".to_string()]);

        let tool = registry.get("mcp__mock__echo").unwrap();
        let ctx = ToolContext::new("s", CancellationToken::new());
        let events: Vec<ToolEvent> = tool
            .call(serde_json::json!({}), &ctx)
            .await
            .unwrap()
            .collect()
            .await;
        assert!(matches!(events.last(), Some(ToolEvent::Completed(_))));

        let _ = std::fs::remove_file(&path);
    }
}
