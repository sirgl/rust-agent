//! `mcp-client`: connect to configured MCP (Model Context Protocol) servers and
//! expose their tools inside the turn loop as provider-independent
//! [`agent_core::Tool`]s.
//!
//! Like every capability in the **services layer**, MCP tools are written once
//! and shared by every decision-layer provider (Anthropic, replay, scripted,
//! ...). They depend only on `agent-core` trait seams, never on a specific
//! provider.
//!
//! ## Seam
//!
//! The actual wire transport to an MCP server (stdio / JSON-RPC) is hidden
//! behind the [`McpConnection`] trait so the tool-wrapping and registration
//! logic is fully testable against an in-memory stub server without any child
//! process or socket. A concrete transport can implement [`McpConnection`] and
//! plug in unchanged.
//!
//! ## Flow
//!
//! 1. [`McpConnection::list_tools`] enumerates the tools a server advertises.
//! 2. [`register_mcp_tools`] wraps each advertised tool in an [`McpTool`] and
//!    registers it in the shared [`ToolRegistry`], namespaced so tools from
//!    different servers cannot collide.
//! 3. When the engine dispatches an [`McpTool`], its
//!    [`call`](agent_core::Tool::call) forwards the JSON arguments to
//!    [`McpConnection::call_tool`] and maps the [`McpToolResult`] into
//!    [`ToolEvent`]s.

#![warn(missing_docs)]

pub mod stdio;

use std::sync::Arc;

use agent_core::{AgentError, Result, Tool, ToolContext, ToolEvent, ToolRegistry};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use tracing::debug;

/// Static configuration for one MCP server the agent should connect to.
///
/// v1 only records how to launch/identify a server; establishing the concrete
/// transport is the responsibility of an [`McpConnection`] implementation. This
/// struct is what config wiring in the binary parses/holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerConfig {
    /// A short, unique label for the server, used to namespace its tools in the
    /// [`ToolRegistry`] (e.g. `search` -> tool `mcp__search__query`).
    pub name: String,
    /// The command used to launch the server (for a stdio transport).
    pub command: String,
    /// Arguments passed to the launch command.
    pub args: Vec<String>,
}

impl McpServerConfig {
    /// Create a new server config with no launch arguments.
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
            args: Vec::new(),
        }
    }

    /// Builder-style setter for launch arguments.
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }
}

/// A tool advertised by an MCP server.
#[derive(Debug, Clone, PartialEq)]
pub struct McpToolDescriptor {
    /// The tool name as advertised by the server (unqualified).
    pub name: String,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// JSON schema describing the tool's input arguments.
    pub input_schema: serde_json::Value,
    /// Whether the tool should require explicit user permission before running.
    ///
    /// MCP tools are external code; a server can mark a tool destructive so the
    /// engine gates it behind `session/request_permission` like the built-in
    /// destructive tools. Defaults to `false` via [`McpToolDescriptor::new`].
    pub requires_permission: bool,
}

impl McpToolDescriptor {
    /// Create a descriptor with an empty-object schema and no description.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            input_schema: serde_json::json!({ "type": "object" }),
            requires_permission: false,
        }
    }

    /// Builder-style setter for the input schema.
    pub fn with_schema(mut self, schema: serde_json::Value) -> Self {
        self.input_schema = schema;
        self
    }

    /// Builder-style setter for the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Builder-style setter marking the tool as requiring permission.
    pub fn requiring_permission(mut self) -> Self {
        self.requires_permission = true;
        self
    }
}

/// The result of invoking an MCP tool.
#[derive(Debug, Clone, PartialEq)]
pub struct McpToolResult {
    /// Whether the call succeeded (`false` maps to a failed tool call).
    pub is_error: bool,
    /// The tool output rendered as a JSON value fed back into the conversation.
    pub content: serde_json::Value,
}

impl McpToolResult {
    /// A successful result carrying the given content.
    pub fn ok(content: serde_json::Value) -> Self {
        Self {
            is_error: false,
            content,
        }
    }

    /// A failed result carrying an error message.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            is_error: true,
            content: serde_json::Value::String(message.into()),
        }
    }
}

/// A connected MCP server.
///
/// This is the transport seam: implementations may talk to a real server over
/// stdio/JSON-RPC, or, in tests, serve canned tools/results in memory. The
/// tool-wrapping and registry code depends only on this trait.
#[async_trait]
pub trait McpConnection: Send + Sync {
    /// The label of the server (used to namespace its tools).
    fn server_name(&self) -> &str;

    /// Enumerate the tools this server exposes.
    async fn list_tools(&self) -> Result<Vec<McpToolDescriptor>>;

    /// Invoke a tool by its (unqualified) name with JSON arguments.
    async fn call_tool(&self, name: &str, args: serde_json::Value) -> Result<McpToolResult>;
}

/// Build the namespaced registry name for an MCP tool.
///
/// Tools are exposed as `mcp__<server>__<tool>` so that tools from different
/// servers (or with the same name as a built-in) cannot collide in the shared
/// [`ToolRegistry`].
pub fn qualified_tool_name(server: &str, tool: &str) -> String {
    format!("mcp__{server}__{tool}")
}

/// An [`agent_core::Tool`] wrapping a single tool exposed by an MCP server.
pub struct McpTool {
    connection: Arc<dyn McpConnection>,
    descriptor: McpToolDescriptor,
    qualified_name: String,
}

impl McpTool {
    /// Wrap a server tool described by `descriptor`, reachable through
    /// `connection`.
    pub fn new(connection: Arc<dyn McpConnection>, descriptor: McpToolDescriptor) -> Self {
        let qualified_name = qualified_tool_name(connection.server_name(), &descriptor.name);
        Self {
            connection,
            descriptor,
            qualified_name,
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.qualified_name
    }

    fn schema(&self) -> serde_json::Value {
        self.descriptor.input_schema.clone()
    }

    fn requires_permission(&self) -> bool {
        self.descriptor.requires_permission
    }

    async fn call(
        &self,
        args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        debug!(tool = %self.qualified_name, "mcp tool call");
        match self
            .connection
            .call_tool(&self.descriptor.name, args)
            .await
        {
            Ok(result) if result.is_error => Ok(events(vec![
                ToolEvent::Started,
                ToolEvent::Failed(render_error(&result.content)),
            ])),
            Ok(result) => Ok(events(vec![
                ToolEvent::Started,
                ToolEvent::Completed(result.content),
            ])),
            Err(err) => Ok(events(vec![
                ToolEvent::Started,
                ToolEvent::Failed(format!("mcp tool `{}` failed: {err}", self.qualified_name)),
            ])),
        }
    }
}

/// Discover the tools exposed by `connection` and register each as an
/// [`McpTool`] in `registry`, returning the qualified names registered.
///
/// This is the single entry point the binary's config wiring calls per
/// configured server.
pub async fn register_mcp_tools(
    registry: &mut ToolRegistry,
    connection: Arc<dyn McpConnection>,
) -> Result<Vec<String>> {
    let descriptors = connection.list_tools().await?;
    if descriptors.is_empty() {
        return Err(AgentError::Other(format!(
            "mcp server `{}` advertised no tools",
            connection.server_name()
        )));
    }
    let mut registered = Vec::with_capacity(descriptors.len());
    for descriptor in descriptors {
        let tool = McpTool::new(connection.clone(), descriptor);
        registered.push(tool.name().to_string());
        registry.register(Arc::new(tool));
    }
    debug!(
        server = connection.server_name(),
        count = registered.len(),
        "registered mcp tools"
    );
    Ok(registered)
}

/// Render a stream of pre-computed [`ToolEvent`]s (work is done eagerly in
/// `call`, then replayed to the engine).
fn events(events: Vec<ToolEvent>) -> BoxStream<'static, ToolEvent> {
    Box::pin(stream::iter(events))
}

/// Render an MCP error payload into a human-readable message.
fn render_error(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use agent_core::CancellationToken;
    use futures::StreamExt;

    /// An in-memory stub MCP server serving canned tools and results, and
    /// recording the calls it received.
    #[derive(Default)]
    struct StubServer {
        name: String,
        tools: Vec<McpToolDescriptor>,
        calls: Mutex<Vec<(String, serde_json::Value)>>,
        next_result: Mutex<Option<McpToolResult>>,
    }

    impl StubServer {
        fn new(name: &str, tools: Vec<McpToolDescriptor>) -> Self {
            Self {
                name: name.to_string(),
                tools,
                ..Default::default()
            }
        }
    }

    #[async_trait]
    impl McpConnection for StubServer {
        fn server_name(&self) -> &str {
            &self.name
        }

        async fn list_tools(&self) -> Result<Vec<McpToolDescriptor>> {
            Ok(self.tools.clone())
        }

        async fn call_tool(&self, name: &str, args: serde_json::Value) -> Result<McpToolResult> {
            self.calls
                .lock()
                .unwrap()
                .push((name.to_string(), args.clone()));
            match self.next_result.lock().unwrap().take() {
                Some(r) => Ok(r),
                None => Ok(McpToolResult::ok(serde_json::json!({ "echo": args }))),
            }
        }
    }

    fn ctx() -> ToolContext {
        ToolContext::new("sess-test", CancellationToken::new())
    }

    async fn collect(stream: BoxStream<'static, ToolEvent>) -> Vec<ToolEvent> {
        stream.collect().await
    }

    #[tokio::test]
    async fn discovers_and_namespaces_tools() {
        let server = Arc::new(StubServer::new(
            "search",
            vec![
                McpToolDescriptor::new("query").with_description("search the web"),
                McpToolDescriptor::new("index"),
            ],
        ));
        let mut registry = ToolRegistry::new();
        let registered = register_mcp_tools(&mut registry, server).await.unwrap();

        assert_eq!(registry.len(), 2);
        assert!(registry.contains("mcp__search__query"));
        assert!(registry.contains("mcp__search__index"));
        assert_eq!(
            registered,
            vec![
                "mcp__search__query".to_string(),
                "mcp__search__index".to_string()
            ]
        );
    }

    #[tokio::test]
    async fn empty_server_is_an_error() {
        let server = Arc::new(StubServer::new("empty", vec![]));
        let mut registry = ToolRegistry::new();
        let result = register_mcp_tools(&mut registry, server).await;
        assert!(matches!(result, Err(AgentError::Other(_))));
    }

    #[tokio::test]
    async fn call_forwards_args_and_maps_success() {
        let server = Arc::new(StubServer::new(
            "search",
            vec![McpToolDescriptor::new("query")],
        ));
        let mut registry = ToolRegistry::new();
        register_mcp_tools(&mut registry, server.clone())
            .await
            .unwrap();

        let tool = registry.get("mcp__search__query").unwrap();
        let stream = tool
            .call(serde_json::json!({ "q": "rust" }), &ctx())
            .await
            .unwrap();
        let evts = collect(stream).await;

        assert!(matches!(evts[0], ToolEvent::Started));
        match evts.last() {
            Some(ToolEvent::Completed(v)) => {
                assert_eq!(v, &serde_json::json!({ "echo": { "q": "rust" } }));
            }
            other => panic!("unexpected: {other:?}"),
        }
        // The unqualified tool name is forwarded to the server.
        let calls = server.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "query");
        assert_eq!(calls[0].1, serde_json::json!({ "q": "rust" }));
    }

    #[tokio::test]
    async fn error_result_maps_to_failed_event() {
        let server = Arc::new(StubServer::new(
            "search",
            vec![McpToolDescriptor::new("query")],
        ));
        *server.next_result.lock().unwrap() = Some(McpToolResult::error("boom"));
        let mut registry = ToolRegistry::new();
        register_mcp_tools(&mut registry, server).await.unwrap();

        let tool = registry.get("mcp__search__query").unwrap();
        let stream = tool.call(serde_json::json!({}), &ctx()).await.unwrap();
        let evts = collect(stream).await;
        assert!(matches!(evts.last(), Some(ToolEvent::Failed(m)) if m == "boom"));
    }

    #[tokio::test]
    async fn descriptor_permission_and_schema_are_exposed() {
        let server = Arc::new(StubServer::new(
            "fs",
            vec![McpToolDescriptor::new("dangerous")
                .with_schema(serde_json::json!({ "type": "object", "properties": {} }))
                .requiring_permission()],
        ));
        let mut registry = ToolRegistry::new();
        register_mcp_tools(&mut registry, server).await.unwrap();
        let tool = registry.get("mcp__fs__dangerous").unwrap();
        assert!(tool.requires_permission());
        assert_eq!(
            tool.schema(),
            serde_json::json!({ "type": "object", "properties": {} })
        );
    }
}
