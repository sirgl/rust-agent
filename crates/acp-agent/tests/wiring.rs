//! Integration tests for the Step 6 config wiring: the subagent delegation tool
//! is present in the assembled registry, and MCP tools discovered from a stub
//! server register through the public `acp_agent` helper.

use std::sync::Arc;

use acp_agent::{register_configured_mcp_tools, AgentConfig, McpConnectionTrait, McpToolDescriptor};
use agent_core::{Result, ToolRegistry};
use async_trait::async_trait;

/// A minimal in-memory MCP server used to exercise discovery wiring without a
/// real transport.
struct StubServer;

#[async_trait]
impl McpConnectionTrait for StubServer {
    fn server_name(&self) -> &str {
        "demo"
    }
    async fn list_tools(&self) -> Result<Vec<McpToolDescriptor>> {
        Ok(vec![McpToolDescriptor::new("echo")])
    }
    async fn call_tool(
        &self,
        _name: &str,
        args: serde_json::Value,
    ) -> Result<mcp_client_reexport::McpToolResult> {
        Ok(mcp_client_reexport::McpToolResult::ok(args))
    }
}

// The result type isn't re-exported by acp_agent; pull it from the crate.
mod mcp_client_reexport {
    pub use mcp_client::McpToolResult;
}

#[test]
fn build_registry_includes_subagent_tool() {
    let config = AgentConfig::default();
    let factory = acp_agent::canned_replay_factory();
    let registry = acp_agent::build_registry(&config, &factory);
    assert!(
        registry.contains("delegate"),
        "subagent delegation tool should be registered when enabled"
    );
}

#[test]
fn build_registry_omits_subagent_when_disabled() {
    let mut config = AgentConfig::default();
    config.enable_subagents = false;
    let factory = acp_agent::canned_replay_factory();
    let registry = acp_agent::build_registry(&config, &factory);
    assert!(!registry.contains("delegate"));
}

#[tokio::test]
async fn mcp_tools_register_through_wiring_helper() {
    let mut registry = ToolRegistry::new();
    let registered = register_configured_mcp_tools(&mut registry, Arc::new(StubServer))
        .await
        .expect("registration succeeds");
    assert_eq!(registered, vec!["mcp__demo__echo".to_string()]);
    assert!(registry.contains("mcp__demo__echo"));
}
