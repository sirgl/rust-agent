//! Configuration for the `acp-agent` binary.
//!
//! Configuration is intentionally minimal for v1: it carries the agent
//! identity advertised during `initialize` and an optional system prompt seeded
//! into every new session. Values are read from the environment so the binary
//! can be configured without a config file, while tests construct
//! [`AgentConfig`] directly.

use mcp_client::McpServerConfig;
use subagents::DEFAULT_MAX_DEPTH;

/// Static configuration for the agent process.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Programmatic implementation name advertised to the client.
    pub name: String,
    /// Implementation version advertised to the client.
    pub version: String,
    /// Optional system prompt seeded into every new session.
    pub system_prompt: Option<String>,
    /// Whether to expose the subagent delegation tool in the tool registry.
    pub enable_subagents: bool,
    /// Maximum subagent nesting depth enforced by the delegation tool.
    pub subagent_max_depth: usize,
    /// MCP servers whose tools should be discovered and registered.
    pub mcp_servers: Vec<McpServerConfig>,
    /// Whether to run in orchestrated mode (an orchestrator LLM drives
    /// sub-agents via `run_subagent`) instead of a plain turn loop.
    ///
    /// This is a basic wiring toggle; full orchestrated front-end UX is out of
    /// scope for v1. When enabled, callers assemble an orchestrator from the
    /// `orchestrated` crate using the same backend factories.
    pub enable_orchestrated: bool,
    /// Whether to expose the `orchestrate` tool in the chat tool registry, so
    /// the chat agent can launch the orchestration pipeline itself mid-turn.
    pub enable_orchestrate_tool: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "rust-acp-agent".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            system_prompt: None,
            enable_subagents: true,
            subagent_max_depth: DEFAULT_MAX_DEPTH,
            mcp_servers: Vec::new(),
            enable_orchestrated: false,
            enable_orchestrate_tool: false,
        }
    }
}

impl AgentConfig {
    /// Load configuration from the environment, falling back to defaults.
    ///
    /// Recognized variables:
    /// - `ACP_AGENT_NAME` overrides the advertised implementation name.
    /// - `ACP_AGENT_SYSTEM_PROMPT` seeds a system prompt into new sessions.
    /// - `ACP_ORCHESTRATED` (`1`/`true`/`yes`/`on`) enables orchestrated mode.
    #[must_use]
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(name) = std::env::var("ACP_AGENT_NAME") {
            if !name.is_empty() {
                config.name = name;
            }
        }
        if let Ok(prompt) = std::env::var("ACP_AGENT_SYSTEM_PROMPT") {
            if !prompt.is_empty() {
                config.system_prompt = Some(prompt);
            }
        }
        if let Ok(v) = std::env::var("ACP_ORCHESTRATED") {
            config.enable_orchestrated = matches!(v.trim(), "1" | "true" | "yes" | "on");
        }
        if let Ok(v) = std::env::var("ACP_ORCHESTRATE_TOOL") {
            config.enable_orchestrate_tool = matches!(v.trim(), "1" | "true" | "yes" | "on");
        }
        config
    }
}
