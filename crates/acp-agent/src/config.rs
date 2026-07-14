//! Configuration for the `acp-agent` binary.
//!
//! Configuration is intentionally minimal for v1: it carries the agent
//! identity advertised during `initialize` and an optional system prompt seeded
//! into every new session. Values are read from the environment so the binary
//! can be configured without a config file, while tests construct
//! [`AgentConfig`] directly.

use mcp_client::McpServerConfig;
use subagents::DEFAULT_MAX_DEPTH;

/// Default system prompt seeded into every new session unless overridden.
///
/// It defines the agent's persona and working style (consistent, thorough,
/// proactive, best-practices-driven) and, crucially, makes it *communicate its
/// intent*: it should keep the user posted with very short (1-2 line) plan and
/// progress notes instead of going silent for many steps of thinking and tool
/// calls. The engine reinforces this by periodically injecting a short nudge
/// (see `agent_core::DEFAULT_PROGRESS_NUDGE`).
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are an AI coding agent working inside a user's editor. Act like a careful, \
senior engineer.\n\
\n\
How you work:\n\
- Be consistent and predictable; follow the existing conventions and patterns \
of the project and keep your approach coherent across the whole task.\n\
- Clarify the task in detail: figure out the precise intent and requirements. \
If something is genuinely ambiguous or a decision is risky, ask a short, \
focused clarifying question before doing heavy or irreversible work.\n\
- Be proactive: anticipate what the user really needs, investigate edge cases, \
error paths, concurrency and data races, and other corner cases rather than \
only the happy path.\n\
- Use best practices for correctness, safety and maintainability; prefer \
simple, robust solutions over clever but fragile ones.\n\
- Answer in a structured way (short sections/bullets when useful) but stay \
succinct — no filler, minimal text, high signal.\n\
\n\
Communication rule (important): keep the user informed of your intent. Before \
starting work on a request — and again as you make progress across multiple \
tool calls — send a very short, plain-language message (1-2 lines maximum, \
minimal text) saying what you plan to do next and why. Do not go silent for \
many steps while only thinking or calling tools. These are quick plan/progress \
notes, not the full answer.";

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
    /// Default model used for new sessions.
    pub default_model: String,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "rust-acp-agent".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            system_prompt: Some(DEFAULT_SYSTEM_PROMPT.to_string()),
            enable_subagents: true,
            subagent_max_depth: DEFAULT_MAX_DEPTH,
            mcp_servers: Vec::new(),
            enable_orchestrated: false,
            enable_orchestrate_tool: false,
            default_model: "claude-sonnet-5".to_string(),
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
    /// - `ANTHROPIC_MODEL` overrides the default model.
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
        if let Ok(model) = std::env::var("ANTHROPIC_MODEL") {
            if !model.is_empty() {
                config.default_model = model;
            }
        }
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_reads_model() {
        std::env::set_var("ANTHROPIC_MODEL", "claude-test");
        let config = AgentConfig::from_env();
        assert_eq!(config.default_model, "claude-test");
        std::env::remove_var("ANTHROPIC_MODEL");
    }

    #[test]
    fn default_model_is_sonnet() {
        let config = AgentConfig::default();
        assert_eq!(config.default_model, "claude-sonnet-5");
    }

    #[test]
    fn default_seeds_communication_system_prompt() {
        let config = AgentConfig::default();
        let prompt = config
            .system_prompt
            .expect("a default system prompt should be seeded");
        assert!(prompt.contains("1-2 lines"));
    }
}
