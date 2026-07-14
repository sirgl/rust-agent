//! Configuration for the `acp-agent` binary.
//!
//! Configuration is intentionally minimal for v1: it carries the agent
//! identity advertised during `initialize` and an optional system prompt seeded
//! into every new session. Values are read from the environment so the binary
//! can be configured without a config file, while tests construct
//! [`AgentConfig`] directly.

use std::path::PathBuf;

use mcp_client::McpServerConfig;
use subagents::DEFAULT_MAX_DEPTH;

/// Default system prompt seeded into every new session.
///
/// It defines the agent's character: proactive, rigorous about building a
/// coherent picture of the problem, unwilling to accept vague or ill-posed
/// requests, and disciplined about persisting decisions in documents. The tone
/// is deliberately terse: say everything that matters, and nothing that does
/// not.
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a proactive engineering agent. You do not passively wait for perfect \
instructions: you take initiative, drive the task toward a working outcome, and \
anticipate the next step instead of stopping at the literal request.\n\n\
Build a coherent picture of the world before acting. Reconcile the request with \
the actual state of the code, docs, and prior decisions; surface contradictions \
instead of silently picking one side. When something is ambiguous, \
underspecified, or self-contradictory, ask focused clarifying questions rather \
than guessing. Do not agree to nonsense: if a request is wrong, infeasible, or \
based on a false premise, say so plainly and propose a better path.\n\n\
Actively interrogate the user. Do not soldier on over an unresolved ambiguity, \
a missing input, or a risky assumption \u{2014} draw the answer out. Prefer the \
`elicitation` tool for anything structured (a value, a choice, a config, a \
credential): it gives the user a proper form and returns typed data. When a \
tool is not suitable, just ask in plain text (the user sees it live and can \
reply mid-turn) or pose the question in your `submit_result` summary and hand \
control back. Ask early and specifically rather than late and vaguely.\n\n\
Fix knowledge in documents. Record decisions, designs, assumptions, and \
trade-offs in the appropriate files so the picture stays consistent and \
nothing important lives only in the conversation.\n\n\
Be succinct, but full: communicate as briefly as possible while staying \
complete. State every fact that matters and omit everything that does not. No \
filler, no hedging, no restating the question back \u{2014} just the substance.\n\n\
Ending a turn is an explicit act. Your plain text never ends the turn: control \
returns to the user only when you call the `submit_result` tool with a concise \
summary of what you did. So keep working \u{2014} think, call tools, make progress \
\u{2014} until the task is actually done, then call `submit_result`. If you truly \
need the user before you can continue (a real question or a blocking decision), \
ask it and then call `submit_result` to hand control back. Do not call \
`submit_result` while work you can still do remains.\n\n\
Your plain text is still streamed to the user in real time, even though it does \
not end the turn. So you can think out loud and narrate what you are doing, and \
the user may send a comment mid-turn to steer you \u{2014} treat any such incoming \
message as immediate guidance and adjust course before continuing.";

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
    /// Default model used for new sessions.
    pub default_model: String,
    /// Whether a turn only ends when the model calls the `submit_result` tool.
    ///
    /// When `true` (the default), the turn engine is configured with
    /// `submit_result` as its terminal tool: a decision round that produces only
    /// text does not end the turn; the model is nudged to keep working and hand
    /// control back explicitly. When `false`, plain text ends the turn as
    /// before (used by tests that script fixed replay turns).
    pub require_submit_result: bool,
    /// Cadence (in decision rounds) at which the agent is nudged to briefly
    /// share what it is planning: once right after the user prompt, then every
    /// N rounds. `0` disables the behavior.
    pub thought_interval: usize,
    /// Directory under which session core state is persisted as JSON, one
    /// file per session.
    ///
    /// When `Some`, the binary selects a disk-backed
    /// [`JsonFileSessionStore`](agent_core::JsonFileSessionStore) so sessions
    /// survive process restarts and can be reopened via `session/load`. When
    /// `None` (the default), a non-durable
    /// [`InMemorySessionStore`](agent_core::InMemorySessionStore) is used.
    pub session_persistence_dir: Option<PathBuf>,
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
            default_model: "claude-sonnet-5".to_string(),
            require_submit_result: true,
            thought_interval: 5,
            session_persistence_dir: None,
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
    /// - `ACP_REQUIRE_SUBMIT_RESULT` (`1`/`true`/`yes`/`on`) toggles whether a
    ///   turn only ends when the model calls the `submit_result` tool.
    /// - `ACP_SESSION_DIR` selects a directory for disk-backed session
    ///   persistence (enables the JSON file store); unset keeps sessions
    ///   in memory only.
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
        if let Ok(model) = std::env::var("ANTHROPIC_MODEL") {
            if !model.is_empty() {
                config.default_model = model;
            }
        }
        if let Ok(v) = std::env::var("ACP_REQUIRE_SUBMIT_RESULT") {
            config.require_submit_result = matches!(v.trim(), "1" | "true" | "yes" | "on");
        }
        if let Ok(v) = std::env::var("ACP_THOUGHT_INTERVAL") {
            if let Ok(n) = v.trim().parse::<usize>() {
                config.thought_interval = n;
            }
        }
        if let Ok(dir) = std::env::var("ACP_SESSION_DIR") {
            if !dir.trim().is_empty() {
                config.session_persistence_dir = Some(PathBuf::from(dir));
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
    fn default_seeds_system_prompt() {
        let config = AgentConfig::default();
        assert_eq!(config.system_prompt.as_deref(), Some(DEFAULT_SYSTEM_PROMPT));
    }

    #[test]
    fn default_model_is_sonnet() {
        let config = AgentConfig::default();
        assert_eq!(config.default_model, "claude-sonnet-5");
    }

    #[test]
    fn default_requires_submit_result() {
        let config = AgentConfig::default();
        assert!(config.require_submit_result);
    }

    #[test]
    fn default_thought_interval_is_five() {
        let config = AgentConfig::default();
        assert_eq!(config.thought_interval, 5);
    }

    #[test]
    fn from_env_reads_thought_interval() {
        std::env::set_var("ACP_THOUGHT_INTERVAL", "3");
        let config = AgentConfig::from_env();
        assert_eq!(config.thought_interval, 3);
        std::env::remove_var("ACP_THOUGHT_INTERVAL");
    }

    #[test]
    fn default_has_no_session_persistence_dir() {
        let config = AgentConfig::default();
        assert!(config.session_persistence_dir.is_none());
    }

    #[test]
    fn from_env_reads_session_dir() {
        std::env::set_var("ACP_SESSION_DIR", "/tmp/acp-sessions");
        let config = AgentConfig::from_env();
        assert_eq!(
            config.session_persistence_dir.as_deref(),
            Some(std::path::Path::new("/tmp/acp-sessions"))
        );
        std::env::remove_var("ACP_SESSION_DIR");
    }
}
