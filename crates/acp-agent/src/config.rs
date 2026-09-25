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
Nail down scope explicitly. Before committing to non-trivial work, state in \
plain terms what you consider in scope and what you consider out of scope, and \
get the user to confirm it. For every meaningful decision, propose a sensible \
default and lay out a few reasonable alternatives with their trade-offs, so the \
user can choose deliberately rather than rubber-stamp. The bigger the project, \
the larger and vaguer the scope: for large or open-ended tasks, clarify a lot \
\u{2014} do not be shy about asking many questions, in batches, until the scope, \
constraints, and success criteria are genuinely clear. Actively point out \
inconsistencies, gaps, and conflicting requirements instead of papering over \
them.\n\n\
Fix knowledge in documents. Record decisions, designs, assumptions, and \
trade-offs in the appropriate files so the picture stays consistent and \
nothing important lives only in the conversation.\n\n\
For non-trivial work, create and maintain the canonical todo list with the \
provided todo tools. Treat it as a live model of the work, not a promise made \
once: revise it when evidence changes scope or sequencing, preserve stable item \
IDs where the work is the same, and mark an item completed only after its \
outcome has been verified. Keep the list aligned with actual progress; do not \
claim completion merely because an action was attempted.\n\n\
Be succinct, but full: communicate as briefly as possible while staying \
complete. State every fact that matters and omit everything that does not. No \
filler, no hedging, no restating the question back \u{2014} just the substance.\n\n\
Your messages are rendered as Markdown, so use it to make answers clear: \
headings, bold/italic, bullet and numbered lists, tables, links, and fenced \
code blocks with a language tag for code. Keep formatting purposeful \u{2014} it \
should aid readability, not add noise.\n\n\
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
    /// Whether to expose the `orchestrate` tool in the chat tool registry, so
    /// the chat agent can launch the orchestration pipeline itself mid-turn.
    pub enable_orchestrate_tool: bool,
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
    /// The default store uses `<temp>/rust-acp-agent/sessions`. It lets a
    /// session survive a process restart. Set this field to `None` to use the
    /// non-durable [`InMemorySessionStore`](agent_core::InMemorySessionStore).
    pub session_persistence_dir: Option<PathBuf>,
    /// Whether to start the LLM-inspector HTTP server at process startup.
    ///
    /// When enabled (the default), the binary launches a small web UI that shows
    /// the real requests sent to the LLM, grouped by session. Disable it with
    /// `ACP_INSPECTOR=0`.
    pub inspector_enabled: bool,
    /// Address the inspector HTTP server binds to (see [`inspector_enabled`]).
    ///
    /// Defaults to `127.0.0.1:7878`; override via `ACP_INSPECTOR_ADDR`.
    ///
    /// [`inspector_enabled`]: AgentConfig::inspector_enabled
    pub inspector_addr: String,
    /// Directory under which the inspector writes per-session JSONL logs of
    /// every captured LLM request (typed + provider).
    ///
    /// When `Some`, the inspector attaches a file recorder so requests are
    /// durably logged to `<dir>/<session_id>.jsonl` in addition to the
    /// in-memory UI buffer. Defaults to `<temp>/rust-acp-agent/llm-logs`;
    /// override via `ACP_INSPECTOR_LOG_DIR`, or set it to `off`/empty to
    /// disable file logging.
    pub inspector_log_dir: Option<PathBuf>,
}

/// The default directory for the inspector's per-session JSONL request logs:
/// `<system temp dir>/rust-acp-agent/llm-logs`.
#[must_use]
pub fn default_inspector_log_dir() -> PathBuf {
    std::env::temp_dir().join("rust-acp-agent").join("llm-logs")
}

fn default_session_persistence_dir() -> PathBuf {
    std::env::temp_dir().join("rust-acp-agent").join("sessions")
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
            default_model: "claude-haiku-4-5".to_string(),
            require_submit_result: true,
            thought_interval: 5,
            session_persistence_dir: Some(default_session_persistence_dir()),
            inspector_enabled: true,
            inspector_addr: "127.0.0.1:7878".to_string(),
            inspector_log_dir: Some(default_inspector_log_dir()),
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
    /// - `ANTHROPIC_MODEL` / `DEEPSEEK_MODEL` / `ACP_MODEL` override the default
    ///   model (`ACP_MODEL` wins when several are set).
    /// - `ACP_REQUIRE_SUBMIT_RESULT` (`1`/`true`/`yes`/`on`) toggles whether a
    ///   turn only ends when the model calls the `submit_result` tool.
    /// - `ACP_SESSION_DIR` selects the session directory. Use `off` to keep
    ///   sessions in memory only.
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
        // Model override precedence: ACP_MODEL > DEEPSEEK_MODEL > ANTHROPIC_MODEL.
        if let Ok(model) = std::env::var("ANTHROPIC_MODEL") {
            if !model.is_empty() {
                config.default_model = model;
            }
        }
        if let Ok(model) = std::env::var("DEEPSEEK_MODEL") {
            if !model.is_empty() {
                config.default_model = model;
            }
        }
        if let Ok(model) = std::env::var("ACP_MODEL") {
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
            let trimmed = dir.trim();
            config.session_persistence_dir =
                if trimmed.is_empty() || matches!(trimmed, "0" | "off" | "no" | "false") {
                    None
                } else {
                    Some(PathBuf::from(trimmed))
                };
        }
        if let Ok(v) = std::env::var("ACP_INSPECTOR") {
            config.inspector_enabled = !matches!(v.trim(), "0" | "false" | "no" | "off");
        }
        if let Ok(addr) = std::env::var("ACP_INSPECTOR_ADDR") {
            if !addr.trim().is_empty() {
                config.inspector_addr = addr.trim().to_string();
            }
        }
        if let Ok(dir) = std::env::var("ACP_INSPECTOR_LOG_DIR") {
            let trimmed = dir.trim();
            config.inspector_log_dir =
                if trimmed.is_empty() || matches!(trimmed, "0" | "off" | "no" | "false") {
                    None
                } else {
                    Some(PathBuf::from(trimmed))
                };
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
        assert!(DEFAULT_SYSTEM_PROMPT.contains("canonical todo list"));
        assert!(DEFAULT_SYSTEM_PROMPT.contains("revise it when evidence changes"));
        assert!(DEFAULT_SYSTEM_PROMPT.contains("completed only after"));
        assert!(DEFAULT_SYSTEM_PROMPT.contains("verified"));
    }

    #[test]
    fn default_model_is_haiku() {
        let config = AgentConfig::default();
        assert_eq!(config.default_model, "claude-haiku-4-5");
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
    fn default_persists_sessions_under_system_temp() {
        let config = AgentConfig::default();
        assert_eq!(
            config.session_persistence_dir.as_deref(),
            Some(default_session_persistence_dir().as_path())
        );
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
