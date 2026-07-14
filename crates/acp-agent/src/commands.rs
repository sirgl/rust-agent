//! A small dispatcher for the agent's built-in slash commands.
//!
//! Commands are recognized from raw prompt text (as delivered by an ACP client
//! naming an advertised command, or typed directly in the CLI REPL) and mapped
//! to an action that produces display text — without running a model turn.
//!
//! Keeping this in one place lets both the ACP prompt path and the CLI REPL
//! share a single source of truth for command parsing and rendering, and gives
//! us a seam to grow additional commands later.

use agent_core::SessionState;

use crate::usage::{format_usage, USAGE_COMMAND};

/// A recognized built-in slash command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentCommand {
    /// Report cumulative token usage and an estimated cost for the session.
    Usage,
}

/// Parse `text` (a raw user prompt) into a built-in [`AgentCommand`], if it
/// names one. Matching mirrors the historical behavior: the trimmed text must
/// equal the command string exactly.
#[must_use]
pub fn parse_command(text: &str) -> Option<AgentCommand> {
    match text.trim() {
        USAGE_COMMAND => Some(AgentCommand::Usage),
        _ => None,
    }
}

/// Run a recognized [`AgentCommand`] against the session, returning the display
/// text to surface to the client. This never runs a model turn.
#[must_use]
pub fn run_command(command: AgentCommand, session: &SessionState, model: &str) -> String {
    match command {
        AgentCommand::Usage => format_usage(session, model),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_usage_command_and_ignores_others() {
        assert_eq!(parse_command("/usage"), Some(AgentCommand::Usage));
        assert_eq!(parse_command("  /usage  "), Some(AgentCommand::Usage));
        assert_eq!(parse_command("hello"), None);
        assert_eq!(parse_command("/usage extra"), None);
    }

    #[test]
    fn run_command_matches_format_usage() {
        let session = SessionState::new("s1");
        let expected = format_usage(&session, "claude-sonnet-5");
        let actual = run_command(AgentCommand::Usage, &session, "claude-sonnet-5");
        assert_eq!(actual, expected);
    }
}
