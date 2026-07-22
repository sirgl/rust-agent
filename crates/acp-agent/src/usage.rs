//! The built-in `/usage` slash command: report cumulative token usage and an
//! estimated dollar cost for a session.
//!
//! This mirrors the CLI REPL's `render_usage`, but formats the summary as a
//! single string so the ACP agent can stream it back to the client as an
//! `AgentMessageChunk` without running a model turn.

use agent_core::SessionState;

use crate::model_spec;

/// The slash command (typed by the user in a prompt) that triggers a usage
/// report instead of a normal turn.
pub const USAGE_COMMAND: &str = "/usage";

/// Format a human-readable cumulative token-usage summary for a session,
/// including reused (cached) tokens and an estimated cost derived from the
/// model's pricing. Returns `None` pricing text when the model is unknown.
pub fn format_usage(session: &SessionState, model: &str) -> String {
    let u = &session.usage;
    let mut out = String::new();
    out.push_str(&format!("Token usage this session (model: {model}):\n"));
    out.push_str(&format!("  input (fresh):    {}\n", u.input_tokens));
    out.push_str(&format!(
        "  cached (reused):  {}\n",
        u.cache_read_input_tokens
    ));
    out.push_str(&format!(
        "  cache write:      {}\n",
        u.cache_creation_input_tokens
    ));
    out.push_str(&format!("  output:           {}\n", u.output_tokens));
    out.push_str(&format!("  total input:      {}\n", u.total_input_tokens()));
    match model_spec(model) {
        Some(spec) => {
            let dollars = ((u.input_tokens as f64) * spec.input_price_per_mtok
                + (u.cache_read_input_tokens as f64) * spec.cache_read_price_per_mtok()
                + (u.cache_creation_input_tokens as f64) * spec.cache_write_price_per_mtok()
                + (u.output_tokens as f64) * spec.output_price_per_mtok)
                / 1_000_000.0;
            out.push_str(&format!("  estimated cost:   ${dollars:.4}"));
        }
        None => out.push_str("  estimated cost:   n/a (unknown model pricing)"),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::TokenUsage;

    #[test]
    fn format_usage_includes_totals_and_cost() {
        let mut session = SessionState::new("s1");
        session.usage = TokenUsage {
            input_tokens: 1000,
            cache_read_input_tokens: 500,
            cache_creation_input_tokens: 200,
            output_tokens: 300,
        };
        let text = format_usage(&session, "claude-sonnet-5");
        assert!(text.contains("Token usage this session"));
        assert!(text.contains("cached (reused):  500"));
        // Known model -> a dollar figure rather than the n/a note.
        assert!(text.contains("estimated cost:   $"));
    }

    #[test]
    fn format_usage_reports_unknown_pricing() {
        let session = SessionState::new("s1");
        let text = format_usage(&session, "totally-unknown-model");
        assert!(text.contains("n/a (unknown model pricing)"));
    }
}
