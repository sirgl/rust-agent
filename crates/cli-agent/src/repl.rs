//! The interactive REPL loop.
//!
//! [`run_repl`] reads user messages line-by-line from `input`, drives a
//! [`TurnEngine`] turn per message against a persistent [`SessionState`] (so the
//! chat has memory within a run), and streams the assistant's answer to `output`
//! via a [`TerminalUpdateSink`]. It reuses the exact same engine, decision
//! backend, and tool registry as the ACP server; only the client seam
//! ([`LocalClientAccess`](crate::LocalClientAccess)) and the presentation sink
//! differ.
//!
//! The loop is generic over any [`BufRead`]/[`Write`] pair so tests can feed a
//! scripted input and capture the streamed output. The binary passes locked
//! stdin/stdout.

use std::io::{BufRead, Write};
use std::sync::Arc;

use acp_agent::{AgentDeps, ModelSelection};
use agent_core::{
    CancellationToken, ClientAccess, SessionState, StopReason, ToolDescriptor, ToolRegistry,
    TurnEngine,
};

use crate::sink::TerminalUpdateSink;

/// Options controlling REPL behavior.
#[derive(Debug, Clone, Copy)]
pub struct ReplOptions {
    /// Whether destructive tools prompt for `[y/N]` permission on stdin.
    pub interactive_permissions: bool,
    /// Whether to install a `Ctrl-C` handler that cancels the in-flight turn.
    pub handle_ctrl_c: bool,
}

impl Default for ReplOptions {
    fn default() -> Self {
        Self {
            interactive_permissions: true,
            handle_ctrl_c: true,
        }
    }
}

/// Build the [`ToolDescriptor`]s advertised to the decision layer for a session.
fn tool_descriptors(tools: &ToolRegistry) -> Vec<ToolDescriptor> {
    tools
        .names()
        .filter_map(|name| tools.get(name).map(|tool| (name, tool)))
        .map(|(name, tool)| ToolDescriptor {
            name: name.to_string(),
            schema: tool.schema(),
            requires_permission: tool.requires_permission(),
        })
        .collect()
}

/// Run the interactive chat loop until end-of-input or an `exit`/`quit` command.
///
/// # Errors
///
/// Returns an error only if writing to `output` fails; per-turn engine errors
/// are reported inline and do not abort the loop.
pub async fn run_repl<R, W>(
    deps: AgentDeps,
    client: Arc<dyn ClientAccess>,
    mut input: R,
    mut output: W,
    options: ReplOptions,
) -> std::io::Result<()>
where
    R: BufRead + Send,
    W: Write + Send,
{
    let mut session = SessionState::new("cli-session");
    session.system_prompt = deps.config.system_prompt.clone();
    session.available_tools = tool_descriptors(&deps.tools);

    writeln!(
        output,
        "cli-agent interactive chat. Type a message and press Enter. Ctrl-D or 'exit' to quit."
    )?;

    loop {
        write!(output, "\n> ")?;
        output.flush()?;

        let mut line = String::new();
        let read = input.read_line(&mut line)?;
        if read == 0 {
            // EOF (Ctrl-D): leave the loop cleanly.
            writeln!(output, "\nbye")?;
            break;
        }
        let message = line.trim();
        if message.is_empty() {
            continue;
        }
        if matches!(message, "exit" | "quit") {
            writeln!(output, "bye")?;
            break;
        }
        if let Some(command) = acp_agent::parse_command(message) {
            let text = acp_agent::run_command(command, &session, &deps.config.default_model);
            writeln!(output, "\n{text}")?;
            continue;
        }

        session.push_user_text(message);

        let cancel = CancellationToken::new();
        let selection = ModelSelection::for_config(&deps.config);
        let next_turn = (deps.next_turn_factory)(&selection);
        let engine = TurnEngine::new(next_turn, deps.tools.clone()).with_client(client.clone());

        // Scope the sink so its mutable borrow of `output` is released before we
        // write the trailing stop-reason note below.
        let stop = {
            let mut sink = TerminalUpdateSink::new(&mut output)
                .interactive(options.interactive_permissions);
            run_turn(&engine, &mut session, &mut sink, &cancel, options.handle_ctrl_c).await
        };

        match stop {
            Ok(reason) => render_stop_reason(&mut output, reason)?,
            Err(err) => writeln!(output, "\n[error] {err}")?,
        }
    }

    Ok(())
}

/// Drive a single turn, optionally racing it against `Ctrl-C` so an interrupt
/// cancels the turn (via the shared token) without ending the process.
async fn run_turn<W: Write + Send>(
    engine: &TurnEngine,
    session: &mut SessionState,
    sink: &mut TerminalUpdateSink<W>,
    cancel: &CancellationToken,
    handle_ctrl_c: bool,
) -> agent_core::Result<StopReason> {
    if !handle_ctrl_c {
        return engine.run_prompt(session, sink, cancel).await;
    }

    tokio::select! {
        result = engine.run_prompt(session, sink, cancel) => result,
        _ = tokio::signal::ctrl_c() => {
            // Trip the token; the engine observes it and returns promptly.
            cancel.cancel();
            engine.run_prompt(session, sink, cancel).await
        }
    }
}

/// Print a short trailing note for how the turn ended.
fn render_stop_reason<W: Write>(output: &mut W, reason: StopReason) -> std::io::Result<()> {
    match reason {
        StopReason::EndTurn | StopReason::ToolUse => writeln!(output),
        StopReason::Cancelled => writeln!(output, "\n[cancelled]"),
        StopReason::MaxTokens => writeln!(output, "\n[stopped: max tokens]"),
        StopReason::Refusal => writeln!(output, "\n[refused]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use agent_core::ToolRegistry;

    fn replay_deps() -> AgentDeps {
        AgentDeps {
            next_turn_factory: acp_agent::canned_replay_factory(),
            tools: ToolRegistry::new(),
            config: acp_agent::AgentConfig::default(),
            store: acp_agent::default_store(),
        }
    }

    #[tokio::test]
    async fn streams_replayed_answer_and_exits() {
        let deps = replay_deps();
        let client: Arc<dyn ClientAccess> = Arc::new(crate::LocalClientAccess::new());
        let input = Cursor::new(b"hello\nexit\n".to_vec());
        let mut output: Vec<u8> = Vec::new();

        run_repl(
            deps,
            client,
            input,
            &mut output,
            ReplOptions {
                interactive_permissions: false,
                handle_ctrl_c: false,
            },
        )
        .await
        .expect("repl should complete");

        let text = String::from_utf8(output).expect("utf8");
        assert!(text.contains("Hello from the replay backend!"));
        assert!(text.contains("without any network"));
        assert!(text.contains("bye"));
    }

    #[tokio::test]
    async fn usage_command_prints_totals_and_cost() {
        let deps = replay_deps();
        let client: Arc<dyn ClientAccess> = Arc::new(crate::LocalClientAccess::new());
        // Ask for usage before any turn, then exit.
        let input = Cursor::new(b"/usage\nexit\n".to_vec());
        let mut output: Vec<u8> = Vec::new();

        run_repl(
            deps,
            client,
            input,
            &mut output,
            ReplOptions {
                interactive_permissions: false,
                handle_ctrl_c: false,
            },
        )
        .await
        .expect("repl should complete");

        let text = String::from_utf8(output).expect("utf8");
        assert!(text.contains("Token usage this session"));
        assert!(text.contains("cached (reused):"));
        // Default model (claude-sonnet-5) has known pricing -> a dollar figure.
        assert!(text.contains("estimated cost:   $"));
        // No turn ran, so the replay answer must not appear.
        assert!(!text.contains("replay backend"));
    }

    #[tokio::test]
    async fn empty_lines_are_skipped_and_eof_exits() {
        let deps = replay_deps();
        let client: Arc<dyn ClientAccess> = Arc::new(crate::LocalClientAccess::new());
        // Blank line then EOF (no explicit exit) should still terminate.
        let input = Cursor::new(b"\n".to_vec());
        let mut output: Vec<u8> = Vec::new();

        run_repl(
            deps,
            client,
            input,
            &mut output,
            ReplOptions {
                interactive_permissions: false,
                handle_ctrl_c: false,
            },
        )
        .await
        .expect("repl should complete");

        let text = String::from_utf8(output).expect("utf8");
        // No turn ran, but the loop ended on EOF.
        assert!(text.contains("bye"));
        assert!(!text.contains("replay backend"));
    }
}
