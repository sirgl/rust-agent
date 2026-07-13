# CLI agent: interactive terminal chat

The `cli-agent` crate is a thin, line-based REPL front-end over the same turn
machinery used by the ACP stdio server (`TurnEngine`, `NextTurnService`,
`ToolRegistry`). Only the two seams differ from `acp-agent`:

- **`LocalClientAccess`** (`local_client.rs`) — implements `agent_core::ClientAccess`
  against the real machine: `read_text_file`/`write_text_file` via `std::fs`, and
  `run_terminal` via `std::process::Command` (run on a blocking thread), mapped
  into `TerminalOutcome` (combined stdout+stderr, byte-bounded truncation, exit
  code, and Unix terminating signal). This lets the built-in fs/terminal tools
  work with no editor behind the agent.
- **`TerminalUpdateSink`** (`sink.rs`) — implements `agent_core::UpdateSink`,
  streaming `MessageChunk`/`ThinkingChunk` to the writer with flushing, rendering
  `Plan` and `ToolCall`/`ToolCallUpdate` as concise status lines, and prompting
  `[y/N]` on stdin before gated (destructive) tools when interactive.

## Behavior

- **Conversation continuity** — a single `SessionState` persists across REPL
  iterations, so the chat has memory within a run.
- **Backend selection** — `main.rs` reuses `acp_agent::default_deps()`, so the CLI
  uses the Anthropic backend when an API key resolves (env `ANTHROPIC_API_KEY` or
  the default `token.properties` file) and otherwise falls back to the
  deterministic replay backend.
- **Cancellation** — each turn runs under a fresh `CancellationToken`; `Ctrl-C`
  trips the token (the turn is cancelled, the REPL survives) via a
  `tokio::select!` race with `tokio::signal::ctrl_c()`.
- **Exit** — `Ctrl-D` (EOF), `exit`, or `quit` leave the loop cleanly.
- **Logging** — `tracing` is routed to stderr so logs never corrupt the chat
  stream on stdout.

## Launch

```
cargo run -p cli-agent
```

Then type a message and press Enter; the assistant's answer streams back inline.
Set `RUST_LOG=debug` (captured on stderr) for verbose tracing.
