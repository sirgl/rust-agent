# MCP client & subagents (Step 6)

This note records the final constraints, limitations, and accepted decisions for
the `mcp-client` and `subagents` capability crates and their wiring into
`acp-agent`.

## Accepted decisions

- **Provider-independent, written once.** Both crates are services-layer
  capabilities implementing `agent_core::Tool`. They depend only on the
  `agent-core` trait seams — never on a decision-layer provider. Which
  `NextTurnService` a subagent uses is injected, not hard-coded.
- **MCP transport behind a trait seam.** The wire transport to an MCP server is
  abstracted by `mcp_client::McpConnection` (`server_name` / `list_tools` /
  `call_tool`). Tool-wrapping, namespacing, and registry logic depend only on
  this trait, so they are fully testable against an in-memory stub with no child
  process or socket. A concrete stdio/JSON-RPC transport can implement the trait
  and plug in unchanged.
- **Tool namespacing.** MCP tools are registered as `mcp__<server>__<tool>` so
  tools from different servers (or colliding with a built-in) cannot clash in
  the shared `ToolRegistry`. The unqualified name is what is forwarded to the
  server on `call_tool`.
- **Recursion depth limit for subagents.** `TurnEngine` propagates its `depth`
  into every dispatched `ToolContext`; a spawned nested engine runs at
  `ctx.depth + 1`. `SubagentTool` refuses delegation when
  `ctx.depth >= max_depth`, returning a graceful failed tool result rather than
  recursing without bound. Default limit: `subagents::DEFAULT_MAX_DEPTH` (2).
- **Config wiring.** `AgentConfig` gains `enable_subagents`,
  `subagent_max_depth`, and `mcp_servers`. `acp_agent::build_registry` assembles
  the built-in tools plus (when enabled) the subagent tool.
  `acp_agent::register_configured_mcp_tools` registers a connected server's
  tools; it is async/fallible and kept out of the synchronous registry assembly.
- **Real MCP stdio transport, wired to `session/new`.** `mcp_client::stdio::StdioMcpConnection`
  implements `McpConnection` over a genuine child process: it launches the
  configured command, performs the MCP `initialize` handshake, and serves
  `list_tools`/`call_tool` via `tools/list`/`tools/call` (newline-delimited
  JSON-RPC over stdin/stdout). `acp-agent`'s `session/new` handler consumes the
  ACP-native `NewSessionRequest.mcp_servers` field: each `McpServer::Stdio`
  entry is connected this way and its tools registered into a *per-session*
  copy of the tool registry (base tools + that session's MCP tools), which is
  what `session/prompt` actually dispatches against. A server that fails to
  connect or advertises no tools is logged and skipped rather than failing
  `session/new`.
- **Default Anthropic API key at startup.** `turn_anthropic::AnthropicConfig::from_env`
  first checks `ANTHROPIC_API_KEY`; if unset/empty it falls back to
  `turn_anthropic::load_default_api_key`, which reads a `KEY=...` property from
  a `token.properties` file (checked, in order: `ACP_TOKEN_PROPERTIES_PATH` env
  override, the workspace root derived from this crate's `CARGO_MANIFEST_DIR`,
  then the literal development path). `acp_agent::default_deps` (used by the
  binary's `main.rs`) prefers the Anthropic backend when a key resolves from
  either source, otherwise falls back to the replay backend. The key is never
  logged or printed; only the source path is traced at `debug` level, and a
  missing/unreadable file falls back cleanly (`None`) with no error.

## Limitations (v1)

- **Only the stdio MCP transport is implemented.** The ACP `McpServer` schema
  also defines `http`, `sse`, and (behind the `unstable_mcp_over_acp` cargo
  feature, not enabled here) `acp` transports. `session/new`'s
  `connect_session_mcp_servers` recognizes those variants but only logs a
  warning and skips them (`McpServer` is `#[non_exhaustive]`, so a catch-all
  arm also covers any future variant) -- it does not fail the session. Adding a
  concrete HTTP/SSE client is future work; the trait seam (`McpConnection`)
  already supports plugging one in unchanged.
- **MCP servers are connected per-session, not shared/pooled.** Every
  `session/new` call that carries `mcp_servers` spawns fresh child processes for
  each `stdio` entry; there is no cross-session process reuse or connection
  pooling in v1.
- **Subagent progress is aggregated, not live.** The nested turn is driven to a
  stop reason while a capturing sink records assistant-visible text; that text is
  then streamed back to the parent as `ToolEvent::OutputDelta` followed by
  `Completed`. Ordering is preserved but deltas are not forwarded token-by-token
  across the tool boundary.
- **Nested tool permissions.** Delegated turns grant tool permissions by default
  (the capturing sink uses the default grant). Interactive permission
  forwarding for subagent tool calls is out of scope for v1.
- **MCP permission gating.** An MCP server may mark a tool
  `requires_permission`; the engine then gates it via
  `session/request_permission` exactly like a destructive built-in.

## Testing

- `mcp-client`: stub-server unit tests for discovery/namespacing, arg
  forwarding, success/error mapping, empty-server error, and schema/permission
  exposure. `stdio::tests` additionally drives the *real* `StdioMcpConnection`
  transport against a self-contained shell-script mock MCP server (a genuine
  child process speaking newline-delimited JSON-RPC), covering `initialize` +
  `tools/list` + `tools/call` and registration into a `ToolRegistry`.
- `subagents`: replay-backed unit tests for delegation + streamed output,
  depth-limit refusal, missing-argument handling, and depth propagation into a
  nested engine's inner tools.
- `acp-agent`: `tests/wiring.rs` asserts the subagent tool is present/absent per
  config and that MCP tools register through the public helper.
  `tests/mcp_session_wiring.rs` proves the full runtime path end-to-end: a
  `session/new` request carrying `mcp_servers: [McpServer::Stdio(...)]`
  (pointing at the same kind of real child-process mock server) results in the
  MCP tool being genuinely dispatchable from a subsequent `session/prompt`
  turn, and an unsupported transport (`McpServer::Http`) is skipped without
  failing session creation.
- `turn-anthropic`: `request::tests` covers `token.properties` parsing
  (comments, quoted values, missing/empty `KEY`, malformed lines, a real temp
  file, and a missing file falling back to `None`) without ever touching
  process-global environment variables, avoiding test-parallelism flakiness.
