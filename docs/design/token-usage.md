# Token usage: caching, output budget, large-output offloading, thinking

This note documents the token-efficiency features of the Anthropic backend and
the turn engine, and the accepted decisions behind them.

## 1. Prompt caching (maximize token reuse) — ON by default

`turn-anthropic`'s `build_body` inserts Anthropic `cache_control: {type:
ephemeral}` breakpoints so stable prefixes are reused across turns instead of
re-billed as fresh input tokens:

- **System prompt**: emitted as a structured text block with a cache
  breakpoint (so it is cached together with the tool definitions that precede
  it in Anthropic's cache hierarchy `tools -> system -> messages`).
- **Tools**: a single breakpoint on the *last* tool definition caches the whole
  (rarely-changing) tool block.
- **Conversation tail**: a breakpoint on the last content block of the last
  message caches the growing conversation prefix, so the next turn (which only
  appends new messages) is a cache hit up to that point.

Controlled by `AnthropicConfig.prompt_caching` (default `true`); disable with
`ANTHROPIC_PROMPT_CACHING=0`. When disabled, `system` is sent as a plain string
and no breakpoints are added.

## 2. Response token budget (`max_tokens`) — raised & configurable

Anthropic *requires* a `max_tokens` field, so it cannot be removed. The previous
default of `4096` truncated large generations (the model returned
`stop_reason: max_tokens` mid-answer). The default is now
`AnthropicConfig::DEFAULT_MAX_TOKENS = 32000` and can be overridden with
`ANTHROPIC_MAX_TOKENS`.

## 3. Large tool-output offloading — ON by default

Very large tool results (a big file read, a chatty terminal command, or a
verbose MCP tool) would otherwise flood the context and burn tokens. The turn
engine (`TurnEngine::maybe_offload_large_output`) applies uniformly to **every**
tool it dispatches — built-in, **MCP**, and subagents — because they all funnel
their results through the same path:

- If a tool result exceeds `MAX_TOOL_RESULT_CHARS` (8000 chars), the full output
  is written to a file (`<tmp>/acp-agent-tool-outputs/<session>-<toolcall>.txt`)
  via the client's `write_text_file` (which now creates parent directories).
- Only a head+tail **preview** is fed back to the model, annotated with the
  total size and the saved file path, plus an instruction to use `fs_read` to
  view the full output on demand.
- If no client filesystem is available, the result is still truncated (with a
  note) so the context stays bounded.

## 4. Extended thinking — ON by default

The full thinking pipeline is wired end-to-end for both live display and
faithful replay:

- **Live display**: `mapper -> TurnEvent::Thinking(delta) ->
  EngineOutput::ThinkingChunk -> ACP AgentThoughtChunk / CLI thinking line`.
- **Preservation**: on `content_block_stop` the mapper also emits the complete
  block — `TurnEvent::ThinkingBlock(ThinkingRecord)` — carrying the reasoning
  text **and its `signature`** (or the opaque `data` of a `redacted_thinking`
  block). The engine stores it verbatim in the typed history
  (`HistoryEntry::Thinking`), and the compiler renders it back as an Anthropic
  `thinking` / `redacted_thinking` content block.

`build_body` emits the right thinking dialect based on
`AnthropicConfig.thinking` (`ThinkingConfig`):

- **Adaptive** (default; Claude Sonnet 5 / Opus 4.7+): `thinking: {type:
  "adaptive", display: "summarized"}`. Newer models **reject** the legacy
  `{type: "enabled"}` form (HTTP 400), and default thinking display to
  *omitted*, so `display: "summarized"` is requested to surface visible
  reasoning. Depth is steered by `output_config.effort` (default high).
- **Budget** (legacy; Opus 4.6 and earlier): `thinking: {type: "enabled",
  budget_tokens: N}`, guaranteeing `max_tokens > budget_tokens`.
- **Disabled**: `thinking: {type: "disabled"}`.

**Default: on** (adaptive with visible summarized reasoning). Env overrides:
`ANTHROPIC_THINKING=0`/`false` disables; `ANTHROPIC_THINKING=1`/`true` forces
adaptive; `ANTHROPIC_THINKING_BUDGET=<n>` (or `ANTHROPIC_THINKING=<n>`) selects
legacy budget mode for older models.

### Why it is now safe with tool use

Anthropic requires that, when extended thinking is used together with tools, the
signed `thinking` blocks of the assistant's tool-use turn be replayed verbatim
(unmodified, and in the *same* assistant message as the `tool_use`, thinking
first) on the following request — otherwise the request is rejected. This is now
satisfied:

- The signed block is captured and stored (the crate-wide "preserve & reuse"
  rule) as `HistoryEntry::Thinking`.
- `build_body` **merges consecutive same-role messages** so the separately
  recorded thinking, assistant text, and tool-call entries fold back into one
  assistant message with the ordering `thinking -> text -> tool_use`.
- `redacted_thinking` blocks are preserved and echoed back unchanged too, so
  filtering never silently drops them and breaks the protocol.
