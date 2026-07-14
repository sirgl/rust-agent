# Design note: the compiler/dialect seam and typed history

This note records the architectural decisions around how typed conversation
history is turned into provider requests, and how the `TurnEngine` turn loop
is validated. It exists so later steps (the Anthropic backend, MCP, subagents)
build on the same agreed shape instead of re-litigating it.

## Constraints that drove this design

- The turn engine (`agent-core::TurnEngine`) must stay provider-agnostic: it
  drives the loop against the `NextTurnService` seam only, never against a
  specific LLM SDK.
- Conversation history must be strongly typed, not raw provider JSON, so the
  rest of the workspace (tools, UI, tests) can reason about it without
  string-matching. See `agent-core::history::Conversation`.
- Whatever converts typed history into a provider request must not force a
  one-size intermediate representation (IR) if it isn't justified — different
  providers shape messages differently (e.g. Anthropic's interleaved `content`
  blocks vs. a hypothetical provider with separate tool-call channels).
- Regardless of shape, the conversion must preserve exact incoming content and
  **reuse** previously-compiled/ingested forms rather than lossily re-rendering
  them from the lightweight typed model, whenever a message or tool
  interaction already arrived from an upstream provider/client representation.

## Decision: interface-first compiler/dialect seam, no mandatory IR

`agent-core::compiler::LlmCompiler` is a trait, not a fixed pipeline stage:

```rust
pub trait LlmCompiler: Send + Sync {
    fn compile(&self, ctx: &TurnContext) -> Result<LlmRequest>;
}
```

Each provider backend supplies its own compiler and decides how to render
`ctx.history` (a `Conversation`) into whatever request shape it needs. There is
**no mandatory intermediate representation** between typed history and a
provider's wire format: a dialect/provider adapter can go straight from typed
history to its own request type.

`LlmRequest` / `LlmMessage` / `LlmContentBlock` (role + content blocks) exist in
`agent-core::compiler` as an **optional, default provider-neutral shape** for
compilers whose target wire format is a natural fit for it (used today by
`DefaultCompiler`, and expected to be a good starting point for
`turn-anthropic`). A future provider is free to ignore these types entirely and
produce its own request struct — the seam is the `LlmCompiler` trait, not
these types.

We deliberately did **not** introduce a mandatory IR layer (e.g. a shared
`CompiledUnit`/`CompiledProgram` staged between history and every provider
request). Rationale: with a single concrete backend (Anthropic) landing next,
a forced IR would be speculative generality — it tends to (a) leak
provider-specific concerns back into `agent-core`, or (b) lose fidelity in an
extra translation hop. If a second real provider backend later duplicates
substantial translation logic against `LlmRequest`, introducing a shared IR at
that point is straightforward without breaking the `LlmCompiler` trait itself.

## Hard requirement: preserve and reuse, don't lossily re-render

Typed history supports faithfully preserving content that already went through
a provider/client representation once:

- `HistoryEntry::ToolCall` stores a typed `KnownTool`; `KnownTool::arguments()`
  reconstructs the exact JSON arguments from the typed record instead of
  keeping a separate, possibly-drifting copy.
- `HistoryEntry::Ingested(IngestedRecord { source, payload })` preserves an
  exact, previously-compiled fragment (e.g. a provider message that arrived
  from an upstream/resumed session) verbatim. A compiler MUST check for this
  variant and splice the preserved `payload` back in unchanged rather than
  re-deriving a message from the lightweight typed fields.

`DefaultCompiler::compile` is the reference implementation of this rule: for
`HistoryEntry::Ingested`, it first tries to deserialize the payload as an
`LlmMessage` and reuses it verbatim; only if that fails (the payload isn't in
that shape) does it fall back to a best-effort text rendering, so no
information is silently dropped. See `crates/agent-core/src/compiler.rs`.

## Turn loop and cancellation

`TurnEngine::run_prompt` (Step 2) drives the loop against `NextTurnService` and
a `ToolRegistry`:

1. Build a `TurnContext` from typed history only (`SessionState::turn_context`)
   — never from raw provider messages.
2. Stream decision events, forwarding `TextDelta`/`Thinking`/`Plan` to the
   `UpdateSink` as they arrive.
3. On `ToolCallRequested`, flush any accumulated assistant text to history,
   dispatch the tool (recording a typed `ToolCall`/`ToolResult` pair in
   history regardless of outcome), and continue the same decision stream.
4. On `TurnFinished`, loop for another decision round only if a tool call was
   actually dispatched and the reason is `ToolUse`; otherwise return the stop
   reason.

Cancellation uses a single `agent_core::CancellationToken` shared by the whole
prompt. Both the decision-stream loop and the tool-dispatch loop race
`cancel.cancelled()` against the next stream item (`tokio::select! { biased; ...
}`, cancellation checked first). Once cancellation is observed:

- the loop returns `StopReason::Cancelled` immediately,
- no further `UpdateSink` output is emitted,
- any assistant text accumulated in the current decision round is **not**
  flushed to history (the cancel branch returns before the flush call).

## Anthropic backend: request build and SSE mapping (Step 4)

`turn-anthropic::AnthropicTurnService` is the first real `NextTurnService`. It
uses the `DefaultCompiler` (not a bespoke one): `agent-core`'s `LlmContentBlock`
serde tags (`text` / `tool_use` / `tool_result`) and role names already match
the Anthropic block/role vocabulary 1:1, so `request::build_body` is a
structural pass over `LlmRequest` — it never re-renders from the lightweight
typed history, and any `Ingested` message the compiler spliced in verbatim
flows through unchanged (the preserve-and-reuse rule is honored transitively).

Accepted request-shape decisions (`crates/turn-anthropic/src/request.rs`):

- `system` is a top-level field, kept out of `messages`. `LlmRequest.system`
  plus any stray `System`-role message are concatenated with `\n\n`.
- `stream: true` always; `model`/`max_tokens`/`base_url`/`anthropic-version`
  come from `AnthropicConfig` (env: `ANTHROPIC_API_KEY` required, `ANTHROPIC_MODEL`
  and `ANTHROPIC_BASE_URL` optional). Auth via `x-api-key` header.
- `tools` are emitted as `{name, input_schema}` and omitted entirely when empty.

Accepted SSE parsing/mapping decisions and invariants
(`crates/turn-anthropic/src/{sse,mapper}.rs`):

- **Framing is separate from mapping.** `SseDecoder` is a pure, synchronous,
  byte-chunk-driven framer: it buffers partial lines across arbitrary network
  chunk boundaries, tolerates CRLF, ignores comment lines and every non-`data:`
  field, concatenates multiple `data:` lines with `\n`, and emits one payload
  per blank-line-terminated event. `StreamMapper` consumes the decoded JSON.
- **Event mapping:** `text_delta` -> `TextDelta`, `thinking_delta` -> `Thinking`,
  `input_json_delta` -> accumulated (see below), `signature_delta`/`ping`/
  `message_start` -> ignored, `error` -> `TurnEvent::Error`, `message_stop` ->
  `TurnFinished`.
- **Tool-use JSON is assembled, never parsed incrementally.** `partial_json`
  fragments are concatenated verbatim per content-block `index` (fragments may
  split a single JSON token) and parsed exactly once at `content_block_stop`.
  Empty accumulation -> `{}` (Anthropic omits deltas for zero-arg tools); a
  parse failure -> `{}` with a `tracing::warn!` rather than dropping the call or
  panicking, keeping the turn loop live so the tool layer can surface an error.
- **Interleaved blocks** are tracked independently by `index`, so concurrent
  text and tool-use blocks cannot corrupt each other's accumulation.
- **`stop_reason` mapping:** `end_turn`/`stop_sequence` -> `EndTurn`,
  `max_tokens` -> `MaxTokens`, `tool_use` -> `ToolUse`, `refusal` -> `Refusal`,
  anything else -> `EndTurn` (safe "turn is over").
- **Termination guarantee:** transport/HTTP-status/stream errors are emitted as
  a terminal `TurnEvent::Error`, and a `TurnFinished` is always yielded even if
  the byte stream ends before `message_stop`, so the engine loop never hangs.

Validation: pure unit tests in each module plus `tests/sse_fixture.rs`, which
runs a recorded raw SSE stream (text + split tool-use JSON + `tool_use` stop)
through `SseDecoder` + `StreamMapper` at every chunk size (1..256 bytes) and
asserts an identical `TurnEvent` sequence. A live end-to-end test
(`tests/live.rs`) is `#[ignore]`d and gated on `ANTHROPIC_API_KEY`.

## Decision: services are provider-independent; only dialects are per-provider

This is the load-bearing separation of the whole design and is **not**
negotiable per provider:

- **Services (the "what the agent can do" layer) are global.** Tools, MCP
  clients, subagents and their access to the ACP client (filesystem, terminal)
  are written **once** in the services layer and shared by every decision-layer
  provider. A provider **never** re-implements a service. Adding Anthropic, a
  replay backend, or a future provider does not add or fork a single tool.
- **Only the dialect/compiler path is per-provider.** The provider-specific
  code is confined to (a) the `NextTurnService` implementation that produces
  `TurnEvent`s and (b) the `LlmCompiler` that renders typed history into that
  provider's request shape and ingests its responses. That is the *entire*
  provider surface.

Concretely, the services layer depends only on `agent-core` trait seams:

- `agent-core::Tool` / `ToolRegistry` — the callable-capability seam. Built-in
  tools live in `tools-builtin` and are registered once
  (`tools_builtin::builtin_registry()`); the engine dispatches them regardless
  of which `NextTurnService` produced the `ToolCallRequested` event.
- `agent-core::client::ClientAccess` — a provider-independent abstraction over
  filesystem and terminal capabilities (`read_text_file`, `write_text_file`,
  `run_terminal`) plus interactive `request_elicitation`. Tools depend on this
  trait, never on the concrete `agent-client-protocol` client. The ACP binary
  supplies `acp-agent::LocalClientAccess`, which performs filesystem/terminal
  work **directly on the local machine** (`std::fs` / `std::process::Command`)
  rather than routing it through ACP `fs/*` and `terminal/*` JSON-RPC; only
  elicitation (which has no local equivalent) is still delegated to the
  connected client. The CLI uses the same local-access shape. Tests supply an
  in-memory fake, keeping the services layer testable without a network or a
  real editor.

### Permission modes (Normal / YOLO)

The permission behavior is a per-session ACP config option (`permissions`,
alongside `model`/`effort`):

- **Normal** (default) — every permission-gated (destructive) tool call triggers
  a blocking `session/request_permission` round-trip to the client.
- **YOLO** — `AcpUpdateSink` auto-grants permission-gated tool calls without
  prompting, so the turn never blocks on a client permission response.

The selected mode lives on the session's `ModelSelection` (`PermissionMode`) and
is threaded into `AcpUpdateSink::with_yolo` for each turn.

If any code, spec, or doc ever implies "provider X implements tool Y" or
"provider X needs its own filesystem/terminal code", that is a bug against this
decision: the tool/service is global, and only the request/response rendering
for provider X is provider-specific.

## Built-in tools and the permission-flow contract (Step 5)

`tools-builtin` provides three global tools, each depending only on
`ClientAccess`:

- `fs_read` (`FsReadTool`) — reads a text file. Non-destructive, so
  `requires_permission() == false`.
- `fs_write` (`FsWriteTool`) — writes a text file. Destructive, so
  `requires_permission() == true`.
- `terminal_run` (`TerminalRunTool`) — runs a command (create terminal → wait
  for exit → collect output → release). Destructive, so
  `requires_permission() == true`.

Permission is enforced by the **engine**, not by the tool, and strictly before
any execution begins (`crates/agent-core/src/engine.rs::dispatch_tool`):

1. The tool call is recorded in typed history (`KnownTool::classify`), and a
   `ToolCall { status: Pending }` update is emitted.
2. Unknown tool name → a failed `tool_call_update` and a failed typed
   `ToolResult`; the loop continues (no panic).
3. If `tool.requires_permission()`, the engine calls
   `UpdateSink::request_permission`. On **deny**, it emits a `Failed`
   `tool_call_update`, records a failed `ToolResult`, and returns **without ever
   calling `Tool::call`** — the tool must not observe a denied call. The turn
   loop continues to the next decision round.
4. On **grant** (or for non-permissioned tools), the engine emits
   `InProgress`, calls `Tool::call`, streams `ToolEvent`s as
   `tool_call_update`s, and finishes with `Completed`/`Failed`.
5. Malformed arguments: tools validate up front and return
   `AgentError::InvalidArguments`; a `Tool::call` error becomes a `Failed`
   update + failed `ToolResult` (loop not deadlocked).

Status transitions surfaced to the client are always
`pending → in_progress → completed | failed` (or `pending → failed` when
permission is denied, an argument is malformed, or the tool is unknown — the
`in_progress` step is skipped in those cases).

### Typed history for the new tools

`KnownTool` gained an explicit `FsWrite(FsWriteToolCall { path, content })`
variant alongside the existing `Edit`, `FsRead`, and `TerminalRun` variants, so
built-in tool calls are represented in typed form (not opaque JSON) and
`KnownTool::arguments()` reconstructs the exact input for the compiler
(preserve-and-reuse). Anything else remains `KnownTool::Other`, preserving raw
arguments verbatim.

## Testing approach

The full loop is validated deterministically via `turn-replay::ReplayTurnService`
(a `NextTurnService` that hands out scripted `Vec<TurnEvent>` "rounds", one per
call, in order) plus an in-memory fake `UpdateSink` — see
`crates/agent-core/tests/turn_engine_replay.rs`. Covered scenarios: plain
streaming, tool-call round trips with typed-history assertions, unknown-tool
handling, every `StopReason` variant, cancellation observed both
mid-decision-stream and mid-tool-dispatch (via deterministic side effects
rather than real time delays), and the **permission flow** — granted (tool
executes, success recorded) and denied (tool never invoked, failed result, loop
continues).

The built-in tools are validated in isolation in `tools-builtin`'s own
`#[cfg(test)]` module against an in-memory `ClientAccess` fake: `fs_read`
streaming, `fs_write` recording the write + `requires_permission`,
`terminal_run` failure mapping, malformed-argument rejection
(`InvalidArguments`), and the missing-client error path — no ACP transport
required.
