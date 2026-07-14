---
sessionId: session-260713-223511-1er3
---

# Requirements

## Overview & Goals
Deliver two related capabilities on top of the existing ACP/CLI agent stack:

1. **First-class `/usage` over ACP** — expose the cumulative token-usage/cost report through the ACP
   *command* mechanism (advertised `AvailableCommand` + a dedicated command dispatcher) instead of the
   current ad-hoc string match inside the `session/prompt` handler.
2. **Session/state-management refactor** — introduce a pluggable `SessionStore` abstraction (in-memory
   default + disk-backed JSON persistence), split session state into a *shared core* (conversation
   history + usage) plus *mode-specific context* so chat / orchestrate / future modes reuse the same
   core, and add ACP `session/load` (advertising the `loadSession` capability) so a persisted session can
   be reopened and its history replayed to the client.

The user-visible outcome: a client can disconnect and later `session/load` a previously persisted
session, see its conversation history replayed, continue prompting with intact memory + usage totals,
and run `/usage` as a proper command in both CLI and ACP.

## Scope
**In scope**
- ACP `/usage` as a dispatched command (advertised via `AvailableCommandsUpdate`, handled by a shared
  command dispatcher rather than inline interception in `session/prompt`).
- `SessionStore` trait in `agent-core` with two implementations: `InMemorySessionStore` (default) and
  `JsonFileSessionStore` (disk-backed JSON, one file per session under a configurable directory).
- Refactor of `SessionState` into a serializable *core* (`session_id`, `history`, `usage`,
  `workspace_roots`, `system_prompt`) + a *mode context* seam so mode-local state (e.g. orchestrate's
  `OrchestratedStepContext`, chat has none) is kept separate but can share the core.
- ACP `session/load` handler + `AgentCapabilities.load_session(true)`, including history replay to the
  client via `session/update` notifications and returning current `config_options`.
- Wiring the ACP `session/new`, `session/prompt`, and `_session/inject` paths through the store so state
  is persisted after each turn.

**Out of scope**
- `session/list`, `session/delete`, `session/resume`, `session/fork` ACP methods (only `session/load`).
- Subagent selection changes and MCP transport work (unchanged).
- Cross-process concurrent access / file locking guarantees beyond best-effort single-writer.
- Migrating historical on-disk formats (this is the first persisted format; version it but no migration).

## User Stories
- As an ACP client, I want `/usage` surfaced and handled as an advertised command so I can invoke it
  reliably without the agent guessing from raw prompt text.
- As a user, I want my session (history + token usage) persisted to disk so that after restarting the
  agent I can reload it and keep the conversation context.
- As an ACP client, I want to call `session/load` on a known session id and have the agent replay the
  prior conversation and accept new prompts against it.
- As a maintainer, I want chat and orchestrate modes to share one core session-state type so usage and
  history handling live in a single place.

## Functional Requirements
1. `initialize` advertises `load_session: true` in `AgentCapabilities`.
2. `session/new` creates state through the configured `SessionStore` and persists it.
3. `/usage` invoked via ACP returns the same summary text as today (`format_usage`) as an
   `AgentMessageChunk`, produced by a shared command dispatcher (no model turn), and is advertised in
   `AvailableCommandsUpdate` on both `session/new` and `session/load`.
4. `session/load` with a known persisted `session_id`: loads core state from the store, replays history
   to the client as `session/update` notifications (user + assistant messages, tool calls/results in
   order), re-advertises available commands, and returns `LoadSessionResponse` with current
   `config_options`. Unknown id → `invalid_params` error.
5. After each completed prompt/inject turn, the session's core state (history + usage) is written back to
   the store.
6. `JsonFileSessionStore` round-trips core state: a saved session deserializes to an equal core state.
7. The CLI `/usage` path continues to work and reuses the shared dispatcher/format helper.

## Non-Functional Requirements
- No new hard third-party dependency beyond what the workspace already uses (`serde`, `serde_json`,
  `tokio`). A uuid dependency is NOT introduced (keep the existing `uuid_like` counter).
- Store trait is `async` + `Send + Sync` behind `Arc` so it fits the existing async handlers.
- Persistence failures must be logged and must NOT crash a turn or the session (best-effort durability).

# Technical Design

## Current Implementation
- `crates/agent-core/src/session.rs` — `SessionState` (all fields already serde-friendly **except**
  `available_tools: Vec<ToolDescriptor>`; `ToolDescriptor` derives only `Debug, Clone, PartialEq`, no
  serde) and `TurnContext`. `history::Conversation` and `event::TokenUsage` both derive
  `Serialize/Deserialize`.
- `crates/acp-agent/src/agent.rs` — `build_agent` registers handlers; sessions live in an in-memory
  `Sessions = Arc<StdMutex<HashMap<String, Arc<SessionEntry>>>>`. `/usage` is intercepted inline in the
  `session/prompt` closure (lines ~272-296) by comparing `user_text.trim() == USAGE_COMMAND`.
  `session/new` already advertises the `/usage` command via `AvailableCommandsUpdate`. There is **no**
  `session/load` handler and `AgentCapabilities::default()` is advertised (so `load_session=false`).
- `crates/acp-agent/src/usage.rs` — `format_usage(&SessionState, model)` and `USAGE_COMMAND`.
- `crates/cli-agent/src/repl.rs` — a single persistent `SessionState`, `/usage` handled by
  `render_usage` (duplicated formatting logic vs `usage.rs`).
- `crates/orchestrated/src/context.rs` — `OrchestratedStepContext` (proposal/results/attempts) is the
  orchestrate mode's mode-local state; it does NOT currently reference `SessionState`.

## Key Decisions
- **`SessionStore` lives in `agent-core`** (not acp-agent) so both CLI and ACP and future modes depend on
  it without a backend crate dependency. Rationale: `agent-core` already owns `SessionState`.
- **Persist a `SessionRecord` (core only), not the whole `SessionState`.** `available_tools` is derived
  from the live tool registry at load time (tools depend on MCP servers reconnected during load), so it
  is excluded from the persisted record. Rationale: avoids adding serde to `ToolDescriptor` and avoids
  persisting stale/unusable tool schemas.
- **Core vs mode-context via composition, not trait objects on state.** Introduce `SessionCore` (the
  shared, serializable fields) embedded in `SessionState`, and keep mode-local state (e.g.
  `OrchestratedStepContext`) as a separate object owned by the mode, referencing the same core.
- **`/usage` as a command dispatcher seam.** Add a small `commands` module (in `acp-agent`, reused by
  CLI) that maps a recognized command string to an action producing display text, so both the inline
  prompt path and any future command routing share one place.
- **Store is injected via `AgentDeps`** as `Arc<dyn SessionStore>`, defaulting to `InMemorySessionStore`.
  The binary selects `JsonFileSessionStore` when a persistence dir is configured.

## Proposed Changes
1. **`agent-core`**
   - New module `store.rs`: `SessionRecord` (serializable core snapshot), `SessionStore` trait
     (`async fn save(&self, record) -> Result<()>`, `async fn load(&self, id) -> Result<Option<SessionRecord>>`,
     `async fn list_ids(&self) -> Result<Vec<String>>`), `InMemorySessionStore`, `JsonFileSessionStore`.
   - Refactor `session.rs`: extract `SessionCore` (all serde), embed it in `SessionState`, and provide
     `SessionState::to_record()`/`from_record(record, available_tools)`.
2. **`acp-agent`**
   - New `commands.rs`: `enum AgentCommand { Usage }`, `parse_command(text) -> Option<AgentCommand>`, and
     `run_command(cmd, &SessionState, model) -> String` (delegating to `format_usage`). Advertised
     command list built here.
   - `agent.rs`: add `store: Arc<dyn SessionStore>` to `AgentDeps`; route `session/new`, `session/prompt`,
     `_session/inject` through the store (load-or-create + save-after-turn); replace the inline `/usage`
     block with dispatcher usage; advertise `load_session(true)`; add `session/load` handler.
3. **`cli-agent`**
   - `repl.rs`: reuse `acp_agent::commands` for `/usage`. CLI persistence, if any, must remain behind
     configuration and default to the current behavior.

## Data Models / Contracts
```rust
// agent-core/src/store.rs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionRecord {
    pub version: u32,
    pub session_id: String,
    pub workspace_roots: Vec<String>,
    pub history: Conversation,
    pub system_prompt: Option<String>,
    pub usage: TokenUsage,
}

#[async_trait::async_trait]
pub trait SessionStore: Send + Sync {
    async fn save(&self, record: &SessionRecord) -> crate::Result<()>;
    async fn load(&self, session_id: &str) -> crate::Result<Option<SessionRecord>>;
    async fn list_ids(&self) -> crate::Result<Vec<String>>;
}
```
```rust
// agent-core/src/session.rs (shape)
pub struct SessionCore {
  pub session_id: String,
  pub workspace_roots: Vec<String>,
  pub history: Conversation,
  pub system_prompt: Option<String>,
  pub usage: TokenUsage,
}
// SessionState keeps available_tools in-memory only and can convert to/from SessionRecord.
```

ACP: `LoadSessionRequest { session_id, cwd, mcp_servers, additional_directories }` →
`LoadSessionResponse` with `config_options` set accordingly.

## Reference Analog
- The existing `session/new` handler in `crates/acp-agent/src/agent.rs` is the template for the `session/load`
  handler (state construction, MCP reconnect via `connect_session_mcp_servers`, command advertisement, and
  returning `config_options`).
- The existing inline `/usage` interception block is the template for `commands::run_command`.
- `OrchestratedStepContext` (`crates/orchestrated/src/context.rs`) provides the analog for mode-local state
  (shared core + mode-local guarded state, never holding guards across await).

## Risks & Constraints
- **Store + concurrency model**: `SessionEntry` is per-session with an internal mutex; save must happen after the
  turn's state guard is dropped to avoid holding locks across `.await`.
- **History replay fidelity**: mapping typed `HistoryEntry` back to ACP `session/update` variants must cover user text,
  assistant text, tool calls, and tool results in order. Unknown/ingested entries must be skipped
  gracefully (log + best-effort).
- **Cross-turn save races**: handled by per-session serialization already present; disk writes happen per session id.
- **Disk errors**: non-fatal; log and continue.
- **`AgentDeps` struct is public**: adding the `store` field will require updating all literal construction sites and
  test builders.
- **History replay / unknown versions**: decide behavior for unsupported `SessionRecord.version` (error vs skip logged).

# Testing

## Validation Approach
Use the workspace's existing Cargo test tooling.
Primary commands (offline):
- `cargo test --workspace`
- per-crate during development: `cargo test -p agent-core`, `cargo test -p acp-agent`, `cargo test -p cli-agent`.

Note: tests run offline using the deterministic replay backend (fixtures assert strings like
"Hello from the replay backend!" and usage totals like "Token usage this session").

Additionally validate docs/lints:
- `cargo build --workspace`
- `cargo clippy --workspace --all-targets`

## Key Scenarios
- **`InMemorySessionStore`**: save then load returns an equal `SessionRecord`; load unknown id returns `None`.
- **`JsonFileSessionStore`**: save writes `<dir>/<session_id>.json`; load reads it back to an equal record;
  list_ids returns saved ids; load missing file returns `None`.
- **`SessionState` ↔ `SessionRecord` fidelity**: `to_record()` / `from_record()` preserve history/usage/workspace
  roots/system prompt; `available_tools` is re-derived from the passed tool registry.
- **ACP `/usage` command**: with replay backend, sending `/usage` yields an `AgentMessageChunk` containing the
  expected usage totals text and does not run a model turn (no new assistant generation).
- **ACP `session/load` happy path**: create+persist a session, then `session/load` the same id replays stored user /
  assistant messages as `session/update` notifications; the response re-advertises `/usage` and returns `config_options`.
- **ACP `session/load` unknown id**: returns an `invalid_params` error.
- **`initialize` capability**: advertised `AgentCapabilities.load_session == true`.

## Edge Cases
- Empty history session load.
- History with tool calls + tool results (order preserved).
- Two turns in quick succession for the same session (no lost history; usage only increases).
- Unsupported persisted `SessionRecord.version`.
- Corrupt/partial JSON file: `load` returns an error that the handler maps to a clean ACP error (not a panic).
- Persisted file exists but disk dir missing initially: store creates dir on first save.
- `/usage` case/whitespace handling: command parsing matches current behavior on `trim()`.

## Regression Guards
Run before AND after the change:
- `cargo test -p agent-core`
- `cargo test -p acp-agent`
- `cargo test -p cli-agent`
- `cargo test --workspace`

Explicit test expectations to preserve:
- CLI usage formatting output must stay byte-for-byte compatible where tests assert on exact strings.
- Ensure any existing assertions around replay backend usage/cost (e.g. messages containing "Token usage this session") still pass.

Concrete regression tests to run/keep green:
- `crates/cli-agent/src/repl.rs`
  - `streams_replayed_answer_and_exits`
  - `usage_command_prints_totals_and_cost`
  - `empty_lines_are_skipped_and_eof_exits`
- `crates/acp-agent/tests/`
  - `wiring.rs` (command/advertisement wiring)
  - `e2e_prompt.rs` (prompt flow with replay backend)
  - `config_options.rs` (config option propagation)
  - `e2e_orchestrate.rs` and `session_modes.rs` (mode selection + continuity)
  - `elicitation.rs` (unstable elicitation wiring)

## New & Modified Tests
Add/extend tests in-place so failures point to the relevant feature:

### agent-core
- Unit tests for `SessionStore`:
  - In-memory: save/load/unknown.
  - JSON file store: round-trip, list_ids, missing id, corrupt JSON, unknown record version.
- Unit tests for `SessionState` conversion:
  - `to_record()`/`from_record()` preserve history/usage/workspace roots/system prompt.
  - Re-derivation of `available_tools` from passed registry (not serialized).

### acp-agent
- Integration tests for ACP:
  - `session/load` happy path: create a session with history, persist, load, validate `session/update` replay order and command advertisement.
  - `session/load` unknown id returns `invalid_params`.
  - `initialize` advertises `load_session: true`.
- End-to-end test for `/usage` as a command:
  - Ensure no model turn runs (replay backend should show the same deterministic behavior).

Where to add/extend:
- Prefer extending `crates/acp-agent/tests/wiring.rs` for capability advertisement and `/usage` command end-to-end.
- Add a dedicated `session/load` integration test module (either a new file or in `wiring.rs`) to keep
  replay-order and unknown-id error behavior isolated.

### cli-agent
- Reuse the existing usage test and verify output is unchanged:
  - If output is refactored to reuse shared dispatcher/format helper, update imports only while keeping text identical.

# Assumptions & Open Questions

## Significant
- **ACP command delivery semantics**: ACP clients deliver advertised commands as prompt content naming the command.
  So `/usage` is implemented as a dedicated command dispatcher invoked from the prompt path, not a separate RPC.
  Alternative (a bespoke `_session/command` extension method) is intentionally not used.
- **Persisted shape excludes `available_tools`**: loaded sessions re-derive tools based on the `session/load` request's
  reconnected MCP server set.
- **Persistence directory config**: assume config/env knob selects `JsonFileSessionStore` when configured.
  If unset, default to `InMemorySessionStore`.

## Non-significant / Known caveats
- History replay of unsupported/unknown entry variants is best-effort (skipped) and should be logged.

## Complexity self-check
Complex — cross-cutting change touching `agent-core`, `acp-agent`, and `cli-agent`, plus new public API surface (`AgentDeps.store`).

# Delivery Steps

### ✓ Step 1: Introduce SessionStore + core/record split in agent-core
Goal: `agent-core` exposes a `SessionStore` trait with in-memory + JSON-file implementations and a
serializable `SessionRecord`, and `SessionState` can convert to/from a record.
Scope: `crates/agent-core/src/store.rs` (new), `session.rs`, `lib.rs`.
Acceptance Criteria:
- [ ] `SessionStore` trait + `InMemorySessionStore` + `JsonFileSessionStore` compile and are re-exported.
- [ ] `SessionRecord` derives `Serialize/Deserialize` and includes a `version` field.
- [ ] `SessionState::to_record()` and `from_record(record, available_tools)` preserve history/usage/
      roots/system prompt and re-derive `available_tools` from the argument.
- [ ] Unit tests: in-memory save/load/unknown, JSON round-trip/list/missing/corrupt/unknown-version pass.
Verification: `cargo test -p agent-core` → green

### ✓ Step 2: Make /usage a shared command dispatcher
Goal: a single command module produces the `/usage` report, used by both ACP and CLI, replacing the
inline string interception.
Scope: `crates/acp-agent/src/commands.rs` (new), `usage.rs`, `agent.rs` (usage block), `cli-agent/src/repl.rs`.
Acceptance Criteria:
- [ ] `parse_command("/usage")` returns the usage command; unrelated text returns `None`.
- [ ] `run_command` returns text equal to the existing `format_usage` output for the same state/model.
- [ ] ACP `session/prompt` routes the command through the dispatcher (no behavior change to emitted
      `AgentMessageChunk`).
- [ ] CLI `/usage` uses the dispatcher/format helper; existing CLI usage test still passes.
Verification: `cargo test -p acp-agent -p cli-agent` → green

### ✓ Step 3: Route ACP sessions through the store and persist after turns
Goal: `AgentDeps` carries an `Arc<dyn SessionStore>`; `session/new`, `session/prompt`, and
`_session/inject` create/persist core state through it.
Scope: `crates/acp-agent/src/agent.rs`, `lib.rs`, and all `AgentDeps` construction sites/tests.
Acceptance Criteria:
- [ ] `AgentDeps` has a `store` field with an in-memory default helper; all constructors/tests updated.
- [ ] After a completed prompt/inject turn, the session's record is saved (verified via an in-memory
      store assertion in a test).
- [ ] All existing `acp-agent` and `cli-agent` tests pass unchanged in behavior.
Verification: `cargo test -p acp-agent -p cli-agent` → green

### ✓ Step 4: Implement ACP session/load + loadSession capability
Goal: `initialize` advertises `load_session: true` and a `session/load` handler restores a persisted
session, replays its history, re-advertises commands, and returns `config_options`.
Scope: `crates/acp-agent/src/agent.rs` (initialize + new handler), history→`session/update` mapping.
Acceptance Criteria:
- [ ] `initialize` response has `AgentCapabilities.load_session == true`.
- [ ] `session/load` for a persisted id replays user/assistant/tool history as `session/update`
      notifications, re-advertises `/usage`, and returns a `LoadSessionResponse` with `config_options`.
- [ ] `session/load` for an unknown id returns an `invalid_params` error (no panic).
- [ ] A `session/prompt` after `session/load` sees prior history and continues usage accumulation.
Verification: `cargo test -p acp-agent` → green

### ✓ Step 5: JSON persistence wiring, docs, and full-workspace gate
Goal: the binary selects `JsonFileSessionStore` when a persistence directory is configured; docs updated;
whole workspace builds clean.
Scope: `crates/acp-agent/src/lib.rs`/`config.rs` (store selection), `docs/design/` note, workspace.
Acceptance Criteria:
- [ ] With the persistence dir configured, `default_deps` uses `JsonFileSessionStore`; unset → in-memory.
- [ ] A design doc note describes the store abstraction, record format/version, and `session/load` flow.
- [ ] `cargo build --workspace` and `cargo clippy --workspace --all-targets` are clean (docs on new
      public items present).
Verification: `cargo test --workspace` → green