# ACP session config options: model & effort selection

## Goal

Expose two per-session selectors to ACP clients (Zed and others):

- **Model** — `claude-sonnet-5` (default), `claude-opus-4-8`, `claude-haiku-4-5`
- **Effort** — `low` / `medium` / `high` (default `high`), mapped to the
  Anthropic Messages API `output_config: {"effort": ...}` parameter.

The mechanism is standard ACP v1: `NewSessionResponse.config_options`
(`SessionConfigOption::select(...)` with
`SessionConfigOptionCategory::{Model, ThoughtLevel}` — types already exist in
`agent-client-protocol-schema`) and the `session/set_config_option` request
(`SetSessionConfigOptionRequest` / `SetSessionConfigOptionResponse`).

**The selection is not fixed at `session/new`.** It lives in `SessionEntry` as
mutable state; `session/set_config_option` may change it at any time, and
*every* `session/prompt` reads the current value and builds a fresh
`AnthropicTurnService` for it via the factory. A mid-session change takes
effect on the next prompt (an in-flight turn finishes on its own model — that
is intended).

## Changes by crate

### `turn-anthropic` (`crates/turn-anthropic/src/request.rs`)

- `pub enum Effort { Low, Medium, High }` with `as_str()`
  (`"low"|"medium"|"high"`) and a case-insensitive `from_str_opt()`.
- `AnthropicConfig` gains `pub effort: Option<Effort>`; `new()` sets `None`.
- `from_env()` optionally reads `ANTHROPIC_EFFORT` (invalid value → `None`).
- `build_body()`: when `Some(effort)`, insert top-level
  `"output_config": {"effort": effort.as_str()}`.
- `lib.rs`: `pub use request::Effort;`

`None` means the field is omitted and the API default (`high`) applies —
backwards compatible for the cli-agent / env-only path.

### `acp-agent`

New module `crates/acp-agent/src/selection.rs`:

- `pub struct ModelSelection { pub model: String, pub effort: Effort }`
  (`Effort` from `turn_anthropic`).
- `pub const MODEL_CATALOG: &[(&str, &str)]` — the three model ids plus
  human-readable names.
- `ModelSelection::for_config(config: &AgentConfig)`: model =
  `config.default_model` if it is in the catalog, else `claude-sonnet-5`;
  effort = `High`.
- `config_options(sel) -> Vec<SessionConfigOption>`: two selects —
  id `"model"` (category `Model`) and id `"effort"` (category `ThoughtLevel`).
- `apply_update(sel, config_id, value) -> bool`: validates id + value and
  updates the selection; invalid input → `false`, selection unchanged.

`crates/acp-agent/src/config.rs`:

- `AgentConfig` gains `pub default_model: String`; `Default` =
  `"claude-sonnet-5"`; `from_env()` reads `ANTHROPIC_MODEL` (mirroring what
  `AnthropicConfig::from_env` already does).

`crates/acp-agent/src/agent.rs`:

- `NextTurnFactory` becomes
  `Arc<dyn Fn(&ModelSelection) -> Arc<dyn NextTurnService> + Send + Sync>`.
- `SessionEntry` gains `selection: StdMutex<ModelSelection>`.
- `session/new`: selection = `ModelSelection::for_config(&config)`; respond
  `NewSessionResponse::new(id).config_options(selection::config_options(&sel))`.
- New `session/set_config_option` handler (`SetSessionConfigOptionRequest`):
  unknown session → `invalid_params` error; otherwise `apply_update` and
  respond `SetSessionConfigOptionResponse::new(config_options(&sel))` (the full
  option set, as the schema requires). Invalid id/value → warn and respond
  with the current (unchanged) state.
- `session/prompt`: read the current selection from `SessionEntry` and call
  `factory(&selection)`.

`crates/acp-agent/src/lib.rs`:

- `anthropic_factory()`: the closure takes `&ModelSelection`, clones the base
  `AnthropicConfig`, overrides `model` and `effort: Some(sel.effort)`.
- `canned_replay_factory()`: closure ignores the argument.
- `build_registry()`: the subagent nested backend is built with
  `next_turn_factory(&ModelSelection::for_config(config))` — subagents stay on
  the default model/effort for now (threading selection through `ToolContext`
  is out of scope).

### `cli-agent`

`crates/cli-agent/src/repl.rs` (the `(deps.next_turn_factory)()` call site):
pass `&ModelSelection::for_config(&deps.config)`. Nothing else changes;
`AgentDeps` / `AgentConfig::default()` keep existing test constructors working.

## Tests

- `turn-anthropic` (request.rs unit tests): `output_config` present when
  `Some(effort)`, absent when `None`; `Effort::from_str_opt` roundtrip.
- `acp-agent` (selection.rs unit tests): shape of `config_options`
  (categories, current values); `apply_update` accepts valid ids/values and
  rejects invalid ones.
- `acp-agent` new `tests/config_options.rs` (mirroring the in-process fake
  client harness in `tests/e2e_prompt.rs`):
  1. `session/new` → response contains both config options with the expected
     values;
  2. `session/set_config_option` (model = opus) → response echoes the updated
     `current_value`;
  3. `session/prompt` → a spy factory (`Arc<Mutex<Vec<ModelSelection>>>`)
     captured the selected model/effort.
- Update the factory closures in `tests/e2e_prompt.rs` and
  `tests/mcp_session_wiring.rs` to the new signature.

## Verification

- `cargo build`, then `cargo test -p turn-anthropic -p acp-agent -p cli-agent`,
  then full `cargo test`.
- `#![warn(missing_docs)]` in acp-agent and turn-anthropic — all new public
  items need doc comments.

## Risks / notes

- If the gateway does not support `output_config`, requests fail with HTTP
  400; the escape hatch is `effort: None` (field omitted). The ACP path always
  sends `output_config` because the selection is always concrete (default
  `high`).
- Subagents (`SubagentTool`) run on the default model/effort — a deliberate v1
  limitation.
