# Session persistence & `session/load`

This note describes how session *core* state is persisted and how a persisted
session is reopened over ACP.

## The store abstraction

`agent-core` owns the persistence seam so every mode (chat, orchestrate, and
future modes) and both front-ends (CLI, ACP) can share it without depending on
a backend crate.

- `SessionStore` (`agent-core/src/store.rs`) is an `async`, `Send + Sync` trait
  held behind `Arc<dyn SessionStore>`:
  - `save(&self, record: &SessionRecord) -> Result<()>`
  - `load(&self, session_id: &str) -> Result<Option<SessionRecord>>`
  - `list_ids(&self) -> Result<Vec<String>>`
- `InMemorySessionStore` — the default, non-durable `HashMap`-backed store used
  in tests and whenever persistence is not configured.
- `JsonFileSessionStore` — a disk-backed store that writes one pretty-printed
  JSON file per session at `<dir>/<session_id>.json`. The directory is created
  lazily on the first `save`. A missing file (or missing directory) yields
  `None` / an empty id list rather than an error; malformed JSON surfaces as a
  store error (mapped to a clean ACP error, never a panic).

Persistence is best-effort: failures are logged and never crash a turn.

## Persisted record & versioning

Only the shared *core* of a session is persisted, as a `SessionRecord`
(`agent-core/src/session.rs`):

```
SessionRecord {
    version: u32,          // SESSION_RECORD_VERSION (currently 1)
    session_id: String,
    workspace_roots: Vec<String>,
    history: Conversation,
    system_prompt: Option<String>,
    usage: TokenUsage,
}
```

`available_tools` is intentionally **excluded**: tools depend on the MCP
servers reconnected at load time, so they are re-derived from the live tool
registry via `SessionState::from_record(record, available_tools)` rather than
persisted. `SessionState::to_record()` produces the snapshot.

`version` is stamped from `SESSION_RECORD_VERSION`. This is the first persisted
format, so there is no migration path yet; unsupported versions are handled at
load time (logged / surfaced as an error rather than silently trusted).

## Store selection (wiring)

`AgentDeps` carries `store: Arc<dyn SessionStore>`, defaulting to
`default_store()` (an `InMemorySessionStore`). The binary chooses the backend in
`default_deps()` based on `AgentConfig::session_persistence_dir`:

- set (via the `ACP_SESSION_DIR` environment variable) → `JsonFileSessionStore`
  rooted at that directory, so sessions survive restarts;
- unset → `InMemorySessionStore`.

`session/new`, `session/prompt`, and `_session/inject` all route through the
store: state is loaded-or-created and saved back after each completed turn.

## `session/load` flow

`initialize` advertises `AgentCapabilities.load_session == true`. The
`session/load` handler:

1. loads the `SessionRecord` for the requested `session_id` from the store;
   an unknown id (or a deserialization error) returns an `invalid_params`
   error;
2. reconnects the request's MCP servers and re-derives the live tool registry
   (as `session/new` does);
3. rehydrates `SessionState` via `SessionState::from_record`;
4. replays the stored history to the client in order as `session/update`
   notifications (user text, assistant text, tool calls, tool results);
   unknown/ingested entries are skipped with a log line;
5. re-advertises the available commands (e.g. `/usage`);
6. returns `LoadSessionResponse` with the current `config_options`.

A subsequent `session/prompt` then sees the prior history and continues
accumulating usage.
