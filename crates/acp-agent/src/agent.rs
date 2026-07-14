//! The ACP `Agent`-side wiring for the stdio server.
//!
//! [`build_agent`] assembles an [`agent_client_protocol`] connection builder
//! that implements the agent half of the protocol. The builder registers one
//! handler per supported method:
//!
//! - `initialize` — negotiate the protocol version and advertise capabilities.
//! - `authenticate` — a no-op/passthrough (this agent requires no auth).
//! - `session/new` — allocate a [`SessionState`] and return its id.
//! - `session/prompt` — run a [`TurnEngine`] turn, streaming `session/update`
//!   notifications and returning a `stopReason`.
//! - `session/cancel` — cancel the in-flight turn for a session.
//!
//! The decision layer is injected as a [`NextTurnFactory`] so the same wiring
//! serves the real Anthropic backend (added later) and the deterministic replay
//! backend used by tests, without touching the turn engine.
//!
//! `session/new`'s `mcp_servers` (the ACP-native way for a client to tell the
//! agent which MCP servers to use for a session) is consumed here: each
//! `stdio` server is launched as a real child process via
//! [`mcp_client::stdio::StdioMcpConnection`] and its tools are registered into
//! a *per-session* copy of the tool registry (base tools cloned from
//! [`AgentDeps::tools`] plus whatever the session's MCP servers advertise), so
//! `session/prompt` dispatches against tools that are actually available for
//! that session. Other transports (`http`/`sse`/`acp`) are not yet
//! implemented and are reported per-server as a warning rather than failing
//! the whole session.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use agent_client_protocol::schema::v1::{
    AgentCapabilities, AuthenticateRequest, AuthenticateResponse, AvailableCommand,
    AvailableCommandsUpdate, CancelNotification, Content, ContentBlock, ContentChunk,
    Implementation, InitializeRequest, InitializeResponse, LoadSessionRequest, LoadSessionResponse,
    McpServer, NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse, SessionId,
    SessionNotification, SessionUpdate, SetSessionConfigOptionRequest,
    SetSessionConfigOptionResponse, StopReason as AcpStopReason, TextContent, ToolCall,
    ToolCallContent, ToolCallStatus as AcpToolCallStatus, ToolCallUpdate, ToolCallUpdateFields,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{
    Agent, Client, ConnectTo, JsonRpcRequest, JsonRpcResponse, MetaCapability, MetaCapabilityExt,
    on_receive_notification, on_receive_request,
};
use agent_core::{
    CancellationToken, HistoryEntry, InMemorySessionStore, NextTurnService, SessionRecord,
    SessionState, SessionStore, StopReason, ToolDescriptor, ToolRegistry, TurnEngine, TurnInbox,
};
use serde::{Deserialize, Serialize};
use mcp_client::stdio::StdioMcpConnection;
use mcp_client::{register_mcp_tools, McpConnection};
use tokio::sync::Mutex as AsyncMutex;
use tracing::{debug, info, warn};

use crate::client::AcpClientAccess;
use crate::config::AgentConfig;
use crate::selection::ModelSelection;
use crate::sink::AcpUpdateSink;
use crate::commands::{parse_command, run_command};
use crate::usage::USAGE_COMMAND;

/// Factory that produces a fresh [`NextTurnService`] for each prompt turn.
///
/// A fresh instance per turn keeps stateful backends (e.g. the deterministic
/// replay backend, which consumes scripted rounds) well-behaved across multiple
/// prompts in the same session.
pub type NextTurnFactory = Arc<dyn Fn(&ModelSelection) -> Arc<dyn NextTurnService> + Send + Sync>;

/// Dependencies required to build the agent connection.
#[derive(Clone)]
pub struct AgentDeps {
    /// Produces the decision-layer service for each turn.
    pub next_turn_factory: NextTurnFactory,
    /// Tools exposed to the turn engine.
    pub tools: ToolRegistry,
    /// Static configuration (identity, system prompt).
    pub config: AgentConfig,
    /// Backing store used to persist session *core* state (history + usage)
    /// after each turn, so sessions can survive process restarts.
    ///
    /// Defaults to [`default_store`] (an [`InMemorySessionStore`]) via
    /// construction helpers; the binary swaps in a disk-backed store when a
    /// persistence directory is configured.
    pub store: Arc<dyn SessionStore>,
}

/// Build the default [`SessionStore`]: a non-durable [`InMemorySessionStore`].
///
/// Used as the ergonomic default for [`AgentDeps::store`] in tests and when no
/// persistence directory is configured.
#[must_use]
pub fn default_store() -> Arc<dyn SessionStore> {
    Arc::new(InMemorySessionStore::new())
}

/// Persist the given session's core state via `store`, logging (but not
/// failing) on error.
///
/// Durability is best-effort: a store error must never crash the turn or the
/// session, so failures are logged and swallowed here.
async fn persist_record(store: &Arc<dyn SessionStore>, session_id: &str, record: &SessionRecord) {
    if let Err(err) = store.save(record).await {
        warn!(session_id, %err, "failed to persist session state");
    }
}

/// Wire method name for the steering extension request.
///
/// ACP extension methods must start with `_`.
const INJECT_METHOD: &str = "_session/inject";

/// The `inject` meta capability advertised in `initialize`, signalling that
/// this agent supports the custom [`INJECT_METHOD`] steering request.
struct InjectCapability;

impl MetaCapability for InjectCapability {
    fn key(&self) -> &'static str {
        "inject"
    }
}

/// Parameters for the `_session/inject` steering request.
#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "_session/inject", response = InjectResponse)]
struct InjectRequest {
    /// The session to steer.
    #[serde(rename = "sessionId")]
    session_id: SessionId,
    /// The message to inject, as prompt content blocks.
    #[serde(default)]
    prompt: Vec<ContentBlock>,
}

/// Response acknowledging a `_session/inject` steering request.
#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcResponse)]
struct InjectResponse {
    /// Whether the injected message was accepted into the session.
    received: bool,
    /// Whether the injection started a fresh prompt turn (because no turn was
    /// running at the time), as opposed to steering an in-flight turn.
    #[serde(rename = "startedNewTurn")]
    started_new_turn: bool,
}

/// Per-session runtime state shared between handlers.
struct SessionEntry {
    /// The typed session state, guarded for exclusive access during a turn.
    state: Arc<AsyncMutex<SessionState>>,
    /// The cancellation token for the currently-running (or next) turn.
    cancel: StdMutex<CancellationToken>,
    /// This session's tool registry: the base tools plus any tools
    /// discovered from the session's `mcp_servers` at `session/new` time.
    tools: ToolRegistry,
    /// Current model and effort selection for this session.
    selection: StdMutex<ModelSelection>,
    /// Steering inbox: user messages injected via [`INJECT_METHOD`] and drained
    /// by the running turn between decision rounds.
    inbox: TurnInbox,
    /// Whether a prompt turn is currently running for this session.
    running: Arc<AtomicBool>,
}

/// Shared registry of active sessions, keyed by session id string.
type Sessions = Arc<StdMutex<HashMap<String, Arc<SessionEntry>>>>;

/// Build the ACP agent connection.
///
/// The returned value implements [`ConnectTo<Client>`] and can be driven either
/// over a real transport (e.g. `Stdio`) or, in tests, directly by a client
/// builder via `connect_with`.
pub fn build_agent(deps: AgentDeps) -> impl ConnectTo<Client> {
    let sessions: Sessions = Arc::new(StdMutex::new(HashMap::new()));

    let init_config = deps.config.clone();
    let new_session_tools = deps.tools.clone();
    let new_session_config = deps.config.clone();
    let new_sessions = sessions.clone();
    let prompt_sessions = sessions.clone();
    let prompt_factory = deps.next_turn_factory.clone();
    let prompt_require_submit_result = deps.config.require_submit_result;
    let inject_require_submit_result = deps.config.require_submit_result;
    let prompt_thought_interval = deps.config.thought_interval;
    let inject_thought_interval = deps.config.thought_interval;
    let cancel_sessions = sessions.clone();
    let config_sessions = sessions.clone();
    let inject_sessions = sessions.clone();
    let inject_factory = deps.next_turn_factory.clone();
    let new_store = deps.store.clone();
    let prompt_store = deps.store.clone();
    let inject_store = deps.store.clone();
    let load_sessions = sessions.clone();
    let load_session_tools = deps.tools.clone();
    let load_session_config = deps.config.clone();
    let load_store = deps.store.clone();

    Agent
        .builder()
        .name(deps.config.name.clone())
        // initialize: negotiate protocol version, advertise capabilities.
        .on_receive_request(
            async move |req: InitializeRequest, responder, _cx| {
                let negotiated = req.protocol_version.min(ProtocolVersion::LATEST);
                info!(
                    requested = %req.protocol_version,
                    negotiated = %negotiated,
                    "initialize"
                );
                let response = InitializeResponse::new(negotiated)
                    .agent_capabilities(AgentCapabilities::default().load_session(true))
                    .agent_info(Implementation::new(
                        init_config.name.clone(),
                        init_config.version.clone(),
                    ))
                    // Advertise steering support so clients know they may call
                    // the custom `_session/inject` method.
                    .add_meta_capability(InjectCapability);
                responder.respond(response)
            },
            on_receive_request!(),
        )
        // authenticate: no-op passthrough.
        .on_receive_request(
            async move |_req: AuthenticateRequest, responder, _cx| {
                debug!("authenticate (no-op)");
                responder.respond(AuthenticateResponse::new())
            },
            on_receive_request!(),
        )
        // session/new: allocate session state, connecting any MCP servers the
        // client asked for and registering their tools into this session's own
        // registry.
        .on_receive_request(
            async move |req: NewSessionRequest, responder, cx| {
                let session_id = format!("session-{}", uuid_like());

                let mut tools = new_session_tools.clone();
                connect_session_mcp_servers(&session_id, &req.mcp_servers, &mut tools).await;

                let mut state = SessionState::new(session_id.clone());
                state.workspace_roots = std::iter::once(req.cwd.clone())
                    .chain(req.additional_directories.iter().cloned())
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect();
                state.system_prompt = new_session_config.system_prompt.clone();
                state.available_tools = tool_descriptors(&tools);

                let selection = ModelSelection::for_config(&new_session_config);
                let config_options = selection.config_options();

                // Persist the freshly-created core state so the session exists
                // in the store from the outset (best-effort).
                let record = state.to_record();
                persist_record(&new_store, &session_id, &record).await;

                let entry = Arc::new(SessionEntry {
                    state: Arc::new(AsyncMutex::new(state)),
                    cancel: StdMutex::new(CancellationToken::new()),
                    tools,
                    selection: StdMutex::new(selection),
                    inbox: Arc::new(StdMutex::new(Vec::new())),
                    running: Arc::new(AtomicBool::new(false)),
                });
                new_sessions
                    .lock()
                    .expect("sessions mutex poisoned")
                    .insert(session_id.clone(), entry);

                info!(session_id, "session/new");

                // Advertise the built-in slash commands this agent supports, so
                // clients can surface them (e.g. `/usage`) and route them back as
                // prompts.
                if let Err(err) = cx.send_notification(available_commands_notification(&session_id)) {
                    warn!(session_id, %err, "failed to advertise available commands");
                }

                responder.respond(
                    NewSessionResponse::new(SessionId::new(session_id)).config_options(config_options),
                )
            },
            on_receive_request!(),
        )
        // session/load: restore a persisted session from the store, reconnect
        // its MCP servers, rehydrate state, replay history to the client, and
        // re-advertise available commands.
        .on_receive_request(
            async move |req: LoadSessionRequest, responder, cx| {
                let session_id = req.session_id.0.to_string();

                // Read the persisted core state. A missing session is a client
                // error; a store/deserialization failure is surfaced cleanly
                // rather than panicking.
                let record = match load_store.load(&session_id).await {
                    Ok(Some(record)) => record,
                    Ok(None) => {
                        warn!(session_id, "session/load for unknown session");
                        return responder.respond_with_error(
                            agent_client_protocol::Error::invalid_params()
                                .data(format!("unknown session: {session_id}")),
                        );
                    }
                    Err(err) => {
                        warn!(session_id, %err, "session/load failed to read store");
                        return responder.respond_with_error(
                            agent_client_protocol::Error::invalid_params()
                                .data(format!("failed to load session {session_id}: {err}")),
                        );
                    }
                };

                // Reconnect any MCP servers the client asked for and re-derive
                // the live tool set (tool schemas are not persisted).
                let mut tools = load_session_tools.clone();
                connect_session_mcp_servers(&session_id, &req.mcp_servers, &mut tools).await;

                let available_tools = tool_descriptors(&tools);
                let history = record.history.clone();
                let state = SessionState::from_record(record, available_tools);

                let selection = ModelSelection::for_config(&load_session_config);
                let config_options = selection.config_options();

                let entry = Arc::new(SessionEntry {
                    state: Arc::new(AsyncMutex::new(state)),
                    cancel: StdMutex::new(CancellationToken::new()),
                    tools,
                    selection: StdMutex::new(selection),
                    inbox: Arc::new(StdMutex::new(Vec::new())),
                    running: Arc::new(AtomicBool::new(false)),
                });
                load_sessions
                    .lock()
                    .expect("sessions mutex poisoned")
                    .insert(session_id.clone(), entry);

                info!(session_id, entries = history.len(), "session/load");

                // Replay the stored conversation to the client in order, as
                // `session/update` notifications. Unknown/ingested entries are
                // skipped (best-effort) rather than failing the load.
                let acp_session_id = SessionId::new(session_id.clone());
                for entry in &history.entries {
                    if let Some(update) = history_entry_to_update(entry) {
                        let notification =
                            SessionNotification::new(acp_session_id.clone(), update);
                        if let Err(err) = cx.send_notification(notification) {
                            warn!(session_id, %err, "failed to replay history entry");
                        }
                    }
                }

                // Re-advertise the built-in slash commands so the reloaded
                // session exposes the same commands as a fresh one.
                if let Err(err) = cx.send_notification(available_commands_notification(&session_id)) {
                    warn!(session_id, %err, "failed to advertise available commands");
                }

                responder.respond(LoadSessionResponse::new().config_options(config_options))
            },
            on_receive_request!(),
        )
        // session/prompt: run a turn, streaming updates, return a stop reason.
        .on_receive_request(
            async move |req: PromptRequest, responder, cx| {
                let session_id = req.session_id.0.to_string();
                let Some(entry) = prompt_sessions
                    .lock()
                    .expect("sessions mutex poisoned")
                    .get(&session_id)
                    .cloned()
                else {
                    warn!(session_id, "session/prompt for unknown session");
                    return responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params()
                            .data(format!("unknown session: {session_id}")),
                    );
                };

                let user_text = prompt_text(&req.prompt);

                // Intercept built-in slash commands (e.g. `/usage`): produce the
                // command's display text as a plain agent message, without
                // running a model turn.
                if let Some(command) = parse_command(&user_text) {
                    let model = entry
                        .selection
                        .lock()
                        .expect("selection mutex poisoned")
                        .model
                        .clone();
                    let summary = {
                        let guard = entry.state.lock().await;
                        run_command(command, &guard, &model)
                    };
                    let notification = SessionNotification::new(
                        req.session_id.clone(),
                        SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                            TextContent::new(summary),
                        ))),
                    );
                    if let Err(err) = cx.send_notification(notification) {
                        warn!(session_id, %err, "failed to send /usage summary");
                    }
                    return responder.respond(PromptResponse::new(AcpStopReason::EndTurn));
                }

                // Fresh cancellation token for this turn; stored so session/cancel
                // can reach both the decision stream and running tools.
                let cancel = CancellationToken::new();
                *entry.cancel.lock().expect("cancel mutex poisoned") = cancel.clone();
                let factory = prompt_factory.clone();
                let require_submit_result = prompt_require_submit_result;
                let thought_interval = prompt_thought_interval;
                let tools = entry.tools.clone();
                let acp_session_id = req.session_id.clone();

                // Seed the turn's inbox with the user's prompt; the engine drains
                // it (together with any later steering injections) each round.
                entry
                    .inbox
                    .lock()
                    .expect("inbox mutex poisoned")
                    .push(user_text);
                entry.running.store(true, Ordering::SeqCst);
                let inbox = entry.inbox.clone();
                let running = entry.running.clone();
                let store = prompt_store.clone();

                // The turn must run off the event loop so the sink's blocking
                // permission requests do not deadlock the connection.
                cx.spawn({
                    let cx = cx.clone();
                    let selection = entry.selection.lock().expect("selection mutex poisoned").clone();
                    async move {
                        let next_turn = factory(&selection);
                        let client = Arc::new(AcpClientAccess::new(
                            cx.clone(),
                            acp_session_id.clone(),
                        ));
                        let mut engine = TurnEngine::new(next_turn, tools)
                            .with_client(client)
                            .with_inbox(inbox);
                        if require_submit_result {
                            engine = engine.require_terminal_tool();
                        }
                        if thought_interval > 0 {
                            engine = engine.with_thought_interval(thought_interval);
                        }
                        let mut sink = AcpUpdateSink::new(cx, acp_session_id);

                        let mut guard = entry.state.lock().await;
                        let result = engine.run_prompt(&mut guard, &mut sink, &cancel).await;
                        // Snapshot the core state while still holding the guard,
                        // then drop it before the (awaiting) persist so no lock
                        // is held across the store `.await`.
                        let record = guard.to_record();
                        drop(guard);
                        running.store(false, Ordering::SeqCst);

                        // Persist the updated history + usage after the turn
                        // (best-effort; failures are logged, not fatal).
                        persist_record(&store, &session_id, &record).await;

                        match result {
                            Ok(stop) => {
                                debug!(session_id, ?stop, "session/prompt finished");
                                responder.respond(PromptResponse::new(map_stop_reason(stop)))
                            }
                            Err(err) => {
                                warn!(session_id, %err, "session/prompt failed");
                                responder.respond_with_error(
                                    agent_client_protocol::Error::into_internal_error(err),
                                )
                            }
                        }
                    }
                })?;

                Ok(())
            },
            on_receive_request!(),
        )
        // session/set_session_config_option: update session-level configuration (e.g.
        // model/effort) and return the full set of current options.
        .on_receive_request(
            async move |req: SetSessionConfigOptionRequest, responder, _cx| {
                let session_id = req.session_id.0.to_string();
                let Some(entry) = config_sessions
                    .lock()
                    .expect("sessions mutex poisoned")
                    .get(&session_id)
                    .cloned()
                else {
                    warn!(session_id, "session/set_config_option for unknown session");
                    return responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params()
                            .data(format!("unknown session: {session_id}")),
                    );
                };

                let mut selection = entry.selection.lock().expect("selection mutex poisoned");
                let json_value = if let Some(id) = req.value.as_value_id() {
                    serde_json::Value::String(id.0.to_string())
                } else if let Some(b) = req.value.as_bool() {
                    serde_json::Value::Bool(b)
                } else {
                    serde_json::to_value(req.value).unwrap_or_default()
                };
                selection.apply_update(&req.config_id.0, json_value);
                let config_options = selection.config_options();

                info!(session_id, config_id = %req.config_id.0, "session/set_config_option");
                responder.respond(SetSessionConfigOptionResponse::new(config_options))
            },
            on_receive_request!(),
        )
        // _session/inject: steering — inject a user message into the running
        // turn, or start a fresh turn if none is running.
        .on_receive_request(
            async move |req: InjectRequest, responder, cx| {
                let session_id = req.session_id.0.to_string();
                let Some(entry) = inject_sessions
                    .lock()
                    .expect("sessions mutex poisoned")
                    .get(&session_id)
                    .cloned()
                else {
                    warn!(session_id, method = INJECT_METHOD, "inject for unknown session");
                    return responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params()
                            .data(format!("unknown session: {session_id}")),
                    );
                };

                let text = prompt_text(&req.prompt);
                entry
                    .inbox
                    .lock()
                    .expect("inbox mutex poisoned")
                    .push(text);

                // If a turn is already running, the engine will drain the inbox
                // on its next round: the message is steered into the current
                // prompt turn.
                if entry.running.load(Ordering::SeqCst) {
                    info!(session_id, method = INJECT_METHOD, "steering: injected into running turn");
                    return responder.respond(InjectResponse {
                        received: true,
                        started_new_turn: false,
                    });
                }

                // Otherwise the previous turn already stopped; start a fresh turn
                // so the user perceives a new prompt turn beginning.
                info!(session_id, method = INJECT_METHOD, "steering: starting new turn");
                let cancel = CancellationToken::new();
                *entry.cancel.lock().expect("cancel mutex poisoned") = cancel.clone();
                entry.running.store(true, Ordering::SeqCst);

                let factory = inject_factory.clone();
                let require_submit_result = inject_require_submit_result;
                let thought_interval = inject_thought_interval;
                let tools = entry.tools.clone();
                let inbox = entry.inbox.clone();
                let running = entry.running.clone();
                let acp_session_id = req.session_id.clone();
                let turn_entry = entry.clone();
                let store = inject_store.clone();

                cx.spawn({
                    let cx = cx.clone();
                    let selection =
                        entry.selection.lock().expect("selection mutex poisoned").clone();
                    async move {
                        let next_turn = factory(&selection);
                        let client =
                            Arc::new(AcpClientAccess::new(cx.clone(), acp_session_id.clone()));
                        let mut engine = TurnEngine::new(next_turn, tools)
                            .with_client(client)
                            .with_inbox(inbox);
                        if require_submit_result {
                            engine = engine.require_terminal_tool();
                        }
                        if thought_interval > 0 {
                            engine = engine.with_thought_interval(thought_interval);
                        }
                        let mut sink = AcpUpdateSink::new(cx, acp_session_id);

                        let mut guard = turn_entry.state.lock().await;
                        let result = engine.run_prompt(&mut guard, &mut sink, &cancel).await;
                        // Snapshot core state under the guard, then drop before
                        // the awaiting persist so no lock is held across `.await`.
                        let record = guard.to_record();
                        drop(guard);
                        running.store(false, Ordering::SeqCst);

                        // Persist the updated history + usage after the steering
                        // turn (best-effort).
                        persist_record(&store, &session_id, &record).await;

                        if let Err(err) = result {
                            warn!(session_id, %err, "steering turn failed");
                        } else {
                            debug!(session_id, "steering turn finished");
                        }
                        Ok(())
                    }
                })?;

                responder.respond(InjectResponse {
                    received: true,
                    started_new_turn: true,
                })
            },
            on_receive_request!(),
        )
        // session/cancel: abort the in-flight turn for a session.
        .on_receive_notification(
            async move |notif: CancelNotification, _cx| {
                let session_id = notif.session_id.0.to_string();
                if let Some(entry) = cancel_sessions
                    .lock()
                    .expect("sessions mutex poisoned")
                    .get(&session_id)
                    .cloned()
                {
                    info!(session_id, "session/cancel");
                    entry
                        .cancel
                        .lock()
                        .expect("cancel mutex poisoned")
                        .cancel();
                } else {
                    warn!(session_id, "session/cancel for unknown session");
                }
                Ok(())
            },
            on_receive_notification!(),
        )
}

/// Build the `AvailableCommandsUpdate` notification advertising this agent's
/// built-in slash commands (currently just `/usage`).
///
/// Shared by `session/new` and `session/load` so a reloaded session exposes the
/// same commands as a freshly created one.
fn available_commands_notification(session_id: &str) -> SessionNotification {
    SessionNotification::new(
        SessionId::new(session_id.to_string()),
        SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(vec![
            AvailableCommand::new(
                USAGE_COMMAND.trim_start_matches('/'),
                "Show cumulative token usage and estimated cost for this session.",
            ),
        ])),
    )
}

/// Concatenate the text of a slice of `agent-core` content blocks.
fn blocks_text(blocks: &[agent_core::ContentBlock]) -> String {
    blocks
        .iter()
        .map(|block| match block {
            agent_core::ContentBlock::Text { text } => text.clone(),
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Map a persisted [`HistoryEntry`] to the ACP [`SessionUpdate`] used to replay
/// it to a client during `session/load`.
///
/// Returns `None` for entries that have no faithful `session/update` analog
/// (e.g. previously-ingested provider fragments), which are skipped during
/// replay on a best-effort basis.
fn history_entry_to_update(entry: &HistoryEntry) -> Option<SessionUpdate> {
    match entry {
        HistoryEntry::User(message) => Some(SessionUpdate::UserMessageChunk(ContentChunk::new(
            ContentBlock::Text(TextContent::new(blocks_text(&message.content))),
        ))),
        HistoryEntry::Assistant(message) => Some(SessionUpdate::AgentMessageChunk(
            ContentChunk::new(ContentBlock::Text(TextContent::new(blocks_text(
                &message.content,
            )))),
        )),
        HistoryEntry::ToolCall(record) => {
            let tool_call = ToolCall::new(record.id.0.clone(), record.tool.name().to_string())
                .status(AcpToolCallStatus::Completed);
            Some(SessionUpdate::ToolCall(tool_call))
        }
        HistoryEntry::ToolResult(record) => {
            let status = if record.success {
                AcpToolCallStatus::Completed
            } else {
                AcpToolCallStatus::Failed
            };
            let mut fields = ToolCallUpdateFields::new().status(status);
            let text = blocks_text(&record.content);
            if !text.is_empty() {
                fields = fields.content(vec![ToolCallContent::Content(Content::new(
                    ContentBlock::Text(TextContent::new(text)),
                ))]);
            }
            Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                record.id.0.clone(),
                fields,
            )))
        }
        HistoryEntry::Ingested(record) => {
            debug!(source = %record.source, "session/load: skipping ingested history entry");
            None
        }
    }
}

/// Build [`ToolDescriptor`]s advertised to the decision layer.
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

/// Extract the concatenated text of an incoming prompt's content blocks.
fn prompt_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Map the provider-agnostic [`StopReason`] to the ACP schema stop reason.
fn map_stop_reason(reason: StopReason) -> AcpStopReason {
    match reason {
        StopReason::EndTurn => AcpStopReason::EndTurn,
        StopReason::MaxTokens => AcpStopReason::MaxTokens,
        // ToolUse is an internal "keep looping" signal; if it ever surfaces as a
        // terminal reason, the turn effectively ended.
        StopReason::ToolUse => AcpStopReason::EndTurn,
        StopReason::Cancelled => AcpStopReason::Cancelled,
        StopReason::Refusal => AcpStopReason::Refusal,
    }
}

/// Generate a process-unique, monotonically increasing id suffix.
///
/// A full UUID dependency is unnecessary for session ids that only need to be
/// unique within a single running process.
fn uuid_like() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Connect every MCP server the client requested for this session (via
/// `session/new`'s `mcp_servers`) and register each server's tools into
/// `tools`.
///
/// This is the real runtime path (as opposed to the seam-level unit tests in
/// `mcp-client`): it launches actual child processes for `stdio` servers.
/// A server that fails to connect, or advertises no tools, does not fail
/// `session/new` as a whole -- it is logged and the session simply proceeds
/// without that server's tools, so one misconfigured MCP server cannot take
/// down the whole session.
async fn connect_session_mcp_servers(
    session_id: &str,
    servers: &[McpServer],
    tools: &mut ToolRegistry,
) {
    for server in servers {
        match server {
            McpServer::Stdio(stdio) => {
                let env: Vec<(String, String)> = stdio
                    .env
                    .iter()
                    .map(|var| (var.name.clone(), var.value.clone()))
                    .collect();
                match StdioMcpConnection::connect(
                    stdio.name.clone(),
                    &stdio.command,
                    &stdio.args,
                    &env,
                )
                .await
                {
                    Ok(connection) => {
                        let connection: Arc<dyn McpConnection> = Arc::new(connection);
                        match register_mcp_tools(tools, connection).await {
                            Ok(names) => info!(
                                session_id,
                                server = %stdio.name,
                                tools = ?names,
                                "session/new: connected mcp stdio server"
                            ),
                            Err(err) => warn!(
                                session_id,
                                server = %stdio.name,
                                %err,
                                "session/new: mcp server connected but tool registration failed"
                            ),
                        }
                    }
                    Err(err) => warn!(
                        session_id,
                        server = %stdio.name,
                        %err,
                        "session/new: failed to connect mcp stdio server"
                    ),
                }
            }
            // The http/sse (and, behind the `unstable_mcp_over_acp` feature,
            // acp) MCP transports are part of the ACP schema but do not yet
            // have a concrete client implementation in this agent; the stdio
            // transport above is the one genuinely usable transport in v1
            // (see docs/design/mcp-and-subagents.md). `McpServer` is
            // `#[non_exhaustive]`, so the fallback arm also covers any future
            // transport variant.
            McpServer::Http(http) => warn!(
                session_id,
                server = %http.name,
                "session/new: mcp http transport is not yet supported; server skipped"
            ),
            McpServer::Sse(sse) => warn!(
                session_id,
                server = %sse.name,
                "session/new: mcp sse transport is not yet supported; server skipped"
            ),
            other => warn!(
                session_id,
                ?other,
                "session/new: unsupported mcp server transport; server skipped"
            ),
        }
    }
}
