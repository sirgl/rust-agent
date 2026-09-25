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
use std::sync::{Arc, Mutex as StdMutex};

use agent_client_protocol::schema::v1::{
    AgentCapabilities, AuthenticateRequest, AuthenticateResponse, AvailableCommand,
    AvailableCommandsUpdate, CancelNotification, Content, ContentBlock, ContentChunk,
    Implementation, InitializeRequest, InitializeResponse, LoadSessionRequest, LoadSessionResponse,
    McpServer, NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse, SessionId,
    SessionMode, SessionModeId, SessionModeState, SessionNotification, SessionUpdate,
    SetSessionConfigOptionRequest, SetSessionConfigOptionResponse, SetSessionModeRequest,
    SetSessionModeResponse, StopReason as AcpStopReason, TextContent, ToolCall, ToolCallContent,
    ToolCallStatus as AcpToolCallStatus, ToolCallUpdate, ToolCallUpdateFields,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{
    on_receive_notification, on_receive_request, Agent, Client, ConnectTo, JsonRpcRequest,
    JsonRpcResponse,
};
use agent_core::{
    EngineOutput, HistoryEntry, InMemorySessionStore, NextTurnService, SessionRecord, SessionState,
    SessionStore, StopReason, ToolDescriptor, ToolRegistry, TurnEngine, TurnInbox, UpdateSink,
};
use mcp_client::stdio::StdioMcpConnection;
use mcp_client::{register_mcp_tools, McpConnection};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as AsyncMutex;
use tracing::{debug, info, warn};

use crate::commands::{parse_command, run_command};
use crate::config::AgentConfig;
use crate::local_client::LocalClientAccess;
use crate::selection::ModelSelection;
use crate::sink::{AcpUpdateSink, GenerationScopedSink};
use crate::turn_coordinator::TurnCoordinator;
use crate::usage::USAGE_COMMAND;

/// The mode id for the default plain-chat session mode.
pub const CHAT_MODE_ID: &str = "chat";
/// The mode id for the orchestrated-execution session mode.
pub const ORCHESTRATE_MODE_ID: &str = "orchestrate";

/// Build the ACP [`SessionModeState`] advertised on `session/new`, with the
/// given `current` mode id and both `chat`/`orchestrate` modes available.
fn session_mode_state(current: &str) -> SessionModeState {
    SessionModeState::new(
        SessionModeId::new(current.to_string()),
        vec![
            SessionMode::new(SessionModeId::new(CHAT_MODE_ID), "Chat")
                .description("Plain streaming chat: a single turn engine answers the prompt."),
            SessionMode::new(SessionModeId::new(ORCHESTRATE_MODE_ID), "Orchestrate").description(
                "Orchestrated execution: an orchestrator drives sub-agents to accomplish the goal.",
            ),
        ],
    )
}

/// Whether `mode_id` is one of the advertised session modes.
fn is_known_mode(mode_id: &str) -> bool {
    matches!(mode_id, CHAT_MODE_ID | ORCHESTRATE_MODE_ID)
}

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
    /// Optional provider-neutral observer attached to every turn engine so the
    /// LLM inspector captures the typed request snapshot of every decision
    /// round, for any backend. `None` disables typed capture.
    pub turn_observer: Option<Arc<dyn agent_core::TurnObserver>>,
    /// Optional base URL of the running LLM-inspector UI (e.g.
    /// `http://127.0.0.1:7878`). When set, `session/new` greets each session
    /// with a per-session inspector link as its first message. `None` disables
    /// the greeting.
    pub inspector_base_url: Option<String>,
}

impl AgentDeps {
    /// Build dependencies with observability seams disabled (`turn_observer` and
    /// `inspector_base_url` both `None`).
    ///
    /// A convenience for tests and callers that do not run the inspector, so
    /// adding observability fields does not force every construction site to
    /// mention them.
    #[must_use]
    pub fn new(
        next_turn_factory: NextTurnFactory,
        tools: ToolRegistry,
        config: AgentConfig,
        store: Arc<dyn SessionStore>,
    ) -> Self {
        Self {
            next_turn_factory,
            tools,
            config,
            store,
            turn_observer: None,
            inspector_base_url: None,
        }
    }
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

/// Publish a snapshot of the current session state to the observability seam
/// (the LLM inspector), so the debug UI can show the live per-session state.
///
/// A no-op when no observer is attached. Must be called whenever a session's
/// persisted state changes (creation, after a turn, or on load).
fn publish_session_state(
    observer: &Option<Arc<dyn agent_core::TurnObserver>>,
    state: &SessionState,
    mode: Option<&str>,
) {
    if let Some(observer) = observer {
        let mut snapshot = agent_core::SessionStateSnapshot::from_state(state);
        if let Some(mode) = mode {
            snapshot = snapshot.with_mode(mode);
        }
        observer.on_session_state(&snapshot);
    }
}

/// Wire method name for the steering extension request.
///
/// ACP extension methods must start with `_`.
const STEERING_METHOD: &str = "_session/steering";

/// Parameters for the `_session/steering` request.
#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "_session/steering", response = SteeringResponse)]
struct SteeringRequest {
    /// The session to steer.
    #[serde(rename = "sessionId")]
    session_id: SessionId,
    /// The message to inject, as prompt content blocks.
    #[serde(default)]
    prompt: Vec<ContentBlock>,
}

/// Result of a `_session/steering` request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
enum SteeringOutcome {
    Injected,
    StartedNewTurn,
}

/// Response acknowledging a `_session/steering` request.
#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcResponse)]
struct SteeringResponse {
    outcome: SteeringOutcome,
}

/// Per-session runtime state shared between handlers.
struct SessionEntry {
    /// The typed session state, guarded for exclusive access during a turn.
    state: Arc<AsyncMutex<SessionState>>,
    /// Single source of truth for active-turn ownership and cancellation.
    turns: Arc<TurnCoordinator>,
    /// This session's tool registry: the base tools plus any tools
    /// discovered from the session's `mcp_servers` at `session/new` time.
    tools: ToolRegistry,
    /// The current ACP session mode (`chat` or `orchestrate`), switchable via
    /// `session/set_mode`.
    mode: StdMutex<SessionModeId>,
    /// Current model and effort selection for this session.
    selection: StdMutex<ModelSelection>,
    /// Steering inbox: user messages injected via [`STEERING_METHOD`] and drained
    /// by the running turn between decision rounds.
    inbox: TurnInbox,
    /// A one-shot greeting (e.g. the per-session LLM-inspector link) to be
    /// flushed as the first agent message *inside the first prompt turn*.
    ///
    /// Clients (notably JetBrains) render `session/update` notifications only
    /// while a prompt turn is active, so a greeting sent during `session/new`
    /// is dropped. Deferring it to the first turn makes it actually visible.
    pending_welcome: StdMutex<Option<String>>,
    /// Long-lived orchestrator history and step context, initialized lazily on
    /// the first prompt in orchestrate mode and retained across mode switches.
    orchestration: AsyncMutex<Option<orchestrated::PipelineSession>>,
}

/// Shared registry of active sessions, keyed by session id string.
type Sessions = Arc<StdMutex<HashMap<String, Arc<SessionEntry>>>>;

/// Build the ACP agent connection.
///
/// Without the `unstable_protocol_v2` Cargo feature, the returned connection
/// supports ACP v1 only. With the feature enabled, initialization is routed to
/// the highest compatible v1 or v2 implementation requested by the client.
#[cfg(not(feature = "unstable_protocol_v2"))]
pub fn build_agent(deps: AgentDeps) -> impl ConnectTo<Client> {
    build_agent_v1(deps)
}

/// Build the dual-protocol ACP agent connection.
#[cfg(feature = "unstable_protocol_v2")]
pub fn build_agent(deps: AgentDeps) -> impl ConnectTo<Client> {
    Agent
        .protocol_router()
        .with_v1(build_agent_v1(deps.clone()))
        .with_v2(crate::agent_v2::build_agent_v2(deps))
}

/// Build the ACP v1 implementation used directly or by the protocol router.
fn build_agent_v1(deps: AgentDeps) -> impl ConnectTo<Client> {
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
    let set_mode_sessions = sessions.clone();
    let config_sessions = sessions.clone();
    let inject_sessions = sessions.clone();
    let inject_factory = deps.next_turn_factory.clone();
    let prompt_observer = deps.turn_observer.clone();
    let inject_observer = deps.turn_observer.clone();
    let new_session_observer = deps.turn_observer.clone();
    let load_session_observer = deps.turn_observer.clone();
    let new_session_inspector_url = deps.inspector_base_url.clone();
    let new_store = deps.store.clone();
    let prompt_store = deps.store.clone();
    let inject_store = deps.store.clone();
    let load_sessions = sessions.clone();
    let load_session_tools = deps.tools.clone();
    let load_session_config = deps.config.clone();
    let load_store = deps.store.clone();

    // The initial mode a new session starts in: `orchestrate` when the
    // orchestrated toggle is set (backward-compatible with `ACP_ORCHESTRATED`),
    // otherwise plain `chat`.
    let initial_mode_id = if deps.config.enable_orchestrated {
        ORCHESTRATE_MODE_ID
    } else {
        CHAT_MODE_ID
    };

    Agent
        .builder()
        .name(deps.config.name.clone())
        // initialize: negotiate protocol version, advertise capabilities.
        .on_receive_request(
            async move |req: InitializeRequest, responder, _cx| {
                let negotiated = ProtocolVersion::V1;
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
                    .meta(serde_json::Map::from_iter([(
                        "steering".to_string(),
                        serde_json::json!({ "supported": true }),
                    )]));
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
                let session_id = new_session_id();

                if let Err(err) = LocalClientAccess::with_cwd(&req.cwd) {
                    warn!(session_id, cwd = %req.cwd.display(), %err, "session/new invalid cwd");
                    return responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params().data(err.to_string()),
                    );
                }

                let mut tools = new_session_tools.clone();
                connect_session_mcp_servers(&session_id, &req.mcp_servers, &mut tools).await;

                let mut state = SessionState::new(session_id.clone());
                state.set_workspace(
                    req.cwd.to_string_lossy().into_owned(),
                    req.additional_directories
                        .iter()
                        .map(|path| path.to_string_lossy().into_owned()),
                );
                state.system_prompt = new_session_config.system_prompt.clone();
                state.available_tools = tool_descriptors(&tools);

                let selection = ModelSelection::for_config(&new_session_config);
                let config_options = selection.config_options();

                // Persist the freshly-created core state so the session exists
                // in the store from the outset (best-effort).
                let record = state.to_record();
                persist_record(&new_store, &session_id, &record).await;

                // Publish the initial session state to the debug UI.
                publish_session_state(&new_session_observer, &state, Some(initial_mode_id));

                // Prepare the per-session LLM-inspector greeting (when the
                // inspector is running) and defer it to the first prompt turn,
                // where clients actually render `session/update` messages.
                let pending_welcome = new_session_inspector_url
                    .as_ref()
                    .map(|base_url| inspector_welcome_text(base_url, &session_id));

                let entry = Arc::new(SessionEntry {
                    state: Arc::new(AsyncMutex::new(state)),
                    turns: TurnCoordinator::new(session_id.clone()),
                    tools,
                    mode: StdMutex::new(SessionModeId::new(initial_mode_id)),
                    selection: StdMutex::new(selection),
                    inbox: Arc::new(StdMutex::new(Vec::new())),
                    pending_welcome: StdMutex::new(pending_welcome),
                    orchestration: AsyncMutex::new(None),
                });
                new_sessions
                    .lock()
                    .expect("sessions mutex poisoned")
                    .insert(session_id.clone(), entry);

                info!(
                    session_id,
                    cwd = %req.cwd.display(),
                    initial_mode = initial_mode_id,
                    "session/new"
                );

                // Advertise the built-in slash commands this agent supports, so
                // clients can surface them (e.g. `/usage`) and route them back as
                // prompts.
                if let Err(err) = cx.send_notification(available_commands_notification(&session_id))
                {
                    warn!(session_id, %err, "failed to advertise available commands");
                }

                // The per-session LLM-inspector greeting is deferred to the
                // first prompt turn (see `SessionEntry::pending_welcome`),
                // because clients render `session/update` messages only while a
                // turn is active.
                responder.respond(
                    NewSessionResponse::new(SessionId::new(session_id))
                        .modes(session_mode_state(initial_mode_id))
                        .config_options(config_options),
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
                if let Err(err) = LocalClientAccess::with_cwd(&req.cwd) {
                    warn!(session_id, cwd = %req.cwd.display(), %err, "session/load invalid cwd");
                    return responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params().data(err.to_string()),
                    );
                }
                let mut state = SessionState::from_record(record, available_tools);
                state.set_workspace(
                    req.cwd.to_string_lossy().into_owned(),
                    req.additional_directories
                        .iter()
                        .map(|path| path.to_string_lossy().into_owned()),
                );
                let restored_plan =
                    (!state.todo_list.items.is_empty()).then(|| state.todo_list.to_plan_update());

                // The current ACP request is authoritative for execution
                // context; persist it before exposing the loaded session.
                persist_record(&load_store, &session_id, &state.to_record()).await;

                // Publish the rehydrated session state to the debug UI.
                publish_session_state(&load_session_observer, &state, Some(CHAT_MODE_ID));

                let selection = ModelSelection::for_config(&load_session_config);
                let config_options = selection.config_options();

                let entry = Arc::new(SessionEntry {
                    state: Arc::new(AsyncMutex::new(state)),
                    turns: TurnCoordinator::new(session_id.clone()),
                    tools,
                    mode: StdMutex::new(SessionModeId::new(CHAT_MODE_ID)),
                    selection: StdMutex::new(selection),
                    inbox: Arc::new(StdMutex::new(Vec::new())),
                    pending_welcome: StdMutex::new(None),
                    orchestration: AsyncMutex::new(None),
                });
                load_sessions
                    .lock()
                    .expect("sessions mutex poisoned")
                    .insert(session_id.clone(), entry);

                info!(
                    session_id,
                    cwd = %req.cwd.display(),
                    entries = history.len(),
                    "session/load"
                );

                // Replay the stored conversation to the client in order, as
                // `session/update` notifications. Unknown/ingested entries are
                // skipped (best-effort) rather than failing the load.
                let acp_session_id = SessionId::new(session_id.clone());
                if let Some(plan) = restored_plan {
                    let notification = SessionNotification::new(
                        acp_session_id.clone(),
                        SessionUpdate::Plan(crate::sink::map_plan(&plan)),
                    );
                    if let Err(err) = cx.send_notification(notification) {
                        warn!(session_id, %err, "failed to restore todo plan");
                    }
                }
                for entry in &history.entries {
                    if let Some(update) = history_entry_to_update(entry) {
                        let notification = SessionNotification::new(acp_session_id.clone(), update);
                        if let Err(err) = cx.send_notification(notification) {
                            warn!(session_id, %err, "failed to replay history entry");
                        }
                    }
                }

                // Re-advertise the built-in slash commands so the reloaded
                // session exposes the same commands as a fresh one.
                if let Err(err) = cx.send_notification(available_commands_notification(&session_id))
                {
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
                // Establish output ownership before any turn-scoped greeting,
                // command response, model output, or permission request.
                let lease = entry.turns.begin_turn().await;
                let cancel = lease.cancellation_token();
                let selection = entry
                    .selection
                    .lock()
                    .expect("selection mutex poisoned")
                    .clone();
                let working_directory = {
                    entry
                        .state
                        .lock()
                        .await
                        .working_directory()
                        .map(str::to_owned)
                };
                let Some(working_directory) = working_directory else {
                    lease.finalize_if_current(|| async {}).await;
                    return responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params()
                            .data(format!("session {session_id} has no working directory")),
                    );
                };
                let client = match LocalClientAccess::with_elicitation_and_cwd(
                    cx.clone(),
                    req.session_id.clone(),
                    &working_directory,
                ) {
                    Ok(client) => Arc::new(client),
                    Err(err) => {
                        lease.finalize_if_current(|| async {}).await;
                        warn!(session_id, cwd = working_directory, %err, "session/prompt invalid cwd");
                        return responder.respond_with_error(
                            agent_client_protocol::Error::invalid_params().data(err.to_string()),
                        );
                    }
                };
                let acp_sink = AcpUpdateSink::new(cx.clone(), req.session_id.clone())
                    .with_yolo(selection.permission.is_yolo());
                let mut sink = GenerationScopedSink::new(acp_sink, lease.clone());

                // Flush any deferred one-shot greeting (e.g. the per-session
                // LLM-inspector link) as the first agent message of this turn.
                // Sent here, inside an active prompt turn, it is actually
                // rendered by clients (unlike a `session/new`-time message).
                let pending_welcome = {
                    entry
                        .pending_welcome
                        .lock()
                        .expect("welcome mutex poisoned")
                        .take()
                };
                if let Some(welcome) = pending_welcome {
                    if let Err(err) = sink.send(EngineOutput::MessageChunk(welcome)).await {
                        warn!(session_id, %err, "failed to send inspector welcome message");
                    }
                }

                // Intercept built-in slash commands (e.g. `/usage`): produce the
                // command's display text as a plain agent message, without
                // running a model turn.
                if let Some(command) = parse_command(&user_text) {
                    let model = selection.model.clone();
                    let summary = {
                        let guard = entry.state.lock().await;
                        run_command(command, &guard, &model)
                    };
                    if let Err(err) = sink.send(EngineOutput::MessageChunk(summary)).await {
                        warn!(session_id, %err, "failed to send /usage summary");
                    }
                    lease.finalize_if_current(|| async {}).await;
                    return responder.respond(PromptResponse::new(AcpStopReason::EndTurn));
                }

                let factory = prompt_factory.clone();
                let require_submit_result = prompt_require_submit_result;
                let thought_interval = prompt_thought_interval;
                let observer = prompt_observer.clone();
                let tools = entry.tools.clone();

                // Snapshot the session's current mode so we can branch the turn.
                let orchestrate = {
                    let mode = entry.mode.lock().expect("mode mutex poisoned");
                    mode.0.as_ref() == ORCHESTRATE_MODE_ID
                };

                let inbox = entry.inbox.clone();
                let store = prompt_store.clone();

                // The turn must run off the event loop so the sink's blocking
                // permission requests do not deadlock the connection.
                cx.spawn({
                    async move {
                        let (result, record, final_state, final_mode) = if orchestrate {
                            // Orchestrate mode: the prompt text is the goal; run
                            // the shared orchestration pipeline, streaming through
                            // the same sink and sharing the turn's cancel token.
                            debug!(session_id, "session/prompt: orchestrate mode");
                            // Adapt the selection-aware factory into the zero-arg
                            // `BackendFactory` the orchestration pipeline expects.
                            let sel = selection.clone();
                            let sel_factory = factory.clone();
                            let backend_factory: orchestrated::BackendFactory =
                                Arc::new(move || sel_factory(&sel));
                            let prior_state = entry.state.lock().await.clone();
                            let mut runtime_guard = entry.orchestration.lock().await;
                            if runtime_guard.is_none() {
                                let backend = (backend_factory)();
                                let resolver =
                                    orchestrated::uniform_resolver(backend_factory.clone());
                                let mut pipeline =
                                    orchestrated::PipelineSession::new_observed_with_inbox(
                                        &user_text,
                                        backend,
                                        resolver,
                                        tools,
                                        observer.clone(),
                                        agent_core::AgentPath::root().child("orchestrator"),
                                        Some(session_id.clone()),
                                        inbox.clone(),
                                        Some(client),
                                    );
                                pipeline.inherit_state(&prior_state);
                                *runtime_guard = Some(pipeline);
                            }
                            let pipeline = runtime_guard
                                .as_mut()
                                .expect("orchestration runtime initialized");
                            let result = pipeline.run_prompt(&user_text, &mut sink, &cancel).await;
                            let orchestrated_state = pipeline.session().clone();
                            drop(runtime_guard);
                            {
                                let mut shared_state = entry.state.lock().await;
                                shared_state.workspace_roots =
                                    orchestrated_state.workspace_roots.clone();
                                shared_state.history = orchestrated_state.history.clone();
                                shared_state.usage = orchestrated_state.usage;
                                shared_state.todo_list = orchestrated_state.todo_list.clone();
                            }
                            (
                                result,
                                orchestrated_state.to_record(),
                                orchestrated_state,
                                ORCHESTRATE_MODE_ID,
                            )
                        } else {
                            // Chat mode: seed the turn's inbox with the user's
                            // prompt; the engine drains it (together with any later
                            // steering injections) each round.
                            inbox.lock().expect("inbox mutex poisoned").push(user_text);
                            let next_turn = factory(&selection);
                            let mut engine = TurnEngine::new(next_turn, tools)
                                .with_client(client)
                                .with_inbox(inbox);
                            if require_submit_result {
                                engine = engine.require_terminal_tool();
                            }
                            if thought_interval > 0 {
                                engine = engine.with_thought_interval(thought_interval);
                            }
                            if let Some(observer) = &observer {
                                engine = engine.with_observer(observer.clone());
                            }
                            let mut guard = entry.state.lock().await;
                            let result = engine.run_prompt(&mut guard, &mut sink, &cancel).await;
                            let record = guard.to_record();
                            let final_state = guard.clone();
                            (result, record, final_state, CHAT_MODE_ID)
                        };

                        lease
                            .finalize_if_current(|| async {
                                publish_session_state(&observer, &final_state, Some(final_mode));
                                persist_record(&store, &session_id, &record).await;
                            })
                            .await;

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
        // _session/steering: inject a user message into the running
        // turn, or start a fresh turn if none is running.
        .on_receive_request(
            async move |req: SteeringRequest, responder, cx| {
                let session_id = req.session_id.0.to_string();
                let Some(entry) = inject_sessions
                    .lock()
                    .expect("sessions mutex poisoned")
                    .get(&session_id)
                    .cloned()
                else {
                    warn!(
                        session_id,
                        method = STEERING_METHOD,
                        "steering for unknown session"
                    );
                    return responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params()
                            .data(format!("unknown session: {session_id}")),
                    );
                };

                let text = prompt_text(&req.prompt);

                // If a turn is already running, the engine will drain the inbox
                // on its next round: the message is steered into the current
                // prompt turn.
                if entry.turns.with_current_turn(|| {
                    entry
                        .inbox
                        .lock()
                        .expect("inbox mutex poisoned")
                        .push(text.clone());
                }) {
                    info!(
                        session_id,
                        method = STEERING_METHOD,
                        "steering: injected into running turn"
                    );
                    return responder.respond(SteeringResponse {
                        outcome: SteeringOutcome::Injected,
                    });
                }

                // Otherwise the previous turn already stopped; start a fresh turn
                // so the user perceives a new prompt turn beginning.
                info!(
                    session_id,
                    method = STEERING_METHOD,
                    "steering: starting new turn"
                );
                let lease = entry.turns.begin_turn().await;
                let cancel = lease.cancellation_token();

                let selection = entry
                    .selection
                    .lock()
                    .expect("selection mutex poisoned")
                    .clone();
                let working_directory = {
                    entry
                        .state
                        .lock()
                        .await
                        .working_directory()
                        .map(str::to_owned)
                };
                let Some(working_directory) = working_directory else {
                    lease.finalize_if_current(|| async {}).await;
                    return responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params()
                            .data(format!("session {session_id} has no working directory")),
                    );
                };
                let client = match LocalClientAccess::with_elicitation_and_cwd(
                    cx.clone(),
                    req.session_id.clone(),
                    &working_directory,
                ) {
                    Ok(client) => Arc::new(client),
                    Err(err) => {
                        lease.finalize_if_current(|| async {}).await;
                        warn!(session_id, cwd = working_directory, %err, "_session/steering invalid cwd");
                        return responder.respond_with_error(
                            agent_client_protocol::Error::invalid_params().data(err.to_string()),
                        );
                    }
                };

                let factory = inject_factory.clone();
                let require_submit_result = inject_require_submit_result;
                let thought_interval = inject_thought_interval;
                let observer = inject_observer.clone();
                let tools = entry.tools.clone();
                let inbox = entry.inbox.clone();
                let acp_session_id = req.session_id.clone();
                let turn_entry = entry.clone();
                let store = inject_store.clone();
                let orchestrate = {
                    let mode = entry.mode.lock().expect("mode mutex poisoned");
                    mode.0.as_ref() == ORCHESTRATE_MODE_ID
                };

                cx.spawn({
                    let cx = cx.clone();
                    async move {
                        let next_turn = factory(&selection);
                        let acp_sink = AcpUpdateSink::new(cx, acp_session_id)
                            .with_yolo(selection.permission.is_yolo());
                        let mut sink = GenerationScopedSink::new(acp_sink, lease.clone());

                        let (result, record, final_state, final_mode) = if orchestrate {
                            let sel = selection.clone();
                            let sel_factory = factory.clone();
                            let backend_factory: orchestrated::BackendFactory =
                                Arc::new(move || sel_factory(&sel));
                            let prior_state = turn_entry.state.lock().await.clone();
                            let mut runtime_guard = turn_entry.orchestration.lock().await;
                            if runtime_guard.is_none() {
                                let resolver =
                                    orchestrated::uniform_resolver(backend_factory.clone());
                                let mut pipeline =
                                    orchestrated::PipelineSession::new_observed_with_inbox(
                                        &text,
                                        next_turn,
                                        resolver,
                                        tools,
                                        observer.clone(),
                                        agent_core::AgentPath::root().child("orchestrator"),
                                        Some(session_id.clone()),
                                        inbox.clone(),
                                        Some(client),
                                    );
                                pipeline.inherit_state(&prior_state);
                                *runtime_guard = Some(pipeline);
                            }
                            let pipeline = runtime_guard
                                .as_mut()
                                .expect("orchestration runtime initialized");
                            let result = pipeline.run_prompt(&text, &mut sink, &cancel).await;
                            let orchestrated_state = pipeline.session().clone();
                            drop(runtime_guard);
                            {
                                let mut shared_state = turn_entry.state.lock().await;
                                shared_state.workspace_roots =
                                    orchestrated_state.workspace_roots.clone();
                                shared_state.history = orchestrated_state.history.clone();
                                shared_state.usage = orchestrated_state.usage;
                                shared_state.todo_list = orchestrated_state.todo_list.clone();
                            }
                            (
                                result,
                                orchestrated_state.to_record(),
                                orchestrated_state,
                                ORCHESTRATE_MODE_ID,
                            )
                        } else {
                            inbox.lock().expect("inbox mutex poisoned").push(text);
                            let mut engine = TurnEngine::new(next_turn, tools)
                                .with_client(client)
                                .with_inbox(inbox);
                            if require_submit_result {
                                engine = engine.require_terminal_tool();
                            }
                            if thought_interval > 0 {
                                engine = engine.with_thought_interval(thought_interval);
                            }
                            if let Some(observer) = &observer {
                                engine = engine.with_observer(observer.clone());
                            }
                            let mut guard = turn_entry.state.lock().await;
                            let result = engine.run_prompt(&mut guard, &mut sink, &cancel).await;
                            let record = guard.to_record();
                            let final_state = guard.clone();
                            (result, record, final_state, CHAT_MODE_ID)
                        };

                        lease
                            .finalize_if_current(|| async {
                                publish_session_state(&observer, &final_state, Some(final_mode));
                                persist_record(&store, &session_id, &record).await;
                            })
                            .await;

                        if let Err(err) = result {
                            warn!(session_id, %err, "steering turn failed");
                        } else {
                            debug!(session_id, "steering turn finished");
                        }
                        Ok(())
                    }
                })?;

                responder.respond(SteeringResponse {
                    outcome: SteeringOutcome::StartedNewTurn,
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
                    entry.turns.cancel_active();
                } else {
                    warn!(session_id, "session/cancel for unknown session");
                }
                Ok(())
            },
            on_receive_notification!(),
        )
        // session/set_mode: switch the session's current mode.
        .on_receive_request(
            async move |req: SetSessionModeRequest, responder, _cx| {
                let session_id = req.session_id.0.to_string();
                let mode_id = req.mode_id.0.to_string();

                let Some(entry) = set_mode_sessions
                    .lock()
                    .expect("sessions mutex poisoned")
                    .get(&session_id)
                    .cloned()
                else {
                    warn!(session_id, "session/set_mode for unknown session");
                    return responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params()
                            .data(format!("unknown session: {session_id}")),
                    );
                };

                if !is_known_mode(&mode_id) {
                    warn!(session_id, mode = %mode_id, "session/set_mode: unknown mode");
                    return responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params()
                            .data(format!("unknown session mode: {mode_id}")),
                    );
                }

                *entry.mode.lock().expect("mode mutex poisoned") = req.mode_id.clone();
                info!(session_id, mode = %mode_id, "session/set_mode");
                responder.respond(SetSessionModeResponse::new())
            },
            on_receive_request!(),
        )
}

/// Build the first-message text greeting a new session with its per-session
/// LLM-inspector link.
///
/// The link carries a `?session=<id>` filter so the UI opens straight to this
/// session's captured requests. `base_url` is the inspector's origin (e.g.
/// `http://127.0.0.1:7878`).
fn inspector_welcome_text(base_url: &str, session_id: &str) -> String {
    let base = base_url.trim_end_matches('/');
    format!(
        "🔎 LLM Inspector for this session: {base}/?session={session_id}\n\
         Live view of every request this session sends to the model (typed history + provider body).\n\n"
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
        HistoryEntry::Assistant(message) => {
            Some(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                ContentBlock::Text(TextContent::new(blocks_text(&message.content))),
            )))
        }
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
        HistoryEntry::Thinking(_) => {
            // Thinking blocks are preserved for faithful provider replay but have
            // no user-visible `session/update` analog on load; skip them.
            None
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

/// Generate a session id that stays unique across agent processes.
fn new_session_id() -> String {
    format!("session-{}", uuid::Uuid::new_v4())
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
