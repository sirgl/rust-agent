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
    AgentCapabilities, AuthenticateRequest, AuthenticateResponse, CancelNotification, ContentBlock,
    Implementation, InitializeRequest, InitializeResponse, McpServer, NewSessionRequest,
    NewSessionResponse, PromptRequest, PromptResponse, SessionId, StopReason as AcpStopReason,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{Agent, Client, ConnectTo, on_receive_notification, on_receive_request};
use agent_core::{
    CancellationToken, NextTurnService, SessionState, StopReason, ToolDescriptor, ToolRegistry,
    TurnEngine,
};
use mcp_client::stdio::StdioMcpConnection;
use mcp_client::{register_mcp_tools, McpConnection};
use tokio::sync::Mutex as AsyncMutex;
use tracing::{debug, info, warn};

use crate::client::AcpClientAccess;
use crate::config::AgentConfig;
use crate::sink::AcpUpdateSink;

/// Factory that produces a fresh [`NextTurnService`] for each prompt turn.
///
/// A fresh instance per turn keeps stateful backends (e.g. the deterministic
/// replay backend, which consumes scripted rounds) well-behaved across multiple
/// prompts in the same session.
pub type NextTurnFactory = Arc<dyn Fn() -> Arc<dyn NextTurnService> + Send + Sync>;

/// Dependencies required to build the agent connection.
#[derive(Clone)]
pub struct AgentDeps {
    /// Produces the decision-layer service for each turn.
    pub next_turn_factory: NextTurnFactory,
    /// Tools exposed to the turn engine.
    pub tools: ToolRegistry,
    /// Static configuration (identity, system prompt).
    pub config: AgentConfig,
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
    let cancel_sessions = sessions.clone();

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
                    .agent_capabilities(AgentCapabilities::default())
                    .agent_info(Implementation::new(
                        init_config.name.clone(),
                        init_config.version.clone(),
                    ));
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
            async move |req: NewSessionRequest, responder, _cx| {
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

                let entry = Arc::new(SessionEntry {
                    state: Arc::new(AsyncMutex::new(state)),
                    cancel: StdMutex::new(CancellationToken::new()),
                    tools,
                });
                new_sessions
                    .lock()
                    .expect("sessions mutex poisoned")
                    .insert(session_id.clone(), entry);

                info!(session_id, "session/new");
                responder.respond(NewSessionResponse::new(SessionId::new(session_id)))
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

                // Fresh cancellation token for this turn; stored so session/cancel
                // can reach both the decision stream and running tools.
                let cancel = CancellationToken::new();
                *entry.cancel.lock().expect("cancel mutex poisoned") = cancel.clone();

                let user_text = prompt_text(&req.prompt);
                let factory = prompt_factory.clone();
                let tools = entry.tools.clone();
                let acp_session_id = req.session_id.clone();

                // The turn must run off the event loop so the sink's blocking
                // permission requests do not deadlock the connection.
                cx.spawn({
                    let cx = cx.clone();
                    async move {
                        let next_turn = factory();
                        let client = Arc::new(AcpClientAccess::new(
                            cx.clone(),
                            acp_session_id.clone(),
                        ));
                        let engine = TurnEngine::new(next_turn, tools).with_client(client);
                        let mut sink = AcpUpdateSink::new(cx, acp_session_id);

                        let mut guard = entry.state.lock().await;
                        guard.push_user_text(user_text);
                        let result = engine.run_prompt(&mut guard, &mut sink, &cancel).await;
                        drop(guard);

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
