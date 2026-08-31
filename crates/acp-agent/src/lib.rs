//! # acp-agent
//!
//! The stdio Agent Client Protocol (ACP) server: it implements the agent half
//! of the protocol on top of [`agent_core`]'s [`TurnEngine`](agent_core::TurnEngine),
//! streaming prompt turns to a connected client.
//!
//! The reusable wiring lives in the library so it can be exercised end-to-end
//! by an in-process fake client (see the crate's integration tests); the binary
//! (`main.rs`) is a thin wrapper that boots the stdio transport.
//!
//! The decision layer is injected via a [`NextTurnFactory`], keeping the binary
//! backend-agnostic. For v1 the binary defaults to the deterministic replay
//! backend so a full prompt turn streams end-to-end without a network.

#![warn(missing_docs)]

pub mod agent;
pub mod commands;
pub mod config;
pub mod local_client;
pub mod selection;
pub mod sink;
pub mod turn_coordinator;
pub mod usage;

use std::sync::Arc;

use agent_client_protocol::{Client, ConnectTo, Stdio};
use agent_core::{NextTurnService, StopReason, ToolRegistry, TurnEvent};
use mcp_client::{register_mcp_tools, McpConnection};
use orchestrated::{uniform_resolver, OrchestrateTool};
use subagents::SubagentTool;
use turn_replay::ReplayTurnService;

pub use agent::{build_agent, default_store, AgentDeps, NextTurnFactory};
pub use commands::{parse_command, run_command, AgentCommand};
pub use config::{AgentConfig, DEFAULT_SYSTEM_PROMPT};
pub use local_client::LocalClientAccess;
pub use selection::ModelSelection;
pub use sink::AcpUpdateSink;
pub use turn_coordinator::{TurnCoordinator, TurnLease};
pub use usage::{format_usage, USAGE_COMMAND};
// Re-export the model catalog helpers so downstream tools (e.g. the CLI
// `/usage` command) can price token usage without depending on the backend
// crate directly.
pub use turn_anthropic::{model_spec, ModelSpec};

// Re-export the capability seams so downstream wiring/tests can construct MCP
// connections and subagent tools without depending on the crates directly.
pub use mcp_client::{McpConnection as McpConnectionTrait, McpServerConfig, McpToolDescriptor};

/// Build a [`NextTurnFactory`] that replays a fixed, canned turn.
///
/// Used as the fallback backend for the binary when no Anthropic API key can
/// be resolved (see [`anthropic_factory`]): it lets the server demonstrate a
/// full streaming prompt turn deterministically, without any network access.
#[must_use]
pub fn canned_replay_factory() -> NextTurnFactory {
    Arc::new(|_selection| {
        let service: Arc<dyn NextTurnService> = Arc::new(ReplayTurnService::single(vec![
            TurnEvent::TextDelta("Hello from the replay backend! ".to_string()),
            TurnEvent::TextDelta("This turn was produced without any network.".to_string()),
            TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            },
        ]));
        service
    })
}

/// Build a [`NextTurnFactory`] backed by the Anthropic-shaped Messages API
/// (Claude via Anthropic and/or DeepSeek via its Anthropic-compatible endpoint),
/// if at least one usable API key can be resolved.
///
/// Keys are resolved via [`turn_anthropic::AnthropicConfig::from_env`] /
/// [`turn_anthropic::AnthropicConfig::deepseek_from_env`] (env vars first,
/// then `token.properties`) — the key values themselves are never logged.
/// Returns `None` (letting callers fall back to [`canned_replay_factory`])
/// when neither provider can be configured.
#[must_use]
pub fn anthropic_factory() -> Option<NextTurnFactory> {
    anthropic_factory_with_observer(None)
}

/// Like [`anthropic_factory`], but attaches an optional
/// [`turn_anthropic::RequestObserver`] to every constructed backend so an
/// observability seam (e.g. the LLM inspector) can capture the exact request
/// body of each turn, attributed to its session.
///
/// Each turn is routed by [`turn_anthropic::provider_for`] on the selected
/// model id: DeepSeek models use the DeepSeek key + endpoint, Claude models
/// use the Anthropic key + endpoint.
#[must_use]
pub fn anthropic_factory_with_observer(
    observer: Option<Arc<dyn turn_anthropic::RequestObserver>>,
) -> Option<NextTurnFactory> {
    let anthropic = turn_anthropic::AnthropicConfig::from_env();
    let deepseek = turn_anthropic::AnthropicConfig::deepseek_from_env();
    if anthropic.is_none() && deepseek.is_none() {
        return None;
    }
    Some(Arc::new(move |selection| {
        let mut turn_config = match turn_anthropic::provider_for(&selection.model) {
            turn_anthropic::Provider::DeepSeek => {
                if let Some(cfg) = deepseek.clone() {
                    cfg.with_model(selection.model.clone())
                } else if let Some(cfg) = anthropic.clone() {
                    // DeepSeek selected but only Anthropic is configured —
                    // retarget so the failure mode is a clear upstream 401
                    // rather than silently hitting the wrong host with a
                    // Claude-only key.
                    tracing::warn!(
                        model = %selection.model,
                        "deepseek model selected but DEEPSEEK_API_KEY is not configured"
                    );
                    cfg.with_model(selection.model.clone())
                } else {
                    unreachable!("factory requires at least one provider config")
                }
            }
            turn_anthropic::Provider::Anthropic => {
                if let Some(cfg) = anthropic.clone() {
                    cfg.with_model(selection.model.clone())
                } else if let Some(cfg) = deepseek.clone() {
                    tracing::warn!(
                        model = %selection.model,
                        "anthropic model selected but ANTHROPIC_API_KEY is not configured"
                    );
                    cfg.with_model(selection.model.clone())
                } else {
                    unreachable!("factory requires at least one provider config")
                }
            }
        };
        turn_config.effort = Some(selection.effort);
        let mut service = turn_anthropic::AnthropicTurnService::new(turn_config);
        if let Some(observer) = &observer {
            service = service.with_observer(observer.clone());
        }
        let service: Arc<dyn NextTurnService> = Arc::new(service);
        service
    }))
}

/// Build the base tool registry for a session from `config`: the built-in
/// filesystem/terminal tools plus (when enabled) the subagent delegation tool.
///
/// MCP tools are registered separately via [`register_configured_mcp_tools`]
/// once concrete [`McpConnection`]s have been established, because opening a
/// server connection is an async, fallible operation kept out of this
/// synchronous assembly step.
#[must_use]
pub fn build_registry(config: &AgentConfig, next_turn_factory: &NextTurnFactory) -> ToolRegistry {
    let mut registry = tools_builtin::builtin_registry();
    if config.enable_subagents {
        // The subagent drives its own nested engine using the same decision
        // backend as the parent turn, over a copy of the built-in tools.
        //
        // Subagents remain out of scope for client-driven selection changes
        // in v1, so we build them once with the intentional default selection
        // derived from the agent configuration.
        let selection = ModelSelection::for_config(config);
        let nested_backend = next_turn_factory(&selection);
        let nested_tools = tools_builtin::builtin_registry();
        let subagent = SubagentTool::new(nested_backend, nested_tools)
            .with_max_depth(config.subagent_max_depth);
        registry.register(Arc::new(subagent));
    }
    if config.enable_orchestrate_tool {
        // The `orchestrate` tool lets the chat agent launch the orchestration
        // pipeline itself. It reuses the same decision backend factory for both
        // the orchestrator and (via a uniform resolver) all sub-agent tiers.
        //
        // The orchestration pipeline uses a zero-argument `BackendFactory`, so
        // we adapt the selection-aware factory by binding it to the default
        // selection derived from the agent configuration (client-driven
        // selection changes are out of scope for orchestrated sub-agents in v1).
        let selection = ModelSelection::for_config(config);
        let selection_factory = next_turn_factory.clone();
        let backend_factory: orchestrated::BackendFactory =
            Arc::new(move || selection_factory(&selection));
        let resolver = uniform_resolver(backend_factory.clone());
        let base_tools = tools_builtin::builtin_registry();
        registry.register(Arc::new(OrchestrateTool::new(
            backend_factory,
            resolver,
            base_tools,
        )));
    }
    registry
}

/// Discover and register the tools of an already-connected MCP server into
/// `registry`, namespaced by the server name.
///
/// The binary calls this once per configured server after establishing a
/// concrete [`McpConnection`]; tests call it with an in-memory stub connection.
///
/// # Errors
///
/// Propagates any error from tool discovery (e.g. a server advertising no
/// tools).
pub async fn register_configured_mcp_tools(
    registry: &mut ToolRegistry,
    connection: Arc<dyn McpConnection>,
) -> agent_core::Result<Vec<String>> {
    register_mcp_tools(registry, connection).await
}

/// Assemble the default dependencies for the binary: a real LLM backend when
/// an Anthropic and/or DeepSeek API key can be resolved (env var or the default
/// `token.properties` file — see [`anthropic_factory`]), otherwise the
/// deterministic replay backend, plus the built-in filesystem/terminal tools,
/// the subagent tool, and environment-derived configuration.
#[must_use]
pub fn default_deps() -> AgentDeps {
    default_deps_with_observer(None)
}

/// Like [`default_deps`], but attaches an optional
/// [`turn_anthropic::RequestObserver`] to the LLM backend (when selected),
/// so the LLM inspector can capture the real outgoing requests.
///
/// The observer is only wired into the real Messages backend; the deterministic
/// replay fallback makes no network calls, so there is nothing to observe there.
#[must_use]
pub fn default_deps_with_observer(
    observer: Option<Arc<dyn turn_anthropic::RequestObserver>>,
) -> AgentDeps {
    default_deps_with_observers(observer, None, None)
}

/// Like [`default_deps_with_observer`], but also attaches a provider-neutral
/// [`agent_core::TurnObserver`] to every turn engine (capturing the typed
/// request snapshot for *any* backend, including replay) and records the
/// inspector's base URL so the ACP layer can greet each new session with a
/// per-session inspector link.
#[must_use]
pub fn default_deps_with_observers(
    request_observer: Option<Arc<dyn turn_anthropic::RequestObserver>>,
    turn_observer: Option<Arc<dyn agent_core::TurnObserver>>,
    inspector_base_url: Option<String>,
) -> AgentDeps {
    let mut config = AgentConfig::from_env();
    let has_anthropic = turn_anthropic::AnthropicConfig::from_env().is_some();
    let has_deepseek = turn_anthropic::AnthropicConfig::deepseek_from_env().is_some();
    // If only DeepSeek is configured and the default model still points at a
    // Claude id, retarget so the first turn doesn't 401 against the wrong host.
    if !has_anthropic
        && has_deepseek
        && !turn_anthropic::is_deepseek_model(&config.default_model)
    {
        config.default_model = turn_anthropic::DEFAULT_DEEPSEEK_MODEL.to_string();
    }
    let (next_turn_factory, backend) = match anthropic_factory_with_observer(request_observer) {
        Some(factory) => {
            let backend = match (has_anthropic, has_deepseek) {
                (true, true) => "anthropic+deepseek",
                (true, false) => "anthropic",
                (false, true) => "deepseek",
                (false, false) => "llm",
            };
            (factory, backend)
        }
        None => (canned_replay_factory(), "replay"),
    };
    tracing::info!(backend, "acp-agent: selected default decision backend");
    let tools = build_registry(&config, &next_turn_factory);
    let store = match &config.session_persistence_dir {
        Some(dir) => {
            tracing::info!(dir = %dir.display(), "acp-agent: persisting sessions to disk");
            let store: Arc<dyn agent_core::SessionStore> =
                Arc::new(agent_core::JsonFileSessionStore::new(dir.clone()));
            store
        }
        None => default_store(),
    };
    AgentDeps {
        next_turn_factory,
        tools,
        config,
        store,
        turn_observer,
        inspector_base_url,
    }
}

/// Run the ACP agent over the stdio transport until the connection closes.
///
/// # Errors
///
/// Returns an error if the connection terminates abnormally.
pub async fn run_stdio(deps: AgentDeps) -> Result<(), agent_client_protocol::Error> {
    let agent = build_agent(deps);
    ConnectTo::<Client>::connect_to(agent, Stdio::new()).await
}
