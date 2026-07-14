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
pub mod config;
pub mod local_client;
pub mod selection;
pub mod sink;

use std::sync::Arc;

use agent_client_protocol::{Client, ConnectTo, Stdio};
use agent_core::{NextTurnService, StopReason, ToolRegistry, TurnEvent};
use mcp_client::{register_mcp_tools, McpConnection};
use orchestrated::{uniform_resolver, OrchestrateTool};
use subagents::SubagentTool;
use turn_replay::ReplayTurnService;

pub use agent::{build_agent, AgentDeps, NextTurnFactory};
pub use config::AgentConfig;
pub use local_client::LocalClientAccess;
pub use selection::ModelSelection;
pub use sink::AcpUpdateSink;

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

/// Build a [`NextTurnFactory`] backed by the real Anthropic Messages API, if a
/// usable [`turn_anthropic::AnthropicConfig`] can be resolved.
///
/// Configuration is resolved via [`turn_anthropic::AnthropicConfig::from_env`],
/// which checks `ANTHROPIC_API_KEY` first and otherwise falls back to the
/// default `token.properties` key file (see
/// [`turn_anthropic::load_default_api_key`]) — the key itself is never logged.
/// Returns `None` (letting callers fall back to [`canned_replay_factory`])
/// when no key can be resolved from either source.
#[must_use]
pub fn anthropic_factory() -> Option<NextTurnFactory> {
    let config = turn_anthropic::AnthropicConfig::from_env()?;
    Some(Arc::new(move |selection| {
        let mut turn_config = config.clone();
        turn_config.model = selection.model.clone();
        turn_config.effort = Some(selection.effort);
        let service: Arc<dyn NextTurnService> =
            Arc::new(turn_anthropic::AnthropicTurnService::new(turn_config));
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

/// Assemble the default dependencies for the binary: the real Anthropic
/// backend when an API key can be resolved (env var or the default
/// `token.properties` file — see [`anthropic_factory`]), otherwise the
/// deterministic replay backend, plus the built-in filesystem/terminal tools,
/// the subagent tool, and environment-derived configuration.
#[must_use]
pub fn default_deps() -> AgentDeps {
    let config = AgentConfig::from_env();
    let (next_turn_factory, backend) = match anthropic_factory() {
        Some(factory) => (factory, "anthropic"),
        None => (canned_replay_factory(), "replay"),
    };
    tracing::info!(backend, "acp-agent: selected default decision backend");
    let tools = build_registry(&config, &next_turn_factory);
    AgentDeps {
        next_turn_factory,
        tools,
        config,
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
