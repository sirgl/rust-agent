//! Integration tests for ACP session modes (Step 14): `session/new` advertises
//! `chat`/`orchestrate`, `session/set_mode` switches the current mode, and an
//! unknown mode id is rejected. Driven through an in-process fake client with
//! the deterministic replay backend (no network).

use std::sync::Arc;

use agent_client_protocol::schema::v1::{
    InitializeRequest, NewSessionRequest, SetSessionModeRequest,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::Client;
use agent_core::{NextTurnService, ToolRegistry, TurnEvent};

use acp_agent::{build_agent, AgentConfig, AgentDeps, NextTurnFactory};

/// Build [`AgentDeps`] with an empty replay script and the given config.
fn deps_with_config(config: AgentConfig) -> AgentDeps {
    let factory: NextTurnFactory = Arc::new(move |_selection| {
        let service: Arc<dyn NextTurnService> =
            Arc::new(turn_replay::ReplayTurnService::new(Vec::<Vec<TurnEvent>>::new()));
        service
    });
    AgentDeps {
        next_turn_factory: factory,
        tools: ToolRegistry::new(),
        config,
    }
}

#[tokio::test]
async fn new_session_advertises_both_modes_with_chat_default() {
    let agent = build_agent(deps_with_config(AgentConfig::default()));

    let (current, available): (String, Vec<String>) = Client
        .builder()
        .name("test-client")
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let new_session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            let modes = new_session.modes.expect("modes advertised");
            let current = modes.current_mode_id.0.to_string();
            let available = modes
                .available_modes
                .iter()
                .map(|m| m.id.0.to_string())
                .collect();
            Ok((current, available))
        })
        .await
        .expect("connection completed");

    assert_eq!(current, "chat");
    assert!(available.contains(&"chat".to_string()));
    assert!(available.contains(&"orchestrate".to_string()));
}

#[tokio::test]
async fn orchestrated_flag_makes_orchestrate_the_initial_mode() {
    let config = AgentConfig {
        enable_orchestrated: true,
        ..AgentConfig::default()
    };
    let agent = build_agent(deps_with_config(config));

    let current: String = Client
        .builder()
        .name("test-client")
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let new_session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            let modes = new_session.modes.expect("modes advertised");
            Ok(modes.current_mode_id.0.to_string())
        })
        .await
        .expect("connection completed");

    assert_eq!(current, "orchestrate");
}

#[tokio::test]
async fn set_mode_switches_to_orchestrate() {
    let agent = build_agent(deps_with_config(AgentConfig::default()));

    let ok: bool = Client
        .builder()
        .name("test-client")
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let new_session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            let outcome = cx
                .send_request(SetSessionModeRequest::new(
                    new_session.session_id,
                    "orchestrate",
                ))
                .block_task()
                .await;
            Ok(outcome.is_ok())
        })
        .await
        .expect("connection completed");

    assert!(ok, "set_mode(orchestrate) should succeed");
}

#[tokio::test]
async fn set_mode_rejects_unknown_mode() {
    let agent = build_agent(deps_with_config(AgentConfig::default()));

    let rejected: bool = Client
        .builder()
        .name("test-client")
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let new_session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            let outcome = cx
                .send_request(SetSessionModeRequest::new(
                    new_session.session_id,
                    "bogus-mode",
                ))
                .block_task()
                .await;
            Ok(outcome.is_err())
        })
        .await
        .expect("connection completed");

    assert!(rejected, "unknown mode id should be rejected");
}
