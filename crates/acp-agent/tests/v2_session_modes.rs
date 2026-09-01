#![cfg(feature = "unstable_protocol_v2")]

//! ACP v2 mode tests: modes are ordinary session config options with the
//! standard `mode` category; the removed `session/set_mode` method is unused.

use std::sync::Arc;

use agent_client_protocol::schema::v2::{
    InitializeRequest, NewSessionRequest, SessionConfigId, SessionConfigKind, SessionConfigOption,
    SessionConfigSelectOptions, SessionConfigValueId, SetSessionConfigOptionRequest,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::Client;
use agent_core::{NextTurnService, ToolRegistry, TurnEvent};

use acp_agent::{build_agent, default_store, AgentConfig, AgentDeps, NextTurnFactory};

fn mode_state(options: &[SessionConfigOption]) -> (String, Vec<String>) {
    let option = options
        .iter()
        .find(|option| option.config_id.0.as_ref() == "mode")
        .expect("mode config advertised");
    let SessionConfigKind::Select(select) = &option.kind else {
        panic!("mode must be a select option");
    };
    let SessionConfigSelectOptions::Ungrouped(options) = &select.options else {
        panic!("mode options must be ungrouped");
    };
    (
        select.current_value.0.to_string(),
        options
            .iter()
            .map(|option| option.value.0.to_string())
            .collect(),
    )
}

/// Build [`AgentDeps`] with an empty replay script and the given config.
fn deps_with_config(config: AgentConfig) -> AgentDeps {
    let factory: NextTurnFactory = Arc::new(move |_selection| {
        let service: Arc<dyn NextTurnService> = Arc::new(turn_replay::ReplayTurnService::new(
            Vec::<Vec<TurnEvent>>::new(),
        ));
        service
    });
    AgentDeps {
        next_turn_factory: factory,
        tools: ToolRegistry::new(),
        config,
        store: default_store(),
        turn_observer: None,
        inspector_base_url: None,
    }
}

#[tokio::test]
async fn new_session_advertises_both_modes_with_chat_default() {
    let agent = build_agent(deps_with_config(AgentConfig::default()));

    let (current, available): (String, Vec<String>) = Client
        .v2()
        .name("test-client")
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(
                ProtocolVersion::V2,
                agent_client_protocol::schema::v2::Implementation::new("test-client", "0.1"),
            ))
            .block_task()
            .await?;
            let new_session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            Ok(mode_state(&new_session.config_options))
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
        .v2()
        .name("test-client")
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(
                ProtocolVersion::V2,
                agent_client_protocol::schema::v2::Implementation::new("test-client", "0.1"),
            ))
            .block_task()
            .await?;
            let new_session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            Ok(mode_state(&new_session.config_options).0)
        })
        .await
        .expect("connection completed");

    assert_eq!(current, "orchestrate");
}

#[tokio::test]
async fn mode_config_switches_to_orchestrate() {
    let agent = build_agent(deps_with_config(AgentConfig::default()));

    let ok: bool = Client
        .v2()
        .name("test-client")
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(
                ProtocolVersion::V2,
                agent_client_protocol::schema::v2::Implementation::new("test-client", "0.1"),
            ))
            .block_task()
            .await?;
            let new_session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            let outcome = cx
                .send_request(SetSessionConfigOptionRequest::new(
                    new_session.session_id,
                    SessionConfigId::new("mode"),
                    SessionConfigValueId::new("orchestrate"),
                ))
                .block_task()
                .await;
            Ok(outcome
                .map(|response| mode_state(&response.config_options).0 == "orchestrate")
                .unwrap_or(false))
        })
        .await
        .expect("connection completed");

    assert!(ok, "mode config update should succeed");
}

#[tokio::test]
async fn mode_config_rejects_unknown_mode() {
    let agent = build_agent(deps_with_config(AgentConfig::default()));

    let rejected: bool = Client
        .v2()
        .name("test-client")
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(
                ProtocolVersion::V2,
                agent_client_protocol::schema::v2::Implementation::new("test-client", "0.1"),
            ))
            .block_task()
            .await?;
            let new_session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            let outcome = cx
                .send_request(SetSessionConfigOptionRequest::new(
                    new_session.session_id,
                    SessionConfigId::new("mode"),
                    SessionConfigValueId::new("bogus-mode"),
                ))
                .block_task()
                .await;
            Ok(outcome.is_err())
        })
        .await
        .expect("connection completed");

    assert!(rejected, "unknown mode id should be rejected");
}
