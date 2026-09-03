#![cfg(feature = "unstable_protocol_v2")]

//! Integration test: when an inspector base URL is configured, the agent greets
//! the session with a per-session inspector link as the first message *of the
//! first prompt turn* (clients render `session/update` only during a turn).
//! Driven through an in-process fake client with the replay backend (no network).

use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v2::{
    ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest, SessionUpdate,
    UpdateSessionNotification,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{on_receive_notification, Client};
use agent_core::event::StopReason;
use agent_core::{NextTurnService, ToolRegistry, TurnEvent};

use acp_agent::{build_agent, default_store, AgentConfig, AgentDeps, NextTurnFactory};

mod common;

fn deps_with_inspector_url(url: Option<String>) -> AgentDeps {
    let factory: NextTurnFactory = Arc::new(move |_selection| {
        // Several finishing rounds so the turn completes regardless of any
        // progress nudges the default config may inject.
        let script: Vec<Vec<TurnEvent>> = (0..8)
            .map(|round| {
                let mut events = Vec::new();
                if round == 0 {
                    events.push(TurnEvent::TextDelta("model answer".to_string()));
                }
                events.push(TurnEvent::TurnFinished {
                    stop_reason: StopReason::EndTurn,
                });
                events
            })
            .collect();
        let service: Arc<dyn NextTurnService> =
            Arc::new(turn_replay::ReplayTurnService::new(script));
        service
    });
    AgentDeps {
        next_turn_factory: factory,
        tools: ToolRegistry::new(),
        // Keep the "plain text / EndTurn ends the turn" behavior so the scripted
        // replay turn completes deterministically.
        config: AgentConfig {
            require_submit_result: false,
            ..AgentConfig::default()
        },
        store: default_store(),
        turn_observer: None,
        inspector_base_url: url,
    }
}

/// Concatenate the text of an `AgentMessageChunk` update, if it is one.
fn agent_chunk_text(update: &SessionUpdate) -> Option<String> {
    match update {
        SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
            agent_client_protocol::schema::v2::ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        },
        _ => None,
    }
}

#[tokio::test]
async fn first_prompt_greets_with_per_session_inspector_link() {
    let agent = build_agent(deps_with_inspector_url(Some(
        "http://127.0.0.1:7878".to_string(),
    )));

    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    let completion = updates.clone();

    let session_id: String = Client
        .v2()
        .name("test-client")
        .on_receive_notification(
            async move |notif: UpdateSessionNotification, _cx| {
                collected.lock().unwrap().push(notif.update);
                Ok(())
            },
            on_receive_notification!(),
        )
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
            let session_id = new_session.session_id.clone();
            // The greeting is deferred to the first prompt turn.
            cx.send_request(PromptRequest::new(
                session_id.clone(),
                vec![ContentBlock::from("hi")],
            ))
            .block_task()
            .await?;
            common::wait_for_idle(&completion).await;
            Ok(session_id.0.to_string())
        })
        .await
        .expect("connection completed");

    let agent_messages = updates
        .lock()
        .unwrap()
        .iter()
        .filter_map(agent_chunk_text)
        .collect::<Vec<_>>();
    let greeting = agent_messages
        .first()
        .expect("the first visible agent content must be the inspector link");

    assert!(greeting.contains("LLM Inspector"), "greeting: {greeting}");
    assert!(
        greeting.contains(&format!("http://127.0.0.1:7878/?session={session_id}")),
        "greeting should carry the per-session link: {greeting}"
    );
    assert_eq!(
        agent_messages.get(1).map(String::as_str),
        Some("model answer"),
        "the inspector link must precede model-produced content"
    );
}

#[tokio::test]
async fn session_new_sends_no_greeting_when_inspector_disabled() {
    let agent = build_agent(deps_with_inspector_url(None));

    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();

    Client
        .v2()
        .name("test-client")
        .on_receive_notification(
            async move |notif: UpdateSessionNotification, _cx| {
                collected.lock().unwrap().push(notif.update);
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(
                ProtocolVersion::V2,
                agent_client_protocol::schema::v2::Implementation::new("test-client", "0.1"),
            ))
            .block_task()
            .await?;
            cx.send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            Ok(())
        })
        .await
        .expect("connection completed");

    let has_inspector_msg = updates
        .lock()
        .unwrap()
        .iter()
        .filter_map(agent_chunk_text)
        .any(|t| t.contains("LLM Inspector"));
    assert!(!has_inspector_msg, "no inspector greeting expected");
}
