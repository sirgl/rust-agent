//! Integration tests for ACP `session/load` and the `load_session` capability
//! (Step 4).
//!
//! These drive a real [`agent_client_protocol`] `Client` against the agent
//! built by [`acp_agent::build_agent`], sharing a concrete
//! [`InMemorySessionStore`] so the test can create + persist a session, then
//! reload it and observe the replayed history, re-advertised commands, and
//! continued usage accumulation.

use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v1::{
    AgentCapabilities, ContentBlock, InitializeRequest, LoadSessionRequest, NewSessionRequest,
    PromptRequest, SessionNotification, SessionUpdate,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{Client, on_receive_notification};
use agent_core::{
    InMemorySessionStore, NextTurnService, SessionStore, StopReason as CoreStopReason, TokenUsage,
    ToolRegistry, TurnEvent,
};

use acp_agent::{build_agent, AgentConfig, AgentDeps, NextTurnFactory};

/// Build [`AgentDeps`] that replay the given script, sharing `store`.
fn deps_with_store(script: Vec<Vec<TurnEvent>>, store: Arc<InMemorySessionStore>) -> AgentDeps {
    let factory: NextTurnFactory = Arc::new(move |_selection| {
        let service: Arc<dyn NextTurnService> =
            Arc::new(turn_replay::ReplayTurnService::new(script.clone()));
        service
    });
    AgentDeps {
        next_turn_factory: factory,
        tools: ToolRegistry::new(),
        config: AgentConfig {
            require_submit_result: false,
            ..AgentConfig::default()
        },
        store,
    }
}

/// A single-round script that answers with text plus a usage report.
fn answer_script() -> Vec<Vec<TurnEvent>> {
    vec![vec![
        TurnEvent::TextDelta("Persisted answer.".to_string()),
        TurnEvent::Usage(TokenUsage {
            input_tokens: 3,
            output_tokens: 5,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
        }),
        TurnEvent::TurnFinished {
            stop_reason: CoreStopReason::EndTurn,
        },
    ]]
}

#[tokio::test]
async fn initialize_advertises_load_session_capability() {
    let store = Arc::new(InMemorySessionStore::new());
    let agent = build_agent(deps_with_store(answer_script(), store));

    let caps: AgentCapabilities = Client
        .builder()
        .name("test-client")
        .connect_with(agent, async move |cx| {
            let init = cx
                .send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            Ok(init.agent_capabilities)
        })
        .await
        .expect("connection completed");

    assert!(caps.load_session, "agent must advertise load_session");
}

#[tokio::test]
async fn load_unknown_session_returns_error() {
    let store = Arc::new(InMemorySessionStore::new());
    let agent = build_agent(deps_with_store(answer_script(), store));

    let result: Result<(), String> = Client
        .builder()
        .name("test-client")
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let outcome = cx
                .send_request(LoadSessionRequest::new("no-such-session", "/tmp"))
                .block_task()
                .await;
            match outcome {
                Ok(_) => Ok(Err("expected an error for unknown session".to_string())),
                Err(_) => Ok(Ok(())),
            }
        })
        .await
        .expect("connection completed");

    result.expect("unknown session should fail");
}

#[tokio::test]
async fn load_replays_history_and_continues_usage() {
    let store = Arc::new(InMemorySessionStore::new());
    let agent = build_agent(deps_with_store(answer_script(), store.clone()));

    let session_id = Client
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
            let id = new_session.session_id.0.to_string();

            // First turn: builds and persists history + usage.
            cx.send_request(PromptRequest::new(
                new_session.session_id.clone(),
                vec![ContentBlock::from("hello")],
            ))
            .block_task()
            .await?;

            Ok(id)
        })
        .await
        .expect("connection completed");

    // Snapshot state persisted after the first turn.
    let after_first = store
        .load(&session_id)
        .await
        .expect("store load")
        .expect("session persisted");
    let history_len_after_first = after_first.history.len();
    let usage_after_first = after_first.usage.output_tokens;
    assert!(history_len_after_first >= 2, "user + assistant persisted");
    assert_eq!(usage_after_first, 5, "usage recorded from first turn");

    // Fresh agent sharing the same store: proves load reads from the store.
    let reload_agent = build_agent(deps_with_store(answer_script(), store.clone()));
    let replay_updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let replay_collected = replay_updates.clone();
    let load_session_id = session_id.clone();

    Client
        .builder()
        .name("test-client-2")
        .on_receive_notification(
            async move |notif: SessionNotification, _cx| {
                replay_collected.lock().unwrap().push(notif.update);
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_with(reload_agent, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let loaded = cx
                .send_request(LoadSessionRequest::new(load_session_id.clone(), "/tmp"))
                .block_task()
                .await?;
            // A loaded session returns config options.
            assert!(
                loaded.config_options.is_some(),
                "load returns config_options"
            );

            // A prompt after load must continue against prior history/usage.
            cx.send_request(PromptRequest::new(
                load_session_id,
                vec![ContentBlock::from("again")],
            ))
            .block_task()
            .await?;
            Ok(())
        })
        .await
        .expect("connection completed");

    let replay = replay_updates.lock().unwrap().clone();

    // History was replayed: user + assistant chunks in order, before the new
    // turn's own updates.
    let user_texts: Vec<String> = replay
        .iter()
        .filter_map(|u| match u {
            SessionUpdate::UserMessageChunk(chunk) => match &chunk.content {
                ContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert!(
        user_texts.first().map(String::as_str) == Some("hello"),
        "prior user message replayed first, got {user_texts:?}"
    );

    let assistant_texts: Vec<String> = replay
        .iter()
        .filter_map(|u| match u {
            SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                ContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert!(
        assistant_texts.contains(&"Persisted answer.".to_string()),
        "prior assistant message replayed, got {assistant_texts:?}"
    );

    // Commands were re-advertised on load.
    let advertised = replay
        .iter()
        .any(|u| matches!(u, SessionUpdate::AvailableCommandsUpdate(_)));
    assert!(advertised, "session/load re-advertises available commands");

    // The post-load prompt continued accumulating history + usage.
    let after_second = store
        .load(&session_id)
        .await
        .expect("store load")
        .expect("session persisted");
    assert!(
        after_second.history.len() > history_len_after_first,
        "history grew after post-load prompt"
    );
    assert!(
        after_second.usage.output_tokens > usage_after_first,
        "usage accumulated across the reloaded turn"
    );
}
