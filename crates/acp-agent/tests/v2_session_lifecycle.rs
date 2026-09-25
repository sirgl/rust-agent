#![cfg(feature = "unstable_protocol_v2")]

//! ACP v2 session lifecycle coverage: list, resume/replay, close, and baseline
//! session capability advertisement.

use std::sync::{Arc, Mutex};

use acp_agent::{build_agent, AgentConfig, AgentDeps, NextTurnFactory};
use agent_client_protocol::schema::v2::{
    CloseSessionRequest, ContentBlock, InitializeRequest, ListSessionsRequest, NewSessionRequest,
    PromptRequest, ReplayFrom, ReplayFromStart, ResumeSessionRequest, SessionUpdate,
    UpdateSessionNotification,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{on_receive_notification, Client};
use agent_core::{
    InMemorySessionStore, JsonFileSessionStore, NextTurnService, SessionState, SessionStore,
    StopReason as CoreStopReason, ToolRegistry, TurnEvent,
};

mod common;

fn deps(store: Arc<dyn SessionStore>) -> AgentDeps {
    let factory: NextTurnFactory = Arc::new(move |_selection| {
        Arc::new(turn_replay::ReplayTurnService::new(vec![vec![
            TurnEvent::TextDelta("answer".to_string()),
            TurnEvent::TurnFinished {
                stop_reason: CoreStopReason::EndTurn,
            },
        ]])) as Arc<dyn NextTurnService>
    });
    AgentDeps {
        next_turn_factory: factory,
        tools: ToolRegistry::new(),
        config: AgentConfig {
            require_submit_result: false,
            ..AgentConfig::default()
        },
        store,
        turn_observer: None,
        inspector_base_url: None,
    }
}

#[tokio::test]
async fn resume_survives_agent_restart_with_disk_store() {
    let root = std::env::temp_dir().join(format!("acp-v2-restart-{}", uuid::Uuid::new_v4()));
    let workspace = root.join("workspace");
    let store_dir = root.join("sessions");
    std::fs::create_dir_all(&workspace).unwrap();

    let first_workspace = workspace.clone();
    let session_id = Client
        .v2()
        .name("first-process")
        .connect_with(
            build_agent(deps(Arc::new(JsonFileSessionStore::new(&store_dir)))),
            async move |cx| {
                cx.send_request(initialize()).block_task().await?;
                Ok(cx
                    .send_request(NewSessionRequest::new(first_workspace))
                    .block_task()
                    .await?
                    .session_id)
            },
        )
        .await
        .expect("first agent process completed");

    Client
        .v2()
        .name("second-process")
        .connect_with(
            build_agent(deps(Arc::new(JsonFileSessionStore::new(&store_dir)))),
            async move |cx| {
                cx.send_request(initialize()).block_task().await?;
                cx.send_request(ResumeSessionRequest::new(session_id, workspace))
                    .block_task()
                    .await?;
                Ok(())
            },
        )
        .await
        .expect("second agent process resumed the session");

    std::fs::remove_dir_all(root).unwrap();
}

fn initialize() -> InitializeRequest {
    InitializeRequest::new(
        ProtocolVersion::V2,
        agent_client_protocol::schema::v2::Implementation::new("test-client", "0.1"),
    )
}

#[tokio::test]
async fn initialize_advertises_v2_baseline_and_supported_session_extensions() {
    let store = Arc::new(InMemorySessionStore::new());
    let capabilities = Client
        .v2()
        .name("capability-client")
        .connect_with(build_agent(deps(store)), async move |cx| {
            Ok(cx
                .send_request(initialize())
                .block_task()
                .await?
                .capabilities)
        })
        .await
        .expect("connection completed");

    let session = capabilities.session.expect("session surface advertised");
    assert!(session.additional_directories.is_some());
    assert!(session.mcp.expect("MCP capabilities").stdio.is_some());
}

#[tokio::test]
async fn new_session_is_listed_with_workspace_roots() {
    let store = Arc::new(InMemorySessionStore::new());
    let root = std::env::temp_dir().join(format!("acp-v2-list-{}", std::process::id()));
    let cwd = root.join("project");
    let extra = root.join("extra");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::create_dir_all(&extra).unwrap();
    let request_cwd = cwd.clone();
    let request_extra = extra.clone();
    let filter_cwd = cwd.clone();

    let sessions = Client
        .v2()
        .name("list-client")
        .connect_with(build_agent(deps(store)), async move |cx| {
            cx.send_request(initialize()).block_task().await?;
            cx.send_request(
                NewSessionRequest::new(request_cwd).additional_directories(vec![request_extra]),
            )
            .block_task()
            .await?;
            Ok(cx
                .send_request(ListSessionsRequest::new().cwd(filter_cwd))
                .block_task()
                .await?
                .sessions)
        })
        .await
        .expect("connection completed");

    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].cwd.0, cwd);
    assert_eq!(sessions[0].additional_directories[0].0, extra);
}

#[tokio::test]
async fn resume_from_start_replays_messages_and_commands() {
    let store = Arc::new(InMemorySessionStore::new());
    let mut state = SessionState::new("persisted");
    state.set_workspace("/tmp", std::iter::empty::<String>());
    state.push_user_text("hello");
    state.push_assistant_text("persisted answer");
    store.save(&state.to_record()).await.unwrap();

    let updates: common::Updates = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    let config_options = Client
        .v2()
        .name("resume-client")
        .on_receive_notification(
            async move |notification: UpdateSessionNotification, _cx| {
                collected.lock().unwrap().push(notification.update);
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_with(build_agent(deps(store)), async move |cx| {
            cx.send_request(initialize()).block_task().await?;
            Ok(cx
                .send_request(
                    ResumeSessionRequest::new("persisted", "/tmp")
                        .replay_from(ReplayFrom::from(ReplayFromStart::new())),
                )
                .block_task()
                .await?
                .config_options)
        })
        .await
        .expect("connection completed");

    assert!(config_options
        .iter()
        .any(|option| option.config_id.0.as_ref() == "mode"));
    let snapshot = updates.lock().unwrap();
    assert!(snapshot.iter().any(|update| match update {
        SessionUpdate::UserMessage(message) => message
            .content
            .value()
            .is_some_and(|content| content == &[ContentBlock::from("hello")]),
        _ => false,
    }));
    assert!(snapshot.iter().any(|update| match update {
        SessionUpdate::AgentMessage(message) => message
            .content
            .value()
            .is_some_and(|content| content == &[ContentBlock::from("persisted answer")]),
        _ => false,
    }));
    assert!(snapshot
        .iter()
        .any(|update| matches!(update, SessionUpdate::AvailableCommandsUpdate(_))));
}

#[tokio::test]
async fn resume_rejects_unknown_session_and_changed_cwd() {
    let store = Arc::new(InMemorySessionStore::new());
    let mut state = SessionState::new("known");
    state.set_workspace("/tmp", std::iter::empty::<String>());
    store.save(&state.to_record()).await.unwrap();

    Client
        .v2()
        .name("invalid-resume-client")
        .connect_with(build_agent(deps(store)), async move |cx| {
            cx.send_request(initialize()).block_task().await?;
            assert!(cx
                .send_request(ResumeSessionRequest::new("missing", "/tmp"))
                .block_task()
                .await
                .is_err());
            assert!(cx
                .send_request(ResumeSessionRequest::new("known", "/"))
                .block_task()
                .await
                .is_err());
            Ok(())
        })
        .await
        .expect("connection completed");
}

#[tokio::test]
async fn close_cancels_and_removes_the_live_session_but_keeps_it_resumable() {
    let store = Arc::new(InMemorySessionStore::new());
    let updates: common::Updates = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    let completion = updates.clone();

    Client
        .v2()
        .name("close-client")
        .on_receive_notification(
            async move |notification: UpdateSessionNotification, _cx| {
                collected.lock().unwrap().push(notification.update);
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_with(build_agent(deps(store)), async move |cx| {
            cx.send_request(initialize()).block_task().await?;
            let session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            let id = session.session_id.clone();
            cx.send_request(PromptRequest::new(
                id.clone(),
                vec![ContentBlock::from("hello")],
            ))
            .block_task()
            .await?;
            common::wait_for_idle(&completion).await;

            cx.send_request(CloseSessionRequest::new(id.clone()))
                .block_task()
                .await?;
            assert!(cx
                .send_request(PromptRequest::new(
                    id.clone(),
                    vec![ContentBlock::from("closed")],
                ))
                .block_task()
                .await
                .is_err());
            cx.send_request(ResumeSessionRequest::new(id, "/tmp"))
                .block_task()
                .await?;
            Ok(())
        })
        .await
        .expect("connection completed");
}
