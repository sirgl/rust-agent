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
    PlanEntryStatus, PromptRequest, SessionNotification, SessionUpdate,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{on_receive_notification, Client};
use agent_core::{
    InMemorySessionStore, NextTurnService, PlanStepStatus, SessionState, SessionStateSnapshot,
    SessionStore, StopReason as CoreStopReason, TodoItem, TodoItemId, TodoList, TokenUsage,
    ToolRegistry, TurnEvent, TurnObservation, TurnObserver,
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
        turn_observer: None,
        inspector_base_url: None,
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
async fn new_session_persists_cwd_then_additional_roots_in_request_order() {
    let store = Arc::new(InMemorySessionStore::new());
    let root = std::env::temp_dir().join(format!("rust-acp-new-workspace-{}", std::process::id()));
    let cwd = root.join("project");
    let first = root.join("first");
    let second = root.join("second");
    for directory in [&cwd, &first, &second] {
        std::fs::create_dir_all(directory).unwrap();
    }
    let expected = vec![
        cwd.to_string_lossy().into_owned(),
        first.to_string_lossy().into_owned(),
        second.to_string_lossy().into_owned(),
    ];
    let request_cwd = cwd.clone();
    let request_first = first.clone();
    let request_second = second.clone();

    let session_id = Client
        .builder()
        .name("new-workspace-client")
        .connect_with(
            build_agent(deps_with_store(answer_script(), store.clone())),
            async move |cx| {
                cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                let session = cx
                    .send_request(
                        NewSessionRequest::new(request_cwd)
                            .additional_directories(vec![request_first, request_second]),
                    )
                    .block_task()
                    .await?;
                Ok(session.session_id.0.to_string())
            },
        )
        .await
        .expect("connection completed");

    let record = store.load(&session_id).await.unwrap().unwrap();
    assert_eq!(record.workspace_roots, expected);
}

#[tokio::test]
async fn new_session_rejects_missing_working_directory_with_requested_path() {
    let store = Arc::new(InMemorySessionStore::new());
    let missing = std::env::temp_dir().join(format!(
        "rust-acp-missing-session-cwd-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&missing);
    let requested = missing.to_string_lossy().into_owned();

    let error = Client
        .builder()
        .name("invalid-workspace-client")
        .connect_with(
            build_agent(deps_with_store(answer_script(), store)),
            async move |cx| {
                cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                Ok(cx
                    .send_request(NewSessionRequest::new(missing))
                    .block_task()
                    .await
                    .expect_err("missing cwd must be rejected"))
            },
        )
        .await
        .expect("connection completed");

    assert!(
        error.to_string().contains(&requested),
        "error must name requested cwd: {error}"
    );
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

#[derive(Default)]
struct StateRecorder(Mutex<Vec<SessionStateSnapshot>>);

impl TurnObserver for StateRecorder {
    fn on_turn_request(&self, _observation: &TurnObservation) {}

    fn on_session_state(&self, snapshot: &SessionStateSnapshot) {
        self.0.lock().unwrap().push(snapshot.clone());
    }
}

#[tokio::test]
async fn load_replaces_stale_workspace_and_persists_published_roots() {
    let store = Arc::new(InMemorySessionStore::new());
    let mut state = SessionState::new("workspace-load");
    state.set_workspace("/stale/project", ["/stale/extra"]);
    store.save(&state.to_record()).await.unwrap();

    let observer = Arc::new(StateRecorder::default());
    let mut deps = deps_with_store(answer_script(), store.clone());
    deps.turn_observer = Some(observer.clone());
    let root = std::env::temp_dir().join(format!("rust-acp-load-workspace-{}", std::process::id()));
    let current_project = root.join("current-project");
    let current_extra = root.join("current-extra");
    std::fs::create_dir_all(&current_project).unwrap();
    std::fs::create_dir_all(&current_extra).unwrap();
    let expected = vec![
        current_project.to_string_lossy().into_owned(),
        current_extra.to_string_lossy().into_owned(),
    ];
    let request_project = current_project.clone();
    let request_extra = current_extra.clone();

    Client
        .builder()
        .name("workspace-load-client")
        .connect_with(build_agent(deps), async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            cx.send_request(
                LoadSessionRequest::new("workspace-load", request_project)
                    .additional_directories(vec![request_extra]),
            )
            .block_task()
            .await?;
            Ok(())
        })
        .await
        .expect("connection completed");

    let persisted = store
        .load("workspace-load")
        .await
        .unwrap()
        .expect("loaded state remains persisted");
    assert_eq!(persisted.workspace_roots, expected);
    assert_eq!(
        observer.0.lock().unwrap().last().unwrap().workspace_roots,
        expected
    );
}

#[tokio::test]
async fn load_restores_non_empty_todo_as_one_full_acp_plan() {
    let store = Arc::new(InMemorySessionStore::new());
    let mut state = SessionState::new("todo-load");
    state.todo_list = TodoList::new(vec![
        TodoItem {
            id: TodoItemId::new("second").unwrap(),
            content: "Second item".to_string(),
            status: PlanStepStatus::InProgress,
        },
        TodoItem {
            id: TodoItemId::new("first").unwrap(),
            content: "First item".to_string(),
            status: PlanStepStatus::Completed,
        },
    ])
    .unwrap();
    store.save(&state.to_record()).await.unwrap();

    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    Client
        .builder()
        .name("todo-load-client")
        .on_receive_notification(
            async move |notif: SessionNotification, _cx| {
                collected.lock().unwrap().push(notif.update);
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_with(
            build_agent(deps_with_store(answer_script(), store)),
            async move |cx| {
                cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                cx.send_request(LoadSessionRequest::new("todo-load", "/tmp"))
                    .block_task()
                    .await?;
                Ok(())
            },
        )
        .await
        .expect("connection completed");

    let plans = updates
        .lock()
        .unwrap()
        .iter()
        .filter_map(|update| match update {
            SessionUpdate::Plan(plan) => Some(plan.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0].entries.len(), 2);
    assert_eq!(plans[0].entries[0].content, "Second item");
    assert_eq!(plans[0].entries[0].status, PlanEntryStatus::InProgress);
    assert_eq!(plans[0].entries[1].content, "First item");
    assert_eq!(plans[0].entries[1].status, PlanEntryStatus::Completed);
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
