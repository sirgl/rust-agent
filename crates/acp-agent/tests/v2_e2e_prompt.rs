#![cfg(feature = "unstable_protocol_v2")]

//! End-to-end tests driving `session/prompt` through an in-process fake client.
//!
//! These tests wire a real [`agent_client_protocol`] `Client` connection
//! directly to the agent connection built by [`acp_agent::build_agent`] (the
//! agent builder implements `ConnectTo<Client>`), exercising the full path:
//! `initialize` -> `session/new` -> `session/prompt` -> streamed
//! `session/update` notifications -> `idle(stopReason)`. The decision layer is the
//! deterministic replay backend, so no network is involved.

use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v2::{
    ContentBlock, InitializeRequest, NewSessionRequest, PermissionOptionId, PromptRequest,
    RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SelectedPermissionOutcome, SessionConfigId, SessionConfigValueId, SessionUpdate,
    SetSessionConfigOptionRequest, StateUpdate, StopReason, ToolCallStatus,
    UpdateSessionNotification,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{on_receive_notification, on_receive_request, Client};
use agent_core::{
    AgentError, NextTurnService, Result as CoreResult, StopReason as CoreStopReason, Tool,
    ToolContext, ToolEvent, ToolRegistry, TurnContext, TurnError, TurnEvent,
};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use serde_json::json;

use acp_agent::{build_agent, AgentConfig, AgentDeps, NextTurnFactory};

mod common;

#[derive(Default)]
struct WorkspaceToolBackend {
    contexts: Mutex<Vec<TurnContext>>,
}

#[async_trait]
impl NextTurnService for WorkspaceToolBackend {
    async fn get_next_turn_streaming(
        &self,
        ctx: &TurnContext,
    ) -> CoreResult<BoxStream<'static, TurnEvent>> {
        let round = {
            let mut contexts = self.contexts.lock().unwrap();
            contexts.push(ctx.clone());
            contexts.len()
        };
        let events = if round == 1 {
            vec![
                TurnEvent::ToolCallRequested {
                    id: agent_core::ToolCallId::new("find-project"),
                    name: "terminal_run".to_string(),
                    arguments: json!({
                        "command": "find . -maxdepth 1 -type f | head -30"
                    }),
                },
                TurnEvent::TurnFinished {
                    stop_reason: CoreStopReason::ToolUse,
                },
            ]
        } else {
            vec![
                TurnEvent::TextDelta("project inspected".to_string()),
                TurnEvent::TurnFinished {
                    stop_reason: CoreStopReason::EndTurn,
                },
            ]
        };
        Ok(Box::pin(stream::iter(events)))
    }
}

#[tokio::test]
async fn terminal_shell_command_runs_in_session_workspace_and_reaches_next_round() {
    let root =
        std::env::temp_dir().join(format!("rust-acp-bundle-workspace-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("marker.txt"), "workspace marker").unwrap();

    let backend = Arc::new(WorkspaceToolBackend::default());
    let factory_backend = backend.clone();
    let factory: NextTurnFactory = Arc::new(move |_selection| factory_backend.clone());
    let deps = AgentDeps {
        next_turn_factory: factory,
        tools: tools_builtin::builtin_registry(),
        config: AgentConfig {
            require_submit_result: false,
            ..AgentConfig::default()
        },
        store: acp_agent::default_store(),
        turn_observer: None,
        inspector_base_url: None,
    };

    let request_root = root.clone();
    let updates: common::Updates = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    let completion = updates.clone();
    Client
        .v2()
        .name("workspace-tool-client")
        .on_receive_notification(
            async move |notif: UpdateSessionNotification, _cx| {
                collected.lock().unwrap().push(notif.update);
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_with(build_agent(deps), async move |cx| {
            cx.send_request(InitializeRequest::new(
                ProtocolVersion::V2,
                agent_client_protocol::schema::v2::Implementation::new("test-client", "0.1"),
            ))
            .block_task()
            .await?;
            let session = cx
                .send_request(NewSessionRequest::new(request_root))
                .block_task()
                .await?;
            cx.send_request(PromptRequest::new(
                session.session_id,
                vec![ContentBlock::from("inspect this project")],
            ))
            .block_task()
            .await?;
            common::wait_for_idle(&completion).await;
            Ok(())
        })
        .await
        .expect("connection completed");

    let contexts = backend.contexts.lock().unwrap();
    assert_eq!(contexts.len(), 2);
    assert_eq!(contexts[0].workspace_roots, vec![root.to_string_lossy()]);
    let second_history = serde_json::to_string(&contexts[1].history).unwrap();
    assert!(
        second_history.contains("marker.txt"),
        "terminal output must reach the next typed round: {second_history}"
    );
    let updates = updates.lock().unwrap();
    assert!(updates
        .iter()
        .any(|update| matches!(update, SessionUpdate::TerminalUpdate(_))));
    assert!(updates
        .iter()
        .any(|update| matches!(update, SessionUpdate::TerminalOutputChunk(_))));
}

#[tokio::test]
async fn fs_write_completes_and_changes_the_file() {
    let root = std::env::temp_dir().join(format!("rust-acp-diff-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let file = root.join("sample.txt");
    std::fs::write(&file, "before\nkeep\n").unwrap();
    let file_path = file.to_string_lossy().into_owned();
    let script = vec![
        vec![
            TurnEvent::ToolCallRequested {
                id: agent_core::ToolCallId::new("write-file"),
                name: "fs_write".to_string(),
                arguments: json!({
                    "path": file_path,
                    "content": "after\nkeep\n",
                }),
            },
            TurnEvent::TurnFinished {
                stop_reason: CoreStopReason::ToolUse,
            },
        ],
        vec![TurnEvent::TurnFinished {
            stop_reason: CoreStopReason::EndTurn,
        }],
    ];
    let deps = deps_with_script(script, tools_builtin::builtin_registry());
    let updates: common::Updates = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    let completion = updates.clone();
    let request_root = root.clone();

    Client
        .v2()
        .name("diff-client")
        .on_receive_notification(
            async move |notif: UpdateSessionNotification, _cx| {
                collected.lock().unwrap().push(notif.update);
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_with(build_agent(deps), async move |cx| {
            cx.send_request(InitializeRequest::new(
                ProtocolVersion::V2,
                agent_client_protocol::schema::v2::Implementation::new("test-client", "0.1"),
            ))
            .block_task()
            .await?;
            let session = cx
                .send_request(NewSessionRequest::new(request_root))
                .block_task()
                .await?;
            cx.send_request(PromptRequest::new(
                session.session_id,
                vec![ContentBlock::from("change the file")],
            ))
            .block_task()
            .await?;
            common::wait_for_idle(&completion).await;
            Ok(())
        })
        .await
        .expect("connection completed");

    let snapshot = updates.lock().unwrap().clone();
    assert!(snapshot.iter().any(|update| matches!(
        update,
        SessionUpdate::ToolCallUpdate(update)
            if update.status.value() == Some(&ToolCallStatus::Completed)
    )));
    assert_eq!(std::fs::read_to_string(&file).unwrap(), "after\nkeep\n");
    std::fs::remove_dir_all(root).unwrap();
}

/// Build [`AgentDeps`] whose decision layer replays the given scripted rounds.
fn deps_with_script(script: Vec<Vec<TurnEvent>>, tools: ToolRegistry) -> AgentDeps {
    let factory: NextTurnFactory = Arc::new(move |_selection| {
        let service: Arc<dyn NextTurnService> =
            Arc::new(turn_replay::ReplayTurnService::new(script.clone()));
        service
    });
    AgentDeps {
        next_turn_factory: factory,
        tools,
        // These scripted-replay tests assert fixed turn boundaries, so keep the
        // legacy "plain text ends the turn" behavior. Terminal-tool enforcement
        // is exercised by its own dedicated test below.
        config: AgentConfig {
            require_submit_result: false,
            ..AgentConfig::default()
        },
        store: acp_agent::default_store(),
        turn_observer: None,
        inspector_base_url: None,
    }
}

#[tokio::test]
async fn prompt_turn_persists_core_state_to_store() {
    use agent_core::{InMemorySessionStore, SessionStore};

    let script = vec![vec![
        TurnEvent::TextDelta("Persisted answer.".to_string()),
        TurnEvent::TurnFinished {
            stop_reason: CoreStopReason::EndTurn,
        },
    ]];
    let factory: NextTurnFactory = Arc::new(move |_selection| {
        let service: Arc<dyn NextTurnService> =
            Arc::new(turn_replay::ReplayTurnService::new(script.clone()));
        service
    });

    // Share a concrete in-memory store so the test can inspect what the agent
    // persisted after the turn completes.
    let store = Arc::new(InMemorySessionStore::new());
    let deps = AgentDeps {
        next_turn_factory: factory,
        tools: ToolRegistry::new(),
        config: AgentConfig {
            require_submit_result: false,
            ..AgentConfig::default()
        },
        store: store.clone(),
        turn_observer: None,
        inspector_base_url: None,
    };
    let agent = build_agent(deps);
    let updates: common::Updates = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    let completion = updates.clone();

    let session_id = Client
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
            let id = new_session.session_id.0.to_string();
            cx.send_request(PromptRequest::new(
                new_session.session_id,
                vec![ContentBlock::from("hi")],
            ))
            .block_task()
            .await?;
            common::wait_for_idle(&completion).await;
            Ok(id)
        })
        .await
        .expect("connection completed");

    // The prompt path persists the record before responding, so by the time
    // the request resolved the store must hold the updated core state.
    let record = store
        .load(&session_id)
        .await
        .expect("store load")
        .expect("session persisted");
    assert_eq!(record.session_id, session_id);
    // The turn ran, so the user prompt and assistant answer are in history.
    assert!(
        !record.history.is_empty(),
        "expected persisted history to contain the turn"
    );
}

#[tokio::test]
async fn plain_streaming_turn_streams_chunks_and_end_turn() {
    let script = vec![vec![
        TurnEvent::TextDelta("Hello, ".to_string()),
        TurnEvent::TextDelta("world!".to_string()),
        TurnEvent::TurnFinished {
            stop_reason: CoreStopReason::EndTurn,
        },
    ]];
    let deps = deps_with_script(script, ToolRegistry::new());
    let agent = build_agent(deps);

    let updates: common::Updates = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    let completion = updates.clone();

    let stop = Client
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
            cx.send_request(PromptRequest::new(
                new_session.session_id,
                vec![ContentBlock::from("hi")],
            ))
            .block_task()
            .await?;
            Ok(common::wait_for_idle(&completion).await)
        })
        .await
        .expect("connection completed");

    assert_eq!(stop, StopReason::EndTurn);

    let chunks: Vec<String> = updates
        .lock()
        .unwrap()
        .iter()
        .filter_map(|update| match update {
            SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                ContentBlock::Text(text) => Some(text.text.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(chunks, vec!["Hello, ".to_string(), "world!".to_string()]);
}

struct FailingBackend;

#[async_trait]
impl NextTurnService for FailingBackend {
    async fn get_next_turn_streaming(
        &self,
        _ctx: &TurnContext,
    ) -> CoreResult<BoxStream<'static, TurnEvent>> {
        Err(AgentError::Turn(TurnError::new(
            "provider rejected request",
        )))
    }
}

#[tokio::test]
async fn failed_turn_publishes_refusal_instead_of_success() {
    let factory: NextTurnFactory = Arc::new(|_selection| Arc::new(FailingBackend));
    let deps = AgentDeps {
        next_turn_factory: factory,
        tools: ToolRegistry::new(),
        config: AgentConfig {
            require_submit_result: false,
            ..AgentConfig::default()
        },
        store: acp_agent::default_store(),
        turn_observer: None,
        inspector_base_url: None,
    };
    let updates: common::Updates = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    let completion = updates.clone();

    let stop = Client
        .v2()
        .name("test-client")
        .on_receive_notification(
            async move |notif: UpdateSessionNotification, _cx| {
                collected.lock().unwrap().push(notif.update);
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_with(build_agent(deps), async move |cx| {
            cx.send_request(InitializeRequest::new(
                ProtocolVersion::V2,
                agent_client_protocol::schema::v2::Implementation::new("test-client", "0.1"),
            ))
            .block_task()
            .await?;
            let session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            cx.send_request(PromptRequest::new(
                session.session_id,
                vec![ContentBlock::from("hi")],
            ))
            .block_task()
            .await?;
            Ok(common::wait_for_idle(&completion).await)
        })
        .await
        .expect("connection completed");

    assert_eq!(stop, StopReason::Refusal);
}

#[tokio::test]
async fn prompt_for_unknown_session_returns_error() {
    let deps = deps_with_script(vec![vec![]], ToolRegistry::new());
    let agent = build_agent(deps);

    let result: Result<(), String> = Client
        .v2()
        .name("test-client")
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(
                ProtocolVersion::V2,
                agent_client_protocol::schema::v2::Implementation::new("test-client", "0.1"),
            ))
            .block_task()
            .await?;
            // Never created a session; prompting must fail.
            let outcome = cx
                .send_request(PromptRequest::new(
                    "does-not-exist",
                    vec![ContentBlock::from("hi")],
                ))
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

/// A tool that always succeeds and requires permission.
struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn schema(&self) -> serde_json::Value {
        json!({ "type": "object" })
    }
    fn requires_permission(&self) -> bool {
        true
    }
    async fn call(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> CoreResult<BoxStream<'static, ToolEvent>> {
        Ok(Box::pin(stream::iter(vec![
            ToolEvent::Started,
            ToolEvent::Completed(json!("echoed")),
        ])))
    }
}

/// Build a two-round script: a tool call, then a finishing text round.
fn tool_call_script() -> Vec<Vec<TurnEvent>> {
    vec![
        vec![
            TurnEvent::ToolCallRequested {
                id: agent_core::ToolCallId::new("call-1"),
                name: "echo".to_string(),
                arguments: json!({ "value": "hi" }),
            },
            TurnEvent::TurnFinished {
                stop_reason: CoreStopReason::ToolUse,
            },
        ],
        vec![
            TurnEvent::TextDelta("done".to_string()),
            TurnEvent::TurnFinished {
                stop_reason: CoreStopReason::EndTurn,
            },
        ],
    ]
}

/// Drive a tool-call turn, answering the permission request with `option_id`.
async fn run_tool_turn(option_id: &'static str) -> (StopReason, Vec<SessionUpdate>) {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(EchoTool));
    let deps = deps_with_script(tool_call_script(), tools);
    let agent = build_agent(deps);

    let updates: common::Updates = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    let completion = updates.clone();

    let stop = Client
        .v2()
        .name("test-client")
        .on_receive_notification(
            async move |notif: UpdateSessionNotification, _cx| {
                collected.lock().unwrap().push(notif.update);
                Ok(())
            },
            on_receive_notification!(),
        )
        .on_receive_request(
            async move |_req: RequestPermissionRequest, responder, _cx| {
                responder.respond(RequestPermissionResponse::new(
                    RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                        PermissionOptionId::new(option_id),
                    )),
                ))
            },
            on_receive_request!(),
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
            // Permission-flow tests explicitly opt into Normal because the
            // product default is YOLO and therefore bypasses client prompts.
            cx.send_request(SetSessionConfigOptionRequest::new(
                new_session.session_id.clone(),
                SessionConfigId::new("permissions"),
                SessionConfigValueId::new("normal"),
            ))
            .block_task()
            .await?;
            cx.send_request(PromptRequest::new(
                new_session.session_id,
                vec![ContentBlock::from("please echo")],
            ))
            .block_task()
            .await?;
            Ok(common::wait_for_idle(&completion).await)
        })
        .await
        .expect("connection completed");

    let snapshot = updates.lock().unwrap().clone();
    (stop, snapshot)
}

#[tokio::test]
async fn tool_call_with_granted_permission_completes() {
    let (stop, updates) = run_tool_turn("allow-once").await;
    assert_eq!(stop, StopReason::EndTurn);

    let statuses: Vec<ToolCallStatus> = common::tool_statuses(&updates);
    // Pending (creation) -> InProgress -> Completed.
    assert!(statuses.contains(&ToolCallStatus::InProgress));
    assert!(statuses.contains(&ToolCallStatus::Completed));
    assert!(!statuses.contains(&ToolCallStatus::Failed));
    assert!(updates
        .iter()
        .any(|update| matches!(update, SessionUpdate::ToolCallContentChunk(_))));

    let requires_action = updates
        .iter()
        .position(|update| {
            matches!(
                update,
                SessionUpdate::StateUpdate(StateUpdate::RequiresAction(_))
            )
        })
        .expect("permission request must publish requires_action");
    assert!(updates
        .iter()
        .skip(requires_action + 1)
        .any(|update| { matches!(update, SessionUpdate::StateUpdate(StateUpdate::Running(_))) }));
}

#[tokio::test]
async fn tool_call_with_denied_permission_fails_but_turn_continues() {
    let (stop, updates) = run_tool_turn("reject-once").await;
    // The turn still finishes (loop continues after the denied tool call).
    assert_eq!(stop, StopReason::EndTurn);

    let statuses: Vec<ToolCallStatus> = common::tool_statuses(&updates);
    assert!(statuses.contains(&ToolCallStatus::Failed));
    // A denied tool never enters InProgress.
    assert!(!statuses.contains(&ToolCallStatus::Completed));
}
