//! Regression for overlapping ACP prompts and retired-turn output fencing.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use acp_agent::{build_agent, AgentConfig, AgentDeps, NextTurnFactory};
use agent_client_protocol::schema::v1::{
    ContentBlock, InitializeRequest, NewSessionRequest, PermissionOptionId, PromptRequest,
    RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SelectedPermissionOutcome, SessionNotification, SessionUpdate, StopReason,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{on_receive_notification, on_receive_request, Client};
use agent_core::{
    HistoryEntry, InMemorySessionStore, NextTurnService, PlanStep, PlanStepStatus, PlanUpdate,
    Result, SessionStateSnapshot, SessionStore, StopReason as CoreStopReason, Tool, ToolCallId,
    ToolContext, ToolEvent, ToolRegistry, TurnContext, TurnEvent, TurnObservation, TurnObserver,
};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use tokio::sync::Notify;

struct LateOldTurnService {
    release_late_output: Arc<Notify>,
}

struct DestructiveTool;

#[async_trait]
impl Tool for DestructiveTool {
    fn name(&self) -> &str {
        "destructive_tool"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "additionalProperties": false})
    }

    fn requires_permission(&self) -> bool {
        true
    }

    async fn call(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        Ok(Box::pin(stream::iter([ToolEvent::Completed(
            serde_json::json!({"result": "should not run"}),
        )])))
    }
}

#[async_trait]
impl NextTurnService for LateOldTurnService {
    async fn get_next_turn_streaming(
        &self,
        _ctx: &TurnContext,
    ) -> Result<BoxStream<'static, TurnEvent>> {
        let release = self.release_late_output.clone();
        Ok(Box::pin(stream::unfold(0_u8, move |step| {
            let release = release.clone();
            async move {
                match step {
                    0 => Some((TurnEvent::TextDelta("old-before".to_string()), 1)),
                    1 => {
                        release.notified().await;
                        Some((TurnEvent::TextDelta("old-late-message".to_string()), 2))
                    }
                    2 => Some((
                        TurnEvent::Plan(PlanUpdate {
                            steps: vec![PlanStep {
                                content: "old-late-plan".to_string(),
                                status: PlanStepStatus::InProgress,
                            }],
                        }),
                        3,
                    )),
                    3 => Some((
                        TurnEvent::ToolCallRequested {
                            id: ToolCallId::new("old-late-tool"),
                            name: "destructive_tool".to_string(),
                            arguments: serde_json::json!({}),
                        },
                        4,
                    )),
                    4 => Some((
                        TurnEvent::TurnFinished {
                            stop_reason: CoreStopReason::EndTurn,
                        },
                        5,
                    )),
                    _ => None,
                }
            }
        })))
    }
}

#[derive(Default)]
struct StateObserver {
    states: Mutex<Vec<SessionStateSnapshot>>,
}

impl TurnObserver for StateObserver {
    fn on_turn_request(&self, _observation: &TurnObservation) {}

    fn on_session_state(&self, snapshot: &SessionStateSnapshot) {
        self.states.lock().unwrap().push(snapshot.clone());
    }
}

fn history_contains(snapshot: &SessionStateSnapshot, expected: &str) -> bool {
    snapshot.history.entries.iter().any(|entry| {
        matches!(
            entry,
            HistoryEntry::User(message)
                if message.content.iter().any(|block| matches!(
                    block,
                    agent_core::ContentBlock::Text { text } if text == expected
                ))
        )
    })
}

#[tokio::test]
async fn replacement_prompt_fences_all_late_old_turn_effects() {
    let release_late_output = Arc::new(Notify::new());
    let factory_calls = Arc::new(AtomicUsize::new(0));
    let factory: NextTurnFactory = Arc::new({
        let release_late_output = release_late_output.clone();
        move |_selection| {
            if factory_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                Arc::new(LateOldTurnService {
                    release_late_output: release_late_output.clone(),
                })
            } else {
                Arc::new(turn_replay::ReplayTurnService::single(vec![
                    TurnEvent::Thinking("new-thinking".to_string()),
                    TurnEvent::Plan(PlanUpdate {
                        steps: vec![PlanStep {
                            content: "new-plan".to_string(),
                            status: PlanStepStatus::Completed,
                        }],
                    }),
                    TurnEvent::TextDelta("new-answer".to_string()),
                    TurnEvent::TurnFinished {
                        stop_reason: CoreStopReason::EndTurn,
                    },
                ]))
            }
        }
    });
    let store = Arc::new(InMemorySessionStore::new());
    let observer = Arc::new(StateObserver::default());
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(DestructiveTool));
    let deps = AgentDeps {
        next_turn_factory: factory,
        tools,
        config: AgentConfig {
            require_submit_result: false,
            thought_interval: 0,
            ..AgentConfig::default()
        },
        store: store.clone(),
        turn_observer: Some(observer.clone()),
        inspector_base_url: Some("http://127.0.0.1:7878".to_string()),
    };

    let updates = Arc::new(Mutex::new(Vec::<SessionUpdate>::new()));
    let collected = updates.clone();
    let first_chunk_seen = Arc::new(Notify::new());
    let notify_first = first_chunk_seen.clone();
    let permission_requests = Arc::new(AtomicUsize::new(0));
    let counted_permissions = permission_requests.clone();

    let (session_id, first_stop, second_stop) = Client
        .builder()
        .name("test-client")
        .on_receive_notification(
            async move |notification: SessionNotification, _cx| {
                if matches!(
                    &notification.update,
                    SessionUpdate::AgentMessageChunk(chunk)
                        if matches!(&chunk.content, ContentBlock::Text(text) if text.text == "old-before")
                ) {
                    notify_first.notify_one();
                }
                collected.lock().unwrap().push(notification.update);
                Ok(())
            },
            on_receive_notification!(),
        )
        .on_receive_request(
            async move |request: RequestPermissionRequest, responder, _cx| {
                counted_permissions.fetch_add(1, Ordering::SeqCst);
                let selected = request
                    .options
                    .first()
                    .map(|option| option.option_id.clone())
                    .unwrap_or_else(|| PermissionOptionId::new("allow-once"));
                responder.respond(RequestPermissionResponse::new(
                    RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(selected)),
                ))
            },
            on_receive_request!(),
        )
        .connect_with(build_agent(deps), async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            let id = session.session_id.0.to_string();
            let first = cx
                .send_request(PromptRequest::new(
                    session.session_id.clone(),
                    vec![ContentBlock::from("first-user")],
                ))
                .block_task();

            tokio::pin!(first);
            tokio::select! {
                _ = first_chunk_seen.notified() => {}
                result = &mut first => panic!("first turn stopped before overlap: {result:?}"),
            }
            let second = cx
                .send_request(PromptRequest::new(
                    session.session_id,
                    vec![ContentBlock::from("second-user")],
                ))
                .block_task();
            let release = async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                release_late_output.notify_waiters();
            };
            let (first, second, ()) = tokio::join!(&mut first, second, release);
            Ok((id, first?.stop_reason, second?.stop_reason))
        })
        .await
        .expect("connection completed");

    assert_eq!(first_stop, StopReason::Cancelled);
    assert_eq!(second_stop, StopReason::EndTurn);

    {
        let updates = updates.lock().unwrap();
        let messages = updates
            .iter()
            .filter_map(|update| match update {
                SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                    ContentBlock::Text(text) => Some(text.text.to_string()),
                    _ => None,
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(messages
            .first()
            .is_some_and(|text| text.contains("LLM Inspector")));
        assert!(messages.iter().any(|text| text == "old-before"));
        assert!(messages.iter().any(|text| text == "new-answer"));
        assert!(!messages.iter().any(|text| text == "old-late-message"));
        assert!(!updates.iter().any(|update| matches!(
            update,
            SessionUpdate::Plan(plan)
                if plan.entries.iter().any(|entry| entry.content == "old-late-plan")
        )));
        assert!(!updates.iter().any(|update| matches!(
            update,
            SessionUpdate::ToolCall(call) if call.tool_call_id.0.as_ref() == "old-late-tool"
        )));
    }
    assert_eq!(permission_requests.load(Ordering::SeqCst), 0);

    let record = store
        .load(&session_id)
        .await
        .unwrap()
        .expect("newest session record must exist");
    let persisted = SessionStateSnapshot::from_state(&agent_core::SessionState::from_record(
        record,
        Vec::new(),
    ));
    assert!(history_contains(&persisted, "first-user"));
    assert!(history_contains(&persisted, "second-user"));
    assert!(!persisted.history.entries.iter().any(|entry| matches!(
        entry,
        HistoryEntry::Assistant(message)
            if message.content.iter().any(|block| matches!(
                block,
                agent_core::ContentBlock::Text { text } if text == "old-late-message"
            ))
    )));

    let states = observer.states.lock().unwrap();
    let latest = states
        .last()
        .expect("latest Inspector state must be published");
    assert!(history_contains(latest, "second-user"));
}
