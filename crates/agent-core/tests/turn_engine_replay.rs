//! Integration tests for [`agent_core::TurnEngine::run_prompt`] driven by the
//! deterministic `turn-replay` backend and an in-memory fake [`UpdateSink`].
//!
//! These cover the core turn-loop scenarios required for step 2: plain
//! streaming, tool-call dispatch with typed history, stop-reason handling, and
//! cancellation of both the decision stream and tool dispatch.

use std::sync::Arc;

use agent_core::{
    AgentError, CancellationToken, ClientAccess, ElicitationOutcome, EngineOutput, HistoryEntry,
    NextTurnService, Result, SessionState, StopReason, TerminalOutcome, Tool, ToolCallId,
    ToolCallStatus, ToolContext, ToolEvent, ToolRegistry, TurnContext, TurnEngine, TurnEvent,
    TurnInbox, UpdateSink, MAX_TOOL_RESULT_CHARS,
};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use turn_replay::ReplayTurnService;

/// An in-memory [`UpdateSink`] that just records every output it receives, in
/// order, so tests can assert on the exact sequence forwarded by the engine.
///
/// It can also be configured to deny permission requests, recording the tools
/// for which permission was requested, so the permission-flow scenarios can be
/// asserted deterministically.
#[derive(Default)]
struct FakeSink {
    outputs: Vec<EngineOutput>,
    deny_permission: bool,
    permission_requests: Vec<String>,
}

#[async_trait]
impl UpdateSink for FakeSink {
    async fn send(&mut self, output: EngineOutput) -> Result<()> {
        self.outputs.push(output);
        Ok(())
    }

    async fn request_permission(&mut self, _id: &ToolCallId, tool_name: &str) -> Result<bool> {
        self.permission_requests.push(tool_name.to_string());
        Ok(!self.deny_permission)
    }
}

/// A tool that requires permission and records whether it was actually invoked.
struct PermissionedTool {
    invoked: Arc<std::sync::atomic::AtomicBool>,
}

#[async_trait]
impl Tool for PermissionedTool {
    fn name(&self) -> &str {
        "write_thing"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }
    fn requires_permission(&self) -> bool {
        true
    }
    async fn call(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        self.invoked
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(Box::pin(stream::iter(vec![
            ToolEvent::Started,
            ToolEvent::Completed(serde_json::json!("written")),
        ])))
    }
}

/// A trivial tool that completes immediately with a fixed result.
struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }
    fn requires_permission(&self) -> bool {
        false
    }
    async fn call(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        Ok(Box::pin(stream::iter(vec![
            ToolEvent::Started,
            ToolEvent::Completed(serde_json::json!("echoed")),
        ])))
    }
}

/// A tool that cancels the shared token as soon as it is invoked, before
/// yielding any events, so tests can exercise cancellation observed during
/// tool dispatch.
struct CancellingTool;

#[async_trait]
impl Tool for CancellingTool {
    fn name(&self) -> &str {
        "cancel_tool"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }
    fn requires_permission(&self) -> bool {
        false
    }
    async fn call(
        &self,
        _args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        ctx.cancel.cancel();
        Ok(Box::pin(stream::iter(vec![ToolEvent::Completed(
            serde_json::json!("should never be observed"),
        )])))
    }
}

/// A decision service that, on its first round, injects a steering message
/// into the shared [`TurnInbox`] and then finishes the turn. The engine should
/// notice the pending inbox and loop for a second round rather than ending.
struct SteeringOnFirstRoundService {
    inbox: TurnInbox,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait]
impl NextTurnService for SteeringOnFirstRoundService {
    async fn get_next_turn_streaming(
        &self,
        _ctx: &TurnContext,
    ) -> Result<BoxStream<'static, TurnEvent>> {
        let n = self
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if n == 0 {
            // Simulate a `_session/inject` arriving mid-turn.
            self.inbox
                .lock()
                .unwrap()
                .push("steer me".to_string());
            Ok(Box::pin(stream::iter(vec![
                TurnEvent::TextDelta("first".into()),
                TurnEvent::TurnFinished {
                    stop_reason: StopReason::EndTurn,
                },
            ])))
        } else {
            Ok(Box::pin(stream::iter(vec![
                TurnEvent::TextDelta("second".into()),
                TurnEvent::TurnFinished {
                    stop_reason: StopReason::EndTurn,
                },
            ])))
        }
    }
}

/// A decision service whose stream cancels the shared token as a side effect
/// of producing its second event, letting tests exercise cancellation
/// observed mid-decision-stream deterministically (no reliance on real time).
struct CancellingStreamService {
    cancel: CancellationToken,
}

#[async_trait]
impl NextTurnService for CancellingStreamService {
    async fn get_next_turn_streaming(
        &self,
        _ctx: &TurnContext,
    ) -> Result<BoxStream<'static, TurnEvent>> {
        let cancel = self.cancel.clone();
        let events = vec![
            TurnEvent::TextDelta("a".into()),
            TurnEvent::TextDelta("b".into()),
            TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            },
        ];
        let stream = stream::unfold((0usize, events, cancel), |(i, events, cancel)| async move {
            if i >= events.len() {
                return None;
            }
            if i == 1 {
                // Cancel right before producing the second event so the
                // engine observes cancellation partway through the stream.
                cancel.cancel();
            }
            let event = events[i].clone();
            Some((event, (i + 1, events, cancel)))
        });
        Ok(Box::pin(stream))
    }
}

#[tokio::test]
async fn plain_streaming_turn_forwards_chunks_and_returns_stop_reason() {
    let svc = ReplayTurnService::single(vec![
        TurnEvent::TextDelta("Hel".into()),
        TurnEvent::TextDelta("lo!".into()),
        TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        },
    ]);
    let mut session = SessionState::new("s1");
    session.push_user_text("hi");
    let mut sink = FakeSink::default();
    let cancel = CancellationToken::new();
    let engine = TurnEngine::new(Arc::new(svc), ToolRegistry::new());

    let stop = engine
        .run_prompt(&mut session, &mut sink, &cancel)
        .await
        .unwrap();

    assert_eq!(stop, StopReason::EndTurn);
    assert_eq!(sink.outputs.len(), 2);
    match (&sink.outputs[0], &sink.outputs[1]) {
        (EngineOutput::MessageChunk(a), EngineOutput::MessageChunk(b)) => {
            assert_eq!(a, "Hel");
            assert_eq!(b, "lo!");
        }
        other => panic!("unexpected outputs: {other:?}"),
    }

    // The user message plus a single, concatenated assistant message.
    assert_eq!(session.history.len(), 2);
    match &session.history.entries[1] {
        HistoryEntry::Assistant(msg) => {
            let agent_core::ContentBlock::Text { text } = &msg.content[0];
            assert_eq!(text, "Hello!");
        }
        other => panic!("unexpected entry: {other:?}"),
    }
}

#[tokio::test]
async fn tool_call_round_trip_streams_updates_and_appends_typed_history() {
    let id_marker = "call-1";
    let svc = ReplayTurnService::new(vec![
        vec![
            TurnEvent::ToolCallRequested {
                id: ToolCallId::new(id_marker),
                name: "echo".into(),
                arguments: serde_json::json!({}),
            },
            TurnEvent::TurnFinished {
                stop_reason: StopReason::ToolUse,
            },
        ],
        vec![
            TurnEvent::TextDelta("done".into()),
            TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            },
        ],
    ]);
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool));
    let mut session = SessionState::new("s1");
    let mut sink = FakeSink::default();
    let cancel = CancellationToken::new();
    let engine = TurnEngine::new(Arc::new(svc), registry);

    let stop = engine
        .run_prompt(&mut session, &mut sink, &cancel)
        .await
        .unwrap();

    assert_eq!(stop, StopReason::EndTurn);

    // pending -> in_progress -> completed, then the final text chunk.
    assert_eq!(sink.outputs.len(), 4);
    assert!(matches!(
        &sink.outputs[0],
        EngineOutput::ToolCall { status: ToolCallStatus::Pending, name, .. } if name == "echo"
    ));
    assert!(matches!(
        &sink.outputs[1],
        EngineOutput::ToolCallUpdate { status: ToolCallStatus::InProgress, .. }
    ));
    match &sink.outputs[2] {
        EngineOutput::ToolCallUpdate {
            status: ToolCallStatus::Completed,
            output,
            ..
        } => assert_eq!(output.as_deref(), Some("echoed")),
        other => panic!("unexpected output: {other:?}"),
    }
    assert!(matches!(&sink.outputs[3], EngineOutput::MessageChunk(t) if t == "done"));

    // Typed history: tool call, tool result, then the final assistant text.
    assert_eq!(session.history.len(), 3);
    match &session.history.entries[0] {
        HistoryEntry::ToolCall(rec) => {
            assert_eq!(rec.id.as_str(), id_marker);
            assert_eq!(rec.tool.name(), "echo");
        }
        other => panic!("unexpected entry: {other:?}"),
    }
    match &session.history.entries[1] {
        HistoryEntry::ToolResult(rec) => {
            assert!(rec.success);
            let agent_core::ContentBlock::Text { text } = &rec.content[0];
            assert_eq!(text, "echoed");
        }
        other => panic!("unexpected entry: {other:?}"),
    }
    assert!(matches!(&session.history.entries[2], HistoryEntry::Assistant(_)));
}

#[tokio::test]
async fn unknown_tool_fails_gracefully_without_deadlocking() {
    let svc = ReplayTurnService::new(vec![
        vec![
            TurnEvent::ToolCallRequested {
                id: ToolCallId::new("call-1"),
                name: "does_not_exist".into(),
                arguments: serde_json::json!({}),
            },
            TurnEvent::TurnFinished {
                stop_reason: StopReason::ToolUse,
            },
        ],
        vec![TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        }],
    ]);
    let mut session = SessionState::new("s1");
    let mut sink = FakeSink::default();
    let cancel = CancellationToken::new();
    let engine = TurnEngine::new(Arc::new(svc), ToolRegistry::new());

    let stop = engine
        .run_prompt(&mut session, &mut sink, &cancel)
        .await
        .unwrap();

    assert_eq!(stop, StopReason::EndTurn);
    match &session.history.entries[1] {
        HistoryEntry::ToolResult(rec) => assert!(!rec.success),
        other => panic!("unexpected entry: {other:?}"),
    }
}

#[tokio::test]
async fn stop_reason_is_propagated_as_scripted() {
    for reason in [StopReason::MaxTokens, StopReason::Refusal] {
        let svc = ReplayTurnService::single(vec![TurnEvent::TurnFinished { stop_reason: reason }]);
        let mut session = SessionState::new("s1");
        let mut sink = FakeSink::default();
        let cancel = CancellationToken::new();
        let engine = TurnEngine::new(Arc::new(svc), ToolRegistry::new());

        let stop = engine
            .run_prompt(&mut session, &mut sink, &cancel)
            .await
            .unwrap();

        assert_eq!(stop, reason);
        assert!(sink.outputs.is_empty());
    }
}

#[tokio::test]
async fn tool_use_without_a_tool_call_is_returned_directly() {
    // A backend that signals ToolUse without ever requesting a tool call is an
    // edge case: since no tool call was dispatched, the engine must not loop
    // for another round, it should surface the stop reason as-is.
    let svc = ReplayTurnService::single(vec![TurnEvent::TurnFinished {
        stop_reason: StopReason::ToolUse,
    }]);
    let mut session = SessionState::new("s1");
    let mut sink = FakeSink::default();
    let cancel = CancellationToken::new();
    let engine = TurnEngine::new(Arc::new(svc), ToolRegistry::new());

    let stop = engine
        .run_prompt(&mut session, &mut sink, &cancel)
        .await
        .unwrap();

    assert_eq!(stop, StopReason::ToolUse);
}

#[tokio::test]
async fn turn_error_event_surfaces_as_an_error() {
    let svc = ReplayTurnService::single(vec![TurnEvent::Error(agent_core::TurnError::new(
        "boom",
    ))]);
    let mut session = SessionState::new("s1");
    let mut sink = FakeSink::default();
    let cancel = CancellationToken::new();
    let engine = TurnEngine::new(Arc::new(svc), ToolRegistry::new());

    let err = engine
        .run_prompt(&mut session, &mut sink, &cancel)
        .await
        .unwrap_err();
    assert!(matches!(err, AgentError::Turn(_)));
}

#[tokio::test]
async fn cancellation_mid_decision_stream_stops_promptly_without_persisting_partial_text() {
    let cancel = CancellationToken::new();
    let svc = CancellingStreamService {
        cancel: cancel.clone(),
    };
    let mut session = SessionState::new("s1");
    session.push_user_text("hi");
    let mut sink = FakeSink::default();
    let engine = TurnEngine::new(Arc::new(svc), ToolRegistry::new());

    let stop = engine
        .run_prompt(&mut session, &mut sink, &cancel)
        .await
        .unwrap();

    assert_eq!(stop, StopReason::Cancelled);
    // No update is emitted once cancellation is observed; the scripted
    // TurnFinished event is never reached.
    assert!(sink
        .outputs
        .iter()
        .all(|o| matches!(o, EngineOutput::MessageChunk(_))));
    // Cancellation short-circuits before the accumulated text is flushed into
    // history, so only the original user message remains.
    assert_eq!(session.history.len(), 1);
}

#[tokio::test]
async fn cancellation_during_tool_dispatch_stops_without_completing_the_call() {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(CancellingTool));
    let svc = ReplayTurnService::single(vec![TurnEvent::ToolCallRequested {
        id: ToolCallId::new("call-1"),
        name: "cancel_tool".into(),
        arguments: serde_json::json!({}),
    }]);
    let mut session = SessionState::new("s1");
    let mut sink = FakeSink::default();
    let cancel = CancellationToken::new();
    let engine = TurnEngine::new(Arc::new(svc), registry);

    let stop = engine
        .run_prompt(&mut session, &mut sink, &cancel)
        .await
        .unwrap();

    assert_eq!(stop, StopReason::Cancelled);
    // pending + in_progress only: the completion event from the tool is never
    // observed because cancellation was detected first.
    assert_eq!(sink.outputs.len(), 2);
    assert!(matches!(
        &sink.outputs[0],
        EngineOutput::ToolCall {
            status: ToolCallStatus::Pending,
            ..
        }
    ));
    assert!(matches!(
        &sink.outputs[1],
        EngineOutput::ToolCallUpdate {
            status: ToolCallStatus::InProgress,
            ..
        }
    ));
    // The tool call was recorded, but no result was ever appended.
    assert_eq!(session.history.len(), 1);
    assert!(matches!(
        &session.history.entries[0],
        HistoryEntry::ToolCall(_)
    ));
}

#[tokio::test]
async fn permission_granted_executes_tool_and_records_success() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let invoked = Arc::new(AtomicBool::new(false));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(PermissionedTool {
        invoked: invoked.clone(),
    }));
    let svc = ReplayTurnService::new(vec![
        vec![
            TurnEvent::ToolCallRequested {
                id: ToolCallId::new("call-1"),
                name: "write_thing".into(),
                arguments: serde_json::json!({}),
            },
            TurnEvent::TurnFinished {
                stop_reason: StopReason::ToolUse,
            },
        ],
        vec![TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        }],
    ]);
    let mut session = SessionState::new("s1");
    let mut sink = FakeSink::default(); // grants by default
    let cancel = CancellationToken::new();
    let engine = TurnEngine::new(Arc::new(svc), registry);

    let stop = engine
        .run_prompt(&mut session, &mut sink, &cancel)
        .await
        .unwrap();

    assert_eq!(stop, StopReason::EndTurn);
    assert!(invoked.load(Ordering::SeqCst), "tool should have executed");
    assert_eq!(sink.permission_requests, vec!["write_thing".to_string()]);
    // pending -> in_progress -> completed.
    assert!(matches!(
        sink.outputs.last(),
        Some(EngineOutput::ToolCallUpdate {
            status: ToolCallStatus::Completed,
            ..
        })
    ));
    // Tool call + successful tool result recorded in history.
    match &session.history.entries[1] {
        HistoryEntry::ToolResult(rec) => assert!(rec.success),
        other => panic!("unexpected entry: {other:?}"),
    }
}

#[tokio::test]
async fn permission_denied_skips_execution_and_marks_failed() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let invoked = Arc::new(AtomicBool::new(false));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(PermissionedTool {
        invoked: invoked.clone(),
    }));
    let svc = ReplayTurnService::new(vec![
        vec![
            TurnEvent::ToolCallRequested {
                id: ToolCallId::new("call-1"),
                name: "write_thing".into(),
                arguments: serde_json::json!({}),
            },
            TurnEvent::TurnFinished {
                stop_reason: StopReason::ToolUse,
            },
        ],
        vec![TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        }],
    ]);
    let mut session = SessionState::new("s1");
    let mut sink = FakeSink {
        deny_permission: true,
        ..FakeSink::default()
    };
    let cancel = CancellationToken::new();
    let engine = TurnEngine::new(Arc::new(svc), registry);

    let stop = engine
        .run_prompt(&mut session, &mut sink, &cancel)
        .await
        .unwrap();

    // Loop continues to the next decision round and ends normally.
    assert_eq!(stop, StopReason::EndTurn);
    assert!(
        !invoked.load(Ordering::SeqCst),
        "tool must NOT execute when permission is denied"
    );
    assert_eq!(sink.permission_requests, vec!["write_thing".to_string()]);
    // The denied tool call is surfaced as a failed update, never in_progress.
    assert!(sink.outputs.iter().any(|o| matches!(
        o,
        EngineOutput::ToolCallUpdate {
            status: ToolCallStatus::Failed,
            ..
        }
    )));
    assert!(!sink.outputs.iter().any(|o| matches!(
        o,
        EngineOutput::ToolCallUpdate {
            status: ToolCallStatus::InProgress,
            ..
        }
    )));
    // History records the tool call and a failed result.
    match &session.history.entries[1] {
        HistoryEntry::ToolResult(rec) => assert!(!rec.success),
        other => panic!("unexpected entry: {other:?}"),
    }
}

#[tokio::test]
async fn steering_message_injected_mid_turn_extends_the_turn() {
    use std::sync::atomic::AtomicUsize;

    let inbox: TurnInbox = Arc::new(std::sync::Mutex::new(Vec::new()));
    let svc = SteeringOnFirstRoundService {
        inbox: inbox.clone(),
        calls: Arc::new(AtomicUsize::new(0)),
    };
    let mut session = SessionState::new("s1");
    session.push_user_text("hi");
    let mut sink = FakeSink::default();
    let cancel = CancellationToken::new();
    let engine = TurnEngine::new(Arc::new(svc), ToolRegistry::new()).with_inbox(inbox.clone());

    let stop = engine
        .run_prompt(&mut session, &mut sink, &cancel)
        .await
        .unwrap();

    assert_eq!(stop, StopReason::EndTurn);

    // Both rounds streamed their text: the turn continued instead of ending
    // after the first round because a steering message was pending.
    let chunks: Vec<String> = sink
        .outputs
        .iter()
        .filter_map(|o| match o {
            EngineOutput::MessageChunk(text) => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(chunks, vec!["first".to_string(), "second".to_string()]);

    // The injected message was folded into the conversation as a user message.
    let injected_user_msg = session.history.entries.iter().any(|entry| match entry {
        HistoryEntry::User(msg) => msg
            .content
            .iter()
            .any(|block| matches!(block, agent_core::ContentBlock::Text { text } if text == "steer me")),
        _ => false,
    });
    assert!(injected_user_msg, "injected steering message should be in history");

    // The inbox was fully drained.
    assert!(inbox.lock().unwrap().is_empty());
}

/// A tool that returns a very large output, to exercise output offloading.
struct BigOutputTool {
    output: String,
}

#[async_trait]
impl Tool for BigOutputTool {
    fn name(&self) -> &str {
        "big_output"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }
    fn requires_permission(&self) -> bool {
        false
    }
    async fn call(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        Ok(Box::pin(stream::iter(vec![
            ToolEvent::Started,
            ToolEvent::Completed(serde_json::json!(self.output)),
        ])))
    }
}

#[tokio::test]
async fn thought_interval_injects_nudge_before_first_round() {
    // A single-round turn: the thought nudge must have been injected before the
    // decision call, so it lands in history right after the user prompt.
    let svc = ReplayTurnService::single(vec![
        TurnEvent::TextDelta("on it".into()),
        TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        },
    ]);
    let mut session = SessionState::new("s1");
    session.push_user_text("hi");
    let mut sink = FakeSink::default();
    let cancel = CancellationToken::new();
    let engine = TurnEngine::new(Arc::new(svc), ToolRegistry::new()).with_thought_interval(5);

    engine
        .run_prompt(&mut session, &mut sink, &cancel)
        .await
        .unwrap();

    let nudged = session.history.entries.iter().any(|entry| match entry {
        HistoryEntry::User(msg) => msg.content.iter().any(|block| {
            matches!(block, agent_core::ContentBlock::Text { text }
                if text == agent_core::THOUGHT_NUDGE)
        }),
        _ => false,
    });
    assert!(nudged, "thought nudge should be injected on the first round");
}

/// A custom terminal tool (NOT `submit_result`) declaring `ends_turn() = true`,
/// used to prove the terminal-tool mechanism is decided by the tool itself and
/// works for any number of terminal tools, not a hardcoded name.
struct FinishTool;

#[async_trait]
impl Tool for FinishTool {
    fn name(&self) -> &str {
        "finish"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }
    fn requires_permission(&self) -> bool {
        false
    }
    fn ends_turn(&self) -> bool {
        true
    }
    async fn call(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        Ok(Box::pin(stream::iter(vec![
            ToolEvent::Started,
            ToolEvent::Completed(serde_json::json!("finished")),
        ])))
    }
}

/// A [`ClientAccess`] that records the last `write_text_file` call so tests can
/// assert the full offloaded output was persisted.
#[derive(Default)]
struct CapturingClient {
    written: std::sync::Mutex<Option<(String, String)>>,
}

#[async_trait]
impl ClientAccess for CapturingClient {
    async fn read_text_file(&self, _path: &str) -> Result<String> {
        Ok(String::new())
    }
    async fn write_text_file(&self, path: &str, content: &str) -> Result<()> {
        *self.written.lock().unwrap() = Some((path.to_string(), content.to_string()));
        Ok(())
    }
    async fn run_terminal(&self, _command: &str, _args: &[String]) -> Result<TerminalOutcome> {
        Ok(TerminalOutcome {
            output: String::new(),
            truncated: false,
            exit_code: Some(0),
            signal: None,
        })
    }
    async fn request_elicitation(
        &self,
        _message: &str,
        _requested_schema: serde_json::Value,
    ) -> Result<ElicitationOutcome> {
        Ok(ElicitationOutcome::Cancelled)
    }
}

#[tokio::test]
async fn large_tool_output_is_offloaded_and_truncated_in_history() {
    // A result far larger than the inline threshold.
    let big = "x".repeat(MAX_TOOL_RESULT_CHARS * 3);
    let svc = ReplayTurnService::new(vec![
        vec![
            TurnEvent::ToolCallRequested {
                id: ToolCallId::new("call-1"),
                name: "big_output".into(),
                arguments: serde_json::json!({}),
            },
            TurnEvent::TurnFinished {
                stop_reason: StopReason::ToolUse,
            },
        ],
        vec![TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        }],
    ]);
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(BigOutputTool { output: big.clone() }));
    let client = Arc::new(CapturingClient::default());
    let mut session = SessionState::new("s1");
    let mut sink = FakeSink::default();
    let cancel = CancellationToken::new();
    let engine = TurnEngine::new(Arc::new(svc), registry).with_client(client.clone());

    engine
        .run_prompt(&mut session, &mut sink, &cancel)
        .await
        .unwrap();

    // The full output was persisted via the client.
    let written = client.written.lock().unwrap().clone().expect("output saved");
    assert_eq!(written.1, big);
    assert!(written.0.contains("acp-agent-tool-outputs"));

    // The history tool result is truncated and points to the saved file.
    let result = session
        .history
        .entries
        .iter()
        .find_map(|entry| match entry {
            HistoryEntry::ToolResult(rec) => Some(rec),
            _ => None,
        })
        .expect("tool result recorded");
    let agent_core::ContentBlock::Text { text } = &result.content[0];
    assert!(text.chars().count() < big.chars().count());
    assert!(text.contains("Output truncated"));
    assert!(text.contains(&written.0));
}

#[tokio::test]
async fn small_tool_output_is_kept_inline_verbatim() {
    let small = "just a short result";
    let svc = ReplayTurnService::new(vec![
        vec![
            TurnEvent::ToolCallRequested {
                id: ToolCallId::new("call-1"),
                name: "big_output".into(),
                arguments: serde_json::json!({}),
            },
            TurnEvent::TurnFinished {
                stop_reason: StopReason::ToolUse,
            },
        ],
        vec![TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        }],
    ]);
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(BigOutputTool {
        output: small.to_string(),
    }));
    let client = Arc::new(CapturingClient::default());
    let mut session = SessionState::new("s1");
    let mut sink = FakeSink::default();
    let cancel = CancellationToken::new();
    let engine = TurnEngine::new(Arc::new(svc), registry).with_client(client.clone());

    engine
        .run_prompt(&mut session, &mut sink, &cancel)
        .await
        .unwrap();

    // Nothing offloaded, and the result is kept verbatim.
    assert!(client.written.lock().unwrap().is_none());
    let result = session
        .history
        .entries
        .iter()
        .find_map(|entry| match entry {
            HistoryEntry::ToolResult(rec) => Some(rec),
            _ => None,
        })
        .expect("tool result recorded");
    let agent_core::ContentBlock::Text { text } = &result.content[0];
    assert_eq!(text, small);
}

#[tokio::test]
async fn require_terminal_tool_nudges_on_text_and_ends_on_terminal_tool() {
    // Round 1: plain text + EndTurn (must NOT end the turn — it gets nudged).
    // Round 2: call the custom terminal tool `finish` (ends the turn).
    let svc = ReplayTurnService::new(vec![
        vec![
            TurnEvent::TextDelta("thinking".into()),
            TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            },
        ],
        vec![
            TurnEvent::ToolCallRequested {
                id: ToolCallId::new("call-finish"),
                name: "finish".into(),
                arguments: serde_json::json!({}),
            },
            TurnEvent::TurnFinished {
                stop_reason: StopReason::ToolUse,
            },
        ],
    ]);
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(FinishTool));
    let mut session = SessionState::new("s1");
    session.push_user_text("go");
    let mut sink = FakeSink::default();
    let cancel = CancellationToken::new();
    let engine = TurnEngine::new(Arc::new(svc), registry).require_terminal_tool();

    let stop = engine
        .run_prompt(&mut session, &mut sink, &cancel)
        .await
        .unwrap();

    // The turn ends only because the terminal tool was dispatched.
    assert_eq!(stop, StopReason::EndTurn);

    // The nudge was injected as a user message after the text-only round.
    let nudged = session.history.entries.iter().any(|entry| match entry {
        HistoryEntry::User(msg) => msg.content.iter().any(|block| {
            matches!(block, agent_core::ContentBlock::Text { text }
                if text == agent_core::TERMINAL_TOOL_NUDGE)
        }),
        _ => false,
    });
    assert!(nudged, "expected the terminal-tool nudge in history");
}

#[tokio::test]
async fn without_require_terminal_tool_plain_text_still_ends_turn() {
    // A terminal tool is registered, but enforcement is OFF: plain text ends
    // the turn as before, and the terminal tool is never needed.
    let svc = ReplayTurnService::single(vec![
        TurnEvent::TextDelta("done".into()),
        TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        },
    ]);
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(FinishTool));
    let mut session = SessionState::new("s1");
    session.push_user_text("go");
    let mut sink = FakeSink::default();
    let cancel = CancellationToken::new();
    let engine = TurnEngine::new(Arc::new(svc), registry);

    let stop = engine
        .run_prompt(&mut session, &mut sink, &cancel)
        .await
        .unwrap();

    assert_eq!(stop, StopReason::EndTurn);
}
