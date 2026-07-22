//! The [`Inspector`]: the capture hub that builds [`CapturedRequest`]s and fans
//! them out to attached [`Recorder`]s.
//!
//! It implements two observer seams so a single instance captures the full
//! picture of every decision round:
//!
//! - [`agent_core::TurnObserver`] — the provider-neutral, typed request snapshot
//!   (works for *every* backend, including the deterministic replay backend);
//! - [`turn_anthropic::RequestObserver`] — the exact provider wire body.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use agent_core::{
    SessionStateSnapshot, TraceContext, TraceEvent, TurnObservation, TurnObserver,
    TurnResponseObservation,
};
use turn_anthropic::RequestObserver;

use crate::record::{CapturedPayload, CapturedRequest, CapturedSessionState, ToolInfo};
use crate::recorder::{FileRecorder, MemoryRecorder, Recorder, DEFAULT_CAPACITY};

/// The label used for requests that could not be attributed to a session.
const UNKNOWN_SESSION: &str = "(unknown)";

/// The capture hub for LLM requests.
///
/// Cloning is cheap (shared `Arc` state). Attach it to the engine as a
/// [`TurnObserver`] and to the Anthropic backend as a [`RequestObserver`]; both
/// feed the same fan-out of recorders (an in-memory ring buffer for the UI, plus
/// any [`FileRecorder`]s for on-disk JSONL logs).
#[derive(Clone)]
pub struct Inspector {
    inner: Arc<Inner>,
}

struct Inner {
    memory: Arc<MemoryRecorder>,
    recorders: Vec<Arc<dyn Recorder>>,
    next_id: AtomicU64,
    /// The latest persisted-state snapshot per session (session_id -> state).
    ///
    /// Kept separate from the append-only record stream: only the most recent
    /// state per session is retained, so the UI always shows current memory.
    states: Arc<Mutex<HashMap<String, CapturedSessionState>>>,
}

impl Inspector {
    /// Create an inspector with an in-memory recorder at [`DEFAULT_CAPACITY`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Create an inspector whose in-memory recorder retains at most `capacity`.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let memory = Arc::new(MemoryRecorder::with_capacity(capacity));
        Self {
            inner: Arc::new(Inner {
                memory: memory.clone(),
                recorders: vec![memory],
                next_id: AtomicU64::new(1),
                states: Arc::new(Mutex::new(HashMap::new())),
            }),
        }
    }

    /// Add on-disk JSONL logging under `dir` (one file per session).
    ///
    /// Consumes and returns `self` for builder-style chaining. If the directory
    /// cannot be created the file recorder is skipped (logged), so the inspector
    /// still works in-memory.
    #[must_use]
    pub fn with_file_logging(self, dir: impl AsRef<Path>) -> Self {
        match FileRecorder::new(dir) {
            Ok(file) => self.with_recorder(Arc::new(file)),
            Err(err) => {
                tracing::warn!(%err, "llm-inspector: file logging disabled (cannot create dir)");
                self
            }
        }
    }

    /// Attach an arbitrary additional [`Recorder`].
    #[must_use]
    pub fn with_recorder(self, recorder: Arc<dyn Recorder>) -> Self {
        // `Inner` is shared behind an `Arc`; rebuild it with the extra recorder.
        // This runs only at construction time, so the clone is negligible.
        let inner = Arc::try_unwrap(self.inner).unwrap_or_else(|arc| Inner {
            memory: arc.memory.clone(),
            recorders: arc.recorders.clone(),
            next_id: AtomicU64::new(arc.next_id.load(Ordering::Relaxed)),
            states: arc.states.clone(),
        });
        let mut recorders = inner.recorders;
        recorders.push(recorder);
        Self {
            inner: Arc::new(Inner {
                memory: inner.memory,
                recorders,
                next_id: inner.next_id,
                states: inner.states,
            }),
        }
    }

    /// Snapshot all in-memory records, newest first.
    #[must_use]
    pub fn snapshot(&self) -> Vec<CapturedRequest> {
        self.inner.memory.snapshot()
    }

    /// The number of currently retained in-memory records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.memory.len()
    }

    /// Whether no in-memory records are currently retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.memory.is_empty()
    }

    /// Emit a fully-built record to every attached recorder.
    fn emit(
        &self,
        session_id: Option<&str>,
        agent_path: Option<String>,
        trace: Option<TraceContext>,
        model: Option<String>,
        payload: CapturedPayload,
    ) {
        let sequence_no = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let request = CapturedRequest {
            id: sequence_no,
            sequence_no,
            session_id: session_id.unwrap_or(UNKNOWN_SESSION).to_string(),
            agent_path,
            trace,
            timestamp_ms: now_ms(),
            model,
            payload,
        };
        for recorder in &self.inner.recorders {
            recorder.record(&request);
        }
    }

    /// Record a provider-neutral, typed request snapshot (from the engine).
    pub fn record_typed(&self, observation: &TurnObservation) {
        let history = serde_json::to_value(&observation.history).unwrap_or(serde_json::Value::Null);
        let tools = observation
            .tools
            .iter()
            .map(|t| ToolInfo {
                name: t.name.clone(),
                requires_permission: t.requires_permission,
            })
            .collect();
        self.emit(
            observation.session_id.as_deref(),
            Some(observation.agent_path.clone()),
            Some(observation.trace.clone()),
            None,
            CapturedPayload::Typed {
                system_prompt: observation.system_prompt.clone(),
                tools,
                history,
                num_entries: observation.history.len(),
                depth: observation.depth,
            },
        );
    }

    /// Record a provider-neutral, typed snapshot of a model response (from the
    /// engine), including token usage/reuse and the round's stop reason.
    pub fn record_response(&self, observation: &TurnResponseObservation) {
        let tool_calls =
            serde_json::to_value(&observation.tool_calls).unwrap_or(serde_json::Value::Null);
        let usage = serde_json::to_value(observation.usage).unwrap_or(serde_json::Value::Null);
        let stop_reason = observation
            .stop_reason
            .and_then(|s| serde_json::to_value(s).ok())
            .and_then(|v| v.as_str().map(str::to_owned));
        self.emit(
            observation.session_id.as_deref(),
            Some(observation.agent_path.clone()),
            Some(observation.trace.clone()),
            None,
            CapturedPayload::Response {
                assistant_text: observation.assistant_text.clone(),
                thinking_chars: observation.thinking_chars,
                tool_calls,
                usage,
                stop_reason,
                error: observation.error.clone(),
                duration_ms: observation.duration_ms,
            },
        );
    }

    /// Record an agent log line (captured from `tracing`) for extra debugging.
    pub fn record_log(
        &self,
        session_id: Option<&str>,
        trace: Option<TraceContext>,
        level: &str,
        target: &str,
        message: String,
        fields: serde_json::Value,
    ) {
        self.emit(
            session_id,
            trace.as_ref().map(|t| t.agent_path.clone()),
            trace,
            None,
            CapturedPayload::Log {
                level: level.to_string(),
                target: target.to_string(),
                message,
                fields,
            },
        );
    }

    /// Record a provider-specific compiled wire body.
    pub fn record_provider(
        &self,
        provider: &str,
        session_id: Option<&str>,
        model: &str,
        body: &serde_json::Value,
    ) {
        self.record_provider_correlated(provider, session_id, model, body, None);
    }

    fn record_provider_correlated(
        &self,
        provider: &str,
        session_id: Option<&str>,
        model: &str,
        body: &serde_json::Value,
        trace: Option<TraceContext>,
    ) {
        let num_messages = array_len(body, "messages");
        let num_tools = array_len(body, "tools");
        let size_bytes = serde_json::to_vec(body).map(|v| v.len()).unwrap_or(0);
        self.emit(
            session_id,
            None,
            trace,
            Some(model.to_string()),
            CapturedPayload::Provider {
                provider: provider.to_string(),
                body: body.clone(),
                num_messages,
                num_tools,
                size_bytes,
            },
        );
    }

    /// Record (replacing any previous entry for the same session) the latest
    /// persisted-state snapshot of a session, so the UI can show live state.
    pub fn record_session_state(&self, snapshot: &SessionStateSnapshot) {
        let tools = snapshot
            .tools
            .iter()
            .map(|t| ToolInfo {
                name: t.name.clone(),
                requires_permission: t.requires_permission,
            })
            .collect();
        let captured = CapturedSessionState {
            session_id: snapshot.session_id.clone(),
            timestamp_ms: now_ms(),
            mode: snapshot.mode.clone(),
            workspace_roots: snapshot.workspace_roots.clone(),
            system_prompt: snapshot.system_prompt.clone(),
            usage: serde_json::to_value(snapshot.usage).unwrap_or(serde_json::Value::Null),
            num_entries: snapshot.num_entries,
            tools,
            history: serde_json::to_value(&snapshot.history).unwrap_or(serde_json::Value::Null),
            todo_list: serde_json::to_value(&snapshot.todo_list).unwrap_or(serde_json::Value::Null),
        };
        // Guard is a plain std::Mutex held only for the insert; never across an
        // `.await`, so it is safe to touch on the turn path.
        if let Ok(mut states) = self.inner.states.lock() {
            states.insert(snapshot.session_id.clone(), captured);
        }
    }

    /// Snapshot the latest session-state per session, sorted by session id.
    #[must_use]
    pub fn session_states(&self) -> Vec<CapturedSessionState> {
        let states = match self.inner.states.lock() {
            Ok(states) => states,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut out: Vec<CapturedSessionState> = states.values().cloned().collect();
        out.sort_by(|a, b| a.session_id.cmp(&b.session_id));
        out
    }
}

impl Default for Inspector {
    fn default() -> Self {
        Self::new()
    }
}

/// The engine's provider-neutral seam: captures the typed request snapshot.
impl TurnObserver for Inspector {
    fn on_turn_request(&self, observation: &TurnObservation) {
        self.record_typed(observation);
    }

    fn on_turn_response(&self, observation: &TurnResponseObservation) {
        self.record_response(observation);
    }

    fn on_session_state(&self, snapshot: &SessionStateSnapshot) {
        self.record_session_state(snapshot);
    }

    fn on_trace_event(&self, event: &TraceEvent) {
        let (trace, session_id) = trace_event_context(event);
        self.emit(
            Some(session_id),
            Some(trace.agent_path.clone()),
            Some(trace.clone()),
            None,
            CapturedPayload::Trace {
                event: event.clone(),
            },
        );
    }
}

/// Anthropic's provider seam: captures the exact wire body.
impl RequestObserver for Inspector {
    fn on_request(
        &self,
        session_id: Option<&str>,
        model: &str,
        body: &serde_json::Value,
        trace: Option<&TraceContext>,
    ) {
        self.record_provider_correlated("anthropic", session_id, model, body, trace.cloned());
    }
}

fn trace_event_context(event: &TraceEvent) -> (&TraceContext, &str) {
    match event {
        TraceEvent::TurnStarted { trace, session_id }
        | TraceEvent::TurnFinished {
            trace, session_id, ..
        }
        | TraceEvent::AgentRunStarted { trace, session_id }
        | TraceEvent::AgentRunFinished {
            trace, session_id, ..
        }
        | TraceEvent::RoundStarted { trace, session_id }
        | TraceEvent::RoundFinished {
            trace, session_id, ..
        }
        | TraceEvent::ToolCallStarted {
            trace, session_id, ..
        }
        | TraceEvent::ToolCallFinished {
            trace, session_id, ..
        } => (trace, session_id),
    }
}

/// Current unix time in milliseconds (0 if the clock is before the epoch).
fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Best-effort length of a top-level JSON array field.
fn array_len(body: &serde_json::Value, key: &str) -> usize {
    body.get(key)
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::{AgentRunMetadata, ToolSummary};

    fn trace() -> TraceContext {
        TraceContext {
            turn_id: "turn-1".into(),
            agent_run_id: "agent-1".into(),
            parent_agent_run_id: None,
            spawned_by_tool_call_id: None,
            round_id: Some("round-1".into()),
            agent_path: "root".into(),
            metadata: AgentRunMetadata::default(),
        }
    }

    fn body(n_msgs: usize, n_tools: usize) -> serde_json::Value {
        serde_json::json!({
            "messages": vec![serde_json::json!({"role": "user"}); n_msgs],
            "tools": vec![serde_json::json!({"name": "t"}); n_tools],
        })
    }

    fn observation(session: &str, entries: usize) -> TurnObservation {
        use agent_core::{Conversation, UserMessage};
        let mut history = Conversation::default();
        for _ in 0..entries {
            history.push(agent_core::HistoryEntry::User(UserMessage::text("hi")));
        }
        TurnObservation {
            trace: trace(),
            session_id: Some(session.to_string()),
            agent_path: "root".to_string(),
            system_prompt: Some("sys".to_string()),
            tools: vec![ToolSummary {
                name: "fs_read".to_string(),
                requires_permission: false,
            }],
            workspace_roots: vec!["/workspace/project".to_string()],
            history,
            depth: 0,
        }
    }

    #[test]
    fn records_provider_metadata_newest_first() {
        let insp = Inspector::new();
        insp.record_provider("anthropic", Some("s1"), "m1", &body(2, 1));
        insp.record_provider("anthropic", Some("s2"), "m2", &body(3, 0));
        let snap = insp.snapshot();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].session_id, "s2");
        assert!(snap[0].id > snap[1].id);
        match &snap[0].payload {
            CapturedPayload::Provider {
                num_messages,
                num_tools,
                provider,
                ..
            } => {
                assert_eq!(*num_messages, 3);
                assert_eq!(*num_tools, 0);
                assert_eq!(provider, "anthropic");
            }
            _ => panic!("expected provider record"),
        }
    }

    #[test]
    fn records_typed_snapshot() {
        let insp = Inspector::new();
        insp.record_typed(&observation("s1", 3));
        let snap = insp.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].session_id, "s1");
        assert_eq!(snap[0].model, None);
        match &snap[0].payload {
            CapturedPayload::Typed {
                num_entries,
                tools,
                system_prompt,
                ..
            } => {
                assert_eq!(*num_entries, 3);
                assert_eq!(tools.len(), 1);
                assert_eq!(system_prompt.as_deref(), Some("sys"));
            }
            _ => panic!("expected typed record"),
        }
    }

    fn response(session: &str, agent_path: &str) -> TurnResponseObservation {
        use agent_core::{StopReason, TokenUsage, ToolCallSummary};
        TurnResponseObservation {
            trace: trace(),
            session_id: Some(session.to_string()),
            agent_path: agent_path.to_string(),
            depth: 0,
            assistant_text: "hello".to_string(),
            thinking_chars: 5,
            tool_calls: vec![ToolCallSummary {
                name: "fs_write".to_string(),
                arguments: serde_json::json!({ "path": "/a", "content": "x" }),
            }],
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 20,
                cache_read_input_tokens: 30,
                cache_creation_input_tokens: 0,
            },
            stop_reason: Some(StopReason::ToolUse),
            error: None,
            duration_ms: 42,
        }
    }

    #[test]
    fn records_response_with_usage_and_agent_path() {
        let insp = Inspector::new();
        insp.record_response(&response("s1", "root › delegate"));
        let snap = insp.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].session_id, "s1");
        assert_eq!(snap[0].agent_path.as_deref(), Some("root › delegate"));
        assert_eq!(snap[0].kind(), "response");
        match &snap[0].payload {
            CapturedPayload::Response {
                assistant_text,
                usage,
                stop_reason,
                duration_ms,
                ..
            } => {
                assert_eq!(assistant_text, "hello");
                assert_eq!(*duration_ms, 42);
                assert_eq!(stop_reason.as_deref(), Some("tool_use"));
                assert_eq!(usage["cache_read_input_tokens"], 30);
            }
            _ => panic!("expected response record"),
        }
    }

    #[test]
    fn on_turn_response_seam_records_response() {
        let insp = Inspector::new();
        (&insp as &dyn TurnObserver).on_turn_response(&response("s", "root"));
        assert_eq!(insp.len(), 1);
        assert_eq!(insp.snapshot()[0].kind(), "response");
    }

    #[test]
    fn record_log_captures_level_target_and_fields() {
        let insp = Inspector::new();
        insp.record_log(
            Some("s1"),
            None,
            "INFO",
            "acp_agent::agent",
            "starting".to_string(),
            serde_json::json!({ "n": 3 }),
        );
        let snap = insp.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].kind(), "log");
        match &snap[0].payload {
            CapturedPayload::Log {
                level,
                target,
                message,
                fields,
            } => {
                assert_eq!(level, "INFO");
                assert_eq!(target, "acp_agent::agent");
                assert_eq!(message, "starting");
                assert_eq!(fields["n"], 3);
            }
            _ => panic!("expected log record"),
        }
    }

    #[test]
    fn unknown_session_is_labelled() {
        let insp = Inspector::new();
        insp.record_provider("anthropic", None, "m", &body(1, 0));
        assert_eq!(insp.snapshot()[0].session_id, "(unknown)");
    }

    #[test]
    fn both_observer_seams_feed_the_store() {
        let insp = Inspector::new();
        (&insp as &dyn TurnObserver).on_turn_request(&observation("s", 1));
        (&insp as &dyn RequestObserver).on_request(Some("s"), "m", &body(1, 1), Some(&trace()));
        (&insp as &dyn TurnObserver).on_turn_response(&response("s", "root"));
        let records = insp.snapshot();
        assert_eq!(records.len(), 3);
        assert!(records.iter().all(|record| {
            let trace = record.trace.as_ref().expect("correlated record");
            trace.turn_id == "turn-1"
                && trace.agent_run_id == "agent-1"
                && trace.round_id.as_deref() == Some("round-1")
        }));
    }

    fn state_snapshot(session: &str, entries: usize) -> SessionStateSnapshot {
        use agent_core::{
            PlanStepStatus, SessionState, TodoItem, TodoItemId, TodoList, ToolDescriptor,
        };
        let mut state = SessionState::new(session);
        state.system_prompt = Some("sys".to_string());
        state.available_tools = vec![ToolDescriptor {
            name: "fs_read".to_string(),
            schema: serde_json::json!({}),
            requires_permission: false,
        }];
        for _ in 0..entries {
            state.push_user_text("hi");
        }
        state.todo_list = TodoList::new(vec![TodoItem {
            id: TodoItemId::new("verify").unwrap(),
            content: "Verify behavior".to_string(),
            status: PlanStepStatus::InProgress,
        }])
        .unwrap();
        SessionStateSnapshot::from_state(&state)
    }

    #[test]
    fn records_latest_session_state_per_session() {
        let insp = Inspector::new();
        insp.record_session_state(&state_snapshot("s1", 1));
        insp.record_session_state(&state_snapshot("s2", 2));
        // A newer snapshot for s1 replaces the previous one (latest wins).
        insp.record_session_state(&state_snapshot("s1", 5).with_mode("orchestrate"));

        let states = insp.session_states();
        assert_eq!(states.len(), 2);
        // Sorted by session id.
        assert_eq!(states[0].session_id, "s1");
        assert_eq!(states[0].num_entries, 5);
        assert_eq!(states[0].mode.as_deref(), Some("orchestrate"));
        assert_eq!(states[0].tools.len(), 1);
        assert_eq!(states[0].todo_list["items"][0]["id"], "verify");
        assert_eq!(
            states[0].todo_list["items"][0]["content"],
            "Verify behavior"
        );
        assert_eq!(states[0].todo_list["items"][0]["status"], "in_progress");
        assert_eq!(states[1].session_id, "s2");
        assert_eq!(states[1].num_entries, 2);
        // Session states are separate from the request stream.
        assert!(insp.is_empty());
    }

    #[test]
    fn on_session_state_seam_records_state() {
        let insp = Inspector::new();
        (&insp as &dyn TurnObserver).on_session_state(&state_snapshot("s", 3));
        let states = insp.session_states();
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].num_entries, 3);
        assert_eq!(states[0].system_prompt.as_deref(), Some("sys"));
    }

    #[test]
    fn file_logging_also_captures() {
        let dir = std::env::temp_dir().join(format!("llm-insp-store-{}", std::process::id()));
        let insp = Inspector::new().with_file_logging(&dir);
        insp.record_typed(&observation("sess-1", 2));
        let path = dir.join("sess-1.jsonl");
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("\"kind\":\"typed\""));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
