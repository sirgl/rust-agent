//! The [`AcpUpdateSink`]: maps [`agent_core`] engine output to ACP
//! `session/update` notifications and `session/request_permission` requests.
//!
//! The sink is deliberately thin: it owns a clone of the [`ConnectionTo<Client>`]
//! handle plus the session id, and translates each [`EngineOutput`] emitted by
//! the [`agent_core::TurnEngine`] into the corresponding [`SessionUpdate`],
//! forwarding it as it arrives (no buffering of the full turn). Permission
//! requests are issued as a blocking `session/request_permission` round-trip.
//!
//! NOTE: because permission requests block on a client response, the sink MUST
//! be driven from a spawned task (via [`ConnectionTo::spawn`]), never directly
//! on the JSON-RPC event loop, otherwise `block_task` would deadlock.

use agent_client_protocol::schema::v1::{
    Content, ContentBlock, ContentChunk, PermissionOption, PermissionOptionKind, Plan, PlanEntry,
    PlanEntryPriority, PlanEntryStatus, RequestPermissionOutcome, RequestPermissionRequest,
    SessionId, SessionNotification, SessionUpdate, TextContent, ToolCall, ToolCallContent,
    ToolCallLocation as AcpToolCallLocation, ToolCallStatus as AcpToolCallStatus, ToolCallUpdate,
    ToolCallUpdateFields, ToolKind as AcpToolKind,
};
use agent_client_protocol::{Client, ConnectionTo};
use agent_core::{
    AgentError, EngineOutput, PlanStepStatus, PlanUpdate, Result, ToolCallId, ToolCallLocation,
    ToolCallStatus, ToolKind, UpdateSink,
};
use async_trait::async_trait;
use tracing::debug;

use crate::turn_coordinator::TurnLease;

/// Permission option id used to signal the user granted the tool call.
const ALLOW_OPTION_ID: &str = "allow-once";
/// Permission option id used to signal the user denied the tool call.
const REJECT_OPTION_ID: &str = "reject-once";

/// An [`UpdateSink`] that emits ACP notifications over a live connection.
pub struct AcpUpdateSink {
    cx: ConnectionTo<Client>,
    session_id: SessionId,
    /// When `true` (YOLO permission mode), permission-gated tool calls are
    /// auto-granted without issuing a `session/request_permission` round-trip.
    yolo: bool,
}

impl AcpUpdateSink {
    /// Create a sink bound to a connection and session (Normal permission mode).
    #[must_use]
    pub fn new(cx: ConnectionTo<Client>, session_id: SessionId) -> Self {
        Self {
            cx,
            session_id,
            yolo: false,
        }
    }

    /// Set whether the sink auto-grants permission-gated tool calls (YOLO mode).
    #[must_use]
    pub fn with_yolo(mut self, yolo: bool) -> Self {
        self.yolo = yolo;
        self
    }

    /// Send a single `session/update` notification, mapping transport errors.
    fn notify(&self, update: SessionUpdate) -> Result<()> {
        self.cx
            .send_notification(SessionNotification::new(self.session_id.clone(), update))
            .map_err(|err| AgentError::Other(format!("failed to send session/update: {err}")))
    }
}

/// An output fence that only forwards events while its turn generation is current.
pub struct GenerationScopedSink<S> {
    inner: S,
    lease: TurnLease,
}

impl<S> GenerationScopedSink<S> {
    /// Bind `inner` to the lifecycle ownership represented by `lease`.
    #[must_use]
    pub fn new(inner: S, lease: TurnLease) -> Self {
        Self { inner, lease }
    }
}

#[async_trait]
impl<S: UpdateSink> UpdateSink for GenerationScopedSink<S> {
    async fn send(&mut self, output: EngineOutput) -> Result<()> {
        if !self.lease.is_current() {
            debug!(
                session_id = self.lease.session_id(),
                generation = self.lease.generation(),
                event_kind = engine_output_kind(&output),
                "dropped output from retired turn generation"
            );
            return Ok(());
        }
        self.inner.send(output).await
    }

    async fn request_permission(&mut self, id: &ToolCallId, tool_name: &str) -> Result<bool> {
        if !self.lease.is_current() {
            debug!(
                session_id = self.lease.session_id(),
                generation = self.lease.generation(),
                tool = tool_name,
                "denied permission locally for retired turn generation"
            );
            return Ok(false);
        }
        let cancel = self.lease.cancellation_token();
        tokio::select! {
            biased;
            _ = cancel.cancelled() => Ok(false),
            result = self.inner.request_permission(id, tool_name) => result,
        }
    }
}

fn engine_output_kind(output: &EngineOutput) -> &'static str {
    match output {
        EngineOutput::MessageChunk(_) => "message",
        EngineOutput::ThinkingChunk(_) => "thinking",
        EngineOutput::Plan(_) => "plan",
        EngineOutput::ToolCall { .. } => "tool_call",
        EngineOutput::ToolCallUpdate { .. } => "tool_call_update",
    }
}

#[async_trait]
impl UpdateSink for AcpUpdateSink {
    async fn send(&mut self, output: EngineOutput) -> Result<()> {
        match output {
            EngineOutput::MessageChunk(text) => self.notify(SessionUpdate::AgentMessageChunk(
                ContentChunk::new(text_block(text)),
            )),
            EngineOutput::ThinkingChunk(text) => self.notify(SessionUpdate::AgentThoughtChunk(
                ContentChunk::new(text_block(text)),
            )),
            EngineOutput::Plan(plan) => self.notify(SessionUpdate::Plan(map_plan(&plan))),
            EngineOutput::ToolCall {
                id,
                name: _,
                title,
                kind,
                status,
                locations,
                raw_input,
            } => {
                let mut tool_call = ToolCall::new(id.0, title)
                    .kind(map_kind(kind))
                    .status(map_status(status));
                if !locations.is_empty() {
                    tool_call = tool_call.locations(locations.iter().map(map_location).collect());
                }
                if let Some(input) = raw_input {
                    tool_call = tool_call.raw_input(input);
                }
                self.notify(SessionUpdate::ToolCall(tool_call))
            }
            EngineOutput::ToolCallUpdate {
                id,
                status,
                output,
                raw_output,
            } => {
                let mut fields = ToolCallUpdateFields::new().status(map_status(status));
                if let Some(text) = output {
                    if !text.is_empty() {
                        fields = fields.content(vec![ToolCallContent::Content(Content::new(
                            text_block(text),
                        ))]);
                    }
                }
                if let Some(out) = raw_output {
                    fields = fields.raw_output(out);
                }
                self.notify(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                    id.0, fields,
                )))
            }
        }
    }

    async fn request_permission(&mut self, id: &ToolCallId, tool_name: &str) -> Result<bool> {
        // YOLO mode: auto-grant without prompting the client.
        if self.yolo {
            debug!(tool = tool_name, "permission auto-granted (YOLO mode)");
            return Ok(true);
        }
        let tool_call = ToolCallUpdate::new(
            id.0.clone(),
            ToolCallUpdateFields::new()
                .title(Some(tool_name.to_string()))
                .status(AcpToolCallStatus::Pending),
        );
        let options = vec![
            PermissionOption::new(ALLOW_OPTION_ID, "Allow", PermissionOptionKind::AllowOnce),
            PermissionOption::new(REJECT_OPTION_ID, "Reject", PermissionOptionKind::RejectOnce),
        ];
        let request = RequestPermissionRequest::new(self.session_id.clone(), tool_call, options);

        let response = self
            .cx
            .send_request(request)
            .block_task()
            .await
            .map_err(|err| {
                AgentError::Other(format!("session/request_permission failed: {err}"))
            })?;

        let granted = match response.outcome {
            RequestPermissionOutcome::Selected(selected) => {
                &*selected.option_id.0 == ALLOW_OPTION_ID
            }
            // Cancelled (or any future variant) is treated as "not granted".
            _ => false,
        };
        debug!(tool = tool_name, granted, "permission decision received");
        Ok(granted)
    }
}

/// Wrap a plain string as a text [`ContentBlock`].
fn text_block(text: String) -> ContentBlock {
    ContentBlock::Text(TextContent::new(text))
}

/// Map a provider-agnostic [`ToolKind`] to the ACP schema equivalent.
fn map_kind(kind: ToolKind) -> AcpToolKind {
    match kind {
        ToolKind::Read => AcpToolKind::Read,
        ToolKind::Edit => AcpToolKind::Edit,
        ToolKind::Delete => AcpToolKind::Delete,
        ToolKind::Move => AcpToolKind::Move,
        ToolKind::Search => AcpToolKind::Search,
        ToolKind::Execute => AcpToolKind::Execute,
        ToolKind::Think => AcpToolKind::Think,
        ToolKind::Fetch => AcpToolKind::Fetch,
        ToolKind::Other => AcpToolKind::Other,
    }
}

/// Map a provider-agnostic [`ToolCallLocation`] to the ACP schema equivalent.
fn map_location(location: &ToolCallLocation) -> AcpToolCallLocation {
    let acp = AcpToolCallLocation::new(location.path.clone());
    match location.line {
        Some(line) => acp.line(line),
        None => acp,
    }
}

/// Map an [`agent_core`] tool-call status to the ACP schema equivalent.
fn map_status(status: ToolCallStatus) -> AcpToolCallStatus {
    match status {
        ToolCallStatus::Pending => AcpToolCallStatus::Pending,
        ToolCallStatus::InProgress => AcpToolCallStatus::InProgress,
        ToolCallStatus::Completed => AcpToolCallStatus::Completed,
        ToolCallStatus::Failed => AcpToolCallStatus::Failed,
    }
}

/// Map a provider-agnostic [`PlanUpdate`] to an ACP [`Plan`].
pub(crate) fn map_plan(plan: &PlanUpdate) -> Plan {
    let entries = plan
        .steps
        .iter()
        .map(|step| {
            PlanEntry::new(
                step.content.clone(),
                PlanEntryPriority::Medium,
                map_plan_status(step.status),
            )
        })
        .collect();
    Plan::new(entries)
}

/// Map a plan-step status to the ACP plan-entry status.
fn map_plan_status(status: PlanStepStatus) -> PlanEntryStatus {
    match status {
        PlanStepStatus::Pending => PlanEntryStatus::Pending,
        PlanStepStatus::InProgress => PlanEntryStatus::InProgress,
        PlanStepStatus::Completed => PlanEntryStatus::Completed,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use agent_core::{PlanStep, ToolCallLocation};

    use super::*;
    use crate::turn_coordinator::TurnCoordinator;

    #[derive(Clone, Default)]
    struct CollectingSink {
        outputs: Arc<Mutex<Vec<EngineOutput>>>,
        permission_requests: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl UpdateSink for CollectingSink {
        async fn send(&mut self, output: EngineOutput) -> Result<()> {
            self.outputs.lock().unwrap().push(output);
            Ok(())
        }

        async fn request_permission(&mut self, _id: &ToolCallId, _tool_name: &str) -> Result<bool> {
            self.permission_requests.fetch_add(1, Ordering::SeqCst);
            Ok(true)
        }
    }

    fn every_output_kind() -> Vec<EngineOutput> {
        vec![
            EngineOutput::MessageChunk("message".to_string()),
            EngineOutput::ThinkingChunk("thought".to_string()),
            EngineOutput::Plan(PlanUpdate {
                steps: vec![PlanStep {
                    content: "step".to_string(),
                    status: PlanStepStatus::InProgress,
                }],
            }),
            EngineOutput::ToolCall {
                id: ToolCallId("call".to_string()),
                name: "fs_write".to_string(),
                title: "Write file".to_string(),
                kind: ToolKind::Edit,
                status: ToolCallStatus::Pending,
                locations: vec![ToolCallLocation::new("/tmp/file")],
                raw_input: Some(serde_json::json!({"path": "/tmp/file"})),
            },
            EngineOutput::ToolCallUpdate {
                id: ToolCallId("call".to_string()),
                status: ToolCallStatus::Completed,
                output: Some("done".to_string()),
                raw_output: Some(serde_json::json!({"ok": true})),
            },
        ]
    }

    #[tokio::test]
    async fn current_generation_forwards_every_output_kind_and_permission() {
        let coordinator = TurnCoordinator::with_retirement_grace("s", Duration::ZERO);
        let lease = coordinator.begin_turn().await;
        let inner = CollectingSink::default();
        let outputs = inner.outputs.clone();
        let permissions = inner.permission_requests.clone();
        let mut sink = GenerationScopedSink::new(inner, lease);

        for output in every_output_kind() {
            sink.send(output).await.unwrap();
        }
        assert!(sink
            .request_permission(&ToolCallId("call".to_string()), "fs_write")
            .await
            .unwrap());

        assert_eq!(outputs.lock().unwrap().len(), 5);
        assert_eq!(permissions.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retired_generation_drops_every_output_kind_and_denies_permission() {
        let coordinator = TurnCoordinator::with_retirement_grace("s", Duration::ZERO);
        let retired = coordinator.begin_turn().await;
        let inner = CollectingSink::default();
        let outputs = inner.outputs.clone();
        let permissions = inner.permission_requests.clone();
        let mut sink = GenerationScopedSink::new(inner, retired);
        let _current = coordinator.begin_turn().await;

        for output in every_output_kind() {
            sink.send(output).await.unwrap();
        }
        assert!(!sink
            .request_permission(&ToolCallId("call".to_string()), "fs_write")
            .await
            .unwrap());

        assert!(outputs.lock().unwrap().is_empty());
        assert_eq!(permissions.load(Ordering::SeqCst), 0);
    }
}
