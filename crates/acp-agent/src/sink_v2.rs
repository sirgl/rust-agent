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

use std::collections::HashSet;

use agent_client_protocol::schema::v2::{
    ContentBlock, ContentChunk, Diff as AcpDiff, DiffChange, DiffFileType, MessageId,
    PermissionOption, PermissionOptionKind, PlanEntry, PlanEntryPriority, PlanEntryStatus,
    PlanUpdate as AcpPlanUpdate, PlanUpdateContent, RequestPermissionOutcome,
    RequestPermissionRequest, RequestPermissionSubject, RequiresActionStateUpdate,
    RunningStateUpdate, SessionId, SessionUpdate, StateUpdate, Terminal, TerminalExitStatus,
    TerminalOutputChunk, TerminalUpdate, TextContent, ToolCallContentChunk,
    ToolCallLocation as AcpToolCallLocation, ToolCallStatus as AcpToolCallStatus, ToolCallUpdate,
    ToolKind as AcpToolKind, UpdateSessionNotification,
};
use agent_client_protocol::{Client, ConnectionTo};
use agent_core::{
    AgentError, EngineOutput, FileDiff, PlanStepStatus, PlanUpdate, Result, ToolCallId,
    ToolCallLocation, ToolCallStatus, ToolKind, UpdateSink,
};
use async_trait::async_trait;
use base64::Engine as _;
use tracing::debug;

/// Permission option id used to signal the user granted the tool call.
const ALLOW_OPTION_ID: &str = "allow-once";
/// Permission option id used to signal the user denied the tool call.
const REJECT_OPTION_ID: &str = "reject-once";

#[derive(
    Debug, Clone, serde::Serialize, serde::Deserialize, agent_client_protocol::JsonRpcNotification,
)]
#[notification(method = "session/update")]
#[serde(rename_all = "camelCase")]
struct SpecUpdateSessionNotification {
    session_id: SessionId,
    update: serde_json::Value,
}

/// An [`UpdateSink`] that emits ACP notifications over a live connection.
pub struct AcpUpdateSink {
    cx: ConnectionTo<Client>,
    session_id: SessionId,
    message_id: MessageId,
    thought_id: MessageId,
    terminals: HashSet<String>,
    /// When `true` (YOLO permission mode), permission-gated tool calls are
    /// auto-granted without issuing a `session/request_permission` round-trip.
    yolo: bool,
}

impl AcpUpdateSink {
    /// Create a sink bound to a connection and session (Normal permission mode).
    #[must_use]
    pub fn new(cx: ConnectionTo<Client>, session_id: SessionId) -> Self {
        let suffix = next_message_id();
        Self {
            cx,
            session_id,
            message_id: MessageId::new(format!("agent-{suffix}")),
            thought_id: MessageId::new(format!("thought-{suffix}")),
            terminals: HashSet::new(),
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
            .send_notification(UpdateSessionNotification::new(
                self.session_id.clone(),
                update,
            ))
            .map_err(|err| AgentError::Other(format!("failed to send session/update: {err}")))
    }

    fn notify_spec_update(&self, update: SessionUpdate) -> Result<()> {
        let mut update = serde_json::to_value(update).map_err(|err| {
            AgentError::Other(format!("failed to serialize session/update: {err}"))
        })?;
        normalize_diff_patch(&mut update);
        self.cx
            .send_notification(SpecUpdateSessionNotification {
                session_id: self.session_id.clone(),
                update,
            })
            .map_err(|err| AgentError::Other(format!("failed to send session/update: {err}")))
    }
}

#[async_trait]
impl UpdateSink for AcpUpdateSink {
    async fn send(&mut self, output: EngineOutput) -> Result<()> {
        match output {
            EngineOutput::MessageChunk(text) => self.notify(SessionUpdate::AgentMessageChunk(
                ContentChunk::new(text_block(text), self.message_id.clone()),
            )),
            EngineOutput::ThinkingChunk(text) => self.notify(SessionUpdate::AgentThoughtChunk(
                ContentChunk::new(text_block(text), self.thought_id.clone()),
            )),
            EngineOutput::Plan(plan) => self.notify(SessionUpdate::PlanUpdate(map_plan(&plan))),
            EngineOutput::ToolCall {
                id,
                name: _,
                title,
                kind,
                status,
                locations,
                raw_input,
            } => {
                let tool_call_id = id.0;
                let is_terminal = matches!(kind, ToolKind::Execute);
                let terminal_command = raw_input
                    .as_ref()
                    .and_then(|input| input.get("command"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned);
                let mut tool_call = ToolCallUpdate::new(tool_call_id.clone())
                    .title(title)
                    .kind(map_kind(kind))
                    .status(map_status(status));
                if !locations.is_empty() {
                    tool_call =
                        tool_call.locations(locations.iter().map(map_location).collect::<Vec<_>>());
                }
                if let Some(input) = raw_input {
                    tool_call = tool_call.raw_input(input);
                }
                if is_terminal {
                    tool_call = tool_call.content(vec![Terminal::new(tool_call_id.clone()).into()]);
                    self.terminals.insert(tool_call_id.clone());
                }
                self.notify(SessionUpdate::ToolCallUpdate(tool_call))?;
                if is_terminal {
                    let mut terminal = TerminalUpdate::new(tool_call_id);
                    if let Some(command) = terminal_command {
                        terminal = terminal.command(command);
                    }
                    self.notify(SessionUpdate::TerminalUpdate(terminal))?;
                }
                Ok(())
            }
            EngineOutput::ToolCallUpdate {
                id,
                status,
                output,
                raw_output,
                file_diffs,
            } => {
                let tool_call_id = id.0;
                let is_terminal = self.terminals.contains(&tool_call_id);
                let terminal_finished =
                    matches!(status, ToolCallStatus::Completed | ToolCallStatus::Failed);
                let mut update =
                    ToolCallUpdate::new(tool_call_id.clone()).status(map_status(status));
                if let Some(out) = raw_output {
                    update = update.raw_output(out);
                }
                let has_file_diffs = !file_diffs.is_empty();
                if let Some(diff) = map_file_diffs(&file_diffs) {
                    update = update.content(vec![diff.into()]);
                }
                let update = SessionUpdate::ToolCallUpdate(update);
                if has_file_diffs {
                    self.notify_spec_update(update)?;
                } else {
                    self.notify(update)?;
                }
                if let Some(text) = output.filter(|text| !text.is_empty()) {
                    if is_terminal {
                        let data =
                            base64::engine::general_purpose::STANDARD.encode(text.as_bytes());
                        self.notify(SessionUpdate::TerminalOutputChunk(
                            TerminalOutputChunk::new(tool_call_id.clone(), data),
                        ))?;
                    } else {
                        self.notify(SessionUpdate::ToolCallContentChunk(
                            ToolCallContentChunk::new(tool_call_id.clone(), text_block(text)),
                        ))?;
                    }
                }
                if is_terminal && terminal_finished {
                    self.notify(SessionUpdate::TerminalUpdate(
                        TerminalUpdate::new(tool_call_id.clone())
                            .exit_status(TerminalExitStatus::new()),
                    ))?;
                    self.terminals.remove(&tool_call_id);
                }
                Ok(())
            }
        }
    }

    async fn request_permission(&mut self, id: &ToolCallId, tool_name: &str) -> Result<bool> {
        // YOLO mode: auto-grant without prompting the client.
        if self.yolo {
            debug!(tool = tool_name, "permission auto-granted (YOLO mode)");
            return Ok(true);
        }
        let tool_call = ToolCallUpdate::new(id.0.clone())
            .title(tool_name.to_string())
            .status(AcpToolCallStatus::Pending);
        let options = vec![
            PermissionOption::new(ALLOW_OPTION_ID, "Allow", PermissionOptionKind::AllowOnce),
            PermissionOption::new(REJECT_OPTION_ID, "Reject", PermissionOptionKind::RejectOnce),
        ];
        let request = RequestPermissionRequest::new(self.session_id.clone(), tool_name, options)
            .subject(RequestPermissionSubject::from(tool_call));

        self.notify(SessionUpdate::StateUpdate(StateUpdate::RequiresAction(
            RequiresActionStateUpdate::new(),
        )))?;

        let response = self.cx.send_request(request).block_task().await;
        let running = self.notify(SessionUpdate::StateUpdate(StateUpdate::Running(
            RunningStateUpdate::new(),
        )));
        let response = response.map_err(|err| {
            AgentError::Other(format!("session/request_permission failed: {err}"))
        })?;
        running?;

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

fn normalize_diff_patch(update: &mut serde_json::Value) {
    let Some(content) = update
        .get_mut("content")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return;
    };
    for item in content {
        if item.get("type").and_then(serde_json::Value::as_str) != Some("diff") {
            continue;
        }
        let Some(patch) = item
            .get_mut("patch")
            .and_then(serde_json::Value::as_object_mut)
        else {
            continue;
        };
        if !patch.contains_key("diff") {
            if let Some(text) = patch.remove("text") {
                patch.insert("diff".to_string(), text);
            }
        }
    }
}

fn map_file_diffs(diffs: &[FileDiff]) -> Option<AcpDiff> {
    if diffs.is_empty() {
        return None;
    }
    let mut changes = Vec::with_capacity(diffs.len());
    let mut patch = String::new();
    for diff in diffs {
        let change = if diff.old_text.is_some() {
            DiffChange::modify(diff.path.clone())
        } else {
            DiffChange::add(diff.path.clone())
        }
        .file_type(DiffFileType::Text);
        changes.push(change);
        patch.push_str(&git_patch(diff));
    }
    Some(AcpDiff::patch(patch, changes))
}

fn git_patch(diff: &FileDiff) -> String {
    let old_text = diff.old_text.as_deref().unwrap_or_default();
    let unified = diffy::create_patch(old_text, &diff.new_text).to_string();
    let hunks = unified.lines().skip(2).collect::<Vec<_>>().join("\n");
    let old_path = diff
        .old_text
        .as_ref()
        .map_or("/dev/null", |_| diff.path.as_str());
    let mut patch = format!(
        "diff --git {path} {path}\n--- {old_path}\n+++ {path}\n{hunks}",
        path = diff.path,
    );
    if unified.ends_with('\n') {
        patch.push('\n');
    }
    patch
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

/// Map a provider-agnostic [`PlanUpdate`] to an ACP v2 plan upsert.
pub(crate) fn map_plan(plan: &PlanUpdate) -> AcpPlanUpdate {
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
    AcpPlanUpdate::new(PlanUpdateContent::items("main", entries))
}

/// Generate a process-unique id shared by all chunks of one sink message.
fn next_message_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
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
    use super::*;

    #[test]
    fn diff_patch_uses_the_v2_wire_field() {
        let diff = FileDiff {
            path: "src/main.rs".to_string(),
            old_text: Some("old\n".to_string()),
            new_text: "new\n".to_string(),
        };
        let content = map_file_diffs(&[diff]).expect("file diff");
        let mut update = serde_json::to_value(SessionUpdate::ToolCallUpdate(
            ToolCallUpdate::new("write-file")
                .status(AcpToolCallStatus::Completed)
                .content(vec![content.into()]),
        ))
        .unwrap();

        normalize_diff_patch(&mut update);

        assert_eq!(update["content"][0]["patch"]["format"], "git_patch");
        assert!(update["content"][0]["patch"]["diff"]
            .as_str()
            .expect("patch diff")
            .contains("-old"));
        assert!(update["content"][0]["patch"].get("text").is_none());
    }
}
