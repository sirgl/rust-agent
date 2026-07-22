//! Provider-neutral execution tracing for agent turns and nested runs.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

use crate::{StopReason, TokenUsage};

static NEXT_TRACE_ID: AtomicU64 = AtomicU64::new(1);

/// Create a process-unique, human-readable trace identifier.
#[must_use]
pub fn next_trace_id(kind: &str) -> String {
    format!(
        "{kind}-{}-{}",
        std::process::id(),
        NEXT_TRACE_ID.fetch_add(1, Ordering::Relaxed)
    )
}

/// Optional semantic attributes describing an agent run.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct AgentRunMetadata {
    /// Agent family or mode (`root`, `orchestrator`, `code`, `review`, ...).
    pub kind: Option<String>,
    /// Orchestrated mode id, when applicable.
    pub mode: Option<String>,
    /// Zero-based plan step, when applicable.
    pub step_index: Option<usize>,
    /// One-based attempt number, when applicable.
    pub attempt: Option<usize>,
    /// Provider-neutral model tier, when applicable.
    pub model_tier: Option<String>,
}

/// Stable causal identity shared by records belonging to one execution node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TraceContext {
    /// One user prompt and all work it caused.
    pub turn_id: String,
    /// One concrete root/orchestrator/sub-agent invocation.
    pub agent_run_id: String,
    /// Parent run, absent for the root agent.
    pub parent_agent_run_id: Option<String>,
    /// Tool call in the parent that spawned this run.
    pub spawned_by_tool_call_id: Option<String>,
    /// One decision-layer request/response pair, absent on run-level events.
    pub round_id: Option<String>,
    /// Display-only breadcrumb; identity never depends on this value.
    pub agent_path: String,
    /// Semantic attributes for grouping retries and plan steps.
    pub metadata: AgentRunMetadata,
}

impl TraceContext {
    /// Return this run context scoped to a concrete decision round.
    #[must_use]
    pub fn with_round(&self, round_id: impl Into<String>) -> Self {
        let mut trace = self.clone();
        trace.round_id = Some(round_id.into());
        trace
    }
}

/// Final state of a lifecycle node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceStatus {
    /// Work is currently executing.
    Running,
    /// Work finished normally.
    Completed,
    /// Work returned an error or failed result.
    Failed,
    /// Work observed cancellation.
    Cancelled,
    /// A start event has no matching finish event (derived after a crash).
    Interrupted,
}

/// Append-only lifecycle event consumed by debug observers.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(missing_docs)] // Field names form the self-describing JSON debug-event contract.
pub enum TraceEvent {
    TurnStarted {
        trace: TraceContext,
        session_id: String,
    },
    TurnFinished {
        trace: TraceContext,
        session_id: String,
        status: TraceStatus,
        stop_reason: Option<StopReason>,
        error: Option<String>,
        duration_ms: u64,
    },
    AgentRunStarted {
        trace: TraceContext,
        session_id: String,
    },
    AgentRunFinished {
        trace: TraceContext,
        session_id: String,
        status: TraceStatus,
        stop_reason: Option<StopReason>,
        error: Option<String>,
        duration_ms: u64,
    },
    RoundStarted {
        trace: TraceContext,
        session_id: String,
    },
    RoundFinished {
        trace: TraceContext,
        session_id: String,
        status: TraceStatus,
        stop_reason: Option<StopReason>,
        error: Option<String>,
        usage: TokenUsage,
        duration_ms: u64,
    },
    ToolCallStarted {
        trace: TraceContext,
        session_id: String,
        tool_call_id: String,
        name: String,
        arguments: serde_json::Value,
    },
    ToolCallFinished {
        trace: TraceContext,
        session_id: String,
        tool_call_id: String,
        status: TraceStatus,
        output: Option<String>,
        duration_ms: u64,
    },
}
