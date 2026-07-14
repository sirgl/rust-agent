//! The provider-agnostic streaming event model produced by the decision layer.

use serde::{Deserialize, Serialize};

use crate::error::TurnError;
use crate::history::ThinkingRecord;

/// Identifier for a single tool call within a turn.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolCallId(pub String);

impl ToolCallId {
    /// Create a new tool call id from anything string-like.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the underlying string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ToolCallId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Why a turn finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model reached a natural end of its turn.
    EndTurn,
    /// The maximum token budget was reached.
    MaxTokens,
    /// The turn stopped to allow tool execution and will be resumed.
    ToolUse,
    /// The turn was cancelled by the client.
    Cancelled,
    /// The turn stopped due to a refusal or content policy.
    Refusal,
}

/// A single plan step surfaced to the client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanStep {
    /// Short description of the step.
    pub content: String,
    /// Execution status of the step.
    pub status: PlanStepStatus,
}

/// Status of a plan step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStepStatus {
    /// Not started yet.
    Pending,
    /// Currently in progress.
    InProgress,
    /// Finished.
    Completed,
}

/// A full plan update (the current list of steps).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PlanUpdate {
    /// The ordered plan steps.
    pub steps: Vec<PlanStep>,
}

/// Token usage reported by the model provider for a decision round.
///
/// Fields mirror the Anthropic `usage` object. `input_tokens` and
/// `output_tokens` count *fresh* (uncached) tokens; cached prompt tokens are
/// reported separately: `cache_read_input_tokens` are tokens **reused** from an
/// existing prompt cache (billed at a large discount), while
/// `cache_creation_input_tokens` are tokens written into the cache this round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Fresh (uncached) input tokens.
    pub input_tokens: u64,
    /// Output (sampled) tokens.
    pub output_tokens: u64,
    /// Cached input tokens that were reused this round.
    pub cache_read_input_tokens: u64,
    /// Input tokens written into the prompt cache this round.
    pub cache_creation_input_tokens: u64,
}

impl TokenUsage {
    /// Accumulate another usage record into this one (field-wise sum).
    pub fn add(&mut self, other: TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_read_input_tokens += other.cache_read_input_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
    }

    /// Total input tokens across fresh, cache-read and cache-creation.
    pub fn total_input_tokens(&self) -> u64 {
        self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens
    }
}

/// A streaming decision event describing what the agent does next.
#[derive(Debug, Clone)]
pub enum TurnEvent {
    /// An incremental chunk of assistant-visible text.
    TextDelta(String),
    /// An incremental chunk of internal reasoning ("thinking").
    ///
    /// Emitted live for streaming display. It is *not* persisted on its own; the
    /// complete, signed block arrives separately as [`TurnEvent::ThinkingBlock`]
    /// once the reasoning block closes.
    Thinking(String),
    /// A complete extended-thinking block (with its signature or redacted data).
    ///
    /// Persisted verbatim into history so it can be replayed back to the
    /// provider on the next turn, which providers require when thinking is used
    /// together with tools.
    ThinkingBlock(ThinkingRecord),
    /// An updated plan.
    Plan(PlanUpdate),
    /// The decision layer requests that a tool be executed.
    ToolCallRequested {
        /// Unique id for correlating updates.
        id: ToolCallId,
        /// Name of the tool to invoke.
        name: String,
        /// Arguments as arbitrary JSON.
        arguments: serde_json::Value,
    },
    /// Token usage reported for the current decision round.
    Usage(TokenUsage),
    /// The turn finished with the given stop reason.
    TurnFinished {
        /// The reason the turn ended.
        stop_reason: StopReason,
    },
    /// An error occurred while producing the turn.
    Error(TurnError),
}
