//! Conversation history compaction.
//!
//! Compaction reduces the size of a [`Conversation`] before it is compiled into
//! a provider request, so long-running sessions stay within the model's context
//! window without losing the information the agent actually needs.
//!
//! The design mirrors the three-level model used by Matterhorn (see
//! `docs/design/compaction.md`), expressed here in a type-safe, provider-agnostic
//! way:
//!
//! - [`HeuristicCompactor`] — a cheap, token-free pass that drops "noisy" tool
//!   interactions (searches, reads, directory listings, …) from the *prefix* of
//!   the history while keeping a verbatim *tail* of the most recent entries.
//! - [`LlmCompactor`] — an optional "smart" pass that replaces the raw output of
//!   long tool results with a short LLM-produced summary, falling back to the
//!   original content whenever summarization fails.
//! - [`ChainedCompactor`] — runs several compactors in sequence (typically the
//!   cheap heuristic first, then the LLM pass) so each stage only has to deal
//!   with what the previous one could not shrink.
//!
//! Every compactor preserves the `tool_call`/`tool_result` pairing invariant:
//! when a [`HistoryEntry::ToolCall`] is removed, its matching
//! [`HistoryEntry::ToolResult`] (correlated by [`ToolCallId`]) is removed too,
//! and vice versa, so the resulting conversation is always valid for a provider.

use std::collections::HashSet;

use async_trait::async_trait;

use crate::error::Result;
use crate::event::ToolCallId;
use crate::history::{Conversation, ContentBlock, HistoryEntry, KnownTool, ToolResultRecord};

/// Number of most-recent entries always preserved verbatim (the "tail").
///
/// Mirrors Matterhorn's `KEEP_TAIL`.
pub const DEFAULT_KEEP_TAIL: usize = 10;

/// Do not attempt any compaction while the estimated size stays below this many
/// characters. Acts as the "early exit" guard so short sessions are untouched.
pub const DEFAULT_MIN_CHARS: usize = 24_000;

/// A tool result at or below this many characters is considered "short" and is
/// never sent to the LLM for summarization (mirrors `SHORT_OUTPUT_MAX_CHARS`).
pub const SHORT_OUTPUT_MAX_CHARS: usize = 100;

/// A tool result at or below this many lines is considered "short" and is never
/// summarized (mirrors `SHORT_OUTPUT_MAX_LINES`).
pub const SHORT_OUTPUT_MAX_LINES: usize = 3;

/// System prompt used by [`LlmCompactor`] when summarizing a tool result.
///
/// It explicitly tells the model that its summary *replaces* the raw output,
/// while the originating command stays in the history, so the model must not
/// repeat the command itself.
pub const SUMMARIZER_SYSTEM_PROMPT: &str =
    "Your task is to summarize the output of a single tool/command. \
Your summary will REPLACE the raw output in the conversation history, while the \
command itself stays in the history — so do not repeat the command. Keep only \
the information that is important for continuing the task: results, key values, \
errors and their reasons. Be concise and factual; do not add interpretations.";

/// The default set of tool names treated as "noisy": their results were already
/// consumed when the agent acted on them, so they can be dropped from the prefix.
///
/// Built-in `fs_read` is included by name; unknown ([`KnownTool::Other`]) tools
/// whose names contain any of these fragments are also treated as noisy (see
/// [`is_noisy_tool`]).
pub const DEFAULT_NOISY_TOOLS: &[&str] = &[
    "fs_read",
    "search",
    "grep",
    "glob",
    "list",
    "open",
    "scroll",
];

/// Provider-agnostic single-shot text summarization seam used by
/// [`LlmCompactor`].
///
/// Implemented by the decision layer (backed by a real model) and by test fakes.
/// Kept intentionally minimal — a `(system, user) -> text` call — so compaction
/// never depends on a concrete provider.
#[async_trait]
pub trait TextSummarizer: Send + Sync {
    /// Summarize `user` content under the guidance of the `system` prompt,
    /// returning the summary text.
    async fn summarize(&self, system: &str, user: &str) -> Result<String>;
}

/// Reduces the size of a [`Conversation`] while preserving its validity.
#[async_trait]
pub trait Compactor: Send + Sync {
    /// Return a compacted copy of `conv`.
    ///
    /// Implementations must keep the `tool_call`/`tool_result` pairing intact
    /// and must never fail destructively: on any internal error they should
    /// return the input (or a best-effort partial result) rather than losing
    /// context.
    async fn compact(&self, conv: Conversation) -> Result<Conversation>;
}

/// Estimate the character size of a conversation (used for early-exit guards).
pub fn estimate_len(conv: &Conversation) -> usize {
    conv.entries.iter().map(entry_len).sum()
}

fn entry_len(entry: &HistoryEntry) -> usize {
    match entry {
        HistoryEntry::User(m) => content_len(&m.content),
        HistoryEntry::Assistant(m) => content_len(&m.content),
        HistoryEntry::Thinking(t) => match t {
            crate::history::ThinkingRecord::Thinking { text, signature } => {
                text.len() + signature.len()
            }
            crate::history::ThinkingRecord::Redacted { data } => data.len(),
        },
        HistoryEntry::ToolCall(c) => c.tool.arguments().to_string().len(),
        HistoryEntry::ToolResult(r) => content_len(&r.content),
        HistoryEntry::Ingested(i) => i.payload.to_string().len(),
    }
}

fn content_len(content: &[ContentBlock]) -> usize {
    content
        .iter()
        .map(|b| match b {
            ContentBlock::Text { text } => text.len(),
        })
        .sum()
}

/// Whether `tool` is considered "noisy" against the given set of name fragments.
///
/// A tool is noisy when its [`KnownTool::name`] contains any fragment in
/// `noisy`. This matches both built-in reads (e.g. `fs_read`) and unknown MCP
/// tools whose names embed a fragment such as `search`.
pub fn is_noisy_tool(tool: &KnownTool, noisy: &HashSet<String>) -> bool {
    let name = tool.name();
    noisy.iter().any(|frag| name.contains(frag.as_str()))
}

/// The flat text of a tool result (concatenated text blocks).
fn tool_result_text(result: &ToolResultRecord) -> String {
    result
        .content
        .iter()
        .map(|b| match b {
            ContentBlock::Text { text } => text.as_str(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Whether a tool result is "short" enough to keep verbatim (never summarized).
fn is_short_output(result: &ToolResultRecord) -> bool {
    let text = tool_result_text(result);
    text.len() <= SHORT_OUTPUT_MAX_CHARS && text.lines().count() <= SHORT_OUTPUT_MAX_LINES
}

/// Split index that keeps the last `keep_tail` entries as the tail.
fn tail_split(len: usize, keep_tail: usize) -> usize {
    len.saturating_sub(keep_tail)
}

/// A cheap, token-free compactor that drops noisy tool interactions from the
/// prefix while keeping a verbatim tail.
///
/// See the [module docs](crate::compaction) for the overall model.
#[derive(Debug, Clone)]
pub struct HeuristicCompactor {
    keep_tail: usize,
    min_chars: usize,
    noisy: HashSet<String>,
}

impl Default for HeuristicCompactor {
    fn default() -> Self {
        Self::new()
    }
}

impl HeuristicCompactor {
    /// Create a heuristic compactor with default thresholds and noisy tool set.
    pub fn new() -> Self {
        Self {
            keep_tail: DEFAULT_KEEP_TAIL,
            min_chars: DEFAULT_MIN_CHARS,
            noisy: DEFAULT_NOISY_TOOLS.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Override the number of tail entries kept verbatim.
    pub fn with_keep_tail(mut self, keep_tail: usize) -> Self {
        self.keep_tail = keep_tail;
        self
    }

    /// Override the early-exit character threshold.
    pub fn with_min_chars(mut self, min_chars: usize) -> Self {
        self.min_chars = min_chars;
        self
    }

    /// Replace the set of noisy tool-name fragments.
    pub fn with_noisy_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.noisy = tools.into_iter().map(Into::into).collect();
        self
    }

    /// Compute the set of [`ToolCallId`]s of noisy calls located in `prefix`.
    fn noisy_ids_in_prefix(&self, prefix: &[HistoryEntry]) -> HashSet<ToolCallId> {
        prefix
            .iter()
            .filter_map(|e| match e {
                HistoryEntry::ToolCall(c) if is_noisy_tool(&c.tool, &self.noisy) => {
                    Some(c.id.clone())
                }
                _ => None,
            })
            .collect()
    }
}

#[async_trait]
impl Compactor for HeuristicCompactor {
    async fn compact(&self, conv: Conversation) -> Result<Conversation> {
        // Early exit: nothing to gain on small or short histories.
        if conv.entries.len() <= self.keep_tail || estimate_len(&conv) <= self.min_chars {
            return Ok(conv);
        }

        let split = tail_split(conv.entries.len(), self.keep_tail);
        let (prefix, tail) = conv.entries.split_at(split);

        let drop_ids = self.noisy_ids_in_prefix(prefix);

        let mut entries: Vec<HistoryEntry> = prefix
            .iter()
            .filter(|e| match e {
                // Drop the noisy call itself …
                HistoryEntry::ToolCall(c) => !drop_ids.contains(&c.id),
                // … and its paired result, keeping the pairing invariant.
                HistoryEntry::ToolResult(r) => !drop_ids.contains(&r.id),
                _ => true,
            })
            .cloned()
            .collect();

        entries.extend_from_slice(tail);
        Ok(Conversation { entries })
    }
}

/// A "smart" compactor that replaces long tool-result outputs in the prefix with
/// short LLM-produced summaries.
///
/// Short outputs are passed through untouched, and any summarization failure
/// falls back to the original content so context is never lost.
pub struct LlmCompactor<S: TextSummarizer> {
    summarizer: S,
    keep_tail: usize,
    min_chars: usize,
}

impl<S: TextSummarizer> LlmCompactor<S> {
    /// Create an LLM compactor from a summarizer, with default thresholds.
    pub fn new(summarizer: S) -> Self {
        Self {
            summarizer,
            keep_tail: DEFAULT_KEEP_TAIL,
            min_chars: DEFAULT_MIN_CHARS,
        }
    }

    /// Override the number of tail entries kept verbatim.
    pub fn with_keep_tail(mut self, keep_tail: usize) -> Self {
        self.keep_tail = keep_tail;
        self
    }

    /// Override the early-exit character threshold.
    pub fn with_min_chars(mut self, min_chars: usize) -> Self {
        self.min_chars = min_chars;
        self
    }

    /// Summarize a single tool result, returning a replacement record.
    ///
    /// On failure or empty output, the original record is returned unchanged.
    async fn summarize_result(&self, result: &ToolResultRecord) -> ToolResultRecord {
        let original = tool_result_text(result);
        match self.summarizer.summarize(SUMMARIZER_SYSTEM_PROMPT, &original).await {
            // Guard against unhelpful summaries: empty, or not actually shorter.
            Ok(summary) if !summary.trim().is_empty() && summary.len() < original.len() => {
                ToolResultRecord {
                    id: result.id.clone(),
                    success: result.success,
                    content: vec![ContentBlock::text(summary)],
                }
            }
            _ => result.clone(),
        }
    }
}

#[async_trait]
impl<S: TextSummarizer> Compactor for LlmCompactor<S> {
    async fn compact(&self, conv: Conversation) -> Result<Conversation> {
        if conv.entries.len() <= self.keep_tail || estimate_len(&conv) <= self.min_chars {
            return Ok(conv);
        }

        let split = tail_split(conv.entries.len(), self.keep_tail);
        let (prefix, tail) = conv.entries.split_at(split);

        let mut entries = Vec::with_capacity(conv.entries.len());
        for entry in prefix {
            match entry {
                HistoryEntry::ToolResult(r) if !is_short_output(r) => {
                    entries.push(HistoryEntry::ToolResult(self.summarize_result(r).await));
                }
                other => entries.push(other.clone()),
            }
        }
        entries.extend_from_slice(tail);
        Ok(Conversation { entries })
    }
}

/// Runs a sequence of compactors in order, feeding each one the output of the
/// previous. Typically the cheap [`HeuristicCompactor`] first, then an
/// [`LlmCompactor`].
pub struct ChainedCompactor {
    stages: Vec<Box<dyn Compactor>>,
}

impl ChainedCompactor {
    /// Create a chain from an ordered list of compactor stages.
    pub fn new(stages: Vec<Box<dyn Compactor>>) -> Self {
        Self { stages }
    }
}

#[async_trait]
impl Compactor for ChainedCompactor {
    async fn compact(&self, conv: Conversation) -> Result<Conversation> {
        let mut current = conv;
        for stage in &self.stages {
            current = stage.compact(current).await?;
        }
        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::{EditToolCall, FsReadToolCall};

    fn noisy_conv(pairs: usize) -> Conversation {
        let mut conv = Conversation::new();
        conv.push_user_text("do the task with a fairly long instruction ".repeat(200));
        for i in 0..pairs {
            let id = ToolCallId::new(format!("call-{i}"));
            conv.push_tool_call(id.clone(), KnownTool::FsRead(FsReadToolCall { path: format!("/f{i}") }));
            conv.push_tool_result(id, true, "file contents that are quite long ".repeat(50));
        }
        conv
    }

    #[tokio::test]
    async fn heuristic_drops_noisy_prefix_pairs_but_keeps_tail() {
        let conv = noisy_conv(20);
        let before = conv.entries.len();
        let compactor = HeuristicCompactor::new().with_keep_tail(4).with_min_chars(0);
        let out = compactor.compact(conv).await.unwrap();
        assert!(out.entries.len() < before);
        // Tail (last 4 = 2 noisy pairs) is preserved verbatim.
        let read_calls = out
            .entries
            .iter()
            .filter(|e| matches!(e, HistoryEntry::ToolCall(c) if c.tool.name() == "fs_read"))
            .count();
        assert_eq!(read_calls, 2, "only the tail's fs_read calls survive");
    }

    #[tokio::test]
    async fn heuristic_keeps_edits_in_prefix() {
        let mut conv = Conversation::new();
        conv.push_user_text("x".repeat(30_000));
        let id = ToolCallId::new("edit-1");
        conv.push_tool_call(
            id.clone(),
            KnownTool::Edit(EditToolCall {
                path: "/a".into(),
                old_text: "a".into(),
                new_text: "b".into(),
            }),
        );
        conv.push_tool_result(id, true, "ok");
        // pad tail with noisy pairs
        for i in 0..12 {
            let id = ToolCallId::new(format!("r-{i}"));
            conv.push_tool_call(id.clone(), KnownTool::FsRead(FsReadToolCall { path: "/f".into() }));
            conv.push_tool_result(id, true, "y".repeat(1000));
        }
        let out = HeuristicCompactor::new().with_keep_tail(2).compact(conv).await.unwrap();
        let edits = out
            .entries
            .iter()
            .filter(|e| matches!(e, HistoryEntry::ToolCall(c) if c.tool.name() == "edit"))
            .count();
        assert_eq!(edits, 1, "edit calls are always kept");
    }

    #[tokio::test]
    async fn heuristic_early_exit_on_small_history() {
        let mut conv = Conversation::new();
        conv.push_user_text("hi");
        conv.push_assistant_text("hello");
        let out = HeuristicCompactor::new().compact(conv.clone()).await.unwrap();
        assert_eq!(out, conv);
    }

    struct StubSummarizer;

    #[async_trait]
    impl TextSummarizer for StubSummarizer {
        async fn summarize(&self, _system: &str, _user: &str) -> Result<String> {
            Ok("SUMMARY".to_string())
        }
    }

    struct FailingSummarizer;

    #[async_trait]
    impl TextSummarizer for FailingSummarizer {
        async fn summarize(&self, _system: &str, _user: &str) -> Result<String> {
            Err(crate::error::AgentError::Other("boom".into()))
        }
    }

    #[tokio::test]
    async fn llm_summarizes_long_results_and_keeps_short_ones() {
        let mut conv = Conversation::new();
        conv.push_user_text("z".repeat(30_000));
        let long = ToolCallId::new("long");
        conv.push_tool_call(long.clone(), KnownTool::FsRead(FsReadToolCall { path: "/f".into() }));
        conv.push_tool_result(long, true, "verbose output line\n".repeat(50));
        let short = ToolCallId::new("short");
        conv.push_tool_call(short.clone(), KnownTool::FsRead(FsReadToolCall { path: "/g".into() }));
        conv.push_tool_result(short, true, "ok");
        // pad tail
        for i in 0..12 {
            conv.push_assistant_text(format!("step {i}"));
        }
        let out = LlmCompactor::new(StubSummarizer)
            .with_keep_tail(2)
            .compact(conv)
            .await
            .unwrap();
        let summarized = out.entries.iter().any(|e| {
            matches!(e, HistoryEntry::ToolResult(r) if tool_result_text(r) == "SUMMARY")
        });
        let kept_short = out.entries.iter().any(|e| {
            matches!(e, HistoryEntry::ToolResult(r) if tool_result_text(r) == "ok")
        });
        assert!(summarized, "long output should be summarized");
        assert!(kept_short, "short output should be kept verbatim");
    }

    #[tokio::test]
    async fn llm_falls_back_to_original_on_failure() {
        let mut conv = Conversation::new();
        conv.push_user_text("z".repeat(30_000));
        let id = ToolCallId::new("long");
        conv.push_tool_call(id.clone(), KnownTool::FsRead(FsReadToolCall { path: "/f".into() }));
        let original = "verbose output line\n".repeat(50);
        conv.push_tool_result(id, true, original.clone());
        for i in 0..12 {
            conv.push_assistant_text(format!("step {i}"));
        }
        let out = LlmCompactor::new(FailingSummarizer)
            .with_keep_tail(2)
            .compact(conv)
            .await
            .unwrap();
        let kept = out.entries.iter().any(|e| {
            matches!(e, HistoryEntry::ToolResult(r) if tool_result_text(r) == original)
        });
        assert!(kept, "on failure the original output is preserved");
    }

    #[tokio::test]
    async fn chained_runs_stages_in_order() {
        let conv = noisy_conv(20);
        let before = conv.entries.len();
        let chain = ChainedCompactor::new(vec![
            Box::new(HeuristicCompactor::new().with_keep_tail(4).with_min_chars(0)),
            Box::new(LlmCompactor::new(StubSummarizer).with_keep_tail(4).with_min_chars(0)),
        ]);
        let out = chain.compact(conv).await.unwrap();
        assert!(out.entries.len() < before);
    }
}
