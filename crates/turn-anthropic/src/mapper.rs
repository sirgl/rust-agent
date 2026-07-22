//! Maps decoded Anthropic streaming events into provider-agnostic
//! [`agent_core::TurnEvent`]s.
//!
//! Anthropic's Messages API emits a well-defined sequence of typed SSE events
//! per turn:
//!
//! - `message_start` — opens the assistant message (metadata only here).
//! - `content_block_start` — opens a content block at an `index`; the block is
//!   one of `text`, `thinking`, or `tool_use` (with an `id` and `name`).
//! - `content_block_delta` — incremental content for a block:
//!   - `text_delta` -> [`TurnEvent::TextDelta`]
//!   - `thinking_delta` -> [`TurnEvent::Thinking`]
//!   - `input_json_delta` -> a fragment of the tool-use arguments JSON, which
//!     must be **accumulated** per block and only parsed once the block closes.
//!   - `signature_delta` — ignored (thinking-signature bookkeeping).
//! - `content_block_stop` — closes a block; for `tool_use` this is when the
//!   assembled arguments are parsed and a [`TurnEvent::ToolCallRequested`] is
//!   emitted.
//! - `message_delta` — carries the final `stop_reason`.
//! - `message_stop` — terminates the turn; emits [`TurnEvent::TurnFinished`].
//! - `error` — a mid-stream error; emits [`TurnEvent::Error`].
//! - `ping` — keep-alive; ignored.
//!
//! ## Robustness invariants
//!
//! - Tool-use argument JSON is **never** parsed incrementally: fragments are
//!   concatenated verbatim across arbitrarily many `input_json_delta` events
//!   (which may split a single JSON token) and parsed exactly once at
//!   `content_block_stop`. An empty accumulation is treated as `{}` (Anthropic
//!   omits `input_json_delta` for zero-argument tools). If the assembled text
//!   still fails to parse, the tool call is emitted with `{}` arguments rather
//!   than dropped or panicking, so the turn loop stays live and the tool layer
//!   can surface a structured error.
//! - Interleaved blocks (e.g. text, then tool_use, then more text) are tracked
//!   independently by `index`, so partial tool-use JSON cannot be corrupted by
//!   concurrent text deltas.
//! - Unknown event/delta/block types are ignored rather than erroring.

use std::collections::BTreeMap;

use agent_core::history::ThinkingRecord;
use agent_core::{StopReason, TokenUsage, ToolCallId, TurnError, TurnEvent};
use serde_json::Value;

/// Per-content-block state tracked while streaming.
#[derive(Debug)]
enum BlockState {
    /// A text block (no accumulation needed; deltas are forwarded live).
    Text,
    /// A thinking block: deltas are forwarded live for display *and*
    /// accumulated (text + signature) so the complete signed block can be
    /// emitted at `content_block_stop` for faithful replay.
    Thinking { text: String, signature: String },
    /// A redacted (encrypted, opaque) thinking block; its `data` is captured so
    /// it can be preserved and echoed back unchanged.
    RedactedThinking { data: String },
    /// A tool-use block whose argument JSON is being assembled.
    ToolUse {
        id: String,
        name: String,
        json: String,
    },
}

/// Stateful translator from Anthropic streaming events to [`TurnEvent`]s.
///
/// Feed each decoded SSE `data:` payload (already parsed into a
/// [`serde_json::Value`]) into [`StreamMapper::push`]; it returns zero or more
/// [`TurnEvent`]s. The mapper owns the accumulation state required to reassemble
/// tool-use blocks that arrive as partial JSON fragments.
#[derive(Debug)]
pub struct StreamMapper {
    blocks: BTreeMap<u64, BlockState>,
    stop_reason: StopReason,
    usage: TokenUsage,
}

impl Default for StreamMapper {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamMapper {
    /// Create a fresh mapper for a single decision round.
    pub fn new() -> Self {
        Self {
            blocks: BTreeMap::new(),
            // Default until a `message_delta` provides the real reason.
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
        }
    }

    /// Feed one decoded event and return any resulting turn events.
    pub fn push(&mut self, event: &Value) -> Vec<TurnEvent> {
        match event.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                self.on_message_start(event);
                Vec::new()
            }
            Some("content_block_start") => self.on_block_start(event),
            Some("content_block_delta") => self.on_block_delta(event),
            Some("content_block_stop") => self.on_block_stop(event),
            Some("message_delta") => {
                self.on_message_delta(event);
                Vec::new()
            }
            // Emit the accumulated usage right before the terminal event so the
            // engine can fold it into the session totals.
            Some("message_stop") => vec![
                TurnEvent::Usage(self.usage),
                TurnEvent::TurnFinished {
                    stop_reason: self.stop_reason,
                },
            ],
            Some("error") => vec![TurnEvent::Error(TurnError::new(error_message(event)))],
            // ping and any unknown types carry nothing to emit.
            _ => Vec::new(),
        }
    }

    fn on_block_start(&mut self, event: &Value) -> Vec<TurnEvent> {
        let Some(index) = index_of(event) else {
            return Vec::new();
        };
        let block = event.get("content_block");
        match block.and_then(|b| b.get("type")).and_then(Value::as_str) {
            Some("text") => {
                self.blocks.insert(index, BlockState::Text);
            }
            Some("thinking") => {
                // A thinking block may carry an initial `thinking`/`signature`
                // already at start; capture them so nothing is lost.
                let text = block
                    .and_then(|b| b.get("thinking"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let signature = block
                    .and_then(|b| b.get("signature"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                self.blocks
                    .insert(index, BlockState::Thinking { text, signature });
            }
            Some("redacted_thinking") => {
                let data = block
                    .and_then(|b| b.get("data"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                self.blocks
                    .insert(index, BlockState::RedactedThinking { data });
            }
            Some("tool_use") => {
                let id = block
                    .and_then(|b| b.get("id"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let name = block
                    .and_then(|b| b.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                self.blocks.insert(
                    index,
                    BlockState::ToolUse {
                        id,
                        name,
                        json: String::new(),
                    },
                );
            }
            _ => {}
        }
        Vec::new()
    }

    fn on_block_delta(&mut self, event: &Value) -> Vec<TurnEvent> {
        let Some(index) = index_of(event) else {
            return Vec::new();
        };
        let Some(delta) = event.get("delta") else {
            return Vec::new();
        };
        match delta.get("type").and_then(Value::as_str) {
            Some("text_delta") => {
                if let Some(text) = delta.get("text").and_then(Value::as_str) {
                    return vec![TurnEvent::TextDelta(text.to_string())];
                }
            }
            Some("thinking_delta") => {
                if let Some(text) = delta.get("thinking").and_then(Value::as_str) {
                    // Accumulate for the complete signed block, and forward the
                    // delta live for streaming display.
                    if let Some(BlockState::Thinking { text: acc, .. }) =
                        self.blocks.get_mut(&index)
                    {
                        acc.push_str(text);
                    }
                    return vec![TurnEvent::Thinking(text.to_string())];
                }
            }
            Some("signature_delta") => {
                if let Some(sig) = delta.get("signature").and_then(Value::as_str) {
                    if let Some(BlockState::Thinking { signature, .. }) =
                        self.blocks.get_mut(&index)
                    {
                        signature.push_str(sig);
                    }
                }
            }
            Some("input_json_delta") => {
                if let Some(partial) = delta.get("partial_json").and_then(Value::as_str) {
                    if let Some(BlockState::ToolUse { json, .. }) = self.blocks.get_mut(&index) {
                        json.push_str(partial);
                    }
                }
            }
            // Unknown delta types are intentionally ignored.
            _ => {}
        }
        Vec::new()
    }

    fn on_block_stop(&mut self, event: &Value) -> Vec<TurnEvent> {
        let Some(index) = index_of(event) else {
            return Vec::new();
        };
        match self.blocks.remove(&index) {
            Some(BlockState::ToolUse { id, name, json }) => {
                let arguments = parse_tool_arguments(&json);
                vec![TurnEvent::ToolCallRequested {
                    id: ToolCallId::new(id),
                    name,
                    arguments,
                }]
            }
            // Emit the complete, signed thinking block for verbatim replay.
            Some(BlockState::Thinking { text, signature }) => {
                vec![TurnEvent::ThinkingBlock(ThinkingRecord::Thinking {
                    text,
                    signature,
                })]
            }
            Some(BlockState::RedactedThinking { data }) => {
                vec![TurnEvent::ThinkingBlock(ThinkingRecord::Redacted { data })]
            }
            // Text blocks need no closing event.
            _ => Vec::new(),
        }
    }

    fn on_message_delta(&mut self, event: &Value) {
        if let Some(reason) = event
            .get("delta")
            .and_then(|d| d.get("stop_reason"))
            .and_then(Value::as_str)
        {
            self.stop_reason = map_stop_reason(reason);
        }
        // `message_delta.usage.output_tokens` is the cumulative final output
        // count for the message, so overwrite rather than add.
        if let Some(usage) = event.get("usage") {
            if let Some(out) = usage.get("output_tokens").and_then(Value::as_u64) {
                self.usage.output_tokens = out;
            }
        }
    }

    /// Capture the prompt-side token counts reported at `message_start`.
    ///
    /// Anthropic reports `input_tokens` (fresh) plus the cache read/creation
    /// counts here; the output count starts at ~1 and is finalized in
    /// `message_delta`.
    fn on_message_start(&mut self, event: &Value) {
        let Some(usage) = event.get("message").and_then(|m| m.get("usage")) else {
            return;
        };
        let get = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
        self.usage.input_tokens = get("input_tokens");
        self.usage.cache_read_input_tokens = get("cache_read_input_tokens");
        self.usage.cache_creation_input_tokens = get("cache_creation_input_tokens");
    }
}

/// Extract the `index` field of a content-block event.
fn index_of(event: &Value) -> Option<u64> {
    event.get("index").and_then(Value::as_u64)
}

/// Parse assembled tool-use argument JSON, defaulting to an empty object when
/// the accumulation is empty or fails to parse (see the module invariants).
fn parse_tool_arguments(json: &str) -> Value {
    let trimmed = json.trim();
    if trimmed.is_empty() {
        return Value::Object(Default::default());
    }
    match serde_json::from_str::<Value>(trimmed) {
        Ok(v) => v,
        Err(err) => {
            tracing::warn!(
                error = %err,
                raw = %trimmed,
                "turn-anthropic: failed to parse assembled tool_use arguments; defaulting to empty object",
            );
            Value::Object(Default::default())
        }
    }
}

/// Map an Anthropic `stop_reason` string to the provider-agnostic [`StopReason`].
///
/// Anthropic values include `end_turn`, `max_tokens`, `tool_use`,
/// `stop_sequence`, `refusal`, and `pause_turn`. Anything not explicitly mapped
/// collapses to [`StopReason::EndTurn`], which is the safe "turn is over" value.
fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" | "stop_sequence" => StopReason::EndTurn,
        "max_tokens" => StopReason::MaxTokens,
        "tool_use" => StopReason::ToolUse,
        "refusal" => StopReason::Refusal,
        _ => StopReason::EndTurn,
    }
}

/// Extract a human-readable message from an Anthropic `error` event.
fn error_message(event: &Value) -> String {
    event
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("anthropic stream error: {event}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn push_all(mapper: &mut StreamMapper, events: &[Value]) -> Vec<TurnEvent> {
        events.iter().flat_map(|e| mapper.push(e)).collect()
    }

    #[test]
    fn maps_text_delta() {
        let mut m = StreamMapper::new();
        let out = push_all(
            &mut m,
            &[
                json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}),
                json!({"type":"content_block_stop","index":0}),
            ],
        );
        assert_eq!(out.len(), 2);
        assert!(matches!(&out[0], TurnEvent::TextDelta(t) if t == "Hello"));
        assert!(matches!(&out[1], TurnEvent::TextDelta(t) if t == " world"));
    }

    #[test]
    fn maps_thinking_delta() {
        let mut m = StreamMapper::new();
        let out = push_all(
            &mut m,
            &[
                json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc"}}),
            ],
        );
        assert_eq!(out.len(), 1);
        assert!(matches!(&out[0], TurnEvent::Thinking(t) if t == "hmm"));
    }

    #[test]
    fn emits_complete_signed_thinking_block_on_stop() {
        let mut m = StreamMapper::new();
        let out = push_all(
            &mut m,
            &[
                json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"first "}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"second"}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-"}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"xyz"}}),
                json!({"type":"content_block_stop","index":0}),
            ],
        );
        // Two live deltas, then the complete signed block at stop.
        assert!(matches!(&out[0], TurnEvent::Thinking(t) if t == "first "));
        assert!(matches!(&out[1], TurnEvent::Thinking(t) if t == "second"));
        match out.last().unwrap() {
            TurnEvent::ThinkingBlock(ThinkingRecord::Thinking { text, signature }) => {
                assert_eq!(text, "first second");
                assert_eq!(signature, "sig-xyz");
            }
            other => panic!("expected signed ThinkingBlock, got {other:?}"),
        }
    }

    #[test]
    fn emits_redacted_thinking_block_on_stop() {
        let mut m = StreamMapper::new();
        let out = push_all(
            &mut m,
            &[
                json!({"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"enc-abc"}}),
                json!({"type":"content_block_stop","index":0}),
            ],
        );
        assert_eq!(out.len(), 1);
        match &out[0] {
            TurnEvent::ThinkingBlock(ThinkingRecord::Redacted { data }) => {
                assert_eq!(data, "enc-abc");
            }
            other => panic!("expected redacted ThinkingBlock, got {other:?}"),
        }
    }

    #[test]
    fn assembles_tool_use_from_partial_json() {
        let mut m = StreamMapper::new();
        let out = push_all(
            &mut m,
            &[
                json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"fs_read","input":{}}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\""}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":": \"/tmp/a"}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":".txt\"}"}}),
                json!({"type":"content_block_stop","index":0}),
            ],
        );
        assert_eq!(out.len(), 1);
        match &out[0] {
            TurnEvent::ToolCallRequested {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id.as_str(), "toolu_1");
                assert_eq!(name, "fs_read");
                assert_eq!(arguments["path"], "/tmp/a.txt");
            }
            other => panic!("expected ToolCallRequested, got {other:?}"),
        }
    }

    #[test]
    fn empty_tool_input_defaults_to_object() {
        let mut m = StreamMapper::new();
        let out = push_all(
            &mut m,
            &[
                json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"now","input":{}}}),
                json!({"type":"content_block_stop","index":0}),
            ],
        );
        match &out[0] {
            TurnEvent::ToolCallRequested { arguments, .. } => {
                assert!(arguments.is_object());
                assert_eq!(arguments.as_object().unwrap().len(), 0);
            }
            other => panic!("expected ToolCallRequested, got {other:?}"),
        }
    }

    #[test]
    fn malformed_tool_json_falls_back_to_empty_object() {
        let mut m = StreamMapper::new();
        let out = push_all(
            &mut m,
            &[
                json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"x","input":{}}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{not json"}}),
                json!({"type":"content_block_stop","index":0}),
            ],
        );
        assert_eq!(out.len(), 1);
        assert!(matches!(
            &out[0],
            TurnEvent::ToolCallRequested { arguments, .. } if arguments.is_object()
        ));
    }

    #[test]
    fn interleaved_text_and_tool_blocks_tracked_by_index() {
        let mut m = StreamMapper::new();
        let out = push_all(
            &mut m,
            &[
                json!({"type":"content_block_start","index":0,"content_block":{"type":"text"}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"before"}}),
                json!({"type":"content_block_stop","index":0}),
                json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t2","name":"edit","input":{}}}),
                json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"/a\"}"}}),
                json!({"type":"content_block_stop","index":1}),
            ],
        );
        assert!(matches!(&out[0], TurnEvent::TextDelta(t) if t == "before"));
        assert!(matches!(
            &out[1],
            TurnEvent::ToolCallRequested { name, .. } if name == "edit"
        ));
    }

    #[test]
    fn message_delta_then_stop_emits_finished_with_reason() {
        let mut m = StreamMapper::new();
        let out = push_all(
            &mut m,
            &[
                json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{}}),
                json!({"type":"message_stop"}),
            ],
        );
        // message_stop emits a Usage event followed by TurnFinished.
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], TurnEvent::Usage(_)));
        assert!(matches!(
            out[1],
            TurnEvent::TurnFinished {
                stop_reason: StopReason::ToolUse
            }
        ));
    }

    #[test]
    fn message_stop_without_delta_defaults_end_turn() {
        let mut m = StreamMapper::new();
        let out = m.push(&json!({"type":"message_stop"}));
        assert!(matches!(out[0], TurnEvent::Usage(_)));
        assert!(matches!(
            out[1],
            TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn
            }
        ));
    }

    #[test]
    fn accumulates_usage_across_start_delta_and_stop() {
        let mut m = StreamMapper::new();
        let out = push_all(
            &mut m,
            &[
                json!({"type":"message_start","message":{"usage":{
                    "input_tokens":120,
                    "cache_read_input_tokens":800,
                    "cache_creation_input_tokens":40,
                    "output_tokens":1
                }}}),
                json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":57}}),
                json!({"type":"message_stop"}),
            ],
        );
        let usage = out
            .iter()
            .find_map(|e| match e {
                TurnEvent::Usage(u) => Some(*u),
                _ => None,
            })
            .expect("usage event emitted");
        assert_eq!(usage.input_tokens, 120);
        assert_eq!(usage.cache_read_input_tokens, 800);
        assert_eq!(usage.cache_creation_input_tokens, 40);
        assert_eq!(usage.output_tokens, 57);
    }

    #[test]
    fn error_event_maps_to_error() {
        let mut m = StreamMapper::new();
        let out = m.push(
            &json!({"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}),
        );
        assert_eq!(out.len(), 1);
        match &out[0] {
            TurnEvent::Error(e) => assert_eq!(e.message, "overloaded"),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn ping_and_message_start_emit_nothing() {
        let mut m = StreamMapper::new();
        assert!(m.push(&json!({"type":"ping"})).is_empty());
        assert!(m
            .push(&json!({"type":"message_start","message":{"id":"m"}}))
            .is_empty());
    }

    #[test]
    fn stop_reason_variants() {
        assert_eq!(map_stop_reason("end_turn"), StopReason::EndTurn);
        assert_eq!(map_stop_reason("stop_sequence"), StopReason::EndTurn);
        assert_eq!(map_stop_reason("max_tokens"), StopReason::MaxTokens);
        assert_eq!(map_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(map_stop_reason("refusal"), StopReason::Refusal);
        assert_eq!(map_stop_reason("pause_turn"), StopReason::EndTurn);
    }
}
