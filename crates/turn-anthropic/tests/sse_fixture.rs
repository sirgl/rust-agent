//! End-to-end parsing test: drive a *recorded* raw Anthropic SSE stream through
//! the public [`SseDecoder`] + [`StreamMapper`] pipeline and assert the exact
//! resulting [`TurnEvent`] sequence — no network access required.
//!
//! The fixture below is a lightly-trimmed but faithful capture of the event
//! shapes Anthropic emits for a turn that streams text, then a tool_use block
//! (arguments split across several `input_json_delta` events, including a split
//! mid-token), and finishes with `stop_reason: "tool_use"`.

use agent_core::{StopReason, TurnEvent};
use turn_anthropic::{SseDecoder, StreamMapper};

/// A recorded raw SSE stream (see module docs).
const FIXTURE: &str = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[]}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: ping
data: {\"type\":\"ping\"}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Let me \"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"read it.\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":0}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_42\",\"name\":\"fs_read\",\"input\":{}}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"pa\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"th\\\": \\\"/etc/hosts\\\"}\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":1}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":15}}

event: message_stop
data: {\"type\":\"message_stop\"}

";

/// Run a byte slice through the decode+map pipeline using the given chunk size,
/// simulating arbitrary network fragmentation.
fn run(bytes: &[u8], chunk_size: usize) -> Vec<TurnEvent> {
    let mut decoder = SseDecoder::new();
    let mut mapper = StreamMapper::new();
    let mut events = Vec::new();
    for chunk in bytes.chunks(chunk_size.max(1)) {
        for payload in decoder.push(chunk) {
            let value: serde_json::Value = serde_json::from_str(&payload).expect("valid JSON");
            events.extend(mapper.push(&value));
        }
    }
    events
}

fn assert_expected_sequence(events: &[TurnEvent]) {
    assert_eq!(events.len(), 4, "unexpected events: {events:?}");
    assert!(matches!(&events[0], TurnEvent::TextDelta(t) if t == "Let me "));
    assert!(matches!(&events[1], TurnEvent::TextDelta(t) if t == "read it."));
    match &events[2] {
        TurnEvent::ToolCallRequested {
            id,
            name,
            arguments,
        } => {
            assert_eq!(id.as_str(), "toolu_42");
            assert_eq!(name, "fs_read");
            assert_eq!(arguments["path"], "/etc/hosts");
        }
        other => panic!("expected ToolCallRequested, got {other:?}"),
    }
    assert!(matches!(
        &events[3],
        TurnEvent::TurnFinished {
            stop_reason: StopReason::ToolUse
        }
    ));
}

#[test]
fn recorded_stream_maps_to_expected_events_whole() {
    let events = run(FIXTURE.as_bytes(), FIXTURE.len());
    assert_expected_sequence(&events);
}

#[test]
fn recorded_stream_is_robust_to_arbitrary_chunking() {
    // Every chunk size should produce the identical event sequence, proving the
    // decoder/mapper correctly reassemble events and partial tool-use JSON
    // regardless of where the network splits the byte stream.
    for chunk_size in [1usize, 3, 7, 16, 64, 256] {
        let events = run(FIXTURE.as_bytes(), chunk_size);
        assert_expected_sequence(&events);
    }
}
