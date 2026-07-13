//! A minimal, streaming Server-Sent Events (SSE) framing decoder.
//!
//! Anthropic's Messages API streams events as SSE: each event is a group of
//! lines terminated by a blank line, e.g.
//!
//! ```text
//! event: content_block_delta
//! data: {"type":"content_block_delta", ...}
//!
//! ```
//!
//! This decoder is deliberately transport-agnostic and synchronous: it is fed
//! arbitrary byte chunks (as they arrive from the network) via [`SseDecoder::push`]
//! and returns any fully-framed `data:` payloads decoded so far. Because chunk
//! boundaries do not align with event boundaries, partial lines are buffered
//! internally until completed. Keeping the framing logic pure makes it unit
//! testable against recorded fixtures without any network access.

/// Incremental SSE frame decoder.
///
/// Only the `data:` field is retained; other SSE fields (`event:`, `id:`,
/// `retry:`, comment lines starting with `:`) are ignored because the JSON
/// payload itself carries a discriminating `type` field. Multiple `data:`
/// lines within one event are concatenated with `\n`, per the SSE spec.
#[derive(Debug, Default)]
pub struct SseDecoder {
    /// Bytes received but not yet split into a complete line.
    line_buf: String,
    /// `data:` payload accumulated for the event currently being framed.
    data_buf: String,
    /// Whether the current event has seen at least one `data:` line.
    has_data: bool,
}

impl SseDecoder {
    /// Create an empty decoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk of bytes; returns any complete `data:` payloads that became
    /// available. Invalid UTF-8 is replaced lossily rather than erroring, so a
    /// single malformed byte cannot wedge the stream.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.line_buf.push_str(&String::from_utf8_lossy(chunk));
        let mut out = Vec::new();

        // Process every complete line (terminated by '\n'); keep the remainder.
        while let Some(nl) = self.line_buf.find('\n') {
            let mut line = self.line_buf[..nl].to_string();
            self.line_buf.drain(..=nl);
            // Tolerate CRLF line endings.
            if line.ends_with('\r') {
                line.pop();
            }
            if let Some(payload) = self.consume_line(&line) {
                out.push(payload);
            }
        }
        out
    }

    /// Process a single already-delimited line, returning a completed payload
    /// when the line was the blank separator closing an event with data.
    fn consume_line(&mut self, line: &str) -> Option<String> {
        if line.is_empty() {
            // Blank line: dispatch the accumulated event (if any).
            if self.has_data {
                let payload = std::mem::take(&mut self.data_buf);
                self.has_data = false;
                return Some(payload);
            }
            return None;
        }
        if line.starts_with(':') {
            // Comment line; ignore.
            return None;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            // A leading space after the colon is part of SSE framing, not data.
            let rest = rest.strip_prefix(' ').unwrap_or(rest);
            if self.has_data {
                self.data_buf.push('\n');
            }
            self.data_buf.push_str(rest);
            self.has_data = true;
        }
        // All other fields (event:, id:, retry:, ...) are ignored on purpose.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_single_event() {
        let mut d = SseDecoder::new();
        let out = d.push(b"event: ping\ndata: {\"type\":\"ping\"}\n\n");
        assert_eq!(out, vec!["{\"type\":\"ping\"}".to_string()]);
    }

    #[test]
    fn buffers_across_chunk_boundaries() {
        let mut d = SseDecoder::new();
        assert!(d.push(b"data: {\"a\"").is_empty());
        assert!(d.push(b":1}").is_empty());
        let out = d.push(b"\n\n");
        assert_eq!(out, vec!["{\"a\":1}".to_string()]);
    }

    #[test]
    fn concatenates_multiple_data_lines() {
        let mut d = SseDecoder::new();
        let out = d.push(b"data: line1\ndata: line2\n\n");
        assert_eq!(out, vec!["line1\nline2".to_string()]);
    }

    #[test]
    fn handles_crlf_and_comments() {
        let mut d = SseDecoder::new();
        let out = d.push(b": comment\r\ndata: {\"x\":1}\r\n\r\n");
        assert_eq!(out, vec!["{\"x\":1}".to_string()]);
    }

    #[test]
    fn emits_multiple_events_in_order() {
        let mut d = SseDecoder::new();
        let out = d.push(b"data: a\n\ndata: b\n\n");
        assert_eq!(out, vec!["a".to_string(), "b".to_string()]);
    }
}
