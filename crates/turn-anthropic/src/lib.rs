//! `turn-anthropic`: a real [`agent_core::NextTurnService`] backed by the
//! Anthropic Messages API using native SSE streaming.
//!
//! ## Pipeline
//!
//! ```text
//! TurnContext --DefaultCompiler--> LlmRequest --build_body--> Anthropic JSON
//!   --HTTP(stream)--> raw bytes --SseDecoder--> data payloads
//!   --serde_json--> events --StreamMapper--> TurnEvent stream
//! ```
//!
//! The turn engine consumes the produced `BoxStream<TurnEvent>` exactly as it
//! does for the replay backend, so the Anthropic backend is a drop-in
//! [`agent_core::NextTurnService`] implementation. Errors (transport, HTTP
//! status, decoding) are surfaced as a terminal [`agent_core::TurnEvent::Error`]
//! followed by a [`agent_core::TurnEvent::TurnFinished`] so the loop never hangs.
//!
//! The heavy lifting — request construction, SSE framing, and event mapping —
//! lives in [`request`], [`sse`], and [`mapper`], all of which are pure and unit
//! tested without any network access (see `tests/sse_fixture.rs` for a full
//! recorded-stream integration test).

#![warn(missing_docs)]

pub mod mapper;
pub mod models;
pub mod request;
pub mod sse;

use std::sync::Arc;

use agent_core::{
    DefaultCompiler, LlmCompiler, NextTurnService, Result, StopReason, TurnContext, TurnError,
    TurnEvent,
};
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;

pub use mapper::StreamMapper;
pub use models::{default_max_output_tokens, model_spec, ModelSpec, MODELS};
pub use request::{build_body, load_default_api_key, AnthropicConfig, Effort, ThinkingConfig};
pub use sse::SseDecoder;

/// A real [`NextTurnService`] that answers turns via the Anthropic Messages API.
///
/// Construct it with an [`AnthropicConfig`]; each call to
/// [`get_next_turn_streaming`](NextTurnService::get_next_turn_streaming)
/// compiles the [`TurnContext`], issues one streaming HTTP request, and maps the
/// SSE response into a [`TurnEvent`] stream.
#[derive(Clone)]
pub struct AnthropicTurnService {
    config: AnthropicConfig,
    client: reqwest::Client,
    compiler: Arc<dyn LlmCompiler>,
}

impl std::fmt::Debug for AnthropicTurnService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicTurnService")
            .field("model", &self.config.model)
            .field("base_url", &self.config.base_url)
            .finish_non_exhaustive()
    }
}

impl AnthropicTurnService {
    /// Create a service from a config, using the [`DefaultCompiler`].
    pub fn new(config: AnthropicConfig) -> Self {
        Self::with_compiler(config, Arc::new(DefaultCompiler))
    }

    /// Create a service with a custom [`LlmCompiler`] (e.g. a dialect variant).
    pub fn with_compiler(config: AnthropicConfig, compiler: Arc<dyn LlmCompiler>) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            compiler,
        }
    }

    /// Build a service from environment variables (see [`AnthropicConfig::from_env`]).
    pub fn from_env() -> Option<Self> {
        AnthropicConfig::from_env().map(Self::new)
    }

    /// The configuration this service was built with.
    pub fn config(&self) -> &AnthropicConfig {
        &self.config
    }
}

#[async_trait]
impl NextTurnService for AnthropicTurnService {
    async fn get_next_turn_streaming(
        &self,
        ctx: &TurnContext,
    ) -> Result<BoxStream<'static, TurnEvent>> {
        // Compile typed history into the provider-neutral request, then into the
        // Anthropic wire body. Compilation happens up-front so a compile error
        // is reported synchronously (via `Result`) rather than mid-stream.
        let req = self.compiler.compile(ctx)?;
        let body = build_body(&req, &self.config);

        tracing::debug!(
            model = %self.config.model,
            messages = req.messages.len(),
            tools = req.tools.len(),
            "turn-anthropic: issuing streaming request",
        );

        let request = self
            .client
            .post(&self.config.base_url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", &self.config.anthropic_version)
            .header("content-type", "application/json")
            .json(&body);

        Ok(Box::pin(stream_turn(request)))
    }
}

/// Drive one HTTP streaming request and yield mapped [`TurnEvent`]s.
///
/// Any failure is emitted as a single [`TurnEvent::Error`] followed by a
/// terminal [`TurnEvent::TurnFinished`] so the engine loop always terminates.
fn stream_turn(request: reqwest::RequestBuilder) -> impl futures::Stream<Item = TurnEvent> {
    stream! {
        let response = match request.send().await {
            Ok(r) => r,
            Err(err) => {
                yield TurnEvent::Error(TurnError::new(format!(
                    "anthropic request failed: {err}"
                )));
                yield TurnEvent::TurnFinished { stop_reason: StopReason::EndTurn };
                return;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            yield TurnEvent::Error(TurnError::new(format!(
                "anthropic returned HTTP {status}: {text}"
            )));
            yield TurnEvent::TurnFinished { stop_reason: StopReason::EndTurn };
            return;
        }

        let mut decoder = SseDecoder::new();
        let mut mapper = StreamMapper::new();
        let mut finished = false;
        let mut bytes = response.bytes_stream();

        while let Some(chunk) = bytes.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(err) => {
                    yield TurnEvent::Error(TurnError::new(format!(
                        "anthropic stream error: {err}"
                    )));
                    break;
                }
            };
            for payload in decoder.push(&chunk) {
                let value = match serde_json::from_str::<serde_json::Value>(&payload) {
                    Ok(v) => v,
                    Err(err) => {
                        tracing::warn!(error = %err, raw = %payload, "turn-anthropic: skipping undecodable SSE payload");
                        continue;
                    }
                };
                for event in mapper.push(&value) {
                    if matches!(event, TurnEvent::TurnFinished { .. }) {
                        finished = true;
                    }
                    yield event;
                }
            }
        }

        // Guarantee the loop always sees a terminal event even if the stream was
        // cut short before `message_stop`.
        if !finished {
            yield TurnEvent::TurnFinished { stop_reason: StopReason::EndTurn };
        }
    }
}
