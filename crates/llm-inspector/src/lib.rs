//! # llm-inspector
//!
//! A small, self-contained observability server that shows the **real** requests
//! sent to the LLM in a simple web UI.
//!
//! It has two pieces:
//!
//! - [`Inspector`] — a thread-safe, capacity-bounded store of captured request
//!   bodies. It implements [`turn_anthropic::RequestObserver`], so attaching it
//!   to the Anthropic backend (`AnthropicTurnService::with_observer`) captures
//!   the exact JSON body of every streaming turn, attributed to its session.
//! - [`serve`] — a tiny, dependency-free HTTP server (over `tokio`) that renders
//!   the captured requests, grouped by session, in a browser.
//!
//! Because sessions are first-class in the ACP agent, the UI groups captured
//! requests by `session_id`, so concurrent sessions are easy to tell apart.
//!
//! ```no_run
//! # async fn demo() -> std::io::Result<()> {
//! use llm_inspector::{serve, Inspector};
//! use std::sync::Arc;
//!
//! let inspector = Inspector::new();
//! // Attach `inspector.clone()` to the Anthropic backend as a RequestObserver,
//! // then serve the UI:
//! let addr = serve(inspector, "127.0.0.1:7878".parse().unwrap()).await?;
//! println!("inspector on http://{addr}");
//! # let _ = Arc::new(());
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

mod log_layer;
mod record;
mod recorder;
mod server;
mod store;

pub use log_layer::InspectorLayer;
pub use record::{CapturedPayload, CapturedRequest, CapturedSessionState, ToolInfo};
pub use recorder::{FileRecorder, MemoryRecorder, Recorder, DEFAULT_CAPACITY};
pub use server::{serve, serve_with_fallback};
pub use store::Inspector;

/// Initialize a `tracing` subscriber that writes to stderr **and** mirrors
/// every log event into `inspector` (so the debug server can show agent logs).
///
/// Respects `RUST_LOG` via `EnvFilter` (default `info`), matching
/// [`agent_core::init_tracing`]. Safe to call once at startup; a subsequent call
/// is ignored. Use this instead of `agent_core::init_tracing` when the inspector
/// is enabled.
pub fn init_tracing_with_inspector(inspector: Inspector) {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::{fmt, EnvFilter};

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(InspectorLayer::new(inspector))
        .try_init();
}
