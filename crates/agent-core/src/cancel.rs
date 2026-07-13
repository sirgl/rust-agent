//! Cancellation primitives.
//!
//! A thin re-export/wrapper around [`tokio_util::sync::CancellationToken`] so the
//! rest of the codebase depends on a single, stable cancellation type that can be
//! shared between the decision stream and running tools/subagents.

pub use tokio_util::sync::CancellationToken;
