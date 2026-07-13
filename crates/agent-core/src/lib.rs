//! # agent-core
//!
//! Core trait seams and event model for the ACP agent.
//!
//! This crate defines the abstractions the rest of the workspace builds on:
//! - the decision layer ([`NextTurnService`]) that streams [`TurnEvent`]s,
//! - the services layer ([`Tool`], [`ToolRegistry`]),
//! - session/turn data models ([`SessionState`], [`TurnContext`], [`ToolContext`]),
//! - the output seam ([`UpdateSink`]),
//! - shared errors, [`StopReason`], and the [`CancellationToken`].
//!
//! The [`TurnEngine`] drives the turn loop, consuming [`TurnEvent`]s and
//! dispatching tool calls.

#![warn(missing_docs)]

pub mod cancel;
pub mod client;
pub mod compiler;
pub mod engine;
pub mod error;
pub mod event;
pub mod history;
pub mod service;
pub mod session;
pub mod sink;
pub mod tool;

pub use cancel::CancellationToken;
pub use client::{ClientAccess, TerminalOutcome};
pub use compiler::{
    DefaultCompiler, LlmCompiler, LlmContentBlock, LlmMessage, LlmRequest, LlmRole, LlmToolSchema,
};
pub use engine::TurnEngine;
pub use error::{AgentError, Result, TurnError};
pub use event::{PlanStep, PlanStepStatus, PlanUpdate, StopReason, ToolCallId, TurnEvent};
pub use history::{
    AssistantMessage, ContentBlock, Conversation, EditToolCall, FsReadToolCall, FsWriteToolCall,
    HistoryEntry, IngestedRecord, KnownTool, TerminalRunToolCall, ToolCallRecord, ToolResultRecord,
    UserMessage,
};
pub use service::NextTurnService;
pub use session::{SessionState, ToolDescriptor, TurnContext};
pub use sink::{EngineOutput, ToolCallStatus, UpdateSink};
pub use tool::{Tool, ToolContext, ToolEvent, ToolRegistry, ToolResult};

/// Initialize a default `tracing` subscriber writing to stderr.
///
/// Respects the `RUST_LOG` environment variable via `EnvFilter`. Safe to call
/// once at process startup; subsequent calls are ignored.
pub fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Compile-time assertions that the trait seams are object-safe.
    fn _assert_object_safe(
        _nts: &dyn NextTurnService,
        _tool: &dyn Tool,
        _sink: &dyn UpdateSink,
        _compiler: &dyn LlmCompiler,
    ) {
    }

    #[test]
    fn tool_registry_registers_and_looks_up() {
        // A minimal tool used only to exercise the registry and trait objects.
        use async_trait::async_trait;
        use futures::stream::{self, BoxStream};

        struct NoopTool;

        #[async_trait]
        impl Tool for NoopTool {
            fn name(&self) -> &str {
                "noop"
            }
            fn schema(&self) -> serde_json::Value {
                serde_json::json!({ "type": "object" })
            }
            fn requires_permission(&self) -> bool {
                false
            }
            async fn call(
                &self,
                _args: serde_json::Value,
                _ctx: &ToolContext,
            ) -> Result<BoxStream<'static, ToolEvent>> {
                Ok(Box::pin(stream::iter(vec![ToolEvent::Started])))
            }
        }

        let mut registry = ToolRegistry::new();
        assert!(registry.is_empty());
        registry.register(Arc::new(NoopTool));
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("noop"));
        let tool = registry.get("noop").expect("tool present");
        assert_eq!(tool.name(), "noop");
        assert!(!tool.requires_permission());
    }

    #[test]
    fn session_state_builds_turn_context() {
        let mut session = SessionState::new("sess-1");
        session.push_user_text("hello");
        session.push_assistant_text("hi");
        let ctx = session.turn_context();
        assert_eq!(ctx.history.len(), 2);
        assert_eq!(session.session_id, "sess-1");
    }

    #[test]
    fn known_tool_classify_typed_and_fallback() {
        let edit = KnownTool::classify(
            "edit",
            serde_json::json!({ "path": "/a", "old_text": "x", "new_text": "y" }),
        );
        assert!(matches!(edit, KnownTool::Edit(_)));
        assert_eq!(edit.name(), "edit");

        // Unknown tool falls back to Other, preserving arguments.
        let other = KnownTool::classify("mcp_search", serde_json::json!({ "q": 1 }));
        assert!(matches!(other, KnownTool::Other { .. }));
        assert_eq!(other.name(), "mcp_search");

        // Known name with bad args also falls back gracefully.
        let bad = KnownTool::classify("edit", serde_json::json!({ "path": 5 }));
        assert!(matches!(bad, KnownTool::Other { .. }));
    }

    #[test]
    fn typed_history_records_tool_interaction() {
        let mut session = SessionState::new("sess-2");
        session.push_user_text("edit the file");
        let id = ToolCallId::new("call-1");
        session.push_tool_call(
            id.clone(),
            KnownTool::Edit(EditToolCall {
                path: "/tmp/a.txt".into(),
                old_text: "foo".into(),
                new_text: "bar".into(),
            }),
        );
        session.push_tool_result(id, true, "ok");
        assert_eq!(session.history.len(), 3);
        match &session.history.entries[1] {
            HistoryEntry::ToolCall(rec) => assert_eq!(rec.tool.name(), "edit"),
            other => panic!("unexpected entry: {other:?}"),
        }
    }

    #[test]
    fn conversation_serde_roundtrip() {
        let mut conv = Conversation::new();
        conv.push_user_text("hi");
        conv.push_assistant_text("hello");
        let json = serde_json::to_string(&conv).unwrap();
        let back: Conversation = serde_json::from_str(&json).unwrap();
        assert_eq!(conv, back);
    }

    #[test]
    fn stop_reason_serde_roundtrip() {
        let json = serde_json::to_string(&StopReason::EndTurn).unwrap();
        assert_eq!(json, "\"end_turn\"");
        let back: StopReason = serde_json::from_str(&json).unwrap();
        assert_eq!(back, StopReason::EndTurn);
    }

    #[test]
    fn cancellation_token_signals() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
    }
}
