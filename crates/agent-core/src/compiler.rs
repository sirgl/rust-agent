//! Provider-agnostic compiler seam.
//!
//! ## Design: interface-first, no mandatory intermediate representation
//!
//! The [`LlmCompiler`] trait is the seam between the strongly-typed,
//! provider-agnostic conversation ([`crate::history::Conversation`]) and
//! whatever wire format a concrete provider/dialect needs. Each provider
//! backend (Anthropic today, others later) implements the trait directly and
//! decides how to render typed history into its own request shape.
//!
//! We deliberately do **not** force every compiler through a shared
//! intermediate representation (IR): different providers have different
//! message/content shapes (e.g. Anthropic's interleaved `content` blocks vs. a
//! hypothetical provider using separate tool-call channels), and a one-size IR
//! tends to either leak provider-specific concerns back into `agent-core` or
//! lose fidelity in translation. [`LlmRequest`]/[`LlmMessage`] below are a
//! *default, optional* provider-neutral shape that a compiler may use when it
//! is a natural fit (as [`DefaultCompiler`] does); a provider is free to skip
//! them entirely and produce its own request type instead. An IR would only be
//! introduced later if multiple concrete dialects are found to duplicate
//! substantial translation logic — not by default.
//!
//! ## Hard requirement: preserve and reuse, don't lossily re-render
//!
//! Regardless of which shape a compiler targets, it must preserve exact
//! incoming content and, where a message/tool interaction already arrived from
//! an upstream provider/client and was compiled once, **reuse** that
//! previously-compiled form rather than re-deriving it from the lightweight
//! typed model. The typed history supports this via
//! [`crate::history::HistoryEntry::Ingested`]: a compiler should check for it
//! first and, when present, splice the preserved payload back in verbatim (see
//! [`DefaultCompiler::compile`] for a reference implementation of this rule).

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::history::HistoryEntry;
use crate::session::TurnContext;

/// The role of a compiled provider message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmRole {
    /// A user-authored message.
    User,
    /// An assistant-authored message.
    Assistant,
    /// A system message.
    System,
}

/// A single content block within an [`LlmMessage`].
///
/// This is the *default* provider-neutral shape offered for convenience; it is
/// not a mandatory IR (see the module docs). A compiler that uses [`LlmMessage`]
/// maps typed history fragments into these blocks; a compiler targeting a
/// different wire shape is free to ignore this type entirely.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LlmContentBlock {
    /// Plain text.
    Text {
        /// The textual content.
        text: String,
    },
    /// A request from the assistant to invoke a tool.
    ToolUse {
        /// Correlation id.
        id: String,
        /// Tool name.
        name: String,
        /// Tool arguments as JSON.
        input: serde_json::Value,
    },
    /// The result of a tool invocation fed back to the model.
    ToolResult {
        /// Id of the corresponding tool use.
        tool_use_id: String,
        /// Whether the tool call errored.
        is_error: bool,
        /// Result content rendered as text.
        content: String,
    },
}

/// A provider-neutral message in a compiled request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmMessage {
    /// The message role.
    pub role: LlmRole,
    /// The ordered content blocks.
    pub content: Vec<LlmContentBlock>,
}

/// A provider-neutral tool schema advertised to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmToolSchema {
    /// Tool name.
    pub name: String,
    /// JSON schema for the tool arguments.
    pub input_schema: serde_json::Value,
}

/// A provider-neutral request produced by an [`LlmCompiler`].
///
/// A concrete backend translates this into its own wire format (headers, model
/// id, streaming flags, etc.) at call time.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmRequest {
    /// Optional system prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// The ordered conversation messages.
    pub messages: Vec<LlmMessage>,
    /// Tool schemas available to the model.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<LlmToolSchema>,
}

/// Converts provider-agnostic history/context into a provider-specific request.
///
/// This is the seam that keeps [`crate::NextTurnService`] provider-agnostic:
/// each backend supplies its own compiler, choosing whatever request shape its
/// provider needs. The trait is object-safe so compilers can be stored as
/// `Arc<dyn LlmCompiler>`. Implementations MUST preserve exact incoming
/// content and reuse [`crate::history::HistoryEntry::Ingested`] payloads
/// verbatim rather than re-rendering them from the typed model (see the module
/// docs and [`DefaultCompiler`] for the reference behavior).
pub trait LlmCompiler: Send + Sync {
    /// Compile a [`TurnContext`] into this compiler's target request shape.
    fn compile(&self, ctx: &TurnContext) -> Result<LlmRequest>;
}

/// A reference [`LlmCompiler`] that targets the optional [`LlmRequest`] shape.
///
/// This is a usable default for backends whose wire format is close enough to
/// `LlmRequest` (role + content blocks), and doubles as the executable
/// specification for the "preserve and reuse" rule: whenever it encounters a
/// [`crate::history::HistoryEntry::Ingested`] entry it splices the preserved
/// payload back in unchanged instead of deriving a message from scratch.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultCompiler;

/// Extract the text of a [`crate::history::ContentBlock`], for shapes that
/// only understand plain text (non-text blocks, if ever added, are skipped
/// rather than causing a compile-time break here).
fn block_text(block: &crate::history::ContentBlock) -> &str {
    match block {
        crate::history::ContentBlock::Text { text } => text,
    }
}

impl LlmCompiler for DefaultCompiler {
    fn compile(&self, ctx: &TurnContext) -> Result<LlmRequest> {
        let mut messages = Vec::with_capacity(ctx.history.len());

        for entry in &ctx.history.entries {
            match entry {
                HistoryEntry::User(msg) => messages.push(LlmMessage {
                    role: LlmRole::User,
                    content: msg
                        .content
                        .iter()
                        .map(|block| LlmContentBlock::Text {
                            text: block_text(block).to_string(),
                        })
                        .collect(),
                }),
                HistoryEntry::Assistant(msg) => messages.push(LlmMessage {
                    role: LlmRole::Assistant,
                    content: msg
                        .content
                        .iter()
                        .map(|block| LlmContentBlock::Text {
                            text: block_text(block).to_string(),
                        })
                        .collect(),
                }),
                HistoryEntry::ToolCall(rec) => messages.push(LlmMessage {
                    role: LlmRole::Assistant,
                    content: vec![LlmContentBlock::ToolUse {
                        id: rec.id.as_str().to_string(),
                        name: rec.tool.name().to_string(),
                        // Rebuilt from the typed record rather than a separately
                        // stored copy, so the arguments always match what was
                        // classified into history.
                        input: rec.tool.arguments(),
                    }],
                }),
                HistoryEntry::ToolResult(rec) => {
                    let content = rec
                        .content
                        .iter()
                        .map(block_text)
                        .collect::<Vec<_>>()
                        .join("\n");
                    messages.push(LlmMessage {
                        role: LlmRole::User,
                        content: vec![LlmContentBlock::ToolResult {
                            tool_use_id: rec.id.as_str().to_string(),
                            is_error: !rec.success,
                            content,
                        }],
                    });
                }
                HistoryEntry::Ingested(ingested) => {
                    // Hard requirement: reuse the previously-compiled form
                    // verbatim instead of lossily re-rendering it. If the
                    // payload is itself a compiled `LlmMessage`, splice it in
                    // unchanged; otherwise fall back to a best-effort text
                    // rendering so no information is silently dropped.
                    match serde_json::from_value::<LlmMessage>(ingested.payload.clone()) {
                        Ok(message) => messages.push(message),
                        Err(_) => messages.push(LlmMessage {
                            role: LlmRole::User,
                            content: vec![LlmContentBlock::Text {
                                text: ingested.payload.to_string(),
                            }],
                        }),
                    }
                }
            }
        }

        Ok(LlmRequest {
            system: ctx.system_prompt.clone(),
            messages,
            tools: ctx
                .available_tools
                .iter()
                .map(|t| LlmToolSchema {
                    name: t.name.clone(),
                    input_schema: t.schema.clone(),
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::ToolCallId;
    use crate::history::{EditToolCall, KnownTool, TerminalRunToolCall};
    use crate::session::SessionState;

    #[test]
    fn default_compiler_renders_plain_text_turns() {
        let mut session = SessionState::new("s1");
        session.system_prompt = Some("be helpful".into());
        session.push_user_text("hi");
        session.push_assistant_text("hello");
        let ctx = session.turn_context();

        let req = DefaultCompiler.compile(&ctx).unwrap();
        assert_eq!(req.system.as_deref(), Some("be helpful"));
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, LlmRole::User);
        assert_eq!(req.messages[1].role, LlmRole::Assistant);
    }

    #[test]
    fn default_compiler_rebuilds_tool_call_arguments_from_typed_record() {
        let mut session = SessionState::new("s1");
        let id = ToolCallId::new("call-1");
        session.push_tool_call(
            id.clone(),
            KnownTool::Edit(EditToolCall {
                path: "/a.txt".into(),
                old_text: "x".into(),
                new_text: "y".into(),
            }),
        );
        session.push_tool_result(id, true, "done");
        let ctx = session.turn_context();

        let req = DefaultCompiler.compile(&ctx).unwrap();
        assert_eq!(req.messages.len(), 2);
        match &req.messages[0].content[0] {
            LlmContentBlock::ToolUse { name, input, .. } => {
                assert_eq!(name, "edit");
                assert_eq!(input["path"], "/a.txt");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
        match &req.messages[1].content[0] {
            LlmContentBlock::ToolResult {
                is_error, content, ..
            } => {
                assert!(!is_error);
                assert_eq!(content, "done");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn default_compiler_rebuilds_terminal_run_arguments_from_typed_record() {
        let mut session = SessionState::new("s1");
        let id = ToolCallId::new("call-1");
        session.push_tool_call(
            id.clone(),
            KnownTool::TerminalRun(TerminalRunToolCall {
                command: "git".into(),
                args: vec!["commit".into(), "-m".into(), "x".into()],
            }),
        );
        session.push_tool_result(id, true, "done");
        let ctx = session.turn_context();

        let req = DefaultCompiler.compile(&ctx).unwrap();
        assert_eq!(req.messages.len(), 2);

        match &req.messages[0].content[0] {
            LlmContentBlock::ToolUse { name, input, .. } => {
                assert_eq!(name, "terminal_run");
                assert_eq!(input["command"], "git");
                assert_eq!(input["args"], serde_json::json!(["commit", "-m", "x"]));
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn default_compiler_reuses_ingested_message_verbatim() {
        let mut session = SessionState::new("s1");
        // Simulate an upstream/provider message that was already compiled once
        // (e.g. re-ingested from a resumed session). It carries content that
        // would not round-trip identically through the lossy text-only path
        // (a leading/trailing-whitespace prefix that must be preserved exactly).
        let prebuilt = LlmMessage {
            role: LlmRole::Assistant,
            content: vec![LlmContentBlock::Text {
                text: "  exact prefix preserved\nverbatim".into(),
            }],
        };
        session.push_ingested("anthropic", serde_json::to_value(&prebuilt).unwrap());
        let ctx = session.turn_context();

        let req = DefaultCompiler.compile(&ctx).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0], prebuilt);
    }

    #[test]
    fn default_compiler_falls_back_for_non_message_ingested_payload() {
        let mut session = SessionState::new("s1");
        session.push_ingested("opaque", serde_json::json!({"anything": true}));
        let ctx = session.turn_context();

        let req = DefaultCompiler.compile(&ctx).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, LlmRole::User);
    }
}
