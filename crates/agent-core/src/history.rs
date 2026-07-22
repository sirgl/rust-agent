//! Strongly-typed conversation history model.
//!
//! History is represented as typed entries in `agent-core` rather than raw
//! provider messages. Tool interactions are captured explicitly, and well-known
//! built-in tools (e.g. [`KnownTool::Edit`]) are represented in typed form so the
//! rest of the workspace can reason about them without string-matching. Any tool
//! the core does not know about is preserved verbatim via
//! [`KnownTool::Other`].
//!
//! Converting this provider-agnostic model into a provider-specific request is
//! the responsibility of the [`crate::compiler::LlmCompiler`] seam.

use serde::{Deserialize, Serialize};

use crate::event::ToolCallId;
use crate::todo::{TodoItemId, TodoList};

/// A block of content within a user or assistant message.
///
/// Kept intentionally small for v1 (text only); richer block kinds (images,
/// documents, etc.) can be added behind this enum without breaking callers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain UTF-8 text.
    Text {
        /// The textual content.
        text: String,
    },
}

impl ContentBlock {
    /// Convenience constructor for a text block.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }
}

/// A message authored by the end user.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserMessage {
    /// The ordered content blocks of the message.
    pub content: Vec<ContentBlock>,
}

impl UserMessage {
    /// Create a user message from a single text string.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::text(text)],
        }
    }
}

/// A message authored by the assistant (excluding tool calls, which are their
/// own [`HistoryEntry::ToolCall`] entries so they can be typed explicitly).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantMessage {
    /// The ordered content blocks of the message.
    pub content: Vec<ContentBlock>,
}

impl AssistantMessage {
    /// Create an assistant message from a single text string.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::text(text)],
        }
    }
}

/// Typed arguments for the built-in `Edit` tool.
///
/// Represented explicitly so the engine and UI can render edits without parsing
/// opaque JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditToolCall {
    /// Absolute path of the file being edited.
    pub path: String,
    /// The exact text to search for.
    pub old_text: String,
    /// The replacement text.
    pub new_text: String,
}

/// Typed arguments for the built-in `fs_read` tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FsReadToolCall {
    /// Absolute path of the file to read.
    pub path: String,
}

/// Typed arguments for the built-in `fs_write` tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FsWriteToolCall {
    /// Absolute path of the file to write.
    pub path: String,
    /// The exact text content to write.
    pub content: String,
}

/// Typed arguments for the built-in `terminal_run` tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerminalRunToolCall {
    /// The command line to execute.
    pub command: String,

    /// Arguments to pass to the terminal command.
    ///
    /// This has a serde default so classification round-trips stay lossless
    /// even when the upstream payload omits `args` (legacy/partial payloads).
    #[serde(default)]
    pub args: Vec<String>,
}

/// Typed arguments for the built-in `update_todo_list` tool.
///
/// Transparent serialization preserves the tool's `{ "items": [...] }`
/// argument shape while reusing [`TodoList`]'s validation on deserialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct UpdateTodoListToolCall(pub TodoList);

/// Typed arguments for the built-in `mark_todo_completed` tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarkTodoCompletedToolCall {
    /// Stable identifier of the item to complete.
    pub id: TodoItemId,
}

/// A typed view of a tool invocation.
///
/// Built-in tools known to `agent-core` are represented with dedicated variants;
/// everything else is preserved as [`KnownTool::Other`] so no information is lost.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tool", rename_all = "snake_case")]
pub enum KnownTool {
    /// The built-in file `Edit` tool.
    Edit(EditToolCall),
    /// The built-in `fs_read` tool.
    FsRead(FsReadToolCall),
    /// The built-in `fs_write` tool.
    FsWrite(FsWriteToolCall),
    /// The built-in `terminal_run` tool.
    TerminalRun(TerminalRunToolCall),
    /// The built-in `update_todo_list` tool.
    UpdateTodoList(UpdateTodoListToolCall),
    /// The built-in `mark_todo_completed` tool.
    MarkTodoCompleted(MarkTodoCompletedToolCall),
    /// Any tool not known to the core (MCP tools, subagents, etc.).
    Other {
        /// The tool name as advertised by the registry.
        name: String,
        /// Raw JSON arguments.
        arguments: serde_json::Value,
    },
}

impl KnownTool {
    /// The name of the tool, whether built-in or [`KnownTool::Other`].
    pub fn name(&self) -> &str {
        match self {
            KnownTool::Edit(_) => "edit",
            KnownTool::FsRead(_) => "fs_read",
            KnownTool::FsWrite(_) => "fs_write",
            KnownTool::TerminalRun(_) => "terminal_run",
            KnownTool::UpdateTodoList(_) => "update_todo_list",
            KnownTool::MarkTodoCompleted(_) => "mark_todo_completed",
            KnownTool::Other { name, .. } => name,
        }
    }

    /// Reconstruct the raw JSON arguments for this tool call.
    ///
    /// Built-in variants are serialized back to their JSON form; the
    /// [`KnownTool::Other`] variant returns its preserved arguments verbatim.
    /// This lets the compiler rebuild a faithful tool-call fragment without a
    /// separate copy of the original arguments.
    pub fn arguments(&self) -> serde_json::Value {
        match self {
            KnownTool::Edit(v) => serde_json::to_value(v).unwrap_or(serde_json::Value::Null),
            KnownTool::FsRead(v) => serde_json::to_value(v).unwrap_or(serde_json::Value::Null),
            KnownTool::FsWrite(v) => serde_json::to_value(v).unwrap_or(serde_json::Value::Null),
            KnownTool::TerminalRun(v) => serde_json::to_value(v).unwrap_or(serde_json::Value::Null),
            KnownTool::UpdateTodoList(v) => {
                serde_json::to_value(v).unwrap_or(serde_json::Value::Null)
            }
            KnownTool::MarkTodoCompleted(v) => {
                serde_json::to_value(v).unwrap_or(serde_json::Value::Null)
            }
            KnownTool::Other { arguments, .. } => arguments.clone(),
        }
    }

    /// Best-effort classification of a raw `(name, arguments)` pair into a typed
    /// [`KnownTool`]. Unknown names, or known names whose arguments fail to parse,
    /// fall back to [`KnownTool::Other`] rather than erroring.
    pub fn classify(name: &str, arguments: serde_json::Value) -> Self {
        match name {
            "edit" => serde_json::from_value(arguments.clone())
                .map(KnownTool::Edit)
                .unwrap_or(KnownTool::Other {
                    name: name.to_string(),
                    arguments,
                }),
            "fs_read" => serde_json::from_value(arguments.clone())
                .map(KnownTool::FsRead)
                .unwrap_or(KnownTool::Other {
                    name: name.to_string(),
                    arguments,
                }),
            "fs_write" => serde_json::from_value(arguments.clone())
                .map(KnownTool::FsWrite)
                .unwrap_or(KnownTool::Other {
                    name: name.to_string(),
                    arguments,
                }),
            "terminal_run" => serde_json::from_value(arguments.clone())
                .map(KnownTool::TerminalRun)
                .unwrap_or(KnownTool::Other {
                    name: name.to_string(),
                    arguments,
                }),
            "update_todo_list" => serde_json::from_value(arguments.clone())
                .map(KnownTool::UpdateTodoList)
                .unwrap_or(KnownTool::Other {
                    name: name.to_string(),
                    arguments,
                }),
            "mark_todo_completed" => serde_json::from_value(arguments.clone())
                .map(KnownTool::MarkTodoCompleted)
                .unwrap_or(KnownTool::Other {
                    name: name.to_string(),
                    arguments,
                }),
            _ => KnownTool::Other {
                name: name.to_string(),
                arguments,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_tool_terminal_run_classify_arguments_roundtrip_preserves_args() {
        let input = serde_json::json!({
            "command": "git",
            "args": ["commit", "-m", "x"],
        });

        let tool = KnownTool::classify("terminal_run", input.clone());
        assert_eq!(tool.arguments(), input);
    }

    #[test]
    fn known_todo_tools_classify_and_roundtrip_typed_arguments() {
        let update_args = serde_json::json!({
            "items": [
                {"id": "inspect", "content": "Inspect state", "status": "in_progress"},
                {"id": "verify", "content": "Verify behavior", "status": "pending"}
            ]
        });
        let update = KnownTool::classify("update_todo_list", update_args.clone());
        assert!(matches!(update, KnownTool::UpdateTodoList(_)));
        assert_eq!(update.arguments(), update_args);

        let complete_args = serde_json::json!({"id": "inspect"});
        let complete = KnownTool::classify("mark_todo_completed", complete_args.clone());
        assert!(matches!(complete, KnownTool::MarkTodoCompleted(_)));
        assert_eq!(complete.arguments(), complete_args);

        let invalid = serde_json::json!({
            "items": [{"id": "", "content": "bad", "status": "pending"}]
        });
        assert!(matches!(
            KnownTool::classify("update_todo_list", invalid,),
            KnownTool::Other { .. }
        ));
    }
}

/// A recorded block of assistant "extended thinking".
///
/// Providers such as Anthropic emit reasoning as dedicated `thinking` content
/// blocks that carry a cryptographic `signature`. When extended thinking is used
/// together with tools, the provider **requires** the complete, unmodified
/// thinking block(s) to be sent back on the following request (they must appear
/// in the same assistant turn as the `tool_use`), otherwise it rejects the call.
/// We therefore capture them verbatim here so the compiler can replay them
/// faithfully — an application of the crate-wide "preserve & reuse" rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ThinkingRecord {
    /// A regular thinking block with its opaque cryptographic signature.
    Thinking {
        /// The reasoning text.
        text: String,
        /// The signature that validates this block; must be echoed back unchanged.
        signature: String,
    },
    /// A redacted (encrypted, opaque) thinking block. Its `data` is not human
    /// readable but must still be preserved and echoed back unchanged.
    Redacted {
        /// The opaque encrypted payload.
        data: String,
    },
}

/// A recorded tool call made by the assistant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// Correlation id shared with the matching [`ToolResultRecord`].
    pub id: ToolCallId,
    /// The typed tool invocation.
    pub tool: KnownTool,
}

/// The recorded result of a tool call, fed back into the conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultRecord {
    /// The tool call this result corresponds to.
    pub id: ToolCallId,
    /// Whether the call succeeded.
    pub success: bool,
    /// The result content blocks.
    pub content: Vec<ContentBlock>,
}

/// A previously ingested/compiled fragment preserved verbatim.
///
/// When a message or tool interaction already arrived from an upstream provider
/// or client representation and was compiled once, we keep enough of that
/// compiled form here so downstream stages can *reuse* it instead of lossily
/// re-rendering from the lightweight typed model. See the compiler pipeline
/// (`crate::compiler`) for how this is threaded through as
/// [`crate::compiler::CompiledUnit::Prebuilt`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IngestedRecord {
    /// A short label identifying the origin dialect/provider (e.g. `anthropic`).
    pub source: String,
    /// The exact previously-compiled payload, preserved for faithful reuse.
    pub payload: serde_json::Value,
}

/// A single strongly-typed entry in the conversation history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HistoryEntry {
    /// A user message.
    User(UserMessage),
    /// An assistant message.
    Assistant(AssistantMessage),
    /// A block of assistant extended-thinking (preserved verbatim with its
    /// signature so it can be replayed back to the provider).
    Thinking(ThinkingRecord),
    /// A tool call requested by the assistant.
    ToolCall(ToolCallRecord),
    /// The result of a previously requested tool call.
    ToolResult(ToolResultRecord),
    /// A verbatim, previously-compiled fragment preserved for lossless reuse.
    Ingested(IngestedRecord),
}

/// An ordered, strongly-typed conversation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Conversation {
    /// The ordered history entries.
    pub entries: Vec<HistoryEntry>,
}

impl Conversation {
    /// Create an empty conversation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an entry.
    pub fn push(&mut self, entry: HistoryEntry) {
        self.entries.push(entry);
    }

    /// Append a user text message.
    pub fn push_user_text(&mut self, text: impl Into<String>) {
        self.push(HistoryEntry::User(UserMessage::text(text)));
    }

    /// Append an assistant text message.
    pub fn push_assistant_text(&mut self, text: impl Into<String>) {
        self.push(HistoryEntry::Assistant(AssistantMessage::text(text)));
    }

    /// Append an assistant extended-thinking block.
    pub fn push_thinking(&mut self, record: ThinkingRecord) {
        self.push(HistoryEntry::Thinking(record));
    }

    /// Append a tool call.
    pub fn push_tool_call(&mut self, id: ToolCallId, tool: KnownTool) {
        self.push(HistoryEntry::ToolCall(ToolCallRecord { id, tool }));
    }

    /// Append a tool result.
    pub fn push_tool_result(&mut self, id: ToolCallId, success: bool, text: impl Into<String>) {
        self.push(HistoryEntry::ToolResult(ToolResultRecord {
            id,
            success,
            content: vec![ContentBlock::text(text)],
        }));
    }

    /// Append a previously-ingested/compiled fragment preserved for reuse.
    pub fn push_ingested(&mut self, source: impl Into<String>, payload: serde_json::Value) {
        self.push(HistoryEntry::Ingested(IngestedRecord {
            source: source.into(),
            payload,
        }));
    }

    /// The number of entries in the conversation.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the conversation is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
