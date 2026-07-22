//! The services layer: callable [`Tool`]s and the [`ToolRegistry`].

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::agent::AgentPath;
use crate::cancel::CancellationToken;
use crate::client::ClientAccess;
use crate::error::{AgentError, Result};
use crate::event::ToolCallId;
use crate::observe::SharedTurnObserver;
use crate::session::ToolDescriptor;
use crate::sink::{ToolCallLocation, ToolKind};
use crate::todo::TodoList;
use crate::trace::TraceContext;

/// A streaming event emitted while a tool executes.
#[derive(Debug, Clone)]
pub enum ToolEvent {
    /// The tool has started executing.
    Started,
    /// An incremental chunk of human-readable output.
    OutputDelta(String),
    /// The tool finished successfully with a final result payload.
    Completed(serde_json::Value),
    /// The tool failed with a descriptive message.
    Failed(String),
    /// A validated replacement for the session's canonical todo state.
    TodoListUpdated(TodoList),
}

/// How a tool's human-readable output is presented to the frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolOutputPresentation {
    /// Render lifecycle and output through tool-call updates.
    #[default]
    ToolCall,
    /// Forward output deltas immediately as assistant message chunks.
    StreamingMessage,
    /// Buffer all output deltas and emit one assistant message on completion.
    AccumulatedMessage,
}

/// Context passed to a [`Tool`] when it is invoked.
///
/// Holds access needed to perform work: the session id, a cancellation token
/// shared with the turn engine, and the current subagent nesting depth (used to
/// bound recursion).
#[derive(Clone)]
pub struct ToolContext {
    /// The id of the session this tool call belongs to.
    pub session_id: String,
    /// Cancellation token observed by the tool.
    pub cancel: CancellationToken,
    /// Current subagent nesting depth (0 for a top-level turn).
    pub depth: usize,
    /// Ordered workspace roots; index `0` is the working directory.
    pub workspace_roots: Vec<String>,
    /// Provider-independent access to the ACP client (fs/terminal).
    ///
    /// Optional so tests and non-ACP contexts (e.g. the replay integration
    /// tests) can build a context without a live client. Tools that need it
    /// should obtain it via [`ToolContext::client`], which yields a structured
    /// error when absent rather than panicking.
    pub client: Option<Arc<dyn ClientAccess>>,
    /// The provider-neutral observer of the engine that dispatched this tool,
    /// if any. Tools that spawn nested engines (subagents, orchestration)
    /// should attach it so the nested agent's requests/responses are captured
    /// too. `None` when observation is disabled.
    pub observer: Option<SharedTurnObserver>,
    /// The [`AgentPath`] of the agent that dispatched this tool. Tools that
    /// spawn a nested engine should give it `agent_path.child(<label>)` so the
    /// sub-agent's output is attributed to it specifically.
    pub agent_path: AgentPath,
    /// Trace of the parent decision round that dispatched this tool.
    pub trace: Option<TraceContext>,
    /// Id of this concrete invocation, used to parent a nested agent run.
    pub tool_call_id: Option<ToolCallId>,
    /// Immutable snapshot of the session's canonical todo state at dispatch.
    pub todo_list: TodoList,
}

impl ToolContext {
    /// Create a new tool context for the given session, without a client.
    pub fn new(session_id: impl Into<String>, cancel: CancellationToken) -> Self {
        Self {
            session_id: session_id.into(),
            cancel,
            depth: 0,
            workspace_roots: Vec::new(),
            client: None,
            observer: None,
            agent_path: AgentPath::root(),
            trace: None,
            tool_call_id: None,
            todo_list: TodoList::default(),
        }
    }

    /// Attach a [`ClientAccess`] implementation to this context.
    pub fn with_client(mut self, client: Arc<dyn ClientAccess>) -> Self {
        self.client = Some(client);
        self
    }

    /// Access the [`ClientAccess`] handle, or a structured error if none is set.
    pub fn client(&self) -> Result<&Arc<dyn ClientAccess>> {
        self.client.as_ref().ok_or_else(|| {
            AgentError::Other("no client access available for this tool context".to_string())
        })
    }
}

/// A callable capability invoked by the turn engine.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The unique name used to reference this tool.
    fn name(&self) -> &str;

    /// JSON schema describing the tool's arguments.
    fn schema(&self) -> serde_json::Value;

    /// Whether this tool requires explicit user permission before execution.
    fn requires_permission(&self) -> bool;

    /// The category of this tool, used by clients for icons and UI treatment.
    ///
    /// Defaults to [`ToolKind::Other`].
    fn kind(&self) -> ToolKind {
        ToolKind::Other
    }

    /// A human-readable title describing a specific invocation with `args`.
    ///
    /// Returning `None` (the default) lets the caller fall back to the tool
    /// name. Implementations may derive a richer title from the arguments
    /// (e.g. `"Read /path/to/file"`).
    fn title(&self, _args: &serde_json::Value) -> Option<String> {
        None
    }

    /// File locations this invocation touches, for client "follow-along".
    ///
    /// Defaults to an empty list.
    fn locations(&self, _args: &serde_json::Value) -> Vec<ToolCallLocation> {
        Vec::new()
    }

    /// Whether dispatching this tool ends the current turn and hands control
    /// back to the user.
    ///
    /// Defaults to `false`. A tool that returns `true` is a *terminal tool*:
    /// when the engine is configured to require one (see
    /// [`TurnEngine::require_terminal_tool`](crate::TurnEngine::require_terminal_tool)),
    /// a decision round that produces only text does not end the turn, and the
    /// turn ends only once some terminal tool is dispatched. Multiple terminal
    /// tools may coexist; the engine treats any of them as a valid way to end
    /// the turn.
    fn ends_turn(&self) -> bool {
        false
    }

    /// Choose how the tool's human-readable output is shown to the user.
    fn output_presentation(&self) -> ToolOutputPresentation {
        ToolOutputPresentation::ToolCall
    }

    /// Execute the tool, streaming [`ToolEvent`]s back to the caller.
    async fn call(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>>;
}

/// A registry mapping tool names to their implementations.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool, keyed by its [`Tool::name`].
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Whether a tool with the given name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// The number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Iterate over the names of all registered tools.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(|s| s.as_str())
    }

    /// Snapshot the schemas advertised to the decision layer.
    pub fn descriptors(&self) -> Vec<ToolDescriptor> {
        self.tools
            .iter()
            .map(|(name, tool)| ToolDescriptor {
                name: name.clone(),
                schema: tool.schema(),
                requires_permission: tool.requires_permission(),
            })
            .collect()
    }
}

/// The outcome of dispatching a tool call, fed back into the conversation.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// The tool call this result corresponds to.
    pub id: ToolCallId,
    /// Whether the call succeeded.
    pub success: bool,
    /// A textual summary suitable for feeding back to the decision layer.
    pub content: String,
}
