//! # orchestrated
//!
//! Orchestrated execution: a top-level orchestrator LLM drives work through an
//! *un-fixed* pipeline by repeatedly calling a single `run_subagent` tool, which
//! delegates each implementation/review/planning action to a sub-agent run in a
//! specific [`mode`](crate::mode).
//!
//! This crate depends only on `agent-core` trait seams (plus `turn-replay` as a
//! dev-dependency for deterministic tests); `agent-core` itself is untouched.
//!
//! ## Module map
//! - [`context`] — shared state ([`OrchestratedStepContext`]) threading results,
//!   verdicts, and attempt counters between sub-agent runs.
//! - [`mode`] — the [`SubAgentMode`](mode::SubAgentMode) contract, the three
//!   family behaviors, and the six concrete modes.
//! - [`registry`] — the [`ModeRegistry`](registry::ModeRegistry) resolving a
//!   mode id (with a structured error for unknown modes).

#![warn(missing_docs)]

pub mod action;
pub mod context;
pub mod mode;
pub mod orchestrator;
pub mod pipeline;
pub mod registry;
pub mod submit;
pub mod worker;

pub use action::{RunSubAgentTool, RUN_SUBAGENT_TOOL_NAME};
pub use context::{ModelTier, OrchestratedStepContext, PlanProposal, PlanStepSpec, StepResult};
pub use mode::{
    ExecutorBehavior, ExecutorMode, OrchestratorKind, PlanMode, PlannerBehavior, PreconditionError,
    ReviewMode, ReviewPlanMode, ReviewerBehavior, SubAgentMode, SubAgentRequest, SubmitKind,
    SubmitOutcome, EXECUTOR_MODE_IDS,
};
pub use orchestrator::{Orchestrator, OrchestratorBuilder, ORCHESTRATOR_SYSTEM_PROMPT};
pub use pipeline::{
    run_pipeline, run_pipeline_observed, uniform_resolver, BackendFactory, OrchestrateTool,
    PipelineSession, ORCHESTRATE_TOOL_NAME,
};
pub use registry::{ModeRegistry, UnknownModeError};
pub use submit::{new_slot, SubmitReviewTool, SubmitSlot, SubmitTool};
pub use worker::{
    FnModelResolver, ModelResolver, OrchestratedSubAgentWorker, SubAgentRunOutcome,
    SubAgentRunParams,
};
