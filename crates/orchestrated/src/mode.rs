//! The sub-agent mode contract and the three mode families.
//!
//! A [`SubAgentMode`] describes *one way* the orchestrator can run a sub-agent
//! (implement, review, plan, ...). Behavior is shared **through composition**:
//! each concrete mode holds an `Arc` to a family behavior
//! ([`ExecutorBehavior`] / [`ReviewerBehavior`] / [`PlannerBehavior`]) and only
//! overrides its `id`, prompt prefix, and model specifics.

use std::sync::Arc;

use agent_core::ToolRegistry;

use crate::context::{ModelTier, OrchestratedStepContext, PlanStepSpec};

/// The mode ids of the executor family (used for reviewer preconditions).
pub const EXECUTOR_MODE_IDS: &[&str] = &["code", "setup", "niche"];

/// The high-level family a mode belongs to, from the orchestrator's view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrchestratorKind {
    /// Does the work (code/setup/niche).
    Executor,
    /// Reviews a prior result and reports a verdict.
    Reviewer,
    /// Produces or refines a plan.
    Planner,
}

/// Which submit action a mode injects into its sub-agent turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitKind {
    /// A plain `submit { output }` action (executors, planner).
    Plain,
    /// A structured `submit_review { approved, reasons }` action (reviewers).
    Review,
}

/// A precondition failure that aborts a `run_subagent` call gracefully.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreconditionError {
    /// Human-readable reason the precondition was not met.
    pub message: String,
}

impl PreconditionError {
    /// Create a precondition error with the given message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for PreconditionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for PreconditionError {}

/// A parsed `run_subagent` request.
#[derive(Debug, Clone, PartialEq)]
pub struct SubAgentRequest {
    /// The requested mode id (already validated against the registry).
    pub mode: String,
    /// The 0-based plan step index.
    pub step_index: usize,
    /// Optional free-form instructions from the orchestrator.
    pub instructions: Option<String>,
    /// Optional scope narrowing (files/areas) from the orchestrator.
    pub scope: Option<String>,
    /// Optional explicit model tier override.
    pub model_tier: Option<ModelTier>,
}

/// The typed outcome a sub-agent reports through its submit action.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SubmitOutcome {
    /// Whether the sub-agent reported success (a positive verdict for reviewers).
    pub success: bool,
    /// The assistant-visible output/summary.
    pub output: String,
    /// Optional reviewer reasons (only set by `submit_review`).
    pub reasons: Option<String>,
}

/// One way the orchestrator can run a sub-agent.
///
/// All methods are synchronous policy/prompt scaffolding; the actual nested
/// turn is driven by the worker (see `crate::worker`). Modes are cheap value
/// objects held behind `Arc<dyn SubAgentMode>` in the [`crate::registry::ModeRegistry`].
pub trait SubAgentMode: Send + Sync {
    /// The unique mode id used in `run_subagent(mode=...)`.
    fn id(&self) -> &str;

    /// The family this mode belongs to.
    fn orchestrator_kind(&self) -> OrchestratorKind;

    /// Whether this mode creates/refines a plan (may run without a proposal).
    fn is_plan_creation_mode(&self) -> bool {
        false
    }

    /// Whether the orchestrator may retry this mode for the same step.
    fn supports_retry(&self) -> bool;

    /// Whether the orchestrator may escalate the model tier on retry.
    fn supports_model_escalation(&self) -> bool;

    /// A fixed model tier this mode always runs at, overriding escalation.
    fn fixed_model(&self) -> Option<ModelTier> {
        None
    }

    /// Max sub-agent steps on the first attempt.
    fn initial_max_steps(&self) -> usize;

    /// Max sub-agent steps on retries.
    fn retry_max_steps(&self) -> usize;

    /// Which submit action the sub-agent turn should expose.
    fn submit_kind(&self) -> SubmitKind;

    /// The system prompt seeded into the sub-agent session.
    fn system_prompt(&self, ctx: &OrchestratedStepContext, step_index: usize) -> String;

    /// Derive the tool set the sub-agent may use from the base registry.
    fn tool_capabilities(&self, base: &ToolRegistry) -> ToolRegistry;

    /// Build the issue/task description handed to the sub-agent.
    fn build_issue_description(&self, req: &SubAgentRequest, step: Option<&PlanStepSpec>)
        -> String;

    /// Build pre-chat observations (plan, prior results, verdicts) for context.
    fn build_pre_chat_observations(
        &self,
        ctx: &OrchestratedStepContext,
        step_index: usize,
    ) -> String;

    /// Validate ordering preconditions before running (e.g. review needs a
    /// prior executor result). Returns `Err` to fail the call gracefully.
    fn check_preconditions(
        &self,
        ctx: &OrchestratedStepContext,
        step_index: usize,
    ) -> std::result::Result<(), PreconditionError>;

    /// Interpret a submit outcome as success (reviewers key off the verdict).
    fn parse_success(&self, submit: &SubmitOutcome) -> bool;

    /// Compress the submit outcome into the text stored in the context.
    fn compress_result(&self, submit: &SubmitOutcome) -> String;

    /// A concise summary returned to the orchestrator to decide the next step.
    fn display_text(&self, output: &str) -> String;
}

// ---------------------------------------------------------------------------
// Shared family behaviors (composition).
// ---------------------------------------------------------------------------

/// Shared policy for executor modes (code/setup/niche): full tool access,
/// retries and model escalation, a plain submit action.
#[derive(Debug, Default)]
pub struct ExecutorBehavior;

impl ExecutorBehavior {
    fn supports_retry(&self) -> bool {
        true
    }
    fn supports_model_escalation(&self) -> bool {
        true
    }
    fn initial_max_steps(&self) -> usize {
        50
    }
    fn retry_max_steps(&self) -> usize {
        30
    }
    fn submit_kind(&self) -> SubmitKind {
        SubmitKind::Plain
    }

    /// Executors get the full base tool set.
    fn tool_capabilities(&self, base: &ToolRegistry) -> ToolRegistry {
        base.clone()
    }

    fn system_prompt(
        &self,
        prefix: &str,
        ctx: &OrchestratedStepContext,
        step_index: usize,
    ) -> String {
        let mut p = format!(
            "{prefix}\nYou are an executor sub-agent. Do the work for the current plan step, \
             then call `submit` with a concise summary of what you changed."
        );
        if let Some(step) = ctx.proposal().and_then(|pr| pr.step(step_index).cloned()) {
            p.push_str(&format!("\n\nCurrent step: {}", step.name));
        }
        p
    }

    fn build_issue_description(
        &self,
        req: &SubAgentRequest,
        step: Option<&PlanStepSpec>,
    ) -> String {
        let mut d = String::new();
        if let Some(step) = step {
            d.push_str(&format!("Step: {}\n{}\n", step.name, step.description));
        }
        if let Some(scope) = &req.scope {
            d.push_str(&format!("\nScope: {scope}\n"));
        }
        if let Some(instr) = &req.instructions {
            d.push_str(&format!("\nInstructions: {instr}\n"));
        }
        d
    }

    fn build_pre_chat_observations(
        &self,
        ctx: &OrchestratedStepContext,
        step_index: usize,
    ) -> String {
        let mut obs = String::new();
        if let Some(proposal) = ctx.proposal() {
            obs.push_str(&proposal.render());
        }
        // Surface any prior reviewer verdict for this step to guide a retry.
        for r in ctx.results_for_step(step_index) {
            if !EXECUTOR_MODE_IDS.contains(&r.mode_id.as_str()) {
                obs.push_str(&format!(
                    "\nPrevious {} verdict (success={}): {}\n",
                    r.mode_id, r.success, r.output
                ));
            }
        }
        obs
    }

    fn parse_success(&self, submit: &SubmitOutcome) -> bool {
        submit.success
    }

    fn compress_result(&self, submit: &SubmitOutcome) -> String {
        submit.output.clone()
    }

    fn display_text(&self, output: &str) -> String {
        output.to_string()
    }
}

/// Shared policy for reviewer modes: read-only tools, no retry/escalation, a
/// fixed high-capability model, success determined by the structured verdict.
#[derive(Debug, Default)]
pub struct ReviewerBehavior;

impl ReviewerBehavior {
    fn supports_retry(&self) -> bool {
        false
    }
    fn supports_model_escalation(&self) -> bool {
        false
    }
    fn fixed_model(&self) -> Option<ModelTier> {
        Some(ModelTier::High)
    }
    fn initial_max_steps(&self) -> usize {
        30
    }
    fn retry_max_steps(&self) -> usize {
        30
    }
    fn submit_kind(&self) -> SubmitKind {
        SubmitKind::Review
    }

    /// Reviewers only get read-only tools (those not requiring permission).
    fn tool_capabilities(&self, base: &ToolRegistry) -> ToolRegistry {
        let mut filtered = ToolRegistry::new();
        for name in base.names() {
            if let Some(tool) = base.get(name) {
                if !tool.requires_permission() {
                    filtered.register(tool);
                }
            }
        }
        filtered
    }

    fn system_prompt(&self, prefix: &str) -> String {
        format!(
            "{prefix}\nYou are a reviewer sub-agent. Inspect the prior result and decide \
             whether it satisfies the requirements. Report your verdict via `submit_review` \
             with `approved` and `reasons`."
        )
    }

    fn build_issue_description(
        &self,
        req: &SubAgentRequest,
        step: Option<&PlanStepSpec>,
    ) -> String {
        let mut d = String::from("Review the work for this step.\n");
        if let Some(step) = step {
            d.push_str(&format!("Step: {}\n{}\n", step.name, step.description));
        }
        if let Some(instr) = &req.instructions {
            d.push_str(&format!("\nReview focus: {instr}\n"));
        }
        d
    }

    fn build_pre_chat_observations(
        &self,
        ctx: &OrchestratedStepContext,
        step_index: usize,
    ) -> String {
        let mut obs = String::new();
        for r in ctx.results_for_step(step_index) {
            if EXECUTOR_MODE_IDS.contains(&r.mode_id.as_str()) {
                obs.push_str(&format!(
                    "\nExecutor ({}) result:\n{}\n",
                    r.mode_id, r.output
                ));
            }
        }
        obs
    }

    fn parse_success(&self, submit: &SubmitOutcome) -> bool {
        submit.success
    }

    fn compress_result(&self, submit: &SubmitOutcome) -> String {
        match &submit.reasons {
            Some(reasons) => format!("approved={}: {}", submit.success, reasons),
            None => format!("approved={}", submit.success),
        }
    }

    fn display_text(&self, output: &str) -> String {
        output.to_string()
    }
}

/// Shared policy for the planner mode: produces a plan proposal; no retry.
#[derive(Debug, Default)]
pub struct PlannerBehavior;

impl PlannerBehavior {
    fn supports_retry(&self) -> bool {
        false
    }
    fn supports_model_escalation(&self) -> bool {
        false
    }
    fn initial_max_steps(&self) -> usize {
        30
    }
    fn retry_max_steps(&self) -> usize {
        30
    }
    fn submit_kind(&self) -> SubmitKind {
        SubmitKind::Plain
    }

    fn tool_capabilities(&self, base: &ToolRegistry) -> ToolRegistry {
        // Planner reasons/reads but does not modify: read-only tools only.
        let mut filtered = ToolRegistry::new();
        for name in base.names() {
            if let Some(tool) = base.get(name) {
                if !tool.requires_permission() {
                    filtered.register(tool);
                }
            }
        }
        filtered
    }

    fn parse_success(&self, submit: &SubmitOutcome) -> bool {
        submit.success
    }

    fn compress_result(&self, submit: &SubmitOutcome) -> String {
        submit.output.clone()
    }

    fn display_text(&self, output: &str) -> String {
        output.to_string()
    }
}

// ---------------------------------------------------------------------------
// Concrete executor modes.
// ---------------------------------------------------------------------------

/// A generic executor mode parameterized by id and prompt prefix.
pub struct ExecutorMode {
    id: &'static str,
    prompt_prefix: &'static str,
    behavior: Arc<ExecutorBehavior>,
}

impl ExecutorMode {
    fn new(id: &'static str, prompt_prefix: &'static str, behavior: Arc<ExecutorBehavior>) -> Self {
        Self {
            id,
            prompt_prefix,
            behavior,
        }
    }
}

impl SubAgentMode for ExecutorMode {
    fn id(&self) -> &str {
        self.id
    }
    fn orchestrator_kind(&self) -> OrchestratorKind {
        OrchestratorKind::Executor
    }
    fn supports_retry(&self) -> bool {
        self.behavior.supports_retry()
    }
    fn supports_model_escalation(&self) -> bool {
        self.behavior.supports_model_escalation()
    }
    fn initial_max_steps(&self) -> usize {
        self.behavior.initial_max_steps()
    }
    fn retry_max_steps(&self) -> usize {
        self.behavior.retry_max_steps()
    }
    fn submit_kind(&self) -> SubmitKind {
        self.behavior.submit_kind()
    }
    fn system_prompt(&self, ctx: &OrchestratedStepContext, step_index: usize) -> String {
        self.behavior
            .system_prompt(self.prompt_prefix, ctx, step_index)
    }
    fn tool_capabilities(&self, base: &ToolRegistry) -> ToolRegistry {
        self.behavior.tool_capabilities(base)
    }
    fn build_issue_description(
        &self,
        req: &SubAgentRequest,
        step: Option<&PlanStepSpec>,
    ) -> String {
        self.behavior.build_issue_description(req, step)
    }
    fn build_pre_chat_observations(
        &self,
        ctx: &OrchestratedStepContext,
        step_index: usize,
    ) -> String {
        self.behavior.build_pre_chat_observations(ctx, step_index)
    }
    fn check_preconditions(
        &self,
        _ctx: &OrchestratedStepContext,
        _step_index: usize,
    ) -> std::result::Result<(), PreconditionError> {
        Ok(())
    }
    fn parse_success(&self, submit: &SubmitOutcome) -> bool {
        self.behavior.parse_success(submit)
    }
    fn compress_result(&self, submit: &SubmitOutcome) -> String {
        self.behavior.compress_result(submit)
    }
    fn display_text(&self, output: &str) -> String {
        self.behavior.display_text(output)
    }
}

/// The `code` executor mode: general implementation work.
pub fn code_mode(behavior: Arc<ExecutorBehavior>) -> ExecutorMode {
    ExecutorMode::new("code", "Implement the requested code changes.", behavior)
}

/// The `setup` executor mode: environment/build/configuration work.
pub fn setup_mode(behavior: Arc<ExecutorBehavior>) -> ExecutorMode {
    ExecutorMode::new(
        "setup",
        "Set up the environment, dependencies, or build configuration.",
        behavior,
    )
}

/// The `niche` executor mode: specialized non-standard work.
pub fn niche_mode(behavior: Arc<ExecutorBehavior>) -> ExecutorMode {
    ExecutorMode::new(
        "niche",
        "Handle a specialized/niche task outside ordinary implementation.",
        behavior,
    )
}

// ---------------------------------------------------------------------------
// Concrete reviewer modes.
// ---------------------------------------------------------------------------

/// Reviews an executor result for a plan step.
pub struct ReviewMode {
    behavior: Arc<ReviewerBehavior>,
}

impl ReviewMode {
    /// Create a review mode delegating to the shared reviewer behavior.
    pub fn new(behavior: Arc<ReviewerBehavior>) -> Self {
        Self { behavior }
    }
}

impl SubAgentMode for ReviewMode {
    fn id(&self) -> &str {
        "review"
    }
    fn orchestrator_kind(&self) -> OrchestratorKind {
        OrchestratorKind::Reviewer
    }
    fn supports_retry(&self) -> bool {
        self.behavior.supports_retry()
    }
    fn supports_model_escalation(&self) -> bool {
        self.behavior.supports_model_escalation()
    }
    fn fixed_model(&self) -> Option<ModelTier> {
        self.behavior.fixed_model()
    }
    fn initial_max_steps(&self) -> usize {
        self.behavior.initial_max_steps()
    }
    fn retry_max_steps(&self) -> usize {
        self.behavior.retry_max_steps()
    }
    fn submit_kind(&self) -> SubmitKind {
        self.behavior.submit_kind()
    }
    fn system_prompt(&self, _ctx: &OrchestratedStepContext, _step_index: usize) -> String {
        self.behavior
            .system_prompt("Review the implementation for correctness.")
    }
    fn tool_capabilities(&self, base: &ToolRegistry) -> ToolRegistry {
        self.behavior.tool_capabilities(base)
    }
    fn build_issue_description(
        &self,
        req: &SubAgentRequest,
        step: Option<&PlanStepSpec>,
    ) -> String {
        self.behavior.build_issue_description(req, step)
    }
    fn build_pre_chat_observations(
        &self,
        ctx: &OrchestratedStepContext,
        step_index: usize,
    ) -> String {
        self.behavior.build_pre_chat_observations(ctx, step_index)
    }
    fn check_preconditions(
        &self,
        ctx: &OrchestratedStepContext,
        step_index: usize,
    ) -> std::result::Result<(), PreconditionError> {
        if ctx.has_result_from_any(step_index, EXECUTOR_MODE_IDS) {
            Ok(())
        } else {
            Err(PreconditionError::new(format!(
                "cannot review step {}: no prior executor result exists",
                step_index + 1
            )))
        }
    }
    fn parse_success(&self, submit: &SubmitOutcome) -> bool {
        self.behavior.parse_success(submit)
    }
    fn compress_result(&self, submit: &SubmitOutcome) -> String {
        self.behavior.compress_result(submit)
    }
    fn display_text(&self, output: &str) -> String {
        self.behavior.display_text(output)
    }
}

/// Reviews a plan proposal produced by the planner.
pub struct ReviewPlanMode {
    behavior: Arc<ReviewerBehavior>,
}

impl ReviewPlanMode {
    /// Create a review-plan mode delegating to the shared reviewer behavior.
    pub fn new(behavior: Arc<ReviewerBehavior>) -> Self {
        Self { behavior }
    }
}

impl SubAgentMode for ReviewPlanMode {
    fn id(&self) -> &str {
        "review_plan"
    }
    fn orchestrator_kind(&self) -> OrchestratorKind {
        OrchestratorKind::Reviewer
    }
    fn supports_retry(&self) -> bool {
        self.behavior.supports_retry()
    }
    fn supports_model_escalation(&self) -> bool {
        self.behavior.supports_model_escalation()
    }
    fn fixed_model(&self) -> Option<ModelTier> {
        self.behavior.fixed_model()
    }
    fn initial_max_steps(&self) -> usize {
        self.behavior.initial_max_steps()
    }
    fn retry_max_steps(&self) -> usize {
        self.behavior.retry_max_steps()
    }
    fn submit_kind(&self) -> SubmitKind {
        self.behavior.submit_kind()
    }
    fn system_prompt(&self, _ctx: &OrchestratedStepContext, _step_index: usize) -> String {
        self.behavior
            .system_prompt("Review the proposed plan for completeness and feasibility.")
    }
    fn tool_capabilities(&self, base: &ToolRegistry) -> ToolRegistry {
        self.behavior.tool_capabilities(base)
    }
    fn build_issue_description(
        &self,
        req: &SubAgentRequest,
        step: Option<&PlanStepSpec>,
    ) -> String {
        self.behavior.build_issue_description(req, step)
    }
    fn build_pre_chat_observations(
        &self,
        ctx: &OrchestratedStepContext,
        _step_index: usize,
    ) -> String {
        match ctx.proposal() {
            Some(p) => p.render(),
            None => String::new(),
        }
    }
    fn check_preconditions(
        &self,
        ctx: &OrchestratedStepContext,
        _step_index: usize,
    ) -> std::result::Result<(), PreconditionError> {
        if ctx.proposal().is_some() {
            Ok(())
        } else {
            Err(PreconditionError::new(
                "cannot review a plan: no plan proposal exists yet",
            ))
        }
    }
    fn parse_success(&self, submit: &SubmitOutcome) -> bool {
        self.behavior.parse_success(submit)
    }
    fn compress_result(&self, submit: &SubmitOutcome) -> String {
        self.behavior.compress_result(submit)
    }
    fn display_text(&self, output: &str) -> String {
        self.behavior.display_text(output)
    }
}

// ---------------------------------------------------------------------------
// Concrete planner mode.
// ---------------------------------------------------------------------------

/// Produces or refines the plan proposal.
pub struct PlanMode {
    behavior: Arc<PlannerBehavior>,
}

impl PlanMode {
    /// Create a plan mode delegating to the shared planner behavior.
    pub fn new(behavior: Arc<PlannerBehavior>) -> Self {
        Self { behavior }
    }
}

impl SubAgentMode for PlanMode {
    fn id(&self) -> &str {
        "plan"
    }
    fn orchestrator_kind(&self) -> OrchestratorKind {
        OrchestratorKind::Planner
    }
    fn is_plan_creation_mode(&self) -> bool {
        true
    }
    fn supports_retry(&self) -> bool {
        self.behavior.supports_retry()
    }
    fn supports_model_escalation(&self) -> bool {
        self.behavior.supports_model_escalation()
    }
    fn initial_max_steps(&self) -> usize {
        self.behavior.initial_max_steps()
    }
    fn retry_max_steps(&self) -> usize {
        self.behavior.retry_max_steps()
    }
    fn submit_kind(&self) -> SubmitKind {
        self.behavior.submit_kind()
    }
    fn system_prompt(&self, _ctx: &OrchestratedStepContext, _step_index: usize) -> String {
        "You are a planner sub-agent. Produce an ordered, actionable plan for the goal, \
         then call `submit` with the plan."
            .to_string()
    }
    fn tool_capabilities(&self, base: &ToolRegistry) -> ToolRegistry {
        self.behavior.tool_capabilities(base)
    }
    fn build_issue_description(
        &self,
        req: &SubAgentRequest,
        _step: Option<&PlanStepSpec>,
    ) -> String {
        let mut d = String::from("Create a plan for the goal.\n");
        if let Some(instr) = &req.instructions {
            d.push_str(&format!("\nGuidance: {instr}\n"));
        }
        d
    }
    fn build_pre_chat_observations(
        &self,
        ctx: &OrchestratedStepContext,
        _step_index: usize,
    ) -> String {
        match ctx.proposal() {
            Some(p) => format!("Existing plan to refine:\n{}", p.render()),
            None => String::new(),
        }
    }
    fn check_preconditions(
        &self,
        _ctx: &OrchestratedStepContext,
        _step_index: usize,
    ) -> std::result::Result<(), PreconditionError> {
        Ok(())
    }
    fn parse_success(&self, submit: &SubmitOutcome) -> bool {
        self.behavior.parse_success(submit)
    }
    fn compress_result(&self, submit: &SubmitOutcome) -> String {
        self.behavior.compress_result(submit)
    }
    fn display_text(&self, output: &str) -> String {
        self.behavior.display_text(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{PlanProposal, PlanStepSpec, StepResult};

    fn executor() -> ExecutorMode {
        code_mode(Arc::new(ExecutorBehavior))
    }
    fn review() -> ReviewMode {
        ReviewMode::new(Arc::new(ReviewerBehavior))
    }
    fn review_plan() -> ReviewPlanMode {
        ReviewPlanMode::new(Arc::new(ReviewerBehavior))
    }
    fn plan() -> PlanMode {
        PlanMode::new(Arc::new(PlannerBehavior))
    }

    fn ctx_with_step() -> Arc<OrchestratedStepContext> {
        OrchestratedStepContext::with_proposal(PlanProposal::new(
            "goal",
            vec![PlanStepSpec::new("Implement X", "do the thing")],
        ))
    }

    #[test]
    fn executor_flags_and_submit_kind() {
        let m = executor();
        assert_eq!(m.id(), "code");
        assert_eq!(m.orchestrator_kind(), OrchestratorKind::Executor);
        assert!(m.supports_retry());
        assert!(m.supports_model_escalation());
        assert!(m.fixed_model().is_none());
        assert_eq!(m.submit_kind(), SubmitKind::Plain);
        assert!(!m.is_plan_creation_mode());
        // Executor preconditions are always satisfied.
        assert!(m.check_preconditions(&ctx_with_step(), 0).is_ok());
    }

    #[test]
    fn reviewer_flags_fixed_model_no_retry() {
        let m = review();
        assert_eq!(m.id(), "review");
        assert_eq!(m.orchestrator_kind(), OrchestratorKind::Reviewer);
        assert!(!m.supports_retry());
        assert!(!m.supports_model_escalation());
        assert_eq!(m.fixed_model(), Some(ModelTier::High));
        assert_eq!(m.submit_kind(), SubmitKind::Review);
    }

    #[test]
    fn planner_flags_and_plan_creation() {
        let m = plan();
        assert_eq!(m.id(), "plan");
        assert_eq!(m.orchestrator_kind(), OrchestratorKind::Planner);
        assert!(m.is_plan_creation_mode());
        assert!(!m.supports_retry());
        assert_eq!(m.submit_kind(), SubmitKind::Plain);
        // No plan required to run the planner.
        assert!(m
            .check_preconditions(&OrchestratedStepContext::new(), 0)
            .is_ok());
    }

    #[test]
    fn review_precondition_requires_prior_executor_result() {
        let ctx = ctx_with_step();
        let m = review();
        // No executor result yet -> precondition fails gracefully.
        let err = m.check_preconditions(&ctx, 0).expect_err("should fail");
        assert!(err.message.contains("no prior executor result"));

        // After an executor result is stored, the precondition passes.
        ctx.store_result(StepResult {
            mode_id: "code".into(),
            step_index: 0,
            success: true,
            output: "did it".into(),
        });
        assert!(m.check_preconditions(&ctx, 0).is_ok());
    }

    #[test]
    fn review_plan_precondition_requires_proposal() {
        let m = review_plan();
        // Empty context (no proposal) -> fails.
        assert!(m
            .check_preconditions(&OrchestratedStepContext::new(), 0)
            .is_err());
        // With a proposal -> passes.
        assert!(m.check_preconditions(&ctx_with_step(), 0).is_ok());
    }

    #[test]
    fn reviewer_compress_result_includes_verdict() {
        let m = review();
        let outcome = SubmitOutcome {
            success: false,
            output: String::new(),
            reasons: Some("missing tests".into()),
        };
        assert!(!m.parse_success(&outcome));
        let compressed = m.compress_result(&outcome);
        assert!(compressed.contains("approved=false"));
        assert!(compressed.contains("missing tests"));
    }

    #[test]
    fn executor_pre_chat_surfaces_prior_verdict() {
        let ctx = ctx_with_step();
        ctx.store_result(StepResult {
            mode_id: "review".into(),
            step_index: 0,
            success: false,
            output: "needs fixes".into(),
        });
        let m = executor();
        let obs = m.build_pre_chat_observations(&ctx, 0);
        assert!(obs.contains("Goal: goal"));
        assert!(obs.contains("Previous review verdict"));
        assert!(obs.contains("needs fixes"));
    }

    #[test]
    fn reviewer_tool_capabilities_filter_out_destructive() {
        use agent_core::{Result, Tool, ToolContext, ToolEvent, ToolRegistry};
        use async_trait::async_trait;
        use futures::stream::{self, BoxStream};

        struct FakeTool {
            name: &'static str,
            destructive: bool,
        }
        #[async_trait]
        impl Tool for FakeTool {
            fn name(&self) -> &str {
                self.name
            }
            fn schema(&self) -> serde_json::Value {
                serde_json::json!({ "type": "object" })
            }
            fn requires_permission(&self) -> bool {
                self.destructive
            }
            async fn call(
                &self,
                _args: serde_json::Value,
                _ctx: &ToolContext,
            ) -> Result<BoxStream<'static, ToolEvent>> {
                Ok(Box::pin(stream::iter(vec![ToolEvent::Started])))
            }
        }

        let mut base = ToolRegistry::new();
        base.register(Arc::new(FakeTool {
            name: "fs_read",
            destructive: false,
        }));
        base.register(Arc::new(FakeTool {
            name: "fs_write",
            destructive: true,
        }));

        let reviewer_tools = review().tool_capabilities(&base);
        assert!(reviewer_tools.contains("fs_read"));
        assert!(!reviewer_tools.contains("fs_write"));

        // Executors keep the full set.
        let exec_tools = executor().tool_capabilities(&base);
        assert!(exec_tools.contains("fs_read"));
        assert!(exec_tools.contains("fs_write"));
    }
}
