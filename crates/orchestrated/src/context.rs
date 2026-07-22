//! Shared orchestration state: the [`PlanProposal`] being executed and the
//! per-`(mode, step)` results/attempts accumulated across `run_subagent` calls.
//!
//! This mirrors the `OrchestratedStepContext` of the reference design: it is the
//! single place where results flow *between* sub-agent modes (an executor's
//! output is later read by a reviewer; a reviewer's verdict is read by the next
//! executor retry). It is shared behind an [`Arc`] and guarded by a
//! [`std::sync::Mutex`]; guards are never held across an `.await`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

/// The relative capability/cost of the model backing a sub-agent turn.
///
/// The orchestrator can request a stronger tier (escalation) when a step keeps
/// failing. Tiers are ordered; [`ModelTier::escalated`] moves one step up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTier {
    /// Cheapest / fastest tier.
    Low,
    /// Balanced default tier.
    #[default]
    Medium,
    /// Most capable / most expensive tier.
    High,
}

impl ModelTier {
    /// Parse a tier from a case-insensitive string, if recognized.
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" => Some(ModelTier::Low),
            "medium" | "mid" | "default" => Some(ModelTier::Medium),
            "high" | "max" => Some(ModelTier::High),
            _ => None,
        }
    }

    /// Return the next stronger tier, saturating at [`ModelTier::High`].
    pub fn escalated(self) -> Self {
        match self {
            ModelTier::Low => ModelTier::Medium,
            ModelTier::Medium => ModelTier::High,
            ModelTier::High => ModelTier::High,
        }
    }
}

/// A single step of a [`PlanProposal`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanStepSpec {
    /// Short name of the step.
    pub name: String,
    /// A longer description of what the step entails.
    #[serde(default)]
    pub description: String,
}

impl PlanStepSpec {
    /// Create a step from a name and description.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
        }
    }
}

/// The goal + ordered steps the orchestrator is driving to completion.
///
/// Analogous to the reference `PlanProposal`. It may be seeded up front, or
/// produced mid-run by the planner mode (see [`crate::mode::SubAgentMode::Plan`]).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PlanProposal {
    /// The overall goal of the task.
    pub goal: String,
    /// The ordered plan steps.
    pub steps: Vec<PlanStepSpec>,
}

impl PlanProposal {
    /// Create a proposal from a goal and its steps.
    pub fn new(goal: impl Into<String>, steps: Vec<PlanStepSpec>) -> Self {
        Self {
            goal: goal.into(),
            steps,
        }
    }

    /// Borrow the step at the given 0-based index, if present.
    pub fn step(&self, index: usize) -> Option<&PlanStepSpec> {
        self.steps.get(index)
    }

    /// Render the proposal as a human-readable, numbered plan (1-based).
    pub fn render(&self) -> String {
        let mut out = format!("Goal: {}\n\nPlan:\n", self.goal);
        for (i, step) in self.steps.iter().enumerate() {
            out.push_str(&format!("{}. {}", i + 1, step.name));
            if !step.description.is_empty() {
                out.push_str(&format!(" — {}", step.description));
            }
            out.push('\n');
        }
        out
    }
}

/// The stored outcome of one `run_subagent` invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepResult {
    /// The mode id that produced this result (e.g. `"code"`, `"review"`).
    pub mode_id: String,
    /// The 0-based plan step index this result belongs to.
    pub step_index: usize,
    /// Whether the sub-agent turn succeeded (for reviewers: a positive verdict).
    pub success: bool,
    /// The assistant-visible output of the sub-agent turn.
    pub output: String,
}

/// The key under which a [`StepResult`] is stored: `(mode_id, step_index)`.
type ResultKey = (String, usize);

/// Shared mutable state threaded through every `run_subagent` call.
#[derive(Default)]
pub struct OrchestratedStepContext {
    proposal: Mutex<Option<PlanProposal>>,
    results: Mutex<HashMap<ResultKey, StepResult>>,
    attempts: Mutex<HashMap<ResultKey, usize>>,
}

impl OrchestratedStepContext {
    /// Create an empty context with no plan proposal yet.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Create a context pre-seeded with a plan proposal.
    pub fn with_proposal(proposal: PlanProposal) -> Arc<Self> {
        Arc::new(Self {
            proposal: Mutex::new(Some(proposal)),
            ..Default::default()
        })
    }

    /// Snapshot the current plan proposal, if any.
    pub fn proposal(&self) -> Option<PlanProposal> {
        self.proposal.lock().expect("proposal lock").clone()
    }

    /// Replace the current plan proposal (used by the planner mode).
    pub fn set_proposal(&self, proposal: PlanProposal) {
        *self.proposal.lock().expect("proposal lock") = Some(proposal);
    }

    /// Store the result of a sub-agent invocation, overwriting any prior result
    /// for the same `(mode_id, step_index)`.
    pub fn store_result(&self, result: StepResult) {
        let key = (result.mode_id.clone(), result.step_index);
        self.results
            .lock()
            .expect("results lock")
            .insert(key, result);
    }

    /// Fetch a previously stored result for `(mode_id, step_index)`.
    pub fn result(&self, mode_id: &str, step_index: usize) -> Option<StepResult> {
        self.results
            .lock()
            .expect("results lock")
            .get(&(mode_id.to_string(), step_index))
            .cloned()
    }

    /// Whether a result exists for `(mode_id, step_index)`.
    pub fn has_result(&self, mode_id: &str, step_index: usize) -> bool {
        self.results
            .lock()
            .expect("results lock")
            .contains_key(&(mode_id.to_string(), step_index))
    }

    /// All stored results belonging to the given `step_index`, across modes.
    ///
    /// Used by reviewer modes to find prior executor output for a step and to
    /// enforce ordering preconditions (a review requires a prior executor pass).
    pub fn results_for_step(&self, step_index: usize) -> Vec<StepResult> {
        self.results
            .lock()
            .expect("results lock")
            .values()
            .filter(|r| r.step_index == step_index)
            .cloned()
            .collect()
    }

    /// Whether any stored result at `step_index` was produced by one of the
    /// given `mode_ids` (e.g. the executor family).
    pub fn has_result_from_any(&self, step_index: usize, mode_ids: &[&str]) -> bool {
        self.results
            .lock()
            .expect("results lock")
            .values()
            .any(|r| r.step_index == step_index && mode_ids.contains(&r.mode_id.as_str()))
    }

    /// Record a new attempt for `(mode_id, step_index)` and return the 1-based
    /// attempt number (first attempt returns `1`).
    pub fn next_attempt(&self, mode_id: &str, step_index: usize) -> usize {
        let mut attempts = self.attempts.lock().expect("attempts lock");
        let entry = attempts
            .entry((mode_id.to_string(), step_index))
            .or_insert(0);
        *entry += 1;
        *entry
    }

    /// The number of attempts already made for `(mode_id, step_index)`.
    pub fn attempts(&self, mode_id: &str, step_index: usize) -> usize {
        self.attempts
            .lock()
            .expect("attempts lock")
            .get(&(mode_id.to_string(), step_index))
            .copied()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_tier_parses_and_escalates() {
        assert_eq!(ModelTier::from_str_opt("LOW"), Some(ModelTier::Low));
        assert_eq!(ModelTier::from_str_opt("default"), Some(ModelTier::Medium));
        assert_eq!(ModelTier::from_str_opt("max"), Some(ModelTier::High));
        assert_eq!(ModelTier::from_str_opt("nope"), None);
        assert_eq!(ModelTier::Low.escalated(), ModelTier::Medium);
        assert_eq!(ModelTier::High.escalated(), ModelTier::High);
    }

    #[test]
    fn context_stores_results_and_attempts() {
        let ctx = OrchestratedStepContext::new();
        assert!(ctx.proposal().is_none());
        assert!(!ctx.has_result("code", 0));

        assert_eq!(ctx.next_attempt("code", 0), 1);
        assert_eq!(ctx.next_attempt("code", 0), 2);
        assert_eq!(ctx.attempts("code", 0), 2);

        ctx.store_result(StepResult {
            mode_id: "code".into(),
            step_index: 0,
            success: true,
            output: "done".into(),
        });
        assert!(ctx.has_result("code", 0));
        assert_eq!(ctx.result("code", 0).unwrap().output, "done");

        ctx.set_proposal(PlanProposal::new(
            "goal",
            vec![PlanStepSpec::new("s1", "do s1")],
        ));
        assert_eq!(ctx.proposal().unwrap().steps.len(), 1);
    }

    #[test]
    fn proposal_renders_numbered() {
        let p = PlanProposal::new(
            "Build X",
            vec![
                PlanStepSpec::new("Setup", "install deps"),
                PlanStepSpec::new("Implement", ""),
            ],
        );
        let text = p.render();
        assert!(text.contains("Goal: Build X"));
        assert!(text.contains("1. Setup — install deps"));
        assert!(text.contains("2. Implement"));
    }
}
