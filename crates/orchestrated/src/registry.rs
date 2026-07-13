//! The [`ModeRegistry`]: maps a mode id to its [`SubAgentMode`] implementation.
//!
//! The registry is built once (all six default modes wired to their shared
//! family behaviors) and consulted by the `run_subagent` tool to resolve the
//! `mode` argument. Unknown modes fail with a structured [`UnknownModeError`].

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::mode::{
    code_mode, niche_mode, setup_mode, ExecutorBehavior, PlanMode, PlannerBehavior, ReviewMode,
    ReviewPlanMode, ReviewerBehavior, SubAgentMode,
};

/// Returned by [`ModeRegistry::from_str`] when the mode id is not registered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownModeError {
    /// The unrecognized mode id.
    pub mode: String,
    /// The ids that *are* registered, for a helpful message.
    pub known: Vec<String>,
}

impl std::fmt::Display for UnknownModeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown sub-agent mode '{}' (known: {})",
            self.mode,
            self.known.join(", ")
        )
    }
}

impl std::error::Error for UnknownModeError {}

/// A lookup table of `id -> Arc<dyn SubAgentMode>`.
#[derive(Clone)]
pub struct ModeRegistry {
    modes: BTreeMap<String, Arc<dyn SubAgentMode>>,
}

impl ModeRegistry {
    /// Create an empty registry.
    pub fn empty() -> Self {
        Self {
            modes: BTreeMap::new(),
        }
    }

    /// Register a mode under its own [`SubAgentMode::id`].
    pub fn register(&mut self, mode: Arc<dyn SubAgentMode>) {
        self.modes.insert(mode.id().to_string(), mode);
    }

    /// Build the default registry with all six modes wired to shared behaviors.
    pub fn with_defaults() -> Self {
        let executor = Arc::new(ExecutorBehavior);
        let reviewer = Arc::new(ReviewerBehavior);
        let planner = Arc::new(PlannerBehavior);

        let mut r = Self::empty();
        r.register(Arc::new(code_mode(executor.clone())));
        r.register(Arc::new(setup_mode(executor.clone())));
        r.register(Arc::new(niche_mode(executor)));
        r.register(Arc::new(ReviewMode::new(reviewer.clone())));
        r.register(Arc::new(ReviewPlanMode::new(reviewer)));
        r.register(Arc::new(PlanMode::new(planner)));
        r
    }

    /// Look up a mode by id, returning `None` if not registered.
    pub fn get(&self, id: &str) -> Option<Arc<dyn SubAgentMode>> {
        self.modes.get(id).cloned()
    }

    /// Resolve a mode by id, returning a structured error for unknown modes.
    pub fn from_str(&self, mode: &str) -> Result<Arc<dyn SubAgentMode>, UnknownModeError> {
        self.get(mode).ok_or_else(|| UnknownModeError {
            mode: mode.to_string(),
            known: self.ids().map(|s| s.to_string()).collect(),
        })
    }

    /// Iterate over the registered mode ids (sorted).
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.modes.keys().map(|s| s.as_str())
    }

    /// The number of registered modes.
    pub fn len(&self) -> usize {
        self.modes.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.modes.is_empty()
    }
}

impl Default for ModeRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mode::OrchestratorKind;

    #[test]
    fn defaults_register_all_six_modes() {
        let r = ModeRegistry::with_defaults();
        assert_eq!(r.len(), 6);
        for id in ["code", "setup", "niche", "review", "review_plan", "plan"] {
            assert!(r.get(id).is_some(), "missing mode {id}");
        }
    }

    #[test]
    fn from_str_resolves_and_errors() {
        let r = ModeRegistry::with_defaults();
        let code = r.from_str("code").expect("code resolves");
        assert_eq!(code.id(), "code");
        assert_eq!(code.orchestrator_kind(), OrchestratorKind::Executor);

        let err = match r.from_str("nope") {
            Ok(_) => panic!("unknown mode should not resolve"),
            Err(e) => e,
        };
        assert_eq!(err.mode, "nope");
        assert!(err.known.contains(&"code".to_string()));
        assert!(err.to_string().contains("unknown sub-agent mode 'nope'"));
    }

    #[test]
    fn empty_registry_is_empty() {
        let r = ModeRegistry::empty();
        assert!(r.is_empty());
        assert!(r.from_str("code").is_err());
    }
}
