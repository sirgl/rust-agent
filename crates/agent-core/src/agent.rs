//! Agent identity within a session.
//!
//! A single user session can involve more than one agent: the top-level chat or
//! orchestrator engine, plus any number of nested sub-agents spawned via tools
//! (`subagents::SubagentTool`, the `orchestrated` worker, `run_pipeline`). All of
//! them issue their own decision-layer requests and produce their own responses.
//!
//! [`AgentPath`] gives each such agent run a stable, hierarchical identity so an
//! observer (e.g. the LLM inspector) can attribute every captured request,
//! response, and log line to the exact agent that produced it — and so the UI
//! can show each agent's output separately, grouped as a tree under its session.
//!
//! The root engine of a session has the path `root`; a sub-agent it spawns gets
//! `root › <label>`, and so on. The `depth` of a path (number of segments minus
//! one) matches the engine's subagent nesting depth.

use std::fmt;

use serde::Serialize;

/// The label used for the top-level agent of a session.
pub const ROOT_LABEL: &str = "root";

/// A hierarchical identity for one agent run within a session.
///
/// Cheap to clone. Segments read outermost-first: `["root", "orchestrator",
/// "code#1"]` means a `code#1` sub-agent spawned by an `orchestrator` sub-agent
/// spawned by the `root` engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AgentPath {
    segments: Vec<String>,
}

impl AgentPath {
    /// The root agent path (`root`), used by a session's top-level engine.
    #[must_use]
    pub fn root() -> Self {
        Self {
            segments: vec![ROOT_LABEL.to_string()],
        }
    }

    /// Build a path from explicit segments, falling back to [`AgentPath::root`]
    /// when empty so a path always has at least one segment.
    #[must_use]
    pub fn from_segments(segments: Vec<String>) -> Self {
        if segments.is_empty() {
            Self::root()
        } else {
            Self { segments }
        }
    }

    /// Derive a child path by appending `label` (e.g. a sub-agent's mode or the
    /// delegating tool's name).
    #[must_use]
    pub fn child(&self, label: impl Into<String>) -> Self {
        let mut segments = self.segments.clone();
        segments.push(label.into());
        Self { segments }
    }

    /// The last segment — this agent's own label.
    #[must_use]
    pub fn label(&self) -> &str {
        self.segments
            .last()
            .map_or(ROOT_LABEL, std::string::String::as_str)
    }

    /// The nesting depth (`0` for the root agent).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.segments.len().saturating_sub(1)
    }

    /// The ordered segments, outermost first.
    #[must_use]
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// A stable, machine-friendly key (segments joined with `/`), suitable for
    /// grouping records by agent.
    #[must_use]
    pub fn key(&self) -> String {
        self.segments.join("/")
    }
}

impl Default for AgentPath {
    fn default() -> Self {
        Self::root()
    }
}

impl fmt::Display for AgentPath {
    /// Human-friendly rendering with a ` › ` separator.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.segments.join(" › "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_has_depth_zero_and_root_label() {
        let root = AgentPath::root();
        assert_eq!(root.depth(), 0);
        assert_eq!(root.label(), ROOT_LABEL);
        assert_eq!(root.key(), "root");
        assert_eq!(root.to_string(), "root");
    }

    #[test]
    fn child_extends_path_and_depth() {
        let child = AgentPath::root().child("orchestrator").child("code#1");
        assert_eq!(child.depth(), 2);
        assert_eq!(child.label(), "code#1");
        assert_eq!(child.key(), "root/orchestrator/code#1");
        assert_eq!(child.to_string(), "root › orchestrator › code#1");
    }

    #[test]
    fn from_empty_segments_falls_back_to_root() {
        assert_eq!(AgentPath::from_segments(vec![]), AgentPath::root());
    }
}
