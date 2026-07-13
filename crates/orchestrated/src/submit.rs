//! Submit actions injected into a sub-agent turn.
//!
//! A sub-agent reports its result not through free-form final text but through a
//! typed **submit action**, so success is explicit and robust to phrasing:
//! - executors/planner use [`SubmitTool`] (`submit { output, success? }`),
//! - reviewers use [`SubmitReviewTool`] (`submit_review { approved, reasons }`).
//!
//! Both write a [`SubmitOutcome`] into a per-run [`SubmitSlot`] that the
//! [`crate::worker::OrchestratedSubAgentWorker`] reads once the nested turn
//! finishes. The slot is guarded by a [`Mutex`]; the guard is taken and dropped
//! synchronously inside `call`, never held across an `.await`.

use std::sync::{Arc, Mutex};

use agent_core::{Result, Tool, ToolContext, ToolEvent};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};

pub use crate::mode::SubmitOutcome;

/// A per-run slot the submit tools write into and the worker reads after the turn.
pub type SubmitSlot = Arc<Mutex<Option<SubmitOutcome>>>;

/// Create a fresh, empty submit slot.
pub fn new_slot() -> SubmitSlot {
    Arc::new(Mutex::new(None))
}

/// Wrap a pre-computed sequence of [`ToolEvent`]s as a stream.
fn events(events: Vec<ToolEvent>) -> BoxStream<'static, ToolEvent> {
    Box::pin(stream::iter(events))
}

/// The plain executor/planner submit action: `submit { output, success? }`.
pub struct SubmitTool {
    slot: SubmitSlot,
}

impl SubmitTool {
    /// Create a submit tool writing into `slot`.
    pub fn new(slot: SubmitSlot) -> Self {
        Self { slot }
    }
}

#[async_trait]
impl Tool for SubmitTool {
    fn name(&self) -> &str {
        "submit"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "output": {
                    "type": "string",
                    "description": "A concise summary of the work performed."
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the task was completed successfully (default true)."
                }
            },
            "required": ["output"]
        })
    }

    fn requires_permission(&self) -> bool {
        false
    }

    async fn call(
        &self,
        args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        let output = args
            .get("output")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let success = args.get("success").and_then(|v| v.as_bool()).unwrap_or(true);

        let outcome = SubmitOutcome {
            success,
            output: output.clone(),
            reasons: None,
        };
        // Take and drop the guard synchronously; never held across `.await`.
        {
            *self.slot.lock().expect("submit slot") = Some(outcome);
        }

        Ok(events(vec![
            ToolEvent::Started,
            ToolEvent::Completed(serde_json::json!({ "submitted": true, "output": output })),
        ]))
    }
}

/// The structured reviewer submit action: `submit_review { approved, reasons }`.
pub struct SubmitReviewTool {
    slot: SubmitSlot,
}

impl SubmitReviewTool {
    /// Create a review submit tool writing into `slot`.
    pub fn new(slot: SubmitSlot) -> Self {
        Self { slot }
    }
}

#[async_trait]
impl Tool for SubmitReviewTool {
    fn name(&self) -> &str {
        "submit_review"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "approved": {
                    "type": "boolean",
                    "description": "Whether the reviewed work satisfies the requirements."
                },
                "reasons": {
                    "type": "string",
                    "description": "The reasoning behind the verdict."
                }
            },
            "required": ["approved", "reasons"]
        })
    }

    fn requires_permission(&self) -> bool {
        false
    }

    async fn call(
        &self,
        args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<BoxStream<'static, ToolEvent>> {
        let approved = args.get("approved").and_then(|v| v.as_bool());
        let Some(approved) = approved else {
            return Ok(events(vec![
                ToolEvent::Started,
                ToolEvent::Failed("submit_review requires a boolean `approved` field".into()),
            ]));
        };
        let reasons = args
            .get("reasons")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        let outcome = SubmitOutcome {
            success: approved,
            output: reasons.clone(),
            reasons: Some(reasons.clone()),
        };
        {
            *self.slot.lock().expect("submit slot") = Some(outcome);
        }

        Ok(events(vec![
            ToolEvent::Started,
            ToolEvent::Completed(
                serde_json::json!({ "approved": approved, "reasons": reasons }),
            ),
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::{CancellationToken, ToolEvent};
    use futures::StreamExt;

    fn ctx() -> ToolContext {
        ToolContext::new("s", CancellationToken::new())
    }

    #[tokio::test]
    async fn submit_tool_writes_outcome() {
        let slot = new_slot();
        let tool = SubmitTool::new(slot.clone());
        let stream = tool
            .call(serde_json::json!({ "output": "done work" }), &ctx())
            .await
            .unwrap();
        let evs: Vec<ToolEvent> = stream.collect().await;
        assert!(matches!(evs.last(), Some(ToolEvent::Completed(_))));

        let outcome = slot.lock().unwrap().clone().expect("outcome present");
        assert!(outcome.success);
        assert_eq!(outcome.output, "done work");
        assert!(outcome.reasons.is_none());
    }

    #[tokio::test]
    async fn submit_review_records_verdict() {
        let slot = new_slot();
        let tool = SubmitReviewTool::new(slot.clone());
        tool.call(
            serde_json::json!({ "approved": false, "reasons": "no tests" }),
            &ctx(),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

        let outcome = slot.lock().unwrap().clone().expect("outcome present");
        assert!(!outcome.success);
        assert_eq!(outcome.reasons.as_deref(), Some("no tests"));
    }

    #[tokio::test]
    async fn submit_review_rejects_missing_approved() {
        let slot = new_slot();
        let tool = SubmitReviewTool::new(slot.clone());
        let evs: Vec<ToolEvent> = tool
            .call(serde_json::json!({ "reasons": "x" }), &ctx())
            .await
            .unwrap()
            .collect()
            .await;
        assert!(matches!(evs.last(), Some(ToolEvent::Failed(_))));
        assert!(slot.lock().unwrap().is_none());
    }
}
