//! Parity test: the `orchestrate` tool and a direct `run_pipeline` call, given
//! identical orchestrator and sub-agent scripts, produce identical
//! assistant-visible output — since both funnel through the same
//! `run_pipeline` helper.

use std::sync::{Arc, Mutex};

use agent_core::{
    CancellationToken, EngineOutput, NextTurnService, Result, StopReason, Tool, ToolContext,
    ToolEvent, ToolRegistry, TurnEvent, UpdateSink,
};
use async_trait::async_trait;
use futures::StreamExt;
use orchestrated::{run_pipeline, uniform_resolver, BackendFactory, OrchestrateTool};
use turn_replay::ReplayTurnService;

fn orchestrator_script() -> Vec<Vec<TurnEvent>> {
    vec![
        vec![
            TurnEvent::ToolCallRequested {
                id: agent_core::ToolCallId::new("o1"),
                name: "run_subagent".into(),
                arguments: serde_json::json!({ "mode": "code", "step_index": 1 }),
            },
            TurnEvent::TurnFinished {
                stop_reason: StopReason::ToolUse,
            },
        ],
        vec![
            TurnEvent::TextDelta("orchestration complete".into()),
            TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            },
        ],
    ]
}

fn subagent_script() -> Vec<Vec<TurnEvent>> {
    vec![
        vec![
            TurnEvent::ToolCallRequested {
                id: agent_core::ToolCallId::new("s1"),
                name: "submit".into(),
                arguments: serde_json::json!({ "output": "implemented goal" }),
            },
            TurnEvent::TurnFinished {
                stop_reason: StopReason::ToolUse,
            },
        ],
        vec![TurnEvent::TurnFinished {
            stop_reason: StopReason::EndTurn,
        }],
    ]
}

fn subagent_resolver() -> Arc<dyn orchestrated::ModelResolver> {
    uniform_resolver(Arc::new(|| {
        Arc::new(ReplayTurnService::new(subagent_script())) as Arc<dyn NextTurnService>
    }))
}

#[derive(Default, Clone)]
struct CapturingSink {
    text: Arc<Mutex<String>>,
}

#[async_trait]
impl UpdateSink for CapturingSink {
    async fn send(&mut self, output: EngineOutput) -> Result<()> {
        if let EngineOutput::MessageChunk(chunk) = output {
            self.text.lock().unwrap().push_str(&chunk);
        }
        Ok(())
    }
}

#[tokio::test]
async fn tool_and_direct_pipeline_produce_identical_output() {
    // Direct run_pipeline path.
    let mut direct_sink = CapturingSink::default();
    let cancel = CancellationToken::new();
    let direct_stop = run_pipeline(
        "Build a thing",
        Arc::new(ReplayTurnService::new(orchestrator_script())),
        subagent_resolver(),
        ToolRegistry::new(),
        &mut direct_sink,
        &cancel,
    )
    .await
    .unwrap();
    let direct_text = direct_sink.text.lock().unwrap().clone();

    // OrchestrateTool path (uses run_pipeline internally, captures the same text).
    let factory: BackendFactory = Arc::new(|| {
        Arc::new(ReplayTurnService::new(orchestrator_script())) as Arc<dyn NextTurnService>
    });
    let tool = OrchestrateTool::new(factory, subagent_resolver(), ToolRegistry::new());
    let ctx = ToolContext::new("sess-main", CancellationToken::new());
    let evts: Vec<ToolEvent> = tool
        .call(serde_json::json!({ "goal": "Build a thing" }), &ctx)
        .await
        .unwrap()
        .collect()
        .await;

    let (tool_text, tool_stop) = match evts.last() {
        Some(ToolEvent::Completed(v)) => (
            v["output"].as_str().unwrap_or("").to_string(),
            v["stop_reason"].as_str().unwrap_or("").to_string(),
        ),
        other => panic!("unexpected final event: {other:?}"),
    };

    assert_eq!(direct_text, tool_text);
    assert_eq!(direct_stop, StopReason::EndTurn);
    assert_eq!(tool_stop, "end_turn");
    assert_eq!(direct_text, "orchestration complete");
}
