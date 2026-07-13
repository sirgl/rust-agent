//! Gated live integration test against the real Anthropic Messages API.
//!
//! Ignored by default so CI and offline runs stay hermetic. Run manually with a
//! valid key:
//!
//! ```bash
//! ANTHROPIC_API_KEY=sk-... cargo test -p turn-anthropic --test live -- --ignored --nocapture
//! ```

use agent_core::{NextTurnService, SessionState, TurnEvent};
use futures::StreamExt;
use turn_anthropic::AnthropicTurnService;

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and network access"]
async fn live_prompt_streams_text() {
    let service = AnthropicTurnService::from_env()
        .expect("ANTHROPIC_API_KEY must be set for the live test");

    let mut session = SessionState::new("live-session");
    session.system_prompt = Some("You are a terse assistant. Reply with a single word.".into());
    session.push_user_text("Say the word 'pong' and nothing else.");
    let ctx = session.turn_context();

    let mut stream = service
        .get_next_turn_streaming(&ctx)
        .await
        .expect("request should be issued");

    let mut text = String::new();
    let mut finished = false;
    while let Some(event) = stream.next().await {
        match event {
            TurnEvent::TextDelta(t) => text.push_str(&t),
            TurnEvent::Thinking(_) => {}
            TurnEvent::TurnFinished { stop_reason } => {
                eprintln!("finished: {stop_reason:?}");
                finished = true;
            }
            TurnEvent::Error(e) => panic!("stream error: {e}"),
            other => eprintln!("event: {other:?}"),
        }
    }

    assert!(finished, "stream must end with TurnFinished");
    assert!(!text.trim().is_empty(), "expected some assistant text");
    eprintln!("assistant said: {text:?}");
}
