//! `turn-replay`: a deterministic [`agent_core::NextTurnService`] that replays a
//! scripted sequence of [`TurnEvent`] streams.
//!
//! A real backend (e.g. `turn-anthropic`) may call
//! [`NextTurnService::get_next_turn_streaming`] more than once per prompt: each
//! call produces one decision "round" (a stream of events terminated by
//! [`TurnEvent::TurnFinished`]). `ReplayTurnService` mirrors this by holding an
//! ordered queue of scripted rounds and handing out the next one on every call,
//! which makes the full [`agent_core::TurnEngine`] loop -- including
//! multi-round tool-call turns -- reproducible without any network access.

#![warn(missing_docs)]

use std::collections::VecDeque;
use std::sync::Mutex;

use agent_core::{AgentError, Result, TurnContext, TurnError, TurnEvent};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};

/// A deterministic [`agent_core::NextTurnService`] that replays scripted turn
/// events.
///
/// Construct it with one or more "rounds" (each a `Vec<TurnEvent>` representing
/// a single call to [`get_next_turn_streaming`](agent_core::NextTurnService::get_next_turn_streaming)).
/// Rounds are handed out in order, once each; calling past the end of the
/// script is a programming/test error and yields a [`TurnError`].
#[derive(Debug, Default)]
pub struct ReplayTurnService {
    rounds: Mutex<VecDeque<Vec<TurnEvent>>>,
}

impl ReplayTurnService {
    /// Create a replay service from an explicit sequence of rounds.
    ///
    /// Each round is played back as a single stream by one call to
    /// `get_next_turn_streaming`; scripting more than one round lets tests
    /// exercise the multi-round tool-call loop deterministically.
    pub fn new(rounds: Vec<Vec<TurnEvent>>) -> Self {
        Self {
            rounds: Mutex::new(rounds.into()),
        }
    }

    /// Convenience constructor for a script consisting of a single round.
    pub fn single(events: Vec<TurnEvent>) -> Self {
        Self::new(vec![events])
    }
}

#[async_trait]
impl agent_core::NextTurnService for ReplayTurnService {
    async fn get_next_turn_streaming(
        &self,
        _ctx: &TurnContext,
    ) -> Result<BoxStream<'static, TurnEvent>> {
        let mut rounds = self
            .rounds
            .lock()
            .expect("ReplayTurnService mutex poisoned");
        let round = rounds.pop_front().ok_or_else(|| {
            AgentError::Turn(TurnError::new(
                "turn-replay: get_next_turn_streaming called with no scripted rounds remaining",
            ))
        })?;
        Ok(Box::pin(stream::iter(round)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::{NextTurnService, SessionState, StopReason};
    use futures::StreamExt;

    #[tokio::test]
    async fn replays_a_single_scripted_round_in_order() {
        let svc = ReplayTurnService::single(vec![
            TurnEvent::TextDelta("hi".into()),
            TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            },
        ]);
        let session = SessionState::new("s1");
        let ctx = session.turn_context();
        let mut stream = svc.get_next_turn_streaming(&ctx).await.unwrap();

        match stream.next().await.unwrap() {
            TurnEvent::TextDelta(t) => assert_eq!(t, "hi"),
            other => panic!("unexpected event: {other:?}"),
        }
        match stream.next().await.unwrap() {
            TurnEvent::TurnFinished { stop_reason } => assert_eq!(stop_reason, StopReason::EndTurn),
            other => panic!("unexpected event: {other:?}"),
        }
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn hands_out_rounds_in_order_and_errors_once_exhausted() {
        let svc = ReplayTurnService::new(vec![
            vec![TurnEvent::TurnFinished {
                stop_reason: StopReason::ToolUse,
            }],
            vec![TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn,
            }],
        ]);
        let session = SessionState::new("s1");
        let ctx = session.turn_context();

        let mut first = svc.get_next_turn_streaming(&ctx).await.unwrap();
        assert!(matches!(
            first.next().await.unwrap(),
            TurnEvent::TurnFinished {
                stop_reason: StopReason::ToolUse
            }
        ));

        let mut second = svc.get_next_turn_streaming(&ctx).await.unwrap();
        assert!(matches!(
            second.next().await.unwrap(),
            TurnEvent::TurnFinished {
                stop_reason: StopReason::EndTurn
            }
        ));

        assert!(svc.get_next_turn_streaming(&ctx).await.is_err());
    }
}
