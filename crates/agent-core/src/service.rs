//! The decision layer abstraction: [`NextTurnService`].

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::Result;
use crate::event::TurnEvent;
use crate::session::TurnContext;

/// Decides *what the agent does next* as a stream of [`TurnEvent`]s.
///
/// Concrete implementations are interchangeable: a real Anthropic backend, a
/// deterministic replay backend for tests, or scripted backends. Keeping this
/// seam provider-agnostic is what makes the turn engine fully testable.
#[async_trait]
pub trait NextTurnService: Send + Sync {
    /// Produce the next turn as a stream of decision events.
    async fn get_next_turn_streaming(
        &self,
        ctx: &TurnContext,
    ) -> Result<BoxStream<'static, TurnEvent>>;
}
