//! Per-session ownership and cancellation of prompt turns.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use agent_core::CancellationToken;
use tokio::sync::{Mutex as AsyncMutex, Notify};
use tracing::{debug, info};

/// Default time allowed for a retired turn to stop cooperatively.
pub const DEFAULT_RETIREMENT_GRACE: Duration = Duration::from_millis(250);

#[derive(Debug)]
struct Completion {
    done: AtomicBool,
    notify: Notify,
}

impl Completion {
    fn new() -> Self {
        Self {
            done: AtomicBool::new(false),
            notify: Notify::new(),
        }
    }

    fn signal(&self) {
        self.done.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    async fn wait_for(&self, duration: Duration) -> bool {
        tokio::time::timeout(duration, async {
            loop {
                if self.done.load(Ordering::Acquire) {
                    return;
                }
                let notified = self.notify.notified();
                if self.done.load(Ordering::Acquire) {
                    return;
                }
                notified.await;
            }
        })
        .await
        .is_ok()
    }
}

#[derive(Debug)]
struct ActiveTurn {
    generation: u64,
    cancel: CancellationToken,
    completion: Arc<Completion>,
    retired: bool,
}

#[derive(Debug, Default)]
struct CoordinatorState {
    next_generation: u64,
    active: Option<ActiveTurn>,
}

/// Authoritative lifecycle coordinator for all turns in one ACP session.
#[derive(Debug)]
pub struct TurnCoordinator {
    session_id: String,
    retirement_grace: Duration,
    start_gate: AsyncMutex<()>,
    state: StdMutex<CoordinatorState>,
}

impl TurnCoordinator {
    /// Create a coordinator using the default bounded retirement grace period.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Arc<Self> {
        Self::with_retirement_grace(session_id, DEFAULT_RETIREMENT_GRACE)
    }

    /// Create a coordinator with an explicit grace period.
    #[must_use]
    pub fn with_retirement_grace(
        session_id: impl Into<String>,
        retirement_grace: Duration,
    ) -> Arc<Self> {
        Arc::new(Self {
            session_id: session_id.into(),
            retirement_grace,
            start_gate: AsyncMutex::new(()),
            state: StdMutex::new(CoordinatorState::default()),
        })
    }

    /// Retire any current generation, wait briefly, and allocate a new lease.
    pub async fn begin_turn(self: &Arc<Self>) -> TurnLease {
        let _start_guard = self.start_gate.lock().await;
        let prior = {
            let mut state = self.state.lock().expect("turn coordinator mutex poisoned");
            state.active.as_mut().map(|active| {
                active.retired = true;
                active.cancel.cancel();
                (active.generation, active.completion.clone())
            })
        };

        if let Some((generation, completion)) = prior {
            info!(
                session_id = self.session_id,
                generation, "retiring active turn before replacement"
            );
            let completed = completion.wait_for(self.retirement_grace).await;
            debug!(
                session_id = self.session_id,
                generation, completed, "retirement grace finished"
            );
        }

        let (generation, cancel, completion) = {
            let mut state = self.state.lock().expect("turn coordinator mutex poisoned");
            state.next_generation = state.next_generation.saturating_add(1);
            let generation = state.next_generation;
            let cancel = CancellationToken::new();
            let completion = Arc::new(Completion::new());
            state.active = Some(ActiveTurn {
                generation,
                cancel: cancel.clone(),
                completion: completion.clone(),
                retired: false,
            });
            (generation, cancel, completion)
        };
        debug!(
            session_id = self.session_id,
            generation, "started turn generation"
        );
        TurnLease {
            coordinator: self.clone(),
            generation,
            cancel,
            completion,
        }
    }

    /// Cancel the current generation, if any.
    pub fn cancel_active(&self) -> Option<u64> {
        let mut state = self.state.lock().expect("turn coordinator mutex poisoned");
        state.active.as_mut().map(|active| {
            active.retired = true;
            active.cancel.cancel();
            info!(
                session_id = self.session_id,
                generation = active.generation,
                "cancelled active turn"
            );
            active.generation
        })
    }

    /// Whether a non-retired generation currently accepts output and steering.
    #[must_use]
    pub fn has_current_turn(&self) -> bool {
        self.state
            .lock()
            .expect("turn coordinator mutex poisoned")
            .active
            .as_ref()
            .is_some_and(|active| !active.retired)
    }

    /// Run a synchronous action while atomically confirming a current turn.
    ///
    /// Used by steering so completion cannot race between the current-turn
    /// check and insertion into its inbox.
    pub fn with_current_turn(&self, action: impl FnOnce()) -> bool {
        let state = self.state.lock().expect("turn coordinator mutex poisoned");
        if state.active.as_ref().is_some_and(|active| !active.retired) {
            action();
            true
        } else {
            false
        }
    }

    fn is_current(&self, generation: u64) -> bool {
        self.state
            .lock()
            .expect("turn coordinator mutex poisoned")
            .active
            .as_ref()
            .is_some_and(|active| active.generation == generation && !active.retired)
    }

    fn complete(&self, generation: u64, completion: &Completion) -> bool {
        completion.signal();
        let mut state = self.state.lock().expect("turn coordinator mutex poisoned");
        let current = state
            .active
            .as_ref()
            .is_some_and(|active| active.generation == generation);
        if current {
            state.active = None;
        }
        debug!(
            session_id = self.session_id,
            generation,
            cleared_current = current,
            "turn generation completed"
        );
        current
    }
}

/// A generation-scoped capability to run, cancel, emit, and complete one turn.
#[derive(Clone, Debug)]
pub struct TurnLease {
    coordinator: Arc<TurnCoordinator>,
    generation: u64,
    cancel: CancellationToken,
    completion: Arc<Completion>,
}

impl TurnLease {
    /// ACP session identifier used for structured lifecycle diagnostics.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.coordinator.session_id
    }

    /// Monotonic generation identifier within the session.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Cancellation token shared by the engine and its nested work.
    #[must_use]
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Whether this lease still owns client output and lifecycle side effects.
    #[must_use]
    pub fn is_current(&self) -> bool {
        self.coordinator.is_current(self.generation)
    }

    /// Signal task completion and clear the active turn only if still current.
    pub fn complete(&self) -> bool {
        self.coordinator.complete(self.generation, &self.completion)
    }

    /// Run final side effects and complete only if this lease is still current.
    ///
    /// Replacement starts use the same gate, so persistence cannot pass the
    /// check and then race with a newer generation's persistence.
    pub async fn finalize_if_current<F, Fut>(&self, action: F) -> bool
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = ()>,
    {
        let _start_guard = self.coordinator.start_gate.lock().await;
        if !self.is_current() {
            self.completion.signal();
            debug!(
                session_id = self.coordinator.session_id,
                generation = self.generation,
                "skipped final side effects for retired turn"
            );
            return false;
        }
        action().await;
        self.complete()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn generations_are_monotonic_and_replacement_cancels_old_lease() {
        let coordinator = TurnCoordinator::with_retirement_grace("s", Duration::ZERO);
        let first = coordinator.begin_turn().await;
        let second = coordinator.begin_turn().await;

        assert_eq!(first.generation(), 1);
        assert_eq!(second.generation(), 2);
        assert!(first.cancellation_token().is_cancelled());
        assert!(!first.is_current());
        assert!(second.is_current());
    }

    #[tokio::test]
    async fn completion_signal_cannot_be_missed() {
        let coordinator = TurnCoordinator::with_retirement_grace("s", Duration::from_secs(1));
        let first = coordinator.begin_turn().await;
        first.complete();

        let second = tokio::time::timeout(Duration::from_millis(50), coordinator.begin_turn())
            .await
            .expect("completed generation must not consume the grace period");
        assert_eq!(second.generation(), 2);
    }

    #[tokio::test]
    async fn stale_completion_cannot_clear_newer_turn() {
        let coordinator = TurnCoordinator::with_retirement_grace("s", Duration::ZERO);
        let first = coordinator.begin_turn().await;
        let second = coordinator.begin_turn().await;

        assert!(!first.complete());
        assert!(second.is_current());
        assert!(coordinator.has_current_turn());
        assert!(second.complete());
        assert!(!coordinator.has_current_turn());
    }

    #[tokio::test]
    async fn replacement_wait_is_bounded_for_ignored_cancellation() {
        let grace = Duration::from_millis(20);
        let coordinator = TurnCoordinator::with_retirement_grace("s", grace);
        let _first = coordinator.begin_turn().await;

        let started = tokio::time::Instant::now();
        let second = coordinator.begin_turn().await;
        assert!(started.elapsed() >= grace);
        assert!(second.is_current());
    }

    #[tokio::test]
    async fn cancel_targets_only_the_active_generation() {
        let coordinator = TurnCoordinator::with_retirement_grace("s", Duration::ZERO);
        assert_eq!(coordinator.cancel_active(), None);
        let lease = coordinator.begin_turn().await;
        assert_eq!(coordinator.cancel_active(), Some(lease.generation()));
        assert!(lease.cancellation_token().is_cancelled());
        assert!(!lease.is_current());
    }

    #[tokio::test]
    async fn stale_finalization_skips_side_effects() {
        let coordinator = TurnCoordinator::with_retirement_grace("s", Duration::ZERO);
        let first = coordinator.begin_turn().await;
        let _second = coordinator.begin_turn().await;
        let called = Arc::new(AtomicBool::new(false));
        let marker = called.clone();

        assert!(
            !first
                .finalize_if_current(|| async move {
                    marker.store(true, Ordering::SeqCst);
                })
                .await
        );
        assert!(!called.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn current_finalization_runs_once_and_clears_generation() {
        let coordinator = TurnCoordinator::with_retirement_grace("s", Duration::ZERO);
        let lease = coordinator.begin_turn().await;
        let called = Arc::new(AtomicBool::new(false));
        let marker = called.clone();

        assert!(
            lease
                .finalize_if_current(|| async move {
                    marker.store(true, Ordering::SeqCst);
                })
                .await
        );
        assert!(called.load(Ordering::SeqCst));
        assert!(!coordinator.has_current_turn());
    }

    #[tokio::test]
    async fn steering_action_is_conditioned_on_current_generation() {
        let coordinator = TurnCoordinator::with_retirement_grace("s", Duration::ZERO);
        let called = Arc::new(AtomicBool::new(false));
        let idle_marker = called.clone();
        assert!(!coordinator.with_current_turn(|| {
            idle_marker.store(true, Ordering::SeqCst);
        }));
        assert!(!called.load(Ordering::SeqCst));

        let _lease = coordinator.begin_turn().await;
        let active_marker = called.clone();
        assert!(coordinator.with_current_turn(|| {
            active_marker.store(true, Ordering::SeqCst);
        }));
        assert!(called.load(Ordering::SeqCst));
    }
}
