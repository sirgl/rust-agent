//! A `tracing` [`Layer`] that mirrors agent log events into the [`Inspector`].
//!
//! Attaching this layer to the tracing subscriber lets the debug server show
//! not only the requests/responses exchanged with the model, but also the
//! agent's own log lines (what it is doing internally) — grouped by session
//! when an event carries a `session_id` field.
//!
//! The layer is deliberately cheap and panic-free: it visits an event's fields
//! into a small map, extracts the `message` and any `session_id`, and forwards
//! the rest as structured JSON. It never emits tracing events itself (and skips
//! the inspector's own target) so it can never recurse.

use serde_json::{Map, Value};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

use agent_core::{AgentRunMetadata, TraceContext};

use crate::store::Inspector;

/// The tracing target prefix of this crate, skipped to avoid self-capture.
const SELF_TARGET_PREFIX: &str = "llm_inspector";

/// A [`Layer`] forwarding log events to an [`Inspector`] as `Log` records.
pub struct InspectorLayer {
    inspector: Inspector,
}

impl InspectorLayer {
    /// Create a layer that forwards events into `inspector`.
    #[must_use]
    pub fn new(inspector: Inspector) -> Self {
        Self { inspector }
    }
}

#[derive(Clone)]
struct SpanFields {
    session_id: Option<String>,
    fields: Map<String, Value>,
}

impl<S> Layer<S> for InspectorLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut visitor = LogVisitor::default();
        attrs.record(&mut visitor);
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(SpanFields {
                session_id: visitor.session_id,
                fields: visitor.fields,
            });
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let metadata = event.metadata();
        let target = metadata.target();
        // Never capture our own logs, or we could recurse via a recorder.
        if target.starts_with(SELF_TARGET_PREFIX) {
            return;
        }

        let mut inherited = Map::new();
        let mut inherited_session = None;
        if let Some(scope) = ctx.event_scope(event) {
            for span in scope.from_root() {
                if let Some(fields) = span.extensions().get::<SpanFields>() {
                    if fields.session_id.is_some() {
                        inherited_session.clone_from(&fields.session_id);
                    }
                    inherited.extend(fields.fields.clone());
                }
            }
        }
        let mut visitor = LogVisitor::default();
        event.record(&mut visitor);

        let session_id = visitor.session_id.take().or(inherited_session);
        let message = visitor.message.take().unwrap_or_default();
        inherited.extend(visitor.fields);
        let trace = trace_from_fields(&inherited);
        let fields = Value::Object(inherited);

        self.inspector.record_log(
            session_id.as_deref(),
            trace,
            metadata.level().as_str(),
            target,
            message,
            fields,
        );
    }
}

fn trace_from_fields(fields: &Map<String, Value>) -> Option<TraceContext> {
    let string = |name: &str| {
        fields
            .get(name)
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
    };
    Some(TraceContext {
        turn_id: string("turn_id")?.to_string(),
        agent_run_id: string("agent_run_id")?.to_string(),
        parent_agent_run_id: string("parent_agent_run_id").map(str::to_owned),
        spawned_by_tool_call_id: string("spawned_by_tool_call_id").map(str::to_owned),
        round_id: string("round_id").map(str::to_owned),
        agent_path: string("agent_path").unwrap_or("root").to_string(),
        metadata: AgentRunMetadata::default(),
    })
}

/// Collects an event's fields, pulling out `message` and `session_id`.
#[derive(Default)]
struct LogVisitor {
    message: Option<String>,
    session_id: Option<String>,
    fields: Map<String, Value>,
}

impl LogVisitor {
    /// Route a captured field into the message, session id, or the field map.
    fn insert(&mut self, name: &str, value: Value) {
        match name {
            "message" => self.message = value.as_str().map(str::to_owned),
            // Accept a few common spellings for the session attribution field.
            "session_id" | "session" => {
                self.session_id = value.as_str().map(str::to_owned);
            }
            _ => {
                self.fields.insert(name.to_string(), value);
            }
        }
    }
}

impl Visit for LogVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.insert(field.name(), Value::String(format!("{value:?}")));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.insert(field.name(), Value::String(value.to_string()));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.insert(field.name(), Value::from(value));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.insert(field.name(), Value::from(value));
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.insert(field.name(), Value::from(value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::subscriber;
    use tracing_subscriber::filter::LevelFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::Registry;

    #[test]
    fn captures_event_with_session_and_fields() {
        let inspector = Inspector::new();
        // The `LevelFilter` layer raises the subscriber's max-level hint so the
        // `info!` event below is actually dispatched under `with_default`.
        let subscriber = Registry::default()
            .with(LevelFilter::TRACE)
            .with(InspectorLayer::new(inspector.clone()));

        subscriber::with_default(subscriber, || {
            // Use a non-`llm_inspector` target so the self-capture guard (which
            // skips this crate's own logs) does not drop the event.
            tracing::info!(target: "acp_agent::agent", session_id = "sess-9", step = 3, "doing work");
        });

        let snapshot = inspector.snapshot();
        assert_eq!(snapshot.len(), 1);
        let record = &snapshot[0];
        assert_eq!(record.session_id, "sess-9");
        assert_eq!(record.kind(), "log");
        match &record.payload {
            crate::record::CapturedPayload::Log {
                level,
                message,
                fields,
                ..
            } => {
                assert_eq!(level, "INFO");
                assert_eq!(message, "doing work");
                assert_eq!(fields.get("step"), Some(&Value::from(3)));
            }
            other => panic!("expected a log payload, got {other:?}"),
        }
    }

    #[test]
    fn inherits_trace_identity_from_nested_spans() {
        let inspector = Inspector::new();
        let subscriber = Registry::default()
            .with(LevelFilter::TRACE)
            .with(InspectorLayer::new(inspector.clone()));

        subscriber::with_default(subscriber, || {
            let run = tracing::info_span!(
                "agent_run",
                session_id = "sess-1",
                turn_id = "turn-1",
                agent_run_id = "agent-2",
                parent_agent_run_id = "agent-1",
                spawned_by_tool_call_id = "call-7",
                agent_path = "root › code#1",
            );
            let _run = run.enter();
            let round = tracing::info_span!("round", round_id = "round-3");
            let _round = round.enter();
            tracing::debug!(target: "agent_core::engine", "streaming response");
        });

        let snapshot = inspector.snapshot();
        let trace = snapshot[0].trace.as_ref().expect("span trace");
        assert_eq!(trace.turn_id, "turn-1");
        assert_eq!(trace.agent_run_id, "agent-2");
        assert_eq!(trace.parent_agent_run_id.as_deref(), Some("agent-1"));
        assert_eq!(trace.spawned_by_tool_call_id.as_deref(), Some("call-7"));
        assert_eq!(trace.round_id.as_deref(), Some("round-3"));
        assert_eq!(trace.agent_path, "root › code#1");
    }
}
