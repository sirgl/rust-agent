# ACP Turn Lifecycle and Message Boundaries

## Problem

ACP `SessionUpdate::AgentMessageChunk` carries content but no assistant-message
identifier. The client derives message boundaries from the lifetime of
`session/prompt`. If an obsolete prompt sends a late chunk after a replacement
prompt starts, the client can append that chunk to an assistant block on the
wrong side of the latest user message. Synthetic separators cannot fix this
reliably because the protocol has no operation that explicitly starts a new
assistant message.

The debug bundle `acp-debug-663d8145-20260716-102913.zip` demonstrated this
race: the first assistant block started before a client cancellation timeout, a
new user prompt started, and late assistant deltas were still appended to the
old block. The transport projection was reflecting overlapping server turns;
it was not inventing the overlap during rebase.

## Coordinator invariants

Every ACP session owns one `TurnCoordinator` and at most one current
`TurnLease`:

- generation IDs increase monotonically within the session;
- starting a prompt retires and cancels the prior generation before any output
  belonging to the replacement is emitted;
- replacement waits only for a short bounded cooperative-shutdown grace period;
- a generation fence remains authoritative when the grace period expires;
- `session/cancel` retires the active generation immediately, so it can no
  longer emit updates or request permission;
- completion clears the active turn only when the completing generation is
  still current;
- session-state publication and persistence run under the same generation gate
  as replacement starts, so an obsolete task cannot overwrite newer state;
- typed history remains serialized by the existing exclusive `SessionState`
  lock. The replacement compiles its user message only after the prior mutable
  state borrow has ended.

## Output fence

`GenerationScopedSink<AcpUpdateSink>` checks its lease before forwarding every
provider-independent output kind:

- assistant message chunks;
- thinking chunks;
- complete todo/plan projections;
- tool calls and tool-call updates;
- permission requests.

Stale outputs return success to their obsolete producer but are not sent to the
ACP client. A stale permission request is denied locally. Structured tracing
records only the session ID, generation, and event kind; it never logs the
dropped content.

## Prompt, cancellation, and steering

`session/prompt` acquires a lease before the deferred Inspector greeting,
slash-command response, chat engine, or orchestration engine can emit output.
Chat and orchestrated execution use the same scoped sink and cancellation
token.

`_session/steering` atomically checks whether the current generation accepts
steering. When it does, the message is inserted into the shared inbox while the
coordinator state is stable. Otherwise injection starts a fresh coordinated
turn in the session's selected mode. Long-lived orchestration drains the same
inbox as ordinary chat between decision rounds.

`session/load` remains outside prompt-turn coordination: it reconstructs an
idle session and replays persisted history and the canonical todo plan before a
new turn exists.

## Message-boundary guarantee

The server does not manufacture message IDs or textual separators. Its client-
visible guarantee is instead:

> Once a replacement prompt has retired generation N, generation N cannot emit
> any ACP update or permission request. Only generation N+1 can contribute
> assistant content to the replacement prompt lifecycle.

This is the strongest boundary the ACP v1 schema can express and aligns server
output ownership with the boundary the client already derives from prompt
completion.

## Validation

Focused coordinator tests cover monotonic generation allocation, cancellation,
bounded retirement, race-safe completion notification, atomic steering, and
compare-and-clear cleanup. Sink tests cover every `EngineOutput` variant and
permission requests for both current and stale leases.

The ACP integration regression reproduces the bundle ordering with a
controllable backend: turn A emits one chunk and blocks, turn B retires it, and
turn A's producer becomes ready with late text, plan, and a destructive tool
call. The test asserts that only A's pre-retirement chunk and B's answer are
visible, no stale permission request is sent, B completes normally, and only
the newest Inspector snapshot and persisted `SessionRecord` win.