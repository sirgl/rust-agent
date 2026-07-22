# Todo planning

## Purpose

The agent maintains one canonical, typed todo list per session. It is a durable
state model for non-trivial work, not provider prompt decoration: Anthropic,
replay, future decision backends, ordinary agents, and orchestrated agents all
use the same tools and engine transitions.

## Ownership and transition flow

`SessionState::todo_list` is the only source of truth. A tool receives an
immutable clone in `ToolContext`; it never owns a parallel mutable store. The
tool validates a proposed replacement and emits `ToolEvent::TodoListUpdated`.
`TurnEngine`, while holding the exclusive mutable session reference, validates
the event again, replaces the state, and emits one full `EngineOutput::Plan`.
No mutex guard is held across an await.

The two actions are:

- `update_todo_list { items: [{ id, content, status }] }` replaces the complete
  ordered list atomically. Omission removes an old item; `items: []` intentionally
  clears the plan.
- `mark_todo_completed { id }` changes exactly one stable ID to `completed`.
  It preserves order, content, and every other status. Repeating it is
  idempotent; an unknown ID fails without mutation.

IDs and content must be non-empty after trimming, IDs must be unique, and status
is `pending`, `in_progress`, or `completed`. Validation happens during argument
deserialization and at the engine state boundary.

## ACP projection and typed history

Every successful transition projects the whole list to ACP `Plan`; stable IDs
are intentionally dropped only because ACP `PlanEntry` has no ID field. Full
replacement avoids client-side delta reconciliation. A successful empty
replacement still emits an empty plan so the editor clears stale entries.

Tool calls and full structured results remain in typed history through dedicated
`KnownTool` variants. The exact IDs and statuses are therefore available to the
next decision round without injecting a duplicate todo snapshot into every
request. This preserves stable prompt prefixes and provider token-cache reuse.
Provider-originated `TurnEvent::Plan` remains presentation-only and does not
change canonical state.

## Persistence and inspection

`SessionRecord` serializes the list with a backward-compatible empty default.
`session/load` restores it and immediately re-emits a non-empty ACP plan.
`SessionStateSnapshot` carries it to the LLM Inspector, whose Session State panel
and `/api/state` endpoint expose each item's ID, content, and status.

The periodic thought nudge asks the agent to compare the list with observed and
verified reality, repair stale or incomplete state with the todo tools, keep the
visible update to one or two lines, and call `submit_result` immediately when
work is complete. The default system prompt similarly requires a maintained
todo list for non-trivial work and verification before completion.

## Failure semantics

Malformed status, empty fields, duplicate IDs, and unknown completion IDs produce
a failed tool result. They do not mutate `SessionState` and do not emit ACP plan
updates. Persistence rejects invalid serialized todo state rather than loading a
partially trusted list.