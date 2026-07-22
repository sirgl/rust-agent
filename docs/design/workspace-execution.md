# Session workspace and local execution

## Canonical workspace state

`SessionState.workspace_roots` is the only workspace source of truth. Index
zero is the session working directory; later entries are additional roots in
the order supplied by the ACP client. `SessionState::set_workspace` maintains
that invariant and `working_directory` reads the primary root without creating
a duplicate cwd field.

`session/new` initializes this list from `NewSessionRequest.cwd` and
`additional_directories`. `session/load` treats the current load request as
authoritative and replaces stale persisted roots before publishing Inspector
state and persisting the refreshed record. A missing or non-directory cwd is
rejected with an error that includes the requested path.

## Model context

The ordered roots travel through `TurnContext` and `ToolContext`.
`DefaultCompiler` appends a stable `<workspace_environment>` section to the
system instructions with the working directory, additional roots, and
relative-path semantics. The section is unchanged between rounds in a session,
so existing history prefixes and provider prompt-cache reuse are preserved.

Ordinary subagents and orchestrated workers copy the same roots into their
nested `SessionState`. Root orchestrators also receive the parent's
`ClientAccess`, allowing all descendants to use the same execution boundary.

## Filesystem semantics

Each local `ClientAccess` captures an immutable, validated `PathBuf`; it never
calls process-global `set_current_dir`. Relative read and write paths are joined
to that cwd. Absolute paths pass through unchanged. Paths containing `..` are
allowed: workspace handling is execution context, not a security sandbox.

Large tool-result offload uses the same write boundary, so relative offload
paths follow the session cwd too.

## Terminal semantics

`terminal_run.command` is one complete shell command. Flags, quoting, pipes,
redirects, and command chaining are supported. On Unix it runs through
`/bin/sh -lc`; on Windows it runs through `cmd.exe /C`. Blocking and streaming
child processes both set `current_dir` to the captured session cwd.

`ClientAccess::run_terminal(command, args)` remains as a low-level direct-spawn
compatibility seam. Legacy typed `terminal_run` history containing `args` is
still readable and executes through that direct seam, while the current tool
schema advertises only the full shell-command contract.

## Frontends, isolation, and diagnostics

ACP prompt, `_session/steering`, chat, orchestrate, and nested-agent paths all use
a client created from the session's primary root. The CLI uses its launch
directory for both model context and local tools. Separate sessions capture
separate cwd values, so concurrent commands and relative filesystem operations
cannot cross through process-global state.

Structured local filesystem and terminal logs include cwd (and ACP lifecycle
logs include session id) but never include captured command output. The LLM
Inspector Session State panel labels root zero as `working directory` and shows
the remainder as `additional roots`.