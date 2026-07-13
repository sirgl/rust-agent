//! # cli-agent
//!
//! A simple interactive terminal chat front-end over the shared turn machinery
//! ([`agent_core::TurnEngine`], `NextTurnService`, `ToolRegistry`). It reuses the
//! exact same engine as the ACP stdio server (`acp-agent`); only the two seams
//! differ:
//!
//! - [`local_client::LocalClientAccess`] backs the built-in filesystem/terminal
//!   tools with the real machine (`std::fs`, `std::process::Command`), since
//!   there is no editor to serve ACP `fs/*` / `terminal/*`.
//! - [`sink::TerminalUpdateSink`] renders streaming engine output to the
//!   terminal.
//!
//! The interactive REPL loop ([`repl::run_repl`]) ties these together, reusing
//! `acp_agent::default_deps()` for backend/tool selection.

#![warn(missing_docs)]

pub mod local_client;
pub mod repl;
pub mod sink;

pub use local_client::LocalClientAccess;
pub use repl::{run_repl, ReplOptions};
pub use sink::TerminalUpdateSink;
