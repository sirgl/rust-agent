//! Binary entry point for the interactive terminal chat.
//!
//! Launch with `cargo run -p cli-agent`. It selects the same decision backend as
//! the ACP server (Anthropic when an API key resolves via env or
//! `token.properties`, otherwise the deterministic replay backend), wires the
//! local filesystem/terminal client, and runs the REPL. `tracing` goes to stderr
//! so logs never corrupt the chat stream on stdout.

use std::io::BufReader;
use std::sync::Arc;

use agent_core::ClientAccess;
use cli_agent::{run_repl, LocalClientAccess, ReplOptions};

#[tokio::main]
async fn main() {
    agent_core::init_tracing();

    let deps = acp_agent::default_deps();
    let client: Arc<dyn ClientAccess> = Arc::new(LocalClientAccess::new());

    // Blocking stdin reads happen on a dedicated blocking thread so they never
    // stall the async runtime; stdout carries the chat stream.
    let input = BufReader::new(std::io::stdin());
    let output = std::io::stdout();

    if let Err(err) = run_repl(deps, client, input, output, ReplOptions::default()).await {
        eprintln!("cli-agent: fatal I/O error: {err}");
        std::process::exit(1);
    }
}
