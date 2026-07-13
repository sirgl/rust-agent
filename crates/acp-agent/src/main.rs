//! `acp-agent`: the stdio ACP server binary.
//!
//! Boots `tracing` (to stderr) and serves the agent over stdio using the
//! dependencies assembled by [`acp_agent::default_deps`]. That default prefers
//! the real Anthropic backend whenever an API key can be resolved (either the
//! `ANTHROPIC_API_KEY` environment variable or, failing that, the default
//! `token.properties` key file -- see [`acp_agent::anthropic_factory`]),
//! falling back to the deterministic replay backend when no key is available,
//! so the binary always streams a prompt turn end-to-end even without any
//! network access.

fn main() {
    agent_core::init_tracing();

    let runtime = match tokio::runtime::Runtime::new() {
        Ok(runtime) => runtime,
        Err(err) => {
            tracing::error!(%err, "failed to start tokio runtime");
            std::process::exit(1);
        }
    };

    tracing::info!("acp-agent starting on stdio");
    if let Err(err) = runtime.block_on(acp_agent::run_stdio(acp_agent::default_deps())) {
        tracing::error!(%err, "acp-agent terminated with error");
        std::process::exit(1);
    }
}
