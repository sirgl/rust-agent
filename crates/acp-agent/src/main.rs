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
    // Build the inspector (if enabled) *before* initializing tracing, so its
    // log layer can mirror the agent's own log events into the debug server
    // from the very first line. Constructing the inspector needs no runtime;
    // only `serve` (below) does.
    let config = acp_agent::AgentConfig::from_env();
    let inspector = if config.inspector_enabled {
        let mut inspector = llm_inspector::Inspector::new();
        if let Some(dir) = &config.inspector_log_dir {
            inspector = inspector.with_file_logging(dir);
        }
        Some(inspector)
    } else {
        None
    };

    // Route tracing to stderr, additionally mirroring log events into the
    // inspector when it is enabled.
    match &inspector {
        Some(inspector) => llm_inspector::init_tracing_with_inspector(inspector.clone()),
        None => agent_core::init_tracing(),
    }

    if let Some(dir) = config
        .inspector_log_dir
        .as_ref()
        .filter(|_| inspector.is_some())
    {
        tracing::info!(dir = %dir.display(), "acp-agent: logging LLM requests to disk");
    }

    let runtime = match tokio::runtime::Runtime::new() {
        Ok(runtime) => runtime,
        Err(err) => {
            tracing::error!(%err, "failed to start tokio runtime");
            std::process::exit(1);
        }
    };

    tracing::info!("acp-agent starting on stdio");
    if let Err(err) = runtime.block_on(async move {
        // Serve the inspector UI (when enabled) and attach the same `Inspector`
        // to *both* observability seams: the provider-neutral engine
        // `TurnObserver` (typed request/response snapshots for every backend)
        // and the Anthropic `RequestObserver` (the exact wire body). The
        // inspector also mirrors every capture (and agent log lines) to disk
        // when a log directory is configured.
        let mut request_observer: Option<std::sync::Arc<dyn turn_anthropic::RequestObserver>> = None;
        let mut turn_observer: Option<std::sync::Arc<dyn agent_core::TurnObserver>> = None;
        let mut inspector_base_url: Option<String> = None;

        if let Some(inspector) = inspector {
            match config.inspector_addr.parse() {
                Ok(addr) => match llm_inspector::serve_with_fallback(inspector.clone(), addr).await {
                    Ok(bound) => {
                        tracing::info!(%bound, "acp-agent: LLM inspector UI at http://{bound}");
                        let shared = std::sync::Arc::new(inspector);
                        request_observer = Some(shared.clone());
                        turn_observer = Some(shared);
                        inspector_base_url = Some(format!("http://{bound}"));
                    }
                    Err(err) => {
                        tracing::warn!(%err, addr = %config.inspector_addr, "acp-agent: failed to start LLM inspector; continuing without it");
                    }
                },
                Err(err) => {
                    tracing::warn!(%err, addr = %config.inspector_addr, "acp-agent: invalid inspector address; continuing without it");
                }
            }
        }

        let deps = acp_agent::default_deps_with_observers(
            request_observer,
            turn_observer,
            inspector_base_url,
        );
        acp_agent::run_stdio(deps).await
    }) {
        tracing::error!(%err, "acp-agent terminated with error");
        std::process::exit(1);
    }
}
