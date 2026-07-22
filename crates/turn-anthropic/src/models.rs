//! Static catalog of the Claude models this agent exposes, with the specs the
//! rest of the workspace needs: the per-model `max_tokens` output budget and
//! the pricing used to estimate session cost (see the `/usage` command).
//!
//! Prices are US dollars per **million** tokens (MTok) from Anthropic's public
//! pricing. Prompt-caching rates follow Anthropic's standard multipliers: a
//! 5-minute cache **write** costs 1.25× the base input rate and a cache
//! **read** (a reused/cached token) costs 0.1× the base input rate.

/// Thinking-control dialect accepted by a Claude model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingDialect {
    /// `thinking: { type: "adaptive" }`, controlled by output effort.
    Adaptive,
    /// `thinking: { type: "enabled", budget_tokens: N }`.
    Budget,
    /// Thinking is disabled unless the caller explicitly overrides it.
    Disabled,
}

/// Static specification for a single Claude model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelSpec {
    /// Claude API model identifier (e.g. `claude-sonnet-5`).
    pub id: &'static str,
    /// Short human-readable name.
    pub display_name: &'static str,
    /// One-line description.
    pub description: &'static str,
    /// Maximum output tokens supported on the synchronous Messages API.
    pub max_output_tokens: u32,
    /// Thinking-control dialect accepted by this model.
    pub thinking_dialect: ThinkingDialect,
    /// Whether the model accepts `output_config.effort`.
    pub supports_effort: bool,
    /// Base input price per million tokens (USD).
    pub input_price_per_mtok: f64,
    /// Output price per million tokens (USD).
    pub output_price_per_mtok: f64,
}

impl ModelSpec {
    /// Price per million tokens to *write* a 5-minute prompt-cache entry.
    pub fn cache_write_price_per_mtok(&self) -> f64 {
        self.input_price_per_mtok * 1.25
    }

    /// Price per million tokens to *read* (reuse) a cached prompt token.
    pub fn cache_read_price_per_mtok(&self) -> f64 {
        self.input_price_per_mtok * 0.1
    }
}

/// The models advertised to clients, in preference order (default first).
///
/// Specs reflect Anthropic's current lineup: Sonnet 5, Opus 4.8, Haiku 4.5 and
/// Fable 5.
pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "claude-sonnet-5",
        display_name: "Claude Sonnet 5",
        description: "Best combination of speed and intelligence",
        max_output_tokens: 128_000,
        thinking_dialect: ThinkingDialect::Adaptive,
        supports_effort: true,
        input_price_per_mtok: 3.0,
        output_price_per_mtok: 15.0,
    },
    ModelSpec {
        id: "claude-opus-4-8",
        display_name: "Claude Opus 4.8",
        description: "For complex agentic coding and enterprise work",
        max_output_tokens: 128_000,
        thinking_dialect: ThinkingDialect::Adaptive,
        supports_effort: true,
        input_price_per_mtok: 5.0,
        output_price_per_mtok: 25.0,
    },
    ModelSpec {
        id: "claude-haiku-4-5",
        display_name: "Claude Haiku 4.5",
        description: "Fastest model with near-frontier intelligence",
        max_output_tokens: 64_000,
        thinking_dialect: ThinkingDialect::Budget,
        supports_effort: false,
        input_price_per_mtok: 1.0,
        output_price_per_mtok: 5.0,
    },
    ModelSpec {
        id: "claude-fable-5",
        display_name: "Claude Fable 5",
        description: "Highest capability for long-running agents",
        max_output_tokens: 128_000,
        thinking_dialect: ThinkingDialect::Adaptive,
        supports_effort: true,
        input_price_per_mtok: 10.0,
        output_price_per_mtok: 50.0,
    },
];

/// The default model used when none is configured.
pub const DEFAULT_MODEL: &str = "claude-sonnet-5";

/// Conservative output-token budget for a model that is not in [`MODELS`].
///
/// All current Claude models support at least this many output tokens, so it is
/// a safe fallback that will not truncate responses.
pub const FALLBACK_MAX_OUTPUT_TOKENS: u32 = 8192;

/// Look up the [`ModelSpec`] for a model identifier, if known.
///
/// Matches the exact id and also the dated snapshot form (e.g.
/// `claude-haiku-4-5-20251001` matches the `claude-haiku-4-5` alias).
pub fn model_spec(model: &str) -> Option<&'static ModelSpec> {
    MODELS
        .iter()
        .find(|m| m.id == model || model.starts_with(m.id))
}

/// The maximum output tokens for a model, or [`FALLBACK_MAX_OUTPUT_TOKENS`] when
/// the model is unknown.
pub fn default_max_output_tokens(model: &str) -> u32 {
    model_spec(model)
        .map(|m| m.max_output_tokens)
        .unwrap_or(FALLBACK_MAX_OUTPUT_TOKENS)
}

/// The default thinking dialect for a model.
///
/// Unknown models default to disabled thinking rather than risking an invalid
/// provider request with a control dialect they may not support.
pub fn default_thinking_dialect(model: &str) -> ThinkingDialect {
    model_spec(model)
        .map(|m| m.thinking_dialect)
        .unwrap_or(ThinkingDialect::Disabled)
}

/// Whether a model accepts the Anthropic `output_config.effort` field.
///
/// Unknown models conservatively omit the optional field.
pub fn supports_effort(model: &str) -> bool {
    model_spec(model).is_some_and(|m| m.supports_effort)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_models_have_expected_budgets() {
        assert_eq!(default_max_output_tokens("claude-sonnet-5"), 128_000);
        assert_eq!(default_max_output_tokens("claude-haiku-4-5"), 64_000);
    }

    #[test]
    fn known_models_expose_their_thinking_dialect() {
        assert_eq!(
            default_thinking_dialect("claude-sonnet-5"),
            ThinkingDialect::Adaptive
        );
        assert_eq!(
            default_thinking_dialect("claude-haiku-4-5"),
            ThinkingDialect::Budget
        );
        assert_eq!(
            default_thinking_dialect("unknown-model"),
            ThinkingDialect::Disabled
        );
        assert!(supports_effort("claude-sonnet-5"));
        assert!(!supports_effort("claude-haiku-4-5"));
        assert!(!supports_effort("unknown-model"));
    }

    #[test]
    fn dated_snapshot_matches_alias() {
        let spec = model_spec("claude-haiku-4-5-20251001").expect("known");
        assert_eq!(spec.id, "claude-haiku-4-5");
    }

    #[test]
    fn unknown_model_falls_back() {
        assert!(model_spec("something-else").is_none());
        assert_eq!(
            default_max_output_tokens("something-else"),
            FALLBACK_MAX_OUTPUT_TOKENS
        );
    }

    #[test]
    fn cache_multipliers() {
        let spec = model_spec("claude-sonnet-5").unwrap();
        assert!((spec.cache_write_price_per_mtok() - 3.75).abs() < 1e-9);
        assert!((spec.cache_read_price_per_mtok() - 0.30).abs() < 1e-9);
    }
}
