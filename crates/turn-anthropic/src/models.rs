//! Static catalog of the models this agent exposes (Claude via Anthropic and
//! DeepSeek via its Anthropic-compatible Messages endpoint), with the specs the
//! rest of the workspace needs: the per-model `max_tokens` output budget and
//! the pricing used to estimate session cost (see the `/usage` command).
//!
//! Prices are US dollars per **million** tokens (MTok) from each provider's
//! public pricing. For Claude, prompt-caching rates follow Anthropic's standard
//! multipliers: a 5-minute cache **write** costs 1.25× the base input rate and
//! a cache **read** costs 0.1×. DeepSeek publishes explicit cache-hit and
//! cache-miss input rates, stored directly on each [`ModelSpec`].

/// Which upstream API hosts a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// Anthropic Messages API (`api.anthropic.com`).
    Anthropic,
    /// DeepSeek's Anthropic-compatible Messages endpoint
    /// (`api.deepseek.com/anthropic`).
    DeepSeek,
}

/// Thinking-control dialect accepted by a model on the Anthropic-shaped wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingDialect {
    /// `thinking: { type: "adaptive" }`, controlled by output effort.
    Adaptive,
    /// `thinking: { type: "enabled", budget_tokens: N }`.
    ///
    /// DeepSeek accepts `type: "enabled"` and ignores `budget_tokens`.
    Budget,
    /// Thinking is disabled unless the caller explicitly overrides it.
    Disabled,
}

/// Static specification for a single model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelSpec {
    /// Provider API model identifier (e.g. `claude-sonnet-5`, `deepseek-v4-flash`).
    pub id: &'static str,
    /// Short human-readable name.
    pub display_name: &'static str,
    /// One-line description.
    pub description: &'static str,
    /// Upstream provider that hosts this model.
    pub provider: Provider,
    /// Maximum output tokens supported on the synchronous Messages API.
    pub max_output_tokens: u32,
    /// Thinking-control dialect accepted by this model.
    pub thinking_dialect: ThinkingDialect,
    /// Whether the model accepts `output_config.effort`.
    pub supports_effort: bool,
    /// Base (cache-miss) input price per million tokens (USD).
    pub input_price_per_mtok: f64,
    /// Output price per million tokens (USD).
    pub output_price_per_mtok: f64,
    /// Price per million tokens to *write* a prompt-cache entry (USD).
    pub cache_write_per_mtok: f64,
    /// Price per million tokens to *read* (reuse) a cached prompt token (USD).
    pub cache_read_per_mtok: f64,
}

impl ModelSpec {
    /// Price per million tokens to *write* a prompt-cache entry.
    pub fn cache_write_price_per_mtok(&self) -> f64 {
        self.cache_write_per_mtok
    }

    /// Price per million tokens to *read* (reuse) a cached prompt token.
    pub fn cache_read_price_per_mtok(&self) -> f64 {
        self.cache_read_per_mtok
    }
}

/// Build a Claude [`ModelSpec`] with Anthropic's standard cache multipliers.
const fn claude(
    id: &'static str,
    display_name: &'static str,
    description: &'static str,
    max_output_tokens: u32,
    thinking_dialect: ThinkingDialect,
    supports_effort: bool,
    input_price_per_mtok: f64,
    output_price_per_mtok: f64,
) -> ModelSpec {
    ModelSpec {
        id,
        display_name,
        description,
        provider: Provider::Anthropic,
        max_output_tokens,
        thinking_dialect,
        supports_effort,
        input_price_per_mtok,
        output_price_per_mtok,
        cache_write_per_mtok: input_price_per_mtok * 1.25,
        cache_read_per_mtok: input_price_per_mtok * 0.1,
    }
}

/// Build a DeepSeek [`ModelSpec`] with explicit cache-hit / cache-miss prices.
const fn deepseek(
    id: &'static str,
    display_name: &'static str,
    description: &'static str,
    input_price_per_mtok: f64,
    cache_read_price_per_mtok: f64,
    output_price_per_mtok: f64,
) -> ModelSpec {
    ModelSpec {
        id,
        display_name,
        description,
        provider: Provider::DeepSeek,
        // DeepSeek V4 documents a 384K max output budget.
        max_output_tokens: 384_000,
        // Anthropic-compatible thinking toggle; budget_tokens is ignored.
        thinking_dialect: ThinkingDialect::Budget,
        supports_effort: true,
        input_price_per_mtok,
        output_price_per_mtok,
        // DeepSeek does not bill a separate cache-write tier; treat writes as
        // cache-miss input so `/usage` stays conservative.
        cache_write_per_mtok: input_price_per_mtok,
        cache_read_per_mtok: cache_read_price_per_mtok,
    }
}

/// The models advertised to clients, in preference order (default first).
///
/// Specs reflect Anthropic's current lineup (Sonnet 5, Opus 4.8, Haiku 4.5,
/// Fable 5) and DeepSeek V4 (Flash + Pro) via the Anthropic-compatible endpoint.
pub const MODELS: &[ModelSpec] = &[
    claude(
        "claude-sonnet-5",
        "Claude Sonnet 5",
        "Best combination of speed and intelligence",
        128_000,
        ThinkingDialect::Adaptive,
        true,
        3.0,
        15.0,
    ),
    claude(
        "claude-opus-4-8",
        "Claude Opus 4.8",
        "For complex agentic coding and enterprise work",
        128_000,
        ThinkingDialect::Adaptive,
        true,
        5.0,
        25.0,
    ),
    claude(
        "claude-haiku-4-5",
        "Claude Haiku 4.5",
        "Fastest model with near-frontier intelligence",
        64_000,
        ThinkingDialect::Budget,
        false,
        1.0,
        5.0,
    ),
    claude(
        "claude-fable-5",
        "Claude Fable 5",
        "Highest capability for long-running agents",
        128_000,
        ThinkingDialect::Adaptive,
        true,
        10.0,
        50.0,
    ),
    deepseek(
        "deepseek-v4-flash",
        "DeepSeek V4 Flash",
        "Fast DeepSeek model for agentic coding",
        0.14,
        0.0028,
        0.28,
    ),
    deepseek(
        "deepseek-v4-pro",
        "DeepSeek V4 Pro",
        "Highest-capability DeepSeek model",
        0.435,
        0.003625,
        0.87,
    ),
    // Legacy aliases (map to V4 Flash non-thinking / thinking on DeepSeek's side).
    deepseek(
        "deepseek-chat",
        "DeepSeek Chat",
        "Legacy alias for DeepSeek V4 Flash (non-thinking default)",
        0.14,
        0.0028,
        0.28,
    ),
    deepseek(
        "deepseek-reasoner",
        "DeepSeek Reasoner",
        "Legacy alias for DeepSeek V4 Flash thinking mode",
        0.14,
        0.0028,
        0.28,
    ),
];

/// The default model used when none is configured.
pub const DEFAULT_MODEL: &str = "claude-sonnet-5";

/// The default DeepSeek model used when DeepSeek is selected without an id.
pub const DEFAULT_DEEPSEEK_MODEL: &str = "deepseek-v4-flash";

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

/// Resolve the [`Provider`] for a model id.
///
/// Known catalog entries use their declared provider. Unknown ids that start
/// with `deepseek` are treated as DeepSeek so a free-form model override still
/// routes to the right endpoint; everything else defaults to Anthropic.
pub fn provider_for(model: &str) -> Provider {
    if let Some(spec) = model_spec(model) {
        return spec.provider;
    }
    if model.starts_with("deepseek") {
        Provider::DeepSeek
    } else {
        Provider::Anthropic
    }
}

/// Whether `model` is hosted by DeepSeek.
pub fn is_deepseek_model(model: &str) -> bool {
    matches!(provider_for(model), Provider::DeepSeek)
}

/// Models belonging to a given provider, in catalog order.
pub fn models_for_provider(provider: Provider) -> impl Iterator<Item = &'static ModelSpec> {
    MODELS.iter().filter(move |m| m.provider == provider)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_models_have_expected_budgets() {
        assert_eq!(default_max_output_tokens("claude-sonnet-5"), 128_000);
        assert_eq!(default_max_output_tokens("claude-haiku-4-5"), 64_000);
        assert_eq!(default_max_output_tokens("deepseek-v4-flash"), 384_000);
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
            default_thinking_dialect("deepseek-v4-pro"),
            ThinkingDialect::Budget
        );
        assert_eq!(
            default_thinking_dialect("unknown-model"),
            ThinkingDialect::Disabled
        );
        assert!(supports_effort("claude-sonnet-5"));
        assert!(!supports_effort("claude-haiku-4-5"));
        assert!(supports_effort("deepseek-v4-flash"));
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

    #[test]
    fn deepseek_pricing_and_provider() {
        let flash = model_spec("deepseek-v4-flash").unwrap();
        assert_eq!(flash.provider, Provider::DeepSeek);
        assert!((flash.input_price_per_mtok - 0.14).abs() < 1e-9);
        assert!((flash.cache_read_price_per_mtok() - 0.0028).abs() < 1e-12);
        assert!(is_deepseek_model("deepseek-reasoner"));
        assert!(!is_deepseek_model("claude-sonnet-5"));
        assert_eq!(provider_for("deepseek-custom"), Provider::DeepSeek);
    }
}
