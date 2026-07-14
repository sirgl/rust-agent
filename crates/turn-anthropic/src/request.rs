//! Building an Anthropic Messages API request body from a provider-neutral
//! [`agent_core::LlmRequest`].
//!
//! ## Reuse over re-render
//!
//! `turn-anthropic` does **not** re-derive the wire body from the lightweight
//! typed history directly. It goes through the [`agent_core::LlmCompiler`] seam
//! ([`AnthropicTurnService`](crate::AnthropicTurnService) uses
//! [`agent_core::DefaultCompiler`]), which already honors the "preserve and
//! reuse ingested forms" rule: a previously-compiled `LlmMessage` spliced into
//! history via [`agent_core::HistoryEntry::Ingested`] is passed through
//! verbatim. Because `agent-core`'s [`agent_core::LlmContentBlock`] serde
//! tags (`text` / `tool_use` / `tool_result`) and role names already match the
//! Anthropic block/role vocabulary 1:1, translating `LlmRequest` into the wire
//! body is a structural pass that preserves that content exactly rather than a
//! lossy re-rendering.

use agent_core::{LlmMessage, LlmRequest, LlmRole};
use serde_json::{json, Map, Value};

/// Selection of "effort" (reasoning/thought level) for the response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    /// Low effort (faster, less reasoning).
    Low,
    /// Medium effort (balanced).
    Medium,
    /// High effort (slower, more reasoning).
    High,
}

impl std::str::FromStr for Effort {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" => Ok(Effort::Low),
            "medium" => Ok(Effort::Medium),
            "high" => Ok(Effort::High),
            _ => Err(format!("invalid effort: {}", s)),
        }
    }
}

/// How extended thinking is requested from the model.
///
/// Anthropic has two thinking control dialects depending on the model:
///
/// - **Adaptive** (Claude Sonnet 5, Opus 4.7/4.8 and newer): request
///   `thinking: {type: "adaptive"}` and steer depth via `output_config.effort`.
///   The legacy `{type: "enabled", budget_tokens}` form is *rejected* (HTTP 400)
///   by these models. On these models thinking text defaults to *omitted* (only
///   the `signature` is returned), so `display: "summarized"` is requested to
///   surface visible reasoning.
/// - **Budget** (Opus 4.6 and earlier): the legacy `{type: "enabled",
///   budget_tokens: N}` form.
///
/// The default is [`ThinkingConfig::Adaptive`] to match the default model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinkingConfig {
    /// Thinking turned off. Sent as `{type: "disabled"}` (the documented way to
    /// turn off adaptive thinking, which is on-by-default on newer models).
    Disabled,
    /// Adaptive thinking. `display_summarized` requests visible summarized
    /// reasoning text (otherwise newer models omit the thinking text).
    Adaptive {
        /// Whether to request `display: "summarized"` for visible reasoning.
        display_summarized: bool,
    },
    /// Legacy budget-based extended thinking for older models.
    Budget {
        /// The `budget_tokens` value (clamped to the API minimum).
        budget_tokens: u32,
    },
}

/// Static configuration for talking to the Anthropic Messages API.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// API key sent in the `x-api-key` header.
    pub api_key: String,
    /// Model identifier (e.g. `claude-sonnet-5`).
    pub model: String,
    /// Selection of effort level.
    pub effort: Option<Effort>,
    /// Maximum number of tokens to sample for the response.
    ///
    /// The Anthropic Messages API *requires* `max_tokens`, so it is always sent.
    /// When `None`, a per-model default is derived via
    /// [`default_max_output_tokens`] so that the value matches the selected
    /// model's real output budget instead of an arbitrary fixed cap.
    pub max_tokens: Option<u32>,
    /// Base URL of the messages endpoint (override for tests/proxies).
    pub base_url: String,
    /// Value of the required `anthropic-version` header.
    pub anthropic_version: String,
    /// Extended-thinking configuration (see [`ThinkingConfig`]). Defaults to
    /// adaptive thinking with visible summarized reasoning.
    pub thinking: ThinkingConfig,
    /// Whether to insert prompt-caching (`cache_control`) breakpoints to
    /// maximize reuse of the (usually stable) system prompt + tool definitions
    /// and the growing conversation prefix across turns. Enabled by default.
    pub prompt_caching: bool,
}

impl AnthropicConfig {
    /// Default Anthropic Messages endpoint.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.anthropic.com/v1/messages";
    /// Default pinned API version.
    pub const DEFAULT_VERSION: &'static str = "2023-06-01";
    /// Default model used when none is configured.
    pub const DEFAULT_MODEL: &'static str = "claude-sonnet-5";

    /// Minimum `budget_tokens` Anthropic accepts for extended thinking.
    pub const MIN_THINKING_BUDGET: u32 = 1024;

    /// Default budget used for legacy [`ThinkingConfig::Budget`] mode when a
    /// budget is requested without an explicit value. Extended thinking is now
    /// **on by default** (the typed history preserves and replays the signed
    /// thinking blocks required for tool use), and can be disabled with
    /// `ANTHROPIC_THINKING=0`.
    pub const DEFAULT_THINKING_BUDGET: u32 = 2048;

    /// Default response token budget.
    ///
    /// Anthropic *requires* a `max_tokens` field, so it cannot be removed; but
    /// the old 4096 default truncated large generations (the model returned
    /// `stop_reason: max_tokens` mid-answer). Modern Claude models accept a much
    /// larger output budget, so we default high and let `ANTHROPIC_MAX_TOKENS`
    /// override it.
    pub const DEFAULT_MAX_TOKENS: u32 = 32000;

    /// Create a config with sane defaults for the given key and model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            effort: None,
            max_tokens: None,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            anthropic_version: Self::DEFAULT_VERSION.to_string(),
            thinking: ThinkingConfig::Adaptive {
                display_summarized: true,
            },
            prompt_caching: true,
        }
    }

    /// Build a config from environment variables:
    /// `ANTHROPIC_API_KEY` (optional; falls back to the default
    /// `token.properties` key file described in
    /// [`load_default_api_key`] when unset), `ANTHROPIC_MODEL` (optional),
    /// `ANTHROPIC_EFFORT` (optional), `ANTHROPIC_BASE_URL` (optional),
    /// `ANTHROPIC_THINKING` / `ANTHROPIC_THINKING_BUDGET` (optional; override the
    /// extended-thinking budget or disable it — thinking is **on by default**,
    /// so `ANTHROPIC_THINKING=0`/`false` turns it off), and
    /// `ANTHROPIC_PROMPT_CACHING` (optional; set to a falsy value to disable
    /// prompt caching, which is on by default).
    /// Returns `None` if no API key can be resolved from either source.
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|key| !key.is_empty())
            .or_else(load_default_api_key)?;
        let model =
            std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| Self::DEFAULT_MODEL.to_string());
        let mut cfg = Self::new(api_key, model);
        if let Ok(effort_str) = std::env::var("ANTHROPIC_EFFORT") {
            if let Ok(effort) = effort_str.parse() {
                cfg.effort = Some(effort);
            }
        }
        if let Ok(base) = std::env::var("ANTHROPIC_BASE_URL") {
            cfg.base_url = base;
        }
        if let Ok(raw) = std::env::var("ANTHROPIC_MAX_TOKENS") {
            if let Ok(n) = raw.trim().parse::<u32>() {
                if n > 0 {
                    cfg.max_tokens = Some(n);
                }
            }
        }
        if let Some(thinking) = thinking_override_from_env() {
            cfg.thinking = thinking;
        }
        if let Ok(flag) = std::env::var("ANTHROPIC_PROMPT_CACHING") {
            cfg.prompt_caching = is_truthy(&flag);
        }
        Some(cfg)
    }
}

/// Resolve an extended-thinking override from the environment.
///
/// Returns `None` when no thinking env var is set (keep the config default,
/// which is adaptive thinking on), otherwise the requested [`ThinkingConfig`].
///
/// - `ANTHROPIC_THINKING_BUDGET=<n>` (positive integer) selects the legacy
///   [`ThinkingConfig::Budget`] mode (only for older models that support it).
/// - `ANTHROPIC_THINKING` accepts:
///   - a falsy value / `0` -> [`ThinkingConfig::Disabled`];
///   - a truthy value -> [`ThinkingConfig::Adaptive`] with visible reasoning;
///   - a positive number -> legacy [`ThinkingConfig::Budget`].
///
/// When both are present, `ANTHROPIC_THINKING` is applied last and wins.
fn thinking_override_from_env() -> Option<ThinkingConfig> {
    let mut result: Option<ThinkingConfig> = None;
    if let Ok(raw) = std::env::var("ANTHROPIC_THINKING_BUDGET") {
        if let Ok(n) = raw.trim().parse::<u32>() {
            result = Some(if n > 0 {
                ThinkingConfig::Budget {
                    budget_tokens: n.max(AnthropicConfig::MIN_THINKING_BUDGET),
                }
            } else {
                ThinkingConfig::Disabled
            });
        }
    }
    if let Ok(flag) = std::env::var("ANTHROPIC_THINKING") {
        if let Ok(n) = flag.trim().parse::<u32>() {
            result = Some(if n > 0 {
                ThinkingConfig::Budget {
                    budget_tokens: n.max(AnthropicConfig::MIN_THINKING_BUDGET),
                }
            } else {
                ThinkingConfig::Disabled
            });
        } else if is_truthy(&flag) {
            result = Some(ThinkingConfig::Adaptive {
                display_summarized: true,
            });
        } else {
            result = Some(ThinkingConfig::Disabled);
        }
    }
    result
}

/// Whether a textual flag should be treated as "on".
fn is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// The property name looked up in the default key file.
const TOKEN_PROPERTIES_KEY_NAME: &str = "KEY";

/// Candidate locations for the default `token.properties` file, tried in
/// order. The first entry favors overriding the path in tests/deployments;
/// the second is derived from this crate's location within the workspace
/// (`crates/turn-anthropic/../../token.properties` == the workspace root),
/// which stays correct even if the workspace is checked out elsewhere; the
/// last is the literal path used during development, kept as a final
/// fallback.
fn token_properties_candidates() -> Vec<std::path::PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(path) = std::env::var("ACP_TOKEN_PROPERTIES_PATH") {
        candidates.push(std::path::PathBuf::from(path));
    }
    candidates.push(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("token.properties"),
    );
    candidates.push(std::path::PathBuf::from(
        "/Users/roman.ivanov/IdeaProjects/rust-agent/token.properties",
    ));
    candidates
}

/// Load a default Anthropic API key from a `.properties`-style file (`KEY=...`)
/// when no explicit `ANTHROPIC_API_KEY` environment variable is set.
///
/// This lets the ACP agent start with a working Anthropic backend during local
/// development without exporting an environment variable on every run. The key
/// is treated as a secret end-to-end: this function never logs its value, only
/// the path it was (or was not) found at, and callers must not print it either.
///
/// Returns `None` (falling back cleanly to no Anthropic config, per
/// [`AnthropicConfig::from_env`]) if none of the candidate files exist, are
/// readable, or contain a non-empty `KEY` property.
pub fn load_default_api_key() -> Option<String> {
    token_properties_candidates()
        .into_iter()
        .find_map(|path| read_key_from_path(&path))
}

/// Read and parse the `KEY` property from a single `token.properties`-style
/// file at `path`. Returns `None` (without treating it as an error) if the
/// file does not exist, cannot be read, or has no non-empty `KEY` property.
fn read_key_from_path(path: &std::path::Path) -> Option<String> {
    match std::fs::read_to_string(path) {
        Ok(contents) => {
            let key = parse_key_property(&contents, TOKEN_PROPERTIES_KEY_NAME);
            if key.is_some() {
                tracing::debug!(
                    path = %path.display(),
                    "loaded default Anthropic API key from token.properties"
                );
            } else {
                tracing::debug!(
                    path = %path.display(),
                    "token.properties found but has no non-empty `{TOKEN_PROPERTIES_KEY_NAME}` property"
                );
            }
            key
        }
        Err(err) => {
            tracing::trace!(path = %path.display(), %err, "token.properties not readable");
            None
        }
    }
}

/// Parse a `.properties`-style `key=value` line for `property_name`, skipping
/// blank lines and `#`/`!` comments, trimming surrounding whitespace and
/// matching quotes.
fn parse_key_property(contents: &str, property_name: &str) -> Option<String> {
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('!') {
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        if name.trim() != property_name {
            continue;
        }
        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .unwrap_or(value);
        return (!value.is_empty()).then(|| value.to_string());
    }
    None
}

/// Convert a compiled [`LlmRequest`] plus config into the Anthropic Messages
/// API request JSON body (with `stream: true`).
///
/// System-role messages, should any slip through the compiler, are folded into
/// the top-level `system` field (Anthropic keeps the system prompt out of the
/// `messages` array); user/assistant messages are passed through structurally.
pub fn build_body(req: &LlmRequest, config: &AnthropicConfig) -> Value {
    let mut system_parts: Vec<String> = Vec::new();
    if let Some(system) = &req.system {
        system_parts.push(system.clone());
    }

    let mut messages: Vec<Value> = Vec::with_capacity(req.messages.len());
    for msg in &req.messages {
        if msg.role == LlmRole::System {
            system_parts.push(system_text(msg));
            continue;
        }
        messages.push(serde_json::to_value(msg).unwrap_or_else(|_| json!({})));
    }

    // Merge consecutive messages that share the same role into one message with
    // a concatenated `content` array. This is required for extended thinking +
    // tool use: Anthropic mandates that the signed `thinking` block appear in the
    // *same* assistant message as the following `tool_use` (thinking first). Our
    // typed history stores thinking, assistant text, and the tool call as
    // separate consecutive assistant entries, so merging folds them back into a
    // single, correctly-ordered assistant turn. It also coalesces parallel
    // tool_result user messages, which Anthropic prefers.
    merge_consecutive_same_role(&mut messages);

    let caching = config.prompt_caching;

    // Cache the tail of the conversation so its (immutable) prefix is a cache
    // hit on the next turn, which only appends new messages.
    if caching {
        mark_last_message_cacheable(&mut messages);
    }

    let mut body = Map::new();
    body.insert("model".into(), json!(config.model));
    body.insert("max_tokens".into(), json!(effective_max_tokens(config)));
    body.insert("stream".into(), json!(true));
    body.insert("messages".into(), Value::Array(messages));

    if let Some(effort) = config.effort {
        body.insert("output_config".into(), json!({ "effort": effort }));
    }

    // Configure extended thinking. The response then carries `thinking` blocks
    // (surfaced end-to-end as thoughts, and preserved with their signature for
    // replay). Newer models (Sonnet 5 / Opus 4.7+) require the `adaptive` form
    // and reject `enabled`; older models use the legacy `budget_tokens` form.
    match &config.thinking {
        ThinkingConfig::Disabled => {
            body.insert("thinking".into(), json!({ "type": "disabled" }));
        }
        ThinkingConfig::Adaptive { display_summarized } => {
            let mut thinking = json!({ "type": "adaptive" });
            if *display_summarized {
                thinking["display"] = json!("summarized");
            }
            body.insert("thinking".into(), thinking);
        }
        ThinkingConfig::Budget { budget_tokens } => {
            // `budget_tokens` must be strictly less than `max_tokens`, which
            // `effective_max_tokens` guarantees.
            let budget = (*budget_tokens).max(AnthropicConfig::MIN_THINKING_BUDGET);
            body.insert(
                "thinking".into(),
                json!({ "type": "enabled", "budget_tokens": budget }),
            );
        }
    }

    if !system_parts.is_empty() {
        let text = system_parts.join("\n\n");
        // Cache the (usually stable) system prompt. Placed as a single text
        // block with a cache breakpoint so both tools and system are reused.
        if caching {
            body.insert(
                "system".into(),
                json!([{
                    "type": "text",
                    "text": text,
                    "cache_control": { "type": "ephemeral" },
                }]),
            );
        } else {
            body.insert("system".into(), json!(text));
        }
    }

    if !req.tools.is_empty() {
        let last = req.tools.len() - 1;
        let tools: Vec<Value> = req
            .tools
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut tool = json!({
                    "name": t.name,
                    "input_schema": t.input_schema,
                });
                // A single breakpoint on the last tool caches the whole tool
                // block, which rarely changes between turns.
                if caching && i == last {
                    if let Some(obj) = tool.as_object_mut() {
                        obj.insert("cache_control".into(), json!({ "type": "ephemeral" }));
                    }
                }
                tool
            })
            .collect();
        body.insert("tools".into(), Value::Array(tools));
    }

    Value::Object(body)
}

/// The `max_tokens` actually sent. In legacy [`ThinkingConfig::Budget`] mode it
/// is guaranteed to stay strictly greater than the thinking `budget_tokens` (an
/// Anthropic API requirement); other modes send `max_tokens` unchanged.
fn effective_max_tokens(config: &AnthropicConfig) -> u32 {
    // Resolve the base budget: an explicit override, otherwise the per-model
    // default derived from the model catalog.
    let base = config
        .max_tokens
        .unwrap_or_else(|| crate::models::default_max_output_tokens(&config.model));
    match &config.thinking {
        ThinkingConfig::Budget { budget_tokens } => {
            let budget = (*budget_tokens).max(AnthropicConfig::MIN_THINKING_BUDGET);
            // Leave room for the visible answer on top of the thinking budget.
            base.max(budget.saturating_add(4096))
        }
        ThinkingConfig::Disabled | ThinkingConfig::Adaptive { .. } => base,
    }
}

/// Fold consecutive messages sharing the same `role` into a single message,
/// concatenating their `content` arrays in order. This preserves block ordering
/// within each merged turn (e.g. `thinking` -> `text` -> `tool_use`).
///
/// Messages whose `content` is not an array are left as standalone entries (they
/// act as a boundary that is not merged), so no information is lost.
fn merge_consecutive_same_role(messages: &mut Vec<Value>) {
    if messages.len() < 2 {
        return;
    }
    let mut merged: Vec<Value> = Vec::with_capacity(messages.len());
    for msg in messages.drain(..) {
        let can_merge = merged.last().is_some_and(|prev: &Value| {
            let same_role = prev.get("role") == msg.get("role") && msg.get("role").is_some();
            let both_arrays = prev.get("content").map(Value::is_array).unwrap_or(false)
                && msg.get("content").map(Value::is_array).unwrap_or(false);
            same_role && both_arrays
        });
        if can_merge {
            // Append this message's content blocks onto the previous message.
            let incoming = msg
                .get("content")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if let Some(prev) = merged.last_mut() {
                if let Some(prev_content) = prev.get_mut("content").and_then(Value::as_array_mut) {
                    prev_content.extend(incoming);
                }
            }
        } else {
            merged.push(msg);
        }
    }
    *messages = merged;
}

/// Attach an ephemeral cache breakpoint to the last content block of the last
/// message, so the conversation prefix up to that point can be reused as a
/// cache hit on the following turn. No-op when there are no messages or the
/// last message has no structured content array.
fn mark_last_message_cacheable(messages: &mut [Value]) {
    let Some(last) = messages.last_mut() else {
        return;
    };
    let Some(content) = last.get_mut("content").and_then(Value::as_array_mut) else {
        return;
    };
    if let Some(block) = content.last_mut() {
        if let Some(obj) = block.as_object_mut() {
            obj.insert("cache_control".into(), json!({ "type": "ephemeral" }));
        }
    }
}

/// Flatten the text content of a system-role message.
fn system_text(msg: &LlmMessage) -> String {
    msg.content
        .iter()
        .filter_map(|b| serde_json::to_value(b).ok())
        .filter_map(|v| {
            v.get("text")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::{DefaultCompiler, LlmCompiler, SessionState, ToolDescriptor};

    fn cfg() -> AnthropicConfig {
        AnthropicConfig::new("sk-test", "claude-test")
    }

    #[test]
    fn builds_streaming_body_with_system_and_messages() {
        let mut session = SessionState::new("s1");
        session.system_prompt = Some("be terse".into());
        session.push_user_text("hi");
        session.push_assistant_text("hello");
        let ctx = session.turn_context();
        let req = DefaultCompiler.compile(&ctx).unwrap();

        let body = build_body(&req, &cfg());
        assert_eq!(body["model"], "claude-test");
        // Unknown model -> conservative per-model default.
        assert_eq!(
            body["max_tokens"],
            crate::models::default_max_output_tokens("claude-test")
        );
        assert_eq!(body["stream"], true);
        // With prompt caching on (the default), `system` is a structured block
        // array carrying a cache breakpoint rather than a bare string.
        assert_eq!(body["system"][0]["type"], "text");
        assert_eq!(body["system"][0]["text"], "be terse");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["type"], "text");
        assert_eq!(body["messages"][0]["content"][0]["text"], "hi");
        // No tools -> field omitted entirely.
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn tool_schemas_are_forwarded() {
        let mut session = SessionState::new("s1");
        session.push_user_text("read a file");
        session.available_tools.push(ToolDescriptor {
            name: "fs_read".into(),
            schema: serde_json::json!({"type":"object","properties":{"path":{"type":"string"}}}),
            requires_permission: false,
        });
        let ctx = session.turn_context();
        let req = DefaultCompiler.compile(&ctx).unwrap();

        let body = build_body(&req, &cfg());
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "fs_read");
        assert_eq!(tools[0]["input_schema"]["type"], "object");
    }

    #[test]
    fn tool_use_and_result_blocks_map_to_anthropic_shape() {
        use agent_core::{EditToolCall, KnownTool, ToolCallId};
        let mut session = SessionState::new("s1");
        let id = ToolCallId::new("toolu_1");
        session.push_tool_call(
            id.clone(),
            KnownTool::Edit(EditToolCall {
                path: "/a.txt".into(),
                old_text: "x".into(),
                new_text: "y".into(),
            }),
        );
        session.push_tool_result(id, true, "done");
        let ctx = session.turn_context();
        let req = DefaultCompiler.compile(&ctx).unwrap();

        let body = build_body(&req, &cfg());
        let messages = body["messages"].as_array().unwrap();
        // assistant tool_use, then user tool_result
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"][0]["type"], "tool_use");
        assert_eq!(messages[0]["content"][0]["id"], "toolu_1");
        assert_eq!(messages[0]["content"][0]["name"], "edit");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"][0]["type"], "tool_result");
        assert_eq!(messages[1]["content"][0]["tool_use_id"], "toolu_1");
        assert_eq!(messages[1]["content"][0]["is_error"], false);
    }

    #[test]
    fn parses_key_property_ignoring_comments_and_other_keys() {
        let contents = "# a comment\n! also a comment\nOTHER=nope\nKEY=sk-ant-secret\n";
        assert_eq!(
            parse_key_property(contents, "KEY"),
            Some("sk-ant-secret".to_string())
        );
    }

    #[test]
    fn parses_key_property_with_surrounding_whitespace_and_quotes() {
        let contents = "  KEY = \"sk-ant-quoted\"  \n";
        assert_eq!(
            parse_key_property(contents, "KEY"),
            Some("sk-ant-quoted".to_string())
        );
    }

    #[test]
    fn missing_key_property_returns_none() {
        let contents = "OTHER=value\n";
        assert_eq!(parse_key_property(contents, "KEY"), None);
    }

    #[test]
    fn empty_key_value_returns_none() {
        let contents = "KEY=\n";
        assert_eq!(parse_key_property(contents, "KEY"), None);
    }

    #[test]
    fn lines_without_equals_are_skipped_not_fatal() {
        // A malformed line before the real property must not short-circuit
        // parsing (regression test for the `?`-in-loop bug).
        let contents = "not a property line\nKEY=sk-ant-after-malformed\n";
        assert_eq!(
            parse_key_property(contents, "KEY"),
            Some("sk-ant-after-malformed".to_string())
        );
    }

    #[test]
    fn reads_key_from_an_actual_token_properties_file() {
        let dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = dir.join(format!(
            "turn-anthropic-test-token-{}-{nanos}.properties",
            std::process::id()
        ));
        std::fs::write(&path, "KEY=sk-ant-from-file\n").unwrap();

        assert_eq!(
            read_key_from_path(&path),
            Some("sk-ant-from-file".to_string())
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn effort_parsing() {
        assert_eq!("low".parse::<Effort>().unwrap(), Effort::Low);
        assert_eq!("MEDIUM".parse::<Effort>().unwrap(), Effort::Medium);
        assert_eq!("High".parse::<Effort>().unwrap(), Effort::High);
        assert!("invalid".parse::<Effort>().is_err());
    }

    #[test]
    fn builds_body_with_effort() {
        let mut cfg = cfg();
        cfg.effort = Some(Effort::High);
        let req = DefaultCompiler
            .compile(&SessionState::new("s1").turn_context())
            .unwrap();
        let body = build_body(&req, &cfg);
        assert_eq!(body["output_config"]["effort"], "high");
    }

    #[test]
    fn builds_body_without_effort_omits_field() {
        let cfg = cfg(); // effort is None by default
        let req = DefaultCompiler
            .compile(&SessionState::new("s1").turn_context())
            .unwrap();
        let body = build_body(&req, &cfg);
        assert!(body.get("output_config").is_none());
    }

    #[test]
    fn from_env_reads_effort() {
        std::env::set_var("ANTHROPIC_API_KEY", "sk-test");
        std::env::set_var("ANTHROPIC_EFFORT", "medium");
        let cfg = AnthropicConfig::from_env().unwrap();
        assert_eq!(cfg.effort, Some(Effort::Medium));
        std::env::remove_var("ANTHROPIC_EFFORT");

        std::env::set_var("ANTHROPIC_EFFORT", "invalid");
        let cfg = AnthropicConfig::from_env().unwrap();
        assert_eq!(cfg.effort, None); // Should ignore invalid effort
        std::env::remove_var("ANTHROPIC_EFFORT");
    }

    #[test]
    fn missing_token_properties_file_falls_back_cleanly() {
        let path = std::env::temp_dir().join("turn-anthropic-test-token-does-not-exist.properties");
        let _ = std::fs::remove_file(&path);
        assert_eq!(read_key_from_path(&path), None);
    }

    #[test]
    fn legacy_budget_thinking_enables_and_raises_max_tokens() {
        let mut cfg = cfg();
        cfg.thinking = ThinkingConfig::Budget {
            budget_tokens: 3000,
        };
        let req = DefaultCompiler
            .compile(&SessionState::new("s1").turn_context())
            .unwrap();
        let body = build_body(&req, &cfg);
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 3000);
        // max_tokens must stay strictly above the thinking budget.
        assert!(body["max_tokens"].as_u64().unwrap() > 3000);
    }

    #[test]
    fn legacy_budget_thinking_clamped_to_minimum() {
        let mut cfg = cfg();
        cfg.thinking = ThinkingConfig::Budget { budget_tokens: 10 };
        let req = DefaultCompiler
            .compile(&SessionState::new("s1").turn_context())
            .unwrap();
        let body = build_body(&req, &cfg);
        assert_eq!(
            body["thinking"]["budget_tokens"],
            AnthropicConfig::MIN_THINKING_BUDGET
        );
    }

    #[test]
    fn adaptive_thinking_enabled_by_default() {
        let req = DefaultCompiler
            .compile(&SessionState::new("s1").turn_context())
            .unwrap();
        // The default config enables adaptive thinking with visible reasoning.
        let body = build_body(&req, &cfg());
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["thinking"]["display"], "summarized");
    }

    #[test]
    fn thinking_disabled_sends_disabled_type() {
        let mut cfg = cfg();
        cfg.thinking = ThinkingConfig::Disabled;
        let req = DefaultCompiler
            .compile(&SessionState::new("s1").turn_context())
            .unwrap();
        let body = build_body(&req, &cfg);
        assert_eq!(body["thinking"]["type"], "disabled");
    }

    #[test]
    fn merges_thinking_text_and_tool_use_into_one_assistant_message() {
        use agent_core::history::ThinkingRecord;
        use agent_core::{KnownTool, ToolCallId};

        let mut session = SessionState::new("s1");
        session.push_user_text("do it");
        // A single assistant turn: thinking, then text, then a tool call —
        // recorded as three consecutive assistant history entries.
        session.push_thinking(ThinkingRecord::Thinking {
            text: "let me think".into(),
            signature: "sig-123".into(),
        });
        session.push_assistant_text("on it");
        session.push_tool_call(
            ToolCallId::new("call-1"),
            KnownTool::classify("fs_read", serde_json::json!({ "path": "/a" })),
        );
        let ctx = session.turn_context();
        let req = DefaultCompiler.compile(&ctx).unwrap();
        // Disable caching so we assert purely on merge/ordering.
        let mut cfg = cfg();
        cfg.prompt_caching = false;
        let body = build_body(&req, &cfg);

        let messages = body["messages"].as_array().unwrap();
        // user + one merged assistant message.
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["role"], "assistant");
        let content = messages[1]["content"].as_array().unwrap();
        // thinking first, then text, then tool_use — order preserved.
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[0]["thinking"], "let me think");
        assert_eq!(content[0]["signature"], "sig-123");
        assert_eq!(content[1]["type"], "text");
        assert_eq!(content[2]["type"], "tool_use");
    }

    #[test]
    fn caching_marks_last_tool_and_last_message() {
        let mut session = SessionState::new("s1");
        session.push_user_text("hi");
        session.available_tools.push(ToolDescriptor {
            name: "a".into(),
            schema: serde_json::json!({"type":"object"}),
            requires_permission: false,
        });
        session.available_tools.push(ToolDescriptor {
            name: "b".into(),
            schema: serde_json::json!({"type":"object"}),
            requires_permission: false,
        });
        let ctx = session.turn_context();
        let req = DefaultCompiler.compile(&ctx).unwrap();
        let body = build_body(&req, &cfg());

        let tools = body["tools"].as_array().unwrap();
        // Only the last tool carries a breakpoint (caches the whole tool block).
        assert!(tools[0].get("cache_control").is_none());
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");

        // The last message's last content block is cacheable.
        let messages = body["messages"].as_array().unwrap();
        let last = messages.last().unwrap();
        let last_block = last["content"].as_array().unwrap().last().unwrap();
        assert_eq!(last_block["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn caching_disabled_keeps_plain_system_and_no_breakpoints() {
        let mut cfg = cfg();
        cfg.prompt_caching = false;
        let mut session = SessionState::new("s1");
        session.system_prompt = Some("be terse".into());
        session.push_user_text("hi");
        session.available_tools.push(ToolDescriptor {
            name: "a".into(),
            schema: serde_json::json!({"type":"object"}),
            requires_permission: false,
        });
        let ctx = session.turn_context();
        let req = DefaultCompiler.compile(&ctx).unwrap();
        let body = build_body(&req, &cfg);

        assert_eq!(body["system"], "be terse");
        assert!(body["tools"][0].get("cache_control").is_none());
        let messages = body["messages"].as_array().unwrap();
        let last_block = messages.last().unwrap()["content"]
            .as_array()
            .unwrap()
            .last()
            .unwrap();
        assert!(last_block.get("cache_control").is_none());
    }
}
