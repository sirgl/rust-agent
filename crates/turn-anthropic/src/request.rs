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
    pub max_tokens: u32,
    /// Base URL of the messages endpoint (override for tests/proxies).
    pub base_url: String,
    /// Value of the required `anthropic-version` header.
    pub anthropic_version: String,
}

impl AnthropicConfig {
    /// Default Anthropic Messages endpoint.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.anthropic.com/v1/messages";
    /// Default pinned API version.
    pub const DEFAULT_VERSION: &'static str = "2023-06-01";
    /// Default model used when none is configured.
    pub const DEFAULT_MODEL: &'static str = "claude-sonnet-5";

    /// Create a config with sane defaults for the given key and model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            effort: None,
            max_tokens: 4096,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            anthropic_version: Self::DEFAULT_VERSION.to_string(),
        }
    }

    /// Build a config from environment variables:
    /// `ANTHROPIC_API_KEY` (optional; falls back to the default
    /// `token.properties` key file described in
    /// [`load_default_api_key`] when unset), `ANTHROPIC_MODEL` (optional),
    /// `ANTHROPIC_EFFORT` (optional), `ANTHROPIC_BASE_URL` (optional).
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
        Some(cfg)
    }
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

    let mut body = Map::new();
    body.insert("model".into(), json!(config.model));
    body.insert("max_tokens".into(), json!(config.max_tokens));
    body.insert("stream".into(), json!(true));
    body.insert("messages".into(), Value::Array(messages));

    if let Some(effort) = config.effort {
        body.insert("output_config".into(), json!({ "effort": effort }));
    }

    if !system_parts.is_empty() {
        body.insert("system".into(), json!(system_parts.join("\n\n")));
    }

    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "input_schema": t.input_schema,
                })
            })
            .collect();
        body.insert("tools".into(), Value::Array(tools));
    }

    Value::Object(body)
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
        assert_eq!(body["max_tokens"], 4096);
        assert_eq!(body["stream"], true);
        assert_eq!(body["system"], "be terse");
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
}
