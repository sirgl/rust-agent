//! Model and effort selection for ACP sessions.

use turn_anthropic::{Effort, MODELS};
use agent_client_protocol::schema::v1::{
    SessionConfigId, SessionConfigKind, SessionConfigOption, SessionConfigOptionCategory,
    SessionConfigSelect, SessionConfigSelectOption, SessionConfigSelectOptions,
    SessionConfigValueId,
};

/// Selection of a model and effort level for a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSelection {
    /// The model identifier (e.g. `claude-sonnet-5`).
    pub model: String,
    /// The effort level for the model.
    pub effort: Effort,
}


impl ModelSelection {
    /// Create a new selection based on the agent configuration.
    pub fn for_config(config: &crate::config::AgentConfig) -> Self {
        Self {
            model: config.default_model.clone(),
            effort: Effort::High,
        }
    }

    /// Convert the selection to ACP session configuration options.
    pub fn config_options(&self) -> Vec<SessionConfigOption> {
        vec![
            SessionConfigOption::new(
                SessionConfigId::new("model"),
                "Model",
                SessionConfigKind::Select(SessionConfigSelect::new(
                    SessionConfigValueId::new(self.model.clone()),
                    SessionConfigSelectOptions::Ungrouped(
                        MODELS
                            .iter()
                            .map(|m| {
                                SessionConfigSelectOption::new(
                                    SessionConfigValueId::new(m.id.to_string()),
                                    m.display_name.to_string(),
                                )
                                .description(m.description.to_string())
                            })
                            .collect(),
                    ),
                )),
            )
            .category(SessionConfigOptionCategory::Model),
            SessionConfigOption::new(
                SessionConfigId::new("effort"),
                "Effort",
                SessionConfigKind::Select(SessionConfigSelect::new(
                    SessionConfigValueId::new(self.effort_to_string()),
                    SessionConfigSelectOptions::Ungrouped(vec![
                        SessionConfigSelectOption::new(
                            SessionConfigValueId::new("low"),
                            "Low",
                        ),
                        SessionConfigSelectOption::new(
                            SessionConfigValueId::new("medium"),
                            "Medium",
                        ),
                        SessionConfigSelectOption::new(
                            SessionConfigValueId::new("high"),
                            "High",
                        ),
                    ]),
                )),
            )
            .category(SessionConfigOptionCategory::Model),
        ]
    }

    fn effort_to_string(&self) -> String {
        match self.effort {
            Effort::Low => "low".to_string(),
            Effort::Medium => "medium".to_string(),
            Effort::High => "high".to_string(),
        }
    }

    /// Apply a configuration update from the client.
    ///
    /// Returns `true` if the update was valid and applied, `false` otherwise.
    pub fn apply_update(&mut self, config_id: &str, value: serde_json::Value) -> bool {
        match config_id {
            "model" => {
                if let Some(s) = value.as_str() {
                    if MODELS.iter().any(|m| m.id == s) {
                        self.model = s.to_string();
                        return true;
                    }
                }
            }
            "effort" => {
                if let Some(s) = value.as_str() {
                    if let Ok(effort) = s.parse::<Effort>() {
                        self.effort = effort;
                        return true;
                    }
                }
            }
            _ => {}
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;
    use serde_json::json;
    use agent_client_protocol::schema::v1::SessionConfigOptionValue;

    #[test]
    fn initial_selection_from_config() {
        let config = AgentConfig::default();
        let selection = ModelSelection::for_config(&config);
        assert_eq!(selection.model, "claude-sonnet-5");
        assert_eq!(selection.effort, Effort::High);
    }

    #[test]
    fn generate_config_options() {
        let selection = ModelSelection {
            model: "claude-haiku-4-5".to_string(),
            effort: Effort::Low,
        };
        let options = selection.config_options();
        assert_eq!(options.len(), 2);

        let model_opt = options.iter().find(|o| o.id.0.as_ref() == "model").unwrap();
        assert_eq!(model_opt.name, "Model");
        if let SessionConfigKind::Select(select) = &model_opt.kind {
            assert_eq!(select.current_value.0.as_ref(), "claude-haiku-4-5");
            if let SessionConfigSelectOptions::Ungrouped(opts) = &select.options {
                assert_eq!(opts.len(), MODELS.len());
                assert!(opts
                    .iter()
                    .any(|o| o.value.0.as_ref() == "claude-sonnet-5"));
            } else {
                panic!("expected ungrouped options");
            }
        } else {
            panic!("expected select kind");
        }

        let effort_opt = options
            .iter()
            .find(|o| o.id.0.as_ref() == "effort")
            .unwrap();
        assert_eq!(effort_opt.name, "Effort");
        if let SessionConfigKind::Select(select) = &effort_opt.kind {
            assert_eq!(select.current_value.0.as_ref(), "low");
        } else {
            panic!("expected select kind");
        }
    }

    #[test]
    fn apply_valid_model_update() {
        let mut selection = ModelSelection {
            model: "claude-sonnet-5".to_string(),
            effort: Effort::High,
        };
        assert!(selection.apply_update("model", json!("claude-haiku-4-5")));
        assert_eq!(selection.model, "claude-haiku-4-5");
    }

    #[test]
    fn apply_invalid_model_update_is_ignored() {
        let mut selection = ModelSelection {
            model: "claude-sonnet-5".to_string(),
            effort: Effort::High,
        };
        assert!(!selection.apply_update("model", json!("invalid-model")));
        assert_eq!(selection.model, "claude-sonnet-5");
    }

    #[test]
    fn apply_valid_effort_update() {
        let mut selection = ModelSelection {
            model: "claude-sonnet-5".to_string(),
            effort: Effort::High,
        };
        assert!(selection.apply_update("effort", json!("medium")));
        assert_eq!(selection.effort, Effort::Medium);
    }

    #[test]
    fn apply_invalid_effort_update_is_ignored() {
        let mut selection = ModelSelection {
            model: "claude-sonnet-5".to_string(),
            effort: Effort::High,
        };
        assert!(!selection.apply_update("effort", json!("extra-high")));
        assert_eq!(selection.effort, Effort::High);
    }

    #[test]
    fn apply_update_from_acp_value() {
        let mut selection = ModelSelection {
            model: "claude-sonnet-5".to_string(),
            effort: Effort::High,
        };

        // Simulating how we extract value from SessionConfigOptionValue in agent.rs
        // Using from_value to avoid needing to know exact struct variant field names
        let val: SessionConfigOptionValue = serde_json::from_value(json!({"value": "claude-haiku-4-5"})).unwrap();

        let json_value = if let Some(id) = val.as_value_id() {
            serde_json::Value::String(id.0.to_string())
        } else if let Some(b) = val.as_bool() {
            serde_json::Value::Bool(b)
        } else {
            serde_json::to_value(val).unwrap_or_default()
        };

        assert!(selection.apply_update("model", json_value));
        assert_eq!(selection.model, "claude-haiku-4-5");
    }
}
