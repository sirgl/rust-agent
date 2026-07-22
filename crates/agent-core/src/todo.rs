//! Canonical, validated todo state owned by an agent session.

use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};

use crate::event::{PlanStep, PlanStepStatus, PlanUpdate};

/// Validation failure for a todo identifier or list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TodoValidationError {
    /// An item identifier is empty or contains only whitespace.
    EmptyId,
    /// An item's content is empty or contains only whitespace.
    EmptyContent {
        /// Identifier of the invalid item.
        id: String,
    },
    /// Two items in a replacement list use the same stable identifier.
    DuplicateId {
        /// Identifier that appeared more than once.
        id: String,
    },
}

impl fmt::Display for TodoValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyId => f.write_str("todo item id must not be empty"),
            Self::EmptyContent { id } => {
                write!(f, "todo item '{id}' content must not be empty")
            }
            Self::DuplicateId { id } => write!(f, "duplicate todo item id: {id}"),
        }
    }
}

impl std::error::Error for TodoValidationError {}

/// Stable, caller-chosen identifier of a todo item.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct TodoItemId(String);

impl TodoItemId {
    /// Validate and create an identifier.
    pub fn new(id: impl Into<String>) -> Result<Self, TodoValidationError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(TodoValidationError::EmptyId);
        }
        Ok(Self(id))
    }

    /// Borrow the identifier text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TodoItemId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for TodoItemId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let id = String::deserialize(deserializer)?;
        Self::new(id).map_err(serde::de::Error::custom)
    }
}

/// One ordered item in a session's canonical todo list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoItem {
    /// Stable identifier used by later mutations.
    pub id: TodoItemId,
    /// Human-readable work description.
    pub content: String,
    /// Current progress status.
    pub status: PlanStepStatus,
}

/// The complete, ordered todo list for one session.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct TodoList {
    /// Ordered items. A full replacement preserves this order exactly.
    pub items: Vec<TodoItem>,
}

impl TodoList {
    /// Validate and create a complete replacement list.
    pub fn new(items: Vec<TodoItem>) -> Result<Self, TodoValidationError> {
        let list = Self { items };
        list.validate()?;
        Ok(list)
    }

    /// Validate identifier uniqueness and non-empty content without mutation.
    pub fn validate(&self) -> Result<(), TodoValidationError> {
        let mut ids = HashSet::with_capacity(self.items.len());
        for item in &self.items {
            if item.content.trim().is_empty() {
                return Err(TodoValidationError::EmptyContent {
                    id: item.id.to_string(),
                });
            }
            if !ids.insert(item.id.as_str()) {
                return Err(TodoValidationError::DuplicateId {
                    id: item.id.to_string(),
                });
            }
        }
        Ok(())
    }

    /// Project the canonical state into the ID-less plan presentation model.
    #[must_use]
    pub fn to_plan_update(&self) -> PlanUpdate {
        PlanUpdate {
            steps: self
                .items
                .iter()
                .map(|item| PlanStep {
                    content: item.content.clone(),
                    status: item.status,
                })
                .collect(),
        }
    }
}

impl<'de> Deserialize<'de> for TodoList {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TodoListData {
            items: Vec<TodoItem>,
        }

        let data = TodoListData::deserialize(deserializer)?;
        Self::new(data.items).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, content: &str, status: PlanStepStatus) -> TodoItem {
        TodoItem {
            id: TodoItemId::new(id).unwrap(),
            content: content.to_string(),
            status,
        }
    }

    #[test]
    fn validates_ids_content_and_duplicates() {
        assert_eq!(TodoItemId::new("  "), Err(TodoValidationError::EmptyId));
        assert!(matches!(
            TodoList::new(vec![item("a", " ", PlanStepStatus::Pending)]),
            Err(TodoValidationError::EmptyContent { id }) if id == "a"
        ));
        assert!(matches!(
            TodoList::new(vec![
                item("a", "first", PlanStepStatus::Pending),
                item("a", "second", PlanStepStatus::Completed),
            ]),
            Err(TodoValidationError::DuplicateId { id }) if id == "a"
        ));
    }

    #[test]
    fn serde_rejects_invalid_state_and_preserves_order() {
        assert!(serde_json::from_value::<TodoList>(serde_json::json!({
            "items": [{"id": "", "content": "x", "status": "pending"}]
        }))
        .is_err());
        assert!(serde_json::from_value::<TodoList>(serde_json::json!({
            "items": [
                {"id": "a", "content": "x", "status": "pending"},
                {"id": "a", "content": "y", "status": "completed"}
            ]
        }))
        .is_err());

        let list = TodoList::new(vec![
            item("b", "second", PlanStepStatus::InProgress),
            item("a", "first", PlanStepStatus::Pending),
        ])
        .unwrap();
        let decoded: TodoList =
            serde_json::from_value(serde_json::to_value(&list).unwrap()).unwrap();
        assert_eq!(decoded, list);
        assert_eq!(decoded.items[0].id.as_str(), "b");
    }

    #[test]
    fn plan_projection_drops_ids_only_at_presentation_boundary() {
        let list = TodoList::new(vec![item(
            "stable-id",
            "verify behavior",
            PlanStepStatus::Completed,
        )])
        .unwrap();
        assert_eq!(
            list.to_plan_update(),
            PlanUpdate {
                steps: vec![PlanStep {
                    content: "verify behavior".to_string(),
                    status: PlanStepStatus::Completed,
                }]
            }
        );
    }
}
