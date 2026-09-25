//! Pluggable persistence for session *core* state.
//!
//! A [`SessionStore`] saves and loads [`SessionRecord`] snapshots so that a
//! session's memory (history + usage) can survive process restarts. Two
//! implementations are provided:
//!
//! - [`InMemorySessionStore`] — the default, non-durable store used in tests
//!   and when persistence is not configured.
//! - [`JsonFileSessionStore`] — a disk-backed store that writes one JSON file
//!   per session under a configurable directory.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;

use crate::error::{AgentError, Result};
use crate::session::SessionRecord;

fn deserialize_session_record(contents: &str) -> serde_json::Result<SessionRecord> {
    match serde_json::from_str(contents) {
        Ok(record) => Ok(record),
        Err(original_error) => {
            let mut value: serde_json::Value = match serde_json::from_str(contents) {
                Ok(value) => value,
                Err(_) => return Err(original_error),
            };
            if !repair_legacy_thinking_entries(&mut value) {
                return Err(original_error);
            }
            serde_json::from_value(value)
        }
    }
}

fn repair_legacy_thinking_entries(value: &mut serde_json::Value) -> bool {
    let Some(entries) = value
        .pointer_mut("/history/entries")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return false;
    };

    let mut repaired = false;
    for entry in entries {
        let Some(object) = entry.as_object_mut() else {
            continue;
        };
        if object.contains_key("thinking_kind") {
            continue;
        }

        let kind = object
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        match kind.as_deref() {
            Some("thinking")
                if object.get("text").is_some() && object.get("signature").is_some() =>
            {
                object.insert(
                    "thinking_kind".to_string(),
                    serde_json::Value::String("thinking".to_string()),
                );
                repaired = true;
            }
            Some("redacted") if object.get("data").is_some() => {
                object.insert(
                    "kind".to_string(),
                    serde_json::Value::String("thinking".to_string()),
                );
                object.insert(
                    "thinking_kind".to_string(),
                    serde_json::Value::String("redacted".to_string()),
                );
                repaired = true;
            }
            _ => {}
        }
    }
    repaired
}

/// A pluggable backing store for persisted session core state.
///
/// Implementations must be safe to share across async tasks (`Send + Sync`)
/// and are typically held behind an `Arc<dyn SessionStore>`.
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Persist (create or overwrite) the given session record.
    async fn save(&self, record: &SessionRecord) -> Result<()>;

    /// Load a session record by id, returning `None` if it does not exist.
    async fn load(&self, session_id: &str) -> Result<Option<SessionRecord>>;

    /// List the ids of all persisted sessions.
    async fn list_ids(&self) -> Result<Vec<String>>;
}

/// An in-memory [`SessionStore`] backed by a `HashMap`.
///
/// State is lost when the process exits; this is the default store used when
/// no persistence directory is configured.
#[derive(Debug, Default)]
pub struct InMemorySessionStore {
    records: Mutex<HashMap<String, SessionRecord>>,
}

impl InMemorySessionStore {
    /// Create a new, empty in-memory store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn save(&self, record: &SessionRecord) -> Result<()> {
        let mut guard = self
            .records
            .lock()
            .map_err(|_| AgentError::Other("session store mutex poisoned".into()))?;
        guard.insert(record.session_id.clone(), record.clone());
        Ok(())
    }

    async fn load(&self, session_id: &str) -> Result<Option<SessionRecord>> {
        let guard = self
            .records
            .lock()
            .map_err(|_| AgentError::Other("session store mutex poisoned".into()))?;
        Ok(guard.get(session_id).cloned())
    }

    async fn list_ids(&self) -> Result<Vec<String>> {
        let guard = self
            .records
            .lock()
            .map_err(|_| AgentError::Other("session store mutex poisoned".into()))?;
        Ok(guard.keys().cloned().collect())
    }
}

/// A disk-backed [`SessionStore`] that writes one JSON file per session.
///
/// Files are stored as `<dir>/<session_id>.json`. The directory is created
/// lazily on the first successful save.
#[derive(Debug, Clone)]
pub struct JsonFileSessionStore {
    dir: PathBuf,
}

impl JsonFileSessionStore {
    /// Create a store rooted at the given directory.
    ///
    /// The directory is not created eagerly; it is created on the first
    /// [`save`](SessionStore::save).
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The directory in which session files are stored.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Compute the on-disk path for a given session id.
    fn path_for(&self, session_id: &str) -> PathBuf {
        self.dir.join(format!("{session_id}.json"))
    }
}

#[async_trait]
impl SessionStore for JsonFileSessionStore {
    async fn save(&self, record: &SessionRecord) -> Result<()> {
        std::fs::create_dir_all(&self.dir)
            .map_err(|e| AgentError::Other(format!("failed to create session dir: {e}")))?;
        let path = self.path_for(&record.session_id);
        let json = serde_json::to_string_pretty(record)?;
        std::fs::write(&path, json)
            .map_err(|e| AgentError::Other(format!("failed to write session file: {e}")))?;
        Ok(())
    }

    async fn load(&self, session_id: &str) -> Result<Option<SessionRecord>> {
        let path = self.path_for(session_id);
        match std::fs::read_to_string(&path) {
            Ok(contents) => {
                let record = deserialize_session_record(&contents)?;
                Ok(Some(record))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(AgentError::Other(format!(
                "failed to read session file: {e}"
            ))),
        }
    }

    async fn list_ids(&self) -> Result<Vec<String>> {
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => {
                return Err(AgentError::Other(format!(
                    "failed to read session dir: {e}"
                )))
            }
        };
        let mut ids = Vec::new();
        for entry in entries {
            let entry =
                entry.map_err(|e| AgentError::Other(format!("failed to read dir entry: {e}")))?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    ids.push(stem.to_string());
                }
            }
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::{HistoryEntry, ThinkingRecord};
    use crate::session::{SessionState, SESSION_RECORD_VERSION};

    fn sample_record(id: &str) -> SessionRecord {
        let mut state = SessionState::new(id);
        state.workspace_roots = vec!["/tmp/root".into()];
        state.system_prompt = Some("be helpful".into());
        state.push_user_text("hello");
        state.push_assistant_text("hi there");
        state.to_record()
    }

    #[tokio::test]
    async fn in_memory_save_load_and_unknown() {
        let store = InMemorySessionStore::new();
        let record = sample_record("sess-1");
        store.save(&record).await.unwrap();

        let loaded = store.load("sess-1").await.unwrap();
        assert_eq!(loaded, Some(record));

        let missing = store.load("does-not-exist").await.unwrap();
        assert_eq!(missing, None);
    }

    #[tokio::test]
    async fn in_memory_list_ids() {
        let store = InMemorySessionStore::new();
        store.save(&sample_record("a")).await.unwrap();
        store.save(&sample_record("b")).await.unwrap();
        let mut ids = store.list_ids().await.unwrap();
        ids.sort();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
    }

    #[tokio::test]
    async fn json_file_round_trip() {
        let dir = std::env::temp_dir().join(format!("agent-core-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = JsonFileSessionStore::new(&dir);
        let record = sample_record("sess-json");

        store.save(&record).await.unwrap();
        // File is written at <dir>/<id>.json.
        assert!(dir.join("sess-json.json").exists());

        let loaded = store.load("sess-json").await.unwrap();
        assert_eq!(loaded, Some(record));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn json_file_list_ids() {
        let dir =
            std::env::temp_dir().join(format!("agent-core-store-list-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = JsonFileSessionStore::new(&dir);
        store.save(&sample_record("x")).await.unwrap();
        store.save(&sample_record("y")).await.unwrap();

        let mut ids = store.list_ids().await.unwrap();
        ids.sort();
        assert_eq!(ids, vec!["x".to_string(), "y".to_string()]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn json_file_missing_returns_none() {
        let dir =
            std::env::temp_dir().join(format!("agent-core-store-missing-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = JsonFileSessionStore::new(&dir);

        // Missing directory entirely.
        assert_eq!(store.load("nope").await.unwrap(), None);
        assert_eq!(store.list_ids().await.unwrap(), Vec::<String>::new());

        // Directory exists but file missing.
        store.save(&sample_record("present")).await.unwrap();
        assert_eq!(store.load("absent").await.unwrap(), None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn json_file_corrupt_returns_error() {
        let dir =
            std::env::temp_dir().join(format!("agent-core-store-corrupt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("broken.json"), "{ not valid json").unwrap();

        let store = JsonFileSessionStore::new(&dir);
        let err = store.load("broken").await;
        assert!(err.is_err(), "corrupt JSON should surface an error");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn json_file_load_repairs_legacy_duplicate_thinking_kinds() {
        let dir = std::env::temp_dir().join(format!(
            "agent-core-store-legacy-thinking-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("legacy-thinking.json"),
            r#"{
  "version": 1,
  "session_id": "legacy-thinking",
  "workspace_roots": [],
  "history": {
    "entries": [
      {
        "kind": "thinking",
        "kind": "thinking",
        "text": "consider the options",
        "signature": "signature"
      },
      {
        "kind": "thinking",
        "kind": "redacted",
        "data": "opaque"
      }
    ]
  },
  "system_prompt": null,
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0
  }
}"#,
        )
        .unwrap();

        let store = JsonFileSessionStore::new(&dir);
        let record = store.load("legacy-thinking").await.unwrap().unwrap();

        assert!(matches!(
            &record.history.entries[0],
            HistoryEntry::Thinking(ThinkingRecord::Thinking { text, signature })
                if text == "consider the options" && signature == "signature"
        ));
        assert!(matches!(
            &record.history.entries[1],
            HistoryEntry::Thinking(ThinkingRecord::Redacted { data }) if data == "opaque"
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn json_file_unknown_version_still_deserializes() {
        // A record with an unexpected version still deserializes; the handler
        // layer is responsible for deciding how to treat unsupported versions.
        let dir =
            std::env::temp_dir().join(format!("agent-core-store-version-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = JsonFileSessionStore::new(&dir);

        let mut record = sample_record("versioned");
        record.version = SESSION_RECORD_VERSION + 999;
        store.save(&record).await.unwrap();

        let loaded = store.load("versioned").await.unwrap().unwrap();
        assert_eq!(loaded.version, SESSION_RECORD_VERSION + 999);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
