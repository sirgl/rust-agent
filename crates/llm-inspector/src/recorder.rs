//! The [`Recorder`] sink abstraction and its concrete implementations.
//!
//! A recorder is a destination for [`CapturedRequest`]s. The [`crate::Inspector`]
//! fans every captured request out to all attached recorders, so where captured
//! requests go (in-memory for the UI, JSONL files on disk, or both) is a
//! composition concern rather than something baked into the capture path.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::record::CapturedRequest;

/// A destination for captured LLM requests.
///
/// Implementations must be cheap and non-blocking enough to run inline on the
/// turn path, and must never panic: a recorder failure must never break a turn.
pub trait Recorder: Send + Sync {
    /// Persist or buffer a captured request.
    fn record(&self, request: &CapturedRequest);
}

/// The default maximum number of captured requests retained in memory.
///
/// Older records are evicted once this many are stored, so a long-running agent
/// never grows the inspector's memory without bound.
pub const DEFAULT_CAPACITY: usize = 500;

/// An in-memory, capacity-bounded ring buffer of captured requests.
///
/// This backs the live web UI: [`snapshot`](MemoryRecorder::snapshot) returns
/// the retained records newest-first.
pub struct MemoryRecorder {
    sends: Mutex<VecDeque<CapturedRequest>>,
    capacity: usize,
}

impl MemoryRecorder {
    /// Create a recorder retaining at most `capacity` records (min 1).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            sends: Mutex::new(VecDeque::new()),
            capacity: capacity.max(1),
        }
    }

    /// Snapshot all retained records, newest first.
    #[must_use]
    pub fn snapshot(&self) -> Vec<CapturedRequest> {
        let sends = self.sends.lock().expect("memory recorder mutex poisoned");
        sends.iter().rev().cloned().collect()
    }

    /// The number of currently retained records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sends
            .lock()
            .expect("memory recorder mutex poisoned")
            .len()
    }

    /// Whether no records are currently retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for MemoryRecorder {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }
}

impl Recorder for MemoryRecorder {
    fn record(&self, request: &CapturedRequest) {
        // Guard is a plain std::Mutex held only for the push/trim; never across
        // an `.await`, so it is safe on the turn path.
        let mut sends = self.sends.lock().expect("memory recorder mutex poisoned");
        sends.push_back(request.clone());
        while sends.len() > self.capacity {
            sends.pop_front();
        }
    }
}

/// A recorder that appends captured requests as newline-delimited JSON (JSONL),
/// one file per session, under a base directory.
///
/// Each session's records land in `<dir>/<session_id>.jsonl`, so the on-disk log
/// mirrors the UI's per-session grouping and is trivially greppable/tailable.
/// File handles are cached per session; write failures are logged and swallowed
/// so logging never breaks a turn.
pub struct FileRecorder {
    dir: PathBuf,
    files: Mutex<HashMap<String, File>>,
}

impl FileRecorder {
    /// Create a file recorder writing under `dir`, creating it if needed.
    ///
    /// # Errors
    ///
    /// Returns an error only if the base directory cannot be created.
    pub fn new(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            files: Mutex::new(HashMap::new()),
        })
    }

    /// The base directory this recorder writes to.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The path of the JSONL log file for `session_id`.
    #[must_use]
    pub fn log_path(&self, session_id: &str) -> PathBuf {
        self.dir.join(format!("{}.jsonl", sanitize(session_id)))
    }

    fn append_line(&self, session_id: &str, line: &str) -> std::io::Result<()> {
        let mut files = self.files.lock().expect("file recorder mutex poisoned");
        if !files.contains_key(session_id) {
            let path = self.log_path(session_id);
            let file = OpenOptions::new().create(true).append(true).open(path)?;
            files.insert(session_id.to_string(), file);
        }
        let file = files.get_mut(session_id).expect("file just inserted");
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
        file.flush()
    }
}

impl Recorder for FileRecorder {
    fn record(&self, request: &CapturedRequest) {
        let line = match serde_json::to_string(request) {
            Ok(line) => line,
            Err(err) => {
                tracing::warn!(%err, "llm-inspector: failed to serialize record for file log");
                return;
            }
        };
        if let Err(err) = self.append_line(&request.session_id, &line) {
            tracing::warn!(%err, session_id = %request.session_id, "llm-inspector: failed to append to log file");
        }
    }
}

/// Replace path-hostile characters so a session id is a safe file name.
fn sanitize(session_id: &str) -> String {
    session_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::CapturedPayload;

    fn typed_request(id: u64, session: &str) -> CapturedRequest {
        CapturedRequest {
            id,
            sequence_no: id,
            session_id: session.to_string(),
            agent_path: Some("root".to_string()),
            trace: None,
            timestamp_ms: 0,
            model: None,
            payload: CapturedPayload::Typed {
                system_prompt: None,
                tools: Vec::new(),
                history: serde_json::json!([]),
                num_entries: 0,
                depth: 0,
            },
        }
    }

    #[test]
    fn memory_recorder_evicts_and_orders_newest_first() {
        let rec = MemoryRecorder::with_capacity(2);
        rec.record(&typed_request(1, "s"));
        rec.record(&typed_request(2, "s"));
        rec.record(&typed_request(3, "s"));
        let snap = rec.snapshot();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].id, 3);
        assert_eq!(snap[1].id, 2);
    }

    #[test]
    fn file_recorder_appends_jsonl_per_session() {
        let dir = std::env::temp_dir().join(format!("llm-insp-test-{}", std::process::id()));
        let rec = FileRecorder::new(&dir).unwrap();
        rec.record(&typed_request(1, "sess/a"));
        rec.record(&typed_request(2, "sess/a"));
        let path = rec.log_path("sess/a");
        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"id\":1"));
        assert!(lines[1].contains("\"id\":2"));
        // The session id was sanitized for the file name.
        assert!(path.to_string_lossy().contains("sess_a.jsonl"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
