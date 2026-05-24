use crate::AgentEvent;
use anyhow::Context;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Library error surface for `agent-session`. Callers most commonly want to
/// distinguish "no session matches this id/name" from underlying IO so they
/// can offer a useful "did you mean … (latest)" hint; rarer cases flow
/// through `Other(anyhow::Error)`.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// Caller asked for a specific session by id/name but no file in the
    /// store matched.
    #[error("session not found: {id} (in {root})")]
    NotFound { id: String, root: PathBuf },

    /// `last_session_path` was called against an empty store.
    #[error("no sessions found in {root}")]
    NoSessions { root: PathBuf },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type SessionResult<T> = std::result::Result<T, SessionError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub ts: DateTime<Utc>,
    pub session_id: String,
    pub event: AgentEvent,
}

#[derive(Debug, Clone)]
pub struct SessionStore {
    root: PathBuf,
}

impl SessionStore {
    pub fn new(root: impl Into<PathBuf>) -> SessionResult<Self> {
        let root = root.into();
        fs::create_dir_all(&root).with_context(|| format!("create {}", root.display()))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn start(&self) -> SessionResult<SessionWriter> {
        let id = Uuid::new_v4().to_string();
        let path = self.root.join(format!("{id}.jsonl"));
        let file = OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("create session {}", path.display()))?;
        Ok(SessionWriter { id, path, file })
    }

    pub fn last_session_path(&self) -> SessionResult<PathBuf> {
        let mut sessions = fs::read_dir(&self.root)
            .with_context(|| format!("read {}", self.root.display()))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "jsonl"))
            .filter_map(|entry| {
                let modified = entry.metadata().ok()?.modified().ok()?;
                Some((modified, entry.path()))
            })
            .collect::<Vec<_>>();
        sessions.sort_by_key(|(modified, _)| *modified);
        sessions
            .pop()
            .map(|(_, path)| path)
            .ok_or_else(|| SessionError::NoSessions {
                root: self.root.clone(),
            })
    }

    pub fn resolve(&self, session: Option<&str>) -> SessionResult<PathBuf> {
        match session {
            None | Some("last") => self.last_session_path(),
            Some(value) => {
                let path = Path::new(value);
                if path.is_absolute() || path.exists() {
                    Ok(path.to_path_buf())
                } else {
                    Ok(self.root.join(format!("{value}.jsonl")))
                }
            }
        }
    }

    pub fn read(&self, session: Option<&str>) -> SessionResult<Vec<SessionRecord>> {
        let path = self.resolve(session)?;
        read_records(&path)
    }
}

pub struct SessionWriter {
    id: String,
    path: PathBuf,
    file: File,
}

impl SessionWriter {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn append(&mut self, event: AgentEvent) -> SessionResult<()> {
        let record = SessionRecord {
            ts: Utc::now(),
            session_id: self.id.clone(),
            event,
        };
        serde_json::to_writer(&mut self.file, &record).context("serialize session event")?;
        self.file
            .write_all(b"\n")
            .context("write session newline")?;
        self.file.flush().context("flush session log")?;
        Ok(())
    }
}

pub fn read_records(path: &Path) -> SessionResult<Vec<SessionRecord>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line.context("read session line")?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(serde_json::from_str(&line).context("parse session record")?);
    }
    Ok(records)
}
