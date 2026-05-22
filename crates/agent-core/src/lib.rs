use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: Value,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, args: Value) -> Self {
        Self {
            id: format!("call_{}", uuid_like()),
            name: name.into(),
            args,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub name: String,
    pub ok: bool,
    pub content: Value,
}

impl ToolResult {
    pub fn ok(call: &ToolCall, content: Value) -> Self {
        Self {
            call_id: call.id.clone(),
            name: call.name.clone(),
            ok: true,
            content,
        }
    }

    pub fn error(call: &ToolCall, message: impl Into<String>) -> Self {
        Self {
            call_id: call.id.clone(),
            name: call.name.clone(),
            ok: false,
            content: serde_json::json!({ "status": "error", "message": message.into() }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutcome {
    pub data: Value,
    pub next_prompt: Option<String>,
    pub should_exit: bool,
}

impl StepOutcome {
    pub fn continue_with(data: Value, next_prompt: impl Into<String>) -> Self {
        Self {
            data,
            next_prompt: Some(next_prompt.into()),
            should_exit: false,
        }
    }

    pub fn done(data: Value) -> Self {
        Self {
            data,
            next_prompt: None,
            should_exit: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolContext {
    pub cwd: PathBuf,
    pub skills_dir: PathBuf,
    pub memory_dir: PathBuf,
    pub sessions_dir: PathBuf,
}

impl ToolContext {
    pub fn new(cwd: impl Into<PathBuf>, skills_dir: impl Into<PathBuf>) -> Self {
        let cwd = cwd.into();
        Self {
            memory_dir: cwd.join("memory"),
            sessions_dir: cwd.join("sessions"),
            cwd,
            skills_dir: skills_dir.into(),
        }
    }

    pub fn with_cwd(cwd: impl Into<PathBuf>) -> Self {
        let cwd = cwd.into();
        Self {
            skills_dir: cwd.join("skills"),
            memory_dir: cwd.join("memory"),
            sessions_dir: cwd.join("sessions"),
            cwd,
        }
    }

    pub fn with_paths(
        cwd: impl Into<PathBuf>,
        skills_dir: impl Into<PathBuf>,
        memory_dir: impl Into<PathBuf>,
        sessions_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            cwd: cwd.into(),
            skills_dir: skills_dir.into(),
            memory_dir: memory_dir.into(),
            sessions_dir: sessions_dir.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    RunStarted {
        goal: String,
        cwd: PathBuf,
    },
    ToolStarted {
        call: ToolCall,
    },
    ToolFinished {
        result: ToolResult,
    },
    TurnSummary {
        turn: usize,
        summary: String,
    },
    CheckpointUpdated {
        key_info: String,
        related_skill: Option<String>,
    },
    LongTermUpdateStarted {
        reason: String,
        evidence: Option<String>,
    },
    LongTermUpdateSettled {
        decision: String,
        target: Option<String>,
        reason: String,
        evidence: Option<String>,
        changed: bool,
    },
    Reflection {
        summary: String,
    },
    RunFinished {
        status: String,
        summary: String,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("invalid arguments for {tool}: {source}")]
    InvalidArguments {
        tool: String,
        source: serde_json::Error,
    },
    #[error("{0}")]
    Failed(String),
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T>(&mut self, tool: T)
    where
        T: Tool + 'static,
    {
        self.tools.insert(tool.name().to_string(), Box::new(tool));
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(String::as_str).collect()
    }

    pub fn infos(&self) -> Vec<ToolInfo> {
        self.tools
            .values()
            .map(|tool| ToolInfo {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
            })
            .collect()
    }

    pub fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let tool = self
            .tools
            .get(&call.name)
            .ok_or_else(|| ToolError::UnknownTool(call.name.clone()))?;
        tool.execute(ctx, call)
    }
}

fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("{nanos:x}")
}
