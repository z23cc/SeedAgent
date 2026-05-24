use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

pub mod session;
pub mod tui;

/// Whether a run is allowed to mutate the project. Set by `run_goal` and
/// pushed into `agent_tools::run_mode_guard` so individual tools (notably
/// `ShellTool`) can refuse write-shaped operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    ReadOnly,
    #[default]
    Implementation,
}

/// Provenance for the active `RunMode` — appears in session JSONLs and the
/// trace header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModeSource {
    #[default]
    Auto,
    Explicit,
}

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
    /// 1-indexed planner turn from the runtime loop; `0` means "not in
    /// a loop". Tools use it to scale output limits as runs grow longer.
    pub current_turn: usize,
}

impl ToolContext {
    /// Test/one-off convenience: derives skills/memory/sessions dirs as
    /// `cwd.join(...)`. Production uses [`Self::with_paths`] instead.
    pub fn with_cwd(cwd: impl Into<PathBuf>) -> Self {
        let cwd = cwd.into();
        Self {
            skills_dir: cwd.join("skills"),
            memory_dir: cwd.join("memory"),
            sessions_dir: cwd.join("sessions"),
            cwd,
            current_turn: 0,
        }
    }

    /// Per-task cwd may be a subdir (after `/cd`) but the host's
    /// memory/skills/sessions storage stays at the workspace root.
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
            current_turn: 0,
        }
    }

    pub fn with_turn(mut self, turn: usize) -> Self {
        self.current_turn = turn;
        self
    }

    /// Returns a default scaled by turn pressure. `base` is the natural
    /// default; `floor` is the smallest it can shrink to.
    pub fn scaled_default(&self, base: usize, floor: usize) -> usize {
        if self.current_turn < 5 {
            return base;
        }
        let divisor = 1 + self.current_turn / 5;
        (base / divisor).max(floor.min(base))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    RunStarted {
        goal: String,
        cwd: PathBuf,
        /// `serde(default)` so older session JSONLs (no `mode` field)
        /// deserialize as `Implementation`.
        #[serde(default)]
        mode: RunMode,
        #[serde(default)]
        mode_source: ModeSource,
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
    TurnTimings {
        turn: usize,
        planner_ms: u64,
        exec_ms: u64,
        /// Approximation for output tokens when the provider doesn't
        /// report `tokens` directly.
        planner_chars: usize,
        #[serde(default)]
        prompt_chars: usize,
    },
    RunFinished {
        status: String,
        summary: String,
    },
    PlannerRetry {
        turn: usize,
        /// 1-based; `attempt=1` is the first retry after the original failed.
        attempt: usize,
        of: usize,
        /// 0 for parse-retries (no sleep).
        backoff_ms: u64,
        /// `"parse"` for InvalidPlannerJson, `"transport"` for `Planner(_)`.
        kind: String,
        reason: String,
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

    /// Schema rendered next to the tool description in the planner
    /// prompt so the LLM picks correct field names. Convention: derive
    /// `JsonSchema` on the args struct + use the `impl_args_schema!` macro.
    fn args_schema(&self) -> Option<serde_json::Value> {
        None
    }

    /// True iff the tool is purely a read (same inputs ⇒ same output
    /// within a run). Used by the planner loop's per-run memoization
    /// layer. Convention: read tools use the `impl_pure_read!` macro.
    fn is_pure_read(&self) -> bool {
        false
    }
}

/// Compact `JsonSchema` → `Value` for `ToolInfo.args_schema`. Strips
/// `$schema` and `title` (Rust-type-name pollution) so the schema
/// reads cleanly in the planner prompt.
pub fn tool_args_schema<T: schemars::JsonSchema>() -> serde_json::Value {
    let mut generator = schemars::generate::SchemaSettings::default().into_generator();
    let schema = generator.root_schema_for::<T>();
    let mut value = serde_json::to_value(schema).unwrap_or(serde_json::Value::Null);
    strip_schema_noise(&mut value);
    value
}

/// `JsonSchema` emits `title` as the Rust type name — internal noise.
fn strip_schema_noise(value: &mut serde_json::Value) {
    if let serde_json::Value::Object(map) = value {
        map.remove("$schema");
        map.remove("title");
        for child in map.values_mut() {
            strip_schema_noise(child);
        }
    } else if let serde_json::Value::Array(arr) = value {
        for child in arr.iter_mut() {
            strip_schema_noise(child);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args_schema: Option<serde_json::Value>,
    #[serde(default)]
    pub is_pure_read: bool,
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
                args_schema: tool.args_schema(),
                is_pure_read: tool.is_pure_read(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_default_keeps_base_under_pressure_threshold() {
        let ctx = ToolContext::with_cwd(".").with_turn(3);
        assert_eq!(ctx.scaled_default(200, 60), 200);
    }

    #[test]
    fn scaled_default_shrinks_with_turn_growth() {
        let ctx5 = ToolContext::with_cwd(".").with_turn(5);
        let ctx12 = ToolContext::with_cwd(".").with_turn(12);
        let ctx30 = ToolContext::with_cwd(".").with_turn(30);
        assert_eq!(ctx5.scaled_default(200, 60), 100);
        assert_eq!(ctx12.scaled_default(200, 60), 66);
        assert_eq!(ctx30.scaled_default(200, 60), 60); // floor reached
    }

    #[test]
    fn scaled_default_respects_floor_only_when_below_base() {
        let ctx = ToolContext::with_cwd(".").with_turn(100);
        assert_eq!(ctx.scaled_default(16_000, 4_000), 4_000);
    }

    #[test]
    fn current_turn_zero_acts_as_disabled() {
        let ctx = ToolContext::with_cwd(".");
        assert_eq!(ctx.current_turn, 0);
        assert_eq!(ctx.scaled_default(200, 60), 200);
    }
}
