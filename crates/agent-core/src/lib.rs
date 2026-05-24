use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Whether a run is allowed to mutate the project.
///
/// Set by `run_goal` early (after consulting the explicit `--mode` flag and
/// then `agent_runtime::classify_run_mode`) and pushed into the
/// `agent_tools::run_mode_guard` process-singleton so individual tools
/// (notably `ShellTool`, RF27-2) can refuse write-shaped operations when
/// they shouldn't be running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    /// Tool catalog is pared down to discovery-only tools; `run_shell`
    /// rejects commands that look like writes. Set automatically when the
    /// goal text matches the analyze/summarize/explain shape.
    ReadOnly,
    /// Full tool catalog. The default for any goal that doesn't classify
    /// as read-only, and the catch-all for `--mode write` overrides.
    #[default]
    Implementation,
}

/// Records how the active `RunMode` was chosen, so session JSONLs and the
/// trace header can distinguish "auto-classified from goal keywords" from
/// "user pinned it explicitly". Pure provenance — does not affect behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModeSource {
    /// Came from `agent_runtime::classify_run_mode(goal)`.
    #[default]
    Auto,
    /// User-set via `--mode read|write` on the CLI, or `/mode` in the REPL.
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
    /// Current planner turn (1-indexed) when invoked from the runtime loop.
    /// Used by tools to scale default output limits so working memory does not
    /// blow up on long runs. `0` means "not in a loop" — tools use their
    /// natural defaults.
    pub current_turn: usize,
}

impl ToolContext {
    pub fn new(cwd: impl Into<PathBuf>, skills_dir: impl Into<PathBuf>) -> Self {
        let cwd = cwd.into();
        Self {
            memory_dir: cwd.join("memory"),
            sessions_dir: cwd.join("sessions"),
            cwd,
            skills_dir: skills_dir.into(),
            current_turn: 0,
        }
    }

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

    /// Returns a soft default cap scaled by current turn pressure.
    /// `base` is the natural default for the tool (count, bytes, etc.).
    /// `floor` caps how small it can shrink.
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
        /// RF27-1: the run mode this goal is operating under. `#[serde(default)]`
        /// so older session JSONLs (no `mode` field) deserialize cleanly as
        /// `Implementation` (the historical behavior — there was no read-only
        /// gating before).
        #[serde(default)]
        mode: RunMode,
        /// Whether `mode` was auto-classified or explicitly pinned.
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
        /// Output char count from the planner's response (approximation for
        /// output tokens when the provider doesn't report `tokens` directly).
        planner_chars: usize,
        /// RF36-1: input char count of the assembled planner prompt for this
        /// turn (system + user + memory + tool catalog). `serde(default)` so
        /// older session JSONLs read back as `0`.
        #[serde(default)]
        prompt_chars: usize,
    },
    RunFinished {
        status: String,
        summary: String,
    },
    /// RF34-2: emitted when the planner's parse/transport retry path fires.
    /// Lets sessions show how often (and why) the runtime had to back off,
    /// without making the planner itself responsible for that observability.
    PlannerRetry {
        turn: usize,
        /// Attempt index, 1-based. `attempt=1` means "first retry"
        /// (the original call already failed).
        attempt: usize,
        /// Total attempts the runtime is willing to make for this kind.
        of: usize,
        /// Backoff in ms applied before this retry (0 for parse-retries
        /// which retry immediately without sleeping).
        backoff_ms: u64,
        /// `"parse"` for `InvalidPlannerJson` retries, `"transport"` for
        /// `Planner(_)` retries (network/stdio blip).
        kind: String,
        /// Short error string for trace context.
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
        // floor=4_000, base=16_000 → divisor=21 → 16000/21 = 761; max(floor.min(base)=4000) = 4000
        assert_eq!(ctx.scaled_default(16_000, 4_000), 4_000);
    }

    #[test]
    fn current_turn_zero_acts_as_disabled() {
        let ctx = ToolContext::with_cwd(".");
        assert_eq!(ctx.current_turn, 0);
        assert_eq!(ctx.scaled_default(200, 60), 200);
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
