//! Subagent surface: spawn (single + map), nudge (file-based intervention),
//! and the file-protocol primitives the parent and child use to talk.
//!
//! The split exists so the large lib.rs (~3000 LOC) does not keep collecting
//! every subagent-shaped concern in the same file.

use agent_core::{Tool, ToolCall, ToolContext, ToolError, ToolResult};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::env;
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;
use wait_timeout::ChildExt;

use crate::{find_latest_session, simple_id, truncate_text};

pub const SEED_SUBAGENT_DEPTH_ENV: &str = "SEED_SUBAGENT_DEPTH";
pub const SEED_SUBAGENT_MAX_DEPTH: u32 = 3;
pub const SEED_SUBAGENT_WATCH_DIR_ENV: &str = "SEED_SUBAGENT_WATCH_DIR";

pub const SUBAGENT_SIGNAL_KEYINFO: &str = "_keyinfo";
pub const SUBAGENT_SIGNAL_INTERVENE: &str = "_intervene";
pub const SUBAGENT_SIGNAL_STOP: &str = "_stop";

#[derive(Debug, Default, Clone)]
pub struct SubagentSignals {
    pub key_info: Vec<String>,
    pub intervene: Option<String>,
    pub stop: bool,
}

impl SubagentSignals {
    pub fn is_empty(&self) -> bool {
        self.key_info.is_empty() && self.intervene.is_none() && !self.stop
    }
}

/// Read and atomically consume `_keyinfo` / `_intervene` / `_stop` files from
/// the watch dir. Each file is deleted after being read so the same signal is
/// never re-applied. Missing watch dir or missing files yield empty signals.
pub fn consume_subagent_signals(watch_dir: &Path) -> SubagentSignals {
    let mut signals = SubagentSignals::default();
    if !watch_dir.is_dir() {
        return signals;
    }
    for (name, slot) in [
        (SUBAGENT_SIGNAL_KEYINFO, 0),
        (SUBAGENT_SIGNAL_INTERVENE, 1),
    ] {
        let path = watch_dir.join(name);
        if let Ok(content) = fs::read_to_string(&path) {
            let trimmed = content.trim().to_string();
            let _ = fs::remove_file(&path);
            if trimmed.is_empty() {
                continue;
            }
            match slot {
                0 => signals.key_info.push(trimmed),
                1 => signals.intervene = Some(trimmed),
                _ => {}
            }
        }
    }
    let stop_path = watch_dir.join(SUBAGENT_SIGNAL_STOP);
    if stop_path.exists() {
        let _ = fs::remove_file(&stop_path);
        signals.stop = true;
    }
    signals
}

/// Write any subset of subagent signals into the watch dir. Used by the parent
/// `subagent_nudge` tool.
pub fn write_subagent_signals(
    watch_dir: &Path,
    key_info: Option<&str>,
    intervene: Option<&str>,
    stop: bool,
) -> std::io::Result<()> {
    fs::create_dir_all(watch_dir)?;
    if let Some(text) = key_info {
        fs::write(watch_dir.join(SUBAGENT_SIGNAL_KEYINFO), text)?;
    }
    if let Some(text) = intervene {
        fs::write(watch_dir.join(SUBAGENT_SIGNAL_INTERVENE), text)?;
    }
    if stop {
        fs::write(watch_dir.join(SUBAGENT_SIGNAL_STOP), "")?;
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
pub(crate) struct SpawnSubagentArgs {
    pub task: String,
    #[serde(default)]
    pub context_files: Vec<String>,
    #[serde(default)]
    pub max_turns: Option<usize>,
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    /// When true the child uses the parent's cwd → shares memory/skills/L4 with parent.
    /// Default false: child gets its own subagent/<uuid>/ cwd so writes don't race
    /// with the parent's memory layer.
    #[serde(default)]
    pub inherit_memory: Option<bool>,
}

pub struct SpawnSubagentTool;

impl Tool for SpawnSubagentTool {
    fn name(&self) -> &'static str {
        "spawn_subagent"
    }

    fn description(&self) -> &'static str {
        "Delegate an isolated sub-task to a child `seed` process. By default the child runs in its OWN cwd (subagent/<uuid>/) with its own memory/skills/sessions so writes never race with the parent's L4 archive. Pass inherit_memory=true ONLY when the sub-task explicitly needs to read or extend the parent's memory/skill tree. Args: task (string), context_files (optional list of absolute paths to read first), max_turns (default 8), timeout_secs (default 600), provider, model, inherit_memory (default false). Returns the child's final answer plus session_path and subagent_root."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: SpawnSubagentArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        match spawn_one_subagent(&ctx.cwd, &ctx.skills_dir, &args) {
            Ok(outcome) => Ok(ToolResult::ok(call, subagent_result_json(&outcome))),
            Err(SpawnError::Refused(msg)) | Err(SpawnError::Failed(msg)) => {
                Ok(ToolResult::error(call, msg))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SubagentOutcome {
    pub task: String,
    pub answer: String,
    pub session_path: Option<PathBuf>,
    pub subagent_root: PathBuf,
    pub child_cwd: PathBuf,
    pub child_skills_dir: PathBuf,
    pub inherit_memory: bool,
    pub depth: u32,
    pub exit_code: i32,
    pub elapsed_secs: f64,
}

pub(crate) fn subagent_result_json(outcome: &SubagentOutcome) -> Value {
    json!({
        "status": "success",
        "task": outcome.task,
        "answer": outcome.answer,
        "session_path": outcome.session_path,
        "subagent_root": outcome.subagent_root,
        "child_cwd": outcome.child_cwd,
        "child_skills_dir": outcome.child_skills_dir,
        "inherit_memory": outcome.inherit_memory,
        "depth": outcome.depth,
        "exit_code": outcome.exit_code,
        "elapsed_secs": outcome.elapsed_secs,
    })
}

#[derive(Debug)]
pub(crate) enum SpawnError {
    Refused(String),
    Failed(String),
}

pub(crate) fn spawn_one_subagent(
    parent_cwd: &Path,
    parent_skills_dir: &Path,
    args: &SpawnSubagentArgs,
) -> Result<SubagentOutcome, SpawnError> {
    let depth: u32 = env::var(SEED_SUBAGENT_DEPTH_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if depth >= SEED_SUBAGENT_MAX_DEPTH {
        return Err(SpawnError::Refused(format!(
            "spawn_subagent depth {depth} hit limit {SEED_SUBAGENT_MAX_DEPTH}; refusing to recurse"
        )));
    }

    let max_turns = args.max_turns.unwrap_or(8).clamp(1, 32);
    let timeout = Duration::from_secs(args.timeout_secs.unwrap_or(600).clamp(10, 3600));
    let inherit_memory = args.inherit_memory.unwrap_or(false);

    let id = simple_id();
    let subagent_root = parent_cwd.join("subagent").join(&id);
    let sessions_dir = subagent_root.join("sessions");
    fs::create_dir_all(&sessions_dir).map_err(|err| SpawnError::Failed(err.to_string()))?;

    let (child_cwd, child_skills_dir) = if inherit_memory {
        (parent_cwd.to_path_buf(), parent_skills_dir.to_path_buf())
    } else {
        let dir = subagent_root.clone();
        fs::create_dir_all(dir.join("memory"))
            .map_err(|err| SpawnError::Failed(err.to_string()))?;
        fs::create_dir_all(dir.join("skills"))
            .map_err(|err| SpawnError::Failed(err.to_string()))?;
        (dir.clone(), dir.join("skills"))
    };

    let context_json = json!({
        "task": args.task,
        "work_dir": subagent_root,
        "input_files": args.context_files,
        "depth": depth + 1,
        "parent_cwd": parent_cwd,
        "child_cwd": child_cwd,
        "child_skills_dir": child_skills_dir,
        "inherit_memory": inherit_memory,
    });
    let context_path = subagent_root.join("context.json");
    fs::write(
        &context_path,
        serde_json::to_vec_pretty(&context_json).unwrap(),
    )
    .map_err(|err| SpawnError::Failed(err.to_string()))?;

    let seed_exe = env::current_exe()
        .map_err(|err| SpawnError::Failed(format!("locate seed binary: {err}")))?;

    let prompt = if args.context_files.is_empty() {
        args.task.clone()
    } else {
        let mut prompt = args.task.clone();
        prompt.push_str("\n\n[CONTEXT FILES — read these first]\n");
        for path in &args.context_files {
            prompt.push_str("- ");
            prompt.push_str(path);
            prompt.push('\n');
        }
        prompt
    };

    let mut command = Command::new(&seed_exe);
    command
        .arg("--sessions-dir")
        .arg(&sessions_dir)
        .arg("--skills-dir")
        .arg(&child_skills_dir)
        .arg("run")
        .arg(&prompt)
        .arg("--llm")
        .arg("--max-turns")
        .arg(max_turns.to_string())
        .arg("--cwd")
        .arg(&child_cwd)
        .env(SEED_SUBAGENT_DEPTH_ENV, (depth + 1).to_string())
        .env(SEED_SUBAGENT_WATCH_DIR_ENV, &subagent_root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());

    if let Some(provider) = &args.provider {
        command.arg("--provider").arg(provider);
    }
    if let Some(model) = &args.model {
        command.arg("--model").arg(model);
    }

    let started = std::time::Instant::now();
    let mut child = command
        .spawn()
        .map_err(|err| SpawnError::Failed(format!("spawn child seed: {err}")))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| SpawnError::Failed("child stdout not captured".to_string()))?;
    let stdout_handle = thread::spawn(move || {
        let mut buf = String::new();
        let mut reader = BufReader::new(stdout);
        let _ = reader.read_to_string(&mut buf);
        buf
    });

    let status = match child
        .wait_timeout(timeout)
        .map_err(|err| SpawnError::Failed(err.to_string()))?
    {
        Some(status) => status,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_handle.join();
            return Err(SpawnError::Failed(format!(
                "subagent timed out after {}s",
                timeout.as_secs()
            )));
        }
    };
    let stdout_text = stdout_handle.join().unwrap_or_default();
    let exit_code = status.code().unwrap_or(-1);
    if !status.success() {
        return Err(SpawnError::Failed(format!(
            "subagent exited with code {exit_code}; tail: {}",
            truncate_text(&stdout_text, 800)
        )));
    }

    Ok(SubagentOutcome {
        task: args.task.clone(),
        answer: stdout_text.trim().to_string(),
        session_path: find_latest_session(&sessions_dir),
        subagent_root,
        child_cwd,
        child_skills_dir,
        inherit_memory,
        depth: depth + 1,
        exit_code,
        elapsed_secs: started.elapsed().as_secs_f64(),
    })
}

#[derive(Debug, Deserialize)]
pub(crate) struct SpawnSubagentMapArgs {
    pub tasks: Vec<SpawnSubagentArgs>,
    #[serde(default)]
    pub max_parallel: Option<usize>,
}

pub struct SpawnSubagentMapTool;

impl Tool for SpawnSubagentMapTool {
    fn name(&self) -> &'static str {
        "spawn_subagent_map"
    }

    fn description(&self) -> &'static str {
        "Fan out N independent sub-tasks to parallel child seed processes (each isolated by default). Use when several tasks operate on disjoint inputs and produce independent outputs. Args: tasks (array of spawn_subagent argument objects), max_parallel (default 4, capped at 8). Returns an array of per-task outcomes with status/answer/session_path/elapsed_secs. Sequential `spawn_subagent` is simpler when tasks share state or depend on each other."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: SpawnSubagentMapArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        if args.tasks.is_empty() {
            return Ok(ToolResult::error(call, "tasks array must not be empty"));
        }
        let max_parallel = args.max_parallel.unwrap_or(4).clamp(1, 8);
        let total = args.tasks.len();

        let parent_cwd = ctx.cwd.clone();
        let parent_skills = ctx.skills_dir.clone();
        let queue = std::sync::Mutex::new(args.tasks.into_iter().enumerate().collect::<Vec<_>>());
        let results: std::sync::Mutex<Vec<(usize, Value)>> = std::sync::Mutex::new(Vec::new());

        thread::scope(|scope| {
            for _ in 0..max_parallel.min(total) {
                scope.spawn(|| {
                    loop {
                        let next = {
                            let mut q = queue.lock().unwrap();
                            if q.is_empty() {
                                None
                            } else {
                                Some(q.remove(0))
                            }
                        };
                        let Some((idx, task_args)) = next else {
                            break;
                        };
                        let entry = match spawn_one_subagent(&parent_cwd, &parent_skills, &task_args) {
                            Ok(outcome) => {
                                let mut value = subagent_result_json(&outcome);
                                if let Some(obj) = value.as_object_mut() {
                                    obj.insert("index".to_string(), Value::from(idx));
                                }
                                value
                            }
                            Err(SpawnError::Refused(msg)) | Err(SpawnError::Failed(msg)) => {
                                json!({
                                    "status": "error",
                                    "index": idx,
                                    "task": task_args.task,
                                    "message": msg,
                                })
                            }
                        };
                        results.lock().unwrap().push((idx, entry));
                    }
                });
            }
        });

        let mut collected = results.into_inner().unwrap();
        collected.sort_by_key(|(idx, _)| *idx);
        let ordered: Vec<Value> = collected.into_iter().map(|(_, value)| value).collect();
        let ok_count = ordered
            .iter()
            .filter(|value| value["status"] == "success")
            .count();
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "completed",
                "total": total,
                "succeeded": ok_count,
                "failed": total - ok_count,
                "max_parallel": max_parallel,
                "results": ordered,
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct SubagentNudgeArgs {
    #[serde(alias = "watch_dir", alias = "dir", alias = "subagent_root")]
    pub target: PathBuf,
    #[serde(default)]
    pub key_info: Option<String>,
    #[serde(default)]
    pub intervene: Option<String>,
    #[serde(default)]
    pub stop: Option<bool>,
}

pub struct SubagentNudgeTool;

impl Tool for SubagentNudgeTool {
    fn name(&self) -> &'static str {
        "subagent_nudge"
    }

    fn description(&self) -> &'static str {
        "Send a non-blocking signal to a running subagent so the parent can steer it without killing the run. Args: target (subagent_root path returned by spawn_subagent), key_info (text appended to the child's working memory at its next turn), intervene (one-shot system message prepended to the child's next prompt), stop (bool — child finishes at end of its current turn). At least one of key_info/intervene/stop must be set."
    }

    fn execute(&self, _ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: SubagentNudgeArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let stop = args.stop.unwrap_or(false);
        if args.key_info.is_none() && args.intervene.is_none() && !stop {
            return Ok(ToolResult::error(
                call,
                "at least one of key_info / intervene / stop must be provided",
            ));
        }
        if !args.target.is_dir() {
            return Ok(ToolResult::error(
                call,
                format!("watch dir not found: {}", args.target.display()),
            ));
        }
        write_subagent_signals(
            &args.target,
            args.key_info.as_deref(),
            args.intervene.as_deref(),
            stop,
        )
        .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "target": args.target,
                "signals": {
                    "key_info": args.key_info.is_some(),
                    "intervene": args.intervene.is_some(),
                    "stop": stop,
                },
            }),
        ))
    }
}
