use agent_core::{ToolCall, ToolContext, ToolError, ToolRegistry};
#[cfg(test)]
use agent_core::{RunMode, Tool};
use serde_json::{Value, json};
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

/// Cheap, non-destructive fixups for common planner deviations:
/// string-encoded JSON object, `null` args, `{ "args": {...} }` envelope.
/// When repair fails the normal deserialize error flows through.
pub fn repair_tool_args(value: serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::String(s) => {
            let trimmed = s.trim();
            if (trimmed.starts_with('{') && trimmed.ends_with('}'))
                || (trimmed.starts_with('[') && trimmed.ends_with(']'))
            {
                if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
                    return repair_tool_args(parsed);
                }
            }
            Value::String(s)
        }
        Value::Null => Value::Object(serde_json::Map::new()),
        Value::Object(map) if map.len() == 1 && map.contains_key("args") => {
            let mut map = map;
            let inner = map.remove("args").unwrap_or(Value::Null);
            repair_tool_args(inner)
        }
        other => other,
    }
}

pub fn parse_tool_args<T: serde::de::DeserializeOwned>(
    call: &ToolCall,
) -> Result<T, ToolError> {
    let repaired = repair_tool_args(call.args.clone());
    serde_json::from_value(repaired).map_err(|source| ToolError::InvalidArguments {
        tool: call.name.clone(),
        source,
    })
}

mod subagent;
pub use subagent::{
    SEED_SUBAGENT_DEPTH_ENV, SEED_SUBAGENT_MAX_DEPTH, SEED_SUBAGENT_WATCH_DIR_ENV,
    SUBAGENT_SIGNAL_INTERVENE, SUBAGENT_SIGNAL_KEYINFO, SUBAGENT_SIGNAL_STOP, SpawnSubagentMapTool,
    SpawnSubagentTool, SubagentNudgeTool, SubagentSignals, consume_subagent_signals,
    write_subagent_signals,
};


mod sync;
pub use sync::{skill_tools_guard, run_mode_guard, repoprompt_sync};

mod walk;
pub use walk::{WalkOptions, WalkResult, walk_workspace};

/// Wires `Tool::description()` to read
/// `crates/agent-tools/descriptions/<name>.md` at compile time.
///
/// ```rust,ignore
/// impl Tool for MyTool {
///     fn name(&self) -> &'static str { "my_tool" }
///     tool_description!("my_tool");
/// }
/// ```
///
/// **Convention**: the `.md` file must NOT end with a newline (we can't `trim_end()`
/// at compile time. A trailing newline would leak into the planner
/// prompt and waste a token per tool. A test in `lib.rs` verifies all
/// 32 description files are newline-free.
///
/// **Why a macro, not a proc-macro**: forge uses a proc-macro for
/// this (`#[tool_description_file = "..."]`). Ours is a `macro_rules!`
/// + `include_str!` because we don't need the auto-derived
/// `ToolDescription` trait that forge ships — just the file-on-disk
/// editability. Zero new deps, same authoring ergonomics.
#[macro_export]
macro_rules! tool_description {
    ($name:literal) => {
        fn description(&self) -> &'static str {
            include_str!(concat!("../descriptions/", $name, ".md"))
        }
    };
}

/// declare a tool as a pure-read in one line. Usage:
///
/// ```rust,ignore
/// impl Tool for ReadFileTool {
///     fn name(&self) -> &'static str { "read_file" }
///     crate::tool_description!("read_file");
///     crate::impl_args_schema!(ReadFileArgs);
///     crate::impl_pure_read!();  // ← read tool opt-in
///     fn execute(&self, ctx, call) -> ... { ... }
/// }
/// ```
///
/// Only add to tools whose `execute` is pure within a run — same
/// inputs ⇒ same output. Side-effecting tools must NOT use this
/// (would cause "we cached your write" bugs in the memoization layer).
#[macro_export]
macro_rules! impl_pure_read {
    () => {
        fn is_pure_read(&self) -> bool {
            true
        }
    };
}

/// `MyArgs` must `#[derive(schemars::JsonSchema)]`. Hides the
/// `Some(..)` + `tool_args_schema::<T>()` boilerplate.
#[macro_export]
macro_rules! impl_args_schema {
    ($t:ty) => {
        fn args_schema(&self) -> ::std::option::Option<::serde_json::Value> {
            ::std::option::Option::Some(::agent_core::tool_args_schema::<$t>())
        }
    };
}

mod files;
pub use files::{PatchFileTool, ReadFileTool, ReadFilesTool, WriteFileTool};

mod memory_protocol;
pub use memory_protocol::{CompleteLongTermUpdateTool, LongTermUpdateTool, WorkingCheckpointTool};

mod skills;
pub use skills::{SkillFetchTool, SkillListTool, SkillSearchTool};

mod ask_user;
mod memory;
mod repoprompt_bridge;
mod shell;
mod tool_describe;
pub use ask_user::AskUserTool;
pub use memory::{MemoryFetchTool, MemorySearchTool};
pub use repoprompt_bridge::{RepoPromptCallTool, RepoPromptExecTool, RepoPromptToolsTool};
pub use shell::{ShellIntent, ShellTool, shell_command_intent};
pub use tool_describe::ToolDescribeTool;
pub(crate) use repoprompt_bridge::RepoPromptRoutingArgs;

mod plan;
pub use plan::{
    PlanCompleteTool, PlanCreateFromRepoPromptTool, PlanCreateTool, PlanCreateViaRepoPromptTool,
    PlanListTool, PlanNextTool, PlanRecordArtifactTool, PlanRecordHandoffTool,
    PlanRefineViaRepoPromptTool, PlanStatusTool, PlanVerifyTool,
};

/// Built once per process — tools are immutable singletons.
pub fn seed_registry() -> &'static ToolRegistry {
    static REGISTRY: std::sync::OnceLock<ToolRegistry> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(build_seed_registry)
}

fn build_seed_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(MemorySearchTool);
    registry.register(MemoryFetchTool);
    registry.register(SkillListTool);
    registry.register(SkillSearchTool);
    registry.register(SkillFetchTool);
    registry.register(PlanCreateTool);
    registry.register(PlanCreateFromRepoPromptTool);
    registry.register(PlanCreateViaRepoPromptTool);
    registry.register(PlanRefineViaRepoPromptTool);
    registry.register(PlanListTool);
    registry.register(PlanStatusTool);
    registry.register(PlanNextTool);
    registry.register(PlanCompleteTool);
    registry.register(PlanRecordArtifactTool);
    registry.register(PlanRecordHandoffTool);
    registry.register(PlanVerifyTool);
    registry.register(RepoPromptToolsTool);
    registry.register(RepoPromptExecTool);
    registry.register(RepoPromptCallTool);
    registry.register(ReadFileTool);
    registry.register(ReadFilesTool);
    registry.register(PatchFileTool);
    registry.register(WriteFileTool);
    registry.register(ShellTool);
    registry.register(WorkingCheckpointTool);
    registry.register(LongTermUpdateTool);
    registry.register(CompleteLongTermUpdateTool);
    registry.register(SpawnSubagentTool);
    registry.register(SpawnSubagentMapTool);
    registry.register(SubagentNudgeTool);
    registry.register(AskUserTool);
    registry.register(ToolDescribeTool);
    registry
}

/// Tagged set of "mutation flavors" passed to [`durable_write_guard`]
/// to drive overwrite-vs-additive policy. Lives in `lib.rs` (not
/// `files.rs`) because the guard function itself stays here as a shared
/// helper used by both `files.rs` and the memory_protocol region. The
/// `From<WriteMode>` impl that maps `files.rs`'s public-shape enum into
/// this internal one lives next to `WriteMode` in `files.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DurableWriteMode {
    Patch,
    Overwrite,
    Append,
    Prepend,
}

pub(crate) fn durable_write_guard(
    ctx: &ToolContext,
    path: &Path,
    new_text: &str,
    mode: DurableWriteMode,
    existing_nonempty: bool,
) -> Option<String> {
    let target = durable_target(ctx, path)?;
    if is_generated_memory_file(ctx, path) {
        return Some(format!(
            "{} is generated L1/index state; update L2/L3 memory and rebuild the index instead",
            path.display()
        ));
    }
    if mode == DurableWriteMode::Overwrite && existing_nonempty {
        return Some(format!(
            "durable {target} already exists; use patch_file for smallest local edits or append/prepend for additive notes"
        ));
    }
    let violations = agent_memory::durable_memory_violations(new_text);
    if violations.is_empty() {
        None
    } else {
        Some(format!(
            "durable {target} guardrail blocked write to {}: {}",
            path.display(),
            violations.join("; ")
        ))
    }
}

fn durable_target(ctx: &ToolContext, path: &Path) -> Option<&'static str> {
    if path_starts_with(path, &ctx.memory_dir) {
        Some("memory")
    } else if path_starts_with(path, &ctx.skills_dir) {
        Some("skill")
    } else {
        None
    }
}

pub(crate) fn is_durable_path(ctx: &ToolContext, path: &Path) -> bool {
    durable_target(ctx, path).is_some()
}

fn is_generated_memory_file(ctx: &ToolContext, path: &Path) -> bool {
    let path = clean_path(path);
    let memory_dir = clean_path(&ctx.memory_dir);
    path == memory_dir.join("index.json") || path == memory_dir.join("l1_insight.md")
}

fn path_starts_with(path: &Path, base: &Path) -> bool {
    clean_path(path).starts_with(clean_path(base))
}

fn clean_path(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let mut out = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::RootDir | Component::Prefix(_) | Component::Normal(_) => {
                out.push(component.as_os_str());
            }
        }
    }
    out
}

pub(crate) fn resolve_path(cwd: &Path, input: &str) -> PathBuf {
    let path = Path::new(input);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

pub(crate) const REPOPROMPT_LEDGER_PROMPT: &str = "REPOPROMPT_LEDGER: if RepoPrompt produces an export path, call plan_record_artifact before using it; if RepoPrompt agent_run or Codex performs plan work, call plan_record_handoff with backend, role/run/thread id when known, artifact_path, status, and summary.";

pub(crate) fn absolutize(cwd: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        absolute_base(cwd).join(path)
    }
}

fn absolute_base(cwd: &Path) -> PathBuf {
    if cwd.is_absolute() {
        cwd.to_path_buf()
    } else {
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(cwd)
    }
}

fn default_repoprompt_working_dirs(
    ctx: &ToolContext,
    working_dirs: Option<Vec<PathBuf>>,
    default_cwd: bool,
) -> Vec<PathBuf> {
    let mut working_dirs = working_dirs.unwrap_or_default();
    if working_dirs.is_empty() && default_cwd {
        // Skill-queued override consumed here so bindings stay transient.
        if let Some(override_dirs) = repoprompt_sync::take_pending_override() {
            working_dirs = override_dirs;
        } else {
            working_dirs.push(ctx.cwd.clone());
        }
    }
    working_dirs
        .into_iter()
        .map(|path| absolutize(&ctx.cwd, path))
        .collect()
}

pub(crate) fn default_cwd_for_repoprompt_exec(command: &str) -> bool {
    let command = command.trim().to_ascii_lowercase();
    !(command == "windows"
        || command.starts_with("windows ")
        || command == "workspace list"
        || command.starts_with("workspace list ")
        || command == "workspaces"
        || command.starts_with("bind_context")
        || command.starts_with("app_settings"))
}

pub(crate) fn default_cwd_for_repoprompt_tool(tool: agent_repoprompt::RepoPromptTool) -> bool {
    !matches!(
        tool,
        agent_repoprompt::RepoPromptTool::BindContext
            | agent_repoprompt::RepoPromptTool::ManageWorkspaces
            | agent_repoprompt::RepoPromptTool::AppSettings
            | agent_repoprompt::RepoPromptTool::OracleUtils
            | agent_repoprompt::RepoPromptTool::AgentManage
    )
}

pub(crate) fn repoprompt_report_text(output: &agent_repoprompt::RepoPromptOutput) -> String {
    if !output.stdout.trim().is_empty() {
        return output.stdout.clone();
    }
    if let Some(json) = &output.json {
        return serde_json::to_string_pretty(json).unwrap_or_else(|_| json.to_string());
    }
    output.stderr.clone()
}

pub(crate) fn repoprompt_client(
    ctx: &ToolContext,
    routing: RepoPromptRoutingArgs,
    default_cwd: bool,
) -> Result<agent_repoprompt::RepoPromptClient, ToolError> {
    let mut cfg = repoprompt_config(ctx, routing, default_cwd);
    // if we have a cached window_id for this working_dirs set,
    // pre-set it before resolve_repoprompt_window runs. That makes the
    // resolver short-circuit (it returns early when window_id is Some)
    // and skips a ~70ms bind_context CLI call. The cache is populated on
    // every successful resolve below, and invalidated by /cd, by skill
    // overrides, and by run_goal's reset_sync_state.
    if cfg.window_id.is_none()
        && !cfg.working_dirs.is_empty()
        && cfg.context_id.is_none()
    {
        if let Some(cached) = repoprompt_sync::cached_window_id_for(&cfg.working_dirs) {
            cfg.window_id = Some(cached);
        }
    }
    let working_dirs_for_record = cfg.working_dirs.clone();
    resolve_repoprompt_window(&mut cfg)?;
    // If resolve produced a window_id (either freshly-bound or already
    // cached via our pre-set above), record it for next time. We avoid
    // re-recording the same `(working_dirs, window_id)` repeatedly because
    // `record_bound_window` overwrites unconditionally; the cost is just
    // a Mutex lock so it's not worth deduping here.
    if let Some(window_id) = cfg.window_id
        && !working_dirs_for_record.is_empty()
    {
        repoprompt_sync::record_bound_window(working_dirs_for_record, window_id);
    }
    Ok(agent_repoprompt::RepoPromptClient::new(cfg))
}

#[cfg(test)]
fn repoprompt_client_without_bind(
    ctx: &ToolContext,
    routing: RepoPromptRoutingArgs,
    default_cwd: bool,
) -> agent_repoprompt::RepoPromptClient {
    agent_repoprompt::RepoPromptClient::new(repoprompt_config(ctx, routing, default_cwd))
}

fn repoprompt_config(
    ctx: &ToolContext,
    routing: RepoPromptRoutingArgs,
    default_cwd: bool,
) -> agent_repoprompt::RepoPromptClientConfig {
    let mut cfg = agent_repoprompt::RepoPromptClientConfig::default();
    if let Some(cli_path) = routing.cli_path {
        cfg.cli_path = cli_path;
    }
    if let Some(timeout_secs) = routing.timeout_secs {
        cfg.timeout_secs = timeout_secs;
    }
    cfg.window_id = routing.window_id;
    cfg.tab = routing.tab;
    cfg.context_id = routing.context_id;
    cfg.working_dirs = default_repoprompt_working_dirs(ctx, routing.working_dirs, default_cwd);
    cfg.raw_json = routing.raw_json.unwrap_or(false);
    cfg
}

fn resolve_repoprompt_window(
    cfg: &mut agent_repoprompt::RepoPromptClientConfig,
) -> Result<(), ToolError> {
    if cfg.window_id.is_some() || cfg.context_id.is_some() || cfg.working_dirs.is_empty() {
        return Ok(());
    }
    let bind_cfg = agent_repoprompt::RepoPromptClientConfig {
        cli_path: cfg.cli_path.clone(),
        timeout_secs: cfg.timeout_secs,
        raw_json: true,
        ..Default::default()
    };
    // default to `create_if_missing: true` so an `seed`/agent
    // invocation in a cwd that RepoPrompt has never seen auto-registers
    // it as a workspace instead of failing the entire rp call. Live RP
    // CLI honors this — `bind_context op=bind working_dirs=[…] create_if_missing=true`
    let output = agent_repoprompt::RepoPromptClient::new(bind_cfg)
        .call_tool(
            agent_repoprompt::RepoPromptTool::BindContext,
            &json!({
                "op": "bind",
                "working_dirs": cfg
                    .working_dirs
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>(),
                "create_if_missing": true,
            }),
        )
        .map_err(|err| ToolError::Failed(format!("RepoPrompt bind failed: {err}")))?;
    if output.timed_out || output.exit_code != Some(0) {
        let raw = compact_single_line(&repoprompt_report_text(&output), 800);
        let friendly = humanize_rp_bind_failure(&raw);
        return Err(ToolError::Failed(format!(
            "RepoPrompt bind failed before routed call: {friendly} (raw: {raw})"
        )));
    }
    if let Some(window_id) = repoprompt_output_u32(&output, &["window_id", "windowID"]) {
        cfg.window_id = Some(window_id);
        return Ok(());
    }
    Err(ToolError::Failed(
        "RepoPrompt bind succeeded but did not return a window_id".to_string(),
    ))
}

/// Pattern-matches RP CLI raw error tails — unmatched falls through to
/// "see raw error tail" so info isn't dropped.
pub fn humanize_rp_bind_failure(raw: &str) -> &'static str {
    let lower = raw.to_ascii_lowercase();
    if lower.contains("no exact workspace match") || lower.contains("no exact match") {
        "RepoPrompt couldn't bind to this cwd and wouldn't auto-create. \
         Try `rp-cli -e 'bind_context op=bind working_dirs=[\"<path>\"] create_if_missing=true'` manually."
    } else if lower.contains("connection refused")
        || lower.contains("failed to connect")
        || lower.contains("no such process")
    {
        "RepoPrompt app isn't running — launch it (open RepoPrompt.app) or \
         run `repoprompt_cli --launch-app`, then retry."
    } else if lower.contains("multiple repoprompt windows")
        || lower.contains("disambiguate")
    {
        "RepoPrompt has multiple windows open and can't disambiguate. \
         Pass `window_id` explicitly via routing args or close extra RP windows."
    } else if lower.contains("permission denied") {
        "RepoPrompt refused the bind (permissions). Check filesystem ACLs \
         on the target dir, or try a different cwd."
    } else {
        "see raw error tail"
    }
}

pub(crate) fn repoprompt_output_json(output: agent_repoprompt::RepoPromptOutput) -> serde_json::Value {
    let (stdout_text, stdout_stats) = truncate_middle_with_stats(&output.stdout, 20_000);
    let (stderr_text, stderr_stats) = truncate_middle_with_stats(&output.stderr, 6_000);
    json!({
        "status": output.status(),
        "timed_out": output.timed_out,
        "exit_code": output.exit_code,
        "stdout": stdout_text,
        "stdout_truncated": stdout_stats.was_truncated,
        "stdout_original_bytes": stdout_stats.original_bytes,
        "stderr": stderr_text,
        "stderr_truncated": stderr_stats.was_truncated,
        "stderr_original_bytes": stderr_stats.original_bytes,
        "json": output.json,
    })
}

pub(crate) fn repoprompt_ledger_prompt_for_exec(command: &str) -> &'static str {
    let command = command.to_ascii_lowercase();
    if command.contains("builder")
        || command.contains("oracle")
        || command.contains("--export")
        || command.contains("workspace_context")
        || command.contains("context")
    {
        "REPOPROMPT_OUTPUT: if stdout/json includes an export path and a plan is active, call plan_record_artifact with kind context_export or repo_prompt_export before continuing."
    } else {
        REPOPROMPT_LEDGER_PROMPT
    }
}

pub(crate) fn repoprompt_ledger_prompt_for_tool(tool: agent_repoprompt::RepoPromptTool) -> &'static str {
    match tool {
        agent_repoprompt::RepoPromptTool::AgentRun => {
            "REPOPROMPT_AGENT_RUN: if this agent_run executed or verified plan work, call plan_record_handoff with backend=repoprompt, role/model, run/thread id when known, artifact_path when used, status, and summary."
        }
        agent_repoprompt::RepoPromptTool::ContextBuilder
        | agent_repoprompt::RepoPromptTool::OracleSend
        | agent_repoprompt::RepoPromptTool::WorkspaceContext
        | agent_repoprompt::RepoPromptTool::Prompt => {
            "REPOPROMPT_EXPORT: if the output includes oracle_export_path, context export path, or prompt export path and a plan is active, call plan_record_artifact before using it as handoff evidence."
        }
        _ => REPOPROMPT_LEDGER_PROMPT,
    }
}

pub(crate) fn attach_repoprompt_protocol_hint(
    content: &mut Value,
    next_prompt: &'static str,
    cfg: &agent_repoprompt::RepoPromptClientConfig,
) {
    if let Value::Object(map) = content {
        map.insert("next_prompt".to_string(), json!(next_prompt));
        map.insert(
            "routing".to_string(),
            json!({
                "window_id": cfg.window_id,
                "tab": cfg.tab,
                "context_id": cfg.context_id,
                "working_dirs": &cfg.working_dirs,
            }),
        );
    }
}

pub(crate) fn repoprompt_output_string(
    output: &agent_repoprompt::RepoPromptOutput,
    keys: &[&str],
) -> Option<String> {
    output
        .json
        .as_ref()
        .and_then(|json| find_string_by_key(json, keys))
}

fn repoprompt_output_u32(
    output: &agent_repoprompt::RepoPromptOutput,
    keys: &[&str],
) -> Option<u32> {
    output
        .json
        .as_ref()
        .and_then(|json| find_u32_by_key(json, keys))
}

fn find_string_by_key(value: &Value, keys: &[&str]) -> Option<String> {
    match value {
        Value::Object(map) => {
            for key in keys {
                if let Some(value) = map.get(*key).and_then(value_to_non_empty_string) {
                    return Some(value);
                }
            }
            map.values()
                .find_map(|value| find_string_by_key(value, keys))
        }
        Value::Array(values) => values
            .iter()
            .find_map(|value| find_string_by_key(value, keys)),
        _ => None,
    }
}

fn find_u32_by_key(value: &Value, keys: &[&str]) -> Option<u32> {
    match value {
        Value::Object(map) => {
            for key in keys {
                if let Some(value) = map.get(*key).and_then(value_to_u32) {
                    return Some(value);
                }
            }
            map.values().find_map(|value| find_u32_by_key(value, keys))
        }
        Value::Array(values) => values.iter().find_map(|value| find_u32_by_key(value, keys)),
        _ => None,
    }
}

fn value_to_non_empty_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) if !value.trim().is_empty() => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

fn value_to_u32(value: &Value) -> Option<u32> {
    match value {
        Value::Number(value) => value.as_u64().and_then(|value| u32::try_from(value).ok()),
        Value::String(value) => value.parse::<u32>().ok(),
        _ => None,
    }
}

pub(crate) fn compact_single_line(input: &str, max_len: usize) -> String {
    let text = input.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.chars().count() <= max_len {
        return text;
    }
    let keep = max_len / 2;
    let head = text.chars().take(keep).collect::<String>();
    let tail = text
        .chars()
        .rev()
        .take(keep)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("{head} ...[omitted]... {tail}")
}

pub(crate) fn non_empty(value: Option<String>) -> Option<String> {
    value
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

pub(crate) fn plan_title_from_task(task: &str) -> String {
    let mut title = task
        .lines()
        .next()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .unwrap_or("Untitled plan")
        .to_string();
    truncate_utf8(&mut title, 80);
    while title.ends_with(['-', ':', ' ', '，', '。']) {
        title.pop();
    }
    if title.is_empty() {
        "Untitled plan".to_string()
    } else {
        title
    }
}

fn truncate_middle(input: &str, max_len: usize) -> String {
    if input.len() <= max_len {
        return input.to_string();
    }
    let keep = max_len / 2;
    format!(
        "{}\n...[omitted long output]...\n{}",
        safe_prefix(input, keep),
        safe_suffix(input, keep)
    )
}

/// Tool results that surface `*_truncated` + `*_original_bytes` fields
/// let the planner reason about whether the observation is complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncationStats {
    pub was_truncated: bool,
    pub original_bytes: usize,
}

pub fn truncate_middle_with_stats(input: &str, max_len: usize) -> (String, TruncationStats) {
    let original_bytes = input.len();
    if original_bytes <= max_len {
        return (
            input.to_string(),
            TruncationStats {
                was_truncated: false,
                original_bytes,
            },
        );
    }
    (
        truncate_middle(input, max_len),
        TruncationStats {
            was_truncated: true,
            original_bytes,
        },
    )
}

pub(crate) fn truncate_utf8(text: &mut String, limit: usize) {
    if text.len() <= limit {
        return;
    }
    let mut end = limit;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
}

fn safe_prefix(text: &str, limit: usize) -> &str {
    let mut end = limit.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn safe_suffix(text: &str, limit: usize) -> &str {
    let mut start = text.len().saturating_sub(limit);
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}


pub(crate) fn simple_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("{nanos:x}")
}

pub(crate) fn find_latest_session(dir: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut latest: Option<(SystemTime, PathBuf)> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "jsonl")
            && let Ok(meta) = entry.metadata()
            && let Ok(mtime) = meta.modified()
        {
            if latest.as_ref().is_none_or(|(t, _)| mtime > *t) {
                latest = Some((mtime, path));
            }
        }
    }
    latest.map(|(_, path)| path)
}

pub(crate) fn truncate_text(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        let head: String = text.chars().take(max).collect();
        format!("{head}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn registry_exposes_genericagent_memory_tools() {
        let registry = seed_registry();
        let names = registry.names();
        assert!(names.contains(&"update_working_checkpoint"));
        assert!(names.contains(&"start_long_term_update"));
        assert!(names.contains(&"complete_long_term_update"));
        assert!(names.contains(&"memory_search"));
        assert!(names.contains(&"memory_fetch"));
        assert!(names.contains(&"plan_create"));
        assert!(names.contains(&"plan_list"));
        assert!(names.contains(&"plan_status"));
        assert!(names.contains(&"plan_next"));
        assert!(names.contains(&"plan_complete"));
        assert!(names.contains(&"plan_record_artifact"));
        assert!(names.contains(&"plan_record_handoff"));
        assert!(names.contains(&"plan_verify"));
        assert!(names.contains(&"repoprompt_tools"));
        assert!(names.contains(&"repoprompt_exec"));
        assert!(names.contains(&"repoprompt_call"));
        assert!(names.contains(&"spawn_subagent"));
        assert!(names.contains(&"ask_user"));
    }

    #[test]
    fn escape_xml_protects_angle_brackets_and_amps() {
        let escaped = crate::plan::escape_xml("3 < 5 && x > y");
        assert_eq!(escaped, "3 &lt; 5 &amp;&amp; x &gt; y");
    }

    #[test]
    fn registry_exposes_plan_create_via_repoprompt() {
        let registry = seed_registry();
        let names = registry.names();
        assert!(names.contains(&"plan_create_via_repoprompt"));
        assert!(names.contains(&"plan_refine_via_repoprompt"));
    }

    // Empty .md files compile (missing files are build errors). A stray
    // trailing newline would leak into the planner prompt.
    #[test]
    fn every_tool_has_a_clean_description() {
        let registry = seed_registry();
        for info in registry.infos() {
            assert!(
                !info.description.is_empty(),
                "tool `{}` has an empty description (check descriptions/{}.md)",
                info.name,
                info.name
            );
            assert_eq!(
                info.description,
                info.description.trim_end(),
                "tool `{}` description has trailing whitespace — remove the trailing newline from descriptions/{}.md",
                info.name,
                info.name
            );
            assert!(
                info.description.len() >= 20,
                "tool `{}` description looks like a stub (got: {:?})",
                info.name,
                info.description
            );
        }
    }

    #[test]
    fn plan_create_from_repoprompt_imports_steps_and_records_artifact() {
        let root = temp_root();
        let ctx = temp_ctx(&root);
        fs::create_dir_all(&ctx.cwd).unwrap();
        let export_path = ctx.cwd.join("export.md");
        fs::write(
            &export_path,
            "# Refactor cache\n\nWe need to split the cache layer.\n\n## Plan\n\n1. Investigate the existing cache across the codebase.\n2. Add new in-memory store.\n3. Run the integration tests.\n",
        )
        .unwrap();
        let call = ToolCall::new(
            "plan_create_from_repoprompt",
            json!({ "export_path": "export.md" }),
        );
        let result = PlanCreateFromRepoPromptTool.execute(&ctx, &call).unwrap();
        assert!(result.ok, "tool failed: {:?}", result.content);
        let import_stats = &result.content["import_stats"];
        assert_eq!(import_stats["steps_total"].as_u64(), Some(3));
        assert!(import_stats["delegated"].as_u64().unwrap_or_default() >= 1);
        let plan = &result.content["plan"];
        let artifacts = plan["state"]["orchestration"]["artifacts"]
            .as_array()
            .unwrap();
        assert!(
            artifacts
                .iter()
                .any(|a| a["kind"] == "repo_prompt_export" && a["path"].as_str().is_some()),
            "expected RepoPromptExport artifact, got {:?}",
            artifacts
        );
        let items = plan["items"].as_array().unwrap();
        let first = &items[0];
        assert!(
            first["text"].as_str().unwrap_or_default().contains("[D]"),
            "expected [D] marker on item 0, got {:?}",
            first["text"]
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn spawn_subagent_refuses_when_depth_at_limit() {
        let root = temp_root();
        let ctx = temp_ctx(&root);
        fs::create_dir_all(&ctx.cwd).unwrap();
        let prev = env::var(SEED_SUBAGENT_DEPTH_ENV).ok();
        unsafe {
            env::set_var(SEED_SUBAGENT_DEPTH_ENV, SEED_SUBAGENT_MAX_DEPTH.to_string());
        }
        let call = ToolCall::new(
            "spawn_subagent",
            json!({ "task": "noop", "max_turns": 1 }),
        );
        let result = SpawnSubagentTool.execute(&ctx, &call).unwrap();
        unsafe {
            match prev {
                Some(prev_value) => env::set_var(SEED_SUBAGENT_DEPTH_ENV, prev_value),
                None => env::remove_var(SEED_SUBAGENT_DEPTH_ENV),
            }
        }
        assert!(!result.ok);
        let message = result.content["message"].as_str().unwrap_or_default();
        assert!(message.contains("depth"), "got: {message}");
    }

    #[test]
    fn subagent_signals_round_trip_through_write_and_consume() {
        let root = temp_root();
        fs::create_dir_all(&root).unwrap();
        write_subagent_signals(&root, Some("verified port=8080"), Some("switch to plan mode"), true).unwrap();
        assert!(root.join(SUBAGENT_SIGNAL_KEYINFO).is_file());
        assert!(root.join(SUBAGENT_SIGNAL_INTERVENE).is_file());
        assert!(root.join(SUBAGENT_SIGNAL_STOP).is_file());

        let signals = consume_subagent_signals(&root);
        assert_eq!(signals.key_info, vec!["verified port=8080".to_string()]);
        assert_eq!(signals.intervene.as_deref(), Some("switch to plan mode"));
        assert!(signals.stop);

        // files must be consumed (deleted) so the same signal does not re-fire.
        assert!(!root.join(SUBAGENT_SIGNAL_KEYINFO).exists());
        assert!(!root.join(SUBAGENT_SIGNAL_INTERVENE).exists());
        assert!(!root.join(SUBAGENT_SIGNAL_STOP).exists());

        let empty = consume_subagent_signals(&root);
        assert!(empty.is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn read_files_batches_multiple_paths() {
        let root = temp_root();
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("a.md"), "alpha line one\nalpha line two\n").unwrap();
        fs::write(root.join("b.md"), "bravo line one\n").unwrap();
        let ctx = temp_ctx(&root);
        let call = ToolCall::new(
            "read_files",
            json!({ "paths": ["a.md", "b.md"], "show_line_numbers": false }),
        );
        let result = ReadFilesTool.execute(&ctx, &call).unwrap();
        assert!(result.ok);
        assert_eq!(result.content["succeeded"].as_u64(), Some(2));
        assert_eq!(result.content["failed"].as_u64(), Some(0));
        let files = result.content["files"].as_array().unwrap();
        assert_eq!(files.len(), 2);
        assert!(files[0]["content"].as_str().unwrap().contains("alpha"));
        assert!(files[1]["content"].as_str().unwrap().contains("bravo"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn read_files_reports_partial_when_one_path_missing() {
        let root = temp_root();
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("a.md"), "alpha\n").unwrap();
        let ctx = temp_ctx(&root);
        let call = ToolCall::new(
            "read_files",
            json!({ "paths": ["a.md", "nope.md"] }),
        );
        let result = ReadFilesTool.execute(&ctx, &call).unwrap();
        assert_eq!(
            result.content["status"].as_str(),
            Some("partial"),
            "got: {:?}",
            result.content
        );
        assert_eq!(result.content["succeeded"].as_u64(), Some(1));
        assert_eq!(result.content["failed"].as_u64(), Some(1));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn read_files_caps_path_count() {
        let root = temp_root();
        fs::create_dir_all(&root).unwrap();
        let ctx = temp_ctx(&root);
        let paths: Vec<String> = (0..12).map(|i| format!("file{i}.md")).collect();
        let call = ToolCall::new("read_files", json!({ "paths": paths }));
        let result = ReadFilesTool.execute(&ctx, &call).unwrap();
        assert!(!result.ok);
        assert!(
            result.content["message"]
                .as_str()
                .unwrap_or_default()
                .contains("capped"),
            "got: {:?}",
            result.content
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn registry_exposes_read_files() {
        let registry = seed_registry();
        assert!(registry.names().contains(&"read_files"));
    }

    #[test]
    fn spawn_subagent_map_rejects_empty_task_list() {
        let root = temp_root();
        fs::create_dir_all(&root).unwrap();
        let ctx = ToolContext::with_cwd(&root);
        let call = ToolCall::new("spawn_subagent_map", json!({ "tasks": [] }));
        let result = SpawnSubagentMapTool.execute(&ctx, &call).unwrap();
        assert!(!result.ok);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn registry_exposes_spawn_subagent_map_and_nudge() {
        let registry = seed_registry();
        let names = registry.names();
        assert!(names.contains(&"spawn_subagent_map"));
        assert!(names.contains(&"subagent_nudge"));
    }

    #[test]
    fn subagent_nudge_writes_requested_files() {
        let root = temp_root();
        fs::create_dir_all(&root).unwrap();
        let ctx = ToolContext::with_cwd(&root);
        let target = root.join("subagent").join("abc");
        fs::create_dir_all(&target).unwrap();
        let call = ToolCall::new(
            "subagent_nudge",
            json!({ "target": target, "key_info": "do this next" }),
        );
        let result = SubagentNudgeTool.execute(&ctx, &call).unwrap();
        assert!(result.ok);
        assert!(target.join(SUBAGENT_SIGNAL_KEYINFO).is_file());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn subagent_nudge_requires_at_least_one_signal() {
        let root = temp_root();
        fs::create_dir_all(&root).unwrap();
        let ctx = ToolContext::with_cwd(&root);
        let target = root.join("subagent").join("abc");
        fs::create_dir_all(&target).unwrap();
        let call = ToolCall::new("subagent_nudge", json!({ "target": target }));
        let result = SubagentNudgeTool.execute(&ctx, &call).unwrap();
        assert!(!result.ok);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn ask_user_refuses_when_stdin_is_not_a_tty() {
        let root = temp_root();
        let ctx = temp_ctx(&root);
        let call = ToolCall::new("ask_user", json!({ "question": "ok?" }));
        let result = AskUserTool.execute(&ctx, &call).unwrap();
        assert!(!result.ok);
        let message = result.content["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("terminal") || message.contains("interactive"),
            "got: {message}"
        );
    }

    #[test]
    fn repoprompt_tools_lists_full_wrapped_surface() {
        let call = ToolCall::new("repoprompt_tools", json!({}));
        let ctx = ToolContext::with_cwd(".");
        let result = RepoPromptToolsTool.execute(&ctx, &call).unwrap();

        assert!(result.ok);
        assert_eq!(result.content["tools"].as_array().unwrap().len(), 18);
        assert!(
            result.content["tools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| tool["name"] == "agent_run")
        );
    }

    #[test]
    fn plan_next_returns_ledger_summary_and_protocol_prompt() {
        let root = temp_root();
        let ctx = temp_ctx(&root);
        let create = ToolCall::new(
            "plan_create",
            json!({
                "title": "Ledger Protocol",
                "task": "Task",
                "steps": ["Inspect context"]
            }),
        );
        PlanCreateTool.execute(&ctx, &create).unwrap();
        let next = ToolCall::new("plan_next", json!({}));
        let result = PlanNextTool.execute(&ctx, &next).unwrap();

        assert!(result.ok);
        assert_eq!(result.content["ledger_summary"]["artifact_count"], json!(0));
        assert!(
            result.content["next_prompt"]
                .as_str()
                .unwrap()
                .contains("REPOPROMPT_LEDGER")
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn plan_list_tool_returns_counts_and_plans() {
        let root = temp_root();
        let ctx = temp_ctx(&root);
        let empty = PlanListTool
            .execute(&ctx, &ToolCall::new("plan_list", json!({})))
            .unwrap();
        assert!(empty.ok);
        assert_eq!(empty.content["total_count"], json!(0));

        PlanCreateTool
            .execute(
                &ctx,
                &ToolCall::new(
                    "plan_create",
                    json!({
                        "title": "List Tool",
                        "task": "Task",
                        "steps": ["Do one"]
                    }),
                ),
            )
            .unwrap();
        let result = PlanListTool
            .execute(&ctx, &ToolCall::new("plan_list", json!({"limit": 1})))
            .unwrap();

        assert!(result.ok);
        assert_eq!(result.content["total_count"], json!(1));
        assert_eq!(result.content["shown_count"], json!(1));
        assert_eq!(
            result.content["plans"][0]["state"]["title"],
            json!("List Tool")
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn plan_create_accepts_goal_and_items_aliases() {
        let root = temp_root();
        let ctx = temp_ctx(&root);
        let result = PlanCreateTool
            .execute(
                &ctx,
                &ToolCall::new(
                    "plan_create",
                    json!({
                        "goal": "优化当前项目：选择一个小而高价值的改进点。",
                        "items": ["Inspect code", "Implement change"]
                    }),
                ),
            )
            .unwrap();

        assert!(result.ok);
        assert_eq!(
            result.content["plan"]["state"]["title"],
            json!("优化当前项目：选择一个小而高价值的改进点")
        );
        assert_eq!(
            result.content["plan"]["state"]["task"],
            json!("优化当前项目：选择一个小而高价值的改进点。")
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn plan_tools_accept_common_plan_id_aliases() {
        let root = temp_root();
        let ctx = temp_ctx(&root);
        let created = PlanCreateTool
            .execute(
                &ctx,
                &ToolCall::new(
                    "plan_create",
                    json!({
                        "title": "Alias Plan",
                        "task": "Task",
                        "steps": ["Do one"]
                    }),
                ),
            )
            .unwrap();
        let plan_id = created.content["plan"]["state"]["id"]
            .as_str()
            .unwrap()
            .to_string();

        let next = PlanNextTool
            .execute(
                &ctx,
                &ToolCall::new("plan_next", json!({ "plan_id": plan_id.clone() })),
            )
            .unwrap();
        assert_eq!(next.content["plan_id"], json!(plan_id));

        let complete = PlanCompleteTool
            .execute(
                &ctx,
                &ToolCall::new(
                    "plan_complete",
                    json!({ "plan_id": plan_id.clone(), "item_index": 1 }),
                ),
            )
            .unwrap();
        assert!(complete.ok);
        assert!(
            complete.content["plan"]["items"][0]["checked"]
                .as_bool()
                .unwrap()
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn truncate_middle_does_not_split_utf8() {
        let input = "优化当前的项目".repeat(80);
        let output = truncate_middle(&input, 17);

        assert!(output.contains("[omitted long output]"));
    }

    #[test]
    fn repoprompt_routing_defaults_to_cwd_for_workspace_tools() {
        let root = temp_root();
        let ctx = temp_ctx(&root);
        let client = repoprompt_client_without_bind(&ctx, RepoPromptRoutingArgs::default(), true);

        assert_eq!(client.config().working_dirs, vec![root.clone()]);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn repoprompt_routing_skips_cwd_for_discovery_tools() {
        assert!(!default_cwd_for_repoprompt_tool(
            agent_repoprompt::RepoPromptTool::BindContext
        ));
        assert!(default_cwd_for_repoprompt_tool(
            agent_repoprompt::RepoPromptTool::ContextBuilder
        ));
        assert!(!default_cwd_for_repoprompt_exec("workspace list"));
        assert!(default_cwd_for_repoprompt_exec("search \"TODO\""));
    }

    // `repoprompt_exec_accepts_cmd_alias` moved to
    // `crates/agent-tools/src/repoprompt_bridge.rs::tests` since
    // `RepoPromptExecArgs` is now private to that module.

    #[test]
    fn durable_write_guardrails_block_secret_like_memory() {
        let root = temp_root();
        let ctx = temp_ctx(&root);
        fs::create_dir_all(&ctx.memory_dir).unwrap();
        let call = ToolCall::new(
            "write_file",
            json!({
                "path": "memory/global_facts.md",
                "content": "api_key: sk-1234567890abcdef"
            }),
        );

        let result = WriteFileTool.execute(&ctx, &call).unwrap();

        assert!(!result.ok);
        assert!(
            result.content["message"]
                .as_str()
                .unwrap()
                .contains("guardrail")
        );
    }

    #[test]
    fn durable_write_guardrails_block_overwrite_of_existing_memory() {
        let root = temp_root();
        let ctx = temp_ctx(&root);
        fs::create_dir_all(&ctx.memory_dir).unwrap();
        fs::write(
            ctx.memory_dir.join("global_facts.md"),
            "# Global Facts\n\n- verified\n",
        )
        .unwrap();
        let call = ToolCall::new(
            "write_file",
            json!({
                "path": "memory/global_facts.md",
                "content": "# Global Facts\n\n- replaced\n"
            }),
        );

        let result = WriteFileTool.execute(&ctx, &call).unwrap();

        assert!(!result.ok);
        assert!(
            result.content["message"]
                .as_str()
                .unwrap()
                .contains("use patch_file")
        );
    }

    #[test]
    fn durable_write_guardrails_block_generated_l1_edits() {
        let root = temp_root();
        let ctx = temp_ctx(&root);
        fs::create_dir_all(&ctx.memory_dir).unwrap();
        fs::write(ctx.memory_dir.join("l1_insight.md"), "# L1 Insight\n").unwrap();
        let call = ToolCall::new(
            "patch_file",
            json!({
                "path": "memory/l1_insight.md",
                "old_content": "# L1 Insight\n",
                "new_content": "# L1 Insight\n\nmanual edit\n"
            }),
        );

        let result = PatchFileTool.execute(&ctx, &call).unwrap();

        assert!(!result.ok);
        assert!(
            result.content["message"]
                .as_str()
                .unwrap()
                .contains("generated")
        );
    }

    fn temp_ctx(root: &Path) -> ToolContext {
        ToolContext::with_paths(
            root,
            root.join("skills"),
            root.join("memory"),
            root.join("sessions"),
        )
    }

    fn temp_root() -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        // Atomic counter prevents nanos-collision when two test threads
        // sample SystemTime within the same ns tick.
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "seed-tools-test-{}-{nanos}-{n}",
            std::process::id()
        ))
    }

    // --- /5 repoprompt_sync ----------------------------------------
    //
    // These tests poke the process-wide `repoprompt_sync` singleton. Cargo
    // runs tests in parallel by default, so any pair of these would race
    // (one sets, another peeks/clears in between). We serialize them with
    // a dedicated test-only mutex held for the lifetime of each test; the
    // guard is returned so test bodies can hold it via `let _g = ...`.

    use std::sync::Mutex as StdMutex;
    static RP_SYNC_TEST_LOCK: StdMutex<()> = StdMutex::new(());

    fn rp_sync_setup() -> std::sync::MutexGuard<'static, ()> {
        // `lock()` returns Err only if the mutex was poisoned by a previous
        // panicking test. We don't care — the inner data is `()`, so just
        // recover the guard and continue.
        let guard = RP_SYNC_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        repoprompt_sync::reset();
        assert!(repoprompt_sync::peek_pending_override().is_none());
        guard
    }

    #[test]
    fn repoprompt_sync_reset_clears_pending() {
        let _g = rp_sync_setup();
        repoprompt_sync::set_pending_override(vec![PathBuf::from("/skill-dir")]);
        assert!(repoprompt_sync::peek_pending_override().is_some());
        repoprompt_sync::reset();
        assert!(repoprompt_sync::peek_pending_override().is_none());
    }

    #[test]
    fn repoprompt_sync_take_is_consuming() {
        let _g = rp_sync_setup();
        repoprompt_sync::set_pending_override(vec![PathBuf::from("/a"), PathBuf::from("/b")]);
        let first = repoprompt_sync::take_pending_override();
        let second = repoprompt_sync::take_pending_override();
        assert_eq!(
            first,
            Some(vec![PathBuf::from("/a"), PathBuf::from("/b")])
        );
        assert!(second.is_none(), "second take should be None — override is one-shot");
    }

    #[test]
    fn default_working_dirs_consumes_pending_override() {
        let _g = rp_sync_setup();
        let root = temp_root();
        std::fs::create_dir_all(&root).unwrap();
        let ctx = temp_ctx(&root);
        repoprompt_sync::set_pending_override(vec![PathBuf::from("/some/skill/dir")]);
        let dirs = default_repoprompt_working_dirs(&ctx, None, true);
        // Override wins over ctx.cwd default. absolutize keeps absolute paths intact.
        assert_eq!(dirs, vec![PathBuf::from("/some/skill/dir")]);
        // And the override is now consumed.
        assert!(repoprompt_sync::peek_pending_override().is_none());
    }

    #[test]
    fn default_working_dirs_falls_back_to_ctx_cwd_after_override_consumed() {
        let _g = rp_sync_setup();
        let root = temp_root();
        std::fs::create_dir_all(&root).unwrap();
        let ctx = temp_ctx(&root);
        repoprompt_sync::set_pending_override(vec![PathBuf::from("/skill-dir")]);
        // First call consumes.
        let _ = default_repoprompt_working_dirs(&ctx, None, true);
        // Second call: no override → ctx.cwd.
        let dirs = default_repoprompt_working_dirs(&ctx, None, true);
        assert_eq!(dirs, vec![root.clone()]);
    }

    #[test]
    fn default_working_dirs_ignores_override_when_user_provided_dirs() {
        let _g = rp_sync_setup();
        let root = temp_root();
        std::fs::create_dir_all(&root).unwrap();
        let ctx = temp_ctx(&root);
        repoprompt_sync::set_pending_override(vec![PathBuf::from("/skill-dir")]);
        // User explicitly passed working_dirs → override should not fire.
        let explicit = vec![PathBuf::from("/user/picked")];
        let dirs = default_repoprompt_working_dirs(&ctx, Some(explicit.clone()), true);
        assert_eq!(dirs, explicit);
        // Override is still pending — it only consumes when defaults kick in.
        assert!(
            repoprompt_sync::peek_pending_override().is_some(),
            "override should survive a call that didn't need defaulting"
        );
        repoprompt_sync::reset();
    }

    #[test]
    fn default_working_dirs_ignores_override_when_default_cwd_false() {
        let _g = rp_sync_setup();
        let root = temp_root();
        std::fs::create_dir_all(&root).unwrap();
        let ctx = temp_ctx(&root);
        repoprompt_sync::set_pending_override(vec![PathBuf::from("/skill-dir")]);
        // Meta ops like bind_context call with default_cwd=false; the
        // override must not fire (those calls aren't workspace-scoped).
        let dirs = default_repoprompt_working_dirs(&ctx, None, false);
        assert!(dirs.is_empty(), "no defaulting when default_cwd=false");
        assert!(
            repoprompt_sync::peek_pending_override().is_some(),
            "override should survive a default_cwd=false call"
        );
        repoprompt_sync::reset();
    }

    #[test]
    fn queue_skill_repoprompt_binding_sets_override_and_returns_status() {
        let _g = rp_sync_setup();
        let binding = agent_skills::RepoPromptBinding {
            working_dirs: vec![PathBuf::from("/a"), PathBuf::from("/b")],
            context_id: Some("ctx-42".to_string()),
            ..Default::default()
        };
        let v = crate::skills::queue_skill_repoprompt_binding(&binding);
        // default sticky_cwd=false → transient classification.
        assert_eq!(v["status"], "queued_transient");
        assert_eq!(v["applies_to"], "next repoprompt_* tool call only");
        assert_eq!(v["sticky_cwd_target"], serde_json::Value::Null);
        assert_eq!(
            repoprompt_sync::peek_pending_override(),
            Some(vec![PathBuf::from("/a"), PathBuf::from("/b")])
        );
        // No sticky_cwd → no sticky pending.
        assert!(repoprompt_sync::peek_pending_sticky_cwd().is_none());
        repoprompt_sync::reset();
    }

    #[test]
    fn queue_skill_repoprompt_binding_queues_sticky_cwd_when_opted_in() {
        let _g = rp_sync_setup();
        let binding = agent_skills::RepoPromptBinding {
            working_dirs: vec![PathBuf::from("/skill-root")],
            sticky_cwd: true,
            ..Default::default()
        };
        let v = crate::skills::queue_skill_repoprompt_binding(&binding);
        assert_eq!(v["status"], "queued_sticky");
        assert!(
            v["applies_to"]
                .as_str()
                .unwrap_or("")
                .contains("workspace.cwd"),
            "applies_to should mention workspace.cwd, got: {}",
            v["applies_to"]
        );
        assert_eq!(
            v["sticky_cwd_target"]
                .as_str()
                .map(PathBuf::from),
            Some(PathBuf::from("/skill-root"))
        );
        assert_eq!(
            repoprompt_sync::peek_pending_sticky_cwd(),
            Some(PathBuf::from("/skill-root"))
        );
        repoprompt_sync::reset();
    }

    #[test]
    fn queue_skill_repoprompt_binding_noop_when_no_working_dirs() {
        let _g = rp_sync_setup();
        let binding = agent_skills::RepoPromptBinding {
            working_dirs: vec![],
            context_id: Some("ctx-7".to_string()),
            ..Default::default()
        };
        let v = crate::skills::queue_skill_repoprompt_binding(&binding);
        assert_eq!(v["status"], "noop");
        assert!(repoprompt_sync::peek_pending_override().is_none());
    }

    // --- skill_tools_guard -----------------------------------------

    static SKILL_TOOLS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn skill_tools_guard_setup() -> std::sync::MutexGuard<'static, ()> {
        let g = SKILL_TOOLS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        skill_tools_guard::reset();
        g
    }

    #[test]
    fn skill_tools_guard_defaults_to_permit_all() {
        let _g = skill_tools_guard_setup();
        // No restriction set → every tool name permitted.
        assert!(skill_tools_guard::permits("read_file"));
        assert!(skill_tools_guard::permits("write_file"));
        assert!(skill_tools_guard::permits("any_random_tool"));
        assert!(skill_tools_guard::current().is_none());
    }

    #[test]
    fn skill_tools_guard_set_narrows_catalog() {
        let _g = skill_tools_guard_setup();
        skill_tools_guard::set(vec!["read_file".to_string(), "run_shell".to_string()]);
        assert!(skill_tools_guard::permits("read_file"));
        assert!(skill_tools_guard::permits("run_shell"));
        assert!(!skill_tools_guard::permits("write_file"));
        assert!(!skill_tools_guard::permits("memory_search"));
        let active = skill_tools_guard::current().unwrap();
        assert_eq!(active, vec!["read_file".to_string(), "run_shell".to_string()]);
        skill_tools_guard::reset();
    }

    #[test]
    fn skill_tools_guard_empty_list_clears() {
        let _g = skill_tools_guard_setup();
        skill_tools_guard::set(vec!["x".to_string()]);
        assert!(skill_tools_guard::current().is_some());
        // Passing empty = clear.
        skill_tools_guard::set(Vec::new());
        assert!(skill_tools_guard::current().is_none());
        assert!(skill_tools_guard::permits("anything"));
    }

    #[test]
    fn skill_tools_guard_set_replaces_not_intersects() {
        let _g = skill_tools_guard_setup();
        skill_tools_guard::set(vec!["read_file".to_string()]);
        skill_tools_guard::set(vec!["run_shell".to_string()]);
        assert!(!skill_tools_guard::permits("read_file"));
        assert!(skill_tools_guard::permits("run_shell"));
        skill_tools_guard::reset();
    }

    // --- serde aliases on arg structs ----------------------------

    // `memory_search_args_accept_q_alias` moved to
    // `crates/agent-tools/src/memory.rs::tests` since `MemorySearchArgs`
    // is now private to that module.

    // --- humanize_rp_bind_failure --------------------------------

    #[test]
    fn humanize_recognizes_no_match_error() {
        let raw = r#"{"error":"No exact workspace match for /tmp/foo"}"#;
        let msg = humanize_rp_bind_failure(raw);
        assert!(msg.contains("auto-create") || msg.contains("create_if_missing"));
    }

    #[test]
    fn humanize_recognizes_connection_refused() {
        let msg = humanize_rp_bind_failure("connection refused");
        assert!(msg.contains("RepoPrompt app isn't running"));
    }

    #[test]
    fn humanize_recognizes_multiple_windows() {
        let msg = humanize_rp_bind_failure("Multiple RepoPrompt windows detected");
        assert!(msg.contains("window_id") || msg.contains("disambiguate"));
    }

    #[test]
    fn humanize_falls_back_when_unrecognized() {
        let msg = humanize_rp_bind_failure("some new error message we haven't seen");
        assert!(msg.contains("see raw"));
    }

    // --- repair_tool_args ----------------------------------------

    #[test]
    fn repair_unwraps_stringified_json_object() {
        let input = serde_json::Value::String("{\"k\":\"v\"}".to_string());
        let repaired = repair_tool_args(input);
        assert_eq!(repaired, serde_json::json!({"k": "v"}));
    }

    #[test]
    fn repair_unwraps_stringified_json_array() {
        let input = serde_json::Value::String("[1, 2, 3]".to_string());
        let repaired = repair_tool_args(input);
        assert_eq!(repaired, serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn repair_leaves_genuine_string_untouched() {
        // A user-meant string like a goal/prompt shouldn't be re-parsed.
        let input = serde_json::Value::String("hello world".to_string());
        let repaired = repair_tool_args(input);
        assert_eq!(repaired, serde_json::json!("hello world"));
    }

    #[test]
    fn repair_treats_null_as_empty_object() {
        let repaired = repair_tool_args(serde_json::Value::Null);
        assert_eq!(repaired, serde_json::json!({}));
    }

    #[test]
    fn repair_unwraps_sole_args_envelope() {
        let input = serde_json::json!({"args": {"k": "v"}});
        let repaired = repair_tool_args(input);
        assert_eq!(repaired, serde_json::json!({"k": "v"}));
    }

    #[test]
    fn repair_does_not_unwrap_envelope_with_siblings() {
        // {"args": {...}, "other": ...} — caller probably meant the
        // multi-field object; eating "args" would lose data.
        let input = serde_json::json!({"args": {"k": "v"}, "other": 1});
        let repaired = repair_tool_args(input.clone());
        assert_eq!(repaired, input);
    }

    #[test]
    fn repair_handles_nested_stringification() {
        // Double-wrap: object → string → "args" envelope. Recursion peels both.
        let input = serde_json::json!({"args": "{\"k\": \"v\"}"});
        let repaired = repair_tool_args(input);
        assert_eq!(repaired, serde_json::json!({"k": "v"}));
    }

    #[test]
    fn parse_tool_args_succeeds_via_repair_path() {
        use serde::Deserialize;
        #[derive(Debug, Deserialize, PartialEq)]
        struct Foo {
            k: String,
        }
        let call = ToolCall::new(
            "test",
            serde_json::Value::String("{\"k\":\"v\"}".to_string()),
        );
        let parsed: Foo = parse_tool_args(&call).unwrap();
        assert_eq!(parsed, Foo { k: "v".to_string() });
    }

    #[test]
    fn parse_tool_args_returns_invalid_arguments_for_unrepairable() {
        use serde::Deserialize;
        #[derive(Debug, Deserialize)]
        struct Foo {
            #[allow(dead_code)]
            k: String,
        }
        // Plain string can't become Foo even after repair → InvalidArguments.
        let call = ToolCall::new("test", serde_json::Value::String("plain".to_string()));
        let err = parse_tool_args::<Foo>(&call).unwrap_err();
        match err {
            ToolError::InvalidArguments { tool, .. } => assert_eq!(tool, "test"),
            other => panic!("expected InvalidArguments, got {other:?}"),
        }
    }

    // --- truncate_middle_with_stats ------------------------------

    #[test]
    fn truncate_stats_no_truncation_when_under_limit() {
        let (text, stats) = truncate_middle_with_stats("hello", 100);
        assert_eq!(text, "hello");
        assert!(!stats.was_truncated);
        assert_eq!(stats.original_bytes, 5);
    }

    #[test]
    fn truncate_stats_no_truncation_at_exact_limit() {
        let input = "a".repeat(100);
        let (text, stats) = truncate_middle_with_stats(&input, 100);
        assert_eq!(text, input);
        assert!(!stats.was_truncated);
        assert_eq!(stats.original_bytes, 100);
    }

    #[test]
    fn truncate_stats_reports_original_size() {
        let input = "x".repeat(50_000);
        let (text, stats) = truncate_middle_with_stats(&input, 1_000);
        assert!(stats.was_truncated);
        assert_eq!(stats.original_bytes, 50_000);
        // Truncation marker is present.
        assert!(text.contains("omitted long output"));
        // Output IS shorter than input — though it may exceed max_len by
        // the length of the marker; the contract is "shorter than input".
        assert!(text.len() < input.len());
    }

    // --- bound-window cache ---------------------------------------

    #[test]
    fn bound_window_cache_round_trips() {
        let _g = rp_sync_setup();
        let dirs = vec![PathBuf::from("/repo/a")];
        assert!(repoprompt_sync::cached_window_id_for(&dirs).is_none());
        repoprompt_sync::record_bound_window(dirs.clone(), 7);
        assert_eq!(repoprompt_sync::cached_window_id_for(&dirs), Some(7));
        assert_eq!(
            repoprompt_sync::peek_bound_window(),
            Some((dirs.clone(), 7))
        );
    }

    #[test]
    fn bound_window_cache_misses_on_different_dirs() {
        let _g = rp_sync_setup();
        repoprompt_sync::record_bound_window(vec![PathBuf::from("/repo/a")], 1);
        // Same prefix, different path → cache miss (set equality).
        let miss = repoprompt_sync::cached_window_id_for(&[PathBuf::from("/repo/b")]);
        assert_eq!(miss, None);
        // Different order of multi-dir set → also miss (vec order matters).
        let miss2 = repoprompt_sync::cached_window_id_for(&[
            PathBuf::from("/repo/b"),
            PathBuf::from("/repo/a"),
        ]);
        assert_eq!(miss2, None);
    }

    #[test]
    fn clear_bound_window_drops_cache_but_not_override() {
        let _g = rp_sync_setup();
        repoprompt_sync::record_bound_window(vec![PathBuf::from("/repo/a")], 1);
        repoprompt_sync::set_pending_override(vec![PathBuf::from("/repo/skill")]);
        // set_pending_override clears bound (intentional — about to switch).
        assert!(repoprompt_sync::peek_bound_window().is_none());
        // Re-record after, then test the standalone clear:
        repoprompt_sync::record_bound_window(vec![PathBuf::from("/repo/a")], 1);
        assert!(repoprompt_sync::peek_bound_window().is_some());
        assert!(repoprompt_sync::peek_pending_override().is_some());
        repoprompt_sync::clear_bound_window();
        assert!(repoprompt_sync::peek_bound_window().is_none());
        assert!(
            repoprompt_sync::peek_pending_override().is_some(),
            "clear_bound_window must not touch pending_override"
        );
        repoprompt_sync::reset();
    }

    #[test]
    fn reset_clears_both_override_and_bound() {
        let _g = rp_sync_setup();
        repoprompt_sync::record_bound_window(vec![PathBuf::from("/a")], 9);
        repoprompt_sync::set_pending_override(vec![PathBuf::from("/skill")]);
        repoprompt_sync::record_bound_window(vec![PathBuf::from("/a")], 9);
        // set_pending_override above cleared bound, so re-record after.
        assert!(repoprompt_sync::peek_bound_window().is_some());
        assert!(repoprompt_sync::peek_pending_override().is_some());
        repoprompt_sync::reset();
        assert!(repoprompt_sync::peek_bound_window().is_none());
        assert!(repoprompt_sync::peek_pending_override().is_none());
    }

    // --- run_mode_guard + shell_command_intent ----------------------

    static RUN_MODE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn run_mode_setup() -> std::sync::MutexGuard<'static, ()> {
        let g = RUN_MODE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        run_mode_guard::reset();
        g
    }

    #[test]
    fn run_mode_guard_defaults_to_implementation() {
        let _g = run_mode_setup();
        // reset() in setup pins to Implementation explicitly.
        assert_eq!(run_mode_guard::current(), RunMode::Implementation);
    }

    #[test]
    fn run_mode_guard_set_round_trips() {
        let _g = run_mode_setup();
        run_mode_guard::set(RunMode::ReadOnly);
        assert_eq!(run_mode_guard::current(), RunMode::ReadOnly);
        run_mode_guard::set(RunMode::Implementation);
        assert_eq!(run_mode_guard::current(), RunMode::Implementation);
    }

    #[test]
    fn shell_intent_classifies_obvious_reads() {
        for cmd in [
            "ls",
            "ls -la",
            "cat README.md",
            "head -20 file.txt",
            "wc -l Cargo.toml",
            "grep TODO src/",
            "rg --files",
            "find . -name '*.rs'",
            "git status",
            "git log --oneline -5",
            "git diff",
            "pwd",
            "echo hello",
            "cargo check",
            "cargo tree",
        ] {
            assert_eq!(
                shell_command_intent(cmd),
                ShellIntent::Read,
                "expected Read for: {cmd}"
            );
        }
    }

    #[test]
    fn shell_intent_classifies_obvious_writes() {
        for cmd in [
            "echo x > foo.txt",
            "echo x >> log.txt",
            "ls > listing.out",
            "ls | tee tee-out.txt",
            "cat foo | tee bar",
            "rm -rf target",
            "mv a b",
            "mkdir new-dir",
            "cp -r src dst",
            "touch new-file",
            "chmod +x script.sh",
            "chown user file",
            "sed -i 's/a/b/' file",
            "perl -i -pe 's/x/y/' f",
            "git commit -m 'x'",
            "git push origin main",
            "git checkout main",
            "git reset --hard HEAD~1",
            "git add .",
            "cargo install --path .",
            "cargo build",
            "cargo test",
            "npm install foo",
            "pip install pkg",
        ] {
            assert_eq!(
                shell_command_intent(cmd),
                ShellIntent::Write,
                "expected Write for: {cmd}"
            );
        }
    }

    #[test]
    fn shell_intent_treats_unknown_commands_as_ambiguous() {
        for cmd in [
            "make something",     // could read or write
            "python script.py",   // could read or write
            "node app.js",        // could read or write
            "some-custom-tool",   // we don't know
            "",                   // empty
        ] {
            assert_eq!(
                shell_command_intent(cmd),
                ShellIntent::Ambiguous,
                "expected Ambiguous for: {cmd:?}"
            );
        }
    }

    #[test]
    fn shell_intent_write_in_chain_dominates_reads() {
        // `git status && rm -rf target` is a Write — chained writes block.
        assert_eq!(
            shell_command_intent("git status && rm -rf target"),
            ShellIntent::Write
        );
        assert_eq!(
            shell_command_intent("ls; mv old new"),
            ShellIntent::Write
        );
    }

    #[test]
    fn shell_tool_refuses_writes_in_read_only_mode() {
        let _g = run_mode_setup();
        run_mode_guard::set(RunMode::ReadOnly);
        let root = temp_root();
        std::fs::create_dir_all(&root).unwrap();
        let ctx = temp_ctx(&root);
        let call = ToolCall::new(
            "run_shell",
            json!({ "command": "echo x > foo.txt" }),
        );
        let err = ShellTool.execute(&ctx, &call).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("read-only mode") && msg.contains("write"),
            "unexpected error: {msg}"
        );
        // The file MUST NOT have been created.
        assert!(
            !root.join("foo.txt").exists(),
            "read-only block leaked: foo.txt was written"
        );
        run_mode_guard::reset();
    }

    #[test]
    fn shell_tool_allows_reads_in_read_only_mode() {
        let _g = run_mode_setup();
        run_mode_guard::set(RunMode::ReadOnly);
        let root = temp_root();
        std::fs::create_dir_all(&root).unwrap();
        let ctx = temp_ctx(&root);
        // `true` always exits 0 on POSIX, costs nothing.
        let call = ToolCall::new("run_shell", json!({ "command": "true" }));
        let result = ShellTool.execute(&ctx, &call);
        assert!(result.is_ok(), "read-only should allow `true`: {result:?}");
        // Same for `ls` (we registered it as a Read).
        let call = ToolCall::new("run_shell", json!({ "command": "ls" }));
        let result = ShellTool.execute(&ctx, &call);
        assert!(result.is_ok(), "read-only should allow `ls`: {result:?}");
        run_mode_guard::reset();
    }

    #[test]
    fn shell_tool_allows_writes_in_implementation_mode() {
        let _g = run_mode_setup();
        run_mode_guard::set(RunMode::Implementation);
        let root = temp_root();
        std::fs::create_dir_all(&root).unwrap();
        let ctx = temp_ctx(&root);
        let call = ToolCall::new(
            "run_shell",
            json!({ "command": "echo hi > test-file.txt" }),
        );
        let result = ShellTool.execute(&ctx, &call);
        assert!(result.is_ok(), "implementation must allow writes: {result:?}");
        // Confirm the side effect actually happened (sanity).
        assert!(root.join("test-file.txt").exists());
        run_mode_guard::reset();
    }
}
