use agent_core::{Tool, ToolCall, ToolContext, ToolError, ToolRegistry, ToolResult};
use serde::Deserialize;
use serde_json::json;
use std::collections::VecDeque;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;
use wait_timeout::ChildExt;

pub fn seed_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(MemorySearchTool);
    registry.register(MemoryFetchTool);
    registry.register(SkillListTool);
    registry.register(SkillSearchTool);
    registry.register(SkillFetchTool);
    registry.register(PlanCreateTool);
    registry.register(PlanStatusTool);
    registry.register(PlanNextTool);
    registry.register(PlanCompleteTool);
    registry.register(PlanVerifyTool);
    registry.register(RepoPromptToolsTool);
    registry.register(RepoPromptExecTool);
    registry.register(RepoPromptCallTool);
    registry.register(ReadFileTool);
    registry.register(PatchFileTool);
    registry.register(WriteFileTool);
    registry.register(ShellTool);
    registry.register(WorkingCheckpointTool);
    registry.register(LongTermUpdateTool);
    registry.register(CompleteLongTermUpdateTool);
    registry
}

#[derive(Debug, Deserialize)]
struct MemorySearchArgs {
    query: String,
    limit: Option<usize>,
}

pub struct MemorySearchTool;

impl Tool for MemorySearchTool {
    fn name(&self) -> &'static str {
        "memory_search"
    }

    fn description(&self) -> &'static str {
        "Search the L1 memory index across L0 rules, L2 facts, L3 skills, and L4 sessions without loading full bodies."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: MemorySearchArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let paths = memory_paths(ctx);
        let index = agent_memory::rebuild_index(&paths)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let hits = agent_memory::search_index(&index, &args.query, args.limit.unwrap_or(10));
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "query": args.query,
                "index_path": paths.index_path(),
                "results": hits,
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct MemoryFetchArgs {
    id: String,
    max_bytes: Option<usize>,
}

pub struct MemoryFetchTool;

impl Tool for MemoryFetchTool {
    fn name(&self) -> &'static str {
        "memory_fetch"
    }

    fn description(&self) -> &'static str {
        "Fetch one memory document by L1 index id or exact path; use after memory_search."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: MemoryFetchArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let doc = agent_memory::fetch_memory(
            &memory_paths(ctx),
            &args.id,
            args.max_bytes.unwrap_or(16_000),
        )
        .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "entry": doc.entry,
                "body": doc.body,
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct SkillListArgs {
    limit: Option<usize>,
}

pub struct SkillListTool;

impl Tool for SkillListTool {
    fn name(&self) -> &'static str {
        "skill_list"
    }

    fn description(&self) -> &'static str {
        "List lightweight skill metadata without loading full skill bodies."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: SkillListArgs =
            serde_json::from_value(call.args.clone()).unwrap_or(SkillListArgs { limit: None });
        let limit = args.limit.unwrap_or(50).clamp(1, 200);
        let skills = agent_skills::list_skill_infos(&ctx.skills_dir)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "skills_dir": ctx.skills_dir.display().to_string(),
                "skills": skills.into_iter().take(limit).collect::<Vec<_>>(),
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct SkillSearchArgs {
    query: String,
    limit: Option<usize>,
}

pub struct SkillSearchTool;

impl Tool for SkillSearchTool {
    fn name(&self) -> &'static str {
        "skill_search"
    }

    fn description(&self) -> &'static str {
        "Search local skill metadata by name, description, and tags."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: SkillSearchArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let skills = agent_skills::search_skill_infos(
            &ctx.skills_dir,
            &args.query,
            args.limit.unwrap_or(10),
        )
        .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "query": args.query,
                "skills": skills,
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct SkillFetchArgs {
    name: String,
}

pub struct SkillFetchTool;

impl Tool for SkillFetchTool {
    fn name(&self) -> &'static str {
        "skill_fetch"
    }

    fn description(&self) -> &'static str {
        "Fetch the full SKILL.md body for one exact skill name or slug."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: SkillFetchArgs = serde_json::from_value(call.args.clone()).map_err(|source| {
            ToolError::InvalidArguments {
                tool: call.name.clone(),
                source,
            }
        })?;
        let document = agent_skills::fetch_skill(&ctx.skills_dir, &args.name)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "skill": document.info,
                "body": document.body,
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct PlanCreateArgs {
    title: String,
    task: String,
    steps: Option<Vec<String>>,
    source_export_path: Option<PathBuf>,
}

pub struct PlanCreateTool;

impl Tool for PlanCreateTool {
    fn name(&self) -> &'static str {
        "plan_create"
    }

    fn description(&self) -> &'static str {
        "Create a durable GenericAgent-style plan under plans/<id>/ with plan.md, state.json, and a required verification gate."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: PlanCreateArgs = serde_json::from_value(call.args.clone()).map_err(|source| {
            ToolError::InvalidArguments {
                tool: call.name.clone(),
                source,
            }
        })?;
        let snapshot = plan_store(ctx)
            .create(agent_plan::CreatePlan {
                title: args.title,
                task: args.task,
                steps: args.steps.unwrap_or_default(),
                source_export_path: args
                    .source_export_path
                    .map(|path| absolutize(&ctx.cwd, path)),
            })
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "plan": snapshot,
                "next_prompt": "PLAN_MODE: execute only the next unchecked item from plan_next. After each meaningful change, call plan_complete for the completed item. When only [VERIFY] remains, call plan_verify; do not finish until verification returns PASS.",
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct PlanIdArgs {
    id: Option<String>,
}

pub struct PlanStatusTool;

impl Tool for PlanStatusTool {
    fn name(&self) -> &'static str {
        "plan_status"
    }

    fn description(&self) -> &'static str {
        "Read the current or selected plan state, checkbox items, and next unchecked item."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: PlanIdArgs =
            serde_json::from_value(call.args.clone()).unwrap_or(PlanIdArgs { id: None });
        let snapshot = plan_store(ctx)
            .snapshot(args.id.as_deref())
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "plan": snapshot,
            }),
        ))
    }
}

pub struct PlanNextTool;

impl Tool for PlanNextTool {
    fn name(&self) -> &'static str {
        "plan_next"
    }

    fn description(&self) -> &'static str {
        "Return the next unchecked plan item; use this before continuing a plan-mode task."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: PlanIdArgs =
            serde_json::from_value(call.args.clone()).unwrap_or(PlanIdArgs { id: None });
        let snapshot = plan_store(ctx)
            .snapshot(args.id.as_deref())
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "plan_id": snapshot.state.id,
                "next_item": snapshot.next_item,
                "unchecked_count": snapshot.unchecked_count,
                "task_unchecked_count": snapshot.task_unchecked_count,
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct PlanCompleteArgs {
    id: Option<String>,
    item: Option<usize>,
    note: Option<String>,
}

pub struct PlanCompleteTool;

impl Tool for PlanCompleteTool {
    fn name(&self) -> &'static str {
        "plan_complete"
    }

    fn description(&self) -> &'static str {
        "Mark one plan item complete by item index, or mark the current next item when omitted."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: PlanCompleteArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let snapshot = plan_store(ctx)
            .complete(args.id.as_deref(), args.item, args.note.as_deref())
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let next_prompt = if snapshot.task_unchecked_count == 0 && snapshot.next_item.is_some() {
            "PLAN_VERIFY_REQUIRED: all non-verify plan items are complete. Call plan_verify now; do not finish until the independent verification gate returns PASS."
        } else {
            "PLAN_MODE: call plan_next and continue with the next unchecked item. Do not skip verification."
        };
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "plan": snapshot,
                "next_prompt": next_prompt,
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct PlanVerifyArgs {
    id: Option<String>,
    model_id: Option<String>,
    timeout_secs: Option<u64>,
    dry_run: Option<bool>,
    window_id: Option<u32>,
    context_id: Option<String>,
    working_dirs: Option<Vec<PathBuf>>,
}

pub struct PlanVerifyTool;

impl Tool for PlanVerifyTool {
    fn name(&self) -> &'static str {
        "plan_verify"
    }

    fn description(&self) -> &'static str {
        "Create verify_context.json and run an independent RepoPrompt agent_run verifier; PASS marks the plan verified, FAIL appends a [FIX] item."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: PlanVerifyArgs = serde_json::from_value(call.args.clone()).map_err(|source| {
            ToolError::InvalidArguments {
                tool: call.name.clone(),
                source,
            }
        })?;
        let store = plan_store(ctx);
        let verify_context = store
            .write_verify_context(args.id.as_deref())
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        if args.dry_run.unwrap_or(false) {
            let snapshot = store
                .snapshot(Some(&verify_context.plan_id))
                .map_err(|err| ToolError::Failed(err.to_string()))?;
            return Ok(ToolResult::ok(
                call,
                json!({
                    "status": "success",
                    "dry_run": true,
                    "verify_context": verify_context,
                    "plan": snapshot,
                }),
            ));
        }

        let timeout_secs = args.timeout_secs.unwrap_or(300).max(1);
        let message = format!(
            "Independent verification gate for SeedAgent plan `{}`.\n\
Read the verify context JSON first: `{}`.\n\
Then inspect the plan, relevant files, git diff, and available test evidence.\n\
Return a concise report containing exactly one verdict line: `VERDICT: PASS` or `VERDICT: FAIL`.\n\
Do not edit files during verification unless the user explicitly asked for a fixing pass.",
            verify_context.plan_id,
            verify_context
                .plan_file
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("verify_context.json")
                .display()
        );
        let routing = RepoPromptRoutingArgs {
            timeout_secs: Some(timeout_secs + 60),
            window_id: args.window_id,
            context_id: args.context_id,
            working_dirs: args.working_dirs,
            raw_json: Some(true),
            ..Default::default()
        };
        let output = repoprompt_client(ctx, routing)
            .call_tool(
                agent_repoprompt::RepoPromptTool::AgentRun,
                &json!({
                    "op": "start",
                    "model_id": args.model_id.unwrap_or_else(|| "pair".to_string()),
                    "message": message,
                    "timeout": timeout_secs,
                }),
            )
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let report = repoprompt_report_text(&output);
        let snapshot = store
            .record_verification(Some(&verify_context.plan_id), &report)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let next_prompt = match snapshot.state.status {
            agent_plan::PlanStatus::Verified => {
                "PLAN_VERIFIED: independent verification returned PASS. It is now safe to finish with the verified outcome."
            }
            agent_plan::PlanStatus::Failed => {
                "PLAN_FIX_REQUIRED: independent verification returned FAIL. Call plan_next and execute the appended [FIX] item before verifying again."
            }
            _ => {
                "PLAN_VERIFY_PENDING: verifier did not return a clear PASS/FAIL. Inspect the report and rerun plan_verify or ask the user."
            }
        };
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "verify_context": verify_context,
                "repoprompt": repoprompt_output_json(output),
                "report": report,
                "plan": snapshot,
                "next_prompt": next_prompt,
            }),
        ))
    }
}

pub struct RepoPromptToolsTool;

impl Tool for RepoPromptToolsTool {
    fn name(&self) -> &'static str {
        "repoprompt_tools"
    }

    fn description(&self) -> &'static str {
        "List all RepoPrompt tools wrapped by SeedAgent, grouped by capability."
    }

    fn execute(&self, _ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "tools": agent_repoprompt::known_tools(),
            }),
        ))
    }
}

#[derive(Debug, Default, Deserialize)]
struct RepoPromptRoutingArgs {
    cli_path: Option<PathBuf>,
    timeout_secs: Option<u64>,
    window_id: Option<u32>,
    tab: Option<String>,
    context_id: Option<String>,
    working_dirs: Option<Vec<PathBuf>>,
    raw_json: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct RepoPromptExecArgs {
    command: String,
    #[serde(flatten)]
    routing: RepoPromptRoutingArgs,
}

pub struct RepoPromptExecTool;

impl Tool for RepoPromptExecTool {
    fn name(&self) -> &'static str {
        "repoprompt_exec"
    }

    fn description(&self) -> &'static str {
        "Execute a RepoPrompt CLI command chain such as windows, tree, search, select, context, builder, plan, or review."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: RepoPromptExecArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let client = repoprompt_client(ctx, args.routing);
        let output = client
            .exec(&args.command)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(call, repoprompt_output_json(output)))
    }
}

#[derive(Debug, Deserialize)]
struct RepoPromptCallArgs {
    tool: String,
    #[serde(default)]
    args: serde_json::Value,
    #[serde(flatten)]
    routing: RepoPromptRoutingArgs,
}

pub struct RepoPromptCallTool;

impl Tool for RepoPromptCallTool {
    fn name(&self) -> &'static str {
        "repoprompt_call"
    }

    fn description(&self) -> &'static str {
        "Call any wrapped RepoPrompt MCP tool by name with JSON args; use repoprompt_tools to inspect supported tools."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: RepoPromptCallArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let tool = args
            .tool
            .parse::<agent_repoprompt::RepoPromptTool>()
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let mut routing = args.routing;
        routing.raw_json = Some(routing.raw_json.unwrap_or(true));
        let client = repoprompt_client(ctx, routing);
        let output = client
            .call_tool(tool, &args.args)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(call, repoprompt_output_json(output)))
    }
}

#[derive(Debug, Deserialize)]
struct ReadFileArgs {
    path: String,
    start: Option<usize>,
    count: Option<usize>,
    keyword: Option<String>,
    show_line_numbers: Option<bool>,
}

pub struct ReadFileTool;

impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read a bounded slice of a UTF-8 file, optionally around a keyword; args: path, optional start, count, keyword, show_line_numbers."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: ReadFileArgs = serde_json::from_value(call.args.clone()).map_err(|source| {
            ToolError::InvalidArguments {
                tool: call.name.clone(),
                source,
            }
        })?;
        let path = resolve_path(&ctx.cwd, &args.path);
        let start = args.start.unwrap_or(1).max(1);
        let count = args.count.unwrap_or(200).clamp(1, 1000);
        let show_line_numbers = args.show_line_numbers.unwrap_or(true);
        let content = read_file_window(
            &path,
            start,
            count,
            args.keyword.as_deref(),
            show_line_numbers,
        )
        .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "path": path,
                "content": content,
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct PatchFileArgs {
    path: String,
    old_content: String,
    new_content: String,
}

pub struct PatchFileTool;

impl Tool for PatchFileTool {
    fn name(&self) -> &'static str {
        "patch_file"
    }

    fn description(&self) -> &'static str {
        "Replace one unique exact text block in a file; args: path, old_content, new_content."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: PatchFileArgs = serde_json::from_value(call.args.clone()).map_err(|source| {
            ToolError::InvalidArguments {
                tool: call.name.clone(),
                source,
            }
        })?;
        let path = resolve_path(&ctx.cwd, &args.path);
        if args.old_content.is_empty() {
            return Ok(ToolResult::error(call, "old_content must not be empty"));
        }
        let text = fs::read_to_string(&path).map_err(|err| ToolError::Failed(err.to_string()))?;
        let matches = text.matches(&args.old_content).count();
        if matches == 0 {
            return Ok(ToolResult::error(
                call,
                "old_content was not found; read the file again and patch a smaller exact block",
            ));
        }
        if matches > 1 {
            return Ok(ToolResult::error(
                call,
                format!("old_content matched {matches} places; provide a more specific block"),
            ));
        }
        let updated_text = text.replace(&args.old_content, &args.new_content);
        if let Some(message) = durable_write_guard(
            ctx,
            &path,
            &args.new_content,
            DurableWriteMode::Patch,
            false,
        ) {
            return Ok(ToolResult::error(call, message));
        }
        fs::write(&path, updated_text).map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({ "status": "success", "path": path, "matches": matches, "durable_guarded": is_durable_path(ctx, &path) }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct WriteFileArgs {
    path: String,
    content: String,
    mode: Option<WriteMode>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum WriteMode {
    Overwrite,
    Append,
    Prepend,
}

pub struct WriteFileTool;

impl Tool for WriteFileTool {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn description(&self) -> &'static str {
        "Create, overwrite, append, or prepend a file; args: path, content, optional mode overwrite|append|prepend."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: WriteFileArgs = serde_json::from_value(call.args.clone()).map_err(|source| {
            ToolError::InvalidArguments {
                tool: call.name.clone(),
                source,
            }
        })?;
        let path = resolve_path(&ctx.cwd, &args.path);
        let mode = args.mode.unwrap_or(WriteMode::Overwrite);
        let existing_nonempty = fs::read_to_string(&path)
            .map(|text| !text.trim().is_empty())
            .unwrap_or(false);
        if let Some(message) = durable_write_guard(
            ctx,
            &path,
            &args.content,
            DurableWriteMode::from(mode),
            existing_nonempty,
        ) {
            return Ok(ToolResult::error(call, message));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| ToolError::Failed(err.to_string()))?;
        }
        match mode {
            WriteMode::Overwrite => {
                fs::write(&path, &args.content).map_err(|err| ToolError::Failed(err.to_string()))?
            }
            WriteMode::Append => {
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .map_err(|err| ToolError::Failed(err.to_string()))?;
                file.write_all(args.content.as_bytes())
                    .map_err(|err| ToolError::Failed(err.to_string()))?;
            }
            WriteMode::Prepend => {
                let old = fs::read_to_string(&path).unwrap_or_default();
                fs::write(&path, format!("{}{}", args.content, old))
                    .map_err(|err| ToolError::Failed(err.to_string()))?;
            }
        }
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "path": path,
                "written_bytes": args.content.len(),
            }),
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DurableWriteMode {
    Patch,
    Overwrite,
    Append,
    Prepend,
}

impl From<WriteMode> for DurableWriteMode {
    fn from(value: WriteMode) -> Self {
        match value {
            WriteMode::Overwrite => Self::Overwrite,
            WriteMode::Append => Self::Append,
            WriteMode::Prepend => Self::Prepend,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ShellArgs {
    command: String,
    cwd: Option<String>,
    timeout_secs: Option<u64>,
}

pub struct ShellTool;

impl Tool for ShellTool {
    fn name(&self) -> &'static str {
        "run_shell"
    }

    fn description(&self) -> &'static str {
        "Run a shell command in a working directory with a timeout."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: ShellArgs = serde_json::from_value(call.args.clone()).map_err(|source| {
            ToolError::InvalidArguments {
                tool: call.name.clone(),
                source,
            }
        })?;
        let cwd = args
            .cwd
            .as_deref()
            .map(|path| resolve_path(&ctx.cwd, path))
            .unwrap_or_else(|| ctx.cwd.clone());
        let output = run_shell(&args.command, &cwd, args.timeout_secs.unwrap_or(60))
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(call, output))
    }
}

#[derive(Debug, Deserialize)]
struct CheckpointArgs {
    key_info: String,
    related_skill: Option<String>,
}

pub struct WorkingCheckpointTool;

impl Tool for WorkingCheckpointTool {
    fn name(&self) -> &'static str {
        "update_working_checkpoint"
    }

    fn description(&self) -> &'static str {
        "Record verified short-term task context to anchor future planner turns; args: key_info, optional related_skill."
    }

    fn execute(&self, _ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        checkpoint_result(call)
    }
}

fn checkpoint_result(call: &ToolCall) -> Result<ToolResult, ToolError> {
    let args: CheckpointArgs = serde_json::from_value(call.args.clone()).map_err(|source| {
        ToolError::InvalidArguments {
            tool: call.name.clone(),
            source,
        }
    })?;
    Ok(ToolResult::ok(
        call,
        json!({
            "status": "success",
            "key_info": args.key_info,
            "related_skill": args.related_skill,
        }),
    ))
}

#[derive(Debug, Deserialize)]
struct LongTermUpdateArgs {
    reason: String,
    evidence: Option<String>,
}

pub struct LongTermUpdateTool;

impl Tool for LongTermUpdateTool {
    fn name(&self) -> &'static str {
        "start_long_term_update"
    }

    fn description(&self) -> &'static str {
        "Start a GenericAgent-style long-term memory distillation pass after verified reusable evidence; args: reason, optional evidence."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: LongTermUpdateArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let sop_path = ctx.memory_dir.join("memory_management_sop.md");
        let sop = fs::read_to_string(&sop_path).unwrap_or_else(|_| DEFAULT_MEMORY_SOP.to_string());
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "phase": "long_term_memory_settlement",
                "reason": args.reason,
                "evidence": args.evidence,
                "sop_path": sop_path,
                "sop": sop,
                "next_prompt": "LONG_TERM_MEMORY_SETTLEMENT: choose exactly one branch now. Branch A update_l2_global_facts: use only for stable environment facts, durable user preferences, paths, or configuration; first read/fetch memory/global_facts.md, then make the smallest patch/write. Branch B update_l3_skill: use for reusable workflows or troubleshooting patterns; first memory_search for an existing skill, then memory_fetch/read the selected existing SKILL.md, then patch that existing SKILL.md when appropriate; do not create duplicate skills. Branch C skip: if evidence is unverified, temporary, generic, duplicate, or not future-useful. After the write or skip decision, call complete_long_term_update with decision, target, reason, evidence, and changed. Never write secrets.",
                "settlement_branches": [
                    {
                        "decision": "update_l2_global_facts",
                        "when": "The evidence is a stable environment fact, path, configuration, or durable user preference.",
                        "first_step": "memory_fetch id=global-facts or read_file path=memory/global_facts.md",
                        "write_step": "patch_file/write_file with the smallest local update",
                        "complete_step": "complete_long_term_update decision=update_l2_global_facts changed=true"
                    },
                    {
                        "decision": "update_l3_skill",
                        "when": "The evidence is a reusable workflow, prerequisite, failure mode, or troubleshooting pattern.",
                        "first_step": "memory_search for an existing skill, then memory_fetch/read the selected SKILL.md",
                        "write_step": "patch_file the existing skill; avoid duplicate skills",
                        "complete_step": "complete_long_term_update decision=update_l3_skill changed=true"
                    },
                    {
                        "decision": "skip",
                        "when": "The evidence is unverified, temporary, generic, duplicated, secret, or not likely future-useful.",
                        "complete_step": "complete_long_term_update decision=skip changed=false",
                        "finish_step": "finish with 'Skipped long-term memory update: <reason>'"
                    }
                ],
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct CompleteLongTermUpdateArgs {
    decision: String,
    target: Option<String>,
    reason: String,
    evidence: Option<String>,
    changed: bool,
}

pub struct CompleteLongTermUpdateTool;

impl Tool for CompleteLongTermUpdateTool {
    fn name(&self) -> &'static str {
        "complete_long_term_update"
    }

    fn description(&self) -> &'static str {
        "Audit-complete a phase 2 long-term memory settlement; args: decision update_l2_global_facts|update_l3_skill|skip, optional target, reason, optional evidence, changed."
    }

    fn execute(&self, _ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: CompleteLongTermUpdateArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        if !matches!(
            args.decision.as_str(),
            "update_l2_global_facts" | "update_l3_skill" | "skip"
        ) {
            return Ok(ToolResult::error(
                call,
                "decision must be update_l2_global_facts, update_l3_skill, or skip",
            ));
        }
        if args.decision != "skip" && args.target.as_deref().unwrap_or_default().is_empty() {
            return Ok(ToolResult::error(
                call,
                "target is required when decision changes memory",
            ));
        }
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "decision": args.decision,
                "target": args.target,
                "reason": args.reason,
                "evidence": args.evidence,
                "changed": args.changed,
            }),
        ))
    }
}

const DEFAULT_MEMORY_SOP: &str = r#"# Memory Management SOP

Only write long-term memory when the fact is verified by a successful tool call and likely useful in future tasks.

- L2 global facts: stable paths, config, environment constraints, durable user preferences.
- L3 skills: reusable workflows, exact prerequisites, common failure modes, verification commands.
- L4 session archive: completed traces kept as evidence; crystallize with `run --learn`.

Rules:
1. Read the current target first.
2. Make the smallest local update.
3. Skip memory writes for guesses, temporary variables, generic advice, or one-off outputs.
4. Prefer updating a skill over creating a duplicate when the behavior is the same.
"#;

fn memory_paths(ctx: &ToolContext) -> agent_memory::MemoryPaths {
    agent_memory::MemoryPaths::new(
        ctx.memory_dir.clone(),
        ctx.skills_dir.clone(),
        ctx.sessions_dir.clone(),
    )
}

fn durable_write_guard(
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

fn is_durable_path(ctx: &ToolContext, path: &Path) -> bool {
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

fn resolve_path(cwd: &Path, input: &str) -> PathBuf {
    let path = Path::new(input);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

fn read_file_window(
    path: &Path,
    start: usize,
    count: usize,
    keyword: Option<&str>,
    show_line_numbers: bool,
) -> anyhow::Result<String> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    let keyword = keyword.map(str::to_lowercase);

    if let Some(keyword) = keyword {
        let before_size = (count / 3).max(1);
        let mut before = VecDeque::with_capacity(before_size);
        for (idx, line) in reader.lines().enumerate().skip(start - 1) {
            let line_no = idx + 1;
            let line = line?;
            if line.to_lowercase().contains(&keyword) {
                rows.extend(before);
                rows.push((line_no, line));
                break;
            }
            if before.len() == before_size {
                before.pop_front();
            }
            before.push_back((line_no, line));
        }
    } else {
        for (idx, line) in reader.lines().enumerate().skip(start - 1).take(count) {
            rows.push((idx + 1, line?));
        }
    }

    if rows.is_empty() {
        return Ok("[FILE] no matching content".to_string());
    }

    let mut out = format!(
        "[FILE] showing {} lines from {}\n",
        rows.len(),
        path.display()
    );
    for (line_no, mut line) in rows.into_iter().take(count) {
        if line.len() > 8_000 {
            line.truncate(8_000);
            line.push_str(" ... [TRUNCATED]");
        }
        if show_line_numbers {
            out.push_str(&format!("{line_no}|{line}\n"));
        } else {
            out.push_str(&line);
            out.push('\n');
        }
    }
    Ok(out)
}

fn run_shell(command: &str, cwd: &Path, timeout_secs: u64) -> anyhow::Result<serde_json::Value> {
    let mut child = if cfg!(windows) {
        Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", command])
            .current_dir(cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?
    } else {
        Command::new("bash")
            .args(["-lc", command])
            .current_dir(cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?
    };

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let out_handle = thread::spawn(move || read_pipe(stdout));
    let err_handle = thread::spawn(move || read_pipe(stderr));
    let (timed_out, status) = match child.wait_timeout(Duration::from_secs(timeout_secs))? {
        Some(status) => (false, Some(status)),
        None => {
            child.kill()?;
            (true, child.wait().ok())
        }
    };
    let stdout = out_handle.join().unwrap_or_default();
    let stderr = err_handle.join().unwrap_or_default();
    let exit_code = status.and_then(|status| status.code());
    Ok(json!({
        "status": if !timed_out && exit_code == Some(0) { "success" } else { "error" },
        "timed_out": timed_out,
        "exit_code": exit_code,
        "stdout": truncate_middle(&stdout, 12_000),
        "stderr": truncate_middle(&stderr, 4_000),
    }))
}

fn plan_store(ctx: &ToolContext) -> agent_plan::PlanStore {
    agent_plan::PlanStore::new(ctx.cwd.join("plans"))
}

fn absolutize(cwd: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    }
}

fn repoprompt_report_text(output: &agent_repoprompt::RepoPromptOutput) -> String {
    if !output.stdout.trim().is_empty() {
        return output.stdout.clone();
    }
    if let Some(json) = &output.json {
        return serde_json::to_string_pretty(json).unwrap_or_else(|_| json.to_string());
    }
    output.stderr.clone()
}

fn repoprompt_client(
    _ctx: &ToolContext,
    routing: RepoPromptRoutingArgs,
) -> agent_repoprompt::RepoPromptClient {
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
    cfg.working_dirs = routing.working_dirs.unwrap_or_default();
    cfg.raw_json = routing.raw_json.unwrap_or(false);
    agent_repoprompt::RepoPromptClient::new(cfg)
}

fn repoprompt_output_json(output: agent_repoprompt::RepoPromptOutput) -> serde_json::Value {
    json!({
        "status": output.status(),
        "timed_out": output.timed_out,
        "exit_code": output.exit_code,
        "stdout": truncate_middle(&output.stdout, 20_000),
        "stderr": truncate_middle(&output.stderr, 6_000),
        "json": output.json,
    })
}

fn read_pipe(pipe: Option<impl Read>) -> String {
    let Some(mut pipe) = pipe else {
        return String::new();
    };
    let mut buf = String::new();
    let _ = pipe.read_to_string(&mut buf);
    buf
}

fn truncate_middle(input: &str, max_len: usize) -> String {
    if input.len() <= max_len {
        return input.to_string();
    }
    let keep = max_len / 2;
    format!(
        "{}\n...[omitted long output]...\n{}",
        &input[..keep],
        &input[input.len() - keep..]
    )
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
        assert!(names.contains(&"plan_status"));
        assert!(names.contains(&"plan_next"));
        assert!(names.contains(&"plan_complete"));
        assert!(names.contains(&"plan_verify"));
        assert!(names.contains(&"repoprompt_tools"));
        assert!(names.contains(&"repoprompt_exec"));
        assert!(names.contains(&"repoprompt_call"));
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
    fn long_term_update_returns_sop_without_writing_memory() {
        let call = ToolCall::new(
            "start_long_term_update",
            json!({
                "reason": "remember verified shell workflow",
                "evidence": "run_shell exited 0"
            }),
        );
        let ctx = ToolContext::with_cwd(".");
        let result = LongTermUpdateTool.execute(&ctx, &call).unwrap();

        assert!(result.ok);
        assert_eq!(result.content["status"], "success");
        assert_eq!(result.content["phase"], "long_term_memory_settlement");
        assert!(result.content["sop"].as_str().unwrap().contains("verified"));
        assert!(result.content["settlement_branches"].is_array());
        assert!(
            result.content["next_prompt"]
                .as_str()
                .unwrap()
                .contains("complete_long_term_update")
        );
    }

    #[test]
    fn complete_long_term_update_requires_valid_decision_and_target() {
        let valid = ToolCall::new(
            "complete_long_term_update",
            json!({
                "decision": "update_l3_skill",
                "target": "skills/demo/SKILL.md",
                "reason": "captured reusable workflow",
                "evidence": "verified by test",
                "changed": true
            }),
        );
        let result = CompleteLongTermUpdateTool
            .execute(&ToolContext::with_cwd("."), &valid)
            .unwrap();
        assert!(result.ok);
        assert_eq!(result.content["decision"], "update_l3_skill");

        let missing_target = ToolCall::new(
            "complete_long_term_update",
            json!({
                "decision": "update_l2_global_facts",
                "reason": "needs a target",
                "changed": true
            }),
        );
        let result = CompleteLongTermUpdateTool
            .execute(&ToolContext::with_cwd("."), &missing_target)
            .unwrap();
        assert!(!result.ok);
    }

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
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("seed-tools-test-{nanos}"))
    }
}
