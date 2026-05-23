use agent_core::{Tool, ToolCall, ToolContext, ToolError, ToolRegistry, ToolResult};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, SystemTime};
use wait_timeout::ChildExt;

mod subagent;
pub use subagent::{
    SEED_SUBAGENT_DEPTH_ENV, SEED_SUBAGENT_MAX_DEPTH, SEED_SUBAGENT_WATCH_DIR_ENV,
    SUBAGENT_SIGNAL_INTERVENE, SUBAGENT_SIGNAL_KEYINFO, SUBAGENT_SIGNAL_STOP, SpawnSubagentMapTool,
    SpawnSubagentTool, SubagentNudgeTool, SubagentSignals, consume_subagent_signals,
    write_subagent_signals,
};

pub fn seed_registry() -> ToolRegistry {
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
        let default_bytes = ctx.scaled_default(16_000, 4_000);
        let doc = agent_memory::fetch_memory(
            &memory_paths(ctx),
            &args.id,
            args.max_bytes.unwrap_or(default_bytes),
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
        let auto_bind = document
            .info
            .repoprompt
            .as_ref()
            .map(|binding| autobind_repoprompt(ctx, binding));
        let mut content = json!({
            "status": "success",
            "skill": document.info,
            "body": document.body,
        });
        if let Some(outcome) = auto_bind {
            content["repoprompt_autobind"] = outcome;
        }
        Ok(ToolResult::ok(call, content))
    }
}

fn autobind_repoprompt(ctx: &ToolContext, binding: &agent_skills::RepoPromptBinding) -> Value {
    let routing = RepoPromptRoutingArgs {
        raw_json: Some(true),
        ..Default::default()
    };
    let client = match repoprompt_client(ctx, routing, false) {
        Ok(client) => client,
        Err(err) => {
            return json!({ "status": "skipped", "reason": format!("RepoPrompt unavailable: {err}") });
        }
    };
    let mut payload = serde_json::Map::new();
    payload.insert("op".to_string(), Value::String("bind".to_string()));
    if !binding.working_dirs.is_empty() {
        payload.insert(
            "working_dirs".to_string(),
            Value::Array(
                binding
                    .working_dirs
                    .iter()
                    .map(|path| Value::String(path.display().to_string()))
                    .collect(),
            ),
        );
    }
    if let Some(context_id) = &binding.context_id {
        payload.insert(
            "context_id".to_string(),
            Value::String(context_id.clone()),
        );
    }
    let result = match client.call_tool(
        agent_repoprompt::RepoPromptTool::BindContext,
        &Value::Object(payload),
    ) {
        Ok(output) => output,
        Err(err) => {
            return json!({ "status": "error", "reason": format!("bind_context call failed: {err}") });
        }
    };
    let status = if result.timed_out {
        "timeout"
    } else if result.exit_code == Some(0) {
        "bound"
    } else {
        "error"
    };
    json!({
        "status": status,
        "exit_code": result.exit_code,
        "stdout_tail": truncate_text(result.stdout.trim(), 400),
        "working_dirs": binding.working_dirs,
        "context_id": binding.context_id,
    })
}

#[derive(Debug, Deserialize)]
struct PlanCreateArgs {
    #[serde(default)]
    title: Option<String>,
    #[serde(default, alias = "goal")]
    task: Option<String>,
    #[serde(default, alias = "items")]
    steps: Option<Vec<String>>,
    #[serde(default)]
    source_export_path: Option<PathBuf>,
}

pub struct PlanCreateTool;

impl Tool for PlanCreateTool {
    fn name(&self) -> &'static str {
        "plan_create"
    }

    fn description(&self) -> &'static str {
        "Create a durable GenericAgent-style plan under plans/<id>/ with plan.md, state.json, and a required verification gate. Args JSON: {\"title\":\"short title\",\"task\":\"full task\",\"steps\":[\"step 1\"]}. Accepted aliases: goal->task, items->steps. Do not pass plan_id or verification_gate."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: PlanCreateArgs = serde_json::from_value(call.args.clone()).map_err(|source| {
            ToolError::InvalidArguments {
                tool: call.name.clone(),
                source,
            }
        })?;
        let Some(task) = non_empty(args.task) else {
            return Ok(ToolResult::error(
                call,
                "plan_create requires `task` (or alias `goal`)",
            ));
        };
        let title = non_empty(args.title).unwrap_or_else(|| plan_title_from_task(&task));
        let snapshot = plan_store(ctx)
            .create(agent_plan::CreatePlan {
                title,
                task,
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
                "ledger_summary": plan_ledger_summary(&snapshot),
                "next_prompt": plan_mode_next_prompt(&snapshot),
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct PlanCreateFromRepoPromptArgs {
    #[serde(alias = "path")]
    export_path: PathBuf,
    #[serde(default)]
    title: Option<String>,
    #[serde(default, alias = "goal")]
    task: Option<String>,
}

pub struct PlanCreateFromRepoPromptTool;

impl Tool for PlanCreateFromRepoPromptTool {
    fn name(&self) -> &'static str {
        "plan_create_from_repoprompt"
    }

    fn description(&self) -> &'static str {
        "Import a RepoPrompt builder plan export (`builder ... --response-type plan --export`) into a durable seed plan. Parses the export's `## Plan/Steps/Implementation` section into checked items, applies [D]/[P] markers via keyword heuristics, creates the plan, and records the export under the plan's RepoPromptExport artifact ledger. Args: {\"export_path\":\"path/to/export.md\", \"title\":\"optional\", \"task\":\"optional\"}."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: PlanCreateFromRepoPromptArgs = serde_json::from_value(call.args.clone())
            .map_err(|source| ToolError::InvalidArguments {
                tool: call.name.clone(),
                source,
            })?;
        let export_path = absolutize(&ctx.cwd, args.export_path);
        if !export_path.is_file() {
            return Ok(ToolResult::error(
                call,
                format!("export file not found: {}", export_path.display()),
            ));
        }
        let text = fs::read_to_string(&export_path)
            .map_err(|err| ToolError::Failed(format!("read {}: {err}", export_path.display())))?;
        let imported = agent_plan::import_repoprompt_plan(&text);
        if imported.steps.is_empty() {
            return Ok(ToolResult::error(
                call,
                "no plan steps detected in export; expected a `## Plan` (or Steps/Tasks/Implementation) section with list items",
            ));
        }
        let title = non_empty(args.title).unwrap_or(imported.title);
        let task = non_empty(args.task).unwrap_or(imported.task);
        let store = plan_store(ctx);
        let snapshot = store
            .create(agent_plan::CreatePlan {
                title,
                task,
                steps: imported.steps.clone(),
                source_export_path: Some(export_path.clone()),
            })
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let snapshot = store
            .record_artifact(
                Some(&snapshot.state.id),
                agent_plan::RecordPlanArtifact {
                    kind: agent_plan::PlanArtifactKind::RepoPromptExport,
                    path: export_path.clone(),
                    note: Some(format!(
                        "Imported {} steps from RepoPrompt export ({} delegated, {} parallel)",
                        imported.steps.len(),
                        imported.delegated_count,
                        imported.parallel_count
                    )),
                },
            )
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "plan": snapshot,
                "ledger_summary": plan_ledger_summary(&snapshot),
                "next_prompt": plan_mode_next_prompt(&snapshot),
                "import_stats": {
                    "steps_total": imported.steps.len(),
                    "delegated": imported.delegated_count,
                    "parallel": imported.parallel_count,
                    "export_path": export_path,
                },
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct PlanRefineArgs {
    #[serde(default, alias = "id", alias = "plan_id")]
    plan: Option<String>,
    #[serde(default)]
    focus: Option<String>,
    #[serde(default)]
    max_fixes: Option<usize>,
    #[serde(default)]
    chat_id: Option<String>,
    #[serde(default)]
    working_dirs: Option<Vec<PathBuf>>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

pub struct PlanRefineViaRepoPromptTool;

impl Tool for PlanRefineViaRepoPromptTool {
    fn name(&self) -> &'static str {
        "plan_refine_via_repoprompt"
    }

    fn description(&self) -> &'static str {
        "Ask RepoPrompt's oracle to review the current plan and append concrete [FIX] items. Args: plan (id; default=latest active), focus (optional string), max_fixes (default 8), chat_id (continue a prior review chat), working_dirs (override binding), timeout_secs. The oracle returns markdown with a `## Recommended Fixes` section; each item gets appended as a numbered [FIX] step before the [VERIFY] gate and a reviewer handoff is logged on the plan."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: PlanRefineArgs = serde_json::from_value(call.args.clone()).map_err(|source| {
            ToolError::InvalidArguments {
                tool: call.name.clone(),
                source,
            }
        })?;
        let store = plan_store(ctx);
        let snapshot = store
            .snapshot(args.plan.as_deref())
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let plan_body = fs::read_to_string(&snapshot.state.plan_path).map_err(|err| {
            ToolError::Failed(format!(
                "read {}: {err}",
                snapshot.state.plan_path.display()
            ))
        })?;
        let max_fixes = args.max_fixes.unwrap_or(8).clamp(1, 30);
        let focus_block = args
            .focus
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|focus| format!("\n<focus>{}</focus>", escape_xml(focus)))
            .unwrap_or_default();
        let message = format!(
            "You are reviewing an implementation plan that an autonomous agent will execute. \
Look for gaps (missing steps, missing verification), risks (steps that could break things, \
untested assumptions), ambiguity (steps too vague to execute), and ordering issues.\n\n\
<plan>\n{plan_body}\n</plan>{focus_block}\n\n\
Respond with EXACTLY this structure:\n\n\
## Findings\n- 2-5 bullets naming specific concerns.\n\n\
## Recommended Fixes\n- Up to {max_fixes} one-line action items. Each MUST start with an imperative verb \
(Add, Remove, Change, Replace, Investigate, Verify, Update, Refactor). These will be appended verbatim as new \
[FIX] plan steps. Make them executable, not philosophical. Do not repeat existing plan steps. \
If the plan is already complete and needs no fixes, write `- (none)` in this section."
        );

        let timeout = args.timeout_secs.unwrap_or(600).clamp(60, 3600);
        let routing = RepoPromptRoutingArgs {
            timeout_secs: Some(timeout),
            working_dirs: args.working_dirs.clone(),
            raw_json: Some(true),
            ..Default::default()
        };
        let client = repoprompt_client(ctx, routing, true)?;
        let new_chat = args.chat_id.is_none();
        let response = client
            .send_oracle(
                &message,
                agent_repoprompt::OracleMode::Chat,
                args.chat_id.as_deref(),
                new_chat,
            )
            .map_err(|err| ToolError::Failed(format!("oracle_send failed: {err}")))?;
        if !response.is_success() {
            return Ok(ToolResult::error(
                call,
                format!(
                    "oracle_send returned exit_code={:?}; stderr: {}",
                    response.raw_output.exit_code,
                    truncate_text(response.raw_output.stderr.trim(), 800)
                ),
            ));
        }

        let mut fixes = agent_plan::parse_plan_review(&response.response_text);
        if fixes.len() > max_fixes {
            fixes.truncate(max_fixes);
        }
        if fixes.is_empty() {
            return Ok(ToolResult::ok(
                call,
                json!({
                    "status": "no_fixes",
                    "plan_id": snapshot.state.id,
                    "reviewer_chat_id": response.chat_id,
                    "review_summary": truncate_text(&response.response_text, 1200),
                }),
            ));
        }

        let updated = store
            .append_items(Some(&snapshot.state.id), fixes.clone())
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let summary = format!(
            "Appended {} [FIX] items via RepoPrompt oracle review.",
            fixes.len()
        );
        let updated = store
            .record_handoff(
                Some(&updated.state.id),
                agent_plan::RecordPlanHandoff {
                    backend: "repoprompt".to_string(),
                    role: Some("reviewer".to_string()),
                    run_id: response.chat_id.clone(),
                    thread_id: response.chat_id.clone(),
                    artifact_path: response.oracle_export_path.clone(),
                    status: "completed".to_string(),
                    summary: summary.clone(),
                },
            )
            .map_err(|err| ToolError::Failed(err.to_string()))?;

        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "plan": updated,
                "ledger_summary": plan_ledger_summary(&updated),
                "next_prompt": plan_mode_next_prompt(&updated),
                "fixes_appended": fixes,
                "fix_count": fixes.len(),
                "reviewer_chat_id": response.chat_id,
                "review_summary": truncate_text(&response.response_text, 1200),
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct PlanCreateViaRepoPromptArgs {
    task: String,
    #[serde(default)]
    context: Option<String>,
    #[serde(default)]
    hints: Option<String>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    working_dirs: Option<Vec<PathBuf>>,
    #[serde(default)]
    context_id: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

pub struct PlanCreateViaRepoPromptTool;

impl Tool for PlanCreateViaRepoPromptTool {
    fn name(&self) -> &'static str {
        "plan_create_via_repoprompt"
    }

    fn description(&self) -> &'static str {
        "One-shot: ask RepoPrompt's context_builder to draft an implementation plan for `task`, then import the export into a durable seed plan. Args: task (required), optional context (background/constraints), hints (discovery agent guidance), title, working_dirs, context_id, timeout_secs (default 900). Use this when you have a fresh task; use plan_create_from_repoprompt when you already have a builder export on disk."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: PlanCreateViaRepoPromptArgs = serde_json::from_value(call.args.clone())
            .map_err(|source| ToolError::InvalidArguments {
                tool: call.name.clone(),
                source,
            })?;
        let task_text = args.task.trim();
        if task_text.is_empty() {
            return Ok(ToolResult::error(call, "task must not be empty"));
        }

        let mut instructions = format!("<task>{}</task>", escape_xml(task_text));
        if let Some(context) = args.context.as_deref()
            && !context.trim().is_empty()
        {
            instructions.push('\n');
            instructions.push_str(&format!("<context>{}</context>", escape_xml(context)));
        }
        if let Some(hints) = args.hints.as_deref()
            && !hints.trim().is_empty()
        {
            instructions.push('\n');
            instructions.push_str(&format!(
                "<discovery_agent-guidelines>{}</discovery_agent-guidelines>",
                escape_xml(hints)
            ));
        }

        let timeout = args.timeout_secs.unwrap_or(900).clamp(60, 3600);
        let routing = RepoPromptRoutingArgs {
            timeout_secs: Some(timeout),
            working_dirs: args.working_dirs.clone(),
            context_id: args.context_id.clone(),
            raw_json: Some(true),
            ..Default::default()
        };
        let client = repoprompt_client(ctx, routing, true)?;
        let response = client
            .build_context(
                &instructions,
                agent_repoprompt::BuilderResponseType::Plan,
                true,
            )
            .map_err(|err| ToolError::Failed(format!("context_builder failed: {err}")))?;
        if !response.is_success() {
            return Ok(ToolResult::error(
                call,
                format!(
                    "context_builder returned exit_code={:?} timed_out={}; stderr: {}",
                    response.raw_output.exit_code,
                    response.raw_output.timed_out,
                    truncate_text(response.raw_output.stderr.trim(), 800)
                ),
            ));
        }

        let export_path = match response.oracle_export_path.clone() {
            Some(path) => path,
            None => {
                return Ok(ToolResult::error(
                    call,
                    format!(
                        "context_builder did not return oracle_export_path; raw stdout tail: {}",
                        truncate_text(response.raw_output.stdout.trim(), 600)
                    ),
                ));
            }
        };
        let export_text = fs::read_to_string(&export_path).map_err(|err| {
            ToolError::Failed(format!(
                "read context_builder export {}: {err}",
                export_path.display()
            ))
        })?;
        let imported = agent_plan::import_repoprompt_plan(&export_text);
        if imported.steps.is_empty() {
            return Ok(ToolResult::error(
                call,
                format!(
                    "context_builder export at {} contained no recognizable plan steps; raw response: {}",
                    export_path.display(),
                    truncate_text(&response.response_text, 600)
                ),
            ));
        }
        let title = non_empty(args.title)
            .or_else(|| non_empty(Some(imported.title.clone())))
            .unwrap_or_else(|| plan_title_from_task(task_text));
        let task = non_empty(Some(imported.task.clone())).unwrap_or_else(|| task_text.to_string());
        let store = plan_store(ctx);
        let snapshot = store
            .create(agent_plan::CreatePlan {
                title,
                task,
                steps: imported.steps.clone(),
                source_export_path: Some(export_path.clone()),
            })
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let snapshot = store
            .record_artifact(
                Some(&snapshot.state.id),
                agent_plan::RecordPlanArtifact {
                    kind: agent_plan::PlanArtifactKind::RepoPromptExport,
                    path: export_path.clone(),
                    note: Some(format!(
                        "Built via context_builder; {} steps ({} delegated, {} parallel)",
                        imported.steps.len(),
                        imported.delegated_count,
                        imported.parallel_count
                    )),
                },
            )
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "plan": snapshot,
                "ledger_summary": plan_ledger_summary(&snapshot),
                "next_prompt": plan_mode_next_prompt(&snapshot),
                "import_stats": {
                    "steps_total": imported.steps.len(),
                    "delegated": imported.delegated_count,
                    "parallel": imported.parallel_count,
                    "export_path": export_path,
                    "builder_chat_id": response.chat_id,
                },
            }),
        ))
    }
}

fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[derive(Debug, Deserialize)]
struct PlanListArgs {
    limit: Option<usize>,
}

pub struct PlanListTool;

impl Tool for PlanListTool {
    fn name(&self) -> &'static str {
        "plan_list"
    }

    fn description(&self) -> &'static str {
        "List durable plans newest-first so an agent can resume, inspect, or choose a plan by id. Args JSON: {\"limit\":20}; empty args are allowed."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: PlanListArgs =
            serde_json::from_value(call.args.clone()).unwrap_or(PlanListArgs { limit: Some(20) });
        let limit = args.limit.unwrap_or(20);
        let plans = plan_store(ctx)
            .list()
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let shown = if limit == 0 {
            plans.clone()
        } else {
            plans.iter().take(limit).cloned().collect()
        };
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "total_count": plans.len(),
                "shown_count": shown.len(),
                "plans": shown,
                "next_prompt": "Choose a plan id, then call plan_status or plan_next with that id before continuing plan work.",
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct PlanIdArgs {
    #[serde(default, alias = "plan_id")]
    id: Option<String>,
}

pub struct PlanStatusTool;

impl Tool for PlanStatusTool {
    fn name(&self) -> &'static str {
        "plan_status"
    }

    fn description(&self) -> &'static str {
        "Read the current or selected plan state, checkbox items, and next unchecked item. Args JSON: {\"id\":\"plan-...\"}; alias plan_id is accepted; empty args read the active plan."
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
        "Return the next unchecked plan item; use this before continuing a plan-mode task. Args JSON: {\"id\":\"plan-...\"}; alias plan_id is accepted; empty args use the active plan."
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
                "ledger_summary": plan_ledger_summary(&snapshot),
                "next_prompt": plan_mode_next_prompt(&snapshot),
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct PlanCompleteArgs {
    #[serde(default, alias = "plan_id")]
    id: Option<String>,
    #[serde(default, alias = "index", alias = "item_index")]
    item: Option<usize>,
    #[serde(default)]
    note: Option<String>,
}

pub struct PlanCompleteTool;

impl Tool for PlanCompleteTool {
    fn name(&self) -> &'static str {
        "plan_complete"
    }

    fn description(&self) -> &'static str {
        "Mark one plan item complete by item index, or mark the current next item when omitted. Args JSON: {\"id\":\"plan-...\",\"item\":1,\"note\":\"done\"}. Aliases: plan_id->id, item_index/index->item."
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
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "plan": snapshot,
                "ledger_summary": plan_ledger_summary(&snapshot),
                "next_prompt": plan_mode_next_prompt(&snapshot),
            }),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct PlanRecordArtifactArgs {
    #[serde(default, alias = "plan_id")]
    id: Option<String>,
    kind: agent_plan::PlanArtifactKind,
    path: PathBuf,
    #[serde(default)]
    note: Option<String>,
}

pub struct PlanRecordArtifactTool;

impl Tool for PlanRecordArtifactTool {
    fn name(&self) -> &'static str {
        "plan_record_artifact"
    }

    fn description(&self) -> &'static str {
        "Record a RepoPrompt/context/verification artifact path in the plan orchestration ledger. Args JSON: {\"id\":\"plan-...\",\"kind\":\"repoprompt_export\",\"path\":\"/abs/file.md\",\"note\":\"optional\"}. Alias plan_id is accepted."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: PlanRecordArtifactArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let snapshot = plan_store(ctx)
            .record_artifact(
                args.id.as_deref(),
                agent_plan::RecordPlanArtifact {
                    kind: args.kind,
                    path: absolutize(&ctx.cwd, args.path),
                    note: args.note,
                },
            )
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

#[derive(Debug, Deserialize)]
struct PlanRecordHandoffArgs {
    #[serde(default, alias = "plan_id")]
    id: Option<String>,
    backend: String,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    thread_id: Option<String>,
    #[serde(default)]
    artifact_path: Option<PathBuf>,
    #[serde(default)]
    status: Option<String>,
    summary: String,
}

pub struct PlanRecordHandoffTool;

impl Tool for PlanRecordHandoffTool {
    fn name(&self) -> &'static str {
        "plan_record_handoff"
    }

    fn description(&self) -> &'static str {
        "Record a RepoPrompt/Codex/local execution handoff, run id, artifact, status, and summary in the plan ledger. Args JSON: {\"id\":\"plan-...\",\"backend\":\"repoprompt|codex|local\",\"summary\":\"what happened\",\"status\":\"done\"}. Alias plan_id is accepted."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: PlanRecordHandoffArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let snapshot = plan_store(ctx)
            .record_handoff(
                args.id.as_deref(),
                agent_plan::RecordPlanHandoff {
                    backend: args.backend,
                    role: args.role,
                    run_id: args.run_id,
                    thread_id: args.thread_id,
                    artifact_path: args.artifact_path.map(|path| absolutize(&ctx.cwd, path)),
                    status: args.status.unwrap_or_else(|| "recorded".to_string()),
                    summary: args.summary,
                },
            )
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

#[derive(Debug, Deserialize)]
struct PlanVerifyArgs {
    #[serde(default, alias = "plan_id")]
    id: Option<String>,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    dry_run: Option<bool>,
    #[serde(default)]
    window_id: Option<u32>,
    #[serde(default)]
    context_id: Option<String>,
    #[serde(default)]
    working_dirs: Option<Vec<PathBuf>>,
}

pub struct PlanVerifyTool;

impl Tool for PlanVerifyTool {
    fn name(&self) -> &'static str {
        "plan_verify"
    }

    fn description(&self) -> &'static str {
        "Create verify_context.json and run an independent RepoPrompt agent_run verifier; PASS marks the plan verified, FAIL appends a [FIX] item. Args JSON: {\"id\":\"plan-...\",\"dry_run\":false,\"timeout_secs\":300}. Alias plan_id is accepted."
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
        let model_id = args.model_id.unwrap_or_else(|| "pair".to_string());
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
        let output = repoprompt_client(ctx, routing, true)?
            .call_tool(
                agent_repoprompt::RepoPromptTool::AgentRun,
                &json!({
                    "op": "start",
                    "model_id": model_id.clone(),
                    "message": message,
                    "timeout": timeout_secs,
                }),
            )
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let output_status = output.status().to_string();
        let run_id = repoprompt_output_string(&output, &["run_id", "runId", "agent_run_id"]);
        let thread_id =
            repoprompt_output_string(&output, &["thread_id", "threadId", "chat_id", "chatId"]);
        let report = repoprompt_report_text(&output);
        store
            .record_verification(Some(&verify_context.plan_id), &report)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let verify_context_path = verify_context
            .plan_file
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("verify_context.json");
        let snapshot = store
            .record_handoff(
                Some(&verify_context.plan_id),
                agent_plan::RecordPlanHandoff {
                    backend: "repoprompt".to_string(),
                    role: Some(model_id),
                    run_id,
                    thread_id,
                    artifact_path: Some(verify_context_path),
                    status: output_status,
                    summary: compact_single_line(&report, 500),
                },
            )
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
    #[serde(alias = "cmd")]
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
        "Execute a RepoPrompt CLI command chain such as windows, tree, search, select, context, builder, plan, or review. Args JSON: {\"command\":\"tree --mode folders\"}; alias cmd is accepted. Workspace commands default to the current cwd when no routing is supplied."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: RepoPromptExecArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let default_cwd = default_cwd_for_repoprompt_exec(&args.command);
        let client = repoprompt_client(ctx, args.routing, default_cwd)?;
        let output = client
            .exec(&args.command)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let mut content = repoprompt_output_json(output);
        attach_repoprompt_protocol_hint(
            &mut content,
            repoprompt_ledger_prompt_for_exec(&args.command),
            client.config(),
        );
        Ok(ToolResult::ok(call, content))
    }
}

#[derive(Debug, Deserialize)]
struct RepoPromptCallArgs {
    // The planner frequently bleeds its own `PlannedAction.tool_name` field
    // into the inner envelope — accept both spellings so we don't waste a
    // retry turn on what is just a naming conflict.
    #[serde(alias = "tool_name", alias = "name")]
    tool: String,
    #[serde(default, alias = "args_json", alias = "params")]
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
        "Call any wrapped RepoPrompt MCP tool by name with JSON args; workspace tools default to the current cwd when no routing is supplied."
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
        let client = repoprompt_client(ctx, routing, default_cwd_for_repoprompt_tool(tool))?;
        let output = client
            .call_tool(tool, &args.args)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let mut content = repoprompt_output_json(output);
        attach_repoprompt_protocol_hint(
            &mut content,
            repoprompt_ledger_prompt_for_tool(tool),
            client.config(),
        );
        Ok(ToolResult::ok(call, content))
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
        let default_count = ctx.scaled_default(200, 60);
        let count = args.count.unwrap_or(default_count).clamp(1, 1000);
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
struct ReadFilesArgs {
    paths: Vec<String>,
    #[serde(default)]
    start: Option<usize>,
    #[serde(default)]
    count: Option<usize>,
    #[serde(default)]
    keyword: Option<String>,
    #[serde(default)]
    show_line_numbers: Option<bool>,
}

pub struct ReadFilesTool;

impl Tool for ReadFilesTool {
    fn name(&self) -> &'static str {
        "read_files"
    }

    fn description(&self) -> &'static str {
        "Batch-read up to 8 UTF-8 files in one tool call. Prefer this over multiple sequential read_file turns when you already know the paths you need (e.g. surveying several crate entry points). Args: paths (string[], required), start/count/keyword/show_line_numbers (applied to every file). Returns { files: [{path, status, content?, error?}], succeeded, failed }."
    }

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: ReadFilesArgs = serde_json::from_value(call.args.clone()).map_err(|source| {
            ToolError::InvalidArguments {
                tool: call.name.clone(),
                source,
            }
        })?;
        if args.paths.is_empty() {
            return Ok(ToolResult::error(call, "paths must not be empty"));
        }
        let paths = if args.paths.len() > 8 {
            return Ok(ToolResult::error(
                call,
                format!(
                    "read_files capped at 8 paths per call; got {}. Split the request.",
                    args.paths.len()
                ),
            ));
        } else {
            args.paths
        };

        let start = args.start.unwrap_or(1).max(1);
        // Per-file scaling: as we read more files in one turn, shrink each
        // file's window so total output stays bounded.
        let base_default = ctx.scaled_default(200, 60);
        let per_file_default = (base_default / paths.len().max(1)).max(40);
        let count = args.count.unwrap_or(per_file_default).clamp(1, 1000);
        let show_line_numbers = args.show_line_numbers.unwrap_or(true);

        let mut files: Vec<Value> = Vec::with_capacity(paths.len());
        let mut succeeded = 0usize;
        for raw_path in &paths {
            let path = resolve_path(&ctx.cwd, raw_path);
            match read_file_window(&path, start, count, args.keyword.as_deref(), show_line_numbers)
            {
                Ok(content) => {
                    succeeded += 1;
                    files.push(json!({
                        "path": path,
                        "status": "ok",
                        "content": content,
                    }));
                }
                Err(err) => {
                    files.push(json!({
                        "path": path,
                        "status": "error",
                        "error": err.to_string(),
                    }));
                }
            }
        }
        let total = files.len();
        Ok(ToolResult::ok(
            call,
            json!({
                "status": if succeeded == total { "success" } else { "partial" },
                "succeeded": succeeded,
                "failed": total - succeeded,
                "count_per_file": count,
                "files": files,
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
            truncate_utf8(&mut line, 8_000);
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

const REPOPROMPT_LEDGER_PROMPT: &str = "REPOPROMPT_LEDGER: if RepoPrompt produces an export path, call plan_record_artifact before using it; if RepoPrompt agent_run or Codex performs plan work, call plan_record_handoff with backend, role/run/thread id when known, artifact_path, status, and summary.";

fn plan_mode_next_prompt(snapshot: &agent_plan::PlanSnapshot) -> String {
    let phase = if snapshot.task_unchecked_count == 0 && snapshot.next_item.is_some() {
        "PLAN_VERIFY_REQUIRED: all non-verify plan items are complete. Call plan_verify now; do not finish until the independent verification gate returns PASS."
    } else {
        "PLAN_MODE: execute only the next unchecked item from plan_next. After each meaningful change, call plan_complete for the completed item. Do not skip verification."
    };
    format!("{phase} {REPOPROMPT_LEDGER_PROMPT}")
}

fn plan_ledger_summary(snapshot: &agent_plan::PlanSnapshot) -> Value {
    let orchestration = &snapshot.state.orchestration;
    json!({
        "preferred_backend": orchestration.preferred_backend,
        "artifact_count": orchestration.artifacts.len(),
        "handoff_count": orchestration.handoffs.len(),
        "verification_record_count": orchestration.verification_records.len(),
        "has_repoprompt_export": orchestration.repoprompt_export_path.is_some()
            || orchestration.artifacts.iter().any(|artifact| artifact.kind == agent_plan::PlanArtifactKind::RepoPromptExport),
        "has_context_export": orchestration.artifacts.iter().any(|artifact| artifact.kind == agent_plan::PlanArtifactKind::ContextExport),
        "latest_artifact": orchestration.artifacts.last(),
        "latest_handoff": orchestration.handoffs.last(),
        "latest_verification": orchestration.verification_records.last(),
    })
}

fn absolutize(cwd: &Path, path: PathBuf) -> PathBuf {
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
        working_dirs.push(ctx.cwd.clone());
    }
    working_dirs
        .into_iter()
        .map(|path| absolutize(&ctx.cwd, path))
        .collect()
}

fn default_cwd_for_repoprompt_exec(command: &str) -> bool {
    let command = command.trim().to_ascii_lowercase();
    !(command == "windows"
        || command.starts_with("windows ")
        || command == "workspace list"
        || command.starts_with("workspace list ")
        || command == "workspaces"
        || command.starts_with("bind_context")
        || command.starts_with("app_settings"))
}

fn default_cwd_for_repoprompt_tool(tool: agent_repoprompt::RepoPromptTool) -> bool {
    !matches!(
        tool,
        agent_repoprompt::RepoPromptTool::BindContext
            | agent_repoprompt::RepoPromptTool::ManageWorkspaces
            | agent_repoprompt::RepoPromptTool::AppSettings
            | agent_repoprompt::RepoPromptTool::OracleUtils
            | agent_repoprompt::RepoPromptTool::AgentManage
    )
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
    ctx: &ToolContext,
    routing: RepoPromptRoutingArgs,
    default_cwd: bool,
) -> Result<agent_repoprompt::RepoPromptClient, ToolError> {
    let mut cfg = repoprompt_config(ctx, routing, default_cwd);
    resolve_repoprompt_window(&mut cfg)?;
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
                "create_if_missing": false,
            }),
        )
        .map_err(|err| ToolError::Failed(format!("RepoPrompt bind failed: {err}")))?;
    if output.timed_out || output.exit_code != Some(0) {
        return Err(ToolError::Failed(format!(
            "RepoPrompt bind failed before routed call: {}",
            compact_single_line(&repoprompt_report_text(&output), 800)
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

fn repoprompt_ledger_prompt_for_exec(command: &str) -> &'static str {
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

fn repoprompt_ledger_prompt_for_tool(tool: agent_repoprompt::RepoPromptTool) -> &'static str {
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

fn attach_repoprompt_protocol_hint(
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

fn repoprompt_output_string(
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

fn compact_single_line(input: &str, max_len: usize) -> String {
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

fn read_pipe(pipe: Option<impl Read>) -> String {
    let Some(mut pipe) = pipe else {
        return String::new();
    };
    let mut buf = String::new();
    let _ = pipe.read_to_string(&mut buf);
    buf
}

fn non_empty(value: Option<String>) -> Option<String> {
    value
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

fn plan_title_from_task(task: &str) -> String {
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

fn truncate_utf8(text: &mut String, limit: usize) {
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


#[derive(Debug, Deserialize)]
struct AskUserArgs {
    question: String,
    #[serde(default)]
    candidates: Vec<String>,
}

pub struct AskUserTool;

impl Tool for AskUserTool {
    fn name(&self) -> &'static str {
        "ask_user"
    }

    fn description(&self) -> &'static str {
        "Prompt the human operator via stdin when the task cannot proceed without clarification or a decision. Args: question (string), optional candidates (list of suggested answers). Fails on non-interactive stdin so it cannot deadlock CI."
    }

    fn execute(&self, _ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: AskUserArgs = serde_json::from_value(call.args.clone()).map_err(|source| {
            ToolError::InvalidArguments {
                tool: call.name.clone(),
                source,
            }
        })?;
        use std::io::IsTerminal;
        let stdin = std::io::stdin();
        if !stdin.is_terminal() {
            return Ok(ToolResult::error(
                call,
                "stdin is not a terminal; ask_user cannot collect a response in non-interactive mode",
            ));
        }
        let mut stderr = std::io::stderr().lock();
        let _ = writeln!(stderr, "\nseed asks: {}", args.question);
        if !args.candidates.is_empty() {
            for (idx, candidate) in args.candidates.iter().enumerate() {
                let _ = writeln!(stderr, "  {}) {}", idx + 1, candidate);
            }
            let _ = writeln!(stderr, "  reply with a number or your own answer.");
        }
        let _ = write!(stderr, "> ");
        let _ = stderr.flush();

        let mut line = String::new();
        stdin
            .lock()
            .read_line(&mut line)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let trimmed = line.trim().to_string();
        if trimmed.is_empty() {
            return Ok(ToolResult::error(call, "user replied with an empty line"));
        }

        let resolved = if !args.candidates.is_empty()
            && let Ok(idx) = trimmed.parse::<usize>()
            && idx >= 1
            && idx <= args.candidates.len()
        {
            args.candidates[idx - 1].clone()
        } else {
            trimmed.clone()
        };

        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "question": args.question,
                "answer": resolved,
                "raw_input": trimmed,
            }),
        ))
    }
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
        let escaped = escape_xml("3 < 5 && x > y");
        assert_eq!(escaped, "3 &lt; 5 &amp;&amp; x &gt; y");
    }

    #[test]
    fn registry_exposes_plan_create_via_repoprompt() {
        let registry = seed_registry();
        let names = registry.names();
        assert!(names.contains(&"plan_create_via_repoprompt"));
        assert!(names.contains(&"plan_refine_via_repoprompt"));
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
        // SAFETY: tests in this binary run single-threaded for env mutation by convention here.
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

    #[test]
    fn repoprompt_exec_accepts_cmd_alias() {
        let args: RepoPromptExecArgs = serde_json::from_value(json!({
            "cmd": "tree --mode folders"
        }))
        .unwrap();

        assert_eq!(args.command, "tree --mode folders");
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
