//! Plan tools: `plan_create`, `plan_create_from_repoprompt`,
//! `plan_create_via_repoprompt`, `plan_refine_via_repoprompt`,
//! `plan_list`, `plan_status`, `plan_next`, `plan_complete`,
//! `plan_record_artifact`, `plan_record_handoff`, `plan_verify`.
//!
//! extracted from `lib.rs`. This is the largest tool family
//! (~840 LOC). Three things stay in `lib.rs` as `pub(crate)` because
//! they're shared with the RepoPrompt-bridge tools:
//!   - `REPOPROMPT_LEDGER_PROMPT` const (also used by
//!     `repoprompt_ledger_prompt_for_tool`)
//!   - `repoprompt_client` (also used by RP exec/call tools)
//!   - `repoprompt_output_json` (ditto)
//!
//! `plan_store`, `plan_mode_next_prompt`, and `plan_ledger_summary` are
//! plan-only and live in this module as private helpers.

use std::fs;
use std::path::{Path, PathBuf};

use agent_core::{Tool, ToolCall, ToolContext, ToolError, ToolResult};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    REPOPROMPT_LEDGER_PROMPT, RepoPromptRoutingArgs, absolutize, compact_single_line, non_empty,
    parse_tool_args, plan_title_from_task, repoprompt_client, repoprompt_output_json,
    repoprompt_output_string, repoprompt_report_text, truncate_text,
};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
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

    crate::tool_description!("plan_create");

    crate::impl_args_schema!(PlanCreateArgs);

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
            let args: PlanCreateArgs = parse_tool_args(call)?;

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

#[derive(Debug, Deserialize, schemars::JsonSchema)]
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

    crate::tool_description!("plan_create_from_repoprompt");

    crate::impl_args_schema!(PlanCreateFromRepoPromptArgs);

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
            let args: PlanCreateFromRepoPromptArgs = parse_tool_args(call)?;

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

#[derive(Debug, Deserialize, schemars::JsonSchema)]
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

    crate::tool_description!("plan_refine_via_repoprompt");

    crate::impl_args_schema!(PlanRefineArgs);

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
            let args: PlanRefineArgs = parse_tool_args(call)?;

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

#[derive(Debug, Deserialize, schemars::JsonSchema)]
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

    crate::tool_description!("plan_create_via_repoprompt");

    crate::impl_args_schema!(PlanCreateViaRepoPromptArgs);

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
            let args: PlanCreateViaRepoPromptArgs = parse_tool_args(call)?;

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

pub(crate) fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct PlanListArgs {
    limit: Option<usize>,
}

pub struct PlanListTool;

impl Tool for PlanListTool {
    fn name(&self) -> &'static str {
        "plan_list"
    }

    crate::tool_description!("plan_list");

    crate::impl_args_schema!(PlanListArgs);

    crate::impl_pure_read!();

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

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct PlanIdArgs {
    #[serde(default, alias = "plan_id")]
    id: Option<String>,
}

pub struct PlanStatusTool;

impl Tool for PlanStatusTool {
    fn name(&self) -> &'static str {
        "plan_status"
    }

    crate::tool_description!("plan_status");

    crate::impl_args_schema!(PlanIdArgs);

    crate::impl_pure_read!();

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

    crate::tool_description!("plan_next");

    crate::impl_args_schema!(PlanIdArgs);

    crate::impl_pure_read!();

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

#[derive(Debug, Deserialize, schemars::JsonSchema)]
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

    crate::tool_description!("plan_complete");

    crate::impl_args_schema!(PlanCompleteArgs);

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

#[derive(Debug, Deserialize, schemars::JsonSchema)]
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

    crate::tool_description!("plan_record_artifact");

    crate::impl_args_schema!(PlanRecordArtifactArgs);

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

#[derive(Debug, Deserialize, schemars::JsonSchema)]
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

    crate::tool_description!("plan_record_handoff");

    crate::impl_args_schema!(PlanRecordHandoffArgs);

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

#[derive(Debug, Deserialize, schemars::JsonSchema)]
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

    crate::tool_description!("plan_verify");

    crate::impl_args_schema!(PlanVerifyArgs);

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
            let args: PlanVerifyArgs = parse_tool_args(call)?;

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

fn plan_store(ctx: &ToolContext) -> agent_plan::PlanStore {
    agent_plan::PlanStore::new(ctx.cwd.join("plans"))
}
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
