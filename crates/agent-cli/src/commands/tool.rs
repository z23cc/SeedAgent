//! `seed tool ...` one-shot subcommand: parses the typed `ToolCommand`
//! sub-enum into a generic `ToolCall`, runs it through `commands::exec`
//! against a fresh session, prints the JSON result. No LLM, no loop.

use std::env;
use std::path::PathBuf;

use agent_core::{AgentEvent, ToolCall};
use agent_session::SessionStore;
use anyhow::Result;
use clap::{Subcommand, ValueEnum};
use serde_json::json;

use crate::commands::exec::execute_call;
use crate::commands::plan::PlanArtifactKindArg;

#[derive(Debug, Subcommand)]
pub(crate) enum ToolCommand {
    MemorySearch {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    MemoryFetch {
        id: String,
        #[arg(long, default_value_t = 16000)]
        max_bytes: usize,
    },
    SkillList {
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    SkillSearch {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    SkillFetch {
        name: String,
    },
    #[command(name = "plan-create")]
    PlanCreate {
        #[arg(long)]
        title: String,
        #[arg(long)]
        task: String,
        #[arg(long = "step")]
        steps: Vec<String>,
        #[arg(long)]
        source_export_path: Option<PathBuf>,
    },
    #[command(name = "plan-create-from-repoprompt")]
    PlanCreateFromRepoPrompt {
        #[arg(long)]
        path: PathBuf,
        #[arg(long)]
        title: Option<String>,
        #[arg(long)]
        task: Option<String>,
    },
    #[command(name = "plan-create-via-repoprompt")]
    PlanCreateViaRepoPrompt {
        #[arg(long)]
        task: String,
        #[arg(long)]
        context: Option<String>,
        #[arg(long)]
        hints: Option<String>,
        #[arg(long)]
        title: Option<String>,
        #[arg(long = "working-dir")]
        working_dirs: Vec<PathBuf>,
        #[arg(long)]
        context_id: Option<String>,
        #[arg(long, default_value_t = 900)]
        timeout_secs: u64,
    },
    #[command(name = "plan-refine-via-repoprompt")]
    PlanRefineViaRepoPrompt {
        #[arg(long, alias = "id", alias = "plan-id")]
        plan: Option<String>,
        #[arg(long)]
        focus: Option<String>,
        #[arg(long, default_value_t = 8)]
        max_fixes: usize,
        #[arg(long)]
        chat_id: Option<String>,
        #[arg(long = "working-dir")]
        working_dirs: Vec<PathBuf>,
        #[arg(long, default_value_t = 600)]
        timeout_secs: u64,
    },
    #[command(name = "plan-list", alias = "plan-ls")]
    PlanList {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    #[command(name = "plan-status")]
    PlanStatus {
        id: Option<String>,
    },
    #[command(name = "plan-next")]
    PlanNext {
        id: Option<String>,
    },
    #[command(name = "plan-complete")]
    PlanComplete {
        id: Option<String>,
        #[arg(long)]
        item: Option<usize>,
        #[arg(long)]
        note: Option<String>,
    },
    #[command(name = "plan-record-artifact")]
    PlanRecordArtifact {
        id: Option<String>,
        #[arg(long, value_enum)]
        kind: PlanArtifactKindArg,
        #[arg(long)]
        path: PathBuf,
        #[arg(long)]
        note: Option<String>,
    },
    #[command(name = "plan-record-handoff")]
    PlanRecordHandoff {
        id: Option<String>,
        #[arg(long, default_value = "repoprompt")]
        backend: String,
        #[arg(long)]
        role: Option<String>,
        #[arg(long)]
        run_id: Option<String>,
        #[arg(long)]
        thread_id: Option<String>,
        #[arg(long)]
        artifact_path: Option<PathBuf>,
        #[arg(long, default_value = "recorded")]
        status: String,
        #[arg(long)]
        summary: String,
    },
    #[command(name = "plan-verify")]
    PlanVerify {
        id: Option<String>,
        #[arg(long, default_value = "pair")]
        model_id: String,
        #[arg(long, default_value_t = 300)]
        timeout_secs: u64,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        window_id: Option<u32>,
        #[arg(long)]
        context_id: Option<String>,
        #[arg(long = "working-dir")]
        working_dirs: Vec<PathBuf>,
    },
    #[command(name = "repoprompt-tools", alias = "repo-prompt-tools")]
    RepoPromptTools,
    #[command(name = "repoprompt-exec", alias = "repo-prompt-exec")]
    RepoPromptExec {
        command: String,
        #[arg(long)]
        window_id: Option<u32>,
        #[arg(long)]
        context_id: Option<String>,
        #[arg(long = "working-dir")]
        working_dirs: Vec<PathBuf>,
        #[arg(long, default_value_t = 300)]
        timeout_secs: u64,
        #[arg(long)]
        raw_json: bool,
    },
    #[command(name = "repoprompt-call", alias = "repo-prompt-call")]
    RepoPromptCall {
        tool: String,
        #[arg(long, default_value = "{}")]
        args: String,
        #[arg(long)]
        window_id: Option<u32>,
        #[arg(long)]
        context_id: Option<String>,
        #[arg(long = "working-dir")]
        working_dirs: Vec<PathBuf>,
        #[arg(long, default_value_t = 300)]
        timeout_secs: u64,
    },
    ReadFile {
        path: String,
        #[arg(long, default_value_t = 1)]
        start: usize,
        #[arg(long, default_value_t = 200)]
        count: usize,
        #[arg(long)]
        keyword: Option<String>,
    },
    PatchFile {
        path: String,
        #[arg(long)]
        old_content: String,
        #[arg(long)]
        new_content: String,
    },
    WriteFile {
        path: String,
        #[arg(long)]
        content: String,
        #[arg(long, value_enum, default_value_t = WriteModeArg::Overwrite)]
        mode: WriteModeArg,
    },
    RunShell {
        command: String,
        #[arg(long, default_value_t = 60)]
        timeout_secs: u64,
    },
    UpdateWorkingCheckpoint {
        key_info: String,
        #[arg(long)]
        related_skill: Option<String>,
    },
    StartLongTermUpdate {
        reason: String,
        #[arg(long)]
        evidence: Option<String>,
    },
    CompleteLongTermUpdate {
        #[arg(value_enum)]
        decision: SettlementDecisionArg,
        #[arg(long)]
        target: Option<String>,
        #[arg(long)]
        reason: String,
        #[arg(long)]
        evidence: Option<String>,
        #[arg(long)]
        changed: bool,
    },
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum WriteModeArg {
    Overwrite,
    Append,
    Prepend,
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum SettlementDecisionArg {
    UpdateL2GlobalFacts,
    UpdateL3Skill,
    Skip,
}

pub(crate) fn run_tool(
    store: &SessionStore,
    command: ToolCommand,
    cwd: Option<PathBuf>,
    skills_dir: PathBuf,
) -> Result<()> {
    let cwd = cwd.unwrap_or(env::current_dir()?);
    let call = tool_command_to_call(command)?;
    let mut session = store.start()?;
    session.append(AgentEvent::RunStarted {
        goal: format!("tool:{}", call.name),
        cwd: cwd.clone(),
    })?;
    let result = execute_call(&mut session, &cwd, &skills_dir, store.root(), call)?;
    session.append(AgentEvent::RunFinished {
        status: if result.ok { "completed" } else { "failed" }.to_string(),
        summary: format!("Tool {} finished.", result.name),
    })?;
    println!("{}", serde_json::to_string_pretty(&result.content)?);
    println!("session: {}", session.path().display());
    Ok(())
}

fn tool_command_to_call(command: ToolCommand) -> Result<ToolCall> {
    Ok(match command {
        ToolCommand::MemorySearch { query, limit } => ToolCall::new(
            "memory_search",
            json!({
                "query": query,
                "limit": limit,
            }),
        ),
        ToolCommand::MemoryFetch { id, max_bytes } => ToolCall::new(
            "memory_fetch",
            json!({
                "id": id,
                "max_bytes": max_bytes,
            }),
        ),
        ToolCommand::SkillList { limit } => ToolCall::new(
            "skill_list",
            json!({
                "limit": limit,
            }),
        ),
        ToolCommand::SkillSearch { query, limit } => ToolCall::new(
            "skill_search",
            json!({
                "query": query,
                "limit": limit,
            }),
        ),
        ToolCommand::SkillFetch { name } => ToolCall::new(
            "skill_fetch",
            json!({
                "name": name,
            }),
        ),
        ToolCommand::PlanCreate {
            title,
            task,
            steps,
            source_export_path,
        } => ToolCall::new(
            "plan_create",
            json!({
                "title": title,
                "task": task,
                "steps": steps,
                "source_export_path": source_export_path,
            }),
        ),
        ToolCommand::PlanCreateFromRepoPrompt { path, title, task } => ToolCall::new(
            "plan_create_from_repoprompt",
            json!({
                "export_path": path,
                "title": title,
                "task": task,
            }),
        ),
        ToolCommand::PlanCreateViaRepoPrompt {
            task,
            context,
            hints,
            title,
            working_dirs,
            context_id,
            timeout_secs,
        } => ToolCall::new(
            "plan_create_via_repoprompt",
            json!({
                "task": task,
                "context": context,
                "hints": hints,
                "title": title,
                "working_dirs": working_dirs,
                "context_id": context_id,
                "timeout_secs": timeout_secs,
            }),
        ),
        ToolCommand::PlanRefineViaRepoPrompt {
            plan,
            focus,
            max_fixes,
            chat_id,
            working_dirs,
            timeout_secs,
        } => ToolCall::new(
            "plan_refine_via_repoprompt",
            json!({
                "plan": plan,
                "focus": focus,
                "max_fixes": max_fixes,
                "chat_id": chat_id,
                "working_dirs": working_dirs,
                "timeout_secs": timeout_secs,
            }),
        ),
        ToolCommand::PlanList { limit } => ToolCall::new(
            "plan_list",
            json!({
                "limit": limit,
            }),
        ),
        ToolCommand::PlanStatus { id } => ToolCall::new(
            "plan_status",
            json!({
                "id": id,
            }),
        ),
        ToolCommand::PlanNext { id } => ToolCall::new(
            "plan_next",
            json!({
                "id": id,
            }),
        ),
        ToolCommand::PlanComplete { id, item, note } => ToolCall::new(
            "plan_complete",
            json!({
                "id": id,
                "item": item,
                "note": note,
            }),
        ),
        ToolCommand::PlanRecordArtifact {
            id,
            kind,
            path,
            note,
        } => ToolCall::new(
            "plan_record_artifact",
            json!({
                "id": id,
                "kind": agent_plan::PlanArtifactKind::from(kind),
                "path": path,
                "note": note,
            }),
        ),
        ToolCommand::PlanRecordHandoff {
            id,
            backend,
            role,
            run_id,
            thread_id,
            artifact_path,
            status,
            summary,
        } => ToolCall::new(
            "plan_record_handoff",
            json!({
                "id": id,
                "backend": backend,
                "role": role,
                "run_id": run_id,
                "thread_id": thread_id,
                "artifact_path": artifact_path,
                "status": status,
                "summary": summary,
            }),
        ),
        ToolCommand::PlanVerify {
            id,
            model_id,
            timeout_secs,
            dry_run,
            window_id,
            context_id,
            working_dirs,
        } => ToolCall::new(
            "plan_verify",
            json!({
                "id": id,
                "model_id": model_id,
                "timeout_secs": timeout_secs,
                "dry_run": dry_run,
                "window_id": window_id,
                "context_id": context_id,
                "working_dirs": working_dirs,
            }),
        ),
        ToolCommand::RepoPromptTools => ToolCall::new("repoprompt_tools", json!({})),
        ToolCommand::RepoPromptExec {
            command,
            window_id,
            context_id,
            working_dirs,
            timeout_secs,
            raw_json,
        } => ToolCall::new(
            "repoprompt_exec",
            json!({
                "command": command,
                "window_id": window_id,
                "context_id": context_id,
                "working_dirs": working_dirs,
                "timeout_secs": timeout_secs,
                "raw_json": raw_json,
            }),
        ),
        ToolCommand::RepoPromptCall {
            tool,
            args,
            window_id,
            context_id,
            working_dirs,
            timeout_secs,
        } => ToolCall::new(
            "repoprompt_call",
            json!({
                "tool": tool,
                "args": agent_repoprompt::parse_args_json(&args)?,
                "window_id": window_id,
                "context_id": context_id,
                "working_dirs": working_dirs,
                "timeout_secs": timeout_secs,
            }),
        ),
        ToolCommand::ReadFile {
            path,
            start,
            count,
            keyword,
        } => ToolCall::new(
            "read_file",
            json!({
                "path": path,
                "start": start,
                "count": count,
                "keyword": keyword,
            }),
        ),
        ToolCommand::PatchFile {
            path,
            old_content,
            new_content,
        } => ToolCall::new(
            "patch_file",
            json!({
                "path": path,
                "old_content": old_content,
                "new_content": new_content,
            }),
        ),
        ToolCommand::WriteFile {
            path,
            content,
            mode,
        } => ToolCall::new(
            "write_file",
            json!({
                "path": path,
                "content": content,
                "mode": match mode {
                    WriteModeArg::Overwrite => "overwrite",
                    WriteModeArg::Append => "append",
                    WriteModeArg::Prepend => "prepend",
                },
            }),
        ),
        ToolCommand::RunShell {
            command,
            timeout_secs,
        } => ToolCall::new(
            "run_shell",
            json!({
                "command": command,
                "timeout_secs": timeout_secs,
            }),
        ),
        ToolCommand::UpdateWorkingCheckpoint {
            key_info,
            related_skill,
        } => ToolCall::new(
            "update_working_checkpoint",
            json!({
                "key_info": key_info,
                "related_skill": related_skill,
            }),
        ),
        ToolCommand::StartLongTermUpdate { reason, evidence } => ToolCall::new(
            "start_long_term_update",
            json!({
                "reason": reason,
                "evidence": evidence,
            }),
        ),
        ToolCommand::CompleteLongTermUpdate {
            decision,
            target,
            reason,
            evidence,
            changed,
        } => ToolCall::new(
            "complete_long_term_update",
            json!({
                "decision": match decision {
                    SettlementDecisionArg::UpdateL2GlobalFacts => "update_l2_global_facts",
                    SettlementDecisionArg::UpdateL3Skill => "update_l3_skill",
                    SettlementDecisionArg::Skip => "skip",
                },
                "target": target,
                "reason": reason,
                "evidence": evidence,
                "changed": changed,
            }),
        ),
    })
}
