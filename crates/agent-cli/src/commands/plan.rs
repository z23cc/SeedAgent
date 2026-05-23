//! Implements the `seed plan` subcommand: list/create/import/build/refine/
//! status/next/complete/record-artifact/record-handoff/verify/record-verification
//! over the durable PlanStore on disk. Pure dispatch + presentation — no LLM
//! call beyond the optional RepoPrompt handoff for `build`/`refine`/`verify`.

use std::env;
use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Subcommand, ValueEnum};
use serde_json::json;

use crate::absolutize_cli;
use crate::commands::rp::{
    default_repoprompt_working_dirs_cli, repoprompt_client_cli, repoprompt_output_string_cli,
    repoprompt_report_text_cli,
};
use crate::display::compact_single_line_cli;
use crate::plan_repoprompt::{
    BuildPlanArgs, RefineOutcome, RefinePlanArgs, plan_build_via_repoprompt,
    plan_refine_via_repoprompt,
};

#[derive(Debug, Subcommand)]
pub(crate) enum PlanCommand {
    #[command(alias = "ls")]
    List {
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    Create {
        #[arg(long)]
        title: String,
        #[arg(long)]
        task: String,
        #[arg(long = "step")]
        steps: Vec<String>,
        #[arg(long)]
        source_export_path: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    #[command(about = "Import a RepoPrompt builder plan export into a durable seed plan")]
    Import {
        #[arg(long)]
        path: PathBuf,
        #[arg(long)]
        title: Option<String>,
        #[arg(long)]
        task: Option<String>,
        #[arg(long)]
        json: bool,
    },
    #[command(about = "One-shot: ask RepoPrompt context_builder to draft a plan and import it")]
    Build {
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
        #[arg(long)]
        json: bool,
    },
    #[command(about = "Ask RepoPrompt's oracle to review a plan and append [FIX] items")]
    Refine {
        id: Option<String>,
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
        #[arg(long)]
        json: bool,
    },
    Status {
        id: Option<String>,
        #[arg(long)]
        json: bool,
    },
    Next {
        id: Option<String>,
        #[arg(long)]
        json: bool,
    },
    Complete {
        id: Option<String>,
        #[arg(long)]
        item: Option<usize>,
        #[arg(long)]
        note: Option<String>,
        #[arg(long)]
        json: bool,
    },
    RecordArtifact {
        id: Option<String>,
        #[arg(long, value_enum)]
        kind: PlanArtifactKindArg,
        #[arg(long)]
        path: PathBuf,
        #[arg(long)]
        note: Option<String>,
        #[arg(long)]
        json: bool,
    },
    RecordHandoff {
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
        #[arg(long)]
        json: bool,
    },
    Verify {
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
        #[arg(long)]
        json: bool,
    },
    RecordVerification {
        id: Option<String>,
        #[arg(long)]
        report: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum PlanArtifactKindArg {
    RepoPromptExport,
    ContextExport,
    VerificationContext,
    VerificationReport,
    Other,
}

impl From<PlanArtifactKindArg> for agent_plan::PlanArtifactKind {
    fn from(value: PlanArtifactKindArg) -> Self {
        match value {
            PlanArtifactKindArg::RepoPromptExport => agent_plan::PlanArtifactKind::RepoPromptExport,
            PlanArtifactKindArg::ContextExport => agent_plan::PlanArtifactKind::ContextExport,
            PlanArtifactKindArg::VerificationContext => {
                agent_plan::PlanArtifactKind::VerificationContext
            }
            PlanArtifactKindArg::VerificationReport => {
                agent_plan::PlanArtifactKind::VerificationReport
            }
            PlanArtifactKindArg::Other => agent_plan::PlanArtifactKind::Other,
        }
    }
}

pub(crate) fn run_plan(command: PlanCommand, cwd: Option<PathBuf>) -> Result<()> {
    let cwd = cwd.unwrap_or(env::current_dir()?);
    let store = agent_plan::PlanStore::new(cwd.join("plans"));
    match command {
        PlanCommand::List { limit, json } => {
            let plans = store.list()?;
            print_plan_list(&plans, limit, json, store.root())
        }
        PlanCommand::Create {
            title,
            task,
            steps,
            source_export_path,
            json,
        } => {
            let snapshot = store.create(agent_plan::CreatePlan {
                title,
                task,
                steps,
                source_export_path: source_export_path.map(|path| absolutize_cli(&cwd, path)),
            })?;
            print_plan_snapshot(&snapshot, json)
        }
        PlanCommand::Import {
            path,
            title,
            task,
            json,
        } => {
            let export_path = absolutize_cli(&cwd, path);
            if !export_path.is_file() {
                anyhow::bail!("export file not found: {}", export_path.display());
            }
            let text = std::fs::read_to_string(&export_path)?;
            let imported = agent_plan::import_repoprompt_plan(&text);
            if imported.steps.is_empty() {
                anyhow::bail!(
                    "no plan steps detected in {} — expected a `## Plan` / Steps / Tasks heading with list items",
                    export_path.display()
                );
            }
            let snapshot = store.create(agent_plan::CreatePlan {
                title: title.unwrap_or(imported.title),
                task: task.unwrap_or(imported.task),
                steps: imported.steps.clone(),
                source_export_path: Some(export_path.clone()),
            })?;
            let snapshot = store.record_artifact(
                Some(&snapshot.state.id),
                agent_plan::RecordPlanArtifact {
                    kind: agent_plan::PlanArtifactKind::RepoPromptExport,
                    path: export_path,
                    note: Some(format!(
                        "Imported {} steps ({} delegated, {} parallel)",
                        imported.steps.len(),
                        imported.delegated_count,
                        imported.parallel_count
                    )),
                },
            )?;
            print_plan_snapshot(&snapshot, json)
        }
        PlanCommand::Build {
            task,
            context,
            hints,
            title,
            working_dirs,
            context_id,
            timeout_secs,
            json,
        } => {
            let snapshot = plan_build_via_repoprompt(
                &cwd,
                &store,
                BuildPlanArgs {
                    task,
                    context,
                    hints,
                    title,
                    working_dirs,
                    context_id,
                    timeout_secs,
                },
            )?;
            print_plan_snapshot(&snapshot, json)
        }
        PlanCommand::Refine {
            id,
            focus,
            max_fixes,
            chat_id,
            working_dirs,
            timeout_secs,
            json,
        } => {
            let result = plan_refine_via_repoprompt(
                &cwd,
                &store,
                RefinePlanArgs {
                    id,
                    focus,
                    max_fixes,
                    chat_id,
                    working_dirs,
                    timeout_secs,
                },
            )?;
            match result {
                RefineOutcome::Applied { snapshot, fixes } => {
                    eprintln!("appended {} [FIX] item(s):", fixes.len());
                    for fix in &fixes {
                        eprintln!("  • {fix}");
                    }
                    print_plan_snapshot(&snapshot, json)
                }
                RefineOutcome::NoFixes { plan_id, summary } => {
                    eprintln!("reviewer returned no fixes for plan {plan_id}");
                    if json {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&json!({
                                "status": "no_fixes",
                                "plan_id": plan_id,
                                "review_summary": summary,
                            }))?
                        );
                    } else {
                        println!("{summary}");
                    }
                    Ok(())
                }
            }
        }
        PlanCommand::Status { id, json } => {
            let snapshot = store.snapshot(id.as_deref())?;
            print_plan_snapshot(&snapshot, json)
        }
        PlanCommand::Next { id, json } => {
            let snapshot = store.snapshot(id.as_deref())?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "plan_id": snapshot.state.id,
                        "next_item": snapshot.next_item,
                        "unchecked_count": snapshot.unchecked_count,
                        "task_unchecked_count": snapshot.task_unchecked_count,
                    }))?
                );
            } else if let Some(item) = snapshot.next_item {
                println!(
                    "next: #{} line {} {:?} {}",
                    item.index, item.line_number, item.kind, item.text
                );
            } else {
                println!("next: none");
            }
            Ok(())
        }
        PlanCommand::Complete {
            id,
            item,
            note,
            json,
        } => {
            let snapshot = store.complete(id.as_deref(), item, note.as_deref())?;
            print_plan_snapshot(&snapshot, json)
        }
        PlanCommand::RecordArtifact {
            id,
            kind,
            path,
            note,
            json,
        } => {
            let snapshot = store.record_artifact(
                id.as_deref(),
                agent_plan::RecordPlanArtifact {
                    kind: kind.into(),
                    path: absolutize_cli(&cwd, path),
                    note,
                },
            )?;
            print_plan_snapshot(&snapshot, json)
        }
        PlanCommand::RecordHandoff {
            id,
            backend,
            role,
            run_id,
            thread_id,
            artifact_path,
            status,
            summary,
            json,
        } => {
            let snapshot = store.record_handoff(
                id.as_deref(),
                agent_plan::RecordPlanHandoff {
                    backend,
                    role,
                    run_id,
                    thread_id,
                    artifact_path: artifact_path.map(|path| absolutize_cli(&cwd, path)),
                    status,
                    summary,
                },
            )?;
            print_plan_snapshot(&snapshot, json)
        }
        PlanCommand::Verify {
            id,
            model_id,
            timeout_secs,
            dry_run,
            window_id,
            context_id,
            working_dirs,
            json,
        } => {
            let context = store.write_verify_context(id.as_deref())?;
            if dry_run {
                let snapshot = store.snapshot(Some(&context.plan_id))?;
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "dry_run": true,
                            "verify_context": context,
                            "plan": snapshot,
                        }))?
                    );
                } else {
                    println!(
                        "verify context: {}",
                        snapshot.state.verify_context_path.display()
                    );
                    println!("dry run: verifier not started");
                }
                return Ok(());
            }
            let verify_context_path = context
                .plan_file
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("verify_context.json");
            let cfg = agent_repoprompt::RepoPromptClientConfig {
                timeout_secs: timeout_secs + 60,
                window_id,
                context_id,
                working_dirs: default_repoprompt_working_dirs_cli(&cwd, working_dirs, true),
                raw_json: true,
                ..Default::default()
            };
            let message = format!(
                "Independent verification gate for SeedAgent plan `{}`. Read `{}` first. Return `VERDICT: PASS` or `VERDICT: FAIL` with concise evidence. Do not edit files.",
                context.plan_id,
                verify_context_path.display()
            );
            let output = repoprompt_client_cli(cfg)?.call_tool(
                agent_repoprompt::RepoPromptTool::AgentRun,
                &json!({
                    "op": "start",
                    "model_id": model_id.clone(),
                    "message": message,
                    "timeout": timeout_secs,
                }),
            )?;
            let output_status = output.status().to_string();
            let run_id =
                repoprompt_output_string_cli(&output, &["run_id", "runId", "agent_run_id"]);
            let thread_id = repoprompt_output_string_cli(
                &output,
                &["thread_id", "threadId", "chat_id", "chatId"],
            );
            let report = repoprompt_report_text_cli(&output);
            store.record_verification(Some(&context.plan_id), &report)?;
            let snapshot = store.record_handoff(
                Some(&context.plan_id),
                agent_plan::RecordPlanHandoff {
                    backend: "repoprompt".to_string(),
                    role: Some(model_id),
                    run_id,
                    thread_id,
                    artifact_path: Some(verify_context_path),
                    status: output_status,
                    summary: compact_single_line_cli(&report, 500),
                },
            )?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "verify_context": context,
                        "repoprompt": output,
                        "report": report,
                        "plan": snapshot,
                    }))?
                );
            } else {
                println!("{report}");
                print_plan_snapshot(&snapshot, false)?;
            }
            Ok(())
        }
        PlanCommand::RecordVerification { id, report, json } => {
            let snapshot = store.record_verification(id.as_deref(), &report)?;
            print_plan_snapshot(&snapshot, json)
        }
    }
}

fn print_plan_snapshot(snapshot: &agent_plan::PlanSnapshot, as_json: bool) -> Result<()> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(snapshot)?);
        return Ok(());
    }
    println!("plan: {}", snapshot.state.id);
    println!("title: {}", snapshot.state.title);
    println!("status: {:?}", snapshot.state.status);
    println!("file: {}", snapshot.state.plan_path.display());
    println!(
        "items: {} unchecked, {} non-verify unchecked",
        snapshot.unchecked_count, snapshot.task_unchecked_count
    );
    println!(
        "ledger: {} artifacts, {} handoffs, {} verification records",
        snapshot.state.orchestration.artifacts.len(),
        snapshot.state.orchestration.handoffs.len(),
        snapshot.state.orchestration.verification_records.len()
    );
    if let Some(item) = &snapshot.next_item {
        println!("next: #{} {:?} {}", item.index, item.kind, item.text);
    } else {
        println!("next: none");
    }
    Ok(())
}

fn print_plan_list(
    plans: &[agent_plan::PlanSnapshot],
    limit: usize,
    as_json: bool,
    root: &Path,
) -> Result<()> {
    let shown = if limit == 0 {
        plans
    } else {
        &plans[..plans.len().min(limit)]
    };

    if as_json {
        println!("{}", serde_json::to_string_pretty(shown)?);
        return Ok(());
    }

    if plans.is_empty() {
        println!("no plans found in {}", root.display());
        return Ok(());
    }

    println!("plans: {} total, {} shown", plans.len(), shown.len());
    for snapshot in shown {
        let total = snapshot.items.len();
        let done = snapshot.items.iter().filter(|item| item.checked).count();
        println!(
            "- {} [{}] {}/{} done",
            snapshot.state.id,
            plan_status_label(snapshot.state.status),
            done,
            total
        );
        println!("  title: {}", snapshot.state.title);
        println!(
            "  task: {}",
            compact_single_line_cli(&snapshot.state.task, 120)
        );
        println!("  updated: {}", snapshot.state.updated_at);
        if let Some(item) = &snapshot.next_item {
            println!(
                "  next: #{} {} {}",
                item.index,
                plan_item_kind_label(item.kind),
                compact_single_line_cli(&item.text, 140)
            );
        } else {
            println!("  next: none");
        }
    }

    if limit > 0 && plans.len() > shown.len() {
        println!("show more with: seed plan list --limit {}", plans.len());
    }
    Ok(())
}

fn plan_status_label(status: agent_plan::PlanStatus) -> &'static str {
    match status {
        agent_plan::PlanStatus::Active => "active",
        agent_plan::PlanStatus::PendingVerification => "pending_verification",
        agent_plan::PlanStatus::Verified => "verified",
        agent_plan::PlanStatus::Failed => "failed",
    }
}

fn plan_item_kind_label(kind: agent_plan::PlanItemKind) -> &'static str {
    match kind {
        agent_plan::PlanItemKind::Task => "task",
        agent_plan::PlanItemKind::Verify => "verify",
        agent_plan::PlanItemKind::Fix => "fix",
    }
}
