//! RepoPrompt-backed plan helpers: build (context_builder → import) and refine
//! (oracle review → append [FIX] items). Extracted from main.rs to keep that
//! file from collecting every plan-related concern.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::absolutize_cli;

#[derive(derive_setters::Setters)]
#[setters(into, strip_option)]
pub(crate) struct BuildPlanArgs {
    pub task: String,
    pub context: Option<String>,
    pub hints: Option<String>,
    pub title: Option<String>,
    pub working_dirs: Vec<PathBuf>,
    pub context_id: Option<String>,
    pub timeout_secs: u64,
}

pub(crate) fn plan_build_via_repoprompt(
    cwd: &Path,
    store: &agent_plan::PlanStore,
    args: BuildPlanArgs,
) -> Result<agent_plan::PlanSnapshot> {
    let task_text = args.task.trim();
    if task_text.is_empty() {
        anyhow::bail!("--task must not be empty");
    }
    let mut instructions = format!("<task>{}</task>", escape_xml_for_builder(task_text));
    if let Some(context) = args.context.as_deref()
        && !context.trim().is_empty()
    {
        instructions.push('\n');
        instructions.push_str(&format!(
            "<context>{}</context>",
            escape_xml_for_builder(context)
        ));
    }
    if let Some(hints) = args.hints.as_deref()
        && !hints.trim().is_empty()
    {
        instructions.push('\n');
        instructions.push_str(&format!(
            "<discovery_agent-guidelines>{}</discovery_agent-guidelines>",
            escape_xml_for_builder(hints)
        ));
    }
    let mut cfg = agent_repoprompt::RepoPromptClientConfig {
        cli_path: agent_repoprompt::default_cli_path(),
        timeout_secs: args.timeout_secs.clamp(60, 3600),
        raw_json: true,
        ..Default::default()
    };
    if !args.working_dirs.is_empty() {
        cfg.working_dirs = args
            .working_dirs
            .into_iter()
            .map(|path| absolutize_cli(cwd, path))
            .collect();
    } else {
        cfg.working_dirs = vec![cwd.to_path_buf()];
    }
    cfg.context_id = args.context_id.clone();
    let client = agent_repoprompt::RepoPromptClient::new(cfg);
    client.check_available()?;
    let response = client.build_context(
        &instructions,
        agent_repoprompt::BuilderResponseType::Plan,
        true,
    )?;
    if !response.is_success() {
        anyhow::bail!(
            "context_builder returned exit_code={:?} timed_out={}; stderr: {}",
            response.raw_output.exit_code,
            response.raw_output.timed_out,
            response.raw_output.stderr.trim()
        );
    }
    let export_path = response
        .oracle_export_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("context_builder did not return oracle_export_path"))?;
    let export_text = std::fs::read_to_string(&export_path)?;
    let imported = agent_plan::import_repoprompt_plan(&export_text);
    if imported.steps.is_empty() {
        anyhow::bail!(
            "context_builder export at {} contained no recognizable plan steps",
            export_path.display()
        );
    }
    let snapshot = store.create(agent_plan::CreatePlan {
        title: args.title.unwrap_or(imported.title),
        task: imported.task,
        steps: imported.steps.clone(),
        source_export_path: Some(export_path.clone()),
    })?;
    Ok(store.record_artifact(
        Some(&snapshot.state.id),
        agent_plan::RecordPlanArtifact {
            kind: agent_plan::PlanArtifactKind::RepoPromptExport,
            path: export_path,
            note: Some(format!(
                "Built via context_builder; {} steps ({} delegated, {} parallel)",
                imported.steps.len(),
                imported.delegated_count,
                imported.parallel_count
            )),
        },
    )?)
}

pub(crate) fn escape_xml_for_builder(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[derive(derive_setters::Setters)]
#[setters(into, strip_option)]
pub(crate) struct RefinePlanArgs {
    pub id: Option<String>,
    pub focus: Option<String>,
    pub max_fixes: usize,
    pub chat_id: Option<String>,
    pub working_dirs: Vec<PathBuf>,
    pub timeout_secs: u64,
}

pub(crate) enum RefineOutcome {
    Applied {
        snapshot: Box<agent_plan::PlanSnapshot>,
        fixes: Vec<String>,
    },
    NoFixes {
        plan_id: String,
        summary: String,
    },
}

pub(crate) fn plan_refine_via_repoprompt(
    cwd: &Path,
    store: &agent_plan::PlanStore,
    args: RefinePlanArgs,
) -> Result<RefineOutcome> {
    let snapshot = store.snapshot(args.id.as_deref())?;
    let plan_body = std::fs::read_to_string(&snapshot.state.plan_path)?;
    let max_fixes = args.max_fixes.clamp(1, 30);
    let focus_block = args
        .focus
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|focus| format!("\n<focus>{}</focus>", escape_xml_for_builder(focus)))
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

    let mut cfg = agent_repoprompt::RepoPromptClientConfig {
        cli_path: agent_repoprompt::default_cli_path(),
        timeout_secs: args.timeout_secs.clamp(60, 3600),
        raw_json: true,
        ..Default::default()
    };
    if !args.working_dirs.is_empty() {
        cfg.working_dirs = args
            .working_dirs
            .into_iter()
            .map(|path| absolutize_cli(cwd, path))
            .collect();
    } else {
        cfg.working_dirs = vec![cwd.to_path_buf()];
    }
    let client = agent_repoprompt::RepoPromptClient::new(cfg);
    client.check_available()?;
    let new_chat = args.chat_id.is_none();
    let response = client.send_oracle(
        &message,
        agent_repoprompt::OracleMode::Chat,
        args.chat_id.as_deref(),
        new_chat,
    )?;
    if !response.is_success() {
        anyhow::bail!(
            "oracle_send returned exit_code={:?} timed_out={}; stderr: {}",
            response.raw_output.exit_code,
            response.raw_output.timed_out,
            response.raw_output.stderr.trim()
        );
    }

    let mut fixes = agent_plan::parse_plan_review(&response.response_text);
    if fixes.len() > max_fixes {
        fixes.truncate(max_fixes);
    }
    if fixes.is_empty() {
        return Ok(RefineOutcome::NoFixes {
            plan_id: snapshot.state.id,
            summary: response.response_text,
        });
    }
    let updated = store.append_items(Some(&snapshot.state.id), fixes.clone())?;
    let summary = format!(
        "Appended {} [FIX] items via RepoPrompt oracle review.",
        fixes.len()
    );
    let updated = store.record_handoff(
        Some(&updated.state.id),
        agent_plan::RecordPlanHandoff {
            backend: "repoprompt".to_string(),
            role: Some("reviewer".to_string()),
            run_id: response.chat_id.clone(),
            thread_id: response.chat_id.clone(),
            artifact_path: response.oracle_export_path.clone(),
            status: "completed".to_string(),
            summary,
        },
    )?;
    Ok(RefineOutcome::Applied {
        snapshot: Box::new(updated),
        fixes,
    })
}
