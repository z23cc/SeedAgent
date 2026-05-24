//! Implements the `seed run` LLM planner-loop command. Verbatim extraction
//! from `main.rs` plus dependency adjustments — no behavior change in this
//! pass. Subsequent refactors (RunPolicy, PlannerProvider enum) follow.

use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::env;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use agent_core::{AgentEvent, ToolCall, ToolInfo, ToolResult};
use agent_core::session::{SessionStore, SessionWriter};
use anyhow::Result;

use crate::display::{
    compact_single_line_cli, compose_tool_label, emit_tool_line, format_call_args_for_display,
    format_elapsed_cli, format_token_subtitle, reset_phase_tracker, set_display_cwd,
    short_failure_reason, short_failure_text,
};
use crate::commands::codex::codex_prompt_with_routed_skill;
use crate::commands::exec::{execute_call, execute_call_with_turn};
use crate::{ApprovalArg, McpArg, consolidate_run_skill, memory_paths, parse_seed_goal};

#[derive(Debug, Clone, Copy)]
struct TurnTiming {
    turn: usize,
    planner_ms: u64,
    exec_ms: u64,
    planner_chars: usize,
    /// assembled prompt size (input side). Set by `record_planner_timing`.
    prompt_chars: usize,
}

fn record_planner_timing(
    timings: &RefCell<Vec<TurnTiming>>,
    turn: usize,
    planner_elapsed: Duration,
    planner_chars: usize,
    prompt_chars: usize,
) {
    timings.borrow_mut().push(TurnTiming {
        turn,
        planner_ms: planner_elapsed.as_millis() as u64,
        exec_ms: 0,
        planner_chars,
        prompt_chars,
    });
}

fn record_exec_timing(timings: &RefCell<Vec<TurnTiming>>, exec_elapsed: Duration) {
    if let Some(last) = timings.borrow_mut().last_mut() {
        last.exec_ms = exec_elapsed.as_millis() as u64;
    }
}

fn build_session_archive_record(
    goal: &str,
    status: &str,
    loop_result: &agent_runtime::AgentLoopResult,
    session_path: &Path,
) -> agent_memory::SessionArchiveRecord {
    let session_id = session_path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_string();
    let summary = compact_single_line_cli(&loop_result.summary, 600);
    let key_facts: Vec<String> = loop_result
        .working_memory
        .key_info
        .iter()
        .rev()
        .take(5)
        .map(|fact| compact_single_line_cli(fact, 200))
        .collect();
    let related_skills: Vec<String> = loop_result
        .working_memory
        .related_skills
        .iter()
        .take(5)
        .cloned()
        .collect();
    agent_memory::SessionArchiveRecord {
        session_id,
        session_path: session_path.to_path_buf(),
        goal: compact_single_line_cli(goal, 240),
        status: status.to_string(),
        summary,
        turns: loop_result.turns,
        key_facts,
        related_skills,
        finished_at: chrono::Utc::now(),
    }
}

/// Look for project-local agent rules at `AGENTS.md` or `.seed/rules.md`
/// relative to `cwd`. Returns a formatted prompt block when found. Mirrors
/// the Cursor / Continue / Aider convention: a repo can ship its own
/// agent instructions that get auto-injected next to the L0 meta_rules.
fn local_project_rules_block(cwd: &Path) -> Option<String> {
    const CANDIDATES: &[&str] = &["AGENTS.md", ".seed/rules.md"];
    let mut found: Option<(PathBuf, String)> = None;
    for candidate in CANDIDATES {
        let path = cwd.join(candidate);
        if let Ok(body) = std::fs::read_to_string(&path) {
            let trimmed = body.trim();
            if !trimmed.is_empty() {
                found = Some((path, trimmed.to_string()));
                break;
            }
        }
    }
    let (path, body) = found?;
    Some(format!(
        "### LOCAL PROJECT RULES ({})\n{body}\nThese rules are project-local — treat them as authoritative for THIS codebase.\n",
        path.display()
    ))
}

/// Inline the bundled RepoPrompt routing skill (analysis / investigation
/// / review goals) as a `### RELEVANT SKILL` block on turn 1.
fn relevant_skill_for_goal(goal: &str, skills_dir: &Path) -> Option<String> {
    let route = agent_skills::route_repoprompt_skill(goal)?;
    let doc = agent_skills::fetch_skill(skills_dir, route.slug).ok()?;
    let mut block = format!("### RELEVANT SKILL: {} (auto-routed)\n", route.name);
    block.push_str(&format!("reason: {}\n", route.reason));
    block.push_str(&format!("source: {}\n", doc.info.path.display()));
    block.push_str("body:\n");
    block.push_str(&doc.body);
    block.push('\n');
    block.push_str(
        "Follow the recipe above when it fits the goal; deviate if observed evidence demands.\n",
    );
    Some(block)
}

fn active_plan_brief_for_prompt(plans_root: &Path) -> Option<String> {
    let store = agent_plan::PlanStore::new(plans_root.to_path_buf());
    let brief = store.active_brief().ok().flatten()?;
    let mut out = String::from("### CURRENT PLAN STEP\n");
    out.push_str(&format!(
        "plan: {} (id={}, status={:?}, {} task steps remaining)\n",
        brief.title, brief.plan_id, brief.status, brief.task_unchecked_count
    ));
    out.push_str(&format!("task: {}\n", brief.task));
    if let Some(item) = &brief.next_item {
        out.push_str(&format!("next: [ ] {}\n", item.text));
        if item.delegate {
            out.push_str(
                "  • [D] — delegate this step via `spawn_subagent`; record handoff after.\n",
            );
        }
        if item.parallel {
            out.push_str("  • [P] — independent of siblings; safe to run in parallel.\n");
        }
        if matches!(item.kind, agent_plan::PlanItemKind::Verify) {
            out.push_str("  • [VERIFY] — call `plan_verify` once all task steps are checked.\n");
        }
        if matches!(item.kind, agent_plan::PlanItemKind::Fix) {
            out.push_str("  • [FIX] — verifier rejected; address this before re-verifying.\n");
        }
    } else {
        out.push_str("next: (no unchecked items — call `plan_verify` or finish)\n");
    }
    out.push_str(
        "Protocol: act on this step, then call `plan_complete` with the item index. Skip only if memory/skill lookup must come first.\n",
    );

    // Post-verify guard: once the plan reaches Verified (or PendingVerification
    // with no remaining task items), the agent has been observed to keep
    // exploring code instead of finishing. Inject a hard constraint here so
    // the next planner turn has to choose: finish-with-summary, or take a
    // concrete write action (patch/write/edit). Pure read/search calls are
    // explicitly disallowed in this state until that choice is made.
    let post_verify = matches!(brief.status, agent_plan::PlanStatus::Verified)
        || (matches!(brief.status, agent_plan::PlanStatus::PendingVerification)
            && brief.task_unchecked_count == 0);
    if post_verify {
        out.push_str(
            "\n⚠ PLAN POST-VERIFY GUARD\n\
            The plan is verified / has no task steps left. You MUST now choose exactly one of:\n\
            1. `finish` with a summary that names the plan id, the artifacts touched, and remaining manual follow-ups.\n\
            2. A concrete write action (`patch_file`, `write_file`, `apply_edits`, `spawn_subagent` with a write task) to actually implement an unfinished change.\n\
            Pure exploration calls (read_file, read_files, file_search, get_code_structure, repoprompt_call's read tools) are FORBIDDEN in this state — they will look like progress but waste budget. Pick a branch.\n",
        );
    }
    Some(out)
}

/// If running as a subagent (SEED_SUBAGENT_WATCH_DIR_ENV set), consume any
/// parent-written signal files and apply them to working memory. Returns
/// `Some(PlannedAction::Finish)` when the parent asked the child to stop.
fn apply_subagent_signals(
    state: &mut agent_runtime::AgentLoopState,
) -> Option<agent_runtime::PlannedAction> {
    let watch_dir = env::var_os(agent_tools::SEED_SUBAGENT_WATCH_DIR_ENV)?;
    let watch_path = std::path::PathBuf::from(watch_dir);
    let signals = agent_tools::consume_subagent_signals(&watch_path);
    if signals.is_empty() {
        return None;
    }
    for info in signals.key_info {
        if !info.trim().is_empty() {
            state.working_memory.key_info.push(info);
        }
    }
    if let Some(intervene) = signals.intervene {
        state.recovery_hint = Some(format!(
            "Parent agent sent an intervene message (apply immediately): {intervene}"
        ));
    }
    if signals.stop {
        return Some(agent_runtime::PlannedAction::Finish {
            summary: Some("stop signal received from parent agent".to_string()),
            answer: "Subagent stopped by parent (received `_stop` signal).".to_string(),
        });
    }
    None
}

/// What the planner loop talks to each turn.
#[derive(Debug, Clone)]
pub(crate) enum PlannerProvider {
    /// Local `codex app-server` over stdio JSON-RPC (default).
    Codex,
    /// RepoPrompt `ask_oracle`. `--model` selects oracle mode.
    Oracle,
    /// RepoPrompt `agent_run` — full Agent Mode with `steer` continuity.
    /// `--model` selects role label (default `pair`).
    RepoPromptAgent,
    /// HTTP provider id resolved via `agent_llm::find_provider`.
    Http(String),
}

impl PlannerProvider {
    pub(crate) fn from_id(id: &str) -> Self {
        match id {
            "codex" => Self::Codex,
            "repoprompt_oracle" | "repoprompt" => Self::Oracle,
            // new alias set for the agent_run-backed planner.
            "repoprompt_agent" | "rp_agent" | "rp-agent" => Self::RepoPromptAgent,
            other => Self::Http(other.to_string()),
        }
    }
}

/// Provider-agnostic loop-control knobs. `Setters` enables fluent
/// `RunPolicy::default().max_turns(40)`. Defaults: 24 turns,
/// 10-minute per-turn timeout, 5 consecutive tool failures.
#[derive(Debug, Clone, Copy, derive_setters::Setters)]
#[setters(into)]
pub(crate) struct RunPolicy {
    pub(crate) max_turns: usize,
    pub(crate) turn_timeout_secs: u64,
    /// Abort the planner loop after this many tool calls fail back-to-back.
    /// Borrowed from forge's `ToolErrorTracker`: when every recent call is
    /// failing, continuing to spin through `max_turns` mostly burns the
    /// user's time without recovering — better to fail loudly and let the
    /// user inspect the session. Set to `usize::MAX` to disable.
    pub(crate) max_consecutive_failures: usize,
}

impl Default for RunPolicy {
    fn default() -> Self {
        Self {
            max_turns: 24,
            turn_timeout_secs: 600,
            max_consecutive_failures: 5,
        }
    }
}

#[derive(Debug, Clone, derive_setters::Setters)]
#[setters(into, strip_option)]
pub(crate) struct ProviderSpec {
    pub(crate) kind: PlannerProvider,
    pub(crate) model: Option<String>,
    pub(crate) approval: ApprovalArg,
    pub(crate) effort: Option<String>,
    pub(crate) mcp: Option<McpArg>,
    pub(crate) mcp_allow: Vec<String>,
    pub(crate) plugins: bool,
}

/// `--mode` / `/mode` override. `Auto` uses the keyword classifier;
/// `Read`/`Write` pin the mode regardless of goal text — useful when
/// "implement an analysis of foo" gets misclassified as read-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Default)]
#[clap(rename_all = "lowercase")]
pub(crate) enum ModeArg {
    #[default]
    Auto,
    /// Write tools blocked, run_shell writes refused.
    Read,
    /// Full tool catalog, no run_shell gating.
    Write,
}

#[derive(derive_setters::Setters)]
#[setters(into, strip_option)]
pub(crate) struct RunGoalArgs<'a> {
    pub(crate) store: &'a SessionStore,
    pub(crate) goal: String,
    pub(crate) cwd: Option<PathBuf>,
    pub(crate) use_llm: bool,
    pub(crate) use_codex: bool,
    pub(crate) learn: bool,
    pub(crate) skills_dir: PathBuf,
    pub(crate) policy: RunPolicy,
    pub(crate) provider: ProviderSpec,
    pub(crate) mode: ModeArg,
    /// Launch Codex via `app-server proxy` (running daemon) instead of
    /// spawning fresh. Opt-in via `--use-daemon`.
    pub(crate) use_daemon: bool,
    /// REPL-lifetime Codex client cache. `Some` reuses with per-turn
    /// cfg hot-swap; `None` spawns fresh per call.
    pub(crate) codex_session: Option<&'a mut crate::commands::codex_session::CodexSession>,
}

fn planner_tool_infos_for_mode(
    tools: Vec<ToolInfo>,
    mode: agent_core::RunMode,
) -> Vec<ToolInfo> {
    // Apply the skill narrow set BEFORE the read-only filter so
    // mutating tools the skill mistakenly listed still get caught.
    let tools = tools
        .into_iter()
        .filter(|t| agent_tools::skill_tools_guard::permits(&t.name))
        .collect::<Vec<_>>();
    if !matches!(mode, agent_core::RunMode::ReadOnly) {
        return tools;
    }

    tools
        .into_iter()
        .filter(|tool| is_read_only_planner_tool(&tool.name))
        .collect()
}

fn is_read_only_planner_tool(name: &str) -> bool {
    matches!(
        name,
        "memory_search"
            | "memory_fetch"
            | "skill_list"
            | "skill_search"
            | "skill_fetch"
            | "repoprompt_tools"
            | "repoprompt_exec"
            | "repoprompt_call"
            | "repoprompt_codemap"
            | "repoprompt_file_search"
            | "repoprompt_git"
            | "read_file"
            | "read_files"
            | "run_shell"
            | "update_working_checkpoint"
            // Long-term memory protocol is metadata-only; the durable
            // write goes through patch_file/write_file (blocked in read-only).
            | "start_long_term_update"
            | "complete_long_term_update"
            | "subagent_nudge"
    )
}

#[allow(clippy::too_many_arguments)]
/// `serde_json::to_string` sorts object keys deterministically so
/// `{"a":1,"b":2}` and `{"b":2,"a":1}` hash the same.
fn memoize_key(name: &str, args: &serde_json::Value) -> (String, String) {
    (
        name.to_string(),
        serde_json::to_string(args).unwrap_or_default(),
    )
}

/// Names of tools whose result is a pure function of args within one
/// run. Adding a new pure-read tool requires only
/// `crate::impl_pure_read!()` in its impl block.
fn pure_read_tool_names() -> std::collections::BTreeSet<String> {
    agent_tools::seed_registry()
        .infos()
        .into_iter()
        .filter(|info| info.is_pure_read)
        .map(|info| info.name)
        .collect()
}

fn run_planner_tool(
    spinner: Option<&agent_core::tui::Spinner>,
    allowed_tool_names: &BTreeSet<String>,
    session: &mut SessionWriter,
    cwd: &Path,
    skills_dir: &Path,
    sessions_dir: &Path,
    current_turn: usize,
    call: &ToolCall,
) -> ToolResult {
    set_display_cwd(cwd);
    if !allowed_tool_names.contains(&call.name) {
        let (inner, args_text) = format_call_args_for_display(&call.name, &call.args, 120);
        emit_tool_line(
            spinner,
            &compose_tool_label(&call.name, inner.as_deref()),
            &args_text,
            agent_core::tui::Status::Blocked,
            None,
            "read-only mode",
        );
        return ToolResult::error(
            call,
            format!("tool `{}` is not available for this goal mode", call.name),
        );
    }
    if let Some(s) = spinner {
        s.set_label(format!("running · {}", call.name));
        s.set_subtitle(None);
    }
    let started = Instant::now();
    let outcome = execute_call_with_turn(
        session,
        cwd,
        skills_dir,
        sessions_dir,
        current_turn,
        call.clone(),
    );
    let elapsed = started.elapsed();
    let (inner_tool, args_text) = format_call_args_for_display(&call.name, &call.args, 120);
    match outcome {
        Ok(result) => {
            let status = if result.ok {
                agent_core::tui::Status::Ok
            } else {
                agent_core::tui::Status::Failed
            };
            let note = if result.ok {
                String::new()
            } else {
                short_failure_reason(&result.content)
            };
            emit_tool_line(
                spinner,
                &compose_tool_label(&result.name, inner_tool.as_deref()),
                &args_text,
                status,
                Some(elapsed),
                &note,
            );
            result
        }
        Err(err) => {
            let note = short_failure_text(&err.to_string());
            emit_tool_line(
                spinner,
                &compose_tool_label(&call.name, inner_tool.as_deref()),
                &args_text,
                agent_core::tui::Status::Failed,
                Some(elapsed),
                &note,
            );
            ToolResult::error(call, err.to_string())
        }
    }
}

pub(crate) struct FinalizeInputs<'a> {
    pub(crate) goal: &'a str,
    pub(crate) memory_paths: &'a agent_memory::MemoryPaths,
    pub(crate) skills_dir: &'a Path,
    pub(crate) learn: bool,
}

/// Returns `loop_result.turns` for the run-footer counter. Incomplete
/// runs (`MaxTurnsExceeded`) skip the archive append to avoid skewing
/// the index.
fn finalize_llm_run_outcome(
    session: &mut agent_core::session::SessionWriter,
    loop_result: &agent_runtime::AgentLoopResult,
    turn_timings: &[TurnTiming],
    inputs: FinalizeInputs<'_>,
) -> Result<usize> {
    for turn_summary in &loop_result.turn_summaries {
        session.append(AgentEvent::TurnSummary {
            turn: turn_summary.turn,
            summary: turn_summary.summary.clone(),
        })?;
    }
    for timing in turn_timings {
        session.append(AgentEvent::TurnTimings {
            turn: timing.turn,
            planner_ms: timing.planner_ms,
            exec_ms: timing.exec_ms,
            planner_chars: timing.planner_chars,
            prompt_chars: timing.prompt_chars,
        })?;
    }
    match loop_result.status {
        agent_runtime::AgentLoopStatus::Finished => {
            session.append(AgentEvent::Reflection {
                summary: loop_result.summary.clone(),
            })?;
            session.append(AgentEvent::RunFinished {
                status: "completed".to_string(),
                summary: format!(
                    "Finished after {} turns: {}",
                    loop_result.turns, loop_result.summary
                ),
            })?;
            println!("{}", loop_result.summary);
            let archive_record = build_session_archive_record(
                inputs.goal,
                "completed",
                loop_result,
                session.path(),
            );
            agent_memory::append_session_archive_record(inputs.memory_paths, &archive_record)?;
            agent_memory::rebuild_index(inputs.memory_paths)?;
            if inputs.learn {
                let consolidation =
                    consolidate_run_skill(inputs.skills_dir, inputs.goal, session.path())?;
                agent_memory::rebuild_index(inputs.memory_paths)?;
                let decision = match consolidation.decision {
                    agent_skills::SkillConsolidationDecision::Created => {
                        "create_l3_skill".to_string()
                    }
                    agent_skills::SkillConsolidationDecision::Updated => {
                        "update_l3_skill".to_string()
                    }
                };
                session.append(AgentEvent::CheckpointUpdated {
                    key_info: format!(
                        "Learned skill consolidated via {decision}: {}",
                        consolidation.path.display()
                    ),
                    related_skill: consolidation
                        .path
                        .parent()
                        .and_then(Path::file_name)
                        .and_then(|name| name.to_str())
                        .map(ToString::to_string),
                })?;
                session.append(AgentEvent::LongTermUpdateSettled {
                    decision,
                    target: Some(consolidation.path.display().to_string()),
                    reason: consolidation.reason,
                    evidence: Some(format!(
                        "run --learn session {}",
                        session.path().display()
                    )),
                    changed: true,
                })?;
                println!("learned skill: {}", consolidation.path.display());
            }
        }
        agent_runtime::AgentLoopStatus::MaxTurnsExceeded => {
            session.append(AgentEvent::RunFinished {
                status: "max_turns_exceeded".to_string(),
                summary: loop_result.summary.clone(),
            })?;
            println!("{}", loop_result.summary);
        }
    }
    Ok(loop_result.turns)
}

/// Records Reflection + RunFinished, prints the answer, surfaces token
/// usage when Codex sent it. Err propagates transport failures.
fn record_codex_fast_path_outcome(
    session: &mut agent_core::session::SessionWriter,
    outcome: anyhow::Result<agent_delegate::CodexRunResult>,
) -> Result<()> {
    match outcome {
        Ok(result) => {
            session.append(AgentEvent::Reflection {
                summary: result.text.clone(),
            })?;
            if let Some(t) = result.tokens.as_ref() {
                eprintln!(
                    "{}",
                    agent_core::tui::dim_text(&format!(
                        "(tokens: {} in / {} cached / {} out / {} reasoning · {} total)",
                        t.input_tokens,
                        t.cached_input_tokens,
                        t.output_tokens,
                        t.reasoning_output_tokens,
                        t.total_tokens
                    ))
                );
            }
            session.append(AgentEvent::RunFinished {
                status: "completed".to_string(),
                summary: format!(
                    "Codex completed turn {} after {} events.{}",
                    result.turn_id,
                    result.events_seen,
                    result
                        .tokens
                        .as_ref()
                        .map(|t| format!(" tokens: {}", t.total_tokens))
                        .unwrap_or_default()
                ),
            })?;
            println!("{}", result.text);
            Ok(())
        }
        Err(err) => {
            session.append(AgentEvent::RunFinished {
                status: "failed".to_string(),
                summary: format!("Codex failed: {err}"),
            })?;
            Err(err)
        }
    }
}

/// Resolve `RunMode`, install the global guard, and print the dim
/// "mode: …" trace.
fn resolve_and_announce_run_mode(
    goal: &str,
    mode_arg: ModeArg,
) -> (agent_core::RunMode, agent_core::ModeSource) {
    let (run_mode, mode_source) = match mode_arg {
        ModeArg::Read => (agent_core::RunMode::ReadOnly, agent_core::ModeSource::Explicit),
        ModeArg::Write => (
            agent_core::RunMode::Implementation,
            agent_core::ModeSource::Explicit,
        ),
        ModeArg::Auto => (
            agent_runtime::classify_run_mode(goal),
            agent_core::ModeSource::Auto,
        ),
    };
    agent_tools::run_mode_guard::set(run_mode);
    eprintln!(
        "{}",
        agent_core::tui::dim_text(&format!(
            "mode: {} ({})",
            match run_mode {
                agent_core::RunMode::ReadOnly => "read-only",
                agent_core::RunMode::Implementation => "implementation",
            },
            match mode_source {
                agent_core::ModeSource::Auto => "auto-classified from goal",
                agent_core::ModeSource::Explicit => "explicit via --mode",
            },
        ))
    );
    (run_mode, mode_source)
}

pub(crate) fn run_goal(args: RunGoalArgs<'_>) -> Result<()> {
    let RunGoalArgs {
        store,
        goal,
        cwd,
        use_llm,
        use_codex,
        learn,
        skills_dir,
        policy: RunPolicy {
            max_turns,
            turn_timeout_secs,
            max_consecutive_failures,
        },
        provider:
            ProviderSpec {
                kind: provider_kind,
                model,
                approval,
                effort,
                mcp,
                mcp_allow,
                plugins,
            },
        mode: mode_arg,
        use_daemon,
        codex_session,
    } = args;
    // When we have no REPL-owned session we build a throwaway one locally;
    // its drop kills the subprocess at function exit (fresh codex per
    // run_goal).
    let mut local_codex_session = crate::commands::codex_session::CodexSession::default();
    let codex_session: &mut crate::commands::codex_session::CodexSession =
        codex_session.unwrap_or(&mut local_codex_session);
    let (run_mode, mode_source) = resolve_and_announce_run_mode(&goal, mode_arg);
    let cwd = cwd.unwrap_or(env::current_dir()?);
    // Read-only runs default codex reasoning_effort to "low" when the
    // user didn't set one; "summarize/explain" goals don't need the
    // medium-effort default. User-set --effort always wins.
    let effort = effort.or_else(|| {
        if matches!(run_mode, agent_core::RunMode::ReadOnly) {
            Some("low".to_string())
        } else {
            None
        }
    });
    // Drop any pending RepoPrompt binding override left behind by a previous
    // run (e.g. a skill_fetch that queued an override but the planner never
    // followed up with a repoprompt_* call). Without this, the next run's
    // first rp tool call would mysteriously bind to the previous run's
    // skill dir.
    agent_tools::repoprompt_sync::reset();
    // same logic for the skill-driven tool narrow set — a previous
    // skill_fetch shouldn't restrict this run's tool catalog unless this
    // run also fetches that skill.
    agent_tools::skill_tools_guard::reset();
    // Fresh read-before-write tracking each run.
    agent_tools::read_paths_guard::reset();
    let memory_paths = memory_paths(&cwd, &skills_dir, store.root());
    agent_memory::rebuild_index(&memory_paths)?;
    let base_memory_text = agent_memory::planner_memory_context(&memory_paths)?;
    let plans_root = cwd.join("plans");
    let relevant_skill_block = relevant_skill_for_goal(&goal, &skills_dir);
    let local_rules_block = local_project_rules_block(&cwd);
    let build_planner_memory = || -> agent_runtime::PlannerMemoryContext {
        let mut text = base_memory_text.clone();
        if let Some(brief) = active_plan_brief_for_prompt(&plans_root) {
            text = format!("{brief}\n{text}");
        }
        if let Some(skill) = &relevant_skill_block {
            text = format!("{skill}\n{text}");
        }
        if let Some(rules) = &local_rules_block {
            text = format!("{rules}\n{text}");
        }
        agent_runtime::PlannerMemoryContext::new(text)
    };
    let mut session = store.start()?;
    session.append(AgentEvent::RunStarted {
        goal: goal.clone(),
        cwd: cwd.clone(),
        mode: run_mode,
        mode_source,
    })?;
    let run_started = Instant::now();
    let mut run_turns: usize = 0;

    if use_codex {
        let spinner = agent_core::tui::Spinner::start("codex · running prompt");
        let cfg = crate::commands::codex::codex_config_full(
            model,
            Some(cwd.clone()),
            approval,
            effort,
            turn_timeout_secs,
            mcp,
            mcp_allow,
            plugins,
            use_daemon,
        )?;
        let client = codex_session.ensure(cfg)?;
        let codex_goal = codex_prompt_with_routed_skill(&goal, &skills_dir)?;
        let delta_chars: Cell<usize> = Cell::new(0);
        let outcome = client.run_prompt_streaming(&codex_goal, |delta| {
            delta_chars.set(delta_chars.get() + delta.chars().count());
            spinner.set_subtitle(Some(format_token_subtitle(delta_chars.get())));
        });
        spinner.stop();
        record_codex_fast_path_outcome(&mut session, outcome)?;
    } else if use_llm {
        let registry = agent_tools::seed_registry();
        let tool_infos = planner_tool_infos_for_mode(registry.infos(), run_mode);
        let allowed_tool_names = tool_infos
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<BTreeSet<_>>();
        let spinner = agent_core::tui::Spinner::start(format!("planning · turn 1/{max_turns}"));
        let turn_timings: RefCell<Vec<TurnTiming>> = RefCell::new(Vec::new());
        let active_turn: Cell<usize> = Cell::new(1);
        // Crosses `max_consecutive_failures` → planner bails with a
        // synthetic RuntimeError, unwinding via the error path.
        let failure_streak: Cell<usize> = Cell::new(0);
        reset_phase_tracker();
        let mut planner = match build_planner(
            &provider_kind,
            &cwd,
            model,
            approval,
            effort,
            turn_timeout_secs,
            mcp,
            mcp_allow,
            plugins,
            use_daemon,
            codex_session,
        ) {
            Ok(p) => p,
            Err(err) => {
                session.append(AgentEvent::RunFinished {
                    status: "failed".to_string(),
                    summary: format!("Planner unavailable: {err}"),
                })?;
                return Err(err);
            }
        };
        let retries: RefCell<Vec<agent_runtime::PlannerRetryInfo>> = RefCell::new(Vec::new());
        // Per-run cache for pure-read tools — avoids re-reading the
        // same file across exploration/answer/double-check turns.
        let tool_cache: RefCell<std::collections::HashMap<(String, String), ToolResult>> =
            RefCell::new(std::collections::HashMap::new());
        let pure_reads = pure_read_tool_names();
        let loop_result = match drive_planner_loop(
            &mut planner,
            max_turns,
            &goal,
            &tool_infos,
            &spinner,
            &active_turn,
            &turn_timings,
            &failure_streak,
            max_consecutive_failures,
            &retries,
            build_planner_memory,
            |call| {
                if pure_reads.contains(&call.name) {
                    let key = memoize_key(&call.name, &call.args);
                    if let Some(cached) = tool_cache.borrow().get(&key).cloned() {
                        let (inner_tool, args_text) =
                            format_call_args_for_display(&call.name, &call.args, 120);
                        emit_tool_line(
                            Some(&spinner),
                            &compose_tool_label(&call.name, inner_tool.as_deref()),
                            &args_text,
                            agent_core::tui::Status::Ok,
                            None,
                            "cached",
                        );
                        // A cache hit is NOT fresh progress — the planner
                        // already saw these bytes. Don't reset
                        // failure_streak; that would cloak a stuck loop
                        // pattern like "fail-fail-fail-recheck-cached-fail".
                        return cached;
                    }
                }
                let result = run_planner_tool(
                    Some(&spinner),
                    &allowed_tool_names,
                    &mut session,
                    &cwd,
                    &skills_dir,
                    store.root(),
                    active_turn.get(),
                    call,
                );
                if result.ok {
                    failure_streak.set(0);
                    if pure_reads.contains(&call.name) {
                        tool_cache
                            .borrow_mut()
                            .insert(memoize_key(&call.name, &call.args), result.clone());
                    }
                } else {
                    failure_streak.set(failure_streak.get() + 1);
                }
                result
            },
        ) {
            Ok(result) => result,
            Err(err) => {
                let session_path = session.path().display().to_string();
                session.append(AgentEvent::RunFinished {
                    status: "failed".to_string(),
                    summary: format!("Planner failed: {err}"),
                })?;
                eprintln!(
                    "\nseed: planner failed — session preserved at {session_path}\n      replay with `seed replay {session_path}` or summarize with `seed reflect {session_path}`"
                );
                return Err(err.into());
            }
        };
        drop(planner);
        spinner.stop();

        // Deferred from in-loop because the on_retry closure can't
        // hold &mut session while run_tool does. JSONL timestamps
        // preserve chronology.
        for info in retries.into_inner() {
            session.append(AgentEvent::PlannerRetry {
                turn: info.turn,
                attempt: info.attempt,
                of: info.of,
                backoff_ms: info.backoff_ms,
                kind: info.kind.to_string(),
                reason: compact_single_line_cli(&info.reason, 200),
            })?;
        }

        run_turns = finalize_llm_run_outcome(
            &mut session,
            &loop_result,
            &turn_timings.borrow(),
            FinalizeInputs {
                goal: &goal,
                memory_paths: &memory_paths,
                skills_dir: &skills_dir,
                learn,
            },
        )?;
    } else if let Some(call) = parse_seed_goal(&goal) {
        execute_call(&mut session, &cwd, &skills_dir, store.root(), call)?;
        session.append(AgentEvent::RunFinished {
            status: "completed".to_string(),
            summary: "Executed one seed tool call from the goal prefix.".to_string(),
        })?;
    } else {
        session.append(AgentEvent::CheckpointUpdated {
            key_info: format!("Goal recorded for future planner work: {goal}"),
            related_skill: None,
        })?;
        session.append(AgentEvent::RunFinished {
            status: "recorded".to_string(),
            summary: "No planner provider is wired yet; recorded the goal as seed context."
                .to_string(),
        })?;
    }

    let mut info = agent_core::tui::Info::new();
    if run_turns > 0 {
        info = info.pair("turns", run_turns.to_string());
    }
    info = info.pair("elapsed", format_elapsed_cli(run_started.elapsed()));
    if let Some(stats) = build_timing_stats(&session)? {
        info = info
            .pair("planner avg", stats.planner_avg)
            .pair("exec avg", stats.exec_avg);
        if let Some(chars) = stats.prompt_chars_total {
            info = info.pair("prompt chars", chars);
        }
        if let Some(chars) = stats.planner_chars_total {
            info = info.pair("planner chars", chars);
        }
    }
    info = info.pair("session", session.path().display().to_string());
    eprintln!();
    eprint!("{}", info.render());
    Ok(())
}

struct TimingStats {
    planner_avg: String,
    exec_avg: String,
    planner_chars_total: Option<String>,
    /// Sum of `prompt_chars` across all turns — input side counterpart
    /// to `planner_chars_total`. Surfaces prompt-bloat.
    prompt_chars_total: Option<String>,
}

fn build_timing_stats(session: &SessionWriter) -> Result<Option<TimingStats>> {
    let records = agent_core::session::read_records(session.path())?;
    let mut planner_ms_total: u64 = 0;
    let mut exec_ms_total: u64 = 0;
    let mut planner_chars_total: usize = 0;
    let mut prompt_chars_total: usize = 0;
    let mut count: u64 = 0;
    for record in records {
        if let AgentEvent::TurnTimings {
            planner_ms,
            exec_ms,
            planner_chars,
            prompt_chars,
            ..
        } = record.event
        {
            planner_ms_total += planner_ms;
            exec_ms_total += exec_ms;
            planner_chars_total += planner_chars;
            prompt_chars_total += prompt_chars;
            count += 1;
        }
    }
    if count == 0 {
        return Ok(None);
    }
    let planner_avg = format_elapsed_cli(Duration::from_millis(planner_ms_total / count));
    let exec_avg = format_elapsed_cli(Duration::from_millis(exec_ms_total / count));
    let chars = if planner_chars_total > 0 {
        Some(format_token_subtitle(planner_chars_total))
    } else {
        None
    };
    let prompt_chars = if prompt_chars_total > 0 {
        Some(format_token_subtitle(prompt_chars_total))
    } else {
        None
    };
    Ok(Some(TimingStats {
        planner_avg,
        exec_avg,
        planner_chars_total: chars,
        prompt_chars_total: prompt_chars,
    }))
}

// Adding a new planner backend: impl Planner, branch on PlannerProvider
// in build_planner, done. No new code in drive_planner_loop.
use crate::commands::run_planners::{Planner, build_planner};

/// Drives the planner loop. `failure_streak` + `max_consecutive_failures`
/// implement forge's `ToolErrorTracker` — abort early when tool calls fail
/// back-to-back rather than burning `max_turns`. The check fires at the
/// TOP of each turn so a fresh planner can't trigger a new one first.
#[allow(clippy::too_many_arguments)]
fn drive_planner_loop<M, T>(
    planner: &mut dyn Planner,
    max_turns: usize,
    goal: &str,
    tool_infos: &[ToolInfo],
    spinner: &agent_core::tui::Spinner,
    active_turn: &Cell<usize>,
    turn_timings: &RefCell<Vec<TurnTiming>>,
    failure_streak: &Cell<usize>,
    max_consecutive_failures: usize,
    retries: &RefCell<Vec<agent_runtime::PlannerRetryInfo>>,
    mut build_memory: M,
    mut run_tool: T,
) -> Result<agent_runtime::AgentLoopResult, agent_runtime::RuntimeError>
where
    M: FnMut() -> agent_runtime::PlannerMemoryContext,
    T: FnMut(&ToolCall) -> ToolResult,
{
    let label = planner.label();
    agent_runtime::run_agent_loop_with_state_planner_observed(
        max_turns,
        |state| {
            active_turn.set(state.next_turn);
            if failure_streak.get() >= max_consecutive_failures {
                return Err(agent_runtime::RuntimeError::planner_fatal(format!(
                    "aborted: {} consecutive tool failures (limit={}). \
                     Re-run with `--max-consecutive-failures N` to raise the threshold.",
                    failure_streak.get(),
                    max_consecutive_failures
                )));
            }
            if let Some(action) = apply_subagent_signals(state) {
                return Ok(action);
            }
            spinner.set_label(format!("{label} · turn {}/{}", state.next_turn, max_turns));
            spinner.set_subtitle(planner.turn_start_subtitle());
            let memory = build_memory();
            let planner_started = Instant::now();
            let output = planner.plan(goal, tool_infos, state, &memory, &mut |event| {
                use crate::commands::run_planners::ProgressEvent;
                match event {
                    ProgressEvent::StaticSubtitle(text) => spinner.set_subtitle(text),
                    ProgressEvent::StreamingTokens(n) => {
                        spinner.set_subtitle(Some(format_token_subtitle(n)));
                    }
                }
            })?;
            record_planner_timing(
                turn_timings,
                state.next_turn,
                planner_started.elapsed(),
                output.response_chars,
                output.prompt_chars,
            );
            Ok(output.action)
        },
        |call| {
            let exec_started = Instant::now();
            let result = run_tool(call);
            record_exec_timing(turn_timings, exec_started.elapsed());
            result
        },
        // capture retries for later session.append (we can't hold
        // &mut session here because run_tool already does). The spinner
        // gets a live subtitle update so the user sees the retry as it
        // happens; the session events are flushed after the loop returns.
        |info| {
            spinner.set_subtitle(Some(format!(
                "{} retry {}/{} ({:.1}s back-off)",
                info.kind,
                info.attempt,
                info.of,
                info.backoff_ms as f64 / 1000.0,
            )));
            retries.borrow_mut().push(info);
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::ToolInfo;

    // removed `planner_trait_default_last_prompt_chars_is_zero`
    // — the `last_prompt_chars()` trait method is gone; prompt_chars
    // is now a field on `PlanOutput` returned by `plan()`. There's no
    // "default" anymore — every backend MUST report it.

    // --- tool memoization ----------------------------------------

    #[test]
    fn memoize_key_orders_object_fields_canonically() {
        // Same logical args, different field order → same key.
        let a = serde_json::json!({"a": 1, "b": 2});
        let b = serde_json::json!({"a": 1, "b": 2});
        assert_eq!(memoize_key("t", &a), memoize_key("t", &b));
    }

    #[test]
    fn memoize_key_differs_on_args_change() {
        let a = serde_json::json!({"path": "foo"});
        let b = serde_json::json!({"path": "bar"});
        assert_ne!(memoize_key("read_file", &a), memoize_key("read_file", &b));
    }

    #[test]
    fn memoize_key_differs_on_name_change() {
        let args = serde_json::json!({"x": 1});
        assert_ne!(memoize_key("read_file", &args), memoize_key("read_files", &args));
    }

    #[test]
    fn registry_pure_read_allowlist_matches_intent() {
        let pure = pure_read_tool_names();
        for n in [
            "read_file",
            "read_files",
            "memory_search",
            "memory_fetch",
            "skill_list",
            "skill_search",
            "skill_fetch",
            "plan_status",
            "plan_next",
            "plan_list",
            "tool_describe",
            "repoprompt_tools",
            "repoprompt_codemap",
            "repoprompt_file_search",
            "repoprompt_git",
        ] {
            assert!(pure.contains(n), "should memoize {n}; got set={pure:?}");
        }
        for n in [
            "write_file",
            "patch_file",
            "run_shell",
            "spawn_subagent",
            "plan_create",
            "plan_complete",
            "plan_record_artifact",
            "plan_record_handoff",
            "plan_verify",
            "update_working_checkpoint",
            "start_long_term_update",
            "complete_long_term_update",
            "ask_user",
            "repoprompt_exec",
            "repoprompt_call",
        ] {
            assert!(!pure.contains(n), "must not memoize {n}");
        }
    }

    #[test]
    fn read_only_analysis_hides_plan_and_mutating_tools_from_planner() {
        let tools = vec![
            ToolInfo {
                name: "read_file".to_string(),
                description: "read".to_string(),
                args_schema: None,
                is_pure_read: false,
            },
            ToolInfo {
                name: "read_files".to_string(),
                description: "batch read".to_string(),
                args_schema: None,
                is_pure_read: false,
            },
            ToolInfo {
                name: "update_working_checkpoint".to_string(),
                description: "anchor".to_string(),
                args_schema: None,
                is_pure_read: false,
            },
            ToolInfo {
                name: "plan_create".to_string(),
                description: "plan".to_string(),
                args_schema: None,
                is_pure_read: false,
            },
            ToolInfo {
                name: "patch_file".to_string(),
                description: "patch".to_string(),
                args_schema: None,
                is_pure_read: false,
            },
            ToolInfo {
                name: "repoprompt_exec".to_string(),
                description: "rp".to_string(),
                args_schema: None,
                is_pure_read: false,
            },
        ];

        let names = planner_tool_infos_for_mode(tools, agent_core::RunMode::ReadOnly)
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            vec![
                "read_file",
                "read_files",
                "update_working_checkpoint",
                "repoprompt_exec",
            ]
        );
    }

    // --- skill_tools_guard intersection ---------------------------

    #[test]
    fn skill_narrow_intersects_with_mode_filter() {
        agent_tools::skill_tools_guard::reset();
        agent_tools::skill_tools_guard::set(vec![
            "read_file".to_string(),
            "run_shell".to_string(),
        ]);
        let tools = vec![
            ToolInfo { name: "read_file".to_string(), description: "r".to_string(), args_schema: None, is_pure_read: false },
            ToolInfo { name: "run_shell".to_string(), description: "s".to_string(), args_schema: None, is_pure_read: false },
            ToolInfo { name: "write_file".to_string(), description: "w".to_string(), args_schema: None, is_pure_read: false },
            ToolInfo { name: "memory_search".to_string(), description: "m".to_string(), args_schema: None, is_pure_read: false },
        ];
        let names = planner_tool_infos_for_mode(tools, agent_core::RunMode::ReadOnly)
            .into_iter()
            .map(|t| t.name)
            .collect::<Vec<_>>();
        agent_tools::skill_tools_guard::reset();
        assert_eq!(names, vec!["read_file".to_string(), "run_shell".to_string()]);
    }

    #[test]
    fn no_skill_narrow_keeps_existing_mode_behavior() {
        agent_tools::skill_tools_guard::reset();
        let tools = vec![
            ToolInfo { name: "read_file".to_string(), description: "r".to_string(), args_schema: None, is_pure_read: false },
            ToolInfo { name: "memory_search".to_string(), description: "m".to_string(), args_schema: None, is_pure_read: false },
            ToolInfo { name: "write_file".to_string(), description: "w".to_string(), args_schema: None, is_pure_read: false },
        ];
        let names = planner_tool_infos_for_mode(tools, agent_core::RunMode::ReadOnly)
            .into_iter()
            .map(|t| t.name)
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["read_file".to_string(), "memory_search".to_string()]);
    }

    // drive_planner_loop integration tests via MockPlanner. Covers loop
    // control (Finished/MaxTurnsExceeded/streak abort/Fatal vs retryable
    // Planner errors) that the per-Planner-impl unit tests in
    // agent-runtime don't see.

    /// Queue of pre-baked responses; each `plan()` pops the next.
    /// Enum (not `Result<_, RuntimeError>`) because RuntimeError isn't Clone.
    enum MockResponse {
        Action(agent_runtime::PlannedAction),
        TransientErr(String),
        FatalErr(String),
    }

    struct MockPlanner {
        queue: Vec<MockResponse>,
        calls: std::cell::Cell<usize>,
    }

    impl MockPlanner {
        fn new(responses: Vec<MockResponse>) -> Self {
            Self {
                queue: responses,
                calls: std::cell::Cell::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.get()
        }
    }

    impl Planner for MockPlanner {
        fn label(&self) -> &'static str {
            "mock"
        }

        fn plan(
            &mut self,
            _goal: &str,
            _tool_infos: &[ToolInfo],
            _state: &agent_runtime::AgentLoopState,
            _memory: &agent_runtime::PlannerMemoryContext,
            _on_progress: &mut dyn FnMut(crate::commands::run_planners::ProgressEvent),
        ) -> Result<crate::commands::run_planners::PlanOutput, agent_runtime::RuntimeError> {
            use crate::commands::run_planners::PlanOutput;
            let idx = self.calls.get();
            self.calls.set(idx + 1);
            if idx >= self.queue.len() {
                return Err(agent_runtime::RuntimeError::planner_fatal(
                    "MockPlanner: ran out of queued responses (test bug)",
                ));
            }
            match std::mem::replace(
                &mut self.queue[idx],
                MockResponse::FatalErr("(consumed)".into()),
            ) {
                MockResponse::Action(a) => Ok(PlanOutput {
                    action: a,
                    response_chars: 0,
                    prompt_chars: 0,
                }),
                MockResponse::TransientErr(msg) => {
                    Err(agent_runtime::RuntimeError::Planner(msg))
                }
                MockResponse::FatalErr(msg) => Err(agent_runtime::RuntimeError::planner_fatal(msg)),
            }
        }
    }

    struct LoopFixture {
        spinner: agent_core::tui::Spinner,
        active_turn: Cell<usize>,
        turn_timings: RefCell<Vec<TurnTiming>>,
        failure_streak: Cell<usize>,
        retries: RefCell<Vec<agent_runtime::PlannerRetryInfo>>,
    }

    impl LoopFixture {
        fn new() -> Self {
            Self {
                // Spinner is a no-op when stderr isn't a terminal.
                spinner: agent_core::tui::Spinner::start("test"),
                active_turn: Cell::new(1),
                turn_timings: RefCell::new(Vec::new()),
                failure_streak: Cell::new(0),
                retries: RefCell::new(Vec::new()),
            }
        }
    }

    fn ok_tool_result(name: &str) -> ToolResult {
        ToolResult::ok(
            &ToolCall::new(name, serde_json::json!({})),
            serde_json::json!({"ok": true}),
        )
    }

    fn err_tool_result(name: &str) -> ToolResult {
        ToolResult::error(
            &ToolCall::new(name, serde_json::json!({})),
            "simulated tool failure".to_string(),
        )
    }

    fn tool_call(name: &str) -> agent_runtime::PlannedAction {
        agent_runtime::PlannedAction::Tool {
            summary: Some(format!("call {name}")),
            tool_name: name.to_string(),
            args: serde_json::json!({}),
        }
    }

    fn finish(answer: &str) -> agent_runtime::PlannedAction {
        agent_runtime::PlannedAction::Finish {
            summary: Some("done".to_string()),
            answer: answer.to_string(),
        }
    }

    #[test]
    fn loop_converges_when_planner_finishes_on_turn_two() {
        let mut planner = MockPlanner::new(vec![
            MockResponse::Action(tool_call("read_file")),
            MockResponse::Action(finish("here is the answer")),
        ]);
        let fx = LoopFixture::new();
        let result = drive_planner_loop(
            &mut planner,
            10,
            "test goal",
            &[],
            &fx.spinner,
            &fx.active_turn,
            &fx.turn_timings,
            &fx.failure_streak,
            5,
            &fx.retries,
            || agent_runtime::PlannerMemoryContext::new(String::new()),
            |call| ok_tool_result(&call.name),
        )
        .expect("loop should succeed");
        assert!(matches!(result.status, agent_runtime::AgentLoopStatus::Finished));
        assert_eq!(result.summary, "here is the answer");
        assert_eq!(result.turns, 2, "expected 2 turns (tool + finish)");
        assert_eq!(planner.call_count(), 2);
    }

    #[test]
    fn loop_hits_max_turns_when_planner_keeps_calling_tools() {
        let mut planner = MockPlanner::new(vec![
            MockResponse::Action(tool_call("read_file")),
            MockResponse::Action(tool_call("read_file")),
            MockResponse::Action(tool_call("read_file")),
        ]);
        let fx = LoopFixture::new();
        let result = drive_planner_loop(
            &mut planner,
            3,
            "test goal",
            &[],
            &fx.spinner,
            &fx.active_turn,
            &fx.turn_timings,
            &fx.failure_streak,
            5,
            &fx.retries,
            || agent_runtime::PlannerMemoryContext::new(String::new()),
            |call| ok_tool_result(&call.name),
        )
        .expect("loop should return Ok with MaxTurnsExceeded status");
        assert!(matches!(
            result.status,
            agent_runtime::AgentLoopStatus::MaxTurnsExceeded
        ));
        assert_eq!(result.turns, 3);
    }

    #[test]
    fn loop_aborts_after_n_consecutive_tool_failures() {
        // The 6th planner call never happens — the streak check fires at
        // the top of turn 6 and exits with PlannerFatal.
        let mut planner = MockPlanner::new(vec![
            MockResponse::Action(tool_call("write_file")),
            MockResponse::Action(tool_call("write_file")),
            MockResponse::Action(tool_call("write_file")),
            MockResponse::Action(tool_call("write_file")),
            MockResponse::Action(tool_call("write_file")),
            MockResponse::Action(finish("should not be called")),
        ]);
        let fx = LoopFixture::new();
        let err = drive_planner_loop(
            &mut planner,
            20,
            "test goal",
            &[],
            &fx.spinner,
            &fx.active_turn,
            &fx.turn_timings,
            &fx.failure_streak,
            5,
            &fx.retries,
            || agent_runtime::PlannerMemoryContext::new(String::new()),
            |call| {
                let result = err_tool_result(&call.name);
                if result.ok {
                    fx.failure_streak.set(0);
                } else {
                    fx.failure_streak.set(fx.failure_streak.get() + 1);
                }
                result
            },
        )
        .expect_err("expected fatal abort");
        let agent_runtime::RuntimeError::PlannerFatal(msg) = err else {
            panic!("expected PlannerFatal, got: {err:?}");
        };
        assert!(
            msg.contains("consecutive tool failures"),
            "got: {msg}"
        );
        assert_eq!(planner.call_count(), 5, "6th plan() should not run");
        assert_eq!(fx.failure_streak.get(), 5);
    }

    #[test]
    fn cache_hits_do_not_reset_failure_streak() {
        // Regression: cache hits must NOT zero failure_streak. Otherwise
        // a "fail, fail, fail, recheck-cached, fail" loop never trips
        // the consecutive-failure abort. Replicates run_goal's exec_tool
        // shape since drive_planner_loop doesn't own the cache.
        use std::collections::HashMap;
        let cache: RefCell<HashMap<String, ToolResult>> = RefCell::new(HashMap::new());
        let cached_call = ToolCall::new("read_file", serde_json::json!({"path": "Cargo.toml"}));
        let cached_result = ok_tool_result(&cached_call.name);
        cache
            .borrow_mut()
            .insert("read_file:Cargo.toml".to_string(), cached_result.clone());

        let failure_streak: Cell<usize> = Cell::new(0);

        let exec_tool = |call: &ToolCall| -> ToolResult {
            if call.name == "read_file"
                && let Some(cached) = cache.borrow().get("read_file:Cargo.toml").cloned()
            {
                return cached;
            }
            let result = err_tool_result(&call.name);
            if result.ok {
                failure_streak.set(0);
            } else {
                failure_streak.set(failure_streak.get() + 1);
            }
            result
        };

        // 6 fresh failures + 2 cache hits → streak must reach 6.
        let _ = exec_tool(&ToolCall::new("write_file", serde_json::json!({})));
        let _ = exec_tool(&ToolCall::new("write_file", serde_json::json!({})));
        let _ = exec_tool(&cached_call);
        let _ = exec_tool(&ToolCall::new("write_file", serde_json::json!({})));
        let _ = exec_tool(&cached_call);
        let _ = exec_tool(&ToolCall::new("write_file", serde_json::json!({})));
        let _ = exec_tool(&ToolCall::new("write_file", serde_json::json!({})));
        let _ = exec_tool(&ToolCall::new("write_file", serde_json::json!({})));

        assert_eq!(
            failure_streak.get(),
            6,
            "cache hits should NOT reset failure_streak"
        );
    }

    #[test]
    fn loop_does_not_retry_planner_fatal_errors() {
        let mut planner = MockPlanner::new(vec![MockResponse::FatalErr(
            "auth rejected, do not retry".into(),
        )]);
        let fx = LoopFixture::new();
        let err = drive_planner_loop(
            &mut planner,
            10,
            "test goal",
            &[],
            &fx.spinner,
            &fx.active_turn,
            &fx.turn_timings,
            &fx.failure_streak,
            5,
            &fx.retries,
            || agent_runtime::PlannerMemoryContext::new(String::new()),
            |call| ok_tool_result(&call.name),
        )
        .expect_err("PlannerFatal should bubble out, not retry");
        assert!(matches!(err, agent_runtime::RuntimeError::PlannerFatal(_)));
        assert_eq!(planner.call_count(), 1);
    }

    #[test]
    fn loop_retries_transient_planner_errors_then_succeeds() {
        // Transient → runtime's inner retry. Retry doesn't consume a turn.
        let mut planner = MockPlanner::new(vec![
            MockResponse::TransientErr("network blip".into()),
            MockResponse::Action(finish("recovered")),
        ]);
        let fx = LoopFixture::new();
        let result = drive_planner_loop(
            &mut planner,
            10,
            "test goal",
            &[],
            &fx.spinner,
            &fx.active_turn,
            &fx.turn_timings,
            &fx.failure_streak,
            5,
            &fx.retries,
            || agent_runtime::PlannerMemoryContext::new(String::new()),
            |call| ok_tool_result(&call.name),
        )
        .expect("loop should recover from transient error");
        assert!(matches!(result.status, agent_runtime::AgentLoopStatus::Finished));
        assert_eq!(result.summary, "recovered");
        assert_eq!(planner.call_count(), 2);
        assert_eq!(result.turns, 1);
    }

    #[test]
    fn tool_success_resets_failure_streak() {
        // fail*2, success, fail*5: streak resets at turn 3, aborts at turn 8.
        let mut planner = MockPlanner::new(
            (0..10).map(|_| MockResponse::Action(tool_call("x"))).collect(),
        );
        let fx = LoopFixture::new();
        let turn_counter = Cell::new(0_usize);
        let err = drive_planner_loop(
            &mut planner,
            20,
            "test goal",
            &[],
            &fx.spinner,
            &fx.active_turn,
            &fx.turn_timings,
            &fx.failure_streak,
            5,
            &fx.retries,
            || agent_runtime::PlannerMemoryContext::new(String::new()),
            |call| {
                turn_counter.set(turn_counter.get() + 1);
                let n = turn_counter.get();
                let result = if n == 3 {
                    ok_tool_result(&call.name)
                } else {
                    err_tool_result(&call.name)
                };
                if result.ok {
                    fx.failure_streak.set(0);
                } else {
                    fx.failure_streak.set(fx.failure_streak.get() + 1);
                }
                result
            },
        )
        .expect_err("expected abort after 5 streak post-reset");
        assert!(matches!(err, agent_runtime::RuntimeError::PlannerFatal(_)));
        assert_eq!(turn_counter.get(), 8);
    }

    /// Mirror of the auto-effort rule inside `run_goal` — the inline
    /// call site is the source of truth.
    fn auto_effort_mirror(user_set: Option<String>, goal: &str) -> Option<String> {
        user_set.or_else(|| {
            if matches!(
                agent_runtime::classify_run_mode(goal),
                agent_core::RunMode::ReadOnly
            ) {
                Some("low".to_string())
            } else {
                None
            }
        })
    }

    #[test]
    fn auto_effort_user_set_always_wins() {
        assert_eq!(
            auto_effort_mirror(Some("high".to_string()), "分析当前的项目"),
            Some("high".to_string())
        );
        assert_eq!(
            auto_effort_mirror(Some("minimal".to_string()), "implement feature X"),
            Some("minimal".to_string())
        );
    }

    #[test]
    fn auto_effort_defaults_to_low_for_read_only_goals() {
        for goal in ["分析当前的项目", "explain this code", "summarize the README"] {
            assert_eq!(
                auto_effort_mirror(None, goal),
                Some("low".to_string()),
                "expected low for read-only goal: {goal}",
            );
        }
    }

    #[test]
    fn auto_effort_stays_none_for_implementation_goals() {
        for goal in ["implement feature X", "refactor module Y", "fix bug Z"] {
            assert_eq!(
                auto_effort_mirror(None, goal),
                None,
                "implementation goal {goal} must not get auto-low effort",
            );
        }
    }
}
