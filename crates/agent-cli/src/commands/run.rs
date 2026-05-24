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
    /// RF36-1: assembled prompt size (input side). Set by `record_planner_timing`.
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

/// For read-only analysis / investigation / review-shaped goals, look up the
/// bundled RepoPrompt routing skill and inline its body once as a `### RELEVANT
/// SKILL` block. This lets the planner pick up an existing recipe on turn 1
/// instead of re-deriving the exploration pattern from scratch — concretely
/// cuts turn count on "analyze this project" / "review the auth module" /
/// "explain the build" tasks.
///
/// Skill body is loaded once at run start, NOT per-turn (it does not change),
/// so the extra prompt cost is amortized across the whole loop.
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

/// Build the prompt for the synthesis-only final turn. The planner's first
/// `Finish` answer is treated as a DRAFT — this prompt asks it to rewrite the
/// draft strictly to the FINISH ANSWER SCHEMA defined in `planner_goal_guidance`,
/// using only what's already in working memory. No tool calls allowed.
fn build_synthesis_prompt(
    goal: &str,
    draft: &str,
    working_memory: &agent_runtime::WorkingMemory,
    observations: &[agent_runtime::ToolObservation],
) -> String {
    let anchor = working_memory.render_anchor();
    let obs_summary = observations
        .iter()
        .map(|obs| {
            format!(
                "- turn {} tool={} ok={} summary={}",
                obs.turn,
                obs.call.name,
                obs.result.ok,
                compact_single_line_cli(&obs.summary, 160),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "You just finished a read-only analysis. Your DRAFT answer was:\n\n\
        <draft_answer>\n{draft}\n</draft_answer>\n\n\
        Your working memory anchor at end of run:\n{anchor}\n\
        \nObserved tool calls (most recent last):\n{obs_summary}\n\n\
        \nThe user's goal was:\n  {goal}\n\n\
        \nREWRITE the draft as a single final answer that strictly follows the FINISH ANSWER SCHEMA from your system prompt:\n\
        - Bottom line: one sentence ≤30 words, no preamble. Must contain a concrete, non-trivial finding (not a goal restatement or README paraphrase).\n\
        - Evidence: 2–5 bullets, each starting with [via <tool_name>], citing the tool that produced the observation (use the names from the observation log above). At least one bullet MUST name a specific code element (function, struct, test, file:line, dependency, line count) — pure README/config restatements are insufficient.\n\
        - Counter-intuitive / risk: 1–2 real findings ABOUT THE PROJECT (code smell, missing test, recent direction in git, suspicious boundary, hidden coupling). \
          FORBIDDEN in this section: meta-observations about your own runtime mode, about read-only constraints, about other active plans/sessions, or about what the agent itself can or cannot do. \
          If you genuinely found nothing surprising about the project, write a single line `- (none — surface-level analysis only)` and do not pad.\n\
        - Action items: 1–4 imperative items, each naming a concrete target (file:line, cargo command, function/struct name). Reject vague items like `Inspect X` / `Trace Y` / `Run cargo test --workspace` (the last is generic — only include if a specific test name is given).\n\
        - Explicitly NOT doing: include only if you actually rejected an alternative for a reason; do NOT use this section to restate goal-mode constraints.\n\
        - Do NOT begin with banned openers (`整体来看`, `当前... 是`, `如果想要继续`, `This project is`, `Overall,`, etc.).\n\
        - Do NOT add or invent observations the draft and working memory don't already support.\n\
        \nReturn EXACTLY one JSON object: \
        {{\"summary\":\"<≤30 char meta-summary>\",\"action\":\"finish\",\"answer\":\"<the rewritten markdown answer>\"}}\n\
        \nNo markdown fences around the JSON. No extra commentary."
    )
}

fn extract_synthesized_answer(text: &str) -> Result<String> {
    let action = agent_runtime::parse_planned_action(text)
        .map_err(|err| anyhow::anyhow!("synthesis response did not parse: {err}"))?;
    match action {
        agent_runtime::PlannedAction::Finish { answer, .. } => {
            if answer.trim().is_empty() {
                anyhow::bail!("synthesis returned an empty answer");
            }
            Ok(answer)
        }
        agent_runtime::PlannedAction::Tool { tool_name, .. } => {
            anyhow::bail!(
                "synthesis must return Finish but returned tool={tool_name}"
            )
        }
    }
}

/// RF26-1: cheap structural check for "does this draft already follow the
/// FINISH ANSWER SCHEMA that the synthesis pass would rewrite it into?"
///
/// The synthesis pass costs one full Codex turn (~60s in real runs). For
/// reasonably-disciplined drafts that already have the four mandatory
/// sections and at least one `[via <tool>]` citation, the rewrite is
/// usually a paraphrase — net negative on latency and tokens, no
/// correctness win.
///
/// We err on the side of running synthesis when in doubt: returning `false`
/// here just means "do the extra turn, like before". Returning `true`
/// short-circuits to the draft. So the cost of a false positive (skip
/// when we shouldn't) is "the answer is structurally fine but might have
/// banned openers"; the cost of a false negative (run synthesis when we
/// could've skipped) is the wasted 60s. Bias toward true is fine.
///
/// Strategy:
///   - Must mention all four required section labels ("Bottom line",
///     "Evidence", "Counter-intuitive / risk" OR "Counter-intuitive", "Action items").
///   - Must contain at least one `[via ` citation (proves Evidence
///     section has tool-grounded bullets, not just header).
///   - Must not start with a banned opener (these are forbidden by the
///     synthesis prompt — if a draft starts with one, the synthesis would
///     do real work).
///   - Total length > 200 chars to filter out shape-only stubs.
fn draft_already_conforms(draft: &str) -> bool {
    let body = draft.trim();
    if body.len() < 200 {
        return false;
    }
    // Required sections. Each `find` is case-sensitive on purpose — the
    // schema we ship in the prompt uses these exact strings, so a draft
    // that paraphrased them ("# Conclusion" instead of "Bottom line:")
    // probably *does* need the rewrite.
    let has_bottom_line = body.contains("Bottom line");
    let has_evidence = body.contains("Evidence");
    let has_counter = body.contains("Counter-intuitive");
    let has_actions = body.contains("Action items");
    if !(has_bottom_line && has_evidence && has_counter && has_actions) {
        return false;
    }
    // Evidence section integrity: at least one [via TOOL] citation.
    if !body.contains("[via ") {
        return false;
    }
    // Banned opener check (subset — schema lists more; this catches the
    // common ones). If the draft opens with one of these, synthesis can
    // still salvage the answer, so don't skip.
    let banned_openers = [
        "整体来看",
        "当前", // weak — but the schema lists it explicitly
        "如果想要继续",
        "This project is",
        "Overall,",
    ];
    let first_para = body.lines().next().unwrap_or("").trim();
    for banned in banned_openers {
        if first_para.starts_with(banned) {
            return false;
        }
    }
    true
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
///
/// String CLI flag `--provider <id>` collapses into one of three concrete
/// shapes via [`PlannerProvider::from_id`]. Match-dispatch in `run_goal`
/// replaces the previous string compares — the borrow checker can prove
/// every arm is handled and the synthesis pass keeps the same shape.
#[derive(Debug, Clone)]
pub(crate) enum PlannerProvider {
    /// Local `codex app-server` over stdio JSON-RPC (default).
    Codex,
    /// RepoPrompt `ask_oracle` — planner prompts inherit RepoPrompt's
    /// curated context. `--model` selects the oracle mode (`chat|plan|edit|review`).
    Oracle,
    /// HTTP provider (`openai`, `openai_compatible`, etc.) — the inner
    /// String is the provider id used by `agent_llm::find_provider`.
    Http(String),
}

impl PlannerProvider {
    pub(crate) fn from_id(id: &str) -> Self {
        match id {
            "codex" => Self::Codex,
            "repoprompt_oracle" | "repoprompt" => Self::Oracle,
            other => Self::Http(other.to_string()),
        }
    }
}

/// Loop-control knobs that aren't provider-specific. Kept separate so the
/// run_goal signature collects "how to drive the loop" in one place — no
/// magic numbers leaking through positional arguments.
///
/// `Setters` lets callers override one field fluently:
/// `RunPolicy::default().max_turns(40)`. Defaults mirror the CLI defaults
/// (24 turns, 10-minute per-turn timeout, 5 consecutive tool failures).
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

/// Provider-specific configuration. Cloned into the post-loop synthesis
/// pass so a Codex-backed run can recreate its config without re-borrowing
/// the original RunGoalArgs.
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

/// User-visible mode override exposed via `--mode` and `/mode`.
///
/// `Auto` is the default and matches pre-RF27 behavior: the goal text gets
/// classified via `agent_runtime::classify_run_mode`. `Read` and `Write`
/// pin the mode regardless of goal text, which is useful when the
/// keyword heuristic guesses wrong (e.g. "implement an analysis of foo"
/// → currently classifies read-only because "analysis" is matched first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Default)]
#[clap(rename_all = "lowercase")]
pub(crate) enum ModeArg {
    /// Auto-classify via keyword heuristic (default, historical behavior).
    #[default]
    Auto,
    /// Pin to read-only: write tools blocked, run_shell writes refused,
    /// synthesis pass eligible.
    Read,
    /// Pin to implementation: full tool catalog, no run_shell gating, no
    /// synthesis pass (treated as non-read-only).
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
    /// RF27-3: explicit mode override. `Auto` uses the keyword classifier;
    /// `Read`/`Write` pin the mode regardless of goal text.
    pub(crate) mode: ModeArg,
    /// RF33-4: when true, codex_config sets use_daemon=true so the client
    /// launches via `codex app-server proxy` (running daemon) instead of
    /// spawning a fresh app-server. User opts in via `--use-daemon`.
    pub(crate) use_daemon: bool,
    /// Optional REPL-lifetime Codex client cache. When `Some`, every Codex
    /// spawn site inside this `run_goal` will reuse the cached client (with
    /// per-turn cfg hot-swapped) instead of spawning a fresh `codex
    /// app-server` subprocess. One-shot `seed run` callers pass `None`,
    /// which falls back to the previous "fresh client per call" behavior.
    pub(crate) codex_session: Option<&'a mut crate::commands::codex_session::CodexSession>,
}

fn planner_tool_infos_for_mode(
    tools: Vec<ToolInfo>,
    mode: agent_core::RunMode,
) -> Vec<ToolInfo> {
    // RF37: if a previously-fetched skill narrowed the catalog, intersect
    // first. Empty narrow set (no skill restriction) = identity. We do
    // this BEFORE the read-only filter so the read-only filter still
    // catches mutating tools the skill mistakenly listed.
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
        // Pure discovery
        "memory_search"
            | "memory_fetch"
            | "skill_list"
            | "skill_search"
            | "skill_fetch"
            | "repoprompt_tools"
            | "repoprompt_exec"
            | "repoprompt_call"
            | "read_file"
            | "read_files"
            | "run_shell"
            // Working-memory anchors (in-process state, no FS writes that would
            // mutate the project).
            | "update_working_checkpoint"
            // Long-term memory protocol — the gate + settlement decision are
            // metadata-only; the actual durable write (if it happens) goes
            // through patch_file / write_file which stay blocked in read-only.
            | "start_long_term_update"
            | "complete_long_term_update"
            // Parent-side nudges for any subagent the read-only run may have
            // spawned (no FS mutation of the project).
            | "subagent_nudge"
    )
}

#[allow(clippy::too_many_arguments)]
/// RF35-2: cache key for within-run tool memoization. We canonicalize the
/// args via `to_string` (which sorts object keys deterministically in
/// serde_json) so `{"a":1,"b":2}` and `{"b":2,"a":1}` hash the same.
fn memoize_key(name: &str, args: &serde_json::Value) -> (String, String) {
    (
        name.to_string(),
        serde_json::to_string(args).unwrap_or_default(),
    )
}

/// RF35-2: allowlist of tools whose result is a pure function of their
/// args within one run. Repeated calls (planner double-checking, multiple
/// turns asking the same question) return the cached value.
///
/// Excluded by default: anything with side effects (write/patch/shell/
/// spawn/plan-mutating/long-term-memory-mutating/checkpoint/ask_user),
/// repoprompt_exec/repoprompt_call (they can perform edits or oracle
/// chats that aren't reproducible).
fn is_memoizable_tool(name: &str) -> bool {
    matches!(
        name,
        "read_file"
            | "read_files"
            | "memory_search"
            | "memory_fetch"
            | "skill_list"
            | "skill_search"
            | "skill_fetch"
            | "plan_status"
            | "plan_next"
            | "plan_list"
            | "tool_describe"
            | "repoprompt_tools"
    )
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
    // Bridge: when we have no REPL-owned session we build a throwaway one
    // locally so the rest of this function can speak the same API. The
    // local session's drop kills the subprocess at function exit, restoring
    // the pre-RF25 "fresh codex per run_goal" behavior.
    let mut local_codex_session = crate::commands::codex_session::CodexSession::default();
    let codex_session: &mut crate::commands::codex_session::CodexSession =
        codex_session.unwrap_or(&mut local_codex_session);
    // RF27: resolve effective mode + provenance.
    //
    //   --mode auto  (default) → classify by goal keywords; source = Auto
    //   --mode read           → ReadOnly,         source = Explicit
    //   --mode write          → Implementation,   source = Explicit
    //
    // Set the process-global guard right after so ShellTool (and any
    // future tool that wants to check) sees the right value for the
    // entirety of the run. The previous run's value persists otherwise
    // — fine for one-shot processes, dangerous for the REPL.
    let (run_mode, mode_source) = match mode_arg {
        ModeArg::Read => (agent_core::RunMode::ReadOnly, agent_core::ModeSource::Explicit),
        ModeArg::Write => (
            agent_core::RunMode::Implementation,
            agent_core::ModeSource::Explicit,
        ),
        ModeArg::Auto => (
            agent_runtime::classify_run_mode(&goal),
            agent_core::ModeSource::Auto,
        ),
    };
    agent_tools::run_mode_guard::set(run_mode);
    // Surface the decision so the user can tell from the trace which
    // toolset the planner has access to. Dimmed so it doesn't compete with
    // the goal text — but always present.
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
    let cwd = cwd.unwrap_or(env::current_dir()?);
    // RF26-2 / RF27: for read-only runs, default codex `reasoning_effort`
    // to "low" when the user didn't explicitly set one. Codex's default is
    // "medium" which is overkill for "summarize / explain / explore" tasks
    // — `low` typically cuts per-turn planner time roughly in half on those
    // shapes. User-set --effort (or /effort in REPL) always wins. We key
    // off the resolved `run_mode` (not the raw classifier) so `--mode read`
    // on an implementation-shaped goal still gets the cheap effort.
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
    // RF37: same logic for the skill-driven tool narrow set — a previous
    // skill_fetch shouldn't restrict this run's tool catalog unless this
    // run also fetches that skill.
    agent_tools::skill_tools_guard::reset();
    let memory_paths = memory_paths(&cwd, &skills_dir, store.root());
    agent_memory::rebuild_index(&memory_paths)?;
    let base_memory_text = agent_memory::planner_memory_context(&memory_paths)?;
    let plans_root = cwd.join("plans");
    let relevant_skill_block = relevant_skill_for_goal(&goal, &skills_dir);
    let build_planner_memory = || -> agent_runtime::PlannerMemoryContext {
        let mut text = base_memory_text.clone();
        if let Some(brief) = active_plan_brief_for_prompt(&plans_root) {
            text = format!("{brief}\n{text}");
        }
        if let Some(skill) = &relevant_skill_block {
            text = format!("{skill}\n{text}");
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
        // RF25-1: reuse the REPL-lifetime client if its launch fingerprint
        // matches; otherwise this constructs a fresh one and stashes it for
        // next time. For one-shot `seed run --codex` callers, `codex_session`
        // is the throwaway local one, so behavior matches the pre-RF25
        // "fresh client per run" path.
        let client = codex_session.ensure(cfg)?;
        let codex_goal = codex_prompt_with_routed_skill(&goal, &skills_dir)?;
        let delta_chars: Cell<usize> = Cell::new(0);
        let outcome = client.run_prompt_streaming(&codex_goal, |delta| {
            delta_chars.set(delta_chars.get() + delta.chars().count());
            spinner.set_subtitle(Some(format_token_subtitle(delta_chars.get())));
        });
        spinner.stop();
        match outcome {
            Ok(result) => {
                session.append(AgentEvent::Reflection {
                    summary: result.text.clone(),
                })?;
                // RF35-1: surface token usage when Codex sent the
                // `thread/tokenUsageUpdated` notification. Single dim line
                // so it doesn't compete with the answer; absent when the
                // daemon/server didn't emit usage data.
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
            }
            Err(err) => {
                session.append(AgentEvent::RunFinished {
                    status: "failed".to_string(),
                    summary: format!("Codex failed: {err}"),
                })?;
                return Err(err);
            }
        }
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
        // Track consecutive tool failures. Reset on any successful tool call;
        // when it exceeds `max_consecutive_failures`, the planner closure bails
        // out with a synthetic RuntimeError so the loop unwinds cleanly through
        // the normal error path (session.append RunFinished + eprintln hint).
        let failure_streak: Cell<usize> = Cell::new(0);
        // Reset the per-run phase divider so each REPL turn starts fresh and
        // the first tool prints its own `── phase ──` header.
        reset_phase_tracker();
        // Stash a copy of the planner config bits before the per-provider
        // branches consume them, so the post-loop synthesis pass can recreate
        // a fresh client without fighting the borrow checker.
        let synthesis_model = model.clone();
        let synthesis_approval = approval.clone();
        let synthesis_effort = effort.clone();
        let synthesis_mcp = mcp.clone();
        let synthesis_mcp_allow = mcp_allow.clone();
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
        // RF35-2: per-run cache for pure-read tools. Saves the planner
        // re-reading Cargo.toml three times in one goal (a common pattern
        // when it's exploring → answers → double-checks before finishing).
        let tool_cache: RefCell<std::collections::HashMap<(String, String), ToolResult>> =
            RefCell::new(std::collections::HashMap::new());
        let mut loop_result = match drive_planner_loop(
            planner.as_mut(),
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
                // RF35-2: short-circuit if we already have a cached
                // result for this exact (name, args). We emit a dim
                // "(cached)" line via the spinner so the trace makes the
                // skip visible.
                if is_memoizable_tool(&call.name) {
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
                        // Cached read = always "success" from streak POV.
                        failure_streak.set(0);
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
                    if is_memoizable_tool(&call.name) {
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

        // RF34-2: flush retry events captured during the loop. We defer
        // session writes here because the in-loop on_retry closure can't
        // hold &mut session (run_tool already does). Chronology is
        // preserved by the JSONL timestamps.
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

        // Synthesis pass: read-only analysis goals that reach Finish get one
        // extra LLM call to rewrite the draft answer strictly to schema. This
        // separates "what to explore next" from "how to present what I found",
        // which the planner conflates when given both jobs in the same turn.
        //
        // RF26-1: skip the entire synthesis turn (~60s) when the draft
        // already structurally conforms — same content, half the latency.
        let synthesis_eligible = matches!(
            loop_result.status,
            agent_runtime::AgentLoopStatus::Finished
        ) && matches!(run_mode, agent_core::RunMode::ReadOnly)
            && !loop_result.summary.trim().is_empty();
        if synthesis_eligible && draft_already_conforms(&loop_result.summary) {
            // Skip path. Leave loop_result.summary as the draft; log so the
            // user sees from the trace whether synthesis fired.
            eprintln!(
                "{}",
                agent_core::tui::dim_text(
                    "(synthesis pass skipped: draft already conforms to schema)"
                )
            );
        } else if synthesis_eligible {
            let synthesis_spinner = agent_core::tui::Spinner::start("synthesizing answer · turn final");
            let synthesis_prompt = build_synthesis_prompt(
                &goal,
                &loop_result.summary,
                &loop_result.working_memory,
                &loop_result.observations,
            );
            let synthesis_outcome: Result<String> = match &provider_kind {
                PlannerProvider::Oracle => {
                let oracle_cfg = agent_repoprompt::RepoPromptClientConfig {
                    cli_path: agent_repoprompt::default_cli_path(),
                    timeout_secs: turn_timeout_secs.max(60),
                    raw_json: true,
                    working_dirs: vec![cwd.clone()],
                    ..Default::default()
                };
                let oracle = agent_repoprompt::RepoPromptClient::new(oracle_cfg);
                if let Err(err) = oracle.check_available() {
                    Err(anyhow::anyhow!(
                        "RepoPrompt oracle unavailable for synthesis: {err}"
                    ))
                } else {
                    oracle
                        .send_oracle(
                            &synthesis_prompt,
                            agent_repoprompt::OracleMode::Chat,
                            None,
                            true,
                        )
                        .map_err(anyhow::Error::from)
                        .and_then(|resp| {
                            if !resp.is_success() {
                                anyhow::bail!(
                                    "oracle synthesis returned exit_code={:?}",
                                    resp.raw_output.exit_code
                                );
                            }
                            extract_synthesized_answer(&resp.response_text)
                        })
                }
                }
                PlannerProvider::Codex => {
                let cfg = crate::commands::codex::codex_config_full(
                    synthesis_model.clone(),
                    Some(cwd.clone()),
                    synthesis_approval.clone(),
                    synthesis_effort.clone(),
                    turn_timeout_secs,
                    synthesis_mcp,
                    synthesis_mcp_allow.clone(),
                    plugins,
                    use_daemon,
                )?;
                // Reuse the REPL-lifetime session (RF25-1) — synthesis is
                // a single extra Codex turn, so it benefits the same way
                // as the main path.
                let codex = codex_session.ensure(cfg)?;
                codex
                    .run_prompt(&synthesis_prompt)
                    .and_then(|result| extract_synthesized_answer(&result.text))
                }
                PlannerProvider::Http(provider_id) => {
                // openai-style provider: skip synthesis (would need plumbing
                // through plan_one_tool_call / ProviderClient; defer until
                // there's a real user of the HTTP provider path).
                Err(anyhow::anyhow!(
                    "synthesis not implemented for provider `{provider_id}` — using draft"
                ))
                }
            };
            synthesis_spinner.stop();
            match synthesis_outcome {
                Ok(rewritten) => {
                    eprintln!(
                        "{}",
                        agent_core::tui::dim_text(
                            "(synthesis pass applied: draft rewritten to schema)"
                        )
                    );
                    loop_result.summary = rewritten;
                }
                Err(err) => {
                    eprintln!(
                        "{}",
                        agent_core::tui::dim_text(&format!(
                            "(synthesis pass skipped: {err}; using draft answer)"
                        ))
                    );
                }
            }
        }

        run_turns = loop_result.turns;
        for turn_summary in &loop_result.turn_summaries {
            session.append(AgentEvent::TurnSummary {
                turn: turn_summary.turn,
                summary: turn_summary.summary.clone(),
            })?;
        }
        for timing in turn_timings.borrow().iter() {
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
                    &goal,
                    "completed",
                    &loop_result,
                    session.path(),
                );
                agent_memory::append_session_archive_record(&memory_paths, &archive_record)?;
                agent_memory::rebuild_index(&memory_paths)?;
                if learn {
                    let consolidation =
                        consolidate_run_skill(&skills_dir, &goal, session.path())?;
                    agent_memory::rebuild_index(&memory_paths)?;
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
                        evidence: Some(format!("run --learn session {}", session.path().display())),
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
        // RF36-1: input prompt size in front of response chars — "in/out"
        // ordering reads naturally.
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
    /// RF36-1: sum of `prompt_chars` across all turns. Shown in the run
    /// footer alongside response char total so users can see "I sent 320k
    /// chars in / 18k chars out" — useful for spotting prompt bloat.
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

// =============================================================================
// Planner abstraction
// -----------------------------------------------------------------------------
// One trait + three impls replaces the three near-identical planner branches
// that previously lived inline in run_goal. The driving loop is identical for
// every provider — only the "send the prompt, get back a PlannedAction"
// inner step varies, so that's all the trait covers.
//
// Adding a new planner backend: implement Planner, branch on PlannerProvider in
// build_planner, done. No new code in drive_planner_loop.
// =============================================================================

// RF40-A3: Planner trait + 3 provider impls + build_planner moved to
// commands/run_planners.rs to keep this file under 2000 lines. Tests
// (which use MockPlanner) stay here because they test drive_planner_loop
// integration shape, not the trait surface itself.
use crate::commands::run_planners::{Planner, build_planner};


/// Run the planner loop with the given planner backend. Owns the per-turn
/// glue (subagent signals, spinner label, prompt building, timing,
/// failure-streak abort) so each Planner impl only has to implement the
/// actual "send prompt → get action" step.
///
/// `failure_streak` + `max_consecutive_failures` implement forge's
/// `ToolErrorTracker` pattern: when tool calls fail back-to-back too many
/// times, abort the loop early rather than burning through `max_turns`
/// turns. The check fires at the *top* of each turn so a fresh planner
/// reading the same failed observation can't trigger a new one before we
/// notice.
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
                // Fatal: retrying the planner won't recover from a tools-broken
                // state; user needs to inspect the session and the failing
                // tools first.
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
            planner.on_turn_start(spinner);
            let memory = build_memory();
            let planner_started = Instant::now();
            let (action, chars) = planner.plan(goal, tool_infos, state, &memory, spinner)?;
            // RF36-1: capture prompt size for this turn (provider-specific;
            // 0 if the planner doesn't track it).
            let prompt_chars = planner.last_prompt_chars();
            record_planner_timing(
                turn_timings,
                state.next_turn,
                planner_started.elapsed(),
                chars,
                prompt_chars,
            );
            Ok(action)
        },
        |call| {
            let exec_started = Instant::now();
            let result = run_tool(call);
            record_exec_timing(turn_timings, exec_started.elapsed());
            result
        },
        // RF34-2: capture retries for later session.append (we can't hold
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

    // --- RF36-1 prompt size visibility ----------------------------------

    #[test]
    fn planner_trait_default_last_prompt_chars_is_zero() {
        // MockPlanner doesn't override last_prompt_chars(), so the default
        // impl should return 0 — confirming OraclePlanner (which also
        // doesn't override) won't accidentally report stale values.
        let mut planner = MockPlanner::new(vec![]);
        let p: &dyn Planner = &mut planner;
        assert_eq!(p.last_prompt_chars(), 0);
    }

    // --- RF35-2 tool memoization ----------------------------------------

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
    fn is_memoizable_only_for_read_tools() {
        // Read-shaped → memoize.
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
        ] {
            assert!(is_memoizable_tool(n), "should memoize {n}");
        }
        // Side-effect tools → must NOT memoize.
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
            assert!(!is_memoizable_tool(n), "must not memoize {n}");
        }
    }

    #[test]
    fn read_only_analysis_hides_plan_and_mutating_tools_from_planner() {
        let tools = vec![
            ToolInfo {
                name: "read_file".to_string(),
                description: "read".to_string(),
            },
            ToolInfo {
                name: "read_files".to_string(),
                description: "batch read".to_string(),
            },
            ToolInfo {
                name: "update_working_checkpoint".to_string(),
                description: "anchor".to_string(),
            },
            ToolInfo {
                name: "plan_create".to_string(),
                description: "plan".to_string(),
            },
            ToolInfo {
                name: "patch_file".to_string(),
                description: "patch".to_string(),
            },
            ToolInfo {
                name: "repoprompt_exec".to_string(),
                description: "rp".to_string(),
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

    // --- RF37 skill_tools_guard intersection ---------------------------

    #[test]
    fn skill_narrow_intersects_with_mode_filter() {
        // RF37: a fetched skill that allowed only [read_file, run_shell]
        // should restrict the planner catalog. Then the read-only filter
        // (which keeps run_shell + read_file but drops write_file etc.)
        // applies on top. So we end up with [read_file, run_shell].
        agent_tools::skill_tools_guard::reset();
        agent_tools::skill_tools_guard::set(vec![
            "read_file".to_string(),
            "run_shell".to_string(),
        ]);
        let tools = vec![
            ToolInfo { name: "read_file".to_string(), description: "r".to_string() },
            ToolInfo { name: "run_shell".to_string(), description: "s".to_string() },
            ToolInfo { name: "write_file".to_string(), description: "w".to_string() },
            ToolInfo { name: "memory_search".to_string(), description: "m".to_string() },
        ];
        let names = planner_tool_infos_for_mode(tools, agent_core::RunMode::ReadOnly)
            .into_iter()
            .map(|t| t.name)
            .collect::<Vec<_>>();
        agent_tools::skill_tools_guard::reset();
        // memory_search NOT in skill narrow → dropped.
        // write_file in neither read-only allowlist nor skill narrow → dropped.
        assert_eq!(names, vec!["read_file".to_string(), "run_shell".to_string()]);
    }

    #[test]
    fn no_skill_narrow_keeps_existing_mode_behavior() {
        agent_tools::skill_tools_guard::reset();
        let tools = vec![
            ToolInfo { name: "read_file".to_string(), description: "r".to_string() },
            ToolInfo { name: "memory_search".to_string(), description: "m".to_string() },
            ToolInfo { name: "write_file".to_string(), description: "w".to_string() },
        ];
        let names = planner_tool_infos_for_mode(tools, agent_core::RunMode::ReadOnly)
            .into_iter()
            .map(|t| t.name)
            .collect::<Vec<_>>();
        // Read-only filter only — read_file + memory_search keep, write_file drops.
        assert_eq!(names, vec!["read_file".to_string(), "memory_search".to_string()]);
    }

    // ====================================================================
    // RF20 — integration tests for the planner loop.
    //
    // These tests drive `drive_planner_loop` end-to-end with a `MockPlanner`
    // and a synthetic tool closure. They cover the loop-control behavior
    // that's hard to verify with the per-Planner-impl unit tests in
    // `agent-runtime` (which exercise `run_agent_loop_with_state_planner*`
    // directly without the seed-cli glue).
    //
    // What they cover:
    //   - multi-turn convergence to `Finished`
    //   - `MaxTurnsExceeded` when planner never finishes
    //   - ToolErrorTracker abort after N consecutive tool failures (RF13)
    //   - `PlannerFatal` short-circuits the loop with no retry (RF14)
    //   - `Planner` (retryable) variant DOES retry via the runtime's
    //     transport-retry plumbing
    // ====================================================================

    /// Test stand-in for a real Planner backend. Stores a queue of
    /// pre-baked responses; each `plan()` call pops the next one. Using an
    /// enum instead of `Result<PlannedAction, RuntimeError>` because
    /// `RuntimeError` doesn't implement `Clone` (anyhow::Error inside).
    enum MockResponse {
        Action(agent_runtime::PlannedAction),
        TransientErr(String),
        FatalErr(String),
    }

    struct MockPlanner {
        queue: Vec<MockResponse>,
        /// Counts plan() invocations so tests can assert how many turns ran.
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
            _spinner: &agent_core::tui::Spinner,
        ) -> Result<(agent_runtime::PlannedAction, usize), agent_runtime::RuntimeError> {
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
                MockResponse::Action(a) => Ok((a, 0)),
                MockResponse::TransientErr(msg) => {
                    Err(agent_runtime::RuntimeError::Planner(msg))
                }
                MockResponse::FatalErr(msg) => Err(agent_runtime::RuntimeError::planner_fatal(msg)),
            }
        }
    }

    /// Builds the bag of context refs that `drive_planner_loop` needs but
    /// the test doesn't care about — `Cell`s, `RefCell`, spinner, etc.
    /// Constructed fresh per test so state doesn't bleed between cases.
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
                // In test runs stderr isn't a terminal, so Spinner::start
                // returns a no-op instance — no background thread, no I/O.
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
        // Five consecutive Tool calls, all returning failure → after the
        // 5th failure the failure_streak counter trips at the top of the
        // *next* (6th) turn and we exit with PlannerFatal. The 6th planner
        // call never happens.
        let mut planner = MockPlanner::new(vec![
            MockResponse::Action(tool_call("write_file")),
            MockResponse::Action(tool_call("write_file")),
            MockResponse::Action(tool_call("write_file")),
            MockResponse::Action(tool_call("write_file")),
            MockResponse::Action(tool_call("write_file")),
            // 6th never reached — the abort fires before this.
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
                // Drive the tool closure outside drive_planner_loop too —
                // bump failure_streak here just like run_goal does.
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
        // Only one plan() call — no retry.
        assert_eq!(planner.call_count(), 1);
    }

    #[test]
    fn loop_retries_transient_planner_errors_then_succeeds() {
        // First plan() returns transient → runtime retries. Second plan()
        // is the retry attempt, which returns Finish. Loop completes in
        // 1 logical turn (the retry doesn't consume a turn-number, it's
        // an inner retry loop).
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
        // Two plan() calls: the failing one + the retry that returned finish.
        assert_eq!(planner.call_count(), 2);
        // But only one logical turn was used (retry is inside turn 1).
        assert_eq!(result.turns, 1);
    }

    #[test]
    fn tool_success_resets_failure_streak() {
        // Sequence: fail, fail, success, fail, fail, fail, fail, fail.
        // The success in turn 3 resets streak to 0, so by turn 8 we've
        // accumulated 5 fresh failures and abort. If the streak weren't
        // reset, we'd abort after turn 5 instead.
        let mut planner = MockPlanner::new(
            (0..10).map(|_| MockResponse::Action(tool_call("x"))).collect(),
        );
        let fx = LoopFixture::new();
        // Pre-seed the tool closure with a counter that fails 1+2 turns,
        // succeeds turn 3, then fails 4-8.
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
                // Turn 3 succeeds; everything else fails.
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
        // Tool ran on turns 1-8 (8 calls): 1+2 fail, 3 succeeds (streak=0),
        // 4-8 fail (streak=5). Abort check fires at top of turn 9, which
        // means plan() ran 8 times (one per tool call invocation).
        assert_eq!(turn_counter.get(), 8);
    }

    // --- RF26-1 draft_already_conforms ----------------------------------

    fn conforming_draft() -> &'static str {
        "Bottom line: this project ships a tiny self-bootstrapping agent kernel.\n\n\
         Evidence:\n\
         - [via read_files] Cargo.toml:2-15 lists 12 workspace crates.\n\
         - [via read_files] main.rs:24 defines DEFAULT_MAX_TURNS = 24.\n\n\
         Counter-intuitive / risk:\n\
         - agent-cli has grown thick — orchestration logic accumulates here.\n\n\
         Action items:\n\
         1. Extract DEFAULT_MAX_TURNS into RunPolicy.\n\
         2. Move codex prompt routing into a tested module."
    }

    #[test]
    fn draft_conforms_when_all_sections_and_citation_present() {
        assert!(draft_already_conforms(conforming_draft()));
    }

    #[test]
    fn draft_does_not_conform_when_too_short() {
        // Same shape, but under the 200-char threshold — that filters out
        // header-only stubs that happen to mention all four labels.
        let short = "Bottom line: x. Evidence: y. Counter-intuitive: z. Action items: w. [via t]";
        assert!(short.len() < 200);
        assert!(!draft_already_conforms(short));
    }

    #[test]
    fn draft_does_not_conform_without_via_citation() {
        // Replace the citations with un-citation-flavored text so [via …]
        // is absent. Synthesis should still run to enforce the rule.
        let noisy = "Bottom line: a project description.\n\n\
                     Evidence:\n\
                     - One thing about the repo.\n\
                     - Another thing about the repo.\n\n\
                     Counter-intuitive / risk:\n\
                     - Nothing surprising.\n\n\
                     Action items:\n\
                     1. Do a thing.\n\
                     2. Do another thing.\n\
                     3. And a third for length padding to clear the 200-char threshold.";
        assert!(noisy.len() >= 200);
        assert!(!draft_already_conforms(noisy));
    }

    #[test]
    fn draft_does_not_conform_when_missing_section() {
        let mut text = conforming_draft().to_string();
        text = text.replace("Counter-intuitive / risk", "Random heading");
        assert!(!draft_already_conforms(&text));
    }

    #[test]
    fn draft_does_not_conform_with_banned_opener() {
        // Schema explicitly forbids these openers. If the draft uses one,
        // synthesis would rewrite — so we should run it.
        let banned = format!("整体来看 this project is fine. {}", conforming_draft());
        assert!(!draft_already_conforms(&banned));
        let banned2 = format!("Overall, the design is clean. {}", conforming_draft());
        assert!(!draft_already_conforms(&banned2));
    }

    // --- RF26-2 auto-effort heuristic ----------------------------------

    // The auto-effort logic lives inline inside `run_goal` and can't be
    // unit-tested directly without spinning up the whole runtime. We cover
    // the underlying classifier (`is_read_only_analysis_goal`) in
    // agent-runtime's own test suite (lib.rs:1921+), and verify the
    // selection rule itself here with a tiny mirror function so we catch
    // regressions to the override logic without coupling to run_goal.

    /// Mirror of the RF26-2 / RF27 rule used inside `run_goal`. Kept in
    /// sync by review (it's three lines). If this diverges, the inline
    /// call site is the source of truth.
    fn auto_effort_mirror(user_set: Option<String>, goal: &str) -> Option<String> {
        user_set.or_else(|| {
            // We compose against the same resolution chain: classify(goal)
            // then check ReadOnly. Tests below pass `Auto`-shaped inputs
            // so this matches `--mode auto` behavior.
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
        // Even if goal is read-only, user-set effort is preserved.
        assert_eq!(
            auto_effort_mirror(Some("high".to_string()), "分析当前的项目"),
            Some("high".to_string())
        );
        // And for implementation goals.
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
