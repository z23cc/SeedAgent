//! `seed chat` (and `seed` with no args): reedline REPL that loops
//! `agent_tui::Repl::read` → planner / codex / record-only depending on flags.
//!
//! Slash commands short-circuit before the planner. We keep the full handler
//! set here (rather than splitting per-command) because each handler is small
//! and they share the same `InteractiveArgs` + `last_goal` state — splitting
//! would create more boilerplate than the file saves. If this file passes
//! ~700 lines, consider extracting handlers to `commands/slash.rs`.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;

use agent_session::SessionStore;
use anyhow::{Context, Result};

use crate::commands::run::{PlannerProvider, ProviderSpec, RunGoalArgs, RunPolicy, run_goal};
use crate::commands::skill::print_skill_infos;
use crate::{ApprovalArg, Cli, Command, DEFAULT_MAX_TURNS, McpArg, doctor, memory_paths};

pub(crate) fn default_interactive_command() -> Command {
    Command::Chat {
        cwd: None,
        max_turns: DEFAULT_MAX_TURNS,
        learn: false,
        provider: "codex".to_string(),
        model: None,
        approval: ApprovalArg::Deny,
        effort: None,
        turn_timeout_secs: 600,
        mcp: None,
        mcp_allow: Vec::new(),
        plugins: false,
        codex: false,
        record_only: false,
    }
}

pub(crate) struct InteractiveArgs {
    pub(crate) cwd: Option<PathBuf>,
    pub(crate) max_turns: usize,
    pub(crate) learn: bool,
    pub(crate) provider: String,
    pub(crate) model: Option<String>,
    pub(crate) approval: ApprovalArg,
    pub(crate) effort: Option<String>,
    pub(crate) turn_timeout_secs: u64,
    pub(crate) mcp: Option<McpArg>,
    pub(crate) mcp_allow: Vec<String>,
    pub(crate) plugins: bool,
    pub(crate) codex: bool,
    pub(crate) record_only: bool,
}

/// Per-REPL session state that we keep outside `InteractiveArgs` because it's
/// REPL-local, not "what flags did the user pass on the command line."
#[derive(Default)]
struct ReplState {
    /// Last non-slash goal the user submitted. Used by `/retry`.
    last_goal: Option<String>,
}

/// Single source of truth for "where the agent thinks it is" during a REPL
/// session. Codex picks this up at next `start_turn` (via the per-turn
/// `cwd` field documented as "Override the working directory for this turn
/// and subsequent turns"). RepoPrompt picks it up lazily on the next
/// `repoprompt_*` tool call (see `agent-tools::default_repoprompt_working_dirs`).
///
/// Mutated by `/cd <path>` and (RF24-5) by skill autobind. Read by `run_goal`
/// at the top of each REPL iteration to pass as the run's cwd.
#[derive(Debug, Clone)]
pub(crate) struct SeedWorkspace {
    pub(crate) cwd: PathBuf,
    /// What RepoPrompt was last told to bind to. `None` means "we never
    /// pushed a binding from this REPL" — the lazy-align layer can still
    /// inspect RepoPrompt's actual state if needed. Used to avoid
    /// re-calling `bind_context` when nothing changed.
    pub(crate) repoprompt_bound: Option<Vec<PathBuf>>,
}

impl SeedWorkspace {
    pub(crate) fn new(cwd: PathBuf) -> Self {
        Self {
            cwd,
            repoprompt_bound: None,
        }
    }

    /// Update the cwd. Caller is expected to flush downstream effects
    /// (Codex `set_cwd`, RepoPrompt rebind) lazily on the next op.
    /// Returns the previous cwd so callers can show "OLD → NEW".
    pub(crate) fn set_cwd(&mut self, new_cwd: PathBuf) -> PathBuf {
        // Invalidate the RP binding cache — a different cwd means the next
        // RP call should re-bind even if the bound vector technically matched
        // (different working_dirs[0] = different context).
        self.repoprompt_bound = None;
        std::mem::replace(&mut self.cwd, new_cwd)
    }
}

/// Resolve a user-supplied `/cd` target into an absolute, canonicalized
/// directory path.
///
/// Supports:
///   - `~` and `~/...` (tilde expansion via `$HOME`)
///   - relative paths (joined against the current REPL cwd)
///   - absolute paths
///
/// Validates that the target exists and is a directory.
pub(crate) fn resolve_cd_target(target: &str, current_cwd: &Path) -> Result<PathBuf> {
    let expanded: PathBuf = if target == "~" {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(target))
    } else if let Some(rest) = target.strip_prefix("~/") {
        match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join(rest),
            None => PathBuf::from(target),
        }
    } else {
        PathBuf::from(target)
    };
    let abs = if expanded.is_absolute() {
        expanded
    } else {
        current_cwd.join(expanded)
    };
    let canonical = abs
        .canonicalize()
        .with_context(|| format!("cannot resolve {}", abs.display()))?;
    if !canonical.is_dir() {
        anyhow::bail!("not a directory: {}", canonical.display());
    }
    Ok(canonical)
}

pub(crate) fn run_interactive(
    cli: &Cli,
    store: &SessionStore,
    mut args: InteractiveArgs,
) -> Result<()> {
    let initial_cwd = args.cwd.clone().unwrap_or(env::current_dir()?);
    let mut workspace = SeedWorkspace::new(initial_cwd);
    std::fs::create_dir_all(&cli.sessions_dir)?;
    let history_path = cli.sessions_dir.join(".seed_history");
    let mut repl = agent_tui::Repl::new(history_path);
    let mode = if args.record_only {
        "record"
    } else if args.codex {
        "codex"
    } else {
        "planner"
    };
    let mut state = ReplState::default();

    agent_tui::print_banner();
    loop {
        let prompt = agent_tui::PromptState::new(
            workspace.cwd.clone(),
            mode,
            args.provider.clone(),
            args.model.clone(),
        );

        match repl.read(&prompt)? {
            agent_tui::ReplInput::Line(input) => {
                // Shell escape: `!cmd` runs `cmd` in cwd via the user's shell
                // and prints its output. Must precede the slash dispatcher so
                // `!` is not mistaken for an unknown command.
                if let Some(rest) = input.strip_prefix(agent_tui::SHELL_ESCAPE_PREFIX) {
                    if let Err(err) = run_shell_escape(&workspace.cwd, rest.trim()) {
                        agent_tui::print_error(err);
                    }
                    continue;
                }

                if handle_interactive_command(
                    cli,
                    store,
                    &mut args,
                    &mut state,
                    &mut workspace,
                    &input,
                )? {
                    break;
                }

                if input.starts_with('/') {
                    continue;
                }

                state.last_goal = Some(input.clone());
                if let Err(err) = run_goal(RunGoalArgs {
                    store,
                    goal: input,
                    cwd: Some(workspace.cwd.clone()),
                    use_llm: !args.record_only && !args.codex,
                    use_codex: args.codex,
                    learn: args.learn,
                    skills_dir: cli.skills_dir.clone(),
                    policy: RunPolicy {
                        max_turns: args.max_turns,
                        turn_timeout_secs: args.turn_timeout_secs,
                        ..Default::default()
                    },
                    provider: ProviderSpec {
                        kind: PlannerProvider::from_id(&args.provider),
                        model: args.model.clone(),
                        approval: args.approval.clone(),
                        effort: args.effort.clone(),
                        mcp: args.mcp.clone(),
                        mcp_allow: args.mcp_allow.clone(),
                        plugins: args.plugins,
                    },
                }) {
                    agent_tui::print_error(err);
                }
            }
            agent_tui::ReplInput::Empty | agent_tui::ReplInput::Continue => {}
            agent_tui::ReplInput::Exit => break,
        }
    }

    Ok(())
}

fn handle_interactive_command(
    cli: &Cli,
    store: &SessionStore,
    args: &mut InteractiveArgs,
    state: &mut ReplState,
    workspace: &mut SeedWorkspace,
    input: &str,
) -> Result<bool> {
    let trimmed = input.trim();
    // We split into (head, rest) once so handlers can ignore the leading verb
    // and just look at the args. `split_once(' ')` returns None for bare
    // commands like `/help`, which we treat as `head=trimmed, rest=""`.
    let (head, rest) = match trimmed.split_once(char::is_whitespace) {
        Some((h, r)) => (h, r.trim()),
        None => (trimmed, ""),
    };
    match head {
        "/exit" | "/quit" | ":q" => Ok(true),
        "/help" | "?" => {
            agent_tui::print_help();
            Ok(false)
        }
        "/doctor" => {
            doctor::doctor(&cli.skills_dir, store)?;
            Ok(false)
        }
        "/providers" => {
            doctor::show_providers(&args.provider, args.model.as_deref(), false)?;
            Ok(false)
        }
        "/skills" => {
            let skills = agent_skills::list_skill_infos(&cli.skills_dir)?;
            let skills = skills.into_iter().take(20).collect::<Vec<_>>();
            print_skill_infos(&skills);
            Ok(false)
        }
        "/tools" => {
            handle_tools_command();
            Ok(false)
        }
        "/model" => {
            handle_model_command(args, rest)?;
            Ok(false)
        }
        "/effort" => {
            handle_effort_command(args, rest);
            Ok(false)
        }
        "/memory" => {
            handle_memory_command(cli, &workspace.cwd, rest)?;
            Ok(false)
        }
        "/plan" => {
            handle_plan_command(&workspace.cwd, rest)?;
            Ok(false)
        }
        "/plans" => {
            handle_plans_command(&workspace.cwd)?;
            Ok(false)
        }
        "/dump" => {
            handle_dump_command(store);
            Ok(false)
        }
        "/compact" => {
            handle_compact_command(cli, &workspace.cwd)?;
            Ok(false)
        }
        "/new" => {
            handle_new_command(state);
            Ok(false)
        }
        "/retry" => {
            handle_retry_command(cli, store, args, state, &workspace.cwd)?;
            Ok(false)
        }
        "/cd" => {
            handle_cd_command(workspace, rest);
            Ok(false)
        }
        cmd if cmd.starts_with('/') => {
            agent_tui::print_error(format!("unknown command: {cmd}"));
            Ok(false)
        }
        _ => Ok(false),
    }
}

// --- shell escape -----------------------------------------------------------

/// Run a shell command in `cwd` and stream stdout/stderr to the terminal.
/// We use `sh -c` (or `cmd /C` on Windows) so the user can use shell features
/// (pipes, globs, env expansion) the way they'd expect inside an interactive
/// shell. Status code is printed on non-zero exit, otherwise silent.
fn run_shell_escape(cwd: &Path, cmd: &str) -> Result<()> {
    if cmd.is_empty() {
        println!("(empty shell escape — usage: !<command>)");
        return Ok(());
    }
    #[cfg(windows)]
    let mut command = {
        let mut c = StdCommand::new("cmd");
        c.arg("/C").arg(cmd);
        c
    };
    #[cfg(not(windows))]
    let mut command = {
        let mut c = StdCommand::new("sh");
        c.arg("-c").arg(cmd);
        c
    };
    let status = command.current_dir(cwd).status()?;
    if !status.success() {
        let code = status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".to_string());
        println!("(shell exit {code})");
    }
    Ok(())
}

// --- /tools ------------------------------------------------------------------

fn handle_tools_command() {
    let registry = agent_tools::seed_registry();
    let infos = registry.infos();
    println!("registered planner tools ({} total)", infos.len());
    let max_name = infos.iter().map(|i| i.name.len()).max().unwrap_or(0);
    for info in &infos {
        // Description can be multi-line; we only show the first non-empty line
        // here. Run `seed doctor` for full descriptions if you need them.
        let summary = info
            .description
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("")
            .trim();
        println!("  {:<width$}  {summary}", info.name, width = max_name);
    }
}

// --- /model ------------------------------------------------------------------

/// Implements `/model [<slug>|none|list]`:
///   - empty arg → print current provider + model + suggested next steps
///   - "list"    → print the codex models table (only useful for the codex
///                 provider; other providers' model list isn't cached)
///   - "none"    → clear `args.model`, the planner will use the backend's
///                 own default (codex falls back to `~/.codex/config.toml`)
///   - anything else → set as the new model. No validation here — codex /
///                     the provider rejects unknown slugs at runtime, which
///                     surfaces as a planner error on the next turn.
fn handle_model_command(args: &mut InteractiveArgs, rest: &str) -> Result<()> {
    if rest.is_empty() {
        match args.model.as_deref() {
            Some(model) => {
                println!(
                    "model: {model}  (provider: {})\n  /model <slug>  switch\n  /model none    revert to backend default\n  /model list    show codex slugs",
                    args.provider
                );
            }
            None => {
                println!(
                    "model: (backend default)  (provider: {})\n  /model <slug>  switch\n  /model list    show codex slugs",
                    args.provider
                );
            }
        }
        return Ok(());
    }
    if rest == "list" {
        // Only meaningful for codex right now — the cache is codex-specific.
        // For other providers, point the user at `/providers` instead.
        if args.provider != "codex" {
            agent_tui::print_error(format!(
                "/model list only works for the codex provider (current: {}). Try /providers.",
                args.provider
            ));
            return Ok(());
        }
        // Reuse the same renderer as `seed codex-models`. False/false = table, no hidden.
        return crate::commands::codex::run_codex_models(false, false);
    }
    if rest == "none" || rest == "default" || rest == "-" {
        let prev = args.model.take();
        match prev {
            Some(old) => println!("model: cleared (was {old}) — will use backend default"),
            None => println!("model: already at backend default"),
        }
        return Ok(());
    }
    let prev = args.model.replace(rest.to_string());
    match prev {
        Some(old) if old == rest => println!("model: already {rest}"),
        Some(old) => println!("model: {old} → {rest}"),
        None => println!("model: (backend default) → {rest}"),
    }
    println!("  next turn will use the new model");
    Ok(())
}

// --- /effort -----------------------------------------------------------------

/// Implements `/effort [low|medium|high|minimal|none]`. Symmetric with
/// `/model`: empty prints current, "none"/"default"/"-" clears, anything else
/// sets the value. Soft-warns when the provider doesn't honor effort but does
/// not block the set — the user might switch provider next.
fn handle_effort_command(args: &mut InteractiveArgs, rest: &str) {
    if rest.is_empty() {
        match args.effort.as_deref() {
            Some(effort) => println!(
                "effort: {effort}  (provider: {})\n  /effort <level>  switch (low|medium|high|minimal)\n  /effort none     clear",
                args.provider
            ),
            None => println!(
                "effort: (provider default)  (provider: {})\n  /effort <level>  set (low|medium|high|minimal)",
                args.provider
            ),
        }
        return;
    }
    if rest == "none" || rest == "default" || rest == "-" {
        let prev = args.effort.take();
        match prev {
            Some(old) => println!("effort: cleared (was {old}) — will use provider default"),
            None => println!("effort: already at provider default"),
        }
        return;
    }
    let normalized = rest.to_ascii_lowercase();
    if !matches!(
        normalized.as_str(),
        "low" | "medium" | "high" | "minimal"
    ) {
        agent_tui::print_error(format!(
            "unknown effort level: {rest} (try low|medium|high|minimal|none)"
        ));
        return;
    }
    if args.provider != "codex" {
        // Soft warning — set anyway since the user may switch provider.
        eprintln!(
            "note: effort is honored by codex; current provider is {} so this is a no-op until you switch",
            args.provider
        );
    }
    let prev = args.effort.replace(normalized.clone());
    match prev {
        Some(old) if old == normalized => println!("effort: already {normalized}"),
        Some(old) => println!("effort: {old} → {normalized}"),
        None => println!("effort: (provider default) → {normalized}"),
    }
    println!("  next turn will use the new effort");
}

// --- /memory -----------------------------------------------------------------

fn handle_memory_command(cli: &Cli, cwd: &Path, query: &str) -> Result<()> {
    if query.is_empty() {
        println!("usage: /memory <query>");
        return Ok(());
    }
    let paths = memory_paths(cwd, &cli.skills_dir, &cli.sessions_dir);
    let index = agent_memory::load_or_rebuild_index(&paths)?;
    let hits = agent_memory::search_index(&index, query, 8);
    if hits.is_empty() {
        println!("(no memory entries match {query:?})");
        return Ok(());
    }
    println!("memory hits for {query:?} ({} shown)", hits.len());
    for entry in &hits {
        let snippet = entry
            .summary
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("")
            .trim();
        let snippet = if snippet.len() > 120 {
            format!("{}…", &snippet[..120])
        } else {
            snippet.to_string()
        };
        println!(
            "  [{}] {}  {}  ({})",
            entry.layer_label(),
            entry.id,
            snippet,
            entry.path.display()
        );
    }
    Ok(())
}

// --- /plan + /plans ----------------------------------------------------------

fn plan_store_for(cwd: &Path) -> agent_plan::PlanStore {
    agent_plan::PlanStore::new(cwd.join("plans"))
}

fn handle_plan_command(cwd: &Path, rest: &str) -> Result<()> {
    let store = plan_store_for(cwd);
    let id_filter = if rest.is_empty() { None } else { Some(rest) };

    let snapshot = match id_filter {
        Some(id) => Some(store.snapshot(Some(id))?),
        None => {
            // No id given → pick the active plan (highest priority) or fall
            // back to the most recently updated one. Either way: one line per
            // visible entry, then a short next-up block for the selected.
            if let Some(brief) = store.active_brief()? {
                Some(store.snapshot(Some(&brief.plan_id))?)
            } else {
                store.list()?.into_iter().next()
            }
        }
    };

    let Some(snapshot) = snapshot else {
        println!("(no plans found in {})", cwd.join("plans").display());
        println!("  tip: `seed plan create` or have the planner call plan_create");
        return Ok(());
    };

    let total = snapshot.items.len();
    let done = total.saturating_sub(snapshot.unchecked_count);
    println!(
        "plan {} · {} · {:?}",
        snapshot.state.id, snapshot.state.title, snapshot.state.status
    );
    println!(
        "  progress: {done}/{total} done · {} unchecked task items",
        snapshot.task_unchecked_count
    );
    match snapshot.next_item.as_ref() {
        Some(item) => {
            let kind = match item.kind {
                agent_plan::PlanItemKind::Task => "TASK",
                agent_plan::PlanItemKind::Verify => "VERIFY",
                agent_plan::PlanItemKind::Fix => "FIX",
            };
            let flags = match (item.delegate, item.parallel) {
                (true, true) => " [D,P]",
                (true, false) => " [D]",
                (false, true) => " [P]",
                (false, false) => "",
            };
            println!("  next [{kind}{flags}]: {}", item.text);
        }
        None => println!("  next: (all items checked — ready to settle)"),
    }
    println!("  file: {}", snapshot.state.plan_path.display());
    Ok(())
}

fn handle_plans_command(cwd: &Path) -> Result<()> {
    let store = plan_store_for(cwd);
    let snapshots = store.list()?;
    if snapshots.is_empty() {
        println!("(no plans in {})", cwd.join("plans").display());
        return Ok(());
    }
    println!("plans ({} total)", snapshots.len());
    for snap in &snapshots {
        let total = snap.items.len();
        let done = total.saturating_sub(snap.unchecked_count);
        println!(
            "  {}  {}  ({}/{} done · status: {:?})",
            snap.state.id, snap.state.title, done, total, snap.state.status
        );
    }
    Ok(())
}

// --- /dump -------------------------------------------------------------------

fn handle_dump_command(store: &SessionStore) {
    match store.last_session_path() {
        Ok(path) => {
            println!("last session: {}", path.display());
            println!("  seed reflect       # summarize");
            println!("  seed replay        # step-through");
        }
        Err(err) => {
            agent_tui::print_error(err);
        }
    }
}

// --- /compact ----------------------------------------------------------------

/// Forgecode's `/compact` summarizes the running chat into a checkpoint. In
/// seed each REPL turn is its own session and successful runs already
/// auto-archive at the end (see `run.rs:785`), so the residual user-facing
/// gap is "I edited L2/L3 by hand, please refresh the L1 index" or "I
/// abandoned a run mid-way, recompute the picture." Both are
/// `rebuild_index`. Report the before/after entry count so the user sees
/// something happened.
fn handle_compact_command(cli: &Cli, cwd: &Path) -> Result<()> {
    let paths = memory_paths(cwd, &cli.skills_dir, &cli.sessions_dir);
    let before = agent_memory::load_or_rebuild_index(&paths)
        .map(|i| i.entries.len())
        .unwrap_or(0);
    let after = agent_memory::rebuild_index(&paths)?;
    println!(
        "memory index rebuilt: {} → {} entries (memory dir: {})",
        before,
        after.entries.len(),
        paths.memory_dir.display()
    );
    Ok(())
}

// --- /cd ---------------------------------------------------------------------

/// `/cd [<path>]` — change the REPL workspace cwd.
///
/// Empty arg prints the current cwd. A valid path updates `workspace.cwd`;
/// both Codex and RepoPrompt pick it up on their next op:
///   - Codex via per-turn `TurnStartParams.cwd` (always read fresh from
///     `CodexAppServerConfig.cwd`, set at `run_goal` construction time).
///   - RepoPrompt via `default_repoprompt_working_dirs(ctx.cwd)` and (RF24-5)
///     the lazy `bind_context` alignment that runs before each rp tool call.
///
/// Invalid paths print an error and leave state unchanged — we never want
/// to land the REPL in a phantom cwd that future commands silently fail on.
fn handle_cd_command(workspace: &mut SeedWorkspace, rest: &str) {
    if rest.is_empty() {
        println!("cwd: {}", workspace.cwd.display());
        println!("  /cd <path>     change workspace cwd (supports ~ and relative)");
        return;
    }
    match resolve_cd_target(rest, &workspace.cwd) {
        Ok(new_cwd) => {
            if new_cwd == workspace.cwd {
                println!("cwd: already at {} (no change)", new_cwd.display());
                return;
            }
            let old = workspace.set_cwd(new_cwd.clone());
            println!("cwd: {} → {}", old.display(), new_cwd.display());
            println!("  next codex turn + repoprompt call will sync to the new cwd");
        }
        Err(err) => agent_tui::print_error(err),
    }
}

// --- /new --------------------------------------------------------------------

fn handle_new_command(state: &mut ReplState) {
    // Each goal already creates its own session UUID — there's no in-REPL
    // running session to close. /new just resets retry state and confirms.
    let had_retry = state.last_goal.is_some();
    state.last_goal = None;
    if had_retry {
        println!("ready for a new goal (cleared /retry buffer)");
    } else {
        println!("ready for a new goal");
    }
}

// --- /retry ------------------------------------------------------------------

fn handle_retry_command(
    cli: &Cli,
    store: &SessionStore,
    args: &InteractiveArgs,
    state: &mut ReplState,
    cwd: &Path,
) -> Result<()> {
    let Some(goal) = state.last_goal.clone() else {
        agent_tui::print_error("nothing to retry — submit a goal first");
        return Ok(());
    };
    println!("retrying: {goal}");
    let result = run_goal(RunGoalArgs {
        store,
        goal: goal.clone(),
        cwd: Some(cwd.to_path_buf()),
        use_llm: !args.record_only && !args.codex,
        use_codex: args.codex,
        learn: args.learn,
        skills_dir: cli.skills_dir.clone(),
        policy: RunPolicy {
            max_turns: args.max_turns,
            turn_timeout_secs: args.turn_timeout_secs,
            ..Default::default()
        },
        provider: ProviderSpec {
            kind: PlannerProvider::from_id(&args.provider),
            model: args.model.clone(),
            approval: args.approval.clone(),
            effort: args.effort.clone(),
            mcp: args.mcp.clone(),
            mcp_allow: args.mcp_allow.clone(),
            plugins: args.plugins,
        },
    });
    // Refresh last_goal even on failure — the user clearly wants to keep
    // iterating on this goal, so /retry should keep working.
    state.last_goal = Some(goal);
    if let Err(err) = result {
        agent_tui::print_error(err);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Unique-per-test scratch dir under /tmp. Mirrors the convention used in
    /// `agent-skills`, `agent-memory`, etc. — we don't pull in `tempfile` just
    /// for this since the workspace already has a pattern.
    fn scratch_dir(tag: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!(
            "seed-cli-interactive-{}-{}-{}",
            tag,
            std::process::id(),
            n
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn fresh_cli(root: &Path) -> Cli {
        Cli {
            sessions_dir: root.join("sessions"),
            skills_dir: root.join("skills"),
            command: None,
        }
    }

    fn args() -> InteractiveArgs {
        InteractiveArgs {
            cwd: None,
            max_turns: 24,
            learn: false,
            provider: "codex".to_string(),
            model: None,
            approval: ApprovalArg::Deny,
            effort: None,
            turn_timeout_secs: 600,
            mcp: None,
            mcp_allow: Vec::new(),
            plugins: false,
            codex: false,
            record_only: false,
        }
    }

    // --- /model ----------------------------------------------------------

    #[test]
    fn model_command_sets_model_from_default() {
        let mut a = args();
        handle_model_command(&mut a, "gpt-5.5").unwrap();
        assert_eq!(a.model.as_deref(), Some("gpt-5.5"));
    }

    #[test]
    fn model_command_replaces_existing_model() {
        let mut a = args();
        a.model = Some("gpt-5.4".to_string());
        handle_model_command(&mut a, "gpt-5.5").unwrap();
        assert_eq!(a.model.as_deref(), Some("gpt-5.5"));
    }

    #[test]
    fn model_command_none_clears_model() {
        let mut a = args();
        a.model = Some("gpt-5.5".to_string());
        handle_model_command(&mut a, "none").unwrap();
        assert_eq!(a.model, None);
    }

    #[test]
    fn model_command_default_alias_also_clears() {
        let mut a = args();
        a.model = Some("gpt-5.5".to_string());
        handle_model_command(&mut a, "default").unwrap();
        assert_eq!(a.model, None);
    }

    #[test]
    fn model_command_empty_does_not_modify() {
        let mut a = args();
        a.model = Some("gpt-5.5".to_string());
        handle_model_command(&mut a, "").unwrap();
        // Empty just prints; state unchanged.
        assert_eq!(a.model.as_deref(), Some("gpt-5.5"));
    }

    #[test]
    fn model_command_list_with_non_codex_provider_does_not_panic() {
        // For non-codex providers we print an error and bail — should not
        // touch args.model and must not panic on filesystem reads.
        let mut a = args();
        a.provider = "openai".to_string();
        a.model = Some("gpt-4o".to_string());
        handle_model_command(&mut a, "list").unwrap();
        assert_eq!(a.model.as_deref(), Some("gpt-4o"));
        assert_eq!(a.provider, "openai");
    }

    // --- /effort ---------------------------------------------------------

    #[test]
    fn effort_command_sets_level_lowercased() {
        let mut a = args();
        handle_effort_command(&mut a, "HIGH");
        assert_eq!(a.effort.as_deref(), Some("high"));
    }

    #[test]
    fn effort_command_rejects_unknown_level() {
        let mut a = args();
        a.effort = Some("low".to_string());
        handle_effort_command(&mut a, "garbage");
        // Unknown level → state unchanged, error printed.
        assert_eq!(a.effort.as_deref(), Some("low"));
    }

    #[test]
    fn effort_command_none_clears() {
        let mut a = args();
        a.effort = Some("high".to_string());
        handle_effort_command(&mut a, "none");
        assert_eq!(a.effort, None);
    }

    #[test]
    fn effort_command_empty_prints_without_modifying() {
        let mut a = args();
        a.effort = Some("medium".to_string());
        handle_effort_command(&mut a, "");
        assert_eq!(a.effort.as_deref(), Some("medium"));
    }

    #[test]
    fn effort_command_non_codex_still_sets_but_warns() {
        // Soft warning behavior: we set anyway because the user might switch
        // provider next. Verify the state actually changed.
        let mut a = args();
        a.provider = "openai".to_string();
        handle_effort_command(&mut a, "high");
        assert_eq!(a.effort.as_deref(), Some("high"));
    }

    // --- /new + /retry ---------------------------------------------------

    #[test]
    fn new_command_clears_retry_buffer() {
        let mut s = ReplState {
            last_goal: Some("analyze runtime".to_string()),
        };
        handle_new_command(&mut s);
        assert_eq!(s.last_goal, None);
    }

    #[test]
    fn new_command_is_noop_when_buffer_empty() {
        let mut s = ReplState::default();
        handle_new_command(&mut s);
        assert_eq!(s.last_goal, None);
    }

    // --- /memory ---------------------------------------------------------

    #[test]
    fn memory_command_empty_query_prints_usage() {
        let root = scratch_dir("memory-empty");
        let cli = fresh_cli(&root);
        // Empty query → early return Ok, no filesystem touch needed beyond
        // path construction.
        handle_memory_command(&cli, &root, "").unwrap();
    }

    #[test]
    fn memory_command_no_hits_does_not_error() {
        let root = scratch_dir("memory-no-hits");
        std::fs::create_dir_all(root.join("skills")).unwrap();
        std::fs::create_dir_all(root.join("sessions")).unwrap();
        let cli = fresh_cli(&root);
        // Fresh dir → only the seeded L0/L2 stubs exist. A query unlikely to
        // match anything in the defaults should produce zero hits without
        // panicking.
        handle_memory_command(&cli, &root, "xyzzy_no_such_term_42").unwrap();
    }

    // --- /compact --------------------------------------------------------

    #[test]
    fn compact_command_rebuilds_index_in_fresh_dir() {
        let root = scratch_dir("compact-fresh");
        std::fs::create_dir_all(root.join("skills")).unwrap();
        std::fs::create_dir_all(root.join("sessions")).unwrap();
        let cli = fresh_cli(&root);
        // Should succeed even with no pre-existing memory dir — the call
        // path ensures layout, builds index, writes file.
        handle_compact_command(&cli, &root).unwrap();
        assert!(root.join("memory").join("index.json").is_file());
    }

    // --- /plan + /plans --------------------------------------------------

    #[test]
    fn plans_command_handles_empty_directory() {
        let root = scratch_dir("plans-empty");
        // No plans/ directory at all — should print "no plans" without
        // erroring.
        handle_plans_command(&root).unwrap();
    }

    #[test]
    fn plan_command_handles_empty_directory() {
        let root = scratch_dir("plan-empty");
        handle_plan_command(&root, "").unwrap();
    }

    #[test]
    fn plans_command_lists_created_plan() {
        let root = scratch_dir("plans-list");
        let store = agent_plan::PlanStore::new(root.join("plans"));
        store
            .create(agent_plan::CreatePlan {
                title: "Test plan".to_string(),
                task: "do a thing".to_string(),
                steps: vec!["step one".to_string(), "step two".to_string()],
                source_export_path: None,
            })
            .unwrap();
        // Must not error and must find at least one plan via list().
        handle_plans_command(&root).unwrap();
        assert!(!store.list().unwrap().is_empty());
    }

    // --- /tools ----------------------------------------------------------

    #[test]
    fn tools_command_renders_without_panicking() {
        // No state mutations — just exercise the formatter against the real
        // registry to catch regressions like an empty registry or a panic
        // inside `infos()`.
        handle_tools_command();
        let registry = agent_tools::seed_registry();
        assert!(!registry.infos().is_empty());
    }

    // --- shell escape ----------------------------------------------------

    #[test]
    fn shell_escape_empty_input_does_not_run_command() {
        let root = scratch_dir("shell-empty");
        // Should not error and should not actually shell out.
        run_shell_escape(&root, "").unwrap();
    }

    #[test]
    fn shell_escape_runs_true_successfully() {
        // `true` exists on every POSIX system; on Windows the test compiles
        // out under cfg(not(windows)) so skip there.
        #[cfg(not(windows))]
        {
            let root = scratch_dir("shell-true");
            run_shell_escape(&root, "true").unwrap();
        }
    }

    // --- /cd + SeedWorkspace ---------------------------------------------

    #[test]
    fn workspace_set_cwd_returns_previous_and_invalidates_bound() {
        let mut ws = SeedWorkspace::new(PathBuf::from("/tmp/seed-A"));
        ws.repoprompt_bound = Some(vec![PathBuf::from("/tmp/seed-A")]);
        let prev = ws.set_cwd(PathBuf::from("/tmp/seed-B"));
        assert_eq!(prev, PathBuf::from("/tmp/seed-A"));
        assert_eq!(ws.cwd, PathBuf::from("/tmp/seed-B"));
        // Critical: changing cwd must invalidate the cached binding so the
        // next rp call re-binds. Otherwise the lazy-align layer thinks
        // we're still good and the agent talks to the wrong workspace.
        assert!(ws.repoprompt_bound.is_none());
    }

    #[test]
    fn resolve_cd_absolute_existing_dir() {
        let root = scratch_dir("cd-abs");
        let resolved = resolve_cd_target(root.to_str().unwrap(), Path::new("/")).unwrap();
        // canonicalize() may turn /var → /private/var on macOS; just check
        // that the result exists and is a directory.
        assert!(resolved.is_dir(), "{} not a dir", resolved.display());
    }

    #[test]
    fn resolve_cd_relative_resolves_against_current() {
        let root = scratch_dir("cd-rel-parent");
        let child = root.join("child");
        std::fs::create_dir_all(&child).unwrap();
        let resolved = resolve_cd_target("child", &root).unwrap();
        assert_eq!(resolved.canonicalize().unwrap(), child.canonicalize().unwrap());
    }

    #[test]
    fn resolve_cd_nonexistent_errors() {
        let root = scratch_dir("cd-nope");
        let err = resolve_cd_target("does-not-exist-xyz", &root).unwrap_err();
        // We don't pin the exact message — `canonicalize` errors vary by OS —
        // but it should mention the bad path so the user can debug.
        let msg = format!("{err:#}");
        assert!(
            msg.contains("does-not-exist-xyz") || msg.contains("cannot resolve"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn resolve_cd_file_errors_with_not_a_directory() {
        let root = scratch_dir("cd-file");
        let file = root.join("not-a-dir.txt");
        std::fs::write(&file, b"hi").unwrap();
        let err = resolve_cd_target(file.to_str().unwrap(), &root).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("not a directory"), "unexpected error: {msg}");
    }

    #[test]
    fn resolve_cd_tilde_expands_to_home() {
        // Skip if HOME isn't set (some CI). When it is, "~" should resolve
        // to the canonical home dir if it exists.
        if let Some(home) = std::env::var_os("HOME") {
            let home_path = PathBuf::from(home);
            if home_path.is_dir() {
                let resolved = resolve_cd_target("~", Path::new("/")).unwrap();
                assert_eq!(resolved, home_path.canonicalize().unwrap());
            }
        }
    }

    #[test]
    fn cd_command_empty_arg_does_not_mutate_cwd() {
        let root = scratch_dir("cd-empty");
        let mut ws = SeedWorkspace::new(root.clone());
        handle_cd_command(&mut ws, "");
        assert_eq!(ws.cwd, root);
    }

    #[test]
    fn cd_command_invalid_path_does_not_mutate_cwd() {
        let root = scratch_dir("cd-invalid");
        let mut ws = SeedWorkspace::new(root.clone());
        ws.repoprompt_bound = Some(vec![root.clone()]);
        handle_cd_command(&mut ws, "/this/does/not/exist/anywhere/xyz");
        // Error path: cwd stays put, AND the cached binding stays valid
        // (we only invalidate on successful change).
        assert_eq!(ws.cwd, root);
        assert!(ws.repoprompt_bound.is_some());
    }

    #[test]
    fn cd_command_same_dir_is_noop() {
        let root = scratch_dir("cd-same");
        let bound_marker = vec![root.clone()];
        let mut ws = SeedWorkspace::new(root.canonicalize().unwrap());
        ws.repoprompt_bound = Some(bound_marker.clone());
        let same_path = ws.cwd.to_string_lossy().to_string();
        handle_cd_command(&mut ws, &same_path);
        // Same target → no change → binding cache should NOT be invalidated.
        assert_eq!(ws.repoprompt_bound, Some(bound_marker));
    }

    #[test]
    fn cd_command_valid_path_updates_workspace() {
        let root = scratch_dir("cd-valid-root");
        let target = scratch_dir("cd-valid-target");
        let mut ws = SeedWorkspace::new(root.canonicalize().unwrap());
        handle_cd_command(&mut ws, target.to_str().unwrap());
        assert_eq!(ws.cwd, target.canonicalize().unwrap());
        // Successful change invalidates the binding cache.
        assert!(ws.repoprompt_bound.is_none());
    }
}
