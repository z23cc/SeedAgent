//! `seed doctor` + `seed providers` output. Both are read-only diagnostics —
//! no LLM, no state mutation. Lives in its own module so the main binary
//! entry point stays focused on dispatch.

use std::env;
use std::path::Path;

use agent_llm::{ModelId, ProviderRouter};
use agent_session::SessionStore;
use anyhow::Result;

use crate::commands::codex_session::CodexSession;
use crate::commands::run::ModeArg;

pub(crate) fn doctor(skills_dir: &Path, store: &SessionStore) -> Result<()> {
    let registry = agent_tools::seed_registry();
    println!("seed doctor");
    println!("- cwd: {}", env::current_dir()?.display());
    println!("- sessions: {}", store.root().display());
    println!("- skills: {}", skills_dir.display());
    println!("- tui: {}", agent_tui::status());
    println!(
        "- repoprompt: {}",
        agent_repoprompt::default_cli_path().display()
    );
    println!("- tools: {}", registry.names().join(", "));
    println!(
        "- providers: {}",
        agent_llm::built_in_providers()
            .iter()
            .map(|provider| provider.id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("- delegates: codex-app-server, repoprompt-oracle");
    // RF25-3: cwd sync health check. Always run; for one-shot `seed doctor`
    // (no REPL session), we just print env::current_dir() + any cached RP
    // bound state. For REPL `/doctor` callers see `cwd_health_check` below.
    cwd_health_check(&env::current_dir()?, None)?;
    // RF28-1: run-mode health check. Same shape — for one-shot doctor there's
    // no REPL session so we pass None for the REPL-pinned mode and report
    // only the process-global guard's current value (defaults to
    // Implementation if no run has fired yet in this process).
    run_mode_health_check(None)?;
    Ok(())
}

/// RF28-1: surface the active `RunMode` so users can confirm what toolset
/// the next run will have access to. Two pieces of state:
///   - `run_mode_guard::current()`: the *process-global* guard set by the
///     most recent `run_goal`. For one-shot `seed doctor` this is the
///     default (Implementation) since no run has executed yet.
///   - `repl_pin`: the REPL's `args.mode` setting (`Auto`/`Read`/`Write`).
///     `Auto` means "classify each goal"; `Read`/`Write` pin the mode for
///     every subsequent goal until `/mode` changes it again.
///
/// Together they answer "what mode will my next prompt run under?" without
/// the user having to read source code.
pub(crate) fn run_mode_health_check(repl_pin: Option<ModeArg>) -> Result<()> {
    println!("- run-mode:");
    let live = agent_tools::run_mode_guard::current();
    let live_label = match live {
        agent_core::RunMode::ReadOnly => "read-only",
        agent_core::RunMode::Implementation => "implementation",
    };
    println!("    guard:         {live_label}  (set by the most recent run_goal)");
    match repl_pin {
        Some(ModeArg::Auto) => println!(
            "    repl pin:      auto       (each goal classified via keyword)"
        ),
        Some(ModeArg::Read) => println!(
            "    repl pin:      read       (next goal forced read-only)"
        ),
        Some(ModeArg::Write) => println!(
            "    repl pin:      write      (next goal forced implementation)"
        ),
        None => println!(
            "    repl pin:      N/A        (one-shot, set per run via --mode)"
        ),
    }
    Ok(())
}

/// Surface cwd-sync state across the three subsystems that all need to
/// agree on "where the agent is": `workspace.cwd` (REPL's truth), the
/// `CodexSession`'s cached client cwd (only meaningful inside a REPL),
/// and the `repoprompt_sync` bound-window cache (process-global).
///
/// Prints a `MISMATCH` marker next to any value that doesn't match
/// `workspace_cwd`. The goal: one command to debug "agent is reading
/// files from the wrong workspace" without grepping logs.
pub(crate) fn cwd_health_check(
    workspace_cwd: &Path,
    codex_session: Option<&CodexSession>,
) -> Result<()> {
    println!("- cwd-sync:");
    println!("    workspace.cwd: {}", workspace_cwd.display());

    // Codex session (only present in REPL).
    match codex_session {
        Some(cs) if cs.is_live() => match cs.client_cwd() {
            Some(c) => {
                let marker = if c == workspace_cwd { "ok" } else { "MISMATCH" };
                println!("    codex client:  {}  [{marker}]", c.display());
            }
            None => println!("    codex client:  live, cwd unset (next turn will inherit workspace.cwd)"),
        },
        Some(_) => println!("    codex client:  not spawned (REPL session empty)"),
        None => println!("    codex client:  N/A (one-shot, no REPL)"),
    }

    // RepoPrompt bound-window cache.
    match agent_tools::repoprompt_sync::peek_bound_window() {
        Some((dirs, wid)) => {
            let covers = dirs.iter().any(|d| d == workspace_cwd);
            let marker = if covers { "ok" } else { "MISMATCH" };
            println!(
                "    rp bind cache: window={wid} dirs={:?}  [{marker}]",
                dirs.iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
            );
        }
        None => println!("    rp bind cache: (empty — next rp call will bind fresh)"),
    }

    // Pending skill override (RF24-4).
    match agent_tools::repoprompt_sync::peek_pending_override() {
        Some(over) => println!(
            "    rp pending override: {:?} (consumed by next rp call, transient)",
            over.iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
        ),
        None => println!("    rp pending override: (none)"),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_delegate::CodexAppServerConfig;
    use std::path::PathBuf;

    // The health-check function calls `println!` so we don't capture output —
    // instead we verify it runs without panicking under every shape of input
    // (codex session live/dead, RP cache hit/miss, override present/absent).
    // The shape of the printed lines is exercised by the smoke test that
    // calls `seed doctor` end-to-end in RF25-4.

    static RP_SYNC_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn rp_sync_test_guard() -> std::sync::MutexGuard<'static, ()> {
        let g = RP_SYNC_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        agent_tools::repoprompt_sync::reset();
        g
    }

    #[test]
    fn cwd_health_check_runs_without_codex_session() {
        let _g = rp_sync_test_guard();
        cwd_health_check(&PathBuf::from("/tmp/seed-doctor-a"), None).unwrap();
    }

    #[test]
    fn cwd_health_check_runs_with_dead_codex_session() {
        let _g = rp_sync_test_guard();
        let cs = CodexSession::default();
        cwd_health_check(&PathBuf::from("/tmp/seed-doctor-b"), Some(&cs)).unwrap();
    }

    #[test]
    fn cwd_health_check_runs_with_live_codex_session_mismatched_cwd() {
        let _g = rp_sync_test_guard();
        let mut cs = CodexSession::default();
        let mut cfg = CodexAppServerConfig::default();
        cfg.cwd = Some(PathBuf::from("/tmp/seed-codex-cwd"));
        cs.ensure(cfg).unwrap();
        // workspace.cwd different from codex's cwd → MISMATCH path exercised.
        cwd_health_check(&PathBuf::from("/tmp/seed-workspace-different"), Some(&cs))
            .unwrap();
    }

    #[test]
    fn cwd_health_check_runs_with_rp_cache_hit_and_miss() {
        let _g = rp_sync_test_guard();
        let cwd = PathBuf::from("/tmp/seed-doctor-c");
        agent_tools::repoprompt_sync::record_bound_window(vec![cwd.clone()], 7);
        cwd_health_check(&cwd, None).unwrap(); // hit (ok marker)
        cwd_health_check(&PathBuf::from("/tmp/seed-doctor-d"), None).unwrap(); // miss (MISMATCH)
    }

    #[test]
    fn cwd_health_check_runs_with_pending_override() {
        let _g = rp_sync_test_guard();
        agent_tools::repoprompt_sync::set_pending_override(vec![PathBuf::from("/tmp/skill-bind")]);
        cwd_health_check(&PathBuf::from("/tmp/seed-doctor-e"), None).unwrap();
    }

    // --- RF28-1 run_mode_health_check ------------------------------------

    static MODE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn mode_test_guard() -> std::sync::MutexGuard<'static, ()> {
        MODE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    fn run_mode_health_check_one_shot_path() {
        let _g = mode_test_guard();
        // repl_pin = None matches `seed doctor` (no REPL).
        run_mode_health_check(None).unwrap();
    }

    #[test]
    fn run_mode_health_check_each_repl_pin() {
        let _g = mode_test_guard();
        for pin in [ModeArg::Auto, ModeArg::Read, ModeArg::Write] {
            run_mode_health_check(Some(pin)).unwrap();
        }
    }

    #[test]
    fn run_mode_health_check_reflects_guard_state() {
        let _g = mode_test_guard();
        // Exercise both guard states to ensure we don't panic when
        // formatting either label.
        agent_tools::run_mode_guard::set(agent_core::RunMode::ReadOnly);
        run_mode_health_check(None).unwrap();
        agent_tools::run_mode_guard::set(agent_core::RunMode::Implementation);
        run_mode_health_check(None).unwrap();
        agent_tools::run_mode_guard::reset();
    }
}

pub(crate) fn show_providers(
    provider_id: &str,
    model: Option<&str>,
    as_json: bool,
) -> Result<()> {
    let providers = agent_llm::built_in_providers();
    if as_json {
        println!("{}", serde_json::to_string_pretty(&providers)?);
        return Ok(());
    }

    println!("providers");
    println!("- codex local-app-server (default planner; uses local Codex login, no API key)");
    println!(
        "- repoprompt_oracle (opt-in: --provider repoprompt_oracle; planner goes through RepoPrompt ask_oracle so prompts inherit RepoPrompt's curated context; --model selects oracle mode: chat|plan|edit|review)"
    );
    for provider in &providers {
        let models = provider
            .models
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>();
        println!(
            "- {} {:?} {}",
            provider.id.as_str(),
            provider.response,
            if models.is_empty() {
                "(no built-in models)".to_string()
            } else {
                models.join(", ")
            }
        );
    }

    let Some(provider) = providers
        .iter()
        .find(|provider| provider.id.as_str() == provider_id)
    else {
        if provider_id == agent_llm::ProviderId::REPOPROMPT_ORACLE
            || provider_id == "repoprompt"
        {
            println!("route: repoprompt_oracle uses the local RepoPrompt CLI (no HTTP route);");
            println!("       seed run --provider repoprompt_oracle sends each planner turn via");
            println!("       `repoprompt_cli --call oracle_send` and keeps chat_id across turns.");
        } else if provider_id == agent_llm::ProviderId::CODEX {
            println!("route: codex uses the local app-server transport (no HTTP route).");
        } else {
            println!("route: provider {provider_id} not found");
        }
        return Ok(());
    };
    let model = model
        .map(ModelId::from)
        .or_else(|| provider.models.first().map(|model| model.id.clone()))
        .unwrap_or_else(|| ModelId::from("gpt-5.1"));
    let route = ProviderRouter.route(provider, &model);
    let transformed = agent_llm::default_pipeline()
        .transform(provider, agent_llm::ChatRequest::user(model, ""));

    println!("route");
    println!("- provider: {}", provider.id.as_str());
    println!("- backend: {:?}", route.response);
    println!("- endpoint: {}", route.endpoint);
    println!(
        "- transforms: {}",
        agent_llm::default_pipeline().names().join(", ")
    );
    println!(
        "- options: {}",
        serde_json::to_string(&transformed.options)?
    );
    Ok(())
}
