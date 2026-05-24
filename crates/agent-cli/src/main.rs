use agent_core::ToolCall;
use agent_delegate::ApprovalMode;
use agent_session::SessionStore;
use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::json;
use std::env;
use std::path::{Path, PathBuf};

mod commands;
mod display;
mod doctor;
mod plan_repoprompt;
use commands::codex::{DelegateCommand, run_codex_delegate};
use commands::interactive::{InteractiveArgs, default_interactive_command, run_interactive};
use commands::llm::{LlmCommand, run_llm_ask};
use commands::plan::{PlanCommand, run_plan};
use commands::replay::replay;
use commands::rp::{RpCommand, RpRunArgs, run_rp};
use commands::run::{RunGoalArgs, run_goal};
use commands::skill::SkillCommand;
use commands::tool::{ToolCommand, run_tool};

// Read-only analysis usually finishes in 4-6 turns thanks to the streak guard
// and skill-injected recipes. Implementation goals (build a plan, verify,
// implement) regularly need 12-20. 24 leaves headroom for both modes without
// being so large that runaway loops burn the user's time silently.
pub(crate) const DEFAULT_MAX_TURNS: usize = 24;

#[derive(Debug, Parser)]
#[command(name = "seed")]
#[command(about = "SeedAgent: a minimal Rust seed for a self-bootstrapping agent")]
pub(crate) struct Cli {
    #[arg(long, default_value = "sessions")]
    pub(crate) sessions_dir: PathBuf,
    #[arg(long, default_value = "skills")]
    pub(crate) skills_dir: PathBuf,
    #[command(subcommand)]
    command: Option<Command>,
}

/// Subcommands for `seed codex-daemon`. Thin wrappers around
/// `codex app-server daemon ...` so users don't have to remember the
/// full codex CLI path. RF33-4.
#[derive(Debug, Subcommand)]
pub(crate) enum CodexDaemonAction {
    /// Start the daemon (no-op if already running).
    Start,
    /// Stop the running daemon.
    Stop,
    /// Print local CLI + running daemon versions as JSON.
    Status,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Start an interactive prompt.
    Chat {
        #[arg(long)]
        cwd: Option<PathBuf>,
        #[arg(long, default_value_t = DEFAULT_MAX_TURNS)]
        max_turns: usize,
        #[arg(long, help = "Crystallize a successful run into a local SKILL.md")]
        learn: bool,
        #[arg(long, default_value = "codex")]
        provider: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, value_enum, default_value_t = ApprovalArg::Deny)]
        approval: ApprovalArg,
        #[arg(long)]
        effort: Option<String>,
        #[arg(long, default_value_t = 600)]
        turn_timeout_secs: u64,
        #[arg(
            long,
            value_enum,
            help = "MCP policy for Codex; omitted means only RepoPrompt is allowed"
        )]
        mcp: Option<McpArg>,
        #[arg(long = "mcp-allow", help = "Allow one MCP server by name; repeatable")]
        mcp_allow: Vec<String>,
        #[arg(long, help = "Enable Codex plugins while starting app-server")]
        plugins: bool,
        #[arg(long, help = "Delegate each prompt directly to Codex app-server")]
        codex: bool,
        #[arg(long, help = "Record goals locally without invoking an LLM")]
        record_only: bool,
        #[arg(
            long,
            value_enum,
            default_value_t = commands::run::ModeArg::Auto,
            help = "Read-only/write mode: auto (classify), read (force read-only), write (force implementation)"
        )]
        mode: commands::run::ModeArg,
        #[arg(
            long = "use-daemon",
            help = "Connect Codex via `app-server proxy` (running daemon) instead of spawning fresh stdio app-server"
        )]
        use_daemon: bool,
    },
    Doctor,
    Run {
        goal: String,
        #[arg(long)]
        cwd: Option<PathBuf>,
        #[arg(long)]
        llm: bool,
        #[arg(long, default_value_t = DEFAULT_MAX_TURNS)]
        max_turns: usize,
        #[arg(long, help = "Crystallize a successful run into a local SKILL.md")]
        learn: bool,
        #[arg(long)]
        codex: bool,
        #[arg(long, default_value = "codex")]
        provider: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, value_enum, default_value_t = ApprovalArg::Deny)]
        approval: ApprovalArg,
        #[arg(long)]
        effort: Option<String>,
        #[arg(long, default_value_t = 600)]
        turn_timeout_secs: u64,
        #[arg(
            long,
            value_enum,
            help = "MCP policy for Codex; omitted means only RepoPrompt is allowed"
        )]
        mcp: Option<McpArg>,
        #[arg(long = "mcp-allow", help = "Allow one MCP server by name; repeatable")]
        mcp_allow: Vec<String>,
        #[arg(long, help = "Enable Codex plugins while starting app-server")]
        plugins: bool,
        #[arg(
            long,
            value_enum,
            default_value_t = commands::run::ModeArg::Auto,
            help = "Read-only/write mode: auto (classify), read (force read-only), write (force implementation)"
        )]
        mode: commands::run::ModeArg,
        #[arg(
            long = "use-daemon",
            help = "Connect Codex via `app-server proxy` (running daemon) instead of spawning fresh stdio app-server"
        )]
        use_daemon: bool,
    },
    Tool {
        #[command(subcommand)]
        command: ToolCommand,
        #[arg(long)]
        cwd: Option<PathBuf>,
    },
    Plan {
        #[command(subcommand)]
        command: PlanCommand,
        #[arg(long)]
        cwd: Option<PathBuf>,
    },
    Reflect {
        session: Option<String>,
    },
    Replay {
        session: Option<String>,
    },
    Skill {
        #[command(subcommand)]
        command: SkillCommand,
    },
    Providers {
        #[arg(long, default_value = "openai")]
        provider: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        json: bool,
    },
    Llm {
        #[command(subcommand)]
        command: LlmCommand,
    },
    Codex {
        prompt: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        cwd: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = ApprovalArg::Deny)]
        approval: ApprovalArg,
        #[arg(long)]
        effort: Option<String>,
        #[arg(long, default_value_t = 600)]
        turn_timeout_secs: u64,
        #[arg(
            long,
            value_enum,
            help = "MCP policy for Codex; omitted means only RepoPrompt is allowed"
        )]
        mcp: Option<McpArg>,
        #[arg(long = "mcp-allow", help = "Allow one MCP server by name; repeatable")]
        mcp_allow: Vec<String>,
        #[arg(long, help = "Enable Codex plugins while starting app-server")]
        plugins: bool,
    },
    /// List the codex models cached at `$CODEX_HOME/models_cache.json`.
    /// Marks the current default (from `config.toml#model`) with ★. Use the
    /// listed slug with `--model <slug>` on `seed run` / `seed codex`.
    #[command(name = "codex-models", alias = "codex_models")]
    CodexModels {
        #[arg(long, help = "Output the raw JSON instead of a table")]
        json: bool,
        #[arg(
            long,
            help = "Also show models with visibility=hide (e.g. codex-auto-review)"
        )]
        show_hidden: bool,
    },
    /// RF33-4: thin wrapper over `codex app-server daemon …` so users can
    /// start/stop/status the codex daemon from `seed` directly. Pair with
    /// `--use-daemon` on `seed run` / `chat` to actually consume it.
    #[command(name = "codex-daemon", alias = "codex_daemon")]
    CodexDaemon {
        #[command(subcommand)]
        action: CodexDaemonAction,
    },
    Delegate {
        #[command(subcommand)]
        command: DelegateCommand,
    },
    Rp {
        #[command(subcommand)]
        command: RpCommand,
        #[arg(long)]
        cli_path: Option<PathBuf>,
        #[arg(long)]
        window_id: Option<u32>,
        #[arg(long)]
        tab: Option<String>,
        #[arg(long)]
        context_id: Option<String>,
        #[arg(long = "working-dir")]
        working_dirs: Vec<PathBuf>,
        #[arg(long, default_value_t = 300)]
        timeout_secs: u64,
        #[arg(long)]
        raw_json: bool,
    },
}


#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum ApprovalArg {
    Deny,
    AcceptOnce,
    AcceptForSession,
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum McpArg {
    None,
    All,
}

impl From<ApprovalArg> for ApprovalMode {
    fn from(value: ApprovalArg) -> Self {
        match value {
            ApprovalArg::Deny => ApprovalMode::Deny,
            ApprovalArg::AcceptOnce => ApprovalMode::AcceptOnce,
            ApprovalArg::AcceptForSession => ApprovalMode::AcceptForSession,
        }
    }
}

fn main() -> Result<()> {
    let mut cli = Cli::parse();
    let store = SessionStore::new(&cli.sessions_dir)?;
    let command = cli
        .command
        .take()
        .unwrap_or_else(default_interactive_command);

    match command {
        Command::Chat {
            cwd,
            max_turns,
            learn,
            provider,
            model,
            approval,
            effort,
            turn_timeout_secs,
            mcp,
            mcp_allow,
            plugins,
            codex,
            record_only,
            mode,
            use_daemon,
        } => run_interactive(
            &cli,
            &store,
            InteractiveArgs {
                cwd,
                max_turns,
                learn,
                provider,
                model,
                approval,
                effort,
                turn_timeout_secs,
                mcp,
                mcp_allow,
                plugins,
                codex,
                record_only,
                mode,
                use_daemon,
            },
        ),
        Command::Doctor => doctor::doctor(&cli.skills_dir, &store),
        Command::Run {
            goal,
            cwd,
            llm,
            max_turns,
            learn,
            codex,
            provider,
            model,
            approval,
            effort,
            turn_timeout_secs,
            mcp,
            mcp_allow,
            plugins,
            mode,
            use_daemon,
        } => run_goal(RunGoalArgs {
            store: &store,
            goal,
            cwd,
            use_llm: llm,
            use_codex: codex,
            learn,
            skills_dir: cli.skills_dir.clone(),
            policy: commands::run::RunPolicy {
                max_turns,
                turn_timeout_secs,
                ..Default::default()
            },
            provider: commands::run::ProviderSpec {
                kind: commands::run::PlannerProvider::from_id(&provider),
                model,
                approval,
                effort,
                mcp,
                mcp_allow,
                plugins,
            },
            mode,
            use_daemon,
            // One-shot `seed run` — no REPL session to inherit, fall back
            // to the local throwaway session built inside run_goal.
            codex_session: None,
        }),
        Command::Tool { command, cwd } => run_tool(&store, command, cwd, cli.skills_dir.clone()),
        Command::Plan { command, cwd } => run_plan(command, cwd),
        Command::Reflect { session } => {
            let records = store.read(session.as_deref())?;
            println!("{}", agent_skills::reflect_markdown(&records));
            Ok(())
        }
        Command::Replay { session } => replay(&store, session.as_deref()),
        Command::Skill { command } => commands::skill::run_skill(command, &cli.skills_dir, &store),
        Command::Providers {
            provider,
            model,
            json,
        } => doctor::show_providers(&provider, model.as_deref(), json),
        Command::Llm { command } => match command {
            LlmCommand::Ask {
                prompt,
                provider,
                model,
                system,
                effort,
                max_output_tokens,
                raw,
            } => run_llm_ask(
                prompt,
                provider,
                model,
                system,
                effort,
                max_output_tokens,
                raw,
            ),
        },
        Command::Codex {
            prompt,
            model,
            cwd,
            approval,
            effort,
            turn_timeout_secs,
            mcp,
            mcp_allow,
            plugins,
        } => run_codex_delegate(
            prompt,
            cli.skills_dir.clone(),
            model,
            cwd,
            "workspace-write".to_string(),
            "on-request".to_string(),
            approval,
            effort,
            turn_timeout_secs,
            mcp,
            mcp_allow,
            plugins,
        ),
        Command::CodexModels { json, show_hidden } => {
            commands::codex::run_codex_models(json, show_hidden)
        }
        Command::CodexDaemon { action } => commands::codex::run_codex_daemon(action),
        Command::Delegate { command } => match command {
            DelegateCommand::Codex {
                prompt,
                model,
                cwd,
                sandbox,
                approval_policy,
                approval,
                effort,
                turn_timeout_secs,
                mcp,
                mcp_allow,
                plugins,
            } => run_codex_delegate(
                prompt,
                cli.skills_dir.clone(),
                model,
                cwd,
                sandbox,
                approval_policy,
                approval,
                effort,
                turn_timeout_secs,
                mcp,
                mcp_allow,
                plugins,
            ),
        },
        Command::Rp {
            command,
            cli_path,
            window_id,
            tab,
            context_id,
            working_dirs,
            timeout_secs,
            raw_json,
        } => run_rp(RpRunArgs {
            command,
            cli_path,
            window_id,
            tab,
            context_id,
            working_dirs,
            timeout_secs,
            raw_json,
        }),
    }
}




pub(crate) fn absolutize_cli(cwd: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        absolute_base_cli(cwd).join(path)
    }
}

fn absolute_base_cli(cwd: &Path) -> PathBuf {
    if cwd.is_absolute() {
        cwd.to_path_buf()
    } else {
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(cwd)
    }
}


fn consolidate_run_skill(
    skills_dir: &Path,
    goal: &str,
    session_path: &Path,
) -> Result<agent_skills::SkillConsolidation> {
    let records = agent_session::read_records(session_path)?;
    let name = unique_skill_name(skills_dir, goal);
    let binding = agent_skills::query_current_repoprompt_binding();
    Ok(agent_skills::consolidate_skill_with_binding(
        skills_dir,
        &name,
        &records,
        binding.as_ref(),
    )?)
}



pub(crate) fn memory_paths(
    cwd: &Path,
    skills_dir: &Path,
    sessions_dir: &Path,
) -> agent_memory::MemoryPaths {
    agent_memory::MemoryPaths::new(cwd.join("memory"), skills_dir.to_path_buf(), sessions_dir)
}

fn unique_skill_name(skills_dir: &Path, goal: &str) -> String {
    let mut base = agent_skills::slugify(goal);
    if base.is_empty() {
        base = "learned-skill".to_string();
    }
    if base.len() > 64 {
        base.truncate(64);
        while base.ends_with('-') {
            base.pop();
        }
    }

    let mut candidate = base.clone();
    let mut suffix = 2usize;
    while skills_dir.join(&candidate).join("SKILL.md").exists() {
        candidate = format!("{base}-{suffix}");
        suffix += 1;
    }
    candidate
}






pub(crate) fn parse_seed_goal(goal: &str) -> Option<ToolCall> {
    let (prefix, rest) = goal.split_once(':')?;
    let value = rest.trim();
    match prefix.trim().to_ascii_lowercase().as_str() {
        "shell" => Some(ToolCall::new("run_shell", json!({ "command": value }))),
        "read" => Some(ToolCall::new("read_file", json!({ "path": value }))),
        "checkpoint" => Some(ToolCall::new(
            "update_working_checkpoint",
            json!({ "key_info": value }),
        )),
        "memory" => Some(ToolCall::new(
            "start_long_term_update",
            json!({ "reason": value }),
        )),
        _ => None,
    }
}


