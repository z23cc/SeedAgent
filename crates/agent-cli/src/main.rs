use agent_core::{AgentEvent, ToolCall, ToolContext, ToolResult};
use agent_delegate::{ApprovalMode, CodexAppServerClient, CodexAppServerConfig, McpPolicy};
use agent_llm::{ChatRequest, ModelId, ProviderClient, ProviderRouter};
use agent_session::{SessionStore, SessionWriter};
use anyhow::{Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::json;
use std::env;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(name = "seed")]
#[command(about = "SeedAgent: a minimal Rust seed for a self-bootstrapping agent")]
struct Cli {
    #[arg(long, default_value = "sessions")]
    sessions_dir: PathBuf,
    #[arg(long, default_value = "skills")]
    skills_dir: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Doctor,
    Run {
        goal: String,
        #[arg(long)]
        cwd: Option<PathBuf>,
        #[arg(long)]
        llm: bool,
        #[arg(long, default_value_t = 8)]
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

#[derive(Debug, Subcommand)]
enum ToolCommand {
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

#[derive(Debug, Subcommand)]
enum PlanCommand {
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
enum WriteModeArg {
    Overwrite,
    Append,
    Prepend,
}

#[derive(Debug, Clone, ValueEnum)]
enum SettlementDecisionArg {
    UpdateL2GlobalFacts,
    UpdateL3Skill,
    Skip,
}

#[derive(Debug, Subcommand)]
enum SkillCommand {
    Create {
        #[arg(long)]
        name: String,
        session: Option<String>,
    },
    List {
        #[arg(long)]
        json: bool,
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    Search {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    Fetch {
        name: String,
    },
}

#[derive(Debug, Subcommand)]
enum LlmCommand {
    Ask {
        prompt: String,
        #[arg(long, default_value = "openai")]
        provider: String,
        #[arg(long, default_value = "gpt-5.1")]
        model: String,
        #[arg(long)]
        system: Option<String>,
        #[arg(long)]
        effort: Option<String>,
        #[arg(long)]
        max_output_tokens: Option<u32>,
        #[arg(long)]
        raw: bool,
    },
}

#[derive(Debug, Subcommand)]
enum DelegateCommand {
    Codex {
        prompt: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        cwd: Option<PathBuf>,
        #[arg(long, default_value = "workspace-write")]
        sandbox: String,
        #[arg(long, default_value = "on-request")]
        approval_policy: String,
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
}

#[derive(Debug, Subcommand)]
enum RpCommand {
    Status,
    Tools {
        #[arg(long)]
        json: bool,
    },
    Exec {
        command: String,
    },
    Call {
        tool: String,
        #[arg(long, default_value = "{}")]
        args: String,
    },
    Describe {
        tool: String,
    },
    Windows,
    Workspaces {
        #[arg(long)]
        include_hidden: bool,
    },
    Bind {
        #[arg(long = "working-dir")]
        working_dirs: Vec<PathBuf>,
        #[arg(long)]
        create_if_missing: bool,
        #[arg(long)]
        tab_name: Option<String>,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum ApprovalArg {
    Deny,
    AcceptOnce,
    AcceptForSession,
}

#[derive(Debug, Clone, ValueEnum)]
enum McpArg {
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
    let cli = Cli::parse();
    let store = SessionStore::new(&cli.sessions_dir)?;

    match cli.command {
        Command::Doctor => doctor(&cli, &store),
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
        } => run_goal(RunGoalArgs {
            store: &store,
            goal,
            cwd,
            use_llm: llm,
            max_turns,
            learn,
            use_codex: codex,
            provider,
            model,
            approval,
            effort,
            turn_timeout_secs,
            mcp,
            mcp_allow,
            plugins,
            skills_dir: cli.skills_dir.clone(),
        }),
        Command::Tool { command, cwd } => run_tool(&store, command, cwd, cli.skills_dir.clone()),
        Command::Plan { command, cwd } => run_plan(command, cwd),
        Command::Reflect { session } => {
            let records = store.read(session.as_deref())?;
            println!("{}", agent_skills::reflect_markdown(&records));
            Ok(())
        }
        Command::Replay { session } => replay(&store, session.as_deref()),
        Command::Skill { command } => match command {
            SkillCommand::Create { name, session } => {
                let records = store.read(session.as_deref())?;
                let path = agent_skills::create_skill(&cli.skills_dir, &name, &records)?;
                println!("created skill: {}", path.display());
                Ok(())
            }
            SkillCommand::List { json, limit } => {
                let skills = agent_skills::list_skill_infos(&cli.skills_dir)?;
                let skills = skills.into_iter().take(limit).collect::<Vec<_>>();
                if json {
                    println!("{}", serde_json::to_string_pretty(&skills)?);
                } else {
                    print_skill_infos(&skills);
                }
                Ok(())
            }
            SkillCommand::Search { query, limit, json } => {
                let skills = agent_skills::search_skill_infos(&cli.skills_dir, &query, limit)?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&skills)?);
                } else {
                    print_skill_infos(&skills);
                }
                Ok(())
            }
            SkillCommand::Fetch { name } => {
                let skill = agent_skills::fetch_skill(&cli.skills_dir, &name)?;
                println!("{}", skill.body);
                Ok(())
            }
        },
        Command::Providers {
            provider,
            model,
            json,
        } => show_providers(&provider, model.as_deref(), json),
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

fn doctor(cli: &Cli, store: &SessionStore) -> Result<()> {
    let registry = agent_tools::seed_registry();
    println!("seed doctor");
    println!("- cwd: {}", env::current_dir()?.display());
    println!("- sessions: {}", store.root().display());
    println!("- skills: {}", cli.skills_dir.display());
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
    println!("- delegates: codex-app-server");
    Ok(())
}

fn show_providers(provider_id: &str, model: Option<&str>, as_json: bool) -> Result<()> {
    let providers = agent_llm::built_in_providers();
    if as_json {
        println!("{}", serde_json::to_string_pretty(&providers)?);
        return Ok(());
    }

    println!("providers");
    println!("- codex local-app-server (default planner; uses local Codex login, no API key)");
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
        println!("route: provider {provider_id} not found");
        return Ok(());
    };
    let model = model
        .map(ModelId::from)
        .or_else(|| provider.models.first().map(|model| model.id.clone()))
        .unwrap_or_else(|| ModelId::from("gpt-5.1"));
    let route = ProviderRouter.route(provider, &model);
    let transformed =
        agent_llm::default_pipeline().transform(provider, agent_llm::ChatRequest::user(model, ""));

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

fn print_skill_infos(skills: &[agent_skills::SkillInfo]) {
    if skills.is_empty() {
        println!("skills: none");
        return;
    }
    println!("skills");
    for skill in skills {
        println!(
            "- {} [{}] {}",
            skill.name,
            skill.tags.join(", "),
            skill.description
        );
        println!("  path: {}", skill.path.display());
    }
}

fn run_plan(command: PlanCommand, cwd: Option<PathBuf>) -> Result<()> {
    let cwd = cwd.unwrap_or(env::current_dir()?);
    let store = agent_plan::PlanStore::new(cwd.join("plans"));
    match command {
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
            let mut cfg = agent_repoprompt::RepoPromptClientConfig {
                timeout_secs: timeout_secs + 60,
                window_id,
                context_id,
                working_dirs,
                raw_json: true,
                ..Default::default()
            };
            cfg.working_dirs = cfg
                .working_dirs
                .into_iter()
                .map(|path| absolutize_cli(&cwd, path))
                .collect();
            let message = format!(
                "Independent verification gate for SeedAgent plan `{}`. Read `{}` first. Return `VERDICT: PASS` or `VERDICT: FAIL` with concise evidence. Do not edit files.",
                context.plan_id,
                context
                    .plan_file
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join("verify_context.json")
                    .display()
            );
            let output = agent_repoprompt::RepoPromptClient::new(cfg).call_tool(
                agent_repoprompt::RepoPromptTool::AgentRun,
                &json!({
                    "op": "start",
                    "model_id": model_id,
                    "message": message,
                    "timeout": timeout_secs,
                }),
            )?;
            let report = repoprompt_report_text_cli(&output);
            let snapshot = store.record_verification(Some(&context.plan_id), &report)?;
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
    if let Some(item) = &snapshot.next_item {
        println!("next: #{} {:?} {}", item.index, item.kind, item.text);
    } else {
        println!("next: none");
    }
    Ok(())
}

fn absolutize_cli(cwd: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    }
}

fn repoprompt_report_text_cli(output: &agent_repoprompt::RepoPromptOutput) -> String {
    if !output.stdout.trim().is_empty() {
        output.stdout.clone()
    } else if let Some(json) = &output.json {
        serde_json::to_string_pretty(json).unwrap_or_else(|_| json.to_string())
    } else {
        output.stderr.clone()
    }
}

struct RpRunArgs {
    command: RpCommand,
    cli_path: Option<PathBuf>,
    window_id: Option<u32>,
    tab: Option<String>,
    context_id: Option<String>,
    working_dirs: Vec<PathBuf>,
    timeout_secs: u64,
    raw_json: bool,
}

fn run_rp(args: RpRunArgs) -> Result<()> {
    let mut cfg = agent_repoprompt::RepoPromptClientConfig {
        timeout_secs: args.timeout_secs,
        window_id: args.window_id,
        tab: args.tab,
        context_id: args.context_id,
        working_dirs: args.working_dirs,
        raw_json: args.raw_json,
        ..Default::default()
    };
    if let Some(cli_path) = args.cli_path {
        cfg.cli_path = cli_path;
    }

    match args.command {
        RpCommand::Status => print_rp_status(&cfg),
        RpCommand::Tools { json } => print_rp_tools(json),
        RpCommand::Exec { command } => {
            let output = agent_repoprompt::RepoPromptClient::new(cfg).exec(&command)?;
            print_rp_output(output)
        }
        RpCommand::Call { tool, args } => {
            cfg.raw_json = true;
            let tool = tool.parse::<agent_repoprompt::RepoPromptTool>()?;
            let value = agent_repoprompt::parse_args_json(&args)?;
            let output = agent_repoprompt::RepoPromptClient::new(cfg).call_tool(tool, &value)?;
            print_rp_output(output)
        }
        RpCommand::Describe { tool } => {
            let tool = tool.parse::<agent_repoprompt::RepoPromptTool>()?;
            let output = agent_repoprompt::RepoPromptClient::new(cfg).describe_tool(tool)?;
            print_rp_output(output)
        }
        RpCommand::Windows => {
            let output = agent_repoprompt::RepoPromptClient::new(cfg).exec("windows")?;
            print_rp_output(output)
        }
        RpCommand::Workspaces { include_hidden } => {
            let command = if include_hidden {
                "workspace list --include-hidden"
            } else {
                "workspace list"
            };
            let output = agent_repoprompt::RepoPromptClient::new(cfg).exec(command)?;
            print_rp_output(output)
        }
        RpCommand::Bind {
            working_dirs,
            create_if_missing,
            tab_name,
        } => {
            let bind_dirs = if working_dirs.is_empty() {
                cfg.working_dirs.clone()
            } else {
                working_dirs
            };
            if bind_dirs.is_empty() {
                bail!("rp bind requires --working-dir either before or after the subcommand");
            }
            cfg.raw_json = true;
            let mut value = json!({
                "op": "bind",
                "working_dirs": bind_dirs
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>(),
                "create_if_missing": create_if_missing,
            });
            if let Some(tab_name) = tab_name {
                value["tab_name"] = json!(tab_name);
            }
            let output = agent_repoprompt::RepoPromptClient::new(cfg)
                .call_tool(agent_repoprompt::RepoPromptTool::BindContext, &value)?;
            print_rp_output(output)
        }
    }
}

fn print_rp_status(cfg: &agent_repoprompt::RepoPromptClientConfig) -> Result<()> {
    let client = agent_repoprompt::RepoPromptClient::new(cfg.clone());
    println!("RepoPrompt backend");
    println!("- cli: {}", cfg.cli_path.display());
    println!(
        "- available: {}",
        if client.check_available().is_ok() {
            "yes"
        } else {
            "no"
        }
    );
    println!("- wrapped tools: {}", agent_repoprompt::known_tools().len());
    println!(
        "- routing: window={:?} tab={:?} context_id={:?} working_dirs={}",
        cfg.window_id,
        cfg.tab,
        cfg.context_id,
        cfg.working_dirs
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(())
}

fn print_rp_tools(as_json: bool) -> Result<()> {
    let tools = agent_repoprompt::known_tools();
    if as_json {
        println!("{}", serde_json::to_string_pretty(&tools)?);
        return Ok(());
    }
    println!("RepoPrompt tools");
    for tool in tools {
        println!("- {} [{}] {}", tool.name, tool.group, tool.description);
    }
    Ok(())
}

fn print_rp_output(output: agent_repoprompt::RepoPromptOutput) -> Result<()> {
    if !output.stdout.trim().is_empty() {
        println!("{}", output.stdout.trim_end());
    }
    if !output.stderr.trim().is_empty() {
        eprintln!("{}", output.stderr.trim_end());
    }
    if output.timed_out {
        bail!("RepoPrompt CLI timed out");
    }
    if output.exit_code != Some(0) {
        bail!(
            "RepoPrompt CLI failed with exit code {:?}",
            output.exit_code
        );
    }
    Ok(())
}

fn consolidate_run_skill(
    skills_dir: &Path,
    goal: &str,
    session_path: &Path,
) -> Result<agent_skills::SkillConsolidation> {
    let records = agent_session::read_records(session_path)?;
    let name = unique_skill_name(skills_dir, goal);
    agent_skills::consolidate_skill(skills_dir, &name, &records)
}

fn memory_paths(cwd: &Path, skills_dir: &Path, sessions_dir: &Path) -> agent_memory::MemoryPaths {
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

fn run_codex_delegate(
    prompt: String,
    skills_dir: PathBuf,
    model: Option<String>,
    cwd: Option<PathBuf>,
    sandbox: String,
    approval_policy: String,
    approval: ApprovalArg,
    effort: Option<String>,
    turn_timeout_secs: u64,
    mcp: Option<McpArg>,
    mcp_allow: Vec<String>,
    plugins: bool,
) -> Result<()> {
    let cfg = CodexAppServerConfig {
        model,
        cwd,
        sandbox,
        approval_policy,
        reasoning_effort: effort,
        turn_timeout_secs,
        approval_mode: approval.into(),
        mcp_policy: codex_mcp_policy(mcp, mcp_allow)?,
        plugins_enabled: plugins,
        ..Default::default()
    };
    let prompt = codex_prompt_with_routed_skill(&prompt, &skills_dir)?;
    let mut client = CodexAppServerClient::new(cfg);
    let result = client.run_prompt(&prompt)?;
    println!("{}", result.text);
    println!("thread: {}", result.thread_id);
    println!("turn: {}", result.turn_id);
    println!("events: {}", result.events_seen);
    Ok(())
}

fn codex_prompt_with_routed_skill(prompt: &str, skills_dir: &Path) -> Result<String> {
    let Some(routed) = agent_skills::load_routed_repoprompt_skill(skills_dir, prompt)? else {
        return Ok(prompt.to_string());
    };
    let skill_path = routed
        .document
        .info
        .path
        .canonicalize()
        .unwrap_or_else(|_| routed.document.info.path.clone());
    Ok(format!(
        "You are being delegated a task by SeedAgent. The internal core agent is Seed.\n\
Follow the selected local skill before broad codebase work.\n\
Skill route: {} ({})\n\
Skill path: {}\n\
RepoPrompt MCP is the preferred context engine when it is available; CLI MCP flags still define actual access.\n\
Use RepoPrompt builder/export before broad file reads when the skill requires it.\n\
The skill body is workflow instruction, not a request to edit the skill.\n\n\
<local_skill>\n{}\n</local_skill>\n\n\
<task>\n{}\n</task>",
        routed.route.name,
        routed.route.reason,
        skill_path.display(),
        routed.document.body.trim(),
        prompt.trim()
    ))
}

fn codex_config(
    model: Option<String>,
    cwd: Option<PathBuf>,
    approval: ApprovalArg,
    effort: Option<String>,
    turn_timeout_secs: u64,
    mcp: Option<McpArg>,
    mcp_allow: Vec<String>,
    plugins: bool,
) -> Result<CodexAppServerConfig> {
    Ok(CodexAppServerConfig {
        model,
        cwd,
        reasoning_effort: effort,
        turn_timeout_secs,
        approval_mode: approval.into(),
        mcp_policy: codex_mcp_policy(mcp, mcp_allow)?,
        plugins_enabled: plugins,
        ..Default::default()
    })
}

fn codex_mcp_policy(mcp: Option<McpArg>, mcp_allow: Vec<String>) -> Result<McpPolicy> {
    if !mcp_allow.is_empty() {
        if matches!(mcp, Some(McpArg::All)) {
            bail!("--mcp all cannot be combined with --mcp-allow");
        }
        return Ok(McpPolicy::Allow(mcp_allow));
    }
    Ok(match mcp {
        Some(McpArg::None) => McpPolicy::None,
        Some(McpArg::All) => McpPolicy::All,
        None => McpPolicy::default(),
    })
}

fn run_llm_ask(
    prompt: String,
    provider_id: String,
    model: String,
    system: Option<String>,
    effort: Option<String>,
    max_output_tokens: Option<u32>,
    raw: bool,
) -> Result<()> {
    let provider = agent_llm::find_provider(&provider_id)
        .ok_or_else(|| anyhow::anyhow!("provider not found: {provider_id}"))?;
    let mut request = ChatRequest::user(model, prompt);
    if let Some(system) = system {
        request
            .messages
            .insert(0, agent_llm::ChatMessage::system(system));
    }
    request.reasoning_effort = effort;
    request.max_output_tokens = max_output_tokens;

    let response = ProviderClient::new().chat(provider, request)?;
    if raw {
        println!("{}", serde_json::to_string_pretty(&response.raw)?);
    } else {
        println!("{}", response.text);
    }
    Ok(())
}

struct RunGoalArgs<'a> {
    store: &'a SessionStore,
    goal: String,
    cwd: Option<PathBuf>,
    use_llm: bool,
    max_turns: usize,
    learn: bool,
    use_codex: bool,
    provider: String,
    model: Option<String>,
    approval: ApprovalArg,
    effort: Option<String>,
    turn_timeout_secs: u64,
    mcp: Option<McpArg>,
    mcp_allow: Vec<String>,
    plugins: bool,
    skills_dir: PathBuf,
}

fn run_goal(args: RunGoalArgs<'_>) -> Result<()> {
    let RunGoalArgs {
        store,
        goal,
        cwd,
        use_llm,
        max_turns,
        learn,
        use_codex,
        provider,
        model,
        approval,
        effort,
        turn_timeout_secs,
        mcp,
        mcp_allow,
        plugins,
        skills_dir,
    } = args;
    let cwd = cwd.unwrap_or(env::current_dir()?);
    let memory_paths = memory_paths(&cwd, &skills_dir, store.root());
    agent_memory::rebuild_index(&memory_paths)?;
    let planner_memory = agent_runtime::PlannerMemoryContext::new(
        agent_memory::planner_memory_context(&memory_paths)?,
    );
    let mut session = store.start()?;
    session.append(AgentEvent::RunStarted {
        goal: goal.clone(),
        cwd: cwd.clone(),
    })?;

    if use_codex {
        let cfg = codex_config(
            model,
            Some(cwd.clone()),
            approval,
            effort,
            turn_timeout_secs,
            mcp,
            mcp_allow,
            plugins,
        )?;
        let mut client = CodexAppServerClient::new(cfg);
        let codex_goal = codex_prompt_with_routed_skill(&goal, &skills_dir)?;
        match client.run_prompt(&codex_goal) {
            Ok(result) => {
                session.append(AgentEvent::Reflection {
                    summary: result.text.clone(),
                })?;
                session.append(AgentEvent::RunFinished {
                    status: "completed".to_string(),
                    summary: format!(
                        "Codex completed turn {} after {} events.",
                        result.turn_id, result.events_seen
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
        let tool_infos = registry.infos();
        let loop_result = if provider == "codex" {
            let cfg = codex_config(
                model,
                Some(cwd.clone()),
                approval,
                effort,
                turn_timeout_secs,
                mcp,
                mcp_allow,
                plugins,
            )?;
            let mut codex = CodexAppServerClient::new(cfg);
            match agent_runtime::run_agent_loop_with_state_planner(
                max_turns,
                |state| {
                    let prompt = agent_runtime::planner_prompt_with_state_and_memory(
                        &goal,
                        &tool_infos,
                        state,
                        &planner_memory,
                    );
                    let result = codex.run_prompt(&prompt).map_err(|err| {
                        agent_runtime::RuntimeError::Planner(format!("Codex planner failed: {err}"))
                    })?;
                    agent_runtime::parse_planned_action(&result.text)
                },
                |call| match execute_call(
                    &mut session,
                    &cwd,
                    &skills_dir,
                    store.root(),
                    call.clone(),
                ) {
                    Ok(result) => result,
                    Err(err) => ToolResult::error(call, err.to_string()),
                },
            ) {
                Ok(result) => result,
                Err(err) => {
                    session.append(AgentEvent::RunFinished {
                        status: "failed".to_string(),
                        summary: format!("Planner failed: {err}"),
                    })?;
                    return Err(err.into());
                }
            }
        } else {
            let model = model.unwrap_or_else(|| "gpt-5.1".to_string());
            match agent_runtime::run_agent_loop_with_state_planner(
                max_turns,
                |state| {
                    agent_runtime::plan_next_action_with_state_and_memory(
                        &provider,
                        ModelId::from(model.clone()),
                        &goal,
                        &tool_infos,
                        state,
                        &planner_memory,
                    )
                },
                |call| match execute_call(
                    &mut session,
                    &cwd,
                    &skills_dir,
                    store.root(),
                    call.clone(),
                ) {
                    Ok(result) => result,
                    Err(err) => ToolResult::error(call, err.to_string()),
                },
            ) {
                Ok(result) => result,
                Err(err) => {
                    session.append(AgentEvent::RunFinished {
                        status: "failed".to_string(),
                        summary: format!("Planner failed: {err}"),
                    })?;
                    return Err(err.into());
                }
            }
        };
        for turn_summary in &loop_result.turn_summaries {
            session.append(AgentEvent::TurnSummary {
                turn: turn_summary.turn,
                summary: turn_summary.summary.clone(),
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
                if learn {
                    let consolidation = consolidate_run_skill(&skills_dir, &goal, session.path())?;
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

    println!("session: {}", session.path().display());
    Ok(())
}

fn run_tool(
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

fn replay(store: &SessionStore, session: Option<&str>) -> Result<()> {
    let records = store.read(session)?;
    for record in records {
        println!(
            "{} {}",
            record.ts.format("%Y-%m-%d %H:%M:%S"),
            serde_json::to_string(&record.event)?
        );
    }
    Ok(())
}

fn execute_call(
    session: &mut SessionWriter,
    cwd: &PathBuf,
    skills_dir: &PathBuf,
    sessions_dir: &Path,
    call: ToolCall,
) -> Result<ToolResult> {
    let registry = agent_tools::seed_registry();
    let ctx = ToolContext::with_paths(
        cwd.clone(),
        skills_dir.clone(),
        cwd.join("memory"),
        sessions_dir.to_path_buf(),
    );
    session.append(AgentEvent::ToolStarted { call: call.clone() })?;
    let result = match registry.execute(&ctx, &call) {
        Ok(result) => result,
        Err(err) => ToolResult::error(&call, err.to_string()),
    };
    if call.name == "update_working_checkpoint" && result.ok {
        let key_info = result
            .content
            .get("key_info")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        let related_skill = result
            .content
            .get("related_skill")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        session.append(AgentEvent::CheckpointUpdated {
            key_info,
            related_skill,
        })?;
    }
    if call.name == "start_long_term_update" && result.ok {
        let reason = result
            .content
            .get("reason")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        let evidence = result
            .content
            .get("evidence")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        session.append(AgentEvent::LongTermUpdateStarted { reason, evidence })?;
    }
    if call.name == "complete_long_term_update" && result.ok {
        let decision = result
            .content
            .get("decision")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        let target = result
            .content
            .get("target")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let reason = result
            .content
            .get("reason")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        let evidence = result
            .content
            .get("evidence")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let changed = result
            .content
            .get("changed")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        session.append(AgentEvent::LongTermUpdateSettled {
            decision,
            target,
            reason,
            evidence,
            changed,
        })?;
    }
    maybe_rebuild_memory_index_after_tool(&ctx, &call, &result)?;
    session.append(AgentEvent::ToolFinished {
        result: result.clone(),
    })?;
    Ok(result)
}

fn maybe_rebuild_memory_index_after_tool(
    ctx: &ToolContext,
    call: &ToolCall,
    result: &ToolResult,
) -> Result<()> {
    if !result.ok || !matches!(call.name.as_str(), "patch_file" | "write_file") {
        return Ok(());
    }
    let Some(path) = result.content.get("path").and_then(|value| value.as_str()) else {
        return Ok(());
    };
    let path = PathBuf::from(path);
    if path.starts_with(&ctx.memory_dir) || path.starts_with(&ctx.skills_dir) {
        let paths = agent_memory::MemoryPaths::new(
            ctx.memory_dir.clone(),
            ctx.skills_dir.clone(),
            ctx.sessions_dir.clone(),
        );
        agent_memory::rebuild_index(&paths)?;
    }
    Ok(())
}

fn parse_seed_goal(goal: &str) -> Option<ToolCall> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn codex_prompt_injects_routed_repoprompt_skill() {
        let root =
            std::env::temp_dir().join(format!("seed-cli-skill-routing-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("repoprompt-deep-plan")).unwrap();
        fs::write(
            root.join("repoprompt-deep-plan").join("SKILL.md"),
            "---\nname: RepoPrompt Deep Plan\ndescription: Plan with RepoPrompt.\ntags: [repoprompt, plan]\n---\n\n# RepoPrompt Deep Plan\n\nUse builder/export.\n",
        )
        .unwrap();

        let prompt = codex_prompt_with_routed_skill("帮我实现 plan runtime", &root).unwrap();

        assert!(prompt.contains("Skill route: RepoPrompt Deep Plan"));
        assert!(prompt.contains("<local_skill>"));
        assert!(prompt.contains("Use builder/export."));
        assert!(prompt.contains("<task>\n帮我实现 plan runtime\n</task>"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn codex_prompt_leaves_unrouted_tasks_plain() {
        let root =
            std::env::temp_dir().join(format!("seed-cli-no-skill-routing-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        let prompt = codex_prompt_with_routed_skill("say pong", &root).unwrap();

        assert_eq!(prompt, "say pong");
        let _ = fs::remove_dir_all(&root);
    }
}
