//! `seed rp` subcommand and RepoPrompt-CLI helpers shared with `seed plan`
//! (verify) and `seed run` (oracle planner / agent_run). All thin wrappers
//! around `agent_repoprompt::RepoPromptClient` — no LLM, no session state.

use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use clap::Subcommand;
use serde_json::{Value, json};

use crate::absolutize_cli;
use crate::display::compact_single_line_cli;

#[derive(Debug, Subcommand)]
pub(crate) enum RpCommand {
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

pub(crate) fn default_repoprompt_working_dirs_cli(
    cwd: &Path,
    working_dirs: Vec<PathBuf>,
    default_cwd: bool,
) -> Vec<PathBuf> {
    let mut working_dirs = working_dirs;
    if working_dirs.is_empty() && default_cwd {
        working_dirs.push(cwd.to_path_buf());
    }
    working_dirs
        .into_iter()
        .map(|path| absolutize_cli(cwd, path))
        .collect()
}

pub(crate) fn repoprompt_report_text_cli(output: &agent_repoprompt::RepoPromptOutput) -> String {
    if !output.stdout.trim().is_empty() {
        output.stdout.clone()
    } else if let Some(json) = &output.json {
        serde_json::to_string_pretty(json).unwrap_or_else(|_| json.to_string())
    } else {
        output.stderr.clone()
    }
}

pub(crate) fn repoprompt_client_cli(
    mut cfg: agent_repoprompt::RepoPromptClientConfig,
) -> Result<agent_repoprompt::RepoPromptClient> {
    resolve_repoprompt_window_cli(&mut cfg)?;
    Ok(agent_repoprompt::RepoPromptClient::new(cfg))
}

fn resolve_repoprompt_window_cli(cfg: &mut agent_repoprompt::RepoPromptClientConfig) -> Result<()> {
    if cfg.window_id.is_some() || cfg.context_id.is_some() || cfg.working_dirs.is_empty() {
        return Ok(());
    }
    let bind_cfg = agent_repoprompt::RepoPromptClientConfig {
        cli_path: cfg.cli_path.clone(),
        timeout_secs: cfg.timeout_secs,
        raw_json: true,
        ..Default::default()
    };
    let output = agent_repoprompt::RepoPromptClient::new(bind_cfg).call_tool(
        agent_repoprompt::RepoPromptTool::BindContext,
        &json!({
            "op": "bind",
            "working_dirs": cfg
                .working_dirs
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>(),
            "create_if_missing": false,
        }),
    )?;
    if output.timed_out || output.exit_code != Some(0) {
        bail!(
            "RepoPrompt bind failed before routed call: {}",
            compact_single_line_cli(&repoprompt_report_text_cli(&output), 800)
        );
    }
    if let Some(window_id) = repoprompt_output_u32_cli(&output, &["window_id", "windowID"]) {
        cfg.window_id = Some(window_id);
        return Ok(());
    }
    bail!("RepoPrompt bind succeeded but did not return a window_id")
}

pub(crate) fn repoprompt_output_string_cli(
    output: &agent_repoprompt::RepoPromptOutput,
    keys: &[&str],
) -> Option<String> {
    output
        .json
        .as_ref()
        .and_then(|json| find_string_by_key_cli(json, keys))
}

fn repoprompt_output_u32_cli(
    output: &agent_repoprompt::RepoPromptOutput,
    keys: &[&str],
) -> Option<u32> {
    output
        .json
        .as_ref()
        .and_then(|json| find_u32_by_key_cli(json, keys))
}

fn find_string_by_key_cli(value: &Value, keys: &[&str]) -> Option<String> {
    match value {
        Value::Object(map) => {
            for key in keys {
                if let Some(value) = map.get(*key).and_then(value_to_non_empty_string_cli) {
                    return Some(value);
                }
            }
            map.values()
                .find_map(|value| find_string_by_key_cli(value, keys))
        }
        Value::Array(values) => values
            .iter()
            .find_map(|value| find_string_by_key_cli(value, keys)),
        _ => None,
    }
}

fn find_u32_by_key_cli(value: &Value, keys: &[&str]) -> Option<u32> {
    match value {
        Value::Object(map) => {
            for key in keys {
                if let Some(value) = map.get(*key).and_then(value_to_u32_cli) {
                    return Some(value);
                }
            }
            map.values()
                .find_map(|value| find_u32_by_key_cli(value, keys))
        }
        Value::Array(values) => values
            .iter()
            .find_map(|value| find_u32_by_key_cli(value, keys)),
        _ => None,
    }
}

fn value_to_non_empty_string_cli(value: &Value) -> Option<String> {
    match value {
        Value::String(value) if !value.trim().is_empty() => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

fn value_to_u32_cli(value: &Value) -> Option<u32> {
    match value {
        Value::Number(value) => value.as_u64().and_then(|value| u32::try_from(value).ok()),
        Value::String(value) => value.parse::<u32>().ok(),
        _ => None,
    }
}

fn default_cwd_for_repoprompt_exec_cli(command: &str) -> bool {
    let command = command.trim().to_ascii_lowercase();
    !(command == "windows"
        || command.starts_with("windows ")
        || command == "workspace list"
        || command.starts_with("workspace list ")
        || command == "workspaces"
        || command.starts_with("bind_context")
        || command.starts_with("app_settings"))
}

fn default_cwd_for_repoprompt_tool_cli(tool: agent_repoprompt::RepoPromptTool) -> bool {
    !matches!(
        tool,
        agent_repoprompt::RepoPromptTool::BindContext
            | agent_repoprompt::RepoPromptTool::ManageWorkspaces
            | agent_repoprompt::RepoPromptTool::AppSettings
            | agent_repoprompt::RepoPromptTool::OracleUtils
            | agent_repoprompt::RepoPromptTool::AgentManage
    )
}

#[derive(derive_setters::Setters)]
#[setters(into, strip_option)]
pub(crate) struct RpRunArgs {
    pub(crate) command: RpCommand,
    pub(crate) cli_path: Option<PathBuf>,
    pub(crate) window_id: Option<u32>,
    pub(crate) tab: Option<String>,
    pub(crate) context_id: Option<String>,
    pub(crate) working_dirs: Vec<PathBuf>,
    pub(crate) timeout_secs: u64,
    pub(crate) raw_json: bool,
}

pub(crate) fn run_rp(args: RpRunArgs) -> Result<()> {
    let cwd = env::current_dir()?;
    let mut cfg = agent_repoprompt::RepoPromptClientConfig {
        timeout_secs: args.timeout_secs,
        window_id: args.window_id,
        tab: args.tab,
        context_id: args.context_id,
        working_dirs: default_repoprompt_working_dirs_cli(&cwd, args.working_dirs, false),
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
            if cfg.working_dirs.is_empty() && default_cwd_for_repoprompt_exec_cli(&command) {
                cfg.working_dirs.push(cwd.clone());
            }
            let output = repoprompt_client_cli(cfg)?.exec(&command)?;
            print_rp_output(output)
        }
        RpCommand::Call { tool, args } => {
            cfg.raw_json = true;
            let tool = tool.parse::<agent_repoprompt::RepoPromptTool>()?;
            if cfg.working_dirs.is_empty() && default_cwd_for_repoprompt_tool_cli(tool) {
                cfg.working_dirs.push(cwd.clone());
            }
            let value = agent_repoprompt::parse_args_json(&args)?;
            let output = repoprompt_client_cli(cfg)?.call_tool(tool, &value)?;
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
                default_repoprompt_working_dirs_cli(&cwd, working_dirs, false)
            };
            let bind_dirs = if bind_dirs.is_empty() {
                vec![cwd.clone()]
            } else {
                bind_dirs
            };
            let mut bind_cfg = cfg;
            bind_cfg.raw_json = true;
            bind_cfg.working_dirs.clear();
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
            let output = agent_repoprompt::RepoPromptClient::new(bind_cfg)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_repoprompt_working_dirs_default_to_cwd_when_requested() {
        let root =
            std::env::temp_dir().join(format!("seed-cli-rp-default-cwd-{}", std::process::id()));
        let dirs = default_repoprompt_working_dirs_cli(&root, Vec::new(), true);

        assert_eq!(dirs, vec![root]);
    }

    #[test]
    fn cli_repoprompt_discovery_commands_do_not_default_to_cwd() {
        assert!(!default_cwd_for_repoprompt_exec_cli("workspace list"));
        assert!(default_cwd_for_repoprompt_exec_cli("search \"TODO\""));
        assert!(!default_cwd_for_repoprompt_tool_cli(
            agent_repoprompt::RepoPromptTool::BindContext
        ));
        assert!(default_cwd_for_repoprompt_tool_cli(
            agent_repoprompt::RepoPromptTool::AgentRun
        ));
    }
}
