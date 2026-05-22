use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str::FromStr;
use std::thread;
use std::time::Duration;
use wait_timeout::ChildExt;

pub const DEFAULT_REPOPROMPT_CLI: &str = "repoprompt_cli";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RepoPromptToolInfo {
    pub name: &'static str,
    pub group: &'static str,
    pub description: &'static str,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RepoPromptTool {
    AgentManage,
    AgentRun,
    AppSettings,
    ApplyEdits,
    BindContext,
    ContextBuilder,
    FileActions,
    FileSearch,
    GetCodeStructure,
    GetFileTree,
    Git,
    ManageSelection,
    ManageWorkspaces,
    OracleSend,
    OracleUtils,
    Prompt,
    ReadFile,
    WorkspaceContext,
}

impl RepoPromptTool {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AgentManage => "agent_manage",
            Self::AgentRun => "agent_run",
            Self::AppSettings => "app_settings",
            Self::ApplyEdits => "apply_edits",
            Self::BindContext => "bind_context",
            Self::ContextBuilder => "context_builder",
            Self::FileActions => "file_actions",
            Self::FileSearch => "file_search",
            Self::GetCodeStructure => "get_code_structure",
            Self::GetFileTree => "get_file_tree",
            Self::Git => "git",
            Self::ManageSelection => "manage_selection",
            Self::ManageWorkspaces => "manage_workspaces",
            Self::OracleSend => "oracle_send",
            Self::OracleUtils => "oracle_utils",
            Self::Prompt => "prompt",
            Self::ReadFile => "read_file",
            Self::WorkspaceContext => "workspace_context",
        }
    }

    pub fn info(self) -> RepoPromptToolInfo {
        match self {
            Self::AgentManage => RepoPromptToolInfo {
                name: self.as_str(),
                group: "agent",
                description: "Discover agents, manage sessions, read logs, export handoffs, and list workflows.",
            },
            Self::AgentRun => RepoPromptToolInfo {
                name: self.as_str(),
                group: "agent",
                description: "Start, poll, wait, steer, respond to, or cancel Agent Mode runs.",
            },
            Self::AppSettings => RepoPromptToolInfo {
                name: self.as_str(),
                group: "settings",
                description: "Read or update allowlisted RepoPrompt app settings.",
            },
            Self::ApplyEdits => RepoPromptToolInfo {
                name: self.as_str(),
                group: "edit",
                description: "Apply literal replacements, multi-edits, or whole-file rewrites.",
            },
            Self::BindContext => RepoPromptToolInfo {
                name: self.as_str(),
                group: "routing",
                description: "List and bind RepoPrompt windows, tabs, context IDs, or workspace roots.",
            },
            Self::ContextBuilder => RepoPromptToolInfo {
                name: self.as_str(),
                group: "context",
                description: "Auto-explore a repository, build selection, and optionally produce plan/question/review responses.",
            },
            Self::FileActions => RepoPromptToolInfo {
                name: self.as_str(),
                group: "edit",
                description: "Create, delete, or move files.",
            },
            Self::FileSearch => RepoPromptToolInfo {
                name: self.as_str(),
                group: "explore",
                description: "Search by file path and/or file content with filters and context lines.",
            },
            Self::GetCodeStructure => RepoPromptToolInfo {
                name: self.as_str(),
                group: "explore",
                description: "Return codemaps with function and type signatures.",
            },
            Self::GetFileTree => RepoPromptToolInfo {
                name: self.as_str(),
                group: "explore",
                description: "Render repository roots or file/folder trees.",
            },
            Self::Git => RepoPromptToolInfo {
                name: self.as_str(),
                group: "git",
                description: "Safe read-only git status, diff, log, show, and blame.",
            },
            Self::ManageSelection => RepoPromptToolInfo {
                name: self.as_str(),
                group: "context",
                description: "Curate selection as full files, slices, or codemap-only entries.",
            },
            Self::ManageWorkspaces => RepoPromptToolInfo {
                name: self.as_str(),
                group: "routing",
                description: "Manage workspaces and compose tabs.",
            },
            Self::OracleSend => RepoPromptToolInfo {
                name: self.as_str(),
                group: "conversation",
                description: "Ask or continue an oracle conversation in chat, plan, edit, or review mode.",
            },
            Self::OracleUtils => RepoPromptToolInfo {
                name: self.as_str(),
                group: "conversation",
                description: "List oracle models or sessions.",
            },
            Self::Prompt => RepoPromptToolInfo {
                name: self.as_str(),
                group: "context",
                description: "Get, set, append, clear, export, or preset the shared prompt.",
            },
            Self::ReadFile => RepoPromptToolInfo {
                name: self.as_str(),
                group: "explore",
                description: "Read full files, slices, or tails.",
            },
            Self::WorkspaceContext => RepoPromptToolInfo {
                name: self.as_str(),
                group: "context",
                description: "Render or export prompt, selection, codemaps, files, tree, and token context.",
            },
        }
    }

    pub fn all() -> &'static [RepoPromptTool] {
        &[
            Self::AgentManage,
            Self::AgentRun,
            Self::AppSettings,
            Self::ApplyEdits,
            Self::BindContext,
            Self::ContextBuilder,
            Self::FileActions,
            Self::FileSearch,
            Self::GetCodeStructure,
            Self::GetFileTree,
            Self::Git,
            Self::ManageSelection,
            Self::ManageWorkspaces,
            Self::OracleSend,
            Self::OracleUtils,
            Self::Prompt,
            Self::ReadFile,
            Self::WorkspaceContext,
        ]
    }
}

impl FromStr for RepoPromptTool {
    type Err = anyhow::Error;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let normalized = input.trim().replace('-', "_").to_ascii_lowercase();
        RepoPromptTool::all()
            .iter()
            .copied()
            .find(|tool| tool.as_str() == normalized)
            .ok_or_else(|| anyhow::anyhow!("unsupported RepoPrompt tool: {input}"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RepoPromptClientConfig {
    pub cli_path: PathBuf,
    pub timeout_secs: u64,
    pub window_id: Option<u32>,
    pub tab: Option<String>,
    pub context_id: Option<String>,
    pub working_dirs: Vec<PathBuf>,
    pub raw_json: bool,
}

impl Default for RepoPromptClientConfig {
    fn default() -> Self {
        Self {
            cli_path: default_cli_path(),
            timeout_secs: 300,
            window_id: None,
            tab: None,
            context_id: None,
            working_dirs: Vec::new(),
            raw_json: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RepoPromptOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub json: Option<Value>,
}

impl RepoPromptOutput {
    pub fn status(&self) -> &'static str {
        if self.timed_out {
            "timed_out"
        } else if self.exit_code == Some(0) {
            "success"
        } else {
            "failed"
        }
    }
}

#[derive(Debug, Clone)]
pub struct RepoPromptClient {
    cfg: RepoPromptClientConfig,
}

impl RepoPromptClient {
    pub fn new(cfg: RepoPromptClientConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &RepoPromptClientConfig {
        &self.cfg
    }

    pub fn check_available(&self) -> Result<()> {
        let path = &self.cfg.cli_path;
        if !path.is_file() {
            bail!("RepoPrompt CLI not found: {}", path.display());
        }
        Ok(())
    }

    pub fn exec(&self, command: &str) -> Result<RepoPromptOutput> {
        self.check_available()?;
        let args = self.args_for_exec(command);
        self.run(args)
    }

    pub fn call_tool(&self, tool: RepoPromptTool, args: &Value) -> Result<RepoPromptOutput> {
        self.check_available()?;
        let args = self.args_for_call(tool, args)?;
        self.run(args)
    }

    pub fn describe_tool(&self, tool: RepoPromptTool) -> Result<RepoPromptOutput> {
        self.check_available()?;
        let mut args = self.routing_args();
        args.push("--describe".to_string());
        args.push(tool.as_str().to_string());
        self.run(args)
    }

    pub fn args_for_exec(&self, command: &str) -> Vec<String> {
        let mut args = self.routing_args();
        if self.cfg.raw_json {
            args.push("--raw-json".to_string());
        }
        args.push("--exec".to_string());
        args.push(command.to_string());
        args
    }

    pub fn args_for_call(&self, tool: RepoPromptTool, args_json: &Value) -> Result<Vec<String>> {
        let mut args = self.routing_args();
        if self.cfg.raw_json {
            args.push("--raw-json".to_string());
        }
        let mut payload = args_json.clone();
        if let (Some(window_id), Value::Object(map)) = (self.cfg.window_id, &mut payload) {
            map.entry("_windowID".to_string())
                .or_insert_with(|| Value::from(window_id));
        }
        args.push("--call".to_string());
        args.push(tool.as_str().to_string());
        args.push("--json".to_string());
        args.push(serde_json::to_string(&payload)?);
        Ok(args)
    }

    fn routing_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        if let Some(window_id) = self.cfg.window_id {
            args.push("--window".to_string());
            args.push(window_id.to_string());
        }
        if let Some(tab) = &self.cfg.tab {
            args.push("--tab".to_string());
            args.push(tab.clone());
        }
        if let Some(context_id) = &self.cfg.context_id {
            args.push("--context-id".to_string());
            args.push(context_id.clone());
        }
        for working_dir in &self.cfg.working_dirs {
            args.push("--working-dir".to_string());
            args.push(working_dir.display().to_string());
        }
        args
    }

    fn run(&self, args: Vec<String>) -> Result<RepoPromptOutput> {
        let mut child = Command::new(&self.cfg.cli_path)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawn {}", self.cfg.cli_path.display()))?;

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let out_handle = thread::spawn(move || read_pipe(stdout));
        let err_handle = thread::spawn(move || read_pipe(stderr));
        let timeout = Duration::from_secs(self.cfg.timeout_secs.max(1));
        let mut timed_out = false;
        let status = match child.wait_timeout(timeout)? {
            Some(status) => status,
            None => {
                timed_out = true;
                let _ = child.kill();
                child.wait().context("wait for killed RepoPrompt CLI")?
            }
        };

        let stdout = out_handle.join().unwrap_or_default();
        let stderr = err_handle.join().unwrap_or_default();
        let json = serde_json::from_str(stdout.trim()).ok();

        Ok(RepoPromptOutput {
            stdout,
            stderr,
            exit_code: status.code(),
            timed_out,
            json,
        })
    }
}

fn read_pipe(pipe: Option<impl Read>) -> String {
    let Some(mut pipe) = pipe else {
        return String::new();
    };
    let mut out = String::new();
    let _ = pipe.read_to_string(&mut out);
    out
}

pub fn default_cli_path() -> PathBuf {
    if let Some(path) = env::var_os("REPOPROMPT_CLI") {
        return PathBuf::from(path);
    }
    if let Some(home) = env::var_os("HOME") {
        let candidate = PathBuf::from(home).join("RepoPrompt/repoprompt_cli");
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from(DEFAULT_REPOPROMPT_CLI)
}

pub fn parse_args_json(text: &str) -> Result<Value> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(Value::Object(Default::default()));
    }
    let path = Path::new(trimmed);
    if path.is_file() {
        let body =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        return serde_json::from_str(&body).with_context(|| format!("parse {}", path.display()));
    }
    serde_json::from_str(trimmed).context("parse RepoPrompt args JSON")
}

pub fn known_tools() -> Vec<RepoPromptToolInfo> {
    RepoPromptTool::all()
        .iter()
        .map(|tool| tool.info())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn lists_all_repoprompt_tools() {
        let tools = known_tools();
        assert_eq!(tools.len(), 18);
        assert!(tools.iter().any(|tool| tool.name == "agent_run"));
        assert!(tools.iter().any(|tool| tool.name == "apply_edits"));
        assert!(tools.iter().any(|tool| tool.name == "workspace_context"));
    }

    #[test]
    fn parses_tool_names_with_dash_alias() {
        assert_eq!(
            RepoPromptTool::from_str("file-search").unwrap(),
            RepoPromptTool::FileSearch
        );
        assert!(RepoPromptTool::from_str("unknown").is_err());
    }

    #[test]
    fn builds_exec_args_with_routing() {
        let cfg = RepoPromptClientConfig {
            window_id: Some(3),
            tab: Some("Plan".to_string()),
            raw_json: true,
            ..Default::default()
        };
        let args = RepoPromptClient::new(cfg).args_for_exec("windows");

        assert_eq!(
            args,
            vec![
                "--window",
                "3",
                "--tab",
                "Plan",
                "--raw-json",
                "--exec",
                "windows"
            ]
        );
    }

    #[test]
    fn builds_call_args() {
        let cfg = RepoPromptClientConfig {
            context_id: Some("ctx".to_string()),
            raw_json: true,
            ..Default::default()
        };
        let args = RepoPromptClient::new(cfg)
            .args_for_call(RepoPromptTool::FileSearch, &json!({ "pattern": "TODO" }))
            .unwrap();

        assert_eq!(args[0], "--context-id");
        assert_eq!(args[1], "ctx");
        assert!(args.contains(&"file_search".to_string()));
        assert!(args.contains(&r#"{"pattern":"TODO"}"#.to_string()));
    }

    #[test]
    fn embeds_window_id_for_call_routing() {
        let cfg = RepoPromptClientConfig {
            window_id: Some(3),
            ..Default::default()
        };
        let args = RepoPromptClient::new(cfg)
            .args_for_call(RepoPromptTool::AgentManage, &json!({ "op": "list_agents" }))
            .unwrap();

        assert_eq!(args[0], "--window");
        assert_eq!(args[1], "3");
        let payload: Value = serde_json::from_str(args.last().unwrap()).unwrap();
        assert_eq!(payload["_windowID"], json!(3));
        assert_eq!(payload["op"], json!("list_agents"));
    }
}
