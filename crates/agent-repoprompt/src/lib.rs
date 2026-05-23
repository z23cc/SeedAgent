use anyhow::{Context, Result};
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

/// Library error surface for `agent-repoprompt`. The dominant failure mode
/// at the API boundary is "the CLI isn't reachable / mis-installed" — the
/// `CliUnavailable` variant makes that pattern-matchable so callers can
/// gracefully degrade (e.g. the run-loop disables the oracle planner instead
/// of erroring out).
#[derive(Debug, thiserror::Error)]
pub enum RepoPromptError {
    /// `check_available` failed (binary missing, version mismatch, etc).
    /// The inner string is the underlying CLI's stderr or our probe message.
    #[error("RepoPrompt CLI unavailable: {reason}")]
    CliUnavailable { reason: String },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type RepoPromptResult<T> = std::result::Result<T, RepoPromptError>;

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

    pub fn check_available(&self) -> RepoPromptResult<()> {
        let path = &self.cfg.cli_path;
        if !path.is_file() {
            return Err(RepoPromptError::CliUnavailable {
                reason: format!("CLI not found: {}", path.display()),
            });
        }
        Ok(())
    }

    pub fn exec(&self, command: &str) -> RepoPromptResult<RepoPromptOutput> {
        self.check_available()?;
        let args = self.args_for_exec(command);
        Ok(self.run(args)?)
    }

    pub fn build_context(
        &self,
        instructions: &str,
        response_type: BuilderResponseType,
        export_response: bool,
    ) -> RepoPromptResult<ContextBuilderResponse> {
        let mut payload = serde_json::Map::new();
        payload.insert(
            "instructions".to_string(),
            Value::String(instructions.to_string()),
        );
        payload.insert(
            "response_type".to_string(),
            Value::String(response_type.as_str().to_string()),
        );
        if export_response {
            payload.insert("export_response".to_string(), Value::Bool(true));
        }
        let mut cfg = self.cfg.clone();
        cfg.raw_json = true;
        // Builder calls take 30s–5min per RepoPrompt docs; ensure we don't truncate.
        if cfg.timeout_secs < 600 {
            cfg.timeout_secs = 600;
        }
        let raw_client = RepoPromptClient::new(cfg);
        let output =
            raw_client.call_tool(RepoPromptTool::ContextBuilder, &Value::Object(payload))?;
        Ok(ContextBuilderResponse::from_output(output))
    }

    pub fn send_oracle(
        &self,
        message: &str,
        mode: OracleMode,
        chat_id: Option<&str>,
        new_chat: bool,
    ) -> RepoPromptResult<OracleResponse> {
        let mut payload = serde_json::Map::new();
        payload.insert(
            "message".to_string(),
            Value::String(message.to_string()),
        );
        payload.insert(
            "mode".to_string(),
            Value::String(mode.as_str().to_string()),
        );
        if let Some(id) = chat_id {
            payload.insert("chat_id".to_string(), Value::String(id.to_string()));
        }
        if new_chat {
            payload.insert("new_chat".to_string(), Value::Bool(true));
        }
        let mut cfg = self.cfg.clone();
        cfg.raw_json = true;
        let raw_client = RepoPromptClient::new(cfg);
        let output = raw_client.call_tool(RepoPromptTool::OracleSend, &Value::Object(payload))?;
        Ok(OracleResponse::from_output(output))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BuilderResponseType {
    Clarify,
    Question,
    Plan,
    Review,
}

impl BuilderResponseType {
    pub fn as_str(self) -> &'static str {
        match self {
            BuilderResponseType::Clarify => "clarify",
            BuilderResponseType::Question => "question",
            BuilderResponseType::Plan => "plan",
            BuilderResponseType::Review => "review",
        }
    }
}

impl FromStr for BuilderResponseType {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "clarify" => Ok(Self::Clarify),
            "question" => Ok(Self::Question),
            "plan" => Ok(Self::Plan),
            "review" => Ok(Self::Review),
            other => Err(format!("unknown builder response_type: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBuilderResponse {
    pub response_text: String,
    pub chat_id: Option<String>,
    pub oracle_export_path: Option<PathBuf>,
    pub raw_output: RepoPromptOutput,
}

impl ContextBuilderResponse {
    pub fn from_output(output: RepoPromptOutput) -> Self {
        let json = output.json.as_ref();
        let response_text = extract_response_text(json, &output.stdout);
        let chat_id = json
            .and_then(|value| value.get("chat_id"))
            .and_then(Value::as_str)
            .map(ToString::to_string);
        let oracle_export_path = json
            .and_then(|value| value.get("oracle_export_path"))
            .and_then(Value::as_str)
            .map(PathBuf::from);
        Self {
            response_text,
            chat_id,
            oracle_export_path,
            raw_output: output,
        }
    }

    pub fn is_success(&self) -> bool {
        if self.raw_output.timed_out || self.raw_output.exit_code != Some(0) {
            return false;
        }
        if json_signals_error(self.raw_output.json.as_ref()) {
            return false;
        }
        true
    }

    pub fn error_message(&self) -> Option<&str> {
        self.raw_output
            .json
            .as_ref()
            .and_then(|value| value.get("error"))
            .and_then(Value::as_str)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OracleMode {
    Chat,
    Plan,
    Edit,
    Review,
}

impl OracleMode {
    pub fn as_str(self) -> &'static str {
        match self {
            OracleMode::Chat => "chat",
            OracleMode::Plan => "plan",
            OracleMode::Edit => "edit",
            OracleMode::Review => "review",
        }
    }
}

impl FromStr for OracleMode {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "chat" => Ok(Self::Chat),
            "plan" => Ok(Self::Plan),
            "edit" => Ok(Self::Edit),
            "review" => Ok(Self::Review),
            other => Err(format!("unknown oracle mode: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleResponse {
    pub response_text: String,
    pub chat_id: Option<String>,
    pub oracle_export_path: Option<PathBuf>,
    pub raw_output: RepoPromptOutput,
}

impl OracleResponse {
    pub fn from_output(output: RepoPromptOutput) -> Self {
        let json = output.json.as_ref();
        let response_text = extract_response_text(json, &output.stdout);
        let chat_id = json
            .and_then(|value| value.get("chat_id"))
            .and_then(Value::as_str)
            .map(ToString::to_string);
        let oracle_export_path = json
            .and_then(|value| value.get("oracle_export_path"))
            .and_then(Value::as_str)
            .map(PathBuf::from);
        Self {
            response_text,
            chat_id,
            oracle_export_path,
            raw_output: output,
        }
    }

    pub fn is_success(&self) -> bool {
        if self.raw_output.timed_out || self.raw_output.exit_code != Some(0) {
            return false;
        }
        if json_signals_error(self.raw_output.json.as_ref()) {
            return false;
        }
        true
    }

    pub fn error_message(&self) -> Option<&str> {
        self.raw_output
            .json
            .as_ref()
            .and_then(|value| value.get("error"))
            .and_then(Value::as_str)
    }
}

impl RepoPromptClient {

    pub fn call_tool(&self, tool: RepoPromptTool, args: &Value) -> RepoPromptResult<RepoPromptOutput> {
        self.check_available()?;
        let args = self.args_for_call(tool, args)?;
        Ok(self.run(args)?)
    }

    pub fn describe_tool(&self, tool: RepoPromptTool) -> RepoPromptResult<RepoPromptOutput> {
        self.check_available()?;
        let mut args = self.routing_args();
        args.push("--describe".to_string());
        args.push(tool.as_str().to_string());
        Ok(self.run(args)?)
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

    pub fn args_for_call(&self, tool: RepoPromptTool, args_json: &Value) -> RepoPromptResult<Vec<String>> {
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
        args.push(serde_json::to_string(&payload).context("serialize tool args")?);
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
        let json = parse_repoprompt_json_payload(&stdout);

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

pub fn parse_args_json(text: &str) -> RepoPromptResult<Value> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(Value::Object(Default::default()));
    }
    let path = Path::new(trimmed);
    if path.is_file() {
        let body =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        return Ok(
            serde_json::from_str(&body).with_context(|| format!("parse {}", path.display()))?,
        );
    }
    Ok(serde_json::from_str(trimmed).context("parse RepoPrompt args JSON")?)
}

pub fn known_tools() -> Vec<RepoPromptToolInfo> {
    RepoPromptTool::all()
        .iter()
        .map(|tool| tool.info())
        .collect()
}

/// Parse the JSON payload out of the RepoPrompt CLI stdout. The CLI emits
/// `[progress] ...` lines before the final JSON body, so we have to strip them.
/// Returns None when no JSON object is present.
pub fn parse_repoprompt_json_payload(stdout: &str) -> Option<Value> {
    let candidate = stdout
        .lines()
        .rev()
        .find(|line| {
            let trimmed = line.trim();
            (trimmed.starts_with('{') && trimmed.ends_with('}'))
                || (trimmed.starts_with('[') && trimmed.ends_with(']'))
        })
        .map(str::trim)
        .map(ToString::to_string)
        .or_else(|| {
            let cleaned: String = stdout
                .lines()
                .filter(|line| !line.trim().starts_with("[progress]"))
                .collect::<Vec<_>>()
                .join("\n");
            let trimmed = cleaned.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })?;
    serde_json::from_str(&candidate).ok()
}

/// Heuristic: the RepoPrompt CLI signals failure in-band via `{"error":"...","is_error":true}`
/// even when the process exit code is 0. Treat that as a hard error.
fn json_signals_error(json: Option<&Value>) -> bool {
    json.and_then(|value| value.get("is_error"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || json.and_then(|value| value.get("error"))
            .and_then(Value::as_str)
            .is_some_and(|err| !err.trim().is_empty())
}

/// Try the known response field names. Returns a clear error sentinel when
/// JSON was parsed but none of the known fields matched — so callers can detect
/// API drift instead of silently parsing raw stdout as the model reply.
fn extract_response_text(json: Option<&Value>, fallback_stdout: &str) -> String {
    if let Some(value) = json {
        for field in [
            "response",
            "text",
            "message",
            "assistant_message",
            "plan",
            "output",
            "reply",
        ] {
            if let Some(text) = value.get(field).and_then(Value::as_str) {
                return text.to_string();
            }
        }
        // JSON parsed but no known field. Surface a structured error string so
        // downstream parsers (e.g. parse_planned_action) fail loudly rather than
        // attempting to parse the whole envelope as a planner action.
        return format!(
            "[repoprompt-response-error] none of the known response fields (response/text/message/assistant_message/plan/output/reply) were present; raw envelope keys: {}",
            value
                .as_object()
                .map(|obj| obj.keys().cloned().collect::<Vec<_>>().join(","))
                .unwrap_or_default()
        );
    }
    // Last resort: when the CLI returned no parseable JSON at all, the raw
    // stdout is the only signal we have.
    fallback_stdout.trim().to_string()
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
    fn oracle_response_parses_common_field_shapes() {
        let primary = RepoPromptOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(0),
            timed_out: false,
            json: Some(json!({
                "response": "hello world",
                "chat_id": "chat-42",
                "oracle_export_path": "/tmp/oracle.md",
            })),
        };
        let parsed = OracleResponse::from_output(primary);
        assert_eq!(parsed.response_text, "hello world");
        assert_eq!(parsed.chat_id.as_deref(), Some("chat-42"));
        assert_eq!(
            parsed.oracle_export_path,
            Some(PathBuf::from("/tmp/oracle.md"))
        );
        assert!(parsed.is_success());

        let fallback_field = RepoPromptOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(0),
            timed_out: false,
            json: Some(json!({ "text": "fallback" })),
        };
        assert_eq!(
            OracleResponse::from_output(fallback_field).response_text,
            "fallback"
        );

        let raw_only = RepoPromptOutput {
            stdout: "raw answer\n".to_string(),
            stderr: String::new(),
            exit_code: Some(0),
            timed_out: false,
            json: None,
        };
        assert_eq!(OracleResponse::from_output(raw_only).response_text, "raw answer");
    }

    #[test]
    fn builder_response_round_trips_and_parses_export_path() {
        for kind in [
            BuilderResponseType::Clarify,
            BuilderResponseType::Question,
            BuilderResponseType::Plan,
            BuilderResponseType::Review,
        ] {
            let parsed: BuilderResponseType = kind.as_str().parse().unwrap();
            assert_eq!(parsed, kind);
        }
        assert!("nope".parse::<BuilderResponseType>().is_err());

        let output = RepoPromptOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(0),
            timed_out: false,
            json: Some(json!({
                "response": "# Plan\n...",
                "oracle_export_path": "/tmp/plan.md",
                "chat_id": "chat-7",
            })),
        };
        let parsed = ContextBuilderResponse::from_output(output);
        assert_eq!(parsed.response_text, "# Plan\n...");
        assert_eq!(
            parsed.oracle_export_path,
            Some(PathBuf::from("/tmp/plan.md"))
        );
        assert!(parsed.is_success());
    }

    #[test]
    fn oracle_response_treats_in_band_is_error_as_failure() {
        let output = RepoPromptOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(0),
            timed_out: false,
            json: Some(json!({
                "error": "Multiple windows detected",
                "is_error": true,
            })),
        };
        let parsed = OracleResponse::from_output(output);
        assert!(!parsed.is_success(), "is_error=true must override exit_code=0");
        assert_eq!(parsed.error_message(), Some("Multiple windows detected"));
    }

    #[test]
    fn oracle_response_unknown_field_returns_structured_error() {
        let output = RepoPromptOutput {
            stdout: "{\"unexpected\":\"shape\"}".to_string(),
            stderr: String::new(),
            exit_code: Some(0),
            timed_out: false,
            json: Some(json!({ "unexpected": "shape" })),
        };
        let parsed = OracleResponse::from_output(output);
        assert!(
            parsed.response_text.starts_with("[repoprompt-response-error]"),
            "expected loud error, got: {}",
            parsed.response_text
        );
        assert!(parsed.response_text.contains("unexpected"));
    }

    #[test]
    fn parse_repoprompt_json_payload_strips_progress_lines() {
        let stdout = "[progress] oracle_send: Starting Oracle...\n[progress] oracle_send: Oracle complete\n{\"chat_id\":\"abc\",\"response\":\"ok\"}\n";
        let parsed = parse_repoprompt_json_payload(stdout).expect("payload present");
        assert_eq!(parsed["chat_id"], "abc");
        assert_eq!(parsed["response"], "ok");
    }

    #[test]
    fn parse_repoprompt_json_payload_returns_none_for_empty() {
        assert!(parse_repoprompt_json_payload("").is_none());
        assert!(parse_repoprompt_json_payload("[progress] only\n").is_none());
    }

    #[test]
    fn oracle_mode_round_trips() {
        for mode in [OracleMode::Chat, OracleMode::Plan, OracleMode::Edit, OracleMode::Review] {
            let parsed: OracleMode = mode.as_str().parse().unwrap();
            assert_eq!(parsed, mode);
        }
        assert!("nope".parse::<OracleMode>().is_err());
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
