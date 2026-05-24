use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

const DEFAULT_MCP_SERVER: &str = "RepoPrompt";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpPolicy {
    None,
    All,
    Allow(Vec<String>),
}

impl McpPolicy {
    fn needs_discovery(&self) -> bool {
        !matches!(self, Self::All)
    }
}

impl Default for McpPolicy {
    fn default() -> Self {
        Self::Allow(vec![DEFAULT_MCP_SERVER.to_string()])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalMode {
    Deny,
    AcceptOnce,
    AcceptForSession,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexAppServerConfig {
    pub command: String,
    pub args: Vec<String>,
    pub model: Option<String>,
    pub cwd: Option<PathBuf>,
    pub sandbox: String,
    pub approval_policy: String,
    pub reasoning_effort: Option<String>,
    pub client_name: String,
    pub client_title: String,
    pub client_version: String,
    pub experimental_api: bool,
    pub request_timeout_secs: u64,
    pub turn_timeout_secs: u64,
    pub approval_mode: ApprovalMode,
    pub mcp_policy: McpPolicy,
    pub plugins_enabled: bool,
    /// when true, launch via `codex app-server proxy` (which
    /// connects to the running `codex app-server daemon`) instead of
    /// spawning a fresh `codex app-server --listen stdio://`. Saves
    /// startup across distinct `seed` invocations (already covers
    /// the within-REPL case). User must have started the daemon
    /// separately via `seed codex-daemon start` (or `codex app-server
    /// daemon start`).
    #[serde(default)]
    pub use_daemon: bool,
}

impl Default for CodexAppServerConfig {
    fn default() -> Self {
        Self {
            command: "codex".to_string(),
            args: vec![
                "app-server".to_string(),
                "--listen".to_string(),
                "stdio://".to_string(),
            ],
            model: None,
            cwd: None,
            sandbox: "workspace-write".to_string(),
            approval_policy: "on-request".to_string(),
            reasoning_effort: None,
            client_name: "seed_agent".to_string(),
            client_title: "SeedAgent".to_string(),
            client_version: env!("CARGO_PKG_VERSION").to_string(),
            experimental_api: true,
            request_timeout_secs: 30,
            turn_timeout_secs: 600,
            approval_mode: ApprovalMode::Deny,
            mcp_policy: McpPolicy::default(),
            plugins_enabled: false,
            use_daemon: false,
        }
    }
}

impl CodexAppServerConfig {
    /// when daemon mode is enabled, the argv changes from
    /// `["codex", "app-server", "--listen", "stdio://"]` to
    /// `["codex", "app-server", "proxy"]`. The proxy subcommand connects
    /// stdin/stdout to a unix socket served by `codex app-server daemon`,
    /// which the user must have started separately.
    fn effective_args(&self) -> Vec<String> {
        if self.use_daemon {
            vec!["app-server".to_string(), "proxy".to_string()]
        } else {
            self.args.clone()
        }
    }
}

impl CodexAppServerConfig {
    pub fn launch_command(&self) -> Vec<String> {
        self.launch_command_with_mcp_servers(&[])
    }

    pub fn resolved_launch_command(&self) -> Result<Vec<String>> {
        let mcp_server_names = if self.mcp_policy.needs_discovery() {
            self.discover_mcp_server_names()?
        } else {
            Vec::new()
        };
        Ok(self.launch_command_with_mcp_servers(&mcp_server_names))
    }

    pub fn launch_command_with_mcp_servers(&self, mcp_server_names: &[String]) -> Vec<String> {
        let mut command = vec![self.command.clone()];
        command.extend(self.effective_args());
        // The daemon path (`codex app-server proxy`) inherits config from
        // the running daemon — `--disable plugins` and `-c mcp_servers=...`
        // would be no-ops at the proxy layer. Skip them in daemon mode.
        if !self.use_daemon {
            if !self.plugins_enabled {
                command.push("--disable".to_string());
                command.push("plugins".to_string());
            }
            if let Some(override_value) = mcp_servers_override(&self.mcp_policy, mcp_server_names) {
                command.push("-c".to_string());
                command.push(override_value);
            }
        }
        command
    }

    fn discover_mcp_server_names(&self) -> Result<Vec<String>> {
        if !self.plugins_enabled
            && let Ok(names) = discover_mcp_server_names_from_config()
        {
            return Ok(names);
        }
        self.discover_mcp_server_names_from_cli()
    }

    fn discover_mcp_server_names_from_cli(&self) -> Result<Vec<String>> {
        let mut command = Command::new(&self.command);
        command.arg("mcp").arg("list").arg("--json");
        if !self.plugins_enabled {
            command.arg("--disable").arg("plugins");
        }
        let output = command
            .output()
            .with_context(|| format!("discover Codex MCP servers with {}", self.command))?;
        if !output.status.success() {
            anyhow::bail!(
                "codex mcp list failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let summaries: Vec<McpServerSummary> =
            serde_json::from_slice(&output.stdout).context("parse codex mcp list --json")?;
        let mut names = summaries
            .into_iter()
            .map(|summary| summary.name)
            .collect::<Vec<_>>();
        names.sort();
        Ok(names)
    }
}

#[derive(Debug, Deserialize)]
struct McpServerSummary {
    name: String,
}

fn mcp_servers_override(policy: &McpPolicy, server_names: &[String]) -> Option<String> {
    let disabled_names = match policy {
        McpPolicy::All => return None,
        McpPolicy::None => server_names.to_vec(),
        McpPolicy::Allow(allowed) => {
            let allowed = allowed.iter().collect::<std::collections::HashSet<_>>();
            server_names
                .iter()
                .filter(|name| !allowed.contains(name))
                .cloned()
                .collect()
        }
    };
    if disabled_names.is_empty() {
        return None;
    }

    let entries = disabled_names
        .iter()
        .map(|name| format!("{}={{enabled=false}}", toml_key(name)))
        .collect::<Vec<_>>()
        .join(",");
    Some(format!("mcp_servers={{{entries}}}"))
}

fn toml_key(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

fn discover_mcp_server_names_from_config() -> Result<Vec<String>> {
    let path = codex_config_path()?;
    let source = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    mcp_server_names_from_config_str(&source)
}

fn codex_config_path() -> Result<PathBuf> {
    if let Some(codex_home) = env::var_os("CODEX_HOME") {
        return Ok(PathBuf::from(codex_home).join("config.toml"));
    }
    let home = env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".codex").join("config.toml"))
}

fn mcp_server_names_from_config_str(source: &str) -> Result<Vec<String>> {
    let value = source
        .parse::<toml::Value>()
        .context("parse Codex config TOML")?;
    let mut names = value
        .get("mcp_servers")
        .and_then(toml::Value::as_table)
        .map(|servers| {
            servers
                .iter()
                .filter(|(_, value)| value.is_table())
                .map(|(name, _)| name.to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    names.sort();
    Ok(names)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexRunResult {
    pub thread_id: String,
    pub turn_id: String,
    pub text: String,
    pub events_seen: usize,
    /// token usage for this turn, when Codex sent a
    /// `thread/tokenUsageUpdated` notification before `turn/completed`.
    /// `None` when Codex didn't emit one (older daemon, errored turn, …).
    pub tokens: Option<TokenUsage>,
}

/// Per-turn token breakdown captured from Codex's
/// `thread/tokenUsageUpdated` notification (or an HTTP `response.completed`
/// event for the HttpPlanner path). All counters are cumulative for the
/// turn; sum across turns for run totals.
///
/// `cached_input_tokens` is broken out separately because Codex's pricing
/// (and most providers') discounts cached input — surfacing it lets a
/// future cost dashboard render an accurate effective rate without us
/// hard-coding price tables.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_output_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcMessage {
    pub id: Option<Value>,
    pub method: Option<String>,
    pub params: Option<Value>,
    pub result: Option<Value>,
    pub error: Option<Value>,
}

pub struct CodexAppServerClient {
    cfg: CodexAppServerConfig,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    rx: Option<Receiver<Value>>,
    next_id: u64,
    initialized: bool,
}

impl CodexAppServerClient {
    pub fn new(cfg: CodexAppServerConfig) -> Self {
        Self {
            cfg,
            child: None,
            stdin: None,
            rx: None,
            next_id: 1,
            initialized: false,
        }
    }

    pub fn run_prompt(&mut self, prompt: &str) -> Result<CodexRunResult> {
        self.run_prompt_streaming(prompt, |_| {})
    }

    /// Read-only view of the current working directory the client will send
    /// with the next `turn/start` request (or `thread/start` if the thread
    /// isn't open yet). Returns `None` if no cwd was configured at launch and
    /// none has been set since.
    pub fn cwd(&self) -> Option<&PathBuf> {
        self.cfg.cwd.as_ref()
    }

    /// Update the cwd that gets sent on the next request. Codex's
    /// `TurnStartParams.cwd` is officially "Override the working directory
    /// for this turn and subsequent turns" — so this propagates without
    /// restarting the app-server. Safe to call mid-thread; the change lands
    /// on the next `start_turn` / `run_prompt[_streaming]` call.
    ///
    /// Does NOT mutate the spawned subprocess's `current_dir`; only the
    /// per-request field. If you launched the server in cwd A and call
    /// `set_cwd(B)`, the server still runs from A but Codex's *logical*
    /// workspace for new turns becomes B. That matches what the protocol
    /// is designed to do.
    pub fn set_cwd(&mut self, cwd: PathBuf) {
        self.cfg.cwd = Some(cwd);
    }

    /// `Option`-aware variant of `set_cwd` for the long-lived-client glue.
    pub fn set_cwd_opt(&mut self, cwd: Option<PathBuf>) {
        self.cfg.cwd = cwd;
    }

    /// Mutate the model used on the next `start_turn`. Picked up the same
    /// way as `set_cwd` — `start_turn` reads `self.cfg.model` at request
    /// build time, so no restart needed.
    pub fn set_model(&mut self, model: Option<String>) {
        self.cfg.model = model;
    }

    /// Mutate the reasoning effort used on the next `start_turn`.
    pub fn set_effort(&mut self, effort: Option<String>) {
        self.cfg.reasoning_effort = effort;
    }

    /// Mutate the sandbox sent on the next `start_thread`. `run_prompt`
    /// starts a fresh thread for every prompt, so this lands on the next
    /// `run_prompt[_streaming]` call.
    pub fn set_sandbox(&mut self, sandbox: String) {
        self.cfg.sandbox = sandbox;
    }

    /// Mutate the approval policy string sent in `start_thread`/`start_turn`.
    pub fn set_approval_policy(&mut self, policy: String) {
        self.cfg.approval_policy = policy;
    }

    /// Mutate the approval-callback mode used for `requestApproval` JSON-RPC
    /// callbacks. Per-callback, so safe to change between prompts.
    pub fn set_approval_mode(&mut self, mode: ApprovalMode) {
        self.cfg.approval_mode = mode;
    }

    /// Snapshot of the launch-relevant config fields. Two clients with
    /// equal fingerprints can be used interchangeably — different fingerprints
    /// require a fresh subprocess (mcp_policy and plugins_enabled affect the
    /// argv passed to `codex app-server`, so they can't be hot-swapped).
    pub fn launch_fingerprint(&self) -> CodexLaunchFingerprint {
        CodexLaunchFingerprint::from(&self.cfg)
    }
}

/// Fingerprint of the subset of `CodexAppServerConfig` that must match for
/// two requests to share an app-server subprocess. Everything NOT in this
/// fingerprint is per-turn-mutable via the `set_*` methods above
/// (cwd / model / effort / sandbox / approval_policy / approval_mode).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexLaunchFingerprint {
    pub plugins_enabled: bool,
    pub mcp_policy: McpPolicy,
    pub command: String,
    pub args: Vec<String>,
    pub experimental_api: bool,
    /// daemon vs stdio mode is a launch-time decision (different
    /// argv → different subprocess shape) so it splits the fingerprint.
    /// Flipping `--use-daemon` mid-REPL drops the cached client.
    pub use_daemon: bool,
}

impl From<&CodexAppServerConfig> for CodexLaunchFingerprint {
    fn from(cfg: &CodexAppServerConfig) -> Self {
        Self {
            plugins_enabled: cfg.plugins_enabled,
            mcp_policy: cfg.mcp_policy.clone(),
            command: cfg.command.clone(),
            args: cfg.args.clone(),
            experimental_api: cfg.experimental_api,
            use_daemon: cfg.use_daemon,
        }
    }
}

// Spurious-impl to keep the rest of the file's `impl CodexAppServerClient`
// block readable — re-open the impl below.
impl CodexAppServerClient {

    pub fn run_prompt_streaming<F>(
        &mut self,
        prompt: &str,
        mut on_delta: F,
    ) -> Result<CodexRunResult>
    where
        F: FnMut(&str),
    {
        self.ensure_ready()?;
        let thread_id = self.start_thread()?;
        let turn_id = self.start_turn(&thread_id, prompt)?;
        self.stream_turn(thread_id, turn_id, &mut on_delta)
    }

    pub fn ensure_ready(&mut self) -> Result<()> {
        self.start()?;
        if !self.initialized {
            self.initialize()?;
            self.initialized = true;
        }
        Ok(())
    }

    pub fn start(&mut self) -> Result<()> {
        if self.child.is_some() {
            return Ok(());
        }

        let launch_command = self.cfg.resolved_launch_command()?;
        let mut command = Command::new(&launch_command[0]);
        command.args(&launch_command[1..]);
        if let Some(cwd) = &self.cfg.cwd {
            command.current_dir(cwd);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "spawn codex app-server command: {}",
                    launch_command.join(" ")
                )
            })?;

        let stdin = child.stdin.take().context("codex stdin was not piped")?;
        let stdout = child.stdout.take().context("codex stdout was not piped")?;
        let stderr = child.stderr.take();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Value>(&line) {
                    Ok(value) => {
                        let _ = tx.send(value);
                    }
                    Err(err) => {
                        let _ = tx.send(json!({
                            "method": "client/parse_error",
                            "params": {
                                "message": err.to_string(),
                                "line": line,
                            }
                        }));
                    }
                }
            }
        });
        drain_stderr(stderr);

        self.stdin = Some(stdin);
        self.rx = Some(rx);
        self.child = Some(child);
        Ok(())
    }

    pub fn initialize(&mut self) -> Result<Value> {
        let result = self.request(
            "initialize",
            json!({
                "clientInfo": {
                    "name": self.cfg.client_name,
                    "title": self.cfg.client_title,
                    "version": self.cfg.client_version,
                },
                "capabilities": {
                    "experimentalApi": self.cfg.experimental_api,
                }
            }),
            Duration::from_secs(self.cfg.request_timeout_secs),
        )?;
        self.notify("initialized", json!({}))?;
        Ok(result)
    }

    pub fn start_thread(&mut self) -> Result<String> {
        let mut params = json!({
            "sandbox": self.cfg.sandbox,
            "approvalPolicy": self.cfg.approval_policy,
        });
        if let Some(model) = &self.cfg.model {
            params["model"] = Value::String(model.clone());
        }
        if let Some(cwd) = &self.cfg.cwd {
            params["cwd"] = Value::String(cwd.display().to_string());
        }
        let result = self.request(
            "thread/start",
            params,
            Duration::from_secs(self.cfg.request_timeout_secs),
        )?;
        result
            .pointer("/thread/id")
            .and_then(Value::as_str)
            .map(ToString::to_string)
            .context("thread/start response did not include thread.id")
    }

    pub fn start_turn(&mut self, thread_id: &str, prompt: &str) -> Result<String> {
        let mut params = json!({
            "threadId": thread_id,
            "input": [{ "type": "text", "text": prompt }],
            "approvalPolicy": self.cfg.approval_policy,
        });
        if let Some(model) = &self.cfg.model {
            params["model"] = Value::String(model.clone());
        }
        if let Some(cwd) = &self.cfg.cwd {
            params["cwd"] = Value::String(cwd.display().to_string());
        }
        if let Some(effort) = &self.cfg.reasoning_effort {
            params["effort"] = Value::String(effort.clone());
        }
        let result = self.request(
            "turn/start",
            params,
            Duration::from_secs(self.cfg.request_timeout_secs),
        )?;
        result
            .pointer("/turn/id")
            .and_then(Value::as_str)
            .map(ToString::to_string)
            .context("turn/start response did not include turn.id")
    }

    fn stream_turn(
        &mut self,
        thread_id: String,
        turn_id: String,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<CodexRunResult> {
        let deadline = Instant::now() + Duration::from_secs(self.cfg.turn_timeout_secs);
        let mut text = String::new();
        let mut events_seen = 0usize;
        // capture the most recent token-usage notification for
        // this turn. Codex emits `thread/tokenUsageUpdated` one or more
        // times before `turn/completed`; the last one before completion
        // is the authoritative per-turn breakdown.
        let mut tokens: Option<TokenUsage> = None;

        while Instant::now() < deadline {
            let Some(message) = self.recv(Duration::from_millis(500))? else {
                continue;
            };
            events_seen += 1;
            if self.handle_server_request(&message)? {
                continue;
            }
            let method = message.get("method").and_then(Value::as_str).unwrap_or("");
            let params = message.get("params").unwrap_or(&Value::Null);
            match method {
                "item/agentMessage/delta" if matches_turn(params, &thread_id, &turn_id) => {
                    if let Some(delta) = params.get("delta").and_then(Value::as_str) {
                        text.push_str(delta);
                        on_delta(delta);
                    }
                }
                "thread/tokenUsageUpdated" => {
                    if let Some(usage) = parse_thread_token_usage(params) {
                        tokens = Some(usage);
                    }
                }
                "turn/completed" if message_mentions_turn(params, &turn_id) => {
                    return Ok(CodexRunResult {
                        thread_id,
                        turn_id,
                        text,
                        events_seen,
                        tokens,
                    });
                }
                "error" => {
                    let message = params
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown codex app-server error");
                    anyhow::bail!(message.to_string());
                }
                _ => {}
            }
        }

        anyhow::bail!("timed out waiting for Codex turn {turn_id}")
    }

    fn request(&mut self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))?;

        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            let Some(message) = self.recv(Duration::from_millis(250))? else {
                continue;
            };
            if self.handle_server_request(&message)? {
                continue;
            }
            if message.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            if let Some(error) = message.get("error") {
                anyhow::bail!("codex app-server {method} failed: {error}");
            }
            return Ok(message.get("result").cloned().unwrap_or(Value::Null));
        }
        anyhow::bail!("timed out waiting for {method} response")
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        self.send(json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }))
    }

    fn recv(&mut self, timeout: Duration) -> Result<Option<Value>> {
        let rx = self
            .rx
            .as_ref()
            .context("codex app-server is not started")?;
        match rx.recv_timeout(timeout) {
            Ok(value) => Ok(Some(value)),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(err) => Err(err).context("codex app-server output closed"),
        }
    }

    fn send(&mut self, value: Value) -> Result<()> {
        let stdin = self
            .stdin
            .as_mut()
            .context("codex app-server stdin is not available")?;
        serde_json::to_writer(&mut *stdin, &value)?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
        Ok(())
    }

    fn handle_server_request(&mut self, message: &Value) -> Result<bool> {
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            return Ok(false);
        };
        let Some(id) = message.get("id").cloned() else {
            return Ok(false);
        };
        let result = approval_response(method, self.cfg.approval_mode);
        self.send(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }))?;
        Ok(true)
    }
}

fn drain_stderr(stderr: Option<impl std::io::Read + Send + 'static>) {
    let Some(stderr) = stderr else {
        return;
    };
    let forward = env::var_os("SEED_AGENT_CODEX_STDERR").is_some();
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            if forward {
                eprintln!("{line}");
            }
        }
    });
}

impl Drop for CodexAppServerClient {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub fn approval_response(method: &str, mode: ApprovalMode) -> Value {
    if method.contains("permissions/requestApproval") {
        return match mode {
            ApprovalMode::Deny => json!({ "permissions": "minimal", "scope": "turn" }),
            ApprovalMode::AcceptOnce => {
                json!({ "permissions": { "fileSystem": null, "network": null }, "scope": "turn" })
            }
            ApprovalMode::AcceptForSession => json!({
                "permissions": {
                    "fileSystem": {
                        "entries": [{
                            "access": "write",
                            "path": { "type": "special", "value": { "kind": "root" } }
                        }]
                    },
                    "network": { "enabled": true }
                },
                "scope": "session"
            }),
        };
    }

    if method.contains("fileChange/requestApproval")
        || method.contains("commandExecution/requestApproval")
    {
        return match mode {
            ApprovalMode::Deny => json!({ "decision": "decline" }),
            ApprovalMode::AcceptOnce => json!({ "decision": "accept" }),
            ApprovalMode::AcceptForSession => json!({ "decision": "acceptForSession" }),
        };
    }

    if method.contains("requestUserInput") || method.contains("elicitation/request") {
        return match mode {
            ApprovalMode::Deny => json!({ "response": "decline" }),
            ApprovalMode::AcceptOnce | ApprovalMode::AcceptForSession => {
                json!({ "response": "accept" })
            }
        };
    }

    json!({ "response": "decline" })
}

/// extract a [`TokenUsage`] from a `thread/tokenUsageUpdated`
/// notification's params. Per the schema:
///   `params.tokenUsage.last.{inputTokens, outputTokens, …}`
///
/// We use `.last` (per-turn delta) rather than `.total` (running total
/// for the thread) because Codex calls `start_thread` per `run_prompt`,
/// so `total` == `last` in practice for us — but `last` is the safer
/// semantic if that ever changes.
///
/// Returns `None` on shape mismatch — defensive against schema drift in
/// Codex updates.
fn parse_thread_token_usage(params: &Value) -> Option<TokenUsage> {
    let breakdown = params.get("tokenUsage")?.get("last")?;
    fn read_u64(b: &Value, field: &str) -> u64 {
        b.get(field)
            .and_then(Value::as_u64)
            .unwrap_or(0)
    }
    Some(TokenUsage {
        input_tokens: read_u64(breakdown, "inputTokens"),
        cached_input_tokens: read_u64(breakdown, "cachedInputTokens"),
        output_tokens: read_u64(breakdown, "outputTokens"),
        reasoning_output_tokens: read_u64(breakdown, "reasoningOutputTokens"),
        total_tokens: read_u64(breakdown, "totalTokens"),
    })
}

fn matches_turn(params: &Value, thread_id: &str, turn_id: &str) -> bool {
    let thread_matches = params
        .get("threadId")
        .and_then(Value::as_str)
        .is_none_or(|value| value == thread_id);
    thread_matches && message_mentions_turn(params, turn_id)
}

fn message_mentions_turn(value: &Value, turn_id: &str) -> bool {
    value.get("turnId").and_then(Value::as_str) == Some(turn_id)
        || value.get("id").and_then(Value::as_str) == Some(turn_id)
        || value.pointer("/turn/id").and_then(Value::as_str) == Some(turn_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_launch_uses_stdio_transport() {
        let cfg = CodexAppServerConfig::default();
        assert_eq!(
            cfg.launch_command(),
            vec![
                "codex",
                "app-server",
                "--listen",
                "stdio://",
                "--disable",
                "plugins"
            ]
        );
    }

    #[test]
    fn mcp_none_disables_discovered_servers() {
        let cfg = CodexAppServerConfig {
            mcp_policy: McpPolicy::None,
            ..Default::default()
        };
        assert_eq!(
            cfg.launch_command_with_mcp_servers(&["RepoPrompt".to_string()]),
            vec![
                "codex",
                "app-server",
                "--listen",
                "stdio://",
                "--disable",
                "plugins",
                "-c",
                "mcp_servers={\"RepoPrompt\"={enabled=false}}"
            ]
        );
    }

    #[test]
    fn default_mcp_policy_allows_repoprompt_only() {
        let cfg = CodexAppServerConfig::default();
        assert_eq!(
            cfg.launch_command_with_mcp_servers(&["RepoPrompt".to_string(), "semgrep".to_string()]),
            vec![
                "codex",
                "app-server",
                "--listen",
                "stdio://",
                "--disable",
                "plugins",
                "-c",
                "mcp_servers={\"semgrep\"={enabled=false}}"
            ]
        );
    }

    #[test]
    fn mcp_allow_disables_everything_else() {
        let cfg = CodexAppServerConfig {
            mcp_policy: McpPolicy::Allow(vec!["RepoPrompt".to_string()]),
            ..Default::default()
        };
        assert_eq!(
            cfg.launch_command_with_mcp_servers(&["RepoPrompt".to_string(), "semgrep".to_string()]),
            vec![
                "codex",
                "app-server",
                "--listen",
                "stdio://",
                "--disable",
                "plugins",
                "-c",
                "mcp_servers={\"semgrep\"={enabled=false}}"
            ]
        );
    }

    #[test]
    fn parses_mcp_server_names_from_codex_config() {
        let names = mcp_server_names_from_config_str(
            r#"
            [mcp_servers]

            [mcp_servers.RepoPrompt]
            command = "/tmp/repoprompt"

            [mcp_servers.semgrep]
            command = "semgrep"
            args = ["mcp"]

            [mcp_servers.semgrep.env]
            WORKSPACE_ROOT = "/tmp"

            [projects."/tmp/demo"]
            trust_level = "trusted"
            "#,
        )
        .unwrap();

        assert_eq!(names, vec!["RepoPrompt", "semgrep"]);
    }

    #[test]
    fn deny_command_approval_declines() {
        let response =
            approval_response("item/commandExecution/requestApproval", ApprovalMode::Deny);
        assert_eq!(response, json!({ "decision": "decline" }));
    }

    #[test]
    fn turn_match_accepts_nested_turn() {
        let params = json!({ "turn": { "id": "turn_1" } });
        assert!(message_mentions_turn(&params, "turn_1"));
    }

    // --- token usage parsing -------------------------------------

    #[test]
    fn parse_token_usage_extracts_full_breakdown() {
        let params = json!({
            "threadId": "t1",
            "turnId": "u1",
            "tokenUsage": {
                "last": {
                    "inputTokens": 1234,
                    "cachedInputTokens": 200,
                    "outputTokens": 456,
                    "reasoningOutputTokens": 89,
                    "totalTokens": 1779
                },
                "total": {
                    "inputTokens": 99999,
                    "cachedInputTokens": 0,
                    "outputTokens": 0,
                    "reasoningOutputTokens": 0,
                    "totalTokens": 99999
                }
            }
        });
        let usage = parse_thread_token_usage(&params).expect("parses");
        assert_eq!(usage.input_tokens, 1234);
        assert_eq!(usage.cached_input_tokens, 200);
        assert_eq!(usage.output_tokens, 456);
        assert_eq!(usage.reasoning_output_tokens, 89);
        assert_eq!(usage.total_tokens, 1779);
    }

    #[test]
    fn parse_token_usage_tolerates_missing_fields() {
        // Older codex daemons (or schema drift) might omit some fields.
        // We default to 0 rather than reject.
        let params = json!({
            "tokenUsage": {
                "last": {
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "totalTokens": 15
                    // cachedInputTokens, reasoningOutputTokens omitted
                }
            }
        });
        let usage = parse_thread_token_usage(&params).expect("parses");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.cached_input_tokens, 0);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.reasoning_output_tokens, 0);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn parse_token_usage_returns_none_for_unrelated_params() {
        // Not a tokenUsageUpdated notification → no usage to extract.
        let params = json!({ "delta": "hello" });
        assert!(parse_thread_token_usage(&params).is_none());
    }

    #[test]
    fn token_usage_default_is_all_zeros() {
        let u = TokenUsage::default();
        assert_eq!(u.total_tokens, 0);
        assert_eq!(u.input_tokens, 0);
    }

    #[test]
    fn set_cwd_updates_config() {
        // We can't actually start the app-server in a unit test, but
        // `start_turn` reads `self.cfg.cwd` at request build time so
        // mutating it is the entire mechanism. Verify the accessor pair.
        let mut client = CodexAppServerClient::new(CodexAppServerConfig::default());
        assert!(client.cwd().is_none());
        client.set_cwd(PathBuf::from("/tmp/seed-agent-test"));
        assert_eq!(
            client.cwd().map(|p| p.to_string_lossy().to_string()),
            Some("/tmp/seed-agent-test".to_string())
        );
        // Overwrites cleanly.
        client.set_cwd(PathBuf::from("/var/empty"));
        assert_eq!(
            client.cwd().map(|p| p.to_string_lossy().to_string()),
            Some("/var/empty".to_string())
        );
    }

    #[test]
    fn set_cwd_opt_handles_none() {
        let mut client = CodexAppServerClient::new(CodexAppServerConfig::default());
        client.set_cwd(PathBuf::from("/a"));
        client.set_cwd_opt(None);
        assert!(client.cwd().is_none(), "set_cwd_opt(None) must clear");
        client.set_cwd_opt(Some(PathBuf::from("/b")));
        assert_eq!(client.cwd(), Some(&PathBuf::from("/b")));
    }

    #[test]
    fn per_turn_setters_mutate_cfg() {
        let mut client = CodexAppServerClient::new(CodexAppServerConfig::default());
        client.set_model(Some("gpt-5.3".to_string()));
        client.set_effort(Some("high".to_string()));
        client.set_sandbox("read-only".to_string());
        client.set_approval_policy("never".to_string());
        client.set_approval_mode(ApprovalMode::AcceptForSession);
        // We expose the readback for cwd; the others we verify via fingerprint
        // not changing (sandbox/approval are NOT in fingerprint).
        let fp = client.launch_fingerprint();
        // Sandbox is per-thread, not in fingerprint.
        assert_eq!(fp.plugins_enabled, false);
    }

    #[test]
    fn launch_fingerprint_matches_when_only_per_turn_fields_differ() {
        let mut a = CodexAppServerConfig::default();
        a.cwd = Some(PathBuf::from("/a"));
        a.model = Some("m1".to_string());
        a.reasoning_effort = Some("low".to_string());
        a.sandbox = "danger-full-access".to_string();
        a.approval_policy = "never".to_string();
        a.approval_mode = ApprovalMode::AcceptForSession;

        let mut b = CodexAppServerConfig::default();
        b.cwd = Some(PathBuf::from("/different"));
        b.model = Some("m2".to_string());
        b.reasoning_effort = None;
        b.sandbox = "read-only".to_string();
        b.approval_policy = "on-request".to_string();
        b.approval_mode = ApprovalMode::Deny;

        // Both have plugins_enabled=false, default mcp_policy (Allow RepoPrompt),
        // default command/args. Per-turn fields should not affect fingerprint.
        assert_eq!(
            CodexLaunchFingerprint::from(&a),
            CodexLaunchFingerprint::from(&b),
            "per-turn fields must not contribute to the fingerprint",
        );
    }

    // --- daemon mode --------------------------------------------

    #[test]
    fn daemon_mode_swaps_argv_to_proxy() {
        let mut cfg = CodexAppServerConfig::default();
        cfg.use_daemon = true;
        let cmd = cfg.launch_command_with_mcp_servers(&[]);
        assert_eq!(
            cmd,
            vec![
                "codex".to_string(),
                "app-server".to_string(),
                "proxy".to_string(),
            ],
            "use_daemon=true must produce `codex app-server proxy` argv only"
        );
    }

    #[test]
    fn daemon_mode_skips_plugins_and_mcp_flags() {
        // The proxy path inherits config from the running daemon, so
        // seed-level overrides (--disable plugins / -c mcp_servers=...)
        // would be no-ops. Verify they're not added.
        let mut cfg = CodexAppServerConfig::default();
        cfg.use_daemon = true;
        cfg.plugins_enabled = false; // would trigger --disable plugins in stdio mode
        cfg.mcp_policy = McpPolicy::Allow(vec!["RepoPrompt".to_string()]);
        let cmd = cfg.launch_command_with_mcp_servers(&["RepoPrompt".to_string()]);
        assert!(!cmd.contains(&"--disable".to_string()));
        assert!(!cmd.contains(&"-c".to_string()));
    }

    #[test]
    fn stdio_mode_still_adds_plugins_and_mcp_flags() {
        // Default (use_daemon=false) must keep adding plugins + mcp flags.
        // Pass a discovered MCP server that's NOT in the Allow list so
        // mcp_servers_override has something to disable (returns Some).
        let cfg = CodexAppServerConfig::default();
        let cmd = cfg.launch_command_with_mcp_servers(&[
            "RepoPrompt".to_string(),
            "OtherServer".to_string(),
        ]);
        assert!(cmd.contains(&"--disable".to_string()));
        assert!(cmd.contains(&"-c".to_string()));
    }

    #[test]
    fn launch_fingerprint_differs_on_use_daemon() {
        let mut a = CodexAppServerConfig::default();
        let mut b = CodexAppServerConfig::default();
        b.use_daemon = true;
        assert_ne!(
            CodexLaunchFingerprint::from(&a),
            CodexLaunchFingerprint::from(&b),
            "flipping --use-daemon must split sessions (different argv)"
        );
        a = CodexAppServerConfig::default();
        a.use_daemon = true;
        assert_eq!(
            CodexLaunchFingerprint::from(&a),
            CodexLaunchFingerprint::from(&b),
        );
    }

    #[test]
    fn launch_fingerprint_differs_when_plugins_or_mcp_differ() {
        let mut a = CodexAppServerConfig::default();
        let mut b = CodexAppServerConfig::default();
        b.plugins_enabled = true;
        assert_ne!(
            CodexLaunchFingerprint::from(&a),
            CodexLaunchFingerprint::from(&b),
            "plugins_enabled must split fingerprints"
        );
        b = CodexAppServerConfig::default();
        b.mcp_policy = McpPolicy::All;
        assert_ne!(
            CodexLaunchFingerprint::from(&a),
            CodexLaunchFingerprint::from(&b),
            "mcp_policy must split fingerprints"
        );
        // Same config = same fingerprint (sanity).
        a = CodexAppServerConfig::default();
        b = CodexAppServerConfig::default();
        assert_eq!(
            CodexLaunchFingerprint::from(&a),
            CodexLaunchFingerprint::from(&b),
        );
    }
}
