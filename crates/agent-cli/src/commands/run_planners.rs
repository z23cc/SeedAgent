//! RF40-A3 partial: Planner trait + the three provider impls
//! (Oracle / Codex / Http) + `build_planner` factory, extracted from
//! `commands/run.rs` to keep that file from sprawling past 2000 lines.
//!
//! What stays in run.rs: `run_goal` itself, `drive_planner_loop`, all
//! tests (including `MockPlanner` for loop-driver integration tests).
//! `MockPlanner` implements the trait re-exported from here.

use std::cell::Cell;
use std::path::Path;

use agent_core::ToolInfo;
use agent_delegate::CodexAppServerClient;
use agent_llm::ModelId;
use anyhow::Result;

use super::run::PlannerProvider;
use crate::display::format_token_subtitle;
use crate::{ApprovalArg, McpArg};

/// Per-turn provider abstraction. `drive_planner_loop` only needs
/// "send the assembled prompt, get back a PlannedAction + char count".
/// Each provider does its own setup (subprocess, HTTP client, etc.).
pub(crate) trait Planner {
    /// Spinner label prefix per-turn, e.g. "planning" or "oracle".
    fn label(&self) -> &'static str;

    /// Called at the top of every turn before `plan`. Lets impls install a
    /// static subtitle (oracle: "RepoPrompt oracle thinking…"). Default clears
    /// the subtitle so Codex's streaming-token subtitle isn't stuck from the
    /// previous turn.
    fn on_turn_start(&self, spinner: &agent_core::tui::Spinner) {
        spinner.set_subtitle(None);
    }

    /// Produce the next planned action. Returns the action plus a rough
    /// response char count for timing telemetry (Http path can't measure this
    /// cheaply and returns 0).
    fn plan(
        &mut self,
        goal: &str,
        tool_infos: &[ToolInfo],
        state: &agent_runtime::AgentLoopState,
        memory: &agent_runtime::PlannerMemoryContext,
        spinner: &agent_core::tui::Spinner,
    ) -> Result<(agent_runtime::PlannedAction, usize), agent_runtime::RuntimeError>;

    /// RF36-1: input char count of the most-recently-built planner prompt.
    /// Returns `0` if the impl doesn't track it (default). CodexPlanner +
    /// HttpPlanner override to surface accurate counts; OraclePlanner
    /// could but the value is dominated by the working_memory section
    /// which is identical across providers — skip for now.
    fn last_prompt_chars(&self) -> usize {
        0
    }
}

pub(crate) struct OraclePlanner {
    oracle: agent_repoprompt::RepoPromptClient,
    mode: agent_repoprompt::OracleMode,
    chat_id: Option<String>,
}

impl Planner for OraclePlanner {
    fn label(&self) -> &'static str {
        "oracle"
    }

    fn on_turn_start(&self, spinner: &agent_core::tui::Spinner) {
        spinner.set_subtitle(Some("RepoPrompt oracle thinking…".to_string()));
    }

    fn plan(
        &mut self,
        goal: &str,
        tool_infos: &[ToolInfo],
        state: &agent_runtime::AgentLoopState,
        memory: &agent_runtime::PlannerMemoryContext,
        _spinner: &agent_core::tui::Spinner,
    ) -> Result<(agent_runtime::PlannedAction, usize), agent_runtime::RuntimeError> {
        let prompt =
            agent_runtime::planner_prompt_with_state_and_memory(goal, tool_infos, state, memory);
        let is_first_turn = self.chat_id.is_none();
        let chat_id_owned = self.chat_id.clone();
        let response = self
            .oracle
            .send_oracle(&prompt, self.mode, chat_id_owned.as_deref(), is_first_turn)
            .map_err(|err| {
                agent_runtime::RuntimeError::Planner(format!(
                    "RepoPrompt oracle send failed: {err}"
                ))
            })?;
        if !response.is_success() {
            // CLI returned a real response saying "no" — retrying with the
            // same prompt would just repeat the rejection. Mark fatal so the
            // runtime fails fast instead of burning N transport retries.
            return Err(agent_runtime::RuntimeError::PlannerFatal(format!(
                "RepoPrompt oracle returned exit_code={:?}; stderr={}",
                response.raw_output.exit_code,
                response.raw_output.stderr.trim()
            )));
        }
        if let Some(id) = &response.chat_id {
            self.chat_id = Some(id.clone());
        }
        let action = agent_runtime::parse_planned_action(&response.response_text)?;
        let chars = response.response_text.chars().count();
        Ok((action, chars))
    }
}

/// Planner-loop backend for `--provider codex`.
///
/// RF29-1: borrows `&'a mut CodexAppServerClient` from the REPL-lifetime
/// `CodexSession`. The session's `ensure()` decides whether to reuse the
/// existing subprocess or restart based on launch fingerprint, so a chain
/// of REPL goals on `--provider codex` shares one `codex app-server`
/// across goals instead of paying ~300ms startup per goal. The lifetime
/// `'a` is the borrow on `CodexSession`; `Box<dyn Planner + 'a>` carries
/// it through `build_planner`'s return type.
pub(crate) struct CodexPlanner<'a> {
    client: &'a mut CodexAppServerClient,
    /// RF36-1: char count of the last prompt we built. drive_planner_loop
    /// reads this after each `plan()` to record in TurnTimings.
    last_prompt_chars: usize,
}

impl<'a> Planner for CodexPlanner<'a> {
    fn label(&self) -> &'static str {
        "planning"
    }

    fn plan(
        &mut self,
        goal: &str,
        tool_infos: &[ToolInfo],
        state: &agent_runtime::AgentLoopState,
        memory: &agent_runtime::PlannerMemoryContext,
        spinner: &agent_core::tui::Spinner,
    ) -> Result<(agent_runtime::PlannedAction, usize), agent_runtime::RuntimeError> {
        let prompt =
            agent_runtime::planner_prompt_with_state_and_memory(goal, tool_infos, state, memory);
        // RF36-1: record prompt size so the loop driver can surface it
        // in TurnTimings. Using .chars().count() matches the response-side
        // metric — they're both rough proxies for token count.
        self.last_prompt_chars = prompt.chars().count();
        let delta_chars: Cell<usize> = Cell::new(0);
        let result = self
            .client
            .run_prompt_streaming(&prompt, |delta| {
                delta_chars.set(delta_chars.get() + delta.chars().count());
                spinner.set_subtitle(Some(format_token_subtitle(delta_chars.get())));
            })
            .map_err(|err| {
                agent_runtime::RuntimeError::Planner(format!("Codex planner failed: {err}"))
            })?;
        let action = agent_runtime::parse_planned_action(&result.text)?;
        Ok((action, delta_chars.get()))
    }

    fn last_prompt_chars(&self) -> usize {
        self.last_prompt_chars
    }
}

pub(crate) struct HttpPlanner {
    provider_id: String,
    model: String,
    /// RF36-1: same role as `CodexPlanner.last_prompt_chars`. We compute
    /// it by summing the rendered ChatRequest message content lengths,
    /// since the HTTP body isn't a single string we can `.chars().count()`.
    last_prompt_chars: usize,
}

impl Planner for HttpPlanner {
    fn label(&self) -> &'static str {
        "planning"
    }

    fn plan(
        &mut self,
        goal: &str,
        tool_infos: &[ToolInfo],
        state: &agent_runtime::AgentLoopState,
        memory: &agent_runtime::PlannerMemoryContext,
        spinner: &agent_core::tui::Spinner,
    ) -> Result<(agent_runtime::PlannedAction, usize), agent_runtime::RuntimeError> {
        // RF32: stream HTTP responses so the user sees live token counts in
        // the spinner subtitle, matching the Codex path. The blocking
        // ProviderClient internally posts with `stream: true` and parses
        // SSE incrementally; we accumulate the text via the on_delta
        // callback below.
        let provider = agent_llm::find_provider(&self.provider_id).ok_or_else(|| {
            agent_runtime::RuntimeError::PlannerFatal(format!(
                "provider not found: {}",
                self.provider_id
            ))
        })?;
        let request = agent_runtime::planner_request_with_state_and_memory(
            ModelId::from(self.model.clone()),
            goal,
            tool_infos,
            state,
            memory,
        );
        // RF36-1: estimate prompt size by summing message content lengths.
        // Approximate (ignores per-message JSON envelope overhead) but a
        // good proxy for "how full is my context?".
        self.last_prompt_chars = request
            .messages
            .iter()
            .map(|m| m.content.chars().count())
            .sum();
        let delta_chars: Cell<usize> = Cell::new(0);
        let response = agent_llm::ProviderClient::new()
            .chat_streaming(provider, request, |delta| {
                delta_chars.set(delta_chars.get() + delta.chars().count());
                spinner.set_subtitle(Some(format_token_subtitle(delta_chars.get())));
            })
            .map_err(|err| {
                agent_runtime::RuntimeError::Planner(format!("HTTP planner failed: {err}"))
            })?;
        let action = agent_runtime::parse_planned_action(&response.text)?;
        Ok((action, delta_chars.get()))
    }

    fn last_prompt_chars(&self) -> usize {
        self.last_prompt_chars
    }
}

/// Build the concrete planner. Consumes the provider-specific fields out of
/// the ProviderSpec match. Returns None for Oracle when the RepoPrompt CLI
/// isn't reachable (caller records the failure into the session).
//
// Genuinely 9 distinct knobs (Codex needs all of them; Oracle/Http use a
// subset). Bundling into a ProviderSpec struct would force re-cloning in
// run_goal where the post-loop synthesis pass needs separate copies of
// model/approval/effort/mcp/mcp_allow; the explicit list is clearer here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_planner<'a>(
    kind: &PlannerProvider,
    cwd: &Path,
    model: Option<String>,
    approval: ApprovalArg,
    effort: Option<String>,
    turn_timeout_secs: u64,
    mcp: Option<McpArg>,
    mcp_allow: Vec<String>,
    plugins: bool,
    use_daemon: bool,
    codex_session: &'a mut crate::commands::codex_session::CodexSession,
) -> Result<Box<dyn Planner + 'a>> {
    match kind {
        PlannerProvider::Oracle => {
            let oracle_cfg = agent_repoprompt::RepoPromptClientConfig {
                cli_path: agent_repoprompt::default_cli_path(),
                timeout_secs: turn_timeout_secs.max(60),
                raw_json: true,
                working_dirs: vec![cwd.to_path_buf()],
                ..Default::default()
            };
            let oracle = agent_repoprompt::RepoPromptClient::new(oracle_cfg);
            oracle.check_available().map_err(|err| {
                anyhow::anyhow!(
                    "RepoPrompt oracle unavailable: {err}. Re-run with --provider codex to bypass."
                )
            })?;
            let mode: agent_repoprompt::OracleMode = model
                .as_deref()
                .and_then(|m| m.parse().ok())
                .unwrap_or(agent_repoprompt::OracleMode::Chat);
            Ok(Box::new(OraclePlanner {
                oracle,
                mode,
                chat_id: None,
            }))
        }
        PlannerProvider::Codex => {
            let cfg = crate::commands::codex::codex_config_full(
                model,
                Some(cwd.to_path_buf()),
                approval,
                effort,
                turn_timeout_secs,
                mcp,
                mcp_allow,
                plugins,
                use_daemon,
            )?;
            // RF29-1: borrow the live client from the REPL-lifetime session.
            // If the launch fingerprint matches a previously-cached client,
            // we reuse the subprocess (no spawn, ~0ms). Otherwise ensure()
            // drops the old one and spawns fresh.
            let client = codex_session.ensure(cfg)?;
            Ok(Box::new(CodexPlanner {
                client,
                last_prompt_chars: 0,
            }))
        }
        PlannerProvider::Http(provider_id) => {
            let model = model.unwrap_or_else(|| "gpt-5.1".to_string());
            #[allow(clippy::redundant_field_names)]
            Ok(Box::new(HttpPlanner {
                last_prompt_chars: 0,
                provider_id: provider_id.clone(),
                model,
            }))
        }
    }
}
