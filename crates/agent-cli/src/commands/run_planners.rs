//! partial: Planner trait + the three provider impls
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
use crate::{ApprovalArg, McpArg};

/// progress events emitted by a `Planner::plan()` call.
/// The driver translates these into UI (spinner subtitle updates,
/// telemetry, etc.) so the Planner impls stay pure and testable.
///
/// Before this refactor, the trait took `&Spinner` as a parameter and
/// each impl directly called `spinner.set_subtitle(...)`. That coupled
/// every backend to the UI layer and made unit-testing the planners
/// awkward (you needed a fake spinner). With `ProgressEvent`, the
/// planners just call `on_progress(event)` and the driver decides what
/// to do with it.
pub(crate) enum ProgressEvent {
    /// Set the spinner subtitle to a fixed string (e.g. "oracle thinking…"
    /// at turn start). `None` clears any prior subtitle.
    StaticSubtitle(Option<String>),
    /// Streaming token-count update — for Codex/HTTP planners that emit
    /// SSE deltas. The driver renders this as `(N tokens streamed)`.
    StreamingTokens(usize),
}

/// output of one planner turn. Bundles the action with the
/// telemetry fields so `drive_planner_loop` doesn't need to query
/// `last_prompt_chars()` separately afterward.
pub(crate) struct PlanOutput {
    pub action: agent_runtime::PlannedAction,
    /// Rough char count of the planner's response text. Used as a proxy
    /// for output tokens when the provider doesn't report usage directly.
    pub response_chars: usize,
    /// char count of the assembled planner prompt. `0` means
    /// "not tracked" (no provider reports 0 legitimately — they always
    /// have at least the system+user messages).
    pub prompt_chars: usize,
}

/// Per-turn provider abstraction. `drive_planner_loop` only needs
/// "send the assembled prompt, get back a PlannedAction + telemetry".
/// Each provider does its own setup (subprocess, HTTP client, etc.).
///
/// trait is now UI-free. Progress is reported via the
/// `on_progress` callback so impls can be unit-tested without a spinner.
pub(crate) trait Planner {
    /// Spinner label prefix per-turn, e.g. "planning" or "oracle".
    fn label(&self) -> &'static str;

    /// Optional static subtitle to install at the top of each turn
    /// before `plan()` is called. Default `None` clears any prior subtitle.
    /// (Oracle returns `Some("RepoPrompt oracle thinking…")` so the user
    /// knows the call is in flight even though there's no streaming.)
    fn turn_start_subtitle(&self) -> Option<String> {
        None
    }

    /// Produce the next planned action. `on_progress` is called by
    /// streaming backends as token deltas arrive; non-streaming
    /// backends (Oracle) don't call it at all.
    fn plan(
        &mut self,
        goal: &str,
        tool_infos: &[ToolInfo],
        state: &agent_runtime::AgentLoopState,
        memory: &agent_runtime::PlannerMemoryContext,
        on_progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PlanOutput, agent_runtime::RuntimeError>;
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

    fn turn_start_subtitle(&self) -> Option<String> {
        Some("RepoPrompt oracle thinking…".to_string())
    }

    fn plan(
        &mut self,
        goal: &str,
        tool_infos: &[ToolInfo],
        state: &agent_runtime::AgentLoopState,
        memory: &agent_runtime::PlannerMemoryContext,
        _on_progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PlanOutput, agent_runtime::RuntimeError> {
        let prompt =
            agent_runtime::planner_prompt_with_state_and_memory(goal, tool_infos, state, memory);
        let prompt_chars = prompt.chars().count();
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
        Ok(PlanOutput {
            action,
            response_chars: response.response_text.chars().count(),
            prompt_chars,
        })
    }
}

/// Planner-loop backend for `--provider repoprompt-agent`.
///
/// Uses RepoPrompt's `agent_run` tool (full Agent Mode) instead of
/// `ask_oracle`. Differs from [`OraclePlanner`] in two ways:
///
/// 1. **Continuity model**: Oracle uses `chat_id` for one-shot Q&A
///    threads; this uses `session_id` from `agent_run start` and
///    `steer`s subsequent turns through the same session. RepoPrompt's
///    agent infrastructure preserves richer per-turn context.
/// 2. **Role labels**: `model_id` selects among `explore` (fast read),
///    `engineer` (balanced impl), `pair` (highest tier), `design`
///    (writes review markdown). Defaults to `pair` if `--model` is
///    omitted; passing e.g. `--model engineer` overrides.
pub(crate) struct RepoPromptAgentPlanner {
    client: agent_repoprompt::RepoPromptClient,
    /// Role label passed as `model_id` on `start`. One of
    /// `explore|engineer|pair|design`, or a compound model id from
    /// `agent_manage.list_agents`. Default `pair`.
    role: String,
    /// Session UUID returned by the first `start` op. `None` on the
    /// first turn — `plan()` switches to `steer` once this is set.
    session_id: Option<String>,
    /// Per-turn `wait` / `timeout` seconds passed through to
    /// `agent_run`. RepoPrompt's default is 300; we mirror the seed
    /// turn timeout so a long-running agent turn doesn't outlive the
    /// loop's own per-turn budget.
    timeout_secs: u64,
}

impl Planner for RepoPromptAgentPlanner {
    fn label(&self) -> &'static str {
        "rp-agent"
    }

    fn turn_start_subtitle(&self) -> Option<String> {
        Some(format!("RepoPrompt agent ({}) thinking…", self.role))
    }

    fn plan(
        &mut self,
        goal: &str,
        tool_infos: &[ToolInfo],
        state: &agent_runtime::AgentLoopState,
        memory: &agent_runtime::PlannerMemoryContext,
        _on_progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PlanOutput, agent_runtime::RuntimeError> {
        let prompt =
            agent_runtime::planner_prompt_with_state_and_memory(goal, tool_infos, state, memory);
        let prompt_chars = prompt.chars().count();

        // First turn → `start`, returns session_id. Subsequent turns
        // → `steer` on the same session, preserves agent context.
        let args = if let Some(session_id) = &self.session_id {
            serde_json::json!({
                "op": "steer",
                "session_id": session_id,
                "message": prompt,
                "wait": true,
                "timeout_seconds": self.timeout_secs,
            })
        } else {
            serde_json::json!({
                "op": "start",
                "message": prompt,
                "model_id": self.role,
                "timeout": self.timeout_secs,
            })
        };

        let output = self
            .client
            .call_tool(agent_repoprompt::RepoPromptTool::AgentRun, &args)
            .map_err(|err| {
                agent_runtime::RuntimeError::Planner(format!("RepoPrompt agent_run failed: {err}"))
            })?;

        let json = output.json.ok_or_else(|| {
            agent_runtime::RuntimeError::Planner(
                "RepoPrompt agent_run returned no JSON payload".to_string(),
            )
        })?;

        // Capture session_id on first turn for subsequent steers.
        if self.session_id.is_none()
            && let Some(id) = json.get("session_id").and_then(|v| v.as_str())
        {
            self.session_id = Some(id.to_string());
        }

        // Status-driven error handling. `waiting_for_input` is the
        // tricky case — the RP agent wants a user response we don't
        // know how to provide synchronously. Treat it as fatal so the
        // user can re-run with a different role or backend.
        let status = json.get("status").and_then(|v| v.as_str()).unwrap_or("");
        match status {
            "completed" => {}
            "failed" => {
                let msg = json
                    .get("status_text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(no status_text)");
                return Err(agent_runtime::RuntimeError::PlannerFatal(format!(
                    "RepoPrompt agent_run failed: {msg}"
                )));
            }
            "waiting_for_input" => {
                return Err(agent_runtime::RuntimeError::PlannerFatal(format!(
                    "RepoPrompt agent_run paused waiting_for_input — not supported by the planner loop. \
                     Use --provider codex or repoprompt (oracle) for tasks needing interactive approval. \
                     interaction_id={}",
                    json.get("interaction_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                )));
            }
            other => {
                // Include the full response so the user can diagnose
                // (RP sometimes returns error-shaped objects without a
                // `status` field).
                let preview = truncate_for_error(&json);
                return Err(agent_runtime::RuntimeError::Planner(format!(
                    "RepoPrompt agent_run returned unexpected status={:?}; response={preview}",
                    other
                )));
            }
        }

        let assistant_text = json
            .get("assistant_text")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if assistant_text.trim().is_empty() {
            let preview = truncate_for_error(&json);
            return Err(agent_runtime::RuntimeError::Planner(format!(
                "RepoPrompt agent_run returned empty assistant_text; response={preview}"
            )));
        }
        let action = agent_runtime::parse_planned_action(assistant_text)?;
        Ok(PlanOutput {
            action,
            response_chars: assistant_text.chars().count(),
            prompt_chars,
        })
    }
}

/// walk a JSON tree looking for the first `window_id`
/// (snake_case) or `windowID` (camelCase) integer field. RepoPrompt's
/// bind_context response nests it variously depending on whether the
/// workspace was pre-existing vs newly created. Returns `None` if not
/// found.
fn find_window_id(value: &serde_json::Value) -> Option<u32> {
    match value {
        serde_json::Value::Object(map) => {
            for key in &["window_id", "windowID"] {
                if let Some(v) = map.get(*key).and_then(|v| v.as_u64()) {
                    return Some(v as u32);
                }
            }
            for child in map.values() {
                if let Some(v) = find_window_id(child) {
                    return Some(v);
                }
            }
            None
        }
        serde_json::Value::Array(arr) => arr.iter().find_map(find_window_id),
        _ => None,
    }
}

/// Helper: render a `serde_json::Value` as a short string for error
/// surfaces — caps at 400 chars so the user can see what RP actually
/// returned without flooding the trace.
fn truncate_for_error(value: &serde_json::Value) -> String {
    let s = serde_json::to_string(value).unwrap_or_else(|_| "<unrenderable>".to_string());
    if s.chars().count() <= 400 {
        s
    } else {
        let head: String = s.chars().take(400).collect();
        format!("{head}… (truncated, full len {} chars)", s.chars().count())
    }
}

/// Planner-loop backend for `--provider codex`.
///
/// borrows `&'a mut CodexAppServerClient` from the REPL-lifetime
/// `CodexSession`. The session's `ensure()` decides whether to reuse the
/// existing subprocess or restart based on launch fingerprint, so a chain
/// of REPL goals on `--provider codex` shares one `codex app-server`
/// across goals instead of paying ~300ms startup per goal. The lifetime
/// `'a` is the borrow on `CodexSession`; `Box<dyn Planner + 'a>` carries
/// it through `build_planner`'s return type.
pub(crate) struct CodexPlanner<'a> {
    client: &'a mut CodexAppServerClient,
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
        on_progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PlanOutput, agent_runtime::RuntimeError> {
        let prompt =
            agent_runtime::planner_prompt_with_state_and_memory(goal, tool_infos, state, memory);
        let prompt_chars = prompt.chars().count();
        let delta_chars: Cell<usize> = Cell::new(0);
        // streaming callback emits ProgressEvent::StreamingTokens
        // instead of touching the spinner directly. The driver decides
        // how to render it (currently: "(N tokens streamed)" subtitle).
        // Note: we can't simply pass `on_progress` to `run_prompt_streaming`
        // because the cdp callback signature takes &str. Wrap it.
        let result = self
            .client
            .run_prompt_streaming(&prompt, |delta| {
                delta_chars.set(delta_chars.get() + delta.chars().count());
                on_progress(ProgressEvent::StreamingTokens(delta_chars.get()));
            })
            .map_err(|err| {
                agent_runtime::RuntimeError::Planner(format!("Codex planner failed: {err}"))
            })?;
        let action = agent_runtime::parse_planned_action(&result.text)?;
        Ok(PlanOutput {
            action,
            response_chars: delta_chars.get(),
            prompt_chars,
        })
    }
}

pub(crate) struct HttpPlanner {
    provider_id: String,
    model: String,
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
        on_progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PlanOutput, agent_runtime::RuntimeError> {
        // stream HTTP responses so the user sees live token counts.
        // route the token deltas through ProgressEvent
        // instead of touching the spinner directly.
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
        // Estimate prompt size by summing message content lengths.
        // Approximate (ignores per-message JSON envelope overhead) but a
        // good proxy for "how full is my context?".
        let prompt_chars: usize = request
            .messages
            .iter()
            .map(|m| m.content.chars().count())
            .sum();
        let delta_chars: Cell<usize> = Cell::new(0);
        let response = agent_llm::ProviderClient::new()
            .chat_streaming(provider, request, |delta| {
                delta_chars.set(delta_chars.get() + delta.chars().count());
                on_progress(ProgressEvent::StreamingTokens(delta_chars.get()));
            })
            .map_err(|err| {
                agent_runtime::RuntimeError::Planner(format!("HTTP planner failed: {err}"))
            })?;
        let action = agent_runtime::parse_planned_action(&response.text)?;
        Ok(PlanOutput {
            action,
            response_chars: delta_chars.get(),
            prompt_chars,
        })
    }
}

/// concrete enum that implements `Planner` itself via match
/// dispatch, replacing the `Box<dyn Planner + 'a>` returned by
/// `build_planner`. The `'a` lifetime is still required because
/// `CodexPlanner<'a>` borrows `&'a mut CodexAppServerClient` from the
/// REPL-lifetime `CodexSession`; the enum just removes the heap
/// allocation and the indirection layer.
///
/// Tests (`MockPlanner`) still use the trait directly with
/// `&mut dyn Planner` — the trait isn't dead, it's just no longer
/// boxed on the production path.
pub(crate) enum PlannerKind<'a> {
    Oracle(OraclePlanner),
    // full RepoPrompt Agent Mode backend.
    RepoPromptAgent(RepoPromptAgentPlanner),
    Codex(CodexPlanner<'a>),
    Http(HttpPlanner),
}

impl<'a> Planner for PlannerKind<'a> {
    fn label(&self) -> &'static str {
        match self {
            PlannerKind::Oracle(p) => p.label(),
            PlannerKind::RepoPromptAgent(p) => p.label(),
            PlannerKind::Codex(p) => p.label(),
            PlannerKind::Http(p) => p.label(),
        }
    }

    fn turn_start_subtitle(&self) -> Option<String> {
        match self {
            PlannerKind::Oracle(p) => p.turn_start_subtitle(),
            PlannerKind::RepoPromptAgent(p) => p.turn_start_subtitle(),
            PlannerKind::Codex(p) => p.turn_start_subtitle(),
            PlannerKind::Http(p) => p.turn_start_subtitle(),
        }
    }

    fn plan(
        &mut self,
        goal: &str,
        tool_infos: &[ToolInfo],
        state: &agent_runtime::AgentLoopState,
        memory: &agent_runtime::PlannerMemoryContext,
        on_progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PlanOutput, agent_runtime::RuntimeError> {
        match self {
            PlannerKind::Oracle(p) => p.plan(goal, tool_infos, state, memory, on_progress),
            PlannerKind::RepoPromptAgent(p) => p.plan(goal, tool_infos, state, memory, on_progress),
            PlannerKind::Codex(p) => p.plan(goal, tool_infos, state, memory, on_progress),
            PlannerKind::Http(p) => p.plan(goal, tool_infos, state, memory, on_progress),
        }
    }
}

/// Build the concrete planner. Consumes the provider-specific fields out of
/// the ProviderSpec match. Returns a concrete `PlannerKind<'a>` enum
/// rather than a boxed trait object — the production path no
/// longer pays for the `Box` allocation or the vtable indirection.
//
// Genuinely 9 distinct knobs (Codex needs all of them; Oracle/Http use
// a subset). Bundling into a ProviderSpec struct would force re-cloning
// in run_goal; the explicit list is clearer here.
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
) -> Result<PlannerKind<'a>> {
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
            Ok(PlannerKind::Oracle(OraclePlanner {
                oracle,
                mode,
                chat_id: None,
            }))
        }
        PlannerProvider::RepoPromptAgent => {
            // bind FIRST with a minimal config to discover the
            // window_id, THEN build the long-lived client with that
            // window pinned. Each `repoprompt_cli` invocation is a
            // separate subprocess — without `--window <id>` on every
            // subsequent call, RP can't disambiguate "which open
            // RepoPrompt window owns this workspace" and rejects
            // `agent_run` with "Multiple RepoPrompt windows detected".
            //
            // This mirrors `agent-tools::resolve_repoprompt_window`
            // (which lives in agent-tools as `pub(crate)`, so we
            // can't reuse it from agent-cli — pattern is inlined).
            let bind_cfg = agent_repoprompt::RepoPromptClientConfig {
                cli_path: agent_repoprompt::default_cli_path(),
                timeout_secs: turn_timeout_secs.max(60),
                raw_json: true,
                ..Default::default()
            };
            let bind_client = agent_repoprompt::RepoPromptClient::new(bind_cfg);
            bind_client.check_available().map_err(|err| {
                anyhow::anyhow!(
                    "RepoPrompt agent unavailable: {err}. Re-run with --provider codex to bypass."
                )
            })?;
            let bind_output = bind_client
                .call_tool(
                    agent_repoprompt::RepoPromptTool::BindContext,
                    &serde_json::json!({
                        "op": "bind",
                        "working_dirs": [cwd.display().to_string()],
                        "create_if_missing": true,
                    }),
                )
                .map_err(|err| {
                    anyhow::anyhow!(
                        "RepoPrompt agent bind_context failed: {err}. \
                         Ensure RepoPrompt is running and the workspace is registered."
                    )
                })?;
            if bind_output.timed_out || bind_output.exit_code != Some(0) {
                anyhow::bail!(
                    "RepoPrompt bind_context returned non-zero exit: {}",
                    bind_output.stderr.trim()
                );
            }
            // Extract window_id from the bind response so subsequent
            // calls route to the right window.
            let window_id = bind_output
                .json
                .as_ref()
                .and_then(|j| find_window_id(j))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "RepoPrompt bind_context succeeded but did not return a window_id; \
                         response: {}",
                        bind_output
                            .json
                            .as_ref()
                            .map(|j| j.to_string())
                            .unwrap_or_default()
                    )
                })?;
            // Now construct the long-lived agent client with both
            // working_dirs AND window_id pinned. All `start` / `steer`
            // calls will pass `--window <window_id>` and bypass the
            // disambiguation prompt.
            let agent_cfg = agent_repoprompt::RepoPromptClientConfig {
                cli_path: agent_repoprompt::default_cli_path(),
                timeout_secs: turn_timeout_secs.max(60),
                raw_json: true,
                working_dirs: vec![cwd.to_path_buf()],
                window_id: Some(window_id),
                ..Default::default()
            };
            let client = agent_repoprompt::RepoPromptClient::new(agent_cfg);
            let role = model.unwrap_or_else(|| "pair".to_string());
            Ok(PlannerKind::RepoPromptAgent(RepoPromptAgentPlanner {
                client,
                role,
                session_id: None,
                timeout_secs: turn_timeout_secs.max(60),
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
            // borrow the live client from the REPL-lifetime session.
            // If the launch fingerprint matches a previously-cached client,
            // we reuse the subprocess (no spawn, ~0ms). Otherwise ensure()
            // drops the old one and spawns fresh.
            let client = codex_session.ensure(cfg)?;
            Ok(PlannerKind::Codex(CodexPlanner { client }))
        }
        PlannerProvider::Http(provider_id) => {
            let model = model.unwrap_or_else(|| "gpt-5.1".to_string());
            #[allow(clippy::redundant_field_names)]
            Ok(PlannerKind::Http(HttpPlanner {
                provider_id: provider_id.clone(),
                model,
            }))
        }
    }
}
