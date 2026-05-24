use agent_core::{ToolCall, ToolInfo, ToolResult};
use agent_llm::{ChatMessage, ChatRequest, ModelId, ProviderClient, ProviderId};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum PlannedAction {
    Tool {
        #[serde(default)]
        summary: Option<String>,
        tool_name: String,
        args: Value,
    },
    Finish {
        #[serde(default)]
        summary: Option<String>,
        answer: String,
    },
}

impl PlannedAction {
    pub fn into_tool_call(self) -> Option<ToolCall> {
        match self {
            PlannedAction::Tool {
                tool_name, args, ..
            } => Some(ToolCall::new(tool_name, args)),
            PlannedAction::Finish { .. } => None,
        }
    }

    pub fn summary(&self) -> Option<&str> {
        match self {
            PlannedAction::Tool { summary, .. } | PlannedAction::Finish { summary, .. } => summary
                .as_deref()
                .map(str::trim)
                .filter(|text| !text.is_empty()),
        }
    }

    /// Borrowed from GenericAgent's `do_no_tool` interception: catch obviously
    /// broken actions BEFORE the runtime dispatches them as a turn. Anything
    /// returned here gets converted into `InvalidPlannerJson` so the existing
    /// parse-retry path nudges the planner with the specific reason on the
    /// next turn — much cheaper than running the tool and observing failure.
    ///
    /// Intentionally narrow: we only catch the unambiguous cases (empty
    /// tool_name, non-object args, blank Finish answer, obvious placeholder
    /// markers). Per-tool argument validation is the tool's responsibility,
    /// not ours.
    fn sanity_check(&self) -> Result<(), String> {
        match self {
            PlannedAction::Tool {
                tool_name, args, ..
            } => {
                let trimmed = tool_name.trim();
                if trimmed.is_empty() {
                    return Err("`tool_name` is empty — pick a concrete tool from the tools list".to_string());
                }
                if matches!(
                    trimmed.to_ascii_lowercase().as_str(),
                    "todo" | "tbd" | "placeholder" | "..."
                ) {
                    return Err(format!(
                        "`tool_name` looks like a placeholder ({trimmed:?}) — pick a real tool"
                    ));
                }
                if args.is_null() {
                    return Err(format!(
                        "`args` is null for tool `{trimmed}` — pass `{{}}` for zero-arg tools or an object literal otherwise"
                    ));
                }
                if !args.is_object() {
                    return Err(format!(
                        "`args` for tool `{trimmed}` must be a JSON object; got: {args}"
                    ));
                }
                Ok(())
            }
            PlannedAction::Finish { answer, .. } => {
                if answer.trim().is_empty() {
                    return Err("`finish.answer` is blank — write the actual answer here, even if short".to_string());
                }
                Ok(())
            }
        }
    }
}

/// Borrowed from forge's `Error::Retryable(_)` pattern: planner-side
/// failures fall into two buckets — *transient* (network blip, 5xx, stdio
/// hiccup) which the loop will back off and retry, and *fatal* (model
/// returned a non-success response, auth was rejected, user-driven abort)
/// which should fail-fast so we don't burn turn budget on a known-bad
/// state. Planner impls choose the right variant; the runtime's retry
/// logic only re-arms on `Planner(_)` and `InvalidPlannerJson(_)`.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// Planner returned text that didn't parse as a `PlannedAction`. Retried
    /// with a JSON-schema reminder injected as a `recovery_hint`.
    #[error("planner response was not valid action JSON: {0}")]
    InvalidPlannerJson(String),

    /// Transient planner failure (network/transport/stdio). Retried with
    /// exponential backoff up to `max_transport_retries`. Use for "the
    /// request didn't complete" errors where retrying might succeed.
    #[error("planner failed (transient): {0}")]
    Planner(String),

    /// Fatal planner failure. The runtime does NOT retry — the loop
    /// unwinds immediately and the caller surfaces the message. Use for
    /// "the request completed but the response says no" (4xx, auth
    /// rejection) or "the runtime itself decided to give up" (loop abort
    /// via ToolErrorTracker).
    #[error("planner failed (fatal): {0}")]
    PlannerFatal(String),

    #[error(transparent)]
    Llm(#[from] agent_llm::LlmError),
}

impl RuntimeError {
    /// Forge-style convenience: wrap any string into a fatal planner
    /// error. Useful at call sites that want the explicit form rather
    /// than `RuntimeError::PlannerFatal(format!(...))`.
    pub fn planner_fatal(message: impl Into<String>) -> Self {
        Self::PlannerFatal(message.into())
    }
}

#[derive(Debug, Clone)]
pub struct AgentLoopConfig {
    pub provider_id: String,
    pub model: ModelId,
    pub max_turns: usize,
}

impl AgentLoopConfig {
    pub fn new(provider_id: impl Into<String>, model: impl Into<ModelId>) -> Self {
        Self {
            provider_id: provider_id.into(),
            model: model.into(),
            max_turns: 16,
        }
    }
}

pub fn is_read_only_analysis_goal(goal: &str) -> bool {
    let lower = goal.to_lowercase();
    let analysis_terms = [
        "analyze",
        "analyse",
        "analysis",
        "summarize",
        "summary",
        "explain",
        "investigate",
        "understand",
        "inspect",
        "describe",
        "review",
        "分析",
        "总结",
        "解释",
        "了解",
        "梳理",
        "调查",
        "看看",
        "审查",
    ];
    let implementation_terms = [
        "implement",
        "fix",
        "change",
        "modify",
        "edit",
        "write",
        "add",
        "remove",
        "delete",
        "update",
        "refactor",
        "optimize",
        "install",
        "实现",
        "修复",
        "修改",
        "改",
        "写",
        "添加",
        "删除",
        "更新",
        "重构",
        "优化",
        "安装",
    ];

    analysis_terms.iter().any(|term| lower.contains(term))
        && !implementation_terms.iter().any(|term| lower.contains(term))
}

/// RF27-1: typed accessor for the goal classifier. `is_read_only_analysis_goal`
/// is kept as the underlying primitive so the keyword tables stay in one
/// place; this wraps the boolean into the public `RunMode` enum so callers
/// (the CLI, `run_goal`, the AgentEvent emitter) can hand the result
/// around without reasoning about polarity each time.
pub fn classify_run_mode(goal: &str) -> agent_core::RunMode {
    if is_read_only_analysis_goal(goal) {
        agent_core::RunMode::ReadOnly
    } else {
        agent_core::RunMode::Implementation
    }
}

/// `is_deep_analysis_goal` distinguishes "summarize this repo" (light, 2-3
/// reads suffice) from "deeply analyze this project" (the user expects
/// substantive findings beyond README paraphrase). Deep goals get a larger
/// exploration budget and stronger guidance to read actual code logic.
pub fn is_deep_analysis_goal(goal: &str) -> bool {
    if !is_read_only_analysis_goal(goal) {
        return false;
    }
    let lower = goal.to_lowercase();
    let deep_terms = [
        "深入",
        "深度",
        "全面",
        "彻底",
        "仔细",
        "详细",
        "deep",
        "deeply",
        "comprehensive",
        "comprehensively",
        "in depth",
        "thorough",
        "thoroughly",
    ];
    deep_terms.iter().any(|term| lower.contains(term))
}

/// Count consecutive read/search/structure-only tool calls at the tail of the
/// observation list. Resets to 0 the moment we see a synthesis step
/// (`update_working_checkpoint`, a plan_* call, a write/patch tool, or a
/// shell command that mutates).
fn exploration_streak_len(observations: &[ToolObservation]) -> usize {
    observations
        .iter()
        .rev()
        .take_while(|obs| {
            matches!(
                obs.call.name.as_str(),
                "read_file"
                    | "read_files"
                    | "memory_search"
                    | "memory_fetch"
                    | "skill_search"
                    | "skill_list"
                    | "skill_fetch"
                    | "repoprompt_call"
                    | "repoprompt_exec"
                    | "repoprompt_tools"
            ) || (obs.call.name == "run_shell" && !mutating_shell_command(&obs.call.args))
        })
        .count()
}

/// Heuristic: a shell call counts as exploration (not mutation) when its
/// command does not start with a typical write verb. Conservative on the side
/// of "treat as mutation" so the exploration streak guard doesn't accidentally
/// fire after a real `cargo build`.
fn mutating_shell_command(args: &Value) -> bool {
    let Some(cmd) = args
        .get("command")
        .or_else(|| args.get("cmd"))
        .and_then(Value::as_str)
    else {
        return false;
    };
    let head = cmd.split_whitespace().next().unwrap_or("");
    matches!(
        head,
        "rm" | "mv" | "cp"
            | "touch"
            | "mkdir"
            | "sed"
            | "tee"
            | "cargo"
            | "npm"
            | "yarn"
            | "pnpm"
            | "git"
            | "make"
            | "pip"
            | "brew"
    )
}

const DEEP_ANALYSIS_GUIDANCE: &str = "Goal mode: DEEP read-only analysis. The user used words like `深入`/`全面`/`comprehensive` — they want findings beyond README paraphrase. \
Do not create a durable plan, do not mutate files or memory.\n\
\n\
DEEP ANALYSIS BUDGET: up to 8 evidence-gathering tool calls (vs 5 for light analysis). Use them wisely:\n\
- 1-2 calls for repo shape (tree / Cargo.toml / README) — same as light mode\n\
- 3-5 calls for ACTUAL CODE LOGIC — read the largest module's body (not just first 200 lines), use `read_files` to look at 3+ entry points at once, run `file_search` for non-obvious patterns (TODO/FIXME/unwrap/panic/recent struct names)\n\
- 1-2 calls for VERIFICATION — `cargo metadata --no-deps`, `git log --oneline -n 30` (via run_shell) to see recent direction\n\
- 1 call optional for cross-checking a suspicion (e.g., 'is this struct also in crate B?')\n\
\n\
QUALITY BAR: a 'deep' analysis answer is unacceptable if its Evidence section only paraphrases README/config/file tree. \
At least 1 Evidence bullet MUST cite something you actually read in the code (function name, struct name, test count, file size, surprising dependency, etc.). \
If after exploration you have nothing beyond surface-level facts, say so explicitly in the answer rather than padding with restated README content.\n\
\n\
FINISH ANSWER SCHEMA (mandatory, same as light analysis):\n\
\n\
Bottom line: <one sentence, ≤30 words. The most useful sentence the user could read. Must contain a concrete, non-trivial finding — not a goal restatement or README paraphrase.>\n\
\n\
Evidence:\n\
- [via <tool_name>] <observation NAMING a specific code element: file:line, function, struct, test, dependency, line count, etc. NOT a README paraphrase.>\n\
- ... (2-5 items max. Each MUST start with `[via <tool>]` AND name something concrete.)\n\
\n\
Counter-intuitive / risk:\n\
- <findings about the PROJECT — code smell, recent change direction, missing test, hidden dependency, weak boundary. NOT about your own runtime mode, NOT about goal restrictions.>\n\
- ... (1-3 items, OR `- (none — surface-level only)` if you genuinely found none. Do not pad.)\n\
\n\
Action items (ROI-sorted, 1-4 items):\n\
1. <imperative verb> <specific target: file:line, `cargo test -p X foo`, struct/fn name>. Expected impact: <one phrase>.\n\
2. ...\n\
\n\
Explicitly NOT doing:\n\
- <action you considered and rejected, with reason>. (Skip section if none.)\n\
\n\
FORBIDDEN OPENERS — same banlist as light analysis: 整体来看 / 当前... 是 / 高价值优化点应优先 / 如果想要继续 / 建议下一步 / This project is / Overall, / In summary,\n\
FORBIDDEN CONTENT in Counter-intuitive: meta-observations about your tool mode or about other active plans/sessions. That section is for findings ABOUT THE CODE / PROJECT, not about your own constraints.\n\
SPECIFICITY RULE: every Action item must name a concrete target (path, command, function name). Vague items like '改进 X' / '加强 Y' are rejected.\n";

fn planner_goal_guidance(goal: &str) -> &'static str {
    if is_deep_analysis_goal(goal) {
        return DEEP_ANALYSIS_GUIDANCE;
    }
    if is_read_only_analysis_goal(goal) {
        "Goal mode: read-only analysis/investigation. Do not create a durable plan, do not call plan_* tools, do not mutate files or memory, and do not run a verification gate. \
        BUDGET: cap evidence-gathering at ~5 read/search/structure tool calls. After that, finish unless a contradiction in observations forces one more lookup. \
        Reading the same file twice with different windows, or running the same search with broader parameters, counts toward the budget. \
        For listing/summary questions (file tree, crate names, README highlights), 2-3 read calls plus a finish is usually enough — do not enumerate exhaustively when the question doesn't require it. \
        Prefer batched `read_files` over consecutive `read_file` turns whenever you already know which paths matter.\n\
        \n\
        FINISH ANSWER SCHEMA (mandatory for read-only goals). Use EXACTLY these sections in this order, in the same language as the goal:\n\
        \n\
        Bottom line: <one sentence, ≤30 words. The single most useful sentence the user could read. No preamble.>\n\
        \n\
        Evidence:\n\
        - [via <tool_name>] <observation that is NOT just a paraphrase of the goal or README; include the file:line or value you actually saw>\n\
        - ... (2-4 items max. Each MUST start with `[via <tool>]` citing the tool call that produced it.)\n\
        \n\
        Counter-intuitive / risk:\n\
        - <one or two findings that are NOT obvious from the goal — something the user probably did not expect, OR an explicit risk in the code/system. If you genuinely found nothing surprising, write a single line `- (none — surface-level analysis only)` and do not pad.>\n\
        \n\
        Action items (ROI-sorted, 0-3 items, each ≤1 line):\n\
        1. <imperative verb> <specific target: file:line, `cargo test -p X test_name`, or a named function/struct>. Expected impact: <one phrase>.\n\
        2. ...\n\
        \n\
        Explicitly NOT doing:\n\
        - <action you considered and rejected> — <reason>. (Skip this section entirely if there are no rejected alternatives worth naming.)\n\
        \n\
        FORBIDDEN OPENERS in the final answer (rewrite if you catch yourself starting with these):\n\
        - 整体来看 / 整体上 / 总的来说\n\
        - 当前(项目|代码|实现) X 是\n\
        - 高价值优化点应优先\n\
        - 如果想要继续 / 若继续做优化\n\
        - 建议下一步\n\
        - This project is / Overall, / In summary,\n\
        \n\
        SPECIFICITY RULE: every Action item must name a concrete target (path, command, function name). \
        Rewrite any item that reads like \"考虑改进 X\" / \"加强 Y\" / \"补强 Z\" — replace with `crate-name/path:line` or `cargo test -p crate name`. \
        If you cannot point at a specific target, the item does not belong in the answer.\n"
    } else {
        "Goal mode: implementation or action. Use durable plans for multi-step implementation work when they reduce risk.\n"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolObservation {
    pub turn: usize,
    pub summary: String,
    pub call: ToolCall,
    pub result: ToolResult,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkingMemory {
    pub current_turn: usize,
    pub history: Vec<String>,
    pub key_info: Vec<String>,
    pub related_skills: Vec<String>,
    pub guard_hints: Vec<String>,
    pub long_term_update: Option<LongTermUpdateState>,
    /// Compacted "earlier than the rolling history window" turn summaries.
    /// When `history` overflows its 30-entry cap, the drained entries get
    /// folded into one short line per overflow event (turn range + first/last
    /// exemplars) and pushed here. Borrowed from GenericAgent's `_fold_earlier`
    /// pattern — long runs keep some signal of early-turn context instead of
    /// dropping it entirely. `serde(default)` keeps replay of pre-fold
    /// sessions working.
    #[serde(default)]
    pub earlier_summary: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LongTermUpdateState {
    pub reason: String,
    pub evidence: Option<String>,
    pub sop_path: Option<String>,
    pub next_prompt: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlannerMemoryContext {
    pub content: String,
}

impl PlannerMemoryContext {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    fn is_empty(&self) -> bool {
        self.content.trim().is_empty()
    }
}

impl WorkingMemory {
    pub fn from_observations(
        observations: &[ToolObservation],
        current_turn: usize,
        max_turns: usize,
    ) -> Self {
        let mut memory = Self {
            current_turn,
            ..Self::default()
        };
        for observation in observations {
            memory.record_summary(observation.turn, &observation.summary);
            memory.observe_tool_result(&observation.call, &observation.result);
        }
        memory.refresh_guard_hints(observations, max_turns);
        memory
    }

    fn record_summary(&mut self, turn: usize, summary: &str) {
        if summary.trim().is_empty() {
            return;
        }
        push_history_with_fold(
            &mut self.history,
            &mut self.earlier_summary,
            format!("turn {turn}: {}", compact_text(summary, 180)),
            30,
        );
    }

    fn observe_tool_result(&mut self, call: &ToolCall, result: &ToolResult) {
        if !result.ok {
            return;
        }
        if call.name == "start_long_term_update" {
            self.observe_long_term_update(result);
            return;
        }
        if call.name == "complete_long_term_update" {
            self.observe_complete_long_term_update(result);
            return;
        }
        if call.name != "update_working_checkpoint" {
            return;
        }
        if let Some(key_info) = result
            .content
            .get("key_info")
            .and_then(|value| value.as_str())
        {
            push_limited_unique(&mut self.key_info, compact_text(key_info, 260), 20);
        }
        if let Some(skill) = result
            .content
            .get("related_skill")
            .and_then(|value| value.as_str())
        {
            push_limited_unique(&mut self.related_skills, compact_text(skill, 120), 20);
        }
    }

    fn observe_long_term_update(&mut self, result: &ToolResult) {
        let reason = result
            .content
            .get("reason")
            .and_then(|value| value.as_str())
            .unwrap_or("long-term memory update")
            .to_string();
        let evidence = result
            .content
            .get("evidence")
            .and_then(|value| value.as_str())
            .map(|value| compact_text(value, 400));
        let sop_path = result
            .content
            .get("sop_path")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let next_prompt = result
            .content
            .get("next_prompt")
            .and_then(|value| value.as_str())
            .unwrap_or(LONG_TERM_SETTLEMENT_PROMPT)
            .to_string();
        self.long_term_update = Some(LongTermUpdateState {
            reason: compact_text(&reason, 260),
            evidence,
            sop_path,
            next_prompt: compact_text(&next_prompt, 1_200),
        });
    }

    fn observe_complete_long_term_update(&mut self, result: &ToolResult) {
        let decision = result
            .content
            .get("decision")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown");
        let changed = result
            .content
            .get("changed")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let reason = result
            .content
            .get("reason")
            .and_then(|value| value.as_str())
            .unwrap_or("long-term update settled");
        let target = result
            .content
            .get("target")
            .and_then(|value| value.as_str());
        let mut info = format!(
            "long-term memory settled: decision={decision}; changed={changed}; reason={reason}"
        );
        if let Some(target) = target {
            info.push_str(&format!("; target={target}"));
        }
        push_limited_unique(&mut self.key_info, compact_text(&info, 320), 20);
        self.long_term_update = None;
    }

    fn refresh_guard_hints(&mut self, observations: &[ToolObservation], max_turns: usize) {
        self.guard_hints.clear();

        if consecutive_failures(observations) >= 2 {
            self.guard_hints.push(
                "Two tool calls in a row failed; switch strategy, read the current state again, or ask the user instead of retrying blindly."
                    .to_string(),
            );
        }

        if repeated_last_call(observations) {
            self.guard_hints.push(
                "The same tool call repeated without progress; change arguments or choose a different tool."
                    .to_string(),
            );
        }

        if self.current_turn > 1 && self.current_turn.is_multiple_of(7) {
            self.guard_hints.push(
                "Turn guard (every 7): summarize the current situation, update_working_checkpoint for verified context if useful, then change strategy if progress is weak."
                    .to_string(),
            );
        }

        // Distillation nudge: if the agent has been running a few turns but has
        // not crystallized any key facts via update_working_checkpoint, each
        // turn is re-deriving "what do I already know" from raw observations.
        // Push an explicit reminder so the next turn turns evidence into a
        // single short anchor line.
        if self.current_turn >= 3
            && self.current_turn.is_multiple_of(3)
            && self.key_info.len() < 3
            && !observations.is_empty()
        {
            self.guard_hints.push(
                "Checkpoint nudge: you have run several turns with few <key_info> anchors. \
                If the recent tool results contain a verified, reusable fact (a path that exists, \
                a config value confirmed, a working command), call `update_working_checkpoint` \
                with one short key_info line so later turns don't re-derive it from raw observations."
                    .to_string(),
            );
        }

        // Stop-exploring nudge: count consecutive read/search/structure calls
        // (no synthesis, no checkpoint, no finish). After 5+ in a row, the
        // marginal value of one more lookup is usually less than just finishing
        // with the evidence already in hand. Soft hint here; the
        // `AgentLoopState::prepare_turn` escalates to a hard runtime override
        // once the streak passes 7.
        let exploration_streak = exploration_streak_len(observations);
        if exploration_streak >= 5 {
            self.guard_hints.push(format!(
                "Exploration streak guard: the last {exploration_streak} turns were all read/search/structure calls. \
                Unless an observation contradicts your current synthesis, the next turn should `finish` (or, if you need one more pass, batch it via `read_files` to avoid another solo lookup)."
            ));
        }

        if self.current_turn > 1 && self.current_turn.is_multiple_of(10) {
            self.guard_hints.push(
                "Memory refresh (every 10 turns): re-read L0 meta_rules and l1_insight via memory_fetch to recenter; drop any stale assumptions accumulated in working memory."
                    .to_string(),
            );
        }

        if max_turns > 5 && self.current_turn + 3 >= max_turns && self.current_turn + 1 < max_turns
        {
            self.guard_hints.push(
                "Budget warning: within 3 turns of max_turns. If the task is not converging, call ask_user with a focused question instead of burning the remaining budget."
                    .to_string(),
            );
        }

        if max_turns > 1 && self.current_turn >= max_turns {
            self.guard_hints.push(
                "Final turn guard: finish with the best verified answer, or explicitly state what is blocked."
                    .to_string(),
            );
        } else if max_turns > 2 && self.current_turn + 1 == max_turns {
            self.guard_hints.push(
                "Near max turns: prefer finishing if enough evidence exists; otherwise make one high-signal tool call."
                    .to_string(),
            );
        }
    }

    pub fn render_anchor(&self) -> String {
        let mut out = format!(
            "### [WORKING MEMORY]\nCurrent turn: {}\n",
            self.current_turn
        );
        if !self.earlier_summary.is_empty() {
            out.push_str("<earlier_context>\n");
            out.push_str(&self.earlier_summary.join("\n"));
            out.push_str("\n</earlier_context>\n");
        }
        if !self.history.is_empty() {
            out.push_str("<history>\n");
            out.push_str(&self.history.join("\n"));
            out.push_str("\n</history>\n");
        }
        if !self.key_info.is_empty() {
            out.push_str("<key_info>\n");
            out.push_str(&self.key_info.join("\n"));
            out.push_str("\n</key_info>\n");
        }
        if !self.related_skills.is_empty() {
            out.push_str("<related_skills>\n");
            out.push_str(&self.related_skills.join("\n"));
            out.push_str("\n</related_skills>\n");
        }
        if !self.guard_hints.is_empty() {
            out.push_str("<guard_hints>\n");
            out.push_str(&self.guard_hints.join("\n"));
            out.push_str("\n</guard_hints>\n");
        }
        if let Some(update) = &self.long_term_update {
            out.push_str("<long_term_update>\n");
            out.push_str(&format!("reason: {}\n", update.reason));
            if let Some(evidence) = &update.evidence {
                out.push_str(&format!("evidence: {evidence}\n"));
            }
            if let Some(sop_path) = &update.sop_path {
                out.push_str(&format!("sop_path: {sop_path}\n"));
            }
            out.push_str(&update.next_prompt);
            out.push_str("\n</long_term_update>\n");
        }
        out
    }
}

const LONG_TERM_SETTLEMENT_PROMPT: &str = "LONG_TERM_MEMORY_SETTLEMENT: choose exactly one branch now. Branch A update_l2_global_facts: use only for stable environment facts, durable user preferences, paths, or configuration; first read/fetch memory/global_facts.md, then make the smallest patch/write. Branch B update_l3_skill: use for reusable workflows or troubleshooting patterns; first memory_search for an existing skill, then memory_fetch/read it, then patch the existing SKILL.md when appropriate. Branch C skip: if evidence is unverified, temporary, generic, duplicate, or not future-useful. After the write or skip decision, call complete_long_term_update with decision, target, reason, evidence, and changed. Never write secrets.";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLoopState {
    pub next_turn: usize,
    pub max_turns: usize,
    pub observations: Vec<ToolObservation>,
    pub working_memory: WorkingMemory,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery_hint: Option<String>,
}

impl AgentLoopState {
    pub fn new(max_turns: usize) -> Self {
        let max_turns = max_turns.max(1);
        Self {
            next_turn: 1,
            max_turns,
            observations: Vec::new(),
            working_memory: WorkingMemory::from_observations(&[], 1, max_turns),
            recovery_hint: None,
        }
    }

    pub fn from_observations(
        observations: &[ToolObservation],
        next_turn: usize,
        max_turns: usize,
    ) -> Self {
        Self {
            next_turn,
            max_turns,
            observations: observations.to_vec(),
            working_memory: WorkingMemory::from_observations(observations, next_turn, max_turns),
            recovery_hint: None,
        }
    }

    pub fn recovery_hint(&self) -> Option<&str> {
        self.recovery_hint.as_deref()
    }

    fn prepare_turn(&mut self, turn: usize) {
        self.next_turn = turn;
        self.working_memory.current_turn = turn;
        self.working_memory
            .refresh_guard_hints(&self.observations, self.max_turns);
        self.recovery_hint = None;
        // Escalate the soft "exploration streak" guard hint into a hard runtime
        // override once it has been ignored long enough. The soft hint fires
        // inside `refresh_guard_hints` at streak ≥ 5 and lives in
        // working_memory.guard_hints; the implementation planner ignores it
        // because its goal_guidance encourages action. At streak ≥ 7 we copy
        // the imperative form into `recovery_hint` so it lands as a prepended
        // system instruction (same channel as parse-retry guidance), which
        // the LLM does not get to opt out of.
        let exploration_streak = exploration_streak_len(&self.observations);
        if exploration_streak >= 7 {
            self.recovery_hint = Some(format!(
                "STOP exploring. The last {exploration_streak} turns were all read/search/structure calls without synthesis. \
                On THIS turn you must do exactly one of: (a) `finish` with a concise answer drawn from what you already have, \
                (b) call `update_working_checkpoint` if there's a verified fact worth anchoring before you finish, or \
                (c) take a concrete write action (`patch_file`/`write_file`/`apply_edits`/`spawn_subagent` with a write task) \
                that actually moves the goal forward. Another read/search call is not an option."
            ));
        }
    }

    fn record_turn_summary(&mut self, turn: usize, summary: &str) {
        self.working_memory.record_summary(turn, summary);
    }

    fn push_observation(&mut self, observation: ToolObservation) {
        self.working_memory
            .observe_tool_result(&observation.call, &observation.result);
        self.observations.push(observation);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnSummary {
    pub turn: usize,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLoopResult {
    pub status: AgentLoopStatus,
    pub summary: String,
    pub turns: usize,
    pub turn_summaries: Vec<TurnSummary>,
    pub observations: Vec<ToolObservation>,
    pub working_memory: WorkingMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentLoopStatus {
    Finished,
    MaxTurnsExceeded,
}

pub fn plan_one_tool_call(
    provider_id: &str,
    model: impl Into<ModelId>,
    goal: &str,
    tools: &[ToolInfo],
) -> Result<PlannedAction, RuntimeError> {
    let provider = agent_llm::find_provider(provider_id)
        .ok_or_else(|| agent_llm::LlmError::ProviderNotFound(provider_id.to_string()))?;
    let request = planner_request(model, goal, tools);
    let response = ProviderClient::new().chat(provider, request)?;
    parse_planned_action(&response.text)
}

pub fn run_agent_loop<F>(
    config: AgentLoopConfig,
    goal: &str,
    tools: &[ToolInfo],
    execute_tool: F,
) -> Result<AgentLoopResult, RuntimeError>
where
    F: FnMut(&ToolCall) -> ToolResult,
{
    let provider_id = config.provider_id.clone();
    let model = config.model.clone();
    run_agent_loop_with_state_planner(
        config.max_turns,
        |state| plan_next_action_with_state(&provider_id, model.clone(), goal, tools, state),
        execute_tool,
    )
}

pub fn run_agent_loop_with_planner<P, F>(
    max_turns: usize,
    mut plan_next: P,
    execute_tool: F,
) -> Result<AgentLoopResult, RuntimeError>
where
    P: FnMut(&[ToolObservation]) -> Result<PlannedAction, RuntimeError>,
    F: FnMut(&ToolCall) -> ToolResult,
{
    run_agent_loop_with_state_planner(
        max_turns,
        |state| plan_next(&state.observations),
        execute_tool,
    )
}

pub const DEFAULT_PLANNER_PARSE_RETRIES: usize = 1;
pub const DEFAULT_PLANNER_TRANSPORT_RETRIES: usize = 2;

pub fn run_agent_loop_with_state_planner<P, F>(
    max_turns: usize,
    plan_next: P,
    execute_tool: F,
) -> Result<AgentLoopResult, RuntimeError>
where
    P: FnMut(&mut AgentLoopState) -> Result<PlannedAction, RuntimeError>,
    F: FnMut(&ToolCall) -> ToolResult,
{
    run_agent_loop_with_state_planner_retries(
        max_turns,
        DEFAULT_PLANNER_PARSE_RETRIES,
        DEFAULT_PLANNER_TRANSPORT_RETRIES,
        plan_next,
        execute_tool,
        |_| {},
    )
}

/// RF34-2: same as `run_agent_loop_with_state_planner` but takes an
/// `on_retry` callback so the CLI can emit `AgentEvent::PlannerRetry`
/// and update the spinner. Existing callers who don't care about retry
/// observability stay on the original entry point.
pub fn run_agent_loop_with_state_planner_observed<P, F, R>(
    max_turns: usize,
    plan_next: P,
    execute_tool: F,
    on_retry: R,
) -> Result<AgentLoopResult, RuntimeError>
where
    P: FnMut(&mut AgentLoopState) -> Result<PlannedAction, RuntimeError>,
    F: FnMut(&ToolCall) -> ToolResult,
    R: FnMut(PlannerRetryInfo),
{
    run_agent_loop_with_state_planner_retries(
        max_turns,
        DEFAULT_PLANNER_PARSE_RETRIES,
        DEFAULT_PLANNER_TRANSPORT_RETRIES,
        plan_next,
        execute_tool,
        on_retry,
    )
}

/// RF34-2: information about one planner retry attempt, passed to the
/// `on_retry` callback so the caller can emit telemetry (AgentEvent,
/// spinner update, log line, etc.) without the runtime depending on any
/// specific output channel.
#[derive(Debug, Clone)]
pub struct PlannerRetryInfo {
    /// Turn number where the retry happened (1-based).
    pub turn: usize,
    /// 1-based attempt index. `attempt=1` is the first retry after the
    /// original call failed; `attempt=2` is the second retry; etc.
    pub attempt: usize,
    /// Total attempts the runtime will make for this `kind`. For parse
    /// retries this is `max_parse_retries`; for transport it's
    /// `max_transport_retries`.
    pub of: usize,
    /// Backoff in milliseconds applied before this retry. Parse retries
    /// don't sleep, so this is 0 for them; transport retries do
    /// exponential backoff.
    pub backoff_ms: u64,
    /// Which retry path fired.
    pub kind: PlannerRetryKind,
    /// Short error string captured at retry time. Best-effort — long
    /// errors are passed through unchanged (callers can truncate).
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannerRetryKind {
    /// `InvalidPlannerJson` — the planner returned malformed JSON. We
    /// nudge it with a recovery hint and let it retry immediately.
    Parse,
    /// `RuntimeError::Planner(_)` — transport/IO failure where retrying
    /// might succeed. Backoff applied.
    Transport,
}

impl PlannerRetryKind {
    fn as_str(self) -> &'static str {
        match self {
            PlannerRetryKind::Parse => "parse",
            PlannerRetryKind::Transport => "transport",
        }
    }
}

impl std::fmt::Display for PlannerRetryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

pub fn run_agent_loop_with_state_planner_retries<P, F, R>(
    max_turns: usize,
    max_parse_retries: usize,
    max_transport_retries: usize,
    mut plan_next: P,
    mut execute_tool: F,
    mut on_retry: R,
) -> Result<AgentLoopResult, RuntimeError>
where
    P: FnMut(&mut AgentLoopState) -> Result<PlannedAction, RuntimeError>,
    F: FnMut(&ToolCall) -> ToolResult,
    R: FnMut(PlannerRetryInfo),
{
    let max_turns = max_turns.max(1);
    let mut state = AgentLoopState::new(max_turns);
    let mut turn_summaries = Vec::new();

    for turn in 1..=max_turns {
        state.prepare_turn(turn);
        let mut parse_left = max_parse_retries;
        let mut transport_left = max_transport_retries;
        let action = loop {
            match plan_next(&mut state) {
                Ok(action) => break action,
                Err(RuntimeError::InvalidPlannerJson(err)) if parse_left > 0 => {
                    parse_left -= 1;
                    let attempt = max_parse_retries - parse_left;
                    on_retry(PlannerRetryInfo {
                        turn,
                        attempt,
                        of: max_parse_retries,
                        backoff_ms: 0,
                        kind: PlannerRetryKind::Parse,
                        reason: err.to_string(),
                    });
                    state.recovery_hint = Some(format!(
                        "Your previous response was rejected because it did not parse as a valid PlannedAction. \
                        Error: {err}\nReturn EXACTLY ONE JSON object with non-empty `summary` and `action: tool|finish`. \
                        Do not wrap it in markdown code fences. Do not include commentary before or after the object."
                    ));
                    continue;
                }
                Err(RuntimeError::Planner(err)) if transport_left > 0 => {
                    transport_left -= 1;
                    let attempt = max_transport_retries - transport_left;
                    let backoff = Duration::from_millis(
                        500_u64.saturating_mul(
                            1 << (max_transport_retries - transport_left - 1) as u64,
                        ),
                    )
                    .min(Duration::from_secs(8));
                    on_retry(PlannerRetryInfo {
                        turn,
                        attempt,
                        of: max_transport_retries,
                        backoff_ms: backoff.as_millis() as u64,
                        kind: PlannerRetryKind::Transport,
                        reason: err.to_string(),
                    });
                    state.recovery_hint = Some(format!(
                        "Previous planner call failed at the transport layer ({err}). \
                        The runtime backed off and is retrying once. Your next response should still \
                        be exactly one valid PlannedAction JSON object."
                    ));
                    std::thread::sleep(backoff);
                    continue;
                }
                Err(other) => return Err(other),
            }
        };
        state.recovery_hint = None;
        let turn_summary = action_summary(&action);
        state.record_turn_summary(turn, &turn_summary);
        turn_summaries.push(TurnSummary {
            turn,
            summary: turn_summary.clone(),
        });
        match action {
            PlannedAction::Finish { answer, .. } => {
                return Ok(AgentLoopResult {
                    status: AgentLoopStatus::Finished,
                    summary: answer,
                    turns: turn,
                    turn_summaries,
                    observations: state.observations,
                    working_memory: state.working_memory,
                });
            }
            PlannedAction::Tool {
                tool_name, args, ..
            } => {
                let call = ToolCall::new(tool_name, args);
                let result = execute_tool(&call);
                state.push_observation(ToolObservation {
                    turn,
                    summary: turn_summary,
                    call,
                    result,
                });
            }
        }
    }

    Ok(AgentLoopResult {
        status: AgentLoopStatus::MaxTurnsExceeded,
        summary: format!("Stopped after {max_turns} turns without a finish action."),
        turns: max_turns,
        turn_summaries,
        observations: state.observations,
        working_memory: state.working_memory,
    })
}

pub fn plan_next_action(
    provider_id: &str,
    model: impl Into<ModelId>,
    goal: &str,
    tools: &[ToolInfo],
    observations: &[ToolObservation],
) -> Result<PlannedAction, RuntimeError> {
    let state = AgentLoopState::from_observations(observations, observations.len() + 1, 0);
    plan_next_action_with_state(provider_id, model, goal, tools, &state)
}

pub fn plan_next_action_with_observations_and_memory(
    provider_id: &str,
    model: impl Into<ModelId>,
    goal: &str,
    tools: &[ToolInfo],
    observations: &[ToolObservation],
    memory_context: &PlannerMemoryContext,
) -> Result<PlannedAction, RuntimeError> {
    let state = AgentLoopState::from_observations(observations, observations.len() + 1, 0);
    plan_next_action_with_state_and_memory(provider_id, model, goal, tools, &state, memory_context)
}

pub fn plan_next_action_with_state(
    provider_id: &str,
    model: impl Into<ModelId>,
    goal: &str,
    tools: &[ToolInfo],
    state: &AgentLoopState,
) -> Result<PlannedAction, RuntimeError> {
    plan_next_action_with_state_and_memory(
        provider_id,
        model,
        goal,
        tools,
        state,
        &PlannerMemoryContext::default(),
    )
}

pub fn plan_next_action_with_state_and_memory(
    provider_id: &str,
    model: impl Into<ModelId>,
    goal: &str,
    tools: &[ToolInfo],
    state: &AgentLoopState,
    memory_context: &PlannerMemoryContext,
) -> Result<PlannedAction, RuntimeError> {
    let provider = agent_llm::find_provider(provider_id)
        .ok_or_else(|| agent_llm::LlmError::ProviderNotFound(provider_id.to_string()))?;
    let request = planner_request_with_state_and_memory(model, goal, tools, state, memory_context);
    let response = ProviderClient::new().chat(provider, request)?;
    parse_planned_action(&response.text)
}

pub fn planner_request(model: impl Into<ModelId>, goal: &str, tools: &[ToolInfo]) -> ChatRequest {
    planner_request_with_observations(model, goal, tools, &[])
}

pub fn planner_request_with_observations(
    model: impl Into<ModelId>,
    goal: &str,
    tools: &[ToolInfo],
    observations: &[ToolObservation],
) -> ChatRequest {
    let state = AgentLoopState::from_observations(observations, observations.len() + 1, 0);
    planner_request_with_state(model, goal, tools, &state)
}

pub fn planner_request_with_state(
    model: impl Into<ModelId>,
    goal: &str,
    tools: &[ToolInfo],
    state: &AgentLoopState,
) -> ChatRequest {
    planner_request_with_state_and_memory(
        model,
        goal,
        tools,
        state,
        &PlannerMemoryContext::default(),
    )
}

pub fn planner_request_with_state_and_memory(
    model: impl Into<ModelId>,
    goal: &str,
    tools: &[ToolInfo],
    state: &AgentLoopState,
    memory_context: &PlannerMemoryContext,
) -> ChatRequest {
    // RF33-3: per-turn tool catalog culling. The prompt currently ships
    // ~30 tools × 1-line descriptions every turn. After turn 1 most of
    // that is redundant repetition. We tier it:
    //
    //   turn 1     → full compact descriptions (~90 chars/tool)
    //   turn 2-4   → compact descriptions (same as turn 1 — descriptions
    //                are still cheap insurance for "what does X do?")
    //   turn 5+    → names only + hint to call `tool_describe` if needed
    //
    // The threshold at turn 5 is empirical: most goals finish in <5
    // turns; once you're past that, the planner has either picked a
    // recipe or is stuck, and dropping descriptions saves ~3KB/turn of
    // input tokens that the prompt cache may or may not dedupe.
    let current_turn = state.working_memory.current_turn;
    let tools_text = if current_turn >= 5 {
        let mut lines: Vec<String> =
            tools.iter().map(|tool| format!("- {}", tool.name)).collect();
        lines.push(String::from(
            "  (descriptions elided — call `tool_describe {name: \"...\"}` if you need one)",
        ));
        lines.join("\n")
    } else {
        tools
            .iter()
            .map(|tool| {
                format!(
                    "- {}: {}",
                    tool.name,
                    compact_tool_description(&tool.description)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let routing = planner_tool_routing_block(tools);
    let system = format!(
        "You are the planner for a minimal local agent. Choose exactly one next action.\n\
Return only JSON, with no markdown.\n\
Every JSON object MUST include a non-empty `summary` (under 30 words) — a physical snapshot of new facts learned this turn and the immediate intent for the next. Responses without `summary` are rejected and you will be asked to retry.\n\
Use update_working_checkpoint for verified short-term context that should anchor later turns.\n\
{goal_guidance}\
For multi-step implementation, create or follow a durable plan with plan_create/plan_next/plan_complete. When a plan uses RepoPrompt, keep its orchestration ledger current: after any RepoPrompt builder/context/oracle export, call plan_record_artifact with the exported path; after any RepoPrompt agent_run or Codex delegate executes plan work, call plan_record_handoff with backend, role/run/thread id when known, artifact_path when used, status, and summary. When all non-verify plan items are complete, call plan_verify and do not finish until the independent verification gate returns PASS.\n\
Use start_long_term_update only when successful evidence should be distilled into durable memory; skip it for guesses or one-off facts.\n\
When WORKING MEMORY contains <long_term_update>, you are in phase 2 settlement. Choose exactly one branch: update L2 global facts, update an existing L3 skill, or skip with a reason. Read/fetch the target before patching or writing. For L3, always memory_search first, memory_fetch the existing skill, then patch that existing SKILL.md; do not create duplicate skills.\n\
After a phase 2 write or skip decision, call complete_long_term_update before finish so the settlement is auditable.\n\
For a tool call, return: {{\"summary\":\"...\",\"action\":\"tool\",\"tool_name\":\"read_file\",\"args\":{{...}}}}\n\
If no tool is needed, return: {{\"summary\":\"...\",\"action\":\"finish\",\"answer\":\"...\"}}\n\
{routing}\
Available tools:\n{tools_text}",
        goal_guidance = planner_goal_guidance(goal),
    );
    let mut user = format!("Goal:\n{goal}");
    if !memory_context.is_empty() {
        user.push_str("\n\n");
        user.push_str(&memory_context.content);
        user.push_str(
            "\n\nUse memory_search before memory_fetch when L1 suggests deeper L2/L3/L4 context may be relevant.",
        );
    }
    user.push_str("\n\n");
    user.push_str(&state.working_memory.render_anchor());
    if !state.observations.is_empty() {
        user.push_str("\n\nPrior tool observations:\n");
        user.push_str(&compact_observations(&state.observations));
        user.push_str(
            "\n\nContinue from these observations. Use another tool if needed, otherwise finish.",
        );
    }
    let mut request = ChatRequest::user(model, user);
    request.messages.insert(0, ChatMessage::system(system));
    if let Some(hint) = state.recovery_hint() {
        request.messages.insert(
            1,
            ChatMessage::system(format!("Runtime override for this turn: {hint}")),
        );
    }
    request.reasoning_effort = Some("minimal".to_string());
    request.max_output_tokens = Some(800);
    request
}

/// Emit a TOOL ROUTING block describing how the planner should choose between
/// overlapping tools (local vs RepoPrompt, sub-agent vs agent_run, etc.). Only
/// includes rules whose required tools are actually exposed this run.
/// Slim a tool description down to its first sentence (or 90 chars) so the
/// per-turn planner prompt does not ship ~3KB of duplicated tool docs. The
/// planner still has the full description available via `seed tool ...` and
/// the registry — this is purely for the LLM's per-turn context.
fn compact_tool_description(desc: &str) -> String {
    let cleaned = desc.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(first_sentence_end) = cleaned.find(". ") {
        let head = &cleaned[..first_sentence_end + 1];
        return head.trim().to_string();
    }
    if cleaned.chars().count() > 90 {
        let head: String = cleaned.chars().take(90).collect();
        return format!("{}…", head.trim());
    }
    cleaned
}

fn planner_tool_routing_block(tools: &[ToolInfo]) -> String {
    let has = |name: &str| tools.iter().any(|tool| tool.name == name);
    let mut rules: Vec<String> = Vec::new();
    if has("repoprompt_call") {
        rules.push(
            "- Cross-file code exploration: prefer `repoprompt_call` (file_search / get_code_structure / get_file_tree / read_file). Local `read_file` and `run_shell rg ...` are fallbacks only when RepoPrompt is unavailable.".to_string(),
        );
        if has("patch_file") || has("write_file") {
            rules.push(
                "- Multi-file edit or refactor: prefer `repoprompt_call apply_edits` (transactional). Local `patch_file`/`write_file` is for single, targeted file changes.".to_string(),
            );
        }
        if has("run_shell") {
            rules.push(
                "- Git inspection: prefer `repoprompt_call git`. Use `run_shell \"git ...\"` only as fallback when RepoPrompt is unreachable.".to_string(),
            );
        }
    }
    if has("spawn_subagent") && has("repoprompt_call") {
        rules.push(
            "- Delegating an isolated >3-file or >100-line sub-task: prefer `repoprompt_call agent_run` (RepoPrompt curates context). Use `spawn_subagent` ONLY when the sub-task needs seed's own plan/memory/skill tools or when RepoPrompt is unavailable.".to_string(),
        );
    } else if has("spawn_subagent") {
        rules.push(
            "- Delegating an isolated sub-task that needs seed's plan/memory/skill tools: use `spawn_subagent` with a tight task description and explicit context_files.".to_string(),
        );
    }
    if has("ask_user") {
        rules.push(
            "- Blocked on a decision or missing input: call `ask_user` with a focused question instead of guessing or burning the turn budget.".to_string(),
        );
    }
    if has("read_files") && has("read_file") {
        rules.push(
            "- Reading 2+ files whose paths you already know: prefer `read_files` (one batch call) over multiple sequential `read_file` turns. Each saved turn is one fewer planner round-trip.".to_string(),
        );
    }
    if rules.is_empty() {
        return String::new();
    }
    let mut block = String::from("### TOOL ROUTING (apply when both options are available)\n");
    for rule in rules {
        block.push_str(&rule);
        block.push('\n');
    }
    block
}

pub fn planner_prompt_with_observations(
    goal: &str,
    tools: &[ToolInfo],
    observations: &[ToolObservation],
) -> String {
    let state = AgentLoopState::from_observations(observations, observations.len() + 1, 0);
    planner_prompt_with_state(goal, tools, &state)
}

pub fn planner_prompt_with_state(goal: &str, tools: &[ToolInfo], state: &AgentLoopState) -> String {
    planner_prompt_with_state_and_memory(goal, tools, state, &PlannerMemoryContext::default())
}

pub fn planner_prompt_with_state_and_memory(
    goal: &str,
    tools: &[ToolInfo],
    state: &AgentLoopState,
    memory_context: &PlannerMemoryContext,
) -> String {
    let request =
        planner_request_with_state_and_memory("planner", goal, tools, state, memory_context);
    let mut rendered = request
        .messages
        .iter()
        .map(|message| format!("### {:?}\n{}", message.role, message.content))
        .collect::<Vec<_>>()
        .join("\n\n");
    if let Some(hint) = state.recovery_hint() {
        rendered = format!("### Runtime Override (this turn only)\n{hint}\n\n{rendered}");
    }
    rendered
}

fn compact_observations(observations: &[ToolObservation]) -> String {
    observations
        .iter()
        .map(|observation| {
            let content = compact_json(&observation.result.content, 1_500);
            format!(
                "- turn {} summary={} tool={}({}) -> ok={} {}",
                observation.turn,
                observation.summary,
                observation.call.name,
                compact_json(&observation.call.args, 500),
                observation.result.ok,
                content
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn compact_json(value: &Value, limit: usize) -> String {
    let mut text = json!(value).to_string();
    if text.len() > limit {
        truncate_utf8(&mut text, limit);
        text.push_str(" ... [truncated]");
    }
    text
}

fn push_limited_unique(items: &mut Vec<String>, value: String, limit: usize) {
    if value.trim().is_empty() {
        return;
    }
    if items.last() == Some(&value) {
        return;
    }
    items.push(value);
    if items.len() > limit {
        let extra = items.len() - limit;
        items.drain(0..extra);
    }
}

/// History-specific push: same dedup/cap shape as `push_limited_unique`, but
/// when items overflow the window they're not just dropped — each overflow
/// event gets folded into one short `"earlier: turns A-B (N turns folded) …"`
/// line appended to `earlier`. Borrowed from GenericAgent's `_fold_earlier`:
/// long runs (50+ turns) keep a compressed pointer to early-turn context
/// instead of the planner re-discovering "what did we already explore" from
/// only the last 30 entries.
///
/// `earlier` itself is capped at 20 fold-events so a runaway loop can't blow
/// the working-memory anchor size.
fn push_history_with_fold(
    history: &mut Vec<String>,
    earlier: &mut Vec<String>,
    value: String,
    limit: usize,
) {
    if value.trim().is_empty() {
        return;
    }
    if history.last() == Some(&value) {
        return;
    }
    history.push(value);
    if history.len() > limit {
        let extra = history.len() - limit;
        let drained: Vec<String> = history.drain(0..extra).collect();
        fold_earlier_entries(earlier, drained);
    }
}

fn fold_earlier_entries(earlier: &mut Vec<String>, drained: Vec<String>) {
    if drained.is_empty() {
        return;
    }
    let first = drained.first().cloned().unwrap_or_default();
    let last = drained.last().cloned().unwrap_or_default();
    let fold_line = if drained.len() == 1 {
        // Single-item drain: keep verbatim, no need to fold.
        first
    } else {
        format!(
            "earlier: {} turns folded — first: {} | last: {}",
            drained.len(),
            compact_text(&first, 80),
            compact_text(&last, 80),
        )
    };
    earlier.push(fold_line);
    if earlier.len() > 20 {
        let extra = earlier.len() - 20;
        earlier.drain(0..extra);
    }
}

fn consecutive_failures(observations: &[ToolObservation]) -> usize {
    observations
        .iter()
        .rev()
        .take_while(|observation| !observation.result.ok)
        .count()
}

fn repeated_last_call(observations: &[ToolObservation]) -> bool {
    let Some(last) = observations.last() else {
        return false;
    };
    observations.iter().rev().take(2).all(|observation| {
        observation.call.name == last.call.name && observation.call.args == last.call.args
    }) && observations.len() >= 2
}

fn action_summary(action: &PlannedAction) -> String {
    if let Some(summary) = action.summary() {
        return compact_text(summary, 220);
    }
    match action {
        PlannedAction::Tool {
            tool_name, args, ..
        } => compact_text(
            &format!("Plan to call {tool_name} with {}", compact_json(args, 200)),
            220,
        ),
        PlannedAction::Finish { answer, .. } => compact_text(answer, 220),
    }
}

fn compact_text(text: &str, limit: usize) -> String {
    let mut text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.len() > limit {
        truncate_utf8(&mut text, limit);
        text.push_str(" ...");
    }
    text
}

fn truncate_utf8(text: &mut String, limit: usize) {
    if text.len() <= limit {
        return;
    }
    let mut end = limit;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
}

pub fn parse_planned_action(text: &str) -> Result<PlannedAction, RuntimeError> {
    let json_text = extract_json_object(text).unwrap_or_else(|| text.trim().to_string());
    let action: PlannedAction = serde_json::from_str(&json_text)
        .map_err(|err| RuntimeError::InvalidPlannerJson(format!("{err}; text={text}")))?;
    if action.summary().is_none() {
        return Err(RuntimeError::InvalidPlannerJson(format!(
            "planner action is missing a non-empty `summary` field; got: {json_text}"
        )));
    }
    if let Err(reason) = action.sanity_check() {
        return Err(RuntimeError::InvalidPlannerJson(format!(
            "{reason}; got: {json_text}"
        )));
    }
    Ok(action)
}

fn extract_json_object(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return Some(trimmed.to_string());
    }

    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if start <= end {
        Some(trimmed[start..=end].to_string())
    } else {
        None
    }
}

pub fn default_provider_id() -> &'static str {
    ProviderId::CODEX
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn history_overflow_accumulates_one_by_one_into_earlier() {
        let mut history: Vec<String> = Vec::new();
        let mut earlier: Vec<String> = Vec::new();
        // Fill exactly to the limit — no overflow yet.
        for n in 1..=30 {
            push_history_with_fold(&mut history, &mut earlier, format!("turn {n}: x"), 30);
        }
        assert_eq!(history.len(), 30);
        assert!(earlier.is_empty());

        // Overflow one turn at a time (the typical loop pattern): each push
        // drains exactly one entry, which goes verbatim into earlier.
        for n in 31..=35 {
            push_history_with_fold(&mut history, &mut earlier, format!("turn {n}: y"), 30);
        }
        assert_eq!(history.len(), 30);
        assert!(history.first().unwrap().starts_with("turn 6:"));
        assert_eq!(earlier.len(), 5);
        assert_eq!(earlier[0], "turn 1: x");
        assert_eq!(earlier[4], "turn 5: x");
    }

    #[test]
    fn bulk_drain_folds_with_count_and_exemplars() {
        let mut history: Vec<String> = Vec::new();
        let mut earlier: Vec<String> = Vec::new();
        // Pre-load 5 entries directly to force a 5-item drain on the next push.
        // Mirrors the WorkingMemory::from_observations restore path where many
        // turns of history land at once.
        for n in 1..=34 {
            history.push(format!("turn {n}: x"));
        }
        push_history_with_fold(&mut history, &mut earlier, "turn 35: y".to_string(), 30);
        assert_eq!(history.len(), 30);
        assert_eq!(earlier.len(), 1);
        let fold_line = &earlier[0];
        assert!(fold_line.contains("5 turns folded"), "got: {fold_line}");
        assert!(fold_line.contains("turn 1:"), "got: {fold_line}");
        assert!(fold_line.contains("turn 5:"), "got: {fold_line}");
    }

    #[test]
    fn earlier_summary_caps_at_20_entries() {
        let mut history: Vec<String> = Vec::new();
        let mut earlier: Vec<String> = Vec::new();
        for n in 1..=30 {
            push_history_with_fold(&mut history, &mut earlier, format!("turn {n}: x"), 30);
        }
        // Push 25 more — each drains one verbatim entry, but earlier caps at 20.
        for n in 31..=55 {
            push_history_with_fold(&mut history, &mut earlier, format!("turn {n}: y"), 30);
        }
        assert_eq!(earlier.len(), 20);
        // Oldest 5 dropped; what's left starts at turn 6 (turn 1-5 evicted).
        assert_eq!(earlier[0], "turn 6: x");
    }

    #[test]
    fn sanity_check_rejects_empty_tool_name() {
        let err = parse_planned_action(
            r#"{"summary":"ok","action":"tool","tool_name":"","args":{}}"#,
        )
        .unwrap_err();
        let RuntimeError::InvalidPlannerJson(msg) = err else {
            panic!("expected InvalidPlannerJson, got: {err:?}");
        };
        assert!(msg.contains("tool_name") && msg.contains("empty"), "got: {msg}");
    }

    #[test]
    fn sanity_check_rejects_placeholder_tool_name() {
        let err = parse_planned_action(
            r#"{"summary":"ok","action":"tool","tool_name":"TODO","args":{}}"#,
        )
        .unwrap_err();
        let RuntimeError::InvalidPlannerJson(msg) = err else {
            panic!("expected InvalidPlannerJson, got: {err:?}");
        };
        assert!(msg.contains("placeholder"), "got: {msg}");
    }

    #[test]
    fn sanity_check_rejects_null_args() {
        let err = parse_planned_action(
            r#"{"summary":"ok","action":"tool","tool_name":"read_file","args":null}"#,
        )
        .unwrap_err();
        let RuntimeError::InvalidPlannerJson(msg) = err else {
            panic!("expected InvalidPlannerJson, got: {err:?}");
        };
        assert!(msg.contains("null"), "got: {msg}");
    }

    #[test]
    fn sanity_check_rejects_non_object_args() {
        let err = parse_planned_action(
            r#"{"summary":"ok","action":"tool","tool_name":"read_file","args":"oops"}"#,
        )
        .unwrap_err();
        let RuntimeError::InvalidPlannerJson(msg) = err else {
            panic!("expected InvalidPlannerJson, got: {err:?}");
        };
        assert!(msg.contains("must be a JSON object"), "got: {msg}");
    }

    #[test]
    fn sanity_check_rejects_blank_finish_answer() {
        let err = parse_planned_action(
            r#"{"summary":"done","action":"finish","answer":"   "}"#,
        )
        .unwrap_err();
        let RuntimeError::InvalidPlannerJson(msg) = err else {
            panic!("expected InvalidPlannerJson, got: {err:?}");
        };
        assert!(msg.contains("blank"), "got: {msg}");
    }

    #[test]
    fn sanity_check_accepts_zero_arg_tool() {
        // Zero-arg tools (e.g. `repoprompt_tools`) pass `args: {}` legitimately.
        let action = parse_planned_action(
            r#"{"summary":"list rp tools","action":"tool","tool_name":"repoprompt_tools","args":{}}"#,
        )
        .expect("zero-arg tool should pass sanity");
        match action {
            PlannedAction::Tool { tool_name, .. } => assert_eq!(tool_name, "repoprompt_tools"),
            _ => panic!("expected Tool"),
        }
    }

    #[test]
    fn parses_tool_action_from_plain_json() {
        let action = parse_planned_action(
            r#"{"summary":"open README","action":"tool","tool_name":"read_file","args":{"path":"README.md"}}"#,
        )
        .unwrap();

        match action {
            PlannedAction::Tool {
                tool_name, args, ..
            } => {
                assert_eq!(tool_name, "read_file");
                assert_eq!(args, json!({ "path": "README.md" }));
            }
            _ => panic!("expected tool"),
        }
    }

    #[test]
    fn parses_json_inside_markdown_noise() {
        let action = parse_planned_action(
            "```json\n{\"summary\":\"finishing\",\"action\":\"finish\",\"answer\":\"done\"}\n```",
        )
        .unwrap();

        match action {
            PlannedAction::Finish { answer, .. } => assert_eq!(answer, "done"),
            _ => panic!("expected finish"),
        }
    }

    #[test]
    fn parses_with_summary() {
        let action = parse_planned_action(
            r#"{"summary":"read the README next","action":"tool","tool_name":"read_file","args":{"path":"README.md"}}"#,
        )
        .unwrap();

        assert_eq!(action.summary(), Some("read the README next"));
    }

    #[test]
    fn rejects_action_without_summary() {
        let err = parse_planned_action(
            r#"{"action":"tool","tool_name":"read_file","args":{"path":"README.md"}}"#,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("summary"), "got: {msg}");
    }

    #[test]
    fn rejects_action_with_empty_summary() {
        let err = parse_planned_action(
            r#"{"summary":"   ","action":"finish","answer":"done"}"#,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("summary"), "got: {msg}");
    }

    #[test]
    fn planner_request_includes_observations() {
        let call = ToolCall::new("read_file", json!({ "path": "README.md" }));
        let result = ToolResult::ok(&call, json!({ "content": "hello" }));
        let request = planner_request_with_observations(
            "gpt-5.1",
            "inspect repo",
            &[ToolInfo {
                name: "read_file".to_string(),
                description: "read".to_string(),
            }],
            &[ToolObservation {
                turn: 1,
                summary: "read README".to_string(),
                call,
                result,
            }],
        );
        assert!(
            request.messages[1]
                .content
                .contains("Prior tool observations")
        );
    }

    #[test]
    fn planner_request_injects_working_checkpoint() {
        let call = ToolCall::new(
            "update_working_checkpoint",
            json!({ "key_info": "Repo root verified", "related_skill": "repo-inspection" }),
        );
        let result = ToolResult::ok(
            &call,
            json!({
                "status": "success",
                "key_info": "Repo root verified",
                "related_skill": "repo-inspection"
            }),
        );
        let request = planner_request_with_observations(
            "gpt-5.1",
            "keep going",
            &[ToolInfo {
                name: "update_working_checkpoint".to_string(),
                description: "checkpoint".to_string(),
            }],
            &[ToolObservation {
                turn: 1,
                summary: "saved repo root".to_string(),
                call,
                result,
            }],
        );
        let user = &request.messages[1].content;

        assert!(user.contains("### [WORKING MEMORY]"));
        assert!(user.contains("<key_info>"));
        assert!(user.contains("Repo root verified"));
        assert!(user.contains("repo-inspection"));
    }

    #[test]
    fn planner_request_injects_loop_guard_hints() {
        let call = ToolCall::new("patch_file", json!({ "path": "README.md" }));
        let first = ToolObservation {
            turn: 1,
            summary: "patch failed".to_string(),
            call: call.clone(),
            result: ToolResult::error(&call, "old content not found"),
        };
        let second = ToolObservation {
            turn: 2,
            summary: "same patch failed again".to_string(),
            call: call.clone(),
            result: ToolResult::error(&call, "old content not found"),
        };
        let state = AgentLoopState::from_observations(&[first, second], 3, 8);
        let prompt = planner_prompt_with_state(
            "fix README",
            &[ToolInfo {
                name: "patch_file".to_string(),
                description: "patch".to_string(),
            }],
            &state,
        );

        assert!(prompt.contains("Two tool calls in a row failed"));
        assert!(prompt.contains("same tool call repeated"));
    }

    #[test]
    fn compact_tool_description_keeps_first_sentence_only() {
        let long = "Delegate an isolated sub-task to a child seed process. By default the child runs in its OWN cwd. Pass inherit_memory=true ONLY when ...";
        let slim = compact_tool_description(long);
        assert_eq!(
            slim,
            "Delegate an isolated sub-task to a child seed process."
        );
    }

    #[test]
    fn compact_tool_description_truncates_when_no_sentence_break() {
        let long = "A".repeat(200);
        let slim = compact_tool_description(&long);
        assert!(slim.ends_with('…'));
        assert!(slim.chars().count() <= 91);
    }

    #[test]
    fn compact_tool_description_passthrough_short_one_liners() {
        let s = "Search the L1 memory index";
        assert_eq!(compact_tool_description(s), s);
    }

    #[test]
    fn tool_routing_block_appears_when_repoprompt_and_subagent_present() {
        let state = AgentLoopState::new(8);
        let tools = vec![
            ToolInfo {
                name: "repoprompt_call".to_string(),
                description: "rp".to_string(),
            },
            ToolInfo {
                name: "spawn_subagent".to_string(),
                description: "sub".to_string(),
            },
            ToolInfo {
                name: "read_file".to_string(),
                description: "read".to_string(),
            },
            ToolInfo {
                name: "patch_file".to_string(),
                description: "patch".to_string(),
            },
            ToolInfo {
                name: "run_shell".to_string(),
                description: "shell".to_string(),
            },
            ToolInfo {
                name: "ask_user".to_string(),
                description: "ask".to_string(),
            },
        ];
        let prompt = planner_prompt_with_state("test", &tools, &state);
        assert!(prompt.contains("TOOL ROUTING"), "block missing: {prompt}");
        assert!(prompt.contains("Cross-file code exploration"));
        assert!(prompt.contains("agent_run"));
        assert!(prompt.contains("ask_user"));
        assert!(prompt.contains("apply_edits"));
        assert!(prompt.contains("`repoprompt_call git`"));
    }

    #[test]
    fn tool_routing_block_omitted_when_only_local_tools_present() {
        let state = AgentLoopState::new(8);
        let tools = vec![ToolInfo {
            name: "read_file".to_string(),
            description: "read".to_string(),
        }];
        let prompt = planner_prompt_with_state("test", &tools, &state);
        assert!(!prompt.contains("TOOL ROUTING"));
    }

    #[test]
    fn turn_budget_guard_fires_when_close_to_max() {
        // turn 5 of max 8 → 3 turns remaining (5+3 >= 8) → budget warning should fire.
        let state = AgentLoopState::from_observations(&[], 5, 8);
        let prompt = planner_prompt_with_state(
            "test",
            &[ToolInfo {
                name: "ask_user".to_string(),
                description: "ask".to_string(),
            }],
            &state,
        );
        assert!(prompt.contains("Budget warning"), "got: {prompt}");
        assert!(prompt.contains("ask_user"));
    }

    #[test]
    fn exploration_streak_guard_fires_after_five_reads() {
        let call = ToolCall::new("read_file", json!({ "path": "x" }));
        let observations: Vec<ToolObservation> = (1..=6)
            .map(|turn| ToolObservation {
                turn,
                summary: format!("read {turn}"),
                call: call.clone(),
                result: ToolResult::ok(&call, json!({ "content": "..." })),
            })
            .collect();
        let state = AgentLoopState::from_observations(&observations, 7, 16);
        let prompt = planner_prompt_with_state(
            "test",
            &[ToolInfo {
                name: "read_file".to_string(),
                description: "read".to_string(),
            }],
            &state,
        );
        assert!(prompt.contains("Exploration streak guard"), "got: {prompt}");
    }

    #[test]
    fn exploration_streak_escalates_to_recovery_hint_at_seven() {
        let call = ToolCall::new("read_file", json!({ "path": "x" }));
        let observations: Vec<ToolObservation> = (1..=8)
            .map(|turn| ToolObservation {
                turn,
                summary: format!("read {turn}"),
                call: call.clone(),
                result: ToolResult::ok(&call, json!({ "content": "..." })),
            })
            .collect();
        let mut state = AgentLoopState::from_observations(&observations, 9, 24);
        // prepare_turn is what writes recovery_hint; from_observations doesn't
        // call it. Simulate one turn-prepare to surface the escalation.
        state.prepare_turn(9);
        assert!(
            state.recovery_hint().is_some(),
            "expected hard recovery_hint at streak ≥ 7"
        );
        let hint = state.recovery_hint().unwrap();
        assert!(hint.contains("STOP exploring"), "got: {hint}");
        assert!(hint.contains("finish") || hint.contains("update_working_checkpoint"));
    }

    #[test]
    fn exploration_streak_resets_after_synthesis_step() {
        let read_call = ToolCall::new("read_file", json!({ "path": "x" }));
        let cp_call = ToolCall::new("update_working_checkpoint", json!({ "key_info": "x" }));
        let mut observations: Vec<ToolObservation> = (1..=4)
            .map(|turn| ToolObservation {
                turn,
                summary: format!("read {turn}"),
                call: read_call.clone(),
                result: ToolResult::ok(&read_call, json!({ "content": "..." })),
            })
            .collect();
        // Insert a synthesis turn that breaks the streak.
        observations.push(ToolObservation {
            turn: 5,
            summary: "checkpoint".to_string(),
            call: cp_call.clone(),
            result: ToolResult::ok(&cp_call, json!({ "key_info": "anchored" })),
        });
        let state = AgentLoopState::from_observations(&observations, 6, 16);
        let prompt = planner_prompt_with_state(
            "test",
            &[ToolInfo {
                name: "read_file".to_string(),
                description: "read".to_string(),
            }],
            &state,
        );
        assert!(!prompt.contains("Exploration streak guard"), "got: {prompt}");
    }

    #[test]
    fn checkpoint_nudge_fires_when_anchor_is_thin() {
        let call = ToolCall::new("read_file", json!({ "path": "x" }));
        let obs = ToolObservation {
            turn: 1,
            summary: "read x".to_string(),
            call: call.clone(),
            result: ToolResult::ok(&call, json!({ "content": "..." })),
        };
        let state = AgentLoopState::from_observations(&[obs], 3, 16);
        let prompt = planner_prompt_with_state(
            "test",
            &[ToolInfo {
                name: "update_working_checkpoint".to_string(),
                description: "anchor".to_string(),
            }],
            &state,
        );
        assert!(prompt.contains("Checkpoint nudge"), "got: {prompt}");
    }

    #[test]
    fn checkpoint_nudge_silent_when_anchor_already_populated() {
        let call = ToolCall::new("update_working_checkpoint", json!({ "key_info": "x" }));
        let obs1 = ToolObservation {
            turn: 1,
            summary: "anchor".to_string(),
            call: call.clone(),
            result: ToolResult::ok(&call, json!({ "key_info": "fact-A" })),
        };
        let obs2 = ToolObservation {
            turn: 2,
            summary: "anchor".to_string(),
            call: call.clone(),
            result: ToolResult::ok(&call, json!({ "key_info": "fact-B" })),
        };
        let obs3 = ToolObservation {
            turn: 3,
            summary: "anchor".to_string(),
            call: call.clone(),
            result: ToolResult::ok(&call, json!({ "key_info": "fact-C" })),
        };
        let state = AgentLoopState::from_observations(&[obs1, obs2, obs3], 6, 16);
        let prompt = planner_prompt_with_state(
            "test",
            &[ToolInfo {
                name: "update_working_checkpoint".to_string(),
                description: "anchor".to_string(),
            }],
            &state,
        );
        assert!(!prompt.contains("Checkpoint nudge"), "got: {prompt}");
    }

    #[test]
    fn memory_refresh_guard_fires_at_multiple_of_ten() {
        let state = AgentLoopState::from_observations(&[], 10, 32);
        let prompt = planner_prompt_with_state(
            "test",
            &[ToolInfo {
                name: "memory_fetch".to_string(),
                description: "fetch".to_string(),
            }],
            &state,
        );
        assert!(prompt.contains("Memory refresh"), "got: {prompt}");
    }

    #[test]
    fn planner_request_injects_l0_l1_memory_context() {
        let state = AgentLoopState::new(4);
        let memory = PlannerMemoryContext::new(
            "### [L0 META RULES]\nKeep memory small.\n\n### [L1 MEMORY INDEX]\n- id=global-facts layer=L2 title=Global Facts path=memory/global_facts.md summary=Stable facts",
        );
        let prompt = planner_prompt_with_state_and_memory(
            "use memory",
            &[ToolInfo {
                name: "memory_search".to_string(),
                description: "search".to_string(),
            }],
            &state,
            &memory,
        );

        assert!(prompt.contains("### [L0 META RULES]"));
        assert!(prompt.contains("id=global-facts"));
        assert!(prompt.contains("Use memory_search before memory_fetch"));
    }

    #[test]
    fn compact_helpers_do_not_split_utf8() {
        let value = json!({
            "goal": "优化当前的项目优化当前的项目优化当前的项目"
        });

        let json_text = compact_json(&value, 17);
        assert!(json_text.contains(" ... [truncated]"));

        let inline = compact_text("优化 当前 的 项目 优化 当前 的 项目", 13);
        assert!(inline.ends_with(" ..."));
    }

    #[test]
    fn read_only_analysis_goal_detection_distinguishes_action_goals() {
        assert!(is_read_only_analysis_goal("分析当前的项目"));
        assert!(is_read_only_analysis_goal("Summarize this repo"));
        assert!(!is_read_only_analysis_goal("优化当前的项目"));
        assert!(!is_read_only_analysis_goal("analyze and fix this module"));
    }

    #[test]
    fn deep_analysis_goal_only_fires_for_deep_keywords() {
        // Light analysis goals → not deep
        assert!(!is_deep_analysis_goal("分析当前的项目"));
        assert!(!is_deep_analysis_goal("Summarize this repo"));
        // Deep keywords → deep
        assert!(is_deep_analysis_goal("深入分析当前的项目"));
        assert!(is_deep_analysis_goal("全面分析架构"));
        assert!(is_deep_analysis_goal("deeply analyze the auth module"));
        assert!(is_deep_analysis_goal("comprehensive review of the runtime"));
        // Implementation goals never qualify (even with deep keywords)
        assert!(!is_deep_analysis_goal("深入重构 main.rs"));
    }

    #[test]
    fn deep_analysis_guidance_appears_in_planner_prompt_when_deep() {
        let state = AgentLoopState::new(24);
        let prompt = planner_prompt_with_state(
            "深入分析当前的项目",
            &[ToolInfo {
                name: "read_files".to_string(),
                description: "batch read".to_string(),
            }],
            &state,
        );
        assert!(prompt.contains("DEEP read-only analysis"), "got: {prompt}");
        assert!(prompt.contains("up to 8 evidence-gathering"));
        assert!(prompt.contains("README paraphrase"));
    }

    #[test]
    fn planner_request_guides_read_only_analysis_to_finish() {
        let request = planner_request_with_state(
            "gpt-5.1",
            "分析当前的项目",
            &[ToolInfo {
                name: "read_file".to_string(),
                description: "read".to_string(),
            }],
            &AgentLoopState::new(4),
        );

        let system = &request.messages[0].content;
        assert!(system.contains("read-only analysis/investigation"));
        assert!(system.contains("Do not create a durable plan"));
    }

    #[test]
    fn planner_request_requires_repoprompt_plan_ledger_updates() {
        let state = AgentLoopState::new(4);
        let prompt = planner_prompt_with_state(
            "implement with RepoPrompt",
            &[
                ToolInfo {
                    name: "repoprompt_call".to_string(),
                    description: "call RepoPrompt".to_string(),
                },
                ToolInfo {
                    name: "plan_record_artifact".to_string(),
                    description: "record artifact".to_string(),
                },
                ToolInfo {
                    name: "plan_record_handoff".to_string(),
                    description: "record handoff".to_string(),
                },
            ],
            &state,
        );

        assert!(prompt.contains("plan_record_artifact"));
        assert!(prompt.contains("plan_record_handoff"));
        assert!(prompt.contains("RepoPrompt agent_run"));
        assert!(prompt.contains("orchestration ledger"));
    }

    #[test]
    fn planner_request_injects_long_term_update_settlement() {
        let call = ToolCall::new(
            "start_long_term_update",
            json!({
                "reason": "remember stable repo root",
                "evidence": "pwd verified /tmp/repo"
            }),
        );
        let result = ToolResult::ok(
            &call,
            json!({
                "status": "success",
                "phase": "long_term_memory_settlement",
                "reason": "remember stable repo root",
                "evidence": "pwd verified /tmp/repo",
                "sop_path": "memory/memory_management_sop.md",
                "next_prompt": "LONG_TERM_MEMORY_SETTLEMENT: choose exactly one branch now. After the write or skip decision, call complete_long_term_update."
            }),
        );
        let state = AgentLoopState::from_observations(
            &[ToolObservation {
                turn: 1,
                summary: "start settlement".to_string(),
                call,
                result,
            }],
            2,
            4,
        );
        let prompt = planner_prompt_with_state(
            "settle memory",
            &[ToolInfo {
                name: "memory_fetch".to_string(),
                description: "fetch".to_string(),
            }],
            &state,
        );

        assert!(prompt.contains("<long_term_update>"));
        assert!(prompt.contains("remember stable repo root"));
        assert!(prompt.contains("choose exactly one branch"));
        assert!(prompt.contains("complete_long_term_update"));
        assert!(prompt.contains("update L2 global facts"));
        assert!(prompt.contains("update an existing L3 skill"));
        assert!(prompt.contains("skip with a reason"));
    }

    #[test]
    fn complete_long_term_update_clears_active_settlement() {
        let start_call = ToolCall::new(
            "start_long_term_update",
            json!({ "reason": "remember workflow", "evidence": "verified" }),
        );
        let start_result = ToolResult::ok(
            &start_call,
            json!({
                "status": "success",
                "reason": "remember workflow",
                "evidence": "verified",
                "next_prompt": "LONG_TERM_MEMORY_SETTLEMENT: choose exactly one branch now."
            }),
        );
        let complete_call = ToolCall::new(
            "complete_long_term_update",
            json!({
                "decision": "update_l3_skill",
                "target": "skills/demo/SKILL.md",
                "reason": "updated existing skill",
                "evidence": "verified",
                "changed": true
            }),
        );
        let complete_result = ToolResult::ok(
            &complete_call,
            json!({
                "status": "success",
                "decision": "update_l3_skill",
                "target": "skills/demo/SKILL.md",
                "reason": "updated existing skill",
                "evidence": "verified",
                "changed": true
            }),
        );
        let state = AgentLoopState::from_observations(
            &[
                ToolObservation {
                    turn: 1,
                    summary: "start".to_string(),
                    call: start_call,
                    result: start_result,
                },
                ToolObservation {
                    turn: 2,
                    summary: "complete".to_string(),
                    call: complete_call,
                    result: complete_result,
                },
            ],
            3,
            4,
        );

        assert!(state.working_memory.long_term_update.is_none());
        assert!(
            state
                .working_memory
                .key_info
                .iter()
                .any(|item| item.contains("decision=update_l3_skill"))
        );
    }

    #[test]
    fn generic_loop_uses_injected_planner() {
        let mut calls = 0usize;
        let result = run_agent_loop_with_planner(
            3,
            |observations| {
                calls += 1;
                if observations.is_empty() {
                    Ok(PlannedAction::Tool {
                        summary: Some("need README context".to_string()),
                        tool_name: "read_file".to_string(),
                        args: json!({ "path": "README.md" }),
                    })
                } else {
                    Ok(PlannedAction::Finish {
                        summary: Some("have enough context".to_string()),
                        answer: "done".to_string(),
                    })
                }
            },
            |call| ToolResult::ok(call, json!({ "content": "ok" })),
        )
        .unwrap();

        assert_eq!(calls, 2);
        assert_eq!(result.status, AgentLoopStatus::Finished);
        assert_eq!(result.observations.len(), 1);
        assert_eq!(result.turn_summaries.len(), 2);
    }

    #[test]
    fn planner_loop_retries_invalid_json_with_recovery_hint() {
        use std::cell::RefCell;
        let calls = RefCell::new(0usize);
        let seen_hint = RefCell::new(false);
        let result = run_agent_loop_with_state_planner(
            3,
            |state| {
                let mut n = calls.borrow_mut();
                *n += 1;
                if *n == 1 {
                    assert!(state.recovery_hint().is_none(), "first turn has no hint");
                    Err(RuntimeError::InvalidPlannerJson(
                        "missing summary".to_string(),
                    ))
                } else {
                    if state.recovery_hint().is_some() {
                        *seen_hint.borrow_mut() = true;
                    }
                    Ok(PlannedAction::Finish {
                        summary: Some("recovered".to_string()),
                        answer: "done".to_string(),
                    })
                }
            },
            |call| ToolResult::ok(call, json!({})),
        )
        .unwrap();

        assert_eq!(*calls.borrow(), 2);
        assert!(*seen_hint.borrow(), "retry must surface recovery_hint to planner");
        assert_eq!(result.status, AgentLoopStatus::Finished);
    }

    #[test]
    fn planner_loop_bails_after_exhausting_retries() {
        let err = run_agent_loop_with_state_planner_retries::<_, _, _>(
            3,
            1,
            0,
            |_state| {
                Err::<PlannedAction, _>(RuntimeError::InvalidPlannerJson(
                    "still bad".to_string(),
                ))
            },
            |call| ToolResult::ok(call, json!({})),
            |_| {},
        )
        .unwrap_err();
        assert!(matches!(err, RuntimeError::InvalidPlannerJson(_)));
    }

    #[test]
    fn planner_loop_retries_transport_errors_then_succeeds() {
        use std::cell::RefCell;
        let calls = RefCell::new(0usize);
        let result = run_agent_loop_with_state_planner_retries(
            3,
            0,
            2,
            |state| {
                let mut n = calls.borrow_mut();
                *n += 1;
                if *n == 1 {
                    Err(RuntimeError::Planner("simulated transport blip".to_string()))
                } else {
                    assert!(state.recovery_hint().is_some(), "retry must carry hint");
                    Ok(PlannedAction::Finish {
                        summary: Some("recovered after retry".to_string()),
                        answer: "done".to_string(),
                    })
                }
            },
            |call| ToolResult::ok(call, json!({})),
            |_| {},
        )
        .unwrap();
        assert_eq!(*calls.borrow(), 2);
        assert_eq!(result.status, AgentLoopStatus::Finished);
    }

    #[test]
    fn planner_loop_bails_after_transport_retries_exhaust() {
        let err = run_agent_loop_with_state_planner_retries::<_, _, _>(
            3,
            0,
            1,
            |_state| {
                Err::<PlannedAction, _>(RuntimeError::Planner(
                    "transport keeps failing".to_string(),
                ))
            },
            |call| ToolResult::ok(call, json!({})),
            |_| {},
        )
        .unwrap_err();
        assert!(matches!(err, RuntimeError::Planner(_)));
    }

    #[test]
    fn state_loop_updates_working_memory_after_checkpoint() {
        let mut saw_checkpoint = false;
        let result = run_agent_loop_with_state_planner(
            3,
            |state| {
                if state.observations.is_empty() {
                    Ok(PlannedAction::Tool {
                        summary: Some("save verified root".to_string()),
                        tool_name: "update_working_checkpoint".to_string(),
                        args: json!({ "key_info": "Repo root verified" }),
                    })
                } else {
                    saw_checkpoint = state
                        .working_memory
                        .key_info
                        .iter()
                        .any(|item| item.contains("Repo root verified"));
                    Ok(PlannedAction::Finish {
                        summary: Some("checkpoint is available".to_string()),
                        answer: "done".to_string(),
                    })
                }
            },
            |call| {
                ToolResult::ok(
                    call,
                    json!({ "status": "success", "key_info": "Repo root verified" }),
                )
            },
        )
        .unwrap();

        assert!(saw_checkpoint);
        assert_eq!(result.status, AgentLoopStatus::Finished);
        assert!(
            result
                .working_memory
                .key_info
                .iter()
                .any(|item| item.contains("Repo root verified"))
        );
    }

    // --- RF33-3 per-turn tool-description culling ----------------------

    #[test]
    fn planner_prompt_includes_descriptions_through_turn_4() {
        let tools = vec![
            ToolInfo {
                name: "read_file".to_string(),
                description: "Read a UTF-8 file from disk.".to_string(),
            },
            ToolInfo {
                name: "write_file".to_string(),
                description: "Write or overwrite a file at the given path.".to_string(),
            },
        ];
        // Turn 1 — full descriptions.
        let mut state = AgentLoopState::from_observations(&[], 1, 0);
        state.working_memory.current_turn = 1;
        let req = planner_request_with_state(
            ModelId::from("x"),
            "demo",
            &tools,
            &state,
        );
        let sys = &req.messages[0].content;
        assert!(sys.contains("Read a UTF-8 file from disk"));
        assert!(sys.contains("Write or overwrite a file"));

        // Turn 4 — descriptions still present (compact path).
        state.working_memory.current_turn = 4;
        let req = planner_request_with_state(
            ModelId::from("x"),
            "demo",
            &tools,
            &state,
        );
        assert!(req.messages[0].content.contains("Read a UTF-8 file"));
    }

    #[test]
    fn planner_prompt_names_only_after_turn_5() {
        let tools = vec![
            ToolInfo {
                name: "read_file".to_string(),
                description: "Read a UTF-8 file from disk.".to_string(),
            },
            ToolInfo {
                name: "write_file".to_string(),
                description: "Write or overwrite a file at the given path.".to_string(),
            },
        ];
        let mut state = AgentLoopState::from_observations(&[], 5, 0);
        state.working_memory.current_turn = 5;
        let req = planner_request_with_state(
            ModelId::from("x"),
            "demo",
            &tools,
            &state,
        );
        let sys = &req.messages[0].content;
        // Names still present.
        assert!(sys.contains("- read_file"));
        assert!(sys.contains("- write_file"));
        // Descriptions gone.
        assert!(
            !sys.contains("Read a UTF-8 file from disk"),
            "turn 5+ should drop tool descriptions to save prompt tokens"
        );
        // Recovery hint present.
        assert!(
            sys.contains("tool_describe"),
            "turn 5+ should hint at tool_describe for description recovery"
        );
    }
}
