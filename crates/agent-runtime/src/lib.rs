use agent_core::{ToolCall, ToolInfo, ToolResult};
use agent_llm::{ChatMessage, ChatRequest, ModelId, ProviderClient, ProviderId};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

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
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("planner response was not valid action JSON: {0}")]
    InvalidPlannerJson(String),
    #[error("planner failed: {0}")]
    Planner(String),
    #[error(transparent)]
    Llm(#[from] agent_llm::LlmError),
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
            max_turns: 8,
        }
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
        push_limited_unique(
            &mut self.history,
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

        if self.current_turn > 1 && self.current_turn % 7 == 0 {
            self.guard_hints.push(
                "Turn guard: summarize the current situation, update_working_checkpoint for verified context if useful, then change strategy if progress is weak."
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
}

impl AgentLoopState {
    pub fn new(max_turns: usize) -> Self {
        let max_turns = max_turns.max(1);
        Self {
            next_turn: 1,
            max_turns,
            observations: Vec::new(),
            working_memory: WorkingMemory::from_observations(&[], 1, max_turns),
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
        }
    }

    fn prepare_turn(&mut self, turn: usize) {
        self.next_turn = turn;
        self.working_memory.current_turn = turn;
        self.working_memory
            .refresh_guard_hints(&self.observations, self.max_turns);
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

pub fn run_agent_loop_with_state_planner<P, F>(
    max_turns: usize,
    mut plan_next: P,
    mut execute_tool: F,
) -> Result<AgentLoopResult, RuntimeError>
where
    P: FnMut(&AgentLoopState) -> Result<PlannedAction, RuntimeError>,
    F: FnMut(&ToolCall) -> ToolResult,
{
    let max_turns = max_turns.max(1);
    let mut state = AgentLoopState::new(max_turns);
    let mut turn_summaries = Vec::new();

    for turn in 1..=max_turns {
        state.prepare_turn(turn);
        let action = plan_next(&state)?;
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
    let tools_text = tools
        .iter()
        .map(|tool| format!("- {}: {}", tool.name, tool.description))
        .collect::<Vec<_>>()
        .join("\n");
    let system = format!(
        "You are the planner for a minimal local agent. Choose exactly one next action.\n\
Return only JSON, with no markdown.\n\
Every JSON object must include summary: a short working-memory snapshot of new facts and current intent.\n\
Use update_working_checkpoint for verified short-term context that should anchor later turns.\n\
For multi-step implementation, create or follow a durable plan with plan_create/plan_next/plan_complete. When all non-verify plan items are complete, call plan_verify and do not finish until the independent verification gate returns PASS.\n\
Use start_long_term_update only when successful evidence should be distilled into durable memory; skip it for guesses or one-off facts.\n\
When WORKING MEMORY contains <long_term_update>, you are in phase 2 settlement. Choose exactly one branch: update L2 global facts, update an existing L3 skill, or skip with a reason. Read/fetch the target before patching or writing. For L3, always memory_search first, memory_fetch the existing skill, then patch that existing SKILL.md; do not create duplicate skills.\n\
After a phase 2 write or skip decision, call complete_long_term_update before finish so the settlement is auditable.\n\
For a tool call, return: {{\"summary\":\"...\",\"action\":\"tool\",\"tool_name\":\"read_file\",\"args\":{{...}}}}\n\
If no tool is needed, return: {{\"summary\":\"...\",\"action\":\"finish\",\"answer\":\"...\"}}\n\
Available tools:\n{tools_text}"
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
    request.reasoning_effort = Some("minimal".to_string());
    request.max_output_tokens = Some(800);
    request
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
    request
        .messages
        .iter()
        .map(|message| format!("### {:?}\n{}", message.role, message.content))
        .collect::<Vec<_>>()
        .join("\n\n")
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
        text.truncate(limit);
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
        text.truncate(limit);
        text.push_str(" ...");
    }
    text
}

pub fn parse_planned_action(text: &str) -> Result<PlannedAction, RuntimeError> {
    let json_text = extract_json_object(text).unwrap_or_else(|| text.trim().to_string());
    serde_json::from_str(&json_text)
        .map_err(|err| RuntimeError::InvalidPlannerJson(format!("{err}; text={text}")))
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
    fn parses_tool_action_from_plain_json() {
        let action = parse_planned_action(
            r#"{"action":"tool","tool_name":"read_file","args":{"path":"README.md"}}"#,
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
        let action =
            parse_planned_action("```json\n{\"action\":\"finish\",\"answer\":\"done\"}\n```")
                .unwrap();

        match action {
            PlannedAction::Finish { answer, .. } => assert_eq!(answer, "done"),
            _ => panic!("expected finish"),
        }
    }

    #[test]
    fn parses_optional_summary() {
        let action = parse_planned_action(
            r#"{"summary":"read the README next","action":"tool","tool_name":"read_file","args":{"path":"README.md"}}"#,
        )
        .unwrap();

        assert_eq!(action.summary(), Some("read the README next"));
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
}
