use agent_core::AgentEvent;
use agent_session::SessionRecord;
use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillInfo {
    pub name: String,
    pub path: PathBuf,
    pub description: String,
    pub tags: Vec<String>,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_backend: Option<String>,
    #[serde(default = "default_autonomous_safe")]
    pub autonomous_safe: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blast_radius: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillDocument {
    pub info: SkillInfo,
    pub body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepoPromptSkillRoute {
    pub name: &'static str,
    pub slug: &'static str,
    pub reason: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedSkill {
    pub route: RepoPromptSkillRoute,
    pub document: SkillDocument,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SkillConsolidationDecision {
    Created,
    Updated,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillConsolidation {
    pub decision: SkillConsolidationDecision,
    pub path: PathBuf,
    pub skill_name: String,
    pub reason: String,
}

pub fn reflect_markdown(records: &[SessionRecord]) -> String {
    let mut goal = "Unknown goal".to_string();
    let mut cwd = None;
    let mut tools = Vec::new();
    let mut successes = Vec::new();
    let mut failures = Vec::new();
    let mut checkpoints = Vec::new();
    let mut final_summary = None;

    for record in records {
        match &record.event {
            AgentEvent::RunStarted { goal: g, cwd: c } => {
                goal = g.clone();
                cwd = Some(c.display().to_string());
            }
            AgentEvent::ToolStarted { call } => tools.push(call.name.clone()),
            AgentEvent::ToolFinished { result } => {
                let line = format!("{} -> {}", result.name, compact_json(&result.content));
                if result.ok {
                    successes.push(line);
                } else {
                    failures.push(line);
                }
            }
            AgentEvent::TurnSummary { turn, summary } => {
                checkpoints.push(format!("turn {turn}: {summary}"));
            }
            AgentEvent::CheckpointUpdated {
                key_info,
                related_skill,
            } => checkpoints.push(match related_skill {
                Some(skill) => format!("{key_info} (related: {skill})"),
                None => key_info.clone(),
            }),
            AgentEvent::LongTermUpdateStarted { reason, evidence } => {
                checkpoints.push(match evidence {
                    Some(evidence) => {
                        format!("long-term update requested: {reason}; evidence: {evidence}")
                    }
                    None => format!("long-term update requested: {reason}"),
                });
            }
            AgentEvent::LongTermUpdateSettled {
                decision,
                target,
                reason,
                evidence,
                changed,
            } => {
                let mut line = format!(
                    "long-term update settled: decision={decision}; changed={changed}; reason={reason}"
                );
                if let Some(target) = target {
                    line.push_str(&format!("; target={target}"));
                }
                if let Some(evidence) = evidence {
                    line.push_str(&format!("; evidence={evidence}"));
                }
                checkpoints.push(line);
            }
            AgentEvent::RunFinished { summary, .. } => final_summary = Some(summary.clone()),
            AgentEvent::Reflection { .. } => {}
        }
    }

    let unique_tools = tools.into_iter().collect::<BTreeSet<_>>();
    let cwd = cwd.unwrap_or_else(|| ".".to_string());
    let final_summary = final_summary.unwrap_or_else(|| "No final summary recorded.".to_string());

    format!(
        r#"# Session Reflection

## Goal
{goal}

## Verified Context
- Working directory: `{cwd}`
- Generated at: {}

## Tools Used
{}

## Successful Evidence
{}

## Failures Or Gaps
{}

## Working Checkpoints
{}

## Reusable Pattern
1. Recreate only the verified context above.
2. Prefer bounded reads before edits.
3. Use exact-match patches for source changes.
4. Run a narrow verification command after each meaningful change.

## Final Summary
{final_summary}
"#,
        Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
        bullet_list(unique_tools),
        bullet_list(successes),
        bullet_list_or_none(failures),
        bullet_list_or_none(checkpoints),
    )
}

pub fn create_skill(
    skills_dir: impl AsRef<Path>,
    name: &str,
    records: &[SessionRecord],
) -> Result<PathBuf> {
    let slug = slugify(name);
    let dir = skills_dir.as_ref().join(&slug);
    fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let path = dir.join("SKILL.md");
    let reflection = reflect_markdown(records);
    let body = format!(
        r#"# {name}

Use this skill when a future task matches the verified pattern below.

{reflection}

## Memory Rule
Only carry forward facts that were verified by successful tool calls. Do not store guesses, volatile state, or one-off command output as durable memory.
"#
    );
    agent_memory::validate_durable_memory_text(&body)
        .with_context(|| format!("validate durable skill {}", path.display()))?;
    fs::write(&path, body).with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

pub fn consolidate_skill(
    skills_dir: impl AsRef<Path>,
    name_hint: &str,
    records: &[SessionRecord],
) -> Result<SkillConsolidation> {
    let skills_dir = skills_dir.as_ref();
    let run = RunLearningContext::from_records(records);
    let candidates = list_skill_infos(skills_dir)?;
    if let Some((score, info)) = best_skill_match(&run, &candidates) {
        let update = learned_update_markdown(&run);
        agent_memory::validate_durable_memory_text(&update)
            .with_context(|| format!("validate learned update {}", info.path.display()))?;
        append_learned_update(&info.path, &update)?;
        return Ok(SkillConsolidation {
            decision: SkillConsolidationDecision::Updated,
            path: info.path,
            skill_name: info.name,
            reason: format!("updated existing skill with overlap score {score}"),
        });
    }

    let path = create_skill(skills_dir, name_hint, records)?;
    let skill_name = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .unwrap_or(name_hint)
        .to_string();
    Ok(SkillConsolidation {
        decision: SkillConsolidationDecision::Created,
        path,
        skill_name,
        reason: "created new skill because no sufficiently similar skill was found".to_string(),
    })
}

pub fn list_skill_infos(skills_dir: impl AsRef<Path>) -> Result<Vec<SkillInfo>> {
    let skills_dir = skills_dir.as_ref();
    if !skills_dir.exists() {
        return Ok(Vec::new());
    }

    let mut infos = Vec::new();
    for entry in
        fs::read_dir(skills_dir).with_context(|| format!("read {}", skills_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let skill_path = if path.is_dir() {
            path.join("SKILL.md")
        } else if path.extension().is_some_and(|ext| ext == "md") {
            path
        } else {
            continue;
        };
        if !skill_path.is_file() {
            continue;
        }
        let body = fs::read_to_string(&skill_path)
            .with_context(|| format!("read {}", skill_path.display()))?;
        infos.push(skill_info_from_body(skills_dir, skill_path, &body));
    }
    infos.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.path.cmp(&b.path)));
    Ok(infos)
}

pub fn search_skill_infos(
    skills_dir: impl AsRef<Path>,
    query: &str,
    limit: usize,
) -> Result<Vec<SkillInfo>> {
    let query_terms = terms(query);
    if query_terms.is_empty() {
        return Ok(list_skill_infos(skills_dir)?
            .into_iter()
            .take(limit.max(1))
            .collect());
    }

    let mut scored = list_skill_infos(skills_dir)?
        .into_iter()
        .filter_map(|info| {
            let haystack = skill_search_haystack(&info).to_ascii_lowercase();
            let score = query_terms
                .iter()
                .filter(|term| haystack.contains(term.as_str()))
                .count();
            (score > 0).then_some((score, info))
        })
        .collect::<Vec<_>>();
    scored.sort_by(|(a_score, a), (b_score, b)| {
        b_score
            .cmp(a_score)
            .then_with(|| a.name.cmp(&b.name))
            .then_with(|| a.path.cmp(&b.path))
    });
    Ok(scored
        .into_iter()
        .take(limit.max(1))
        .map(|(_, info)| info)
        .collect())
}

pub fn fetch_skill(skills_dir: impl AsRef<Path>, name: &str) -> Result<SkillDocument> {
    let skills_dir = skills_dir.as_ref();
    let wanted = name.trim();
    let wanted_slug = slugify(wanted);
    for info in list_skill_infos(skills_dir)? {
        if info.name == wanted || slugify(&info.name) == wanted_slug {
            let body = fs::read_to_string(&info.path)
                .with_context(|| format!("read {}", info.path.display()))?;
            return Ok(SkillDocument { info, body });
        }
    }
    anyhow::bail!("skill not found: {name}")
}

pub fn route_repoprompt_skill(task: &str) -> Option<RepoPromptSkillRoute> {
    let task = task.trim();
    if task.is_empty() {
        return None;
    }
    let lower = task.to_ascii_lowercase();

    if contains_any(
        task,
        &lower,
        &[
            "review",
            "code review",
            "审查",
            "复查",
            "检查代码",
            "风险",
            "找问题",
            "bug",
            "漏洞",
        ],
    ) {
        return Some(RepoPromptSkillRoute {
            name: "RepoPrompt Review",
            slug: "repoprompt-review",
            reason: "task asks for review or risk finding",
        });
    }

    if contains_any(
        task,
        &lower,
        &[
            "plan",
            "planning",
            "implement",
            "implementation",
            "feature",
            "refactor",
            "architecture",
            "design",
            "build",
            "新增",
            "实现",
            "改造",
            "重构",
            "优化",
            "架构",
            "设计",
            "方案",
            "规划",
            "计划",
            "下一步",
            "怎么做",
        ],
    ) {
        return Some(RepoPromptSkillRoute {
            name: "RepoPrompt Deep Plan",
            slug: "repoprompt-deep-plan",
            reason: "task asks for planning or implementation",
        });
    }

    if contains_any(
        task,
        &lower,
        &[
            "investigate",
            "investigation",
            "explore",
            "inspect",
            "trace",
            "understand",
            "summarize this repo",
            "analyze",
            "分析",
            "调研",
            "梳理",
            "理解",
            "看看",
            "调用链",
            "怎么设计",
            "代码库",
        ],
    ) {
        return Some(RepoPromptSkillRoute {
            name: "RepoPrompt Investigate",
            slug: "repoprompt-investigate",
            reason: "task asks for codebase investigation",
        });
    }

    None
}

pub fn load_routed_repoprompt_skill(
    skills_dir: impl AsRef<Path>,
    task: &str,
) -> Result<Option<RoutedSkill>> {
    let Some(route) = route_repoprompt_skill(task) else {
        return Ok(None);
    };
    let skills_dir = skills_dir.as_ref();
    let path = skills_dir.join(route.slug).join("SKILL.md");
    if !path.is_file() {
        return Ok(None);
    }
    let body = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let info = skill_info_from_body(skills_dir, path, &body);
    Ok(Some(RoutedSkill {
        route,
        document: SkillDocument { info, body },
    }))
}

pub fn slugify(input: &str) -> String {
    let mut out = String::new();
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

fn skill_info_from_body(skills_dir: &Path, path: PathBuf, body: &str) -> SkillInfo {
    let (front_matter, content) = split_front_matter(body);
    let fallback_name = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .or_else(|| path.file_stem().and_then(|name| name.to_str()))
        .unwrap_or("unnamed-skill")
        .to_string();
    let name = front_matter
        .and_then(|front_matter| front_matter_value(front_matter, "name"))
        .or_else(|| {
            content
                .lines()
                .find_map(|line| line.trim().strip_prefix("# ").map(str::trim))
                .filter(|name| !name.is_empty())
                .map(ToString::to_string)
        })
        .unwrap_or(fallback_name);
    let description = front_matter
        .and_then(|front_matter| front_matter_value(front_matter, "description"))
        .or_else(|| {
            content
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .filter(|line| !line.starts_with('#'))
                .find(|line| !line.starts_with("Generated at:"))
                .map(ToString::to_string)
        })
        .unwrap_or_else(|| "No description recorded.".to_string());
    let task_type =
        front_matter.and_then(|front_matter| front_matter_value(front_matter, "task_type"));
    let capabilities = front_matter
        .map(|front_matter| front_matter_list(front_matter, "capabilities"))
        .unwrap_or_default();
    let required_tools = front_matter
        .map(|front_matter| {
            let mut tools = front_matter_list(front_matter, "required_tools");
            tools.extend(front_matter_list(front_matter, "tools"));
            dedupe_sorted(tools)
        })
        .unwrap_or_default();
    let preferred_backend =
        front_matter.and_then(|front_matter| front_matter_value(front_matter, "preferred_backend"));
    let autonomous_safe = front_matter
        .and_then(|front_matter| front_matter_bool(front_matter, "autonomous_safe"))
        .unwrap_or_else(default_autonomous_safe);
    let blast_radius =
        front_matter.and_then(|front_matter| front_matter_value(front_matter, "blast_radius"));
    let mut tags = BTreeSet::new();
    if let Some(front_matter) = front_matter {
        for tag in front_matter_list(front_matter, "tags") {
            tags.insert(tag);
        }
    }
    for term in terms(&name) {
        tags.insert(term);
    }
    if let Some(task_type) = &task_type {
        tags.insert(task_type.to_ascii_lowercase());
    }
    if let Some(preferred_backend) = &preferred_backend {
        tags.insert(preferred_backend.to_ascii_lowercase());
    }
    for value in capabilities.iter().chain(required_tools.iter()) {
        tags.insert(value.to_ascii_lowercase());
    }
    for tool in tools_used(content) {
        tags.insert(tool);
    }
    let source = path
        .strip_prefix(skills_dir)
        .ok()
        .and_then(|rel| rel.components().next())
        .map(|_| "local")
        .unwrap_or("external")
        .to_string();

    SkillInfo {
        name,
        path,
        description,
        tags: tags.into_iter().collect(),
        source,
        task_type,
        capabilities,
        required_tools,
        preferred_backend,
        autonomous_safe,
        blast_radius,
    }
}

fn split_front_matter(body: &str) -> (Option<&str>, &str) {
    let Some(rest) = body.strip_prefix("---\n") else {
        return (None, body);
    };
    let Some(end) = rest.find("\n---") else {
        return (None, body);
    };
    let front_matter = &rest[..end];
    let content = &rest[end + "\n---".len()..];
    (
        Some(front_matter),
        content.strip_prefix('\n').unwrap_or(content),
    )
}

fn front_matter_value(front_matter: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}:");
    front_matter.lines().find_map(|line| {
        let line = line.trim();
        let value = line.strip_prefix(&prefix)?.trim();
        let value = value.trim_matches('"').trim_matches('\'');
        (!value.is_empty()).then(|| value.to_string())
    })
}

fn front_matter_list(front_matter: &str, key: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut in_list = false;
    let prefix = format!("{key}:");
    for line in front_matter.lines() {
        let trimmed = line.trim();
        if let Some(value) = trimmed.strip_prefix(&prefix) {
            in_list = true;
            let value = value.trim();
            if value.starts_with('[') && value.ends_with(']') {
                for tag in value
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .split(',')
                {
                    push_metadata_value(&mut values, tag);
                }
            } else if !value.is_empty() {
                push_metadata_value(&mut values, value);
            }
            continue;
        }
        if in_list && trimmed.starts_with("- ") {
            push_metadata_value(&mut values, trimmed.trim_start_matches("- "));
            continue;
        }
        if in_list && trimmed.contains(':') {
            in_list = false;
        }
    }
    dedupe_sorted(values)
}

fn front_matter_bool(front_matter: &str, key: &str) -> Option<bool> {
    let value = front_matter_value(front_matter, key)?;
    match value.to_ascii_lowercase().as_str() {
        "true" | "yes" | "1" => Some(true),
        "false" | "no" | "0" => Some(false),
        _ => None,
    }
}

fn push_metadata_value(values: &mut Vec<String>, value: &str) {
    let value = value.trim().trim_matches('"').trim_matches('\'');
    if !value.is_empty() {
        values.push(value.to_string());
    }
}

fn dedupe_sorted(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn default_autonomous_safe() -> bool {
    true
}

fn skill_search_haystack(info: &SkillInfo) -> String {
    format!(
        "{} {} {} {} {} {} {} {}",
        info.name,
        info.description,
        info.tags.join(" "),
        info.task_type.as_deref().unwrap_or_default(),
        info.capabilities.join(" "),
        info.required_tools.join(" "),
        info.preferred_backend.as_deref().unwrap_or_default(),
        info.blast_radius.as_deref().unwrap_or_default(),
    )
}

fn contains_any(task: &str, lower: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| {
        if needle.is_ascii() {
            lower.contains(needle)
        } else {
            task.contains(needle)
        }
    })
}

fn terms(input: &str) -> Vec<String> {
    input
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-')
        .map(str::trim)
        .filter(|term| !term.is_empty())
        .map(|term| term.to_ascii_lowercase())
        .collect()
}

fn tools_used(body: &str) -> Vec<String> {
    let mut tools = Vec::new();
    let mut in_tools = false;
    for line in body.lines().map(str::trim) {
        if line.starts_with("## ") {
            in_tools = line.eq_ignore_ascii_case("## Tools Used");
            continue;
        }
        if in_tools && let Some(tool) = line.strip_prefix("- ") {
            let tool = tool.trim();
            if !tool.is_empty() && !tool.eq_ignore_ascii_case("none recorded") {
                tools.push(tool.to_string());
            }
        }
    }
    tools
}

#[derive(Debug, Clone)]
struct RunLearningContext {
    goal: String,
    tools: Vec<String>,
    final_summary: String,
    key_terms: BTreeSet<String>,
}

impl RunLearningContext {
    fn from_records(records: &[SessionRecord]) -> Self {
        let mut goal = String::new();
        let mut tools = BTreeSet::new();
        let mut final_summary = String::new();
        let mut checkpoints = Vec::new();
        for record in records {
            match &record.event {
                AgentEvent::RunStarted { goal: value, .. } => goal = value.clone(),
                AgentEvent::ToolStarted { call } => {
                    tools.insert(call.name.clone());
                }
                AgentEvent::TurnSummary { summary, .. }
                | AgentEvent::Reflection { summary }
                | AgentEvent::RunFinished { summary, .. } => {
                    if matches!(record.event, AgentEvent::RunFinished { .. }) {
                        final_summary = summary.clone();
                    } else {
                        checkpoints.push(summary.clone());
                    }
                }
                AgentEvent::CheckpointUpdated { key_info, .. } => {
                    checkpoints.push(key_info.clone())
                }
                AgentEvent::LongTermUpdateStarted { reason, evidence } => {
                    checkpoints.push(reason.clone());
                    if let Some(evidence) = evidence {
                        checkpoints.push(evidence.clone());
                    }
                }
                AgentEvent::LongTermUpdateSettled {
                    decision,
                    reason,
                    evidence,
                    ..
                } => {
                    checkpoints.push(decision.clone());
                    checkpoints.push(reason.clone());
                    if let Some(evidence) = evidence {
                        checkpoints.push(evidence.clone());
                    }
                }
                AgentEvent::ToolFinished { .. } => {}
            }
        }
        let tools = tools.into_iter().collect::<Vec<_>>();
        let mut key_terms = BTreeSet::new();
        for text in std::iter::once(goal.as_str())
            .chain(std::iter::once(final_summary.as_str()))
            .chain(checkpoints.iter().map(String::as_str))
        {
            for term in meaningful_terms(text) {
                key_terms.insert(term);
            }
        }
        for tool in &tools {
            key_terms.insert(tool.to_ascii_lowercase());
        }
        Self {
            goal,
            tools,
            final_summary,
            key_terms,
        }
    }
}

fn best_skill_match(
    run: &RunLearningContext,
    candidates: &[SkillInfo],
) -> Option<(usize, SkillInfo)> {
    let mut scored = candidates
        .iter()
        .filter_map(|info| {
            let mut terms = BTreeSet::new();
            for term in meaningful_terms(&skill_search_haystack(info)) {
                terms.insert(term);
            }
            let overlap = run.key_terms.intersection(&terms).count();
            let tool_overlap = run
                .tools
                .iter()
                .filter(|tool| info.tags.contains(tool))
                .count();
            let score = overlap + tool_overlap * 2;
            (score >= 3).then_some((score, info.clone()))
        })
        .collect::<Vec<_>>();
    scored.sort_by(|(a_score, a), (b_score, b)| {
        b_score
            .cmp(a_score)
            .then_with(|| a.name.cmp(&b.name))
            .then_with(|| a.path.cmp(&b.path))
    });
    scored.into_iter().next()
}

fn learned_update_markdown(run: &RunLearningContext) -> String {
    let tools = if run.tools.is_empty() {
        "none recorded".to_string()
    } else {
        run.tools.join(", ")
    };
    format!(
        "- {}: goal=`{}`; tools={}; summary={}",
        Utc::now().format("%Y-%m-%d"),
        compact_inline(&run.goal, 160),
        tools,
        compact_inline(&run.final_summary, 260)
    )
}

fn append_learned_update(path: &Path, update: &str) -> Result<()> {
    let body = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    if body.contains(update) {
        return Ok(());
    }
    let new_body = if body.contains("## Learned Updates") {
        append_after_heading(&body, "## Learned Updates", update)
    } else if body.contains("## Memory Rule") {
        body.replacen(
            "## Memory Rule",
            &format!("## Learned Updates\n{update}\n\n## Memory Rule"),
            1,
        )
    } else {
        format!("{}\n\n## Learned Updates\n{}\n", body.trim_end(), update)
    };
    fs::write(path, new_body).with_context(|| format!("write {}", path.display()))
}

fn append_after_heading(body: &str, heading: &str, update: &str) -> String {
    let mut out = String::new();
    let mut inserted = false;
    let mut in_section = false;
    for line in body.lines() {
        if line.trim() == heading {
            in_section = true;
            out.push_str(line);
            out.push('\n');
            continue;
        }
        if in_section && !inserted && line.starts_with("## ") {
            out.push_str(update);
            out.push('\n');
            inserted = true;
            in_section = false;
        }
        out.push_str(line);
        out.push('\n');
    }
    if in_section && !inserted {
        out.push_str(update);
        out.push('\n');
    }
    out
}

fn meaningful_terms(input: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "then",
        "into",
        "from",
        "after",
        "before",
        "when",
        "use",
        "used",
        "using",
        "skill",
        "session",
        "finished",
        "completed",
        "summary",
        "goal",
        "turn",
        "tools",
        "update",
        "updates",
        "existing",
        "verified",
    ];
    terms(input)
        .into_iter()
        .filter(|term| term.len() >= 3)
        .filter(|term| !STOP.contains(&term.as_str()))
        .collect()
}

fn compact_inline(text: &str, limit: usize) -> String {
    let mut text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.len() > limit {
        text.truncate(limit);
        text.push_str(" ...");
    }
    text.replace('`', "'")
}

fn compact_json(value: &serde_json::Value) -> String {
    let text = value.to_string();
    if text.len() <= 240 {
        text
    } else {
        format!("{} ... {}", &text[..120], &text[text.len() - 80..])
    }
}

fn bullet_list(items: impl IntoIterator<Item = String>) -> String {
    let lines = items.into_iter().collect::<Vec<_>>();
    if lines.is_empty() {
        "- None recorded".to_string()
    } else {
        lines
            .into_iter()
            .map(|item| format!("- {item}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn bullet_list_or_none(items: impl IntoIterator<Item = String>) -> String {
    bullet_list(items)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::{ToolCall, ToolResult};
    use chrono::Utc;
    use std::fs;

    #[test]
    fn lists_searches_and_fetches_skill_metadata() {
        let root = std::env::temp_dir().join(format!("seed-skill-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("demo")).unwrap();
        fs::write(
            root.join("demo").join("SKILL.md"),
            "# Demo Skill\n\nUse this skill for demo workflows.\n\n## Tools Used\n- run_shell\n",
        )
        .unwrap();

        let infos = list_skill_infos(&root).unwrap();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].name, "Demo Skill");
        assert!(infos[0].tags.contains(&"run_shell".to_string()));

        let fetched = fetch_skill(&root, "demo-skill").unwrap();
        assert!(fetched.body.contains("demo workflows"));

        let results = search_skill_infos(&root, "shell demo", 5).unwrap();
        assert_eq!(results[0].name, "Demo Skill");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn reads_front_matter_metadata() {
        let root =
            std::env::temp_dir().join(format!("seed-skill-front-matter-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("repoprompt-deep-plan")).unwrap();
        fs::write(
            root.join("repoprompt-deep-plan").join("SKILL.md"),
            "---\nname: RepoPrompt Deep Plan\ndescription: Use RepoPrompt builder for grounded implementation plans.\ntask_type: implementation\ncapabilities: [context-build, planning]\nrequired_tools:\n  - RepoPrompt\npreferred_backend: repoprompt\nautonomous_safe: true\nblast_radius: low\ntags: [repoprompt, plan]\n---\n\n# RepoPrompt Deep Plan\n\nUse this skill for planning.\n",
        )
        .unwrap();

        let infos = list_skill_infos(&root).unwrap();

        assert_eq!(infos[0].name, "RepoPrompt Deep Plan");
        assert_eq!(
            infos[0].description,
            "Use RepoPrompt builder for grounded implementation plans."
        );
        assert!(infos[0].tags.contains(&"repoprompt".to_string()));
        assert!(infos[0].tags.contains(&"plan".to_string()));
        assert_eq!(infos[0].task_type.as_deref(), Some("implementation"));
        assert_eq!(infos[0].preferred_backend.as_deref(), Some("repoprompt"));
        assert_eq!(infos[0].blast_radius.as_deref(), Some("low"));
        assert!(infos[0].autonomous_safe);
        assert!(infos[0].capabilities.contains(&"context-build".to_string()));
        assert!(infos[0].required_tools.contains(&"RepoPrompt".to_string()));
        assert!(infos[0].tags.contains(&"context-build".to_string()));
        assert!(infos[0].tags.contains(&"implementation".to_string()));
        let results = search_skill_infos(&root, "context-build repoprompt", 5).unwrap();
        assert_eq!(results[0].name, "RepoPrompt Deep Plan");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn routes_repoprompt_skills_by_task_intent() {
        assert_eq!(
            route_repoprompt_skill("帮我实现新的计划流程").unwrap().slug,
            "repoprompt-deep-plan"
        );
        assert_eq!(
            route_repoprompt_skill("review this code for risks")
                .unwrap()
                .slug,
            "repoprompt-review"
        );
        assert_eq!(
            route_repoprompt_skill("深入分析一下这个代码库")
                .unwrap()
                .slug,
            "repoprompt-investigate"
        );
        assert!(route_repoprompt_skill("say pong").is_none());
    }

    #[test]
    fn consolidate_updates_similar_existing_skill() {
        let root = std::env::temp_dir().join(format!(
            "seed-skill-consolidate-update-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("shell-workflow")).unwrap();
        fs::write(
            root.join("shell-workflow").join("SKILL.md"),
            "# shell-workflow\n\nUse this skill for shell workflow checks.\n\n## Tools Used\n- run_shell\n\n## Memory Rule\nOnly carry forward verified facts.\n",
        )
        .unwrap();

        let records = learning_records("Run a shell workflow check", "run_shell");
        let result = consolidate_skill(&root, "new-shell-workflow", &records).unwrap();
        let body = fs::read_to_string(root.join("shell-workflow").join("SKILL.md")).unwrap();

        assert_eq!(result.decision, SkillConsolidationDecision::Updated);
        assert!(body.contains("## Learned Updates"));
        assert!(body.contains("Run a shell workflow check"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn consolidate_creates_new_skill_when_no_match_exists() {
        let root = std::env::temp_dir().join(format!(
            "seed-skill-consolidate-create-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        let records = learning_records("Analyze a memory index", "memory_search");
        let result = consolidate_skill(&root, "memory-index-analysis", &records).unwrap();

        assert_eq!(result.decision, SkillConsolidationDecision::Created);
        assert!(result.path.is_file());
        let _ = fs::remove_dir_all(&root);
    }

    fn learning_records(goal: &str, tool_name: &str) -> Vec<SessionRecord> {
        let call = ToolCall::new(tool_name, serde_json::json!({}));
        vec![
            record(AgentEvent::RunStarted {
                goal: goal.to_string(),
                cwd: std::env::current_dir().unwrap(),
            }),
            record(AgentEvent::ToolStarted { call: call.clone() }),
            record(AgentEvent::ToolFinished {
                result: ToolResult::ok(&call, serde_json::json!({ "status": "success" })),
            }),
            record(AgentEvent::RunFinished {
                status: "completed".to_string(),
                summary: format!("Finished after 1 turns: {goal}"),
            }),
        ]
    }

    fn record(event: AgentEvent) -> SessionRecord {
        SessionRecord {
            ts: Utc::now(),
            session_id: "test".to_string(),
            event,
        }
    }
}
