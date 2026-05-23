use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Library error surface for `agent-plan`.
///
/// Borrowing the forge_domain pattern: name the failure modes callers will
/// want to pattern-match on; keep `Other(anyhow::Error)` as the escape hatch
/// so internal `bail!` / `context()` chains still propagate via `?` without
/// forcing every internal site to be converted.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    /// Plan id was named but no plan with that id exists. Includes the
    /// store root so the caller can render a useful "ls $root" hint.
    #[error("plan not found: id={id} (in {root})")]
    PlanNotFound { id: String, root: PathBuf },

    /// Snapshot/active-brief lookup ran with no id and the store contains
    /// no plans at all.
    #[error("no plans found in {root}")]
    NoActivePlan { root: PathBuf },

    /// `complete` / item-targeting operation referenced an item index that
    /// doesn't exist in the plan body.
    #[error("plan item not found: index={index}")]
    ItemNotFound { index: usize },

    /// `complete` was called with no item and the plan has no unchecked
    /// items left to complete.
    #[error("plan has no unchecked items")]
    NoUncheckedItems,

    /// `complete` targeted an item that's already checked off.
    #[error("plan item {index} is already complete")]
    AlreadyComplete { index: usize },

    /// `append_items` was called with an empty slice; reject explicitly
    /// instead of producing a no-op write.
    #[error("append_items requires at least one item")]
    EmptyAppend,

    /// Escape hatch for IO / serde / unstructured failures. Internal code
    /// keeps using `anyhow::Result`; this variant absorbs them at the API
    /// boundary via `#[from]`.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Library-facing result alias. Internal code keeps using `anyhow::Result`
/// (which is still in scope as `Result`); public functions opt into the
/// typed surface by returning `PlanResult<T>`. The `Other(anyhow::Error)`
/// variant means `?` from internal anyhow code converts automatically.
pub type PlanResult<T> = std::result::Result<T, PlanError>;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    Active,
    PendingVerification,
    Verified,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanState {
    pub id: String,
    pub title: String,
    pub task: String,
    pub status: PlanStatus,
    pub plan_path: PathBuf,
    pub state_path: PathBuf,
    pub verify_context_path: PathBuf,
    pub repoprompt_export_path: Option<PathBuf>,
    pub source_export_path: Option<PathBuf>,
    #[serde(default)]
    pub orchestration: PlanOrchestration,
    pub verification_report: Option<String>,
    pub verify_attempts: usize,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanBackend {
    Local,
    #[default]
    RepoPrompt,
    Codex,
    Mixed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanArtifactKind {
    RepoPromptExport,
    ContextExport,
    VerificationContext,
    VerificationReport,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanArtifact {
    pub kind: PlanArtifactKind,
    pub path: PathBuf,
    pub note: Option<String>,
    pub recorded_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanHandoff {
    pub backend: String,
    pub role: Option<String>,
    pub run_id: Option<String>,
    pub thread_id: Option<String>,
    pub artifact_path: Option<PathBuf>,
    pub status: String,
    pub summary: String,
    pub recorded_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanVerificationRecord {
    pub attempt: usize,
    pub verdict: String,
    pub report: String,
    pub recorded_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanOrchestration {
    pub preferred_backend: PlanBackend,
    pub source_export_path: Option<PathBuf>,
    pub repoprompt_export_path: Option<PathBuf>,
    pub artifacts: Vec<PlanArtifact>,
    pub handoffs: Vec<PlanHandoff>,
    pub verification_records: Vec<PlanVerificationRecord>,
}

impl Default for PlanOrchestration {
    fn default() -> Self {
        Self {
            preferred_backend: PlanBackend::RepoPrompt,
            source_export_path: None,
            repoprompt_export_path: None,
            artifacts: Vec::new(),
            handoffs: Vec::new(),
            verification_records: Vec::new(),
        }
    }
}

impl PlanOrchestration {
    fn from_exports(
        source_export_path: Option<PathBuf>,
        repoprompt_export_path: Option<PathBuf>,
    ) -> Self {
        let mut orchestration = Self {
            source_export_path,
            repoprompt_export_path: repoprompt_export_path.clone(),
            ..Self::default()
        };
        if let Some(path) = repoprompt_export_path {
            orchestration.record_artifact_once(
                PlanArtifactKind::RepoPromptExport,
                path,
                Some("initial RepoPrompt export copied or referenced at plan creation".to_string()),
            );
        }
        orchestration
    }

    fn record_artifact_once(
        &mut self,
        kind: PlanArtifactKind,
        path: PathBuf,
        note: Option<String>,
    ) {
        if self
            .artifacts
            .iter()
            .any(|artifact| artifact.kind == kind && artifact.path == path)
        {
            return;
        }
        self.artifacts.push(PlanArtifact {
            kind,
            path,
            note,
            recorded_at: Utc::now(),
        });
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanItem {
    pub index: usize,
    pub line_number: usize,
    pub checked: bool,
    pub kind: PlanItemKind,
    pub text: String,
    #[serde(default)]
    pub delegate: bool,
    #[serde(default)]
    pub parallel: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanItemKind {
    Task,
    Verify,
    Fix,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanSnapshot {
    pub state: PlanState,
    pub items: Vec<PlanItem>,
    pub next_item: Option<PlanItem>,
    pub unchecked_count: usize,
    pub task_unchecked_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanBrief {
    pub plan_id: String,
    pub title: String,
    pub task: String,
    pub status: PlanStatus,
    pub task_unchecked_count: usize,
    pub unchecked_count: usize,
    pub next_item: Option<PlanItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VerificationContext {
    pub plan_id: String,
    pub task: String,
    pub plan_file: PathBuf,
    pub state_file: PathBuf,
    pub repoprompt_export_path: Option<PathBuf>,
    pub orchestration: PlanOrchestration,
    pub required_checks: Vec<String>,
    pub instructions: String,
}

#[derive(Debug, Clone)]
pub struct CreatePlan {
    pub title: String,
    pub task: String,
    pub steps: Vec<String>,
    pub source_export_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct RecordPlanArtifact {
    pub kind: PlanArtifactKind,
    pub path: PathBuf,
    pub note: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RecordPlanHandoff {
    pub backend: String,
    pub role: Option<String>,
    pub run_id: Option<String>,
    pub thread_id: Option<String>,
    pub artifact_path: Option<PathBuf>,
    pub status: String,
    pub summary: String,
}

#[derive(Debug, Clone)]
pub struct PlanStore {
    root: PathBuf,
}

impl PlanStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn create(&self, input: CreatePlan) -> PlanResult<PlanSnapshot> {
        fs::create_dir_all(&self.root)
            .with_context(|| format!("create {}", self.root.display()))?;
        let title = clean_title(&input.title).unwrap_or_else(|| "Plan".to_string());
        let id = unique_plan_id(&self.root, &title);
        let dir = self.root.join(&id);
        fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;

        let plan_path = dir.join("plan.md");
        let state_path = dir.join("state.json");
        let verify_context_path = dir.join("verify_context.json");
        let source_export_path = input.source_export_path;
        let repoprompt_export_path = copy_export(&dir, source_export_path.as_deref())?;
        let orchestration = PlanOrchestration::from_exports(
            source_export_path.clone(),
            repoprompt_export_path.clone(),
        );
        let now = Utc::now();
        let state = PlanState {
            id,
            title,
            task: input.task.trim().to_string(),
            status: PlanStatus::Active,
            plan_path,
            state_path,
            verify_context_path,
            repoprompt_export_path,
            source_export_path,
            orchestration,
            verification_report: None,
            verify_attempts: 0,
            created_at: now,
            updated_at: now,
        };
        let steps = normalize_steps(input.steps);
        fs::write(&state.plan_path, render_plan_markdown(&state, &steps))
            .with_context(|| format!("write {}", state.plan_path.display()))?;
        self.write_state(&state)?;
        self.snapshot(Some(&state.id))
    }

    pub fn snapshot(&self, id: Option<&str>) -> PlanResult<PlanSnapshot> {
        let mut state = self.load_state(id)?;
        let items = parse_plan_items(
            &fs::read_to_string(&state.plan_path)
                .with_context(|| format!("read {}", state.plan_path.display()))?,
        );
        state.status = inferred_status(state.status, &items);
        self.write_state(&state)?;
        let next_item = items.iter().find(|item| !item.checked).cloned();
        let unchecked_count = items.iter().filter(|item| !item.checked).count();
        let task_unchecked_count = items
            .iter()
            .filter(|item| !item.checked && item.kind != PlanItemKind::Verify)
            .count();
        Ok(PlanSnapshot {
            state,
            items,
            next_item,
            unchecked_count,
            task_unchecked_count,
        })
    }

    pub fn active_brief(&self) -> PlanResult<Option<PlanBrief>> {
        if !self.root.exists() {
            return Ok(None);
        }
        let snapshots = self.list()?;
        let Some(snapshot) = snapshots.into_iter().find(|snap| {
            matches!(
                snap.state.status,
                PlanStatus::Active | PlanStatus::PendingVerification
            )
        }) else {
            return Ok(None);
        };
        Ok(Some(PlanBrief {
            plan_id: snapshot.state.id,
            title: snapshot.state.title,
            task: snapshot.state.task,
            status: snapshot.state.status,
            task_unchecked_count: snapshot.task_unchecked_count,
            unchecked_count: snapshot.unchecked_count,
            next_item: snapshot.next_item,
        }))
    }

    pub fn list(&self) -> PlanResult<Vec<PlanSnapshot>>{
        if !self.root.exists() {
            return Ok(Vec::new());
        }

        let mut ids = fs::read_dir(&self.root)
            .with_context(|| format!("read {}", self.root.display()))?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().join("state.json").is_file())
            .filter_map(|entry| entry.file_name().to_str().map(ToString::to_string))
            .collect::<Vec<_>>();
        ids.sort();

        let mut snapshots = ids
            .into_iter()
            .map(|id| self.snapshot(Some(&id)))
            .collect::<PlanResult<Vec<_>>>()?;
        snapshots.sort_by(|left, right| {
            right
                .state
                .updated_at
                .cmp(&left.state.updated_at)
                .then_with(|| left.state.id.cmp(&right.state.id))
        });
        Ok(snapshots)
    }

    pub fn complete(
        &self,
        id: Option<&str>,
        item_index: Option<usize>,
        note: Option<&str>,
    ) -> PlanResult<PlanSnapshot> {
        let snapshot = self.snapshot(id)?;
        let item = match item_index {
            Some(index) => snapshot
                .items
                .iter()
                .find(|item| item.index == index)
                .cloned()
                .ok_or(PlanError::ItemNotFound { index })?,
            None => snapshot
                .next_item
                .clone()
                .ok_or(PlanError::NoUncheckedItems)?,
        };
        if item.checked {
            return Err(PlanError::AlreadyComplete { index: item.index });
        }
        let body = fs::read_to_string(&snapshot.state.plan_path)
            .with_context(|| format!("read {}", snapshot.state.plan_path.display()))?;
        let mut lines = body.lines().map(ToString::to_string).collect::<Vec<_>>();
        let Some(line) = lines.get_mut(item.line_number - 1) else {
            return Err(PlanError::Other(anyhow::anyhow!(
                "plan item line missing: {}",
                item.line_number
            )));
        };
        *line = mark_checked(line)?;
        if let Some(note) = note.filter(|note| !note.trim().is_empty()) {
            lines.push(format!(
                "\n> Completed item {} at {}: {}",
                item.index,
                Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
                note.trim()
            ));
        }
        fs::write(&snapshot.state.plan_path, format!("{}\n", lines.join("\n")))
            .with_context(|| format!("write {}", snapshot.state.plan_path.display()))?;
        let mut state = snapshot.state;
        state.updated_at = Utc::now();
        self.write_state(&state)?;
        self.snapshot(Some(&state.id))
    }

    pub fn append_items(
        &self,
        id: Option<&str>,
        items: Vec<String>,
    ) -> PlanResult<PlanSnapshot> {
        if items.is_empty() {
            return Err(PlanError::EmptyAppend);
        }
        let snapshot = self.snapshot(id)?;
        let plan_path = &snapshot.state.plan_path;
        let body = fs::read_to_string(plan_path)
            .with_context(|| format!("read {}", plan_path.display()))?;
        let lines: Vec<&str> = body.lines().collect();
        let verify_idx = lines
            .iter()
            .position(|line| line.contains("- [ ]") && line.contains("[VERIFY]"));
        let next_number = snapshot
            .items
            .iter()
            .filter_map(|item| {
                let (head, _) = item.text.split_once(". ")?;
                head.trim()
                    .trim_start_matches(['[', 'D', 'P', ']', ' '])
                    .parse::<usize>()
                    .ok()
            })
            .max()
            .unwrap_or(snapshot.items.len())
            + 1;

        let mut new_lines: Vec<String> = lines
            .iter()
            .take(verify_idx.unwrap_or(lines.len()))
            .map(|s| s.to_string())
            .collect();
        for (offset, item) in items.iter().enumerate() {
            let trimmed = item.trim();
            if trimmed.is_empty() {
                continue;
            }
            let text = if trimmed.contains("[FIX]") {
                trimmed.to_string()
            } else {
                format!("[FIX] {trimmed}")
            };
            new_lines.push(format!("- [ ] {}. {text}", next_number + offset));
        }
        if let Some(idx) = verify_idx {
            new_lines.extend(lines.iter().skip(idx).map(|s| s.to_string()));
        }
        let trailing_newline = if body.ends_with('\n') { "\n" } else { "" };
        fs::write(plan_path, format!("{}{trailing_newline}", new_lines.join("\n")))
            .with_context(|| format!("write {}", plan_path.display()))?;

        let mut state = snapshot.state;
        state.updated_at = Utc::now();
        self.write_state(&state)?;
        self.snapshot(Some(&state.id))
    }

    pub fn record_artifact(
        &self,
        id: Option<&str>,
        input: RecordPlanArtifact,
    ) -> PlanResult<PlanSnapshot> {
        let mut snapshot = self.snapshot(id)?;
        snapshot
            .state
            .orchestration
            .record_artifact_once(input.kind, input.path, input.note);
        snapshot.state.updated_at = Utc::now();
        self.write_state(&snapshot.state)?;
        self.snapshot(Some(&snapshot.state.id))
    }

    pub fn record_handoff(
        &self,
        id: Option<&str>,
        input: RecordPlanHandoff,
    ) -> PlanResult<PlanSnapshot> {
        let mut snapshot = self.snapshot(id)?;
        snapshot.state.orchestration.handoffs.push(PlanHandoff {
            backend: input.backend,
            role: input.role,
            run_id: input.run_id,
            thread_id: input.thread_id,
            artifact_path: input.artifact_path,
            status: input.status,
            summary: input.summary,
            recorded_at: Utc::now(),
        });
        snapshot.state.updated_at = Utc::now();
        self.write_state(&snapshot.state)?;
        self.snapshot(Some(&snapshot.state.id))
    }

    pub fn write_verify_context(&self, id: Option<&str>) -> PlanResult<VerificationContext> {
        let mut snapshot = self.snapshot(id)?;
        if snapshot.task_unchecked_count > 0 {
            return Err(PlanError::Other(anyhow::anyhow!(
                "plan still has {} unfinished non-verify item(s)",
                snapshot.task_unchecked_count
            )));
        }
        snapshot.state.orchestration.record_artifact_once(
            PlanArtifactKind::VerificationContext,
            snapshot.state.verify_context_path.clone(),
            Some("verification context emitted for independent verifier".to_string()),
        );
        let context = VerificationContext {
            plan_id: snapshot.state.id.clone(),
            task: snapshot.state.task.clone(),
            plan_file: snapshot.state.plan_path.clone(),
            state_file: snapshot.state.state_path.clone(),
            repoprompt_export_path: snapshot.state.repoprompt_export_path.clone(),
            orchestration: snapshot.state.orchestration.clone(),
            required_checks: vec![
                "Read plan.md and verify every checked task against the repository.".to_string(),
                "Run or inspect the narrow checks named by the plan when possible.".to_string(),
                "Return a line beginning with VERDICT: PASS or VERDICT: FAIL.".to_string(),
            ],
            instructions: "Independent verification gate. Do not trust the executor summary; verify from files, git diff, tests, and artifacts. Report concrete failures with file paths and commands.".to_string(),
        };
        fs::write(
            &snapshot.state.verify_context_path,
            serde_json::to_string_pretty(&context).context("serialize verify context")?,
        )
        .with_context(|| format!("write {}", snapshot.state.verify_context_path.display()))?;
        let mut state = snapshot.state;
        state.status = PlanStatus::PendingVerification;
        state.updated_at = Utc::now();
        self.write_state(&state)?;
        Ok(context)
    }

    pub fn record_verification(&self, id: Option<&str>, report: &str) -> PlanResult<PlanSnapshot> {
        let mut snapshot = self.snapshot(id)?;
        let report = report.trim().to_string();
        snapshot.state.verify_attempts += 1;
        snapshot.state.verification_report = Some(report.clone());
        let verify_attempts = snapshot.state.verify_attempts;
        let verdict = verdict_from_report(&report);
        let verdict_label = match verdict {
            VerificationVerdict::Pass => "pass",
            VerificationVerdict::Fail => "fail",
            VerificationVerdict::Unknown => "unknown",
        }
        .to_string();
        snapshot
            .state
            .orchestration
            .verification_records
            .push(PlanVerificationRecord {
                attempt: verify_attempts,
                verdict: verdict_label,
                report: report.clone(),
                recorded_at: Utc::now(),
            });
        let orchestration = snapshot.state.orchestration.clone();
        match verdict {
            VerificationVerdict::Pass => {
                snapshot.state.status = PlanStatus::Verified;
                if let Some(item) = snapshot
                    .items
                    .iter()
                    .find(|item| item.kind == PlanItemKind::Verify && !item.checked)
                    .cloned()
                {
                    self.complete(
                        Some(&snapshot.state.id),
                        Some(item.index),
                        Some("verification passed"),
                    )?;
                    snapshot = self.snapshot(Some(&snapshot.state.id))?;
                    snapshot.state.verify_attempts = verify_attempts;
                    snapshot.state.verification_report = Some(report.clone());
                    snapshot.state.status = PlanStatus::Verified;
                    snapshot.state.orchestration = orchestration.clone();
                }
            }
            VerificationVerdict::Fail => {
                snapshot.state.status = PlanStatus::Failed;
                append_fix_item(&snapshot.state.plan_path, &report)?;
            }
            VerificationVerdict::Unknown => {
                snapshot.state.status = PlanStatus::PendingVerification;
            }
        }
        snapshot.state.orchestration = orchestration;
        snapshot.state.updated_at = Utc::now();
        self.write_state(&snapshot.state)?;
        self.snapshot(Some(&snapshot.state.id))
    }

    fn load_state(&self, id: Option<&str>) -> Result<PlanState> {
        let id = match id {
            Some(id) => id.to_string(),
            None => self.latest_id()?,
        };
        let path = self.root.join(&id).join("state.json");
        let body = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&body).with_context(|| format!("parse {}", path.display()))
    }

    fn latest_id(&self) -> Result<String> {
        let mut plans = fs::read_dir(&self.root)
            .with_context(|| format!("read {}", self.root.display()))?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().join("state.json").is_file())
            .collect::<Vec<_>>();
        plans.sort_by_key(|entry| entry.metadata().and_then(|meta| meta.modified()).ok());
        plans
            .into_iter()
            .rev()
            .find_map(|entry| entry.file_name().to_str().map(ToString::to_string))
            .ok_or_else(|| anyhow::anyhow!("no plans found in {}", self.root.display()))
    }

    fn write_state(&self, state: &PlanState) -> Result<()> {
        fs::write(&state.state_path, serde_json::to_string_pretty(state)?)
            .with_context(|| format!("write {}", state.state_path.display()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerificationVerdict {
    Pass,
    Fail,
    Unknown,
}

fn verdict_from_report(report: &str) -> VerificationVerdict {
    let upper = report.to_ascii_uppercase();
    if upper.contains("VERDICT: PASS") || upper.contains("VERDICT PASS") {
        VerificationVerdict::Pass
    } else if upper.contains("VERDICT: FAIL") || upper.contains("VERDICT FAIL") {
        VerificationVerdict::Fail
    } else {
        VerificationVerdict::Unknown
    }
}

fn unique_plan_id(root: &Path, title: &str) -> String {
    let mut slug = slugify(title);
    if slug.is_empty() {
        slug = "plan".to_string();
    }
    let suffix = Uuid::new_v4().to_string();
    let id = format!("{slug}-{}", &suffix[..8]);
    if !root.join(&id).exists() {
        id
    } else {
        format!("{slug}-{suffix}")
    }
}

fn clean_title(title: &str) -> Option<String> {
    let title = title.trim();
    (!title.is_empty()).then(|| title.to_string())
}

fn normalize_steps(steps: Vec<String>) -> Vec<String> {
    let steps = steps
        .into_iter()
        .map(|step| step.trim().trim_start_matches("- [ ]").trim().to_string())
        .filter(|step| !step.is_empty())
        .collect::<Vec<_>>();
    if steps.is_empty() {
        vec![
            "Use RepoPrompt to identify the relevant files and constraints.".to_string(),
            "Implement the smallest safe change.".to_string(),
            "Run focused verification.".to_string(),
        ]
    } else {
        steps
    }
}

fn copy_export(dir: &Path, source: Option<&Path>) -> Result<Option<PathBuf>> {
    let Some(source) = source else {
        return Ok(None);
    };
    let target = dir.join("repoprompt_export.md");
    if source.is_file() {
        fs::copy(source, &target)
            .with_context(|| format!("copy {} to {}", source.display(), target.display()))?;
        Ok(Some(target))
    } else {
        Ok(Some(source.to_path_buf()))
    }
}

fn render_plan_markdown(state: &PlanState, steps: &[String]) -> String {
    let mut out = format!(
        "# {}\n\nTask: {}\n\nStatus: active\n\n## Steps\n",
        state.title, state.task
    );
    for (idx, step) in steps.iter().enumerate() {
        out.push_str(&format!("- [ ] {}. {}\n", idx + 1, step));
    }
    out.push_str("- [ ] [VERIFY] Independent verification gate\n");
    if let Some(path) = &state.repoprompt_export_path {
        out.push_str("\n## RepoPrompt Export\n");
        out.push_str(&format!("- {}\n", path.display()));
    }
    out.push_str("\n## Orchestration Ledger\n");
    out.push_str("- preferred_backend: repoprompt\n");
    out.push_str("- executor: RepoPrompt performs context, planning, review, and verification work; Seed records artifacts, handoffs, and verdicts.\n");
    if let Some(path) = &state.repoprompt_export_path {
        out.push_str(&format!("- repoprompt_export: {}\n", path.display()));
    }
    out.push_str("\n## Notes\n");
    out
}

pub fn parse_plan_items(body: &str) -> Vec<PlanItem> {
    let mut items = Vec::new();
    for (line_idx, line) in body.lines().enumerate() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed
            .strip_prefix("- [ ]")
            .map(|text| (false, text))
            .or_else(|| trimmed.strip_prefix("- [x]").map(|text| (true, text)))
            .or_else(|| trimmed.strip_prefix("- [X]").map(|text| (true, text)))
        else {
            continue;
        };
        let text = rest.1.trim().to_string();
        let kind = if text.contains("[VERIFY]") {
            PlanItemKind::Verify
        } else if text.contains("[FIX]") {
            PlanItemKind::Fix
        } else {
            PlanItemKind::Task
        };
        let delegate = item_has_marker(&text, "D");
        let parallel = item_has_marker(&text, "P");
        items.push(PlanItem {
            index: items.len() + 1,
            line_number: line_idx + 1,
            checked: rest.0,
            kind,
            text,
            delegate,
            parallel,
        });
    }
    items
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImportedPlan {
    pub title: String,
    pub task: String,
    pub steps: Vec<String>,
    pub delegated_count: usize,
    pub parallel_count: usize,
}

pub fn import_repoprompt_plan(text: &str) -> ImportedPlan {
    let title = extract_first_h1(text).unwrap_or_else(|| "Imported RepoPrompt Plan".to_string());
    let task = extract_task_summary(text, &title);
    let raw_steps = extract_step_lines(text);
    let mut delegated_count = 0;
    let mut parallel_count = 0;
    let steps = raw_steps
        .into_iter()
        .map(|step| {
            let annotated = annotate_step(&step);
            if item_has_marker(&annotated, "D") {
                delegated_count += 1;
            }
            if item_has_marker(&annotated, "P") {
                parallel_count += 1;
            }
            annotated
        })
        .collect::<Vec<_>>();
    ImportedPlan {
        title,
        task,
        steps,
        delegated_count,
        parallel_count,
    }
}

fn extract_first_h1(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        line.trim()
            .strip_prefix("# ")
            .map(|rest| rest.trim().to_string())
            .filter(|title| !title.is_empty())
    })
}

fn extract_task_summary(text: &str, title: &str) -> String {
    // First non-heading, non-empty paragraph that does not start with a list marker.
    let mut paragraph = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !paragraph.is_empty() {
                break;
            }
            continue;
        }
        if trimmed.starts_with('#')
            || trimmed.starts_with("- ")
            || trimmed.starts_with("* ")
            || trimmed.starts_with("```")
        {
            continue;
        }
        if let Some(rest) = trimmed.split_once(". ") {
            // Skip "1. foo" numbered items in the summary search.
            if rest.0.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
        }
        if !paragraph.is_empty() {
            paragraph.push(' ');
        }
        paragraph.push_str(trimmed);
    }
    if paragraph.is_empty() {
        title.to_string()
    } else {
        paragraph
    }
}

pub fn parse_plan_review(text: &str) -> Vec<String> {
    let fix_headings = [
        "recommended fixes",
        "fixes",
        "action items",
        "recommendations",
        "suggested fixes",
        "follow-ups",
    ];
    let mut items = Vec::new();
    let mut in_fixes = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(heading) = trimmed.strip_prefix("## ") {
            let lower = heading.trim().to_ascii_lowercase();
            in_fixes = fix_headings.iter().any(|kw| lower.contains(kw));
            continue;
        }
        if trimmed.starts_with("# ") {
            in_fixes = false;
            continue;
        }
        if !in_fixes {
            continue;
        }
        if let Some(item) = parse_step_line(trimmed) {
            let cleaned = item.trim_start_matches('(').to_string();
            if cleaned.eq_ignore_ascii_case("none") || cleaned.eq_ignore_ascii_case("none)") {
                continue;
            }
            items.push(item);
        }
    }
    items
}

fn extract_step_lines(text: &str) -> Vec<String> {
    let plan_headings = ["plan", "steps", "implementation", "tasks", "todo", "actions"];
    let mut steps = Vec::new();
    let mut in_plan = false;
    let mut found_section = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(heading) = trimmed.strip_prefix("## ") {
            let lower = heading.trim().to_ascii_lowercase();
            in_plan = plan_headings.iter().any(|kw| lower.contains(kw));
            if in_plan {
                found_section = true;
            }
            continue;
        }
        if trimmed.starts_with("# ") {
            in_plan = false;
            continue;
        }
        if in_plan
            && let Some(step) = parse_step_line(trimmed)
        {
            steps.push(step);
        }
    }
    if found_section {
        return steps;
    }
    // Fallback: scan the whole document for top-level list items.
    text.lines()
        .filter_map(|line| parse_step_line(line.trim()))
        .collect()
}

fn parse_step_line(trimmed: &str) -> Option<String> {
    if let Some(rest) = trimmed
        .strip_prefix("- [ ] ")
        .or_else(|| trimmed.strip_prefix("- [x] "))
        .or_else(|| trimmed.strip_prefix("- [X] "))
    {
        return Some(strip_leading_number(rest.trim()));
    }
    if let Some(rest) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
    {
        return Some(strip_leading_number(rest.trim()));
    }
    let mut chars = trimmed.chars();
    let mut digits = String::new();
    for c in chars.by_ref() {
        if c.is_ascii_digit() {
            digits.push(c);
        } else if !digits.is_empty() && (c == '.' || c == ')') {
            let rest: String = chars.collect();
            let value = rest.trim();
            if value.is_empty() {
                return None;
            }
            return Some(value.to_string());
        } else {
            return None;
        }
    }
    None
}

fn strip_leading_number(text: &str) -> String {
    let mut chars = text.chars().peekable();
    let mut digits = String::new();
    while let Some(c) = chars.peek() {
        if c.is_ascii_digit() {
            digits.push(*c);
            chars.next();
        } else {
            break;
        }
    }
    if !digits.is_empty()
        && let Some(sep) = chars.peek()
        && (*sep == '.' || *sep == ')')
    {
        chars.next();
        let remainder: String = chars.collect();
        return remainder.trim().to_string();
    }
    text.to_string()
}

fn annotate_step(step: &str) -> String {
    let lower = step.to_ascii_lowercase();
    let delegate_keywords = [
        "across the codebase",
        "multiple files",
        "many files",
        "all files",
        "every file",
        "refactor",
        "rewrite",
        "investigate",
        "explore",
        "audit",
        "scan ",
        "throughout",
        "全面",
        "重构",
        "调研",
        "全部",
        "排查",
    ];
    let parallel_keywords = [
        "independent",
        "in parallel",
        "concurrently",
        "fan out",
        "并行",
        "同时",
    ];
    let mut markers: Vec<&str> = Vec::new();
    if delegate_keywords.iter().any(|kw| lower.contains(kw)) || step.chars().count() > 200 {
        markers.push("[D]");
    }
    if parallel_keywords.iter().any(|kw| lower.contains(kw)) {
        markers.push("[P]");
    }
    if markers.is_empty() {
        step.to_string()
    } else {
        format!("{} {}", markers.join(" "), step)
    }
}

fn item_has_marker(text: &str, marker: &str) -> bool {
    // Match `[D]` / `[P]` only when present as a standalone token, not inside
    // `[VERIFY]`, `[FIX]`, `[SOP:...]`, or arbitrary substrings.
    let needle = format!("[{marker}]");
    let mut idx = 0;
    while let Some(found) = text[idx..].find(&needle) {
        let absolute = idx + found;
        let before_ok = absolute == 0
            || text.as_bytes()[absolute - 1] == b' '
            || text.as_bytes()[absolute - 1] == b']';
        let after = absolute + needle.len();
        let after_ok = after == text.len()
            || text.as_bytes()[after] == b' '
            || text.as_bytes()[after] == b'['
            || text.as_bytes()[after] == b'.';
        if before_ok && after_ok {
            return true;
        }
        idx = absolute + needle.len();
    }
    false
}

fn mark_checked(line: &str) -> Result<String> {
    if line.contains("- [ ]") {
        Ok(line.replacen("- [ ]", "- [x]", 1))
    } else {
        bail!("line is not an unchecked plan item: {line}")
    }
}

fn inferred_status(current: PlanStatus, items: &[PlanItem]) -> PlanStatus {
    if current == PlanStatus::Verified || current == PlanStatus::Failed {
        return current;
    }
    let non_verify_unchecked = items
        .iter()
        .any(|item| !item.checked && item.kind != PlanItemKind::Verify);
    if non_verify_unchecked {
        PlanStatus::Active
    } else if items
        .iter()
        .any(|item| !item.checked && item.kind == PlanItemKind::Verify)
    {
        PlanStatus::PendingVerification
    } else {
        PlanStatus::Verified
    }
}

fn append_fix_item(plan_path: &Path, report: &str) -> Result<()> {
    let mut body =
        fs::read_to_string(plan_path).with_context(|| format!("read {}", plan_path.display()))?;
    body.push_str(&format!(
        "\n- [ ] [FIX] Address verification failure from {}.\n\n> Verification report: {}\n",
        Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
        compact_inline(report, 500)
    ));
    fs::write(plan_path, body).with_context(|| format!("write {}", plan_path.display()))
}

fn compact_inline(text: &str, limit: usize) -> String {
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

fn slugify(input: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plan_items_detects_delegate_and_parallel_markers() {
        let body = "\
- [ ] 1. [D] Investigate auth subsystem [SOP:repoprompt-investigate]
- [ ] 2. [P] Generate fixture A
- [ ] 3. [P] Generate fixture B
- [ ] 4. Run the test
- [ ] [VERIFY] Independent verification gate
";
        let items = parse_plan_items(body);
        assert_eq!(items.len(), 5);
        assert!(items[0].delegate);
        assert!(!items[0].parallel);
        assert!(items[1].parallel && !items[1].delegate);
        assert!(items[2].parallel);
        assert!(!items[3].delegate && !items[3].parallel);
        assert_eq!(items[4].kind, PlanItemKind::Verify);
        assert!(!items[4].delegate);
    }

    #[test]
    fn parse_plan_review_extracts_recommended_fixes_section() {
        let response = r#"## Findings
- step 3 leaves the cache half-cleared
- there's no rollback on failure

## Recommended Fixes
- Add a rollback step after the cache rewrite
- Verify the new TokenStore behaves under concurrent writes
- (none)

## Closing thoughts
ignore me
"#;
        let fixes = parse_plan_review(response);
        assert_eq!(
            fixes,
            vec![
                "Add a rollback step after the cache rewrite".to_string(),
                "Verify the new TokenStore behaves under concurrent writes".to_string(),
            ]
        );
    }

    #[test]
    fn parse_plan_review_handles_aliases_and_numbered_items() {
        let response = r#"## Action Items
1. Replace the global Mutex with RwLock
2. Investigate the failing CI job
"#;
        let fixes = parse_plan_review(response);
        assert_eq!(
            fixes,
            vec![
                "Replace the global Mutex with RwLock".to_string(),
                "Investigate the failing CI job".to_string(),
            ]
        );
    }

    #[test]
    fn append_items_inserts_before_verify_with_running_numbers() {
        let root = temp_root();
        let store = PlanStore::new(root.join("plans"));
        let snapshot = store
            .create(CreatePlan {
                title: "Refine Demo".to_string(),
                task: "demo".to_string(),
                steps: vec!["Step A".to_string(), "Step B".to_string()],
                source_export_path: None,
            })
            .unwrap();
        let updated = store
            .append_items(
                Some(&snapshot.state.id),
                vec![
                    "Add retry logic".to_string(),
                    "[FIX] Verify caller upgrades cleanly".to_string(),
                ],
            )
            .unwrap();
        let task_items = updated
            .items
            .iter()
            .filter(|item| item.kind != PlanItemKind::Verify)
            .collect::<Vec<_>>();
        assert_eq!(task_items.len(), 4);
        assert!(task_items[2].text.starts_with("3."));
        assert!(task_items[2].text.contains("[FIX]"));
        assert!(task_items[2].text.contains("Add retry logic"));
        assert_eq!(task_items[2].kind, PlanItemKind::Fix);
        assert!(task_items[3].text.starts_with("4."));
        assert_eq!(task_items[3].kind, PlanItemKind::Fix);
        let verify_items = updated
            .items
            .iter()
            .filter(|item| item.kind == PlanItemKind::Verify)
            .collect::<Vec<_>>();
        assert_eq!(verify_items.len(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn imports_repoprompt_plan_with_numbered_steps_and_markers() {
        let export = r#"# Refactor auth module

We want to split the legacy auth middleware into smaller layers so legal can sign off on the
session-token storage change.

## Plan

1. Investigate the existing session middleware across the codebase.
2. Add a new TokenStore abstraction in src/auth/token_store.rs.
3. Generate fixture file A in independent worker.
4. Generate fixture file B in independent worker.
5. Run the integration tests.
"#;
        let plan = import_repoprompt_plan(export);
        assert_eq!(plan.title, "Refactor auth module");
        assert!(plan.task.contains("session-token storage"));
        assert_eq!(plan.steps.len(), 5);
        assert!(plan.steps[0].starts_with("[D] "));
        assert!(plan.steps[2].starts_with("[P] "));
        assert!(plan.steps[3].starts_with("[P] "));
        assert!(plan.delegated_count >= 1);
        assert!(plan.parallel_count >= 2);
    }

    #[test]
    fn imports_fallback_when_no_plan_heading() {
        let export = "# Title\n\nIntro paragraph.\n\n- step one\n- step two\n- step three\n";
        let plan = import_repoprompt_plan(export);
        assert_eq!(plan.steps.len(), 3);
        assert_eq!(plan.steps[0], "step one");
        assert_eq!(plan.title, "Title");
        assert_eq!(plan.task, "Intro paragraph.");
    }

    #[test]
    fn imports_checkbox_list_after_plan_heading() {
        let export = "# Title\n\n## Plan\n- [ ] do A\n- [x] do B already done\n- [ ] 3. do C\n";
        let plan = import_repoprompt_plan(export);
        assert_eq!(plan.steps, vec!["do A", "do B already done", "do C"]);
    }

    #[test]
    fn item_marker_does_not_match_inside_other_brackets() {
        assert!(!item_has_marker("foo [VERIFY] bar", "V"));
        assert!(!item_has_marker("[SOP:demo]", "S"));
        assert!(item_has_marker("[D] task", "D"));
        assert!(item_has_marker("1. [D] task", "D"));
        assert!(item_has_marker("step [P].", "P"));
    }

    #[test]
    fn creates_and_completes_plan_items() {
        let root = temp_root();
        let store = PlanStore::new(root.join("plans"));
        let snapshot = store
            .create(CreatePlan {
                title: "Demo Plan".to_string(),
                task: "Implement a demo".to_string(),
                steps: vec!["Edit file".to_string(), "Run test".to_string()],
                source_export_path: None,
            })
            .unwrap();

        assert_eq!(snapshot.items.len(), 3);
        assert_eq!(snapshot.next_item.unwrap().text, "1. Edit file");

        let snapshot = store
            .complete(Some(&snapshot.state.id), None, Some("edited"))
            .unwrap();

        assert!(snapshot.items[0].checked);
        assert_eq!(snapshot.next_item.unwrap().text, "2. Run test");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn lists_plans_newest_first() {
        let root = temp_root();
        let store = PlanStore::new(root.join("plans"));
        assert!(store.list().unwrap().is_empty());

        let first = store
            .create(CreatePlan {
                title: "First Plan".to_string(),
                task: "Task".to_string(),
                steps: vec!["Do first".to_string()],
                source_export_path: None,
            })
            .unwrap();
        let second = store
            .create(CreatePlan {
                title: "Second Plan".to_string(),
                task: "Task".to_string(),
                steps: vec!["Do second".to_string()],
                source_export_path: None,
            })
            .unwrap();
        let mut first_state = first.state.clone();
        first_state.updated_at = second.state.updated_at + chrono::Duration::seconds(1);
        store.write_state(&first_state).unwrap();

        let plans = store.list().unwrap();
        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].state.id, first.state.id);
        assert_eq!(plans[1].state.id, second.state.id);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn compact_inline_does_not_split_utf8() {
        let text = compact_inline("优化 当前 的 项目 优化 当前 的 项目", 13);

        assert!(text.ends_with(" ..."));
    }

    #[test]
    fn verification_context_requires_non_verify_items_done() {
        let root = temp_root();
        let store = PlanStore::new(root.join("plans"));
        let snapshot = store
            .create(CreatePlan {
                title: "Verify Demo".to_string(),
                task: "Task".to_string(),
                steps: vec!["Do work".to_string()],
                source_export_path: None,
            })
            .unwrap();

        assert!(
            store
                .write_verify_context(Some(&snapshot.state.id))
                .is_err()
        );
        let snapshot = store
            .complete(Some(&snapshot.state.id), Some(1), None)
            .unwrap();
        let context = store
            .write_verify_context(Some(&snapshot.state.id))
            .unwrap();

        assert_eq!(context.plan_id, snapshot.state.id);
        assert!(context.plan_file.is_file());
        assert!(
            context
                .orchestration
                .artifacts
                .iter()
                .any(|artifact| artifact.kind == PlanArtifactKind::VerificationContext)
        );
        assert!(snapshot.state.verify_context_path.is_file());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn records_repoprompt_artifacts_and_handoffs_in_plan_ledger() {
        let root = temp_root();
        let store = PlanStore::new(root.join("plans"));
        let export_path = root.join("oracle-export.md");
        fs::create_dir_all(&root).unwrap();
        fs::write(&export_path, "# RepoPrompt Export\n").unwrap();
        let snapshot = store
            .create(CreatePlan {
                title: "Ledger Demo".to_string(),
                task: "Task".to_string(),
                steps: vec!["Do work".to_string()],
                source_export_path: Some(export_path.clone()),
            })
            .unwrap();

        assert_eq!(
            snapshot.state.orchestration.preferred_backend,
            PlanBackend::RepoPrompt
        );
        assert!(
            snapshot
                .state
                .orchestration
                .repoprompt_export_path
                .is_some()
        );
        assert!(
            snapshot
                .state
                .orchestration
                .artifacts
                .iter()
                .any(|artifact| artifact.kind == PlanArtifactKind::RepoPromptExport)
        );

        let context_path = root.join("context-export.md");
        fs::write(&context_path, "# Context\n").unwrap();
        let snapshot = store
            .record_artifact(
                Some(&snapshot.state.id),
                RecordPlanArtifact {
                    kind: PlanArtifactKind::ContextExport,
                    path: context_path.clone(),
                    note: Some("selection for implementation".to_string()),
                },
            )
            .unwrap();
        let snapshot = store
            .record_handoff(
                Some(&snapshot.state.id),
                RecordPlanHandoff {
                    backend: "repoprompt".to_string(),
                    role: Some("engineer".to_string()),
                    run_id: Some("agent-run-1".to_string()),
                    thread_id: None,
                    artifact_path: Some(context_path),
                    status: "completed".to_string(),
                    summary: "implemented selected plan item".to_string(),
                },
            )
            .unwrap();

        assert!(
            snapshot
                .state
                .orchestration
                .artifacts
                .iter()
                .any(|artifact| artifact.kind == PlanArtifactKind::ContextExport)
        );
        assert_eq!(snapshot.state.orchestration.handoffs.len(), 1);
        assert_eq!(
            snapshot.state.orchestration.handoffs[0].run_id.as_deref(),
            Some("agent-run-1")
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn records_pass_and_fail_verdicts() {
        let root = temp_root();
        let store = PlanStore::new(root.join("plans"));
        let snapshot = store
            .create(CreatePlan {
                title: "Verdict Demo".to_string(),
                task: "Task".to_string(),
                steps: vec!["Do work".to_string()],
                source_export_path: None,
            })
            .unwrap();
        let snapshot = store
            .complete(Some(&snapshot.state.id), Some(1), None)
            .unwrap();
        store
            .write_verify_context(Some(&snapshot.state.id))
            .unwrap();
        let snapshot = store
            .record_verification(Some(&snapshot.state.id), "VERDICT: PASS\nLooks good.")
            .unwrap();

        assert_eq!(snapshot.state.status, PlanStatus::Verified);
        assert_eq!(snapshot.state.verify_attempts, 1);
        assert_eq!(
            snapshot.state.verification_report.as_deref(),
            Some("VERDICT: PASS\nLooks good.")
        );
        assert_eq!(snapshot.state.orchestration.verification_records.len(), 1);
        assert_eq!(
            snapshot.state.orchestration.verification_records[0].verdict,
            "pass"
        );
        assert!(snapshot.items.iter().all(|item| item.checked));

        let failed = store
            .create(CreatePlan {
                title: "Fail Demo".to_string(),
                task: "Task".to_string(),
                steps: vec!["Do work".to_string()],
                source_export_path: None,
            })
            .unwrap();
        let failed = store
            .complete(Some(&failed.state.id), Some(1), None)
            .unwrap();
        let failed = store
            .record_verification(Some(&failed.state.id), "VERDICT: FAIL\nMissing test.")
            .unwrap();

        assert_eq!(failed.state.status, PlanStatus::Failed);
        assert!(
            failed
                .items
                .iter()
                .any(|item| item.kind == PlanItemKind::Fix)
        );
        let _ = fs::remove_dir_all(root);
    }

    fn temp_root() -> PathBuf {
        std::env::temp_dir().join(format!("seed-plan-test-{}", Uuid::new_v4()))
    }

    // Targeted gap tests added in RF18. The existing 14 tests cover happy
    // paths well; these fill in the typed-error branches from RF10, the
    // record_artifact_once dedup invariant, and a couple of edge cases.

    #[test]
    fn complete_with_out_of_range_index_returns_item_not_found() {
        let root = temp_root();
        let store = PlanStore::new(root.join("plans"));
        let snap = store
            .create(CreatePlan {
                title: "X".into(),
                task: "t".into(),
                steps: vec!["only step".into()],
                source_export_path: None,
            })
            .unwrap();
        let err = store
            .complete(Some(&snap.state.id), Some(999), None)
            .unwrap_err();
        assert!(
            matches!(err, PlanError::ItemNotFound { index: 999 }),
            "expected ItemNotFound{{999}}, got: {err:?}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn complete_twice_returns_already_complete() {
        let root = temp_root();
        let store = PlanStore::new(root.join("plans"));
        let snap = store
            .create(CreatePlan {
                title: "X".into(),
                task: "t".into(),
                steps: vec!["one".into()],
                source_export_path: None,
            })
            .unwrap();
        // First complete succeeds.
        store
            .complete(Some(&snap.state.id), Some(1), None)
            .unwrap();
        // Second complete on the same item is the typed error.
        let err = store
            .complete(Some(&snap.state.id), Some(1), None)
            .unwrap_err();
        assert!(
            matches!(err, PlanError::AlreadyComplete { index: 1 }),
            "expected AlreadyComplete{{1}}, got: {err:?}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn complete_on_fully_done_plan_returns_no_unchecked() {
        let root = temp_root();
        let store = PlanStore::new(root.join("plans"));
        let snap = store
            .create(CreatePlan {
                title: "X".into(),
                task: "t".into(),
                steps: vec!["one".into()],
                source_export_path: None,
            })
            .unwrap();
        store
            .complete(Some(&snap.state.id), None, None)
            .unwrap();
        // The only task item is done. `[VERIFY]` is the only unchecked left;
        // complete(None) considers it via snapshot.next_item, so this should
        // actually find the VERIFY item, not NoUncheckedItems. Verify via the
        // VERIFY explicit completion path instead.
        let snap2 = store.snapshot(Some(&snap.state.id)).unwrap();
        // Find the VERIFY item index and complete it explicitly so the only
        // unchecked is gone. Then complete(None) → NoUncheckedItems.
        let verify_idx = snap2
            .items
            .iter()
            .find(|i| matches!(i.kind, PlanItemKind::Verify))
            .map(|i| i.index)
            .expect("verify item present");
        store
            .complete(Some(&snap.state.id), Some(verify_idx), None)
            .unwrap();
        let err = store
            .complete(Some(&snap.state.id), None, None)
            .unwrap_err();
        assert!(
            matches!(err, PlanError::NoUncheckedItems),
            "expected NoUncheckedItems, got: {err:?}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn append_items_empty_returns_typed_error() {
        let root = temp_root();
        let store = PlanStore::new(root.join("plans"));
        let snap = store
            .create(CreatePlan {
                title: "X".into(),
                task: "t".into(),
                steps: vec!["one".into()],
                source_export_path: None,
            })
            .unwrap();
        let err = store
            .append_items(Some(&snap.state.id), Vec::new())
            .unwrap_err();
        assert!(
            matches!(err, PlanError::EmptyAppend),
            "expected EmptyAppend, got: {err:?}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn record_artifact_once_is_idempotent_per_kind_and_path() {
        let mut orch = PlanOrchestration::default();
        let path = PathBuf::from("/tmp/export.md");
        orch.record_artifact_once(
            PlanArtifactKind::RepoPromptExport,
            path.clone(),
            Some("first call".into()),
        );
        orch.record_artifact_once(
            PlanArtifactKind::RepoPromptExport,
            path.clone(),
            Some("second call SHOULD be deduped".into()),
        );
        assert_eq!(
            orch.artifacts.len(),
            1,
            "second call with same kind+path must dedup; got: {:?}",
            orch.artifacts
        );
        assert_eq!(orch.artifacts[0].note.as_deref(), Some("first call"));

        // Different kind → not deduped.
        orch.record_artifact_once(
            PlanArtifactKind::VerificationContext,
            path.clone(),
            None,
        );
        assert_eq!(orch.artifacts.len(), 2);

        // Different path → not deduped.
        orch.record_artifact_once(
            PlanArtifactKind::RepoPromptExport,
            PathBuf::from("/tmp/other.md"),
            None,
        );
        assert_eq!(orch.artifacts.len(), 3);
    }

    #[test]
    fn record_handoff_appends_to_ledger_with_typed_fields() {
        let root = temp_root();
        let store = PlanStore::new(root.join("plans"));
        let snap = store
            .create(CreatePlan {
                title: "X".into(),
                task: "t".into(),
                steps: vec!["one".into()],
                source_export_path: None,
            })
            .unwrap();
        let snap = store
            .record_handoff(
                Some(&snap.state.id),
                RecordPlanHandoff {
                    backend: "codex".into(),
                    role: Some("delegate".into()),
                    run_id: Some("run-1".into()),
                    thread_id: Some("thr-1".into()),
                    artifact_path: Some(PathBuf::from("/tmp/h.json")),
                    status: "completed".into(),
                    summary: "did the thing".into(),
                },
            )
            .unwrap();
        assert_eq!(snap.state.orchestration.handoffs.len(), 1);
        let h = &snap.state.orchestration.handoffs[0];
        assert_eq!(h.backend, "codex");
        assert_eq!(h.role.as_deref(), Some("delegate"));
        assert_eq!(h.run_id.as_deref(), Some("run-1"));
        assert_eq!(h.status, "completed");
        let _ = fs::remove_dir_all(root);
    }
}
