use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

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
    pub verification_report: Option<String>,
    pub verify_attempts: usize,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanItem {
    pub index: usize,
    pub line_number: usize,
    pub checked: bool,
    pub kind: PlanItemKind,
    pub text: String,
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
pub struct VerificationContext {
    pub plan_id: String,
    pub task: String,
    pub plan_file: PathBuf,
    pub state_file: PathBuf,
    pub repoprompt_export_path: Option<PathBuf>,
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

    pub fn create(&self, input: CreatePlan) -> Result<PlanSnapshot> {
        fs::create_dir_all(&self.root)
            .with_context(|| format!("create {}", self.root.display()))?;
        let title = clean_title(&input.title).unwrap_or_else(|| "Plan".to_string());
        let id = unique_plan_id(&self.root, &title);
        let dir = self.root.join(&id);
        fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;

        let plan_path = dir.join("plan.md");
        let state_path = dir.join("state.json");
        let verify_context_path = dir.join("verify_context.json");
        let repoprompt_export_path = copy_export(&dir, input.source_export_path.as_deref())?;
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
            source_export_path: input.source_export_path,
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

    pub fn snapshot(&self, id: Option<&str>) -> Result<PlanSnapshot> {
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

    pub fn complete(
        &self,
        id: Option<&str>,
        item_index: Option<usize>,
        note: Option<&str>,
    ) -> Result<PlanSnapshot> {
        let snapshot = self.snapshot(id)?;
        let item = match item_index {
            Some(index) => snapshot
                .items
                .iter()
                .find(|item| item.index == index)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("plan item not found: {index}"))?,
            None => snapshot
                .next_item
                .clone()
                .ok_or_else(|| anyhow::anyhow!("plan has no unchecked items"))?,
        };
        if item.checked {
            bail!("plan item {} is already complete", item.index);
        }
        let body = fs::read_to_string(&snapshot.state.plan_path)
            .with_context(|| format!("read {}", snapshot.state.plan_path.display()))?;
        let mut lines = body.lines().map(ToString::to_string).collect::<Vec<_>>();
        let Some(line) = lines.get_mut(item.line_number - 1) else {
            bail!("plan item line missing: {}", item.line_number);
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

    pub fn write_verify_context(&self, id: Option<&str>) -> Result<VerificationContext> {
        let snapshot = self.snapshot(id)?;
        if snapshot.task_unchecked_count > 0 {
            bail!(
                "plan still has {} unfinished non-verify item(s)",
                snapshot.task_unchecked_count
            );
        }
        let context = VerificationContext {
            plan_id: snapshot.state.id.clone(),
            task: snapshot.state.task.clone(),
            plan_file: snapshot.state.plan_path.clone(),
            state_file: snapshot.state.state_path.clone(),
            repoprompt_export_path: snapshot.state.repoprompt_export_path.clone(),
            required_checks: vec![
                "Read plan.md and verify every checked task against the repository.".to_string(),
                "Run or inspect the narrow checks named by the plan when possible.".to_string(),
                "Return a line beginning with VERDICT: PASS or VERDICT: FAIL.".to_string(),
            ],
            instructions: "Independent verification gate. Do not trust the executor summary; verify from files, git diff, tests, and artifacts. Report concrete failures with file paths and commands.".to_string(),
        };
        fs::write(
            &snapshot.state.verify_context_path,
            serde_json::to_string_pretty(&context)?,
        )
        .with_context(|| format!("write {}", snapshot.state.verify_context_path.display()))?;
        let mut state = snapshot.state;
        state.status = PlanStatus::PendingVerification;
        state.updated_at = Utc::now();
        self.write_state(&state)?;
        Ok(context)
    }

    pub fn record_verification(&self, id: Option<&str>, report: &str) -> Result<PlanSnapshot> {
        let mut snapshot = self.snapshot(id)?;
        let report = report.trim().to_string();
        snapshot.state.verify_attempts += 1;
        snapshot.state.verification_report = Some(report.clone());
        let verify_attempts = snapshot.state.verify_attempts;
        let verdict = verdict_from_report(&report);
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
        items.push(PlanItem {
            index: items.len() + 1,
            line_number: line_idx + 1,
            checked: rest.0,
            kind,
            text,
        });
    }
    items
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
        text.truncate(limit);
        text.push_str(" ...");
    }
    text
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
        assert!(snapshot.state.verify_context_path.is_file());
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
}
