//! minimal eval suite runner.
//!
//! Reads `evals/*.toml`, runs each goal against a chosen backend,
//! grades the final answer against a regex pattern. Aggregates
//! pass/fail + per-eval timing.
//!
//! **Design**: shells out to `seed run` rather than calling `run_goal`
//! in-process. Tradeoff:
//!   - Pro: zero coupling to run_goal internals — when run_goal's
//!     return shape changes, evals don't need to change.
//!   - Pro: stdout-based grading matches what a user actually sees.
//!   - Pro: each eval is an isolated process — memory leaks /
//!     thread-local state from one eval don't contaminate the next.
//!   - Con: 100-200ms process-spawn overhead per eval. Acceptable
//!     for a suite that runs in CI, not in a hot loop.
//!
//! **Grading**: only regex-match for now. LLM-as-judge would be more
//! flexible but adds an external dependency on whoever judges. Future:
//! add `kind = "rubric"` or `kind = "llm_judge"` variants.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Deserialize;

/// One eval spec, parsed from `evals/<name>.toml`.
#[derive(Debug, Deserialize)]
struct EvalSpec {
    name: String,
    goal: String,
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default = "default_max_turns")]
    max_turns: usize,
    grade: GradeSpec,
}

fn default_mode() -> String {
    "auto".to_string()
}
fn default_max_turns() -> usize {
    8
}

/// Grading rule.
///
/// - `regex`: match the answer text against a Rust regex.
///   Sufficient for factual lookups where the right answer has a
///   stable shape.
/// - `judge`: hand the answer to a configured LLM backend
///   plus a rubric (in TOML); the backend returns PASS/FAIL.
///   Necessary for free-form answers (summaries, comparisons) where
///   regex matching would be either too lax or too strict.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum GradeSpec {
    Regex { pattern: String },
    Judge { rubric: String },
}

/// Per-eval result.
struct EvalOutcome {
    name: String,
    passed: bool,
    elapsed_secs: f64,
    answer_preview: String,
    error: Option<String>,
    turns: Option<usize>,
}

/// Run all `evals/*.toml` against the given provider. Returns exit
/// code 0 iff every eval passed.
pub(crate) fn run_eval(provider: &str, evals_dir: PathBuf, judge_provider: &str) -> Result<i32> {
    if !evals_dir.is_dir() {
        anyhow::bail!(
            "evals dir not found at {} — pass --evals-dir to override",
            evals_dir.display()
        );
    }
    let specs = load_eval_specs(&evals_dir)?;
    if specs.is_empty() {
        eprintln!("no *.toml evals found under {}", evals_dir.display());
        return Ok(0);
    }
    println!(
        "running {} eval(s) against provider={}\n",
        specs.len(),
        provider
    );

    let mut outcomes: Vec<EvalOutcome> = Vec::with_capacity(specs.len());
    for spec in &specs {
        let outcome = run_one_eval(provider, spec, judge_provider);
        // Print as-we-go so a long suite isn't silent.
        let badge = if outcome.passed { "PASS" } else { "FAIL" };
        println!(
            "  [{badge}] {}  ({:.1}s)",
            outcome.name, outcome.elapsed_secs
        );
        if !outcome.passed {
            println!("        answer: {}", outcome.answer_preview);
            if let Some(err) = &outcome.error {
                println!("        error : {err}");
            }
        }
        outcomes.push(outcome);
    }

    let passed = outcomes.iter().filter(|o| o.passed).count();
    let total = outcomes.len();
    println!("\n{passed}/{total} passed");

    if passed == total {
        Ok(0)
    } else {
        Ok(1)
    }
}

/// validate that `--learn` produces useful skills.
///
/// Runs ONE eval three times and reports the comparison:
///   1. baseline (learn off) → T1 turns, A1 answer, P1 grade
///   2. learn run (learn on) → T2 turns, A2 answer, P2 grade
///                              + creates SKILL.md
///   3. post-learn (learn off, but skill now exists) → T3, A3, P3
///
/// Interpretation:
///   - T3 < T1 AND P3 ≥ P1 → skill helped: planner had a clearer path
///   - T3 == T1 → skill not consulted (or didn't matter)
///   - T3 > T1 → skill hurt (the planner spent time loading it for no
///     benefit, or the skill's instructions sent it on a tangent)
///
/// **Does not auto-clean up** the created skill — the trace tells the
/// user which slug to `rm -rf` if they don't want it persisted.
pub(crate) fn run_eval_learn(
    eval_path: &Path,
    provider: &str,
    store: &agent_core::session::SessionStore,
    skills_dir: &Path,
    judge_provider: &str,
) -> Result<i32> {
    let text = std::fs::read_to_string(eval_path)
        .with_context(|| format!("read {}", eval_path.display()))?;
    let spec: EvalSpec = toml::from_str(&text)
        .with_context(|| format!("parse {}", eval_path.display()))?;

    println!(
        "learn-validation: eval={} provider={}\n",
        spec.name, provider
    );

    let mut codex_session = crate::commands::codex_session::CodexSession::default();

    // --- Phase 1: baseline ---
    println!("phase 1: baseline (no learn) ...");
    let baseline = run_one_eval_in_process(
        provider,
        &spec,
        store,
        skills_dir,
        &mut codex_session,
        judge_provider,
        false,
    );
    let baseline_turns = baseline.turns.unwrap_or(0);
    println!(
        "  baseline: turns={} elapsed={:.1}s grade={}",
        baseline_turns,
        baseline.elapsed_secs,
        if baseline.passed { "PASS" } else { "FAIL" }
    );

    // --- Phase 2: learn run ---
    println!("\nphase 2: with --learn (creates SKILL.md) ...");
    let learn_run = run_one_eval_in_process(
        provider,
        &spec,
        store,
        skills_dir,
        &mut codex_session,
        judge_provider,
        true,
    );
    let learn_turns = learn_run.turns.unwrap_or(0);
    println!(
        "  learn run: turns={} elapsed={:.1}s grade={}",
        learn_turns,
        learn_run.elapsed_secs,
        if learn_run.passed { "PASS" } else { "FAIL" }
    );

    // --- Phase 3: post-learn ---
    // Note: --learn is OFF, but the skill from phase 2 is now in
    // skills_dir and will appear in the catalog the planner sees.
    println!("\nphase 3: post-learn (skill in catalog) ...");
    let postlearn = run_one_eval_in_process(
        provider,
        &spec,
        store,
        skills_dir,
        &mut codex_session,
        judge_provider,
        false,
    );
    let postlearn_turns = postlearn.turns.unwrap_or(0);
    println!(
        "  post-learn: turns={} elapsed={:.1}s grade={}",
        postlearn_turns,
        postlearn.elapsed_secs,
        if postlearn.passed { "PASS" } else { "FAIL" }
    );

    // --- Interpretation ---
    println!("\n--- summary ---");
    println!(
        "  T1={}  T2={}  T3={}    (baseline / with-learn / post-learn)",
        baseline_turns, learn_turns, postlearn_turns
    );
    let helped = postlearn_turns > 0
        && baseline_turns > 0
        && (postlearn_turns < baseline_turns
            || (postlearn_turns == baseline_turns && postlearn.passed && !baseline.passed));
    let neutral = postlearn_turns == baseline_turns && postlearn.passed == baseline.passed;
    let hurt = postlearn_turns > baseline_turns;
    let verdict = if helped {
        "skill HELPED: post-learn used fewer turns or upgraded grade"
    } else if hurt {
        "skill HURT: post-learn used MORE turns than baseline"
    } else if neutral {
        "skill NEUTRAL: same turn count + same grade — skill wasn't consulted or didn't matter"
    } else {
        "mixed: see numbers above"
    };
    println!("  verdict: {verdict}");
    println!(
        "\n(SKILL.md was created during phase 2; `rm -rf {}/<slug>` to clean up.)",
        skills_dir.display()
    );

    Ok(if helped { 0 } else { 1 })
}

/// Helper: parse "Finished after N turns:" out of the latest session's
/// RunFinished.summary. Returns None on parse failure (best-effort).
fn last_session_turn_count(store: &agent_core::session::SessionStore) -> Option<usize> {
    let path = store.last_session_path().ok()?;
    let text = std::fs::read_to_string(&path).ok()?;
    let re = regex::Regex::new(r"Finished after (\d+) turns?").ok()?;
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if let Some(summary) = value
            .get("event")
            .and_then(|e| e.get("summary"))
            .and_then(|s| s.as_str())
            && let Some(caps) = re.captures(summary)
            && let Some(m) = caps.get(1)
        {
            return m.as_str().parse().ok();
        }
    }
    None
}

/// in-process variant of [`run_eval`]. Constructs
/// `RunGoalArgs` directly instead of shelling out, reuses a single
/// `CodexSession` across all evals (client cache survives the
/// loop), and reads the final answer back from the session JSONL.
///
/// Faster on suites with N ≥ 5 (no process-spawn × N overhead, no
/// Codex cold-start × N). Use the default shell-out path when you
/// need isolation guarantees (each eval gets a clean process).
pub(crate) fn run_eval_in_process(
    provider: &str,
    evals_dir: PathBuf,
    store: &agent_core::session::SessionStore,
    skills_dir: &Path,
    judge_provider: &str,
) -> Result<i32> {
    if !evals_dir.is_dir() {
        anyhow::bail!(
            "evals dir not found at {} — pass --evals-dir to override",
            evals_dir.display()
        );
    }
    let specs = load_eval_specs(&evals_dir)?;
    if specs.is_empty() {
        eprintln!("no *.toml evals found under {}", evals_dir.display());
        return Ok(0);
    }
    println!(
        "running {} eval(s) in-process against provider={}\n",
        specs.len(),
        provider
    );

    // one shared CodexSession across all evals. For
    // --provider codex this lets 's client reuse kick in;
    // for other providers it's just unused state.
    let mut codex_session = crate::commands::codex_session::CodexSession::default();

    let mut outcomes: Vec<EvalOutcome> = Vec::with_capacity(specs.len());
    for spec in &specs {
        let outcome = run_one_eval_in_process(
            provider,
            spec,
            store,
            skills_dir,
            &mut codex_session,
            judge_provider,
            false, // learn off in the standard eval path
        );
        let badge = if outcome.passed { "PASS" } else { "FAIL" };
        println!(
            "  [{badge}] {}  ({:.1}s)",
            outcome.name, outcome.elapsed_secs
        );
        if !outcome.passed {
            println!("        answer: {}", outcome.answer_preview);
            if let Some(err) = &outcome.error {
                println!("        error : {err}");
            }
        }
        outcomes.push(outcome);
    }

    let passed = outcomes.iter().filter(|o| o.passed).count();
    let total = outcomes.len();
    println!("\n{passed}/{total} passed");
    if passed == total { Ok(0) } else { Ok(1) }
}

fn run_one_eval_in_process(
    provider: &str,
    spec: &EvalSpec,
    store: &agent_core::session::SessionStore,
    skills_dir: &Path,
    codex_session: &mut crate::commands::codex_session::CodexSession,
    judge_provider: &str,
    learn: bool,
) -> EvalOutcome {
    use crate::commands::run::{ModeArg, PlannerProvider, ProviderSpec, RunGoalArgs, RunPolicy, run_goal};
    let mode_arg = match spec.mode.as_str() {
        "read" => ModeArg::Read,
        "write" => ModeArg::Write,
        _ => ModeArg::Auto,
    };

    let started = Instant::now();
    let result = run_goal(RunGoalArgs {
        store,
        goal: spec.goal.clone(),
        cwd: None, // use process cwd
        use_llm: true,
        use_codex: false,
        learn,
        skills_dir: skills_dir.to_path_buf(),
        policy: RunPolicy {
            max_turns: spec.max_turns,
            turn_timeout_secs: 600,
            ..Default::default()
        },
        provider: ProviderSpec {
            kind: PlannerProvider::from_id(provider),
            model: None,
            approval: crate::ApprovalArg::Deny,
            effort: None,
            mcp: None,
            mcp_allow: vec![],
            plugins: false,
        },
        mode: mode_arg,
        use_daemon: false,
        codex_session: Some(codex_session),
    });
    let elapsed_secs = started.elapsed().as_secs_f64();

    if let Err(err) = result {
        return EvalOutcome {
            name: spec.name.clone(),
            passed: false,
            elapsed_secs,
            answer_preview: String::new(),
            error: Some(format!("run_goal failed: {err}")),
            turns: None,
        };
    }

    // After run_goal, the latest session JSONL holds the answer in a
    // RunFinished event. Extract it for grading.
    let answer = match read_last_session_answer(store) {
        Ok(a) => a,
        Err(err) => {
            return EvalOutcome {
                name: spec.name.clone(),
                passed: false,
                elapsed_secs,
                answer_preview: String::new(),
                error: Some(format!("read session: {err}")),
            turns: None,
            };
        }
    };
    let answer_preview: String = answer.trim().chars().take(200).collect();

    // Capture before grade_answer — the judge shells out to seed run
    // and its session would shadow this eval's in last_session_path().
    let turns = last_session_turn_count(store);

    let (passed, error) = grade_answer(&spec.grade, &spec.goal, &answer, judge_provider);
    EvalOutcome {
        name: spec.name.clone(),
        passed,
        elapsed_secs,
        answer_preview,
        error,
        turns,
    }
}

/// dispatch grading. For `Regex`, compile + match. For
/// `Judge`, build a structured prompt (goal + answer + rubric) and
/// shell-out to the judge backend; parse PASS/FAIL from the answer.
fn grade_answer(
    grade: &GradeSpec,
    goal: &str,
    answer: &str,
    judge_provider: &str,
) -> (bool, Option<String>) {
    match grade {
        GradeSpec::Regex { pattern } => match regex::Regex::new(pattern) {
            Ok(re) => (re.is_match(answer), None),
            Err(err) => (false, Some(format!("invalid regex `{pattern}`: {err}"))),
        },
        GradeSpec::Judge { rubric } => judge_grade(goal, answer, rubric, judge_provider),
    }
}

/// ask `judge_provider` to grade `answer` against `rubric`.
/// Returns `(passed, error)` — `error` set only on protocol failure
/// (judge unreachable, no JSON returned, etc.); a clear FAIL from the
/// judge returns `(false, None)`.
///
/// **Prompt shape**: a single structured message asking for one of
/// `PASS` / `FAIL` on the first line, then reasoning. We parse the
/// first line case-insensitively. The judge's full answer is
/// returned in the error slot when FAIL so the user can see why.
fn judge_grade(
    goal: &str,
    answer: &str,
    rubric: &str,
    judge_provider: &str,
) -> (bool, Option<String>) {
    use std::process::Command;
    // Cap the answer at 4000 chars so the judge prompt stays bounded.
    let answer_for_judge: String = answer.chars().take(4000).collect();
    let prompt = format!(
        "You are grading an LLM agent's answer to a task. Reply with `PASS` or `FAIL` on the FIRST LINE, then one short reason line.\n\
        \n\
        TASK:\n{goal}\n\
        \n\
        AGENT ANSWER:\n{answer_for_judge}\n\
        \n\
        RUBRIC:\n{rubric}\n\
        \n\
        Reply now. First line: PASS or FAIL. Second line: one-sentence reason."
    );
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("seed"));
    let output = Command::new(&exe)
        .args([
            "run",
            "--llm",
            "--provider",
            judge_provider,
            "--mode",
            "read",
            "--max-turns",
            "3",
            &prompt,
        ])
        .output();
    let output = match output {
        Ok(o) => o,
        Err(err) => return (false, Some(format!("judge spawn failed: {err}"))),
    };
    if !output.status.success() {
        return (
            false,
            Some(format!(
                "judge exited {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).chars().take(200).collect::<String>()
            )),
        );
    }
    let judge_answer = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let first_line = judge_answer.lines().next().unwrap_or("").trim();
    let verdict = first_line.to_ascii_uppercase();
    if verdict.starts_with("PASS") {
        (true, None)
    } else if verdict.starts_with("FAIL") {
        // Keep the judge's reason in error so the user sees why.
        (false, Some(format!("judge said: {judge_answer}")))
    } else {
        (
            false,
            Some(format!(
                "judge returned non-PASS/FAIL verdict: {judge_answer}"
            )),
        )
    }
}

/// Walk the most recent session JSONL and pull the final answer out
/// of its `RunFinished` (or fall back to `Reflection`) event.
fn read_last_session_answer(store: &agent_core::session::SessionStore) -> Result<String> {
    let path = store.last_session_path()?;
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("read {}", path.display()))?;
    // RunFinished comes last; scan from the end.
    let mut latest_run_finished: Option<String> = None;
    let mut latest_reflection: Option<String> = None;
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let event = value.get("event").and_then(|e| e.as_object());
        let Some(event) = event else { continue };
        match event.get("type").and_then(|t| t.as_str()) {
            Some("run_finished") => {
                if let Some(s) = event.get("summary").and_then(|s| s.as_str()) {
                    latest_run_finished = Some(s.to_string());
                }
            }
            Some("reflection") => {
                if let Some(s) = event.get("summary").and_then(|s| s.as_str()) {
                    latest_reflection = Some(s.to_string());
                }
            }
            _ => {}
        }
    }
    Ok(latest_reflection
        .or(latest_run_finished)
        .unwrap_or_default())
}

fn load_eval_specs(dir: &Path) -> Result<Vec<EvalSpec>> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .collect();
    entries.sort(); // deterministic order
    let mut specs = Vec::new();
    for path in entries {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        let spec: EvalSpec = toml::from_str(&text)
            .with_context(|| format!("parse {}", path.display()))?;
        specs.push(spec);
    }
    Ok(specs)
}

fn run_one_eval(provider: &str, spec: &EvalSpec, judge_provider: &str) -> EvalOutcome {
    let started = Instant::now();
    let output = Command::new(std::env::current_exe().unwrap_or_else(|_| PathBuf::from("seed")))
        .args([
            "run",
            "--llm",
            "--provider",
            provider,
            "--mode",
            &spec.mode,
            "--max-turns",
            &spec.max_turns.to_string(),
            &spec.goal,
        ])
        .output();
    let elapsed_secs = started.elapsed().as_secs_f64();

    let output = match output {
        Ok(o) => o,
        Err(err) => {
            return EvalOutcome {
                name: spec.name.clone(),
                passed: false,
                elapsed_secs,
                answer_preview: String::new(),
                error: Some(format!("spawn failed: {err}")),
            turns: None,
            };
        }
    };

    if !output.status.success() {
        return EvalOutcome {
            name: spec.name.clone(),
            passed: false,
            elapsed_secs,
            answer_preview: String::from_utf8_lossy(&output.stderr).chars().take(200).collect(),
            error: Some(format!("seed run exited {}", output.status)),
            turns: None,
        };
    }

    // `seed run` writes the final answer to stdout; the trailing
    // footer block (`  turns ...`, `  session ...`) appears in
    // dim-text on stderr and won't pollute stdout. Take the entire
    // stdout as the answer.
    let answer = String::from_utf8_lossy(&output.stdout);
    let answer_preview: String = answer.trim().chars().take(200).collect();

    let (passed, error) = grade_answer(&spec.grade, &spec.goal, &answer, judge_provider);
    EvalOutcome {
        name: spec.name.clone(),
        passed,
        elapsed_secs,
        answer_preview,
        error,
        turns: None,
    }
}
