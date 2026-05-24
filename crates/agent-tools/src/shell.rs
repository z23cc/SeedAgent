//! `run_shell` tool + the read-only-mode intent classifier.
//!
//! extracted from `lib.rs`. The classifier
//! (`shell_command_intent`) and `ShellIntent` enum stay `pub` because
//! they're crate-public API surface (defensive, not consumed by other
//! crates today).
//!
//! Shared helpers used: `parse_tool_args`, `resolve_path`,
//! `truncate_middle_with_stats`, `run_mode_guard::current()` — all
//! `pub`/`pub(crate)` in `lib.rs`.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use agent_core::{RunMode, Tool, ToolCall, ToolContext, ToolError, ToolResult};
use serde::Deserialize;
use serde_json::json;
use wait_timeout::ChildExt;

use crate::{parse_tool_args, resolve_path, run_mode_guard, truncate_middle_with_stats};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ShellArgs {
    // planners often emit `cmd` instead of `command`. Same for
    // `timeout_ms` vs `timeout_secs`, `dir` vs `cwd`. Accept the
    // synonyms so we don't waste a turn on a self-correctable mistake.
    #[serde(alias = "cmd", alias = "shell", alias = "script")]
    command: String,
    #[serde(default, alias = "dir", alias = "working_dir")]
    cwd: Option<String>,
    #[serde(default, alias = "timeout", alias = "timeout_seconds")]
    timeout_secs: Option<u64>,
    // planners that came from Claude tool-use sometimes send
    // `timeout_ms`. Accept it as a denominator-converted hint — we don't
    // expose it in the canonical struct (it'd duplicate timeout_secs),
    // but #[serde(deny_unknown_fields)] would otherwise reject it.
    // Default behavior: silently ignored if the canonical field is also
    // present; otherwise convert.
    #[serde(default, alias = "timeoutMs")]
    timeout_ms: Option<u64>,
}

impl ShellArgs {
    /// Resolve the timeout in seconds with synonym fallback.
    fn resolved_timeout_secs(&self) -> Option<u64> {
        self.timeout_secs.or_else(|| {
            self.timeout_ms.map(|ms| (ms + 999) / 1000)
        })
    }
}

/// Classification of a shell command's intent for the read-only guard.
///
/// Conservative: anything we don't explicitly recognize as Read or Write
/// is `Ambiguous` and gets through. The goal is to catch the obvious
/// `echo > foo` / `rm -rf` shapes that would silently violate the
/// read-only contract, not to be a full bash linter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellIntent {
    /// Clearly write-shaped (output redirect, mutating subcommand). Blocked
    /// in `RunMode::ReadOnly`.
    Write,
    /// Clearly read-shaped (ls, cat, git status, …). Always allowed.
    Read,
    /// Neither pattern matched. Allowed everywhere — pure pass-through.
    Ambiguous,
}

/// classify a shell command string to decide whether ShellTool
/// should run it under `RunMode::ReadOnly`.
///
/// Looked-for write signals (any one is enough → `Write`):
///   - Output redirect operators: `>`, `>>`, `|tee`, `&>`
///   - Destructive cmds at start of any chain segment: `rm`, `mv`,
///     `mkdir`, `rmdir`, `cp`, `dd`, `tee`, `install`, `ln`
///   - In-place editors: `sed -i`, `awk -i`, `perl -i`
///   - VCS mutating subcommands: `git commit`, `git push`, `git checkout`,
///     `git reset --hard`, `git rebase`, `git merge`, `git tag`, `git add`,
///     `git rm`, `git mv`, `git stash`
///   - Package managers: `cargo install/build/run/test/publish`,
///     `npm install`, `pip install`, `brew install`
///
/// Read signals (any → `Read`, but write signals take precedence):
///   - `ls`, `cat`, `head`, `tail`, `wc`, `file`, `stat`, `tree`
///   - `grep`, `rg`, `ag`, `find`, `fd`
///   - `git status`, `git log`, `git diff`, `git show`, `git blame`
///   - `pwd`, `whoami`, `which`, `type`, `echo` (without redirect)
pub fn shell_command_intent(cmd: &str) -> ShellIntent {
    let trimmed = cmd.trim();
    if trimmed.is_empty() {
        return ShellIntent::Ambiguous;
    }

    // Redirect operators are unambiguous writes (echo "x" > foo, cmd | tee).
    // Match against a stripped version that removes contiguous spaces around
    // redirect chars so `echo x>foo` and `echo x > foo` both hit.
    let lower = trimmed.to_ascii_lowercase();
    if lower.contains(" > ")
        || lower.contains(" >> ")
        || lower.contains(" >>")
        || lower.contains(">> ")
        || lower.contains(" &> ")
        || lower.contains(" &>>")
        || lower.contains("| tee")
        || lower.contains("|tee")
    {
        return ShellIntent::Write;
    }

    // Inspect each chained segment (split on `&&`, `||`, `;`, `|`). For
    // each segment we look at the first token.
    let segments = lower
        .split(|c: char| c == ';' || c == '|' || c == '&')
        .map(str::trim)
        .filter(|s| !s.is_empty());

    let write_first_tokens = [
        "rm", "mv", "mkdir", "rmdir", "cp", "dd", "tee", "install", "ln", "touch",
        "chmod", "chown",
    ];
    let read_first_tokens = [
        "ls", "cat", "head", "tail", "wc", "file", "stat", "tree", "grep", "rg",
        "ag", "find", "fd", "pwd", "whoami", "which", "type", "echo", "printf",
        "env", "date", "uname", "hostname", "id",
    ];

    let mut any_write = false;
    let mut any_read = false;
    for seg in segments {
        let mut toks = seg.split_whitespace();
        let Some(first) = toks.next() else { continue };

        // In-place editors via flag inspection.
        if matches!(first, "sed" | "awk" | "perl") {
            for tok in toks.clone() {
                if tok == "-i" || tok.starts_with("-i") {
                    return ShellIntent::Write;
                }
            }
        }

        if write_first_tokens.contains(&first) {
            any_write = true;
            continue;
        }

        if first == "git" {
            if let Some(sub) = toks.next() {
                let mutating = [
                    "commit", "push", "checkout", "reset", "rebase", "merge",
                    "tag", "add", "rm", "mv", "stash", "cherry-pick", "revert",
                    "clean", "branch", "pull", "fetch", "init", "clone",
                ];
                let reading = ["status", "log", "diff", "show", "blame", "ls-files", "config"];
                if mutating.contains(&sub) {
                    any_write = true;
                } else if reading.contains(&sub) {
                    any_read = true;
                }
            }
            continue;
        }

        if matches!(first, "cargo" | "npm" | "pip" | "pip3" | "yarn" | "pnpm" | "brew")
        {
            if let Some(sub) = toks.next() {
                let mutating = [
                    "install", "uninstall", "remove", "publish", "build", "run",
                    "test", "bench", "update", "upgrade",
                ];
                let reading = ["check", "tree", "list", "info", "search", "outdated"];
                if mutating.contains(&sub) {
                    any_write = true;
                } else if reading.contains(&sub) {
                    any_read = true;
                }
            }
            continue;
        }

        if read_first_tokens.contains(&first) {
            any_read = true;
        }
    }

    if any_write {
        ShellIntent::Write
    } else if any_read {
        ShellIntent::Read
    } else {
        ShellIntent::Ambiguous
    }
}

pub struct ShellTool;

impl Tool for ShellTool {
    fn name(&self) -> &'static str {
        "run_shell"
    }

    crate::tool_description!("run_shell");

    crate::impl_args_schema!(ShellArgs);

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: ShellArgs = parse_tool_args(call)?;

        // in read-only mode, refuse write-shaped commands. The
        // `is_read_only_planner_tool` allowlist already lets `run_shell`
        // through unconditionally (so `ls`/`git status` work in read-only
        // runs); this is the inner check that closes the `echo > foo`
        // loophole. Ambiguous commands fall through to keep the false
        // positive rate down.
        if matches!(run_mode_guard::current(), RunMode::ReadOnly)
            && matches!(shell_command_intent(&args.command), ShellIntent::Write)
        {
            return Err(ToolError::Failed(format!(
                "run_shell refused in read-only mode: command appears to write \
                 ({:?}). Re-run with --mode write (or `/mode write` in the REPL) \
                 if you really mean to mutate the project.",
                args.command,
            )));
        }
        let cwd = args
            .cwd
            .as_deref()
            .map(|path| resolve_path(&ctx.cwd, path))
            .unwrap_or_else(|| ctx.cwd.clone());
        let output = run_shell(
            &args.command,
            &cwd,
            args.resolved_timeout_secs().unwrap_or(60),
        )
        .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(call, output))
    }
}

fn run_shell(command: &str, cwd: &Path, timeout_secs: u64) -> anyhow::Result<serde_json::Value> {
    let mut child = if cfg!(windows) {
        Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", command])
            .current_dir(cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?
    } else {
        Command::new("bash")
            .args(["-lc", command])
            .current_dir(cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?
    };

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let out_handle = thread::spawn(move || read_pipe(stdout));
    let err_handle = thread::spawn(move || read_pipe(stderr));
    let (timed_out, status) = match child.wait_timeout(Duration::from_secs(timeout_secs))? {
        Some(status) => (false, Some(status)),
        None => {
            child.kill()?;
            (true, child.wait().ok())
        }
    };
    let stdout = out_handle.join().unwrap_or_default();
    let stderr = err_handle.join().unwrap_or_default();
    let exit_code = status.and_then(|status| status.code());
    // emit truncation metadata so the planner can tell when it's
    // only seeing a slice of the real output.
    let (stdout_text, stdout_stats) = truncate_middle_with_stats(&stdout, 12_000);
    let (stderr_text, stderr_stats) = truncate_middle_with_stats(&stderr, 4_000);
    Ok(json!({
        "status": if !timed_out && exit_code == Some(0) { "success" } else { "error" },
        "timed_out": timed_out,
        "exit_code": exit_code,
        "stdout": stdout_text,
        "stdout_truncated": stdout_stats.was_truncated,
        "stdout_original_bytes": stdout_stats.original_bytes,
        "stderr": stderr_text,
        "stderr_truncated": stderr_stats.was_truncated,
        "stderr_original_bytes": stderr_stats.original_bytes,
    }))
}

fn read_pipe(pipe: Option<impl Read>) -> String {
    let Some(mut pipe) = pipe else {
        return String::new();
    };
    let mut buf = String::new();
    let _ = pipe.read_to_string(&mut buf);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- serde aliases on arg structs ----------------------------

    #[test]
    fn shell_args_accept_cmd_alias() {
        // The case from the user's trace: planner sent `cmd` not `command`.
        // After this should now parse cleanly.
        let v = serde_json::json!({"cmd": "echo hi"});
        let args: ShellArgs = serde_json::from_value(v).expect("cmd alias parses");
        assert_eq!(args.command, "echo hi");
    }

    #[test]
    fn shell_args_accept_shell_and_script_aliases() {
        for synonym in ["shell", "script"] {
            let v = serde_json::json!({ synonym: "ls" });
            let args: ShellArgs = serde_json::from_value(v).unwrap_or_else(|e| {
                panic!("alias `{synonym}` should parse: {e}")
            });
            assert_eq!(args.command, "ls");
        }
    }

    #[test]
    fn shell_args_timeout_ms_converts_to_secs() {
        // Planner sends timeout_ms=10000; we have no canonical timeout_ms
        // field but the helper rounds up to 10s.
        let v = serde_json::json!({"command": "ls", "timeout_ms": 10000});
        let args: ShellArgs = serde_json::from_value(v).unwrap();
        assert_eq!(args.resolved_timeout_secs(), Some(10));
    }

    #[test]
    fn shell_args_timeout_seconds_alias_works() {
        let v = serde_json::json!({"command": "ls", "timeout_seconds": 30});
        let args: ShellArgs = serde_json::from_value(v).unwrap();
        assert_eq!(args.resolved_timeout_secs(), Some(30));
    }
}
