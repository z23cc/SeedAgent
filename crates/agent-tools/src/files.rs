//! Filesystem tools: `read_file`, `read_files`, `patch_file`, `write_file`.
//!
//! extracted from `lib.rs` to peel ~340 lines off the central
//! file. Shared helpers (`durable_write_guard`, `is_durable_path`,
//! `resolve_path`, `truncate_utf8`, `DurableWriteMode`) stay in `lib.rs`
//! as `pub(crate)` because other regions (shell, memory_protocol,
//! ask_user) also reach for them.

use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use agent_core::{Tool, ToolCall, ToolContext, ToolError, ToolResult};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    DurableWriteMode, durable_write_guard, is_durable_path, parse_tool_args, resolve_path,
    truncate_utf8,
};

/// Reject writes to a file that already exists on disk but wasn't read
/// this run. Returns `None` when the write is allowed (file didn't
/// exist, OR planner read it earlier, OR escape hatch active).
fn check_read_before_write(path: &Path) -> Option<String> {
    if crate::read_paths_guard::is_disabled() {
        return None;
    }
    if !path.exists() {
        return None;
    }
    if crate::read_paths_guard::was_read(path) {
        return None;
    }
    Some(format!(
        "write to {} blocked: file exists on disk but was not read this run. \
         Call read_file (or read_files) on this path first so you're patching \
         what's actually there, not what you imagine is there. \
         Override with env SEED_DISABLE_READ_BEFORE_WRITE=1 if you genuinely \
         need to overwrite blind (codegen, migrations).",
        path.display()
    ))
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ReadFileArgs {
    // accept the synonym `file` that planners often emit.
    #[serde(alias = "file", alias = "filename", alias = "filepath")]
    path: String,
    #[serde(default, alias = "start_line", alias = "from")]
    start: Option<usize>,
    #[serde(default, alias = "limit", alias = "lines", alias = "max_lines")]
    count: Option<usize>,
    #[serde(default, alias = "pattern", alias = "needle")]
    keyword: Option<String>,
    #[serde(default, alias = "line_numbers")]
    show_line_numbers: Option<bool>,
}

pub struct ReadFileTool;

impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    crate::tool_description!("read_file");

    crate::impl_args_schema!(ReadFileArgs);

    crate::impl_pure_read!();

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: ReadFileArgs = parse_tool_args(call)?;

        let path = resolve_path(&ctx.cwd, &args.path);
        let start = args.start.unwrap_or(1).max(1);
        let default_count = ctx.scaled_default(200, 60);
        let count = args.count.unwrap_or(default_count).clamp(1, 1000);
        let show_line_numbers = args.show_line_numbers.unwrap_or(true);
        let content = read_file_window(
            &path,
            start,
            count,
            args.keyword.as_deref(),
            show_line_numbers,
        )
        .map_err(|err| ToolError::Failed(err.to_string()))?;
        crate::read_paths_guard::record(&path);
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "path": path,
                "content": content,
            }),
        ))
    }
}

/// each entry in `ReadFilesArgs.paths` can be either a bare string
/// (most planners' first-attempt shape) or an object with per-file
/// overrides for `start`/`count`/`keyword`. The latter shape lets a
/// planner say "read main.rs lines 1-50 and lib.rs lines 200-260 in one
/// turn" without splitting into two `read_file` calls. Untagged enum so
/// serde tries each variant in order.
///
/// `path` is the only required field; the other three default to the
/// top-level `ReadFilesArgs` fallback values when omitted. Field aliases
/// match `ReadFileArgs` for consistency.
#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
enum ReadFilesEntry {
    Plain(String),
    Detailed {
        #[serde(alias = "file", alias = "filename", alias = "filepath")]
        path: String,
        #[serde(default, alias = "start_line", alias = "from")]
        start: Option<usize>,
        #[serde(default, alias = "limit", alias = "lines", alias = "max_lines")]
        count: Option<usize>,
        #[serde(default, alias = "pattern", alias = "needle")]
        keyword: Option<String>,
    },
}

impl ReadFilesEntry {
    fn path(&self) -> &str {
        match self {
            ReadFilesEntry::Plain(p) => p,
            ReadFilesEntry::Detailed { path, .. } => path,
        }
    }

    /// Resolve per-file params with fallback to caller-supplied defaults.
    fn resolved(
        &self,
        fallback_start: usize,
        fallback_count: usize,
        fallback_keyword: Option<&str>,
    ) -> (usize, usize, Option<String>) {
        match self {
            ReadFilesEntry::Plain(_) => (
                fallback_start,
                fallback_count,
                fallback_keyword.map(ToString::to_string),
            ),
            ReadFilesEntry::Detailed {
                start,
                count,
                keyword,
                ..
            } => (
                start.unwrap_or(fallback_start).max(1),
                count.unwrap_or(fallback_count).clamp(1, 1000),
                keyword
                    .clone()
                    .or_else(|| fallback_keyword.map(ToString::to_string)),
            ),
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ReadFilesArgs {
    // accept `files`/`filenames` as synonyms for `paths`.
    // each entry can now be a plain string OR a per-file spec object.
    #[serde(alias = "files", alias = "filenames", alias = "filepaths")]
    paths: Vec<ReadFilesEntry>,
    #[serde(default, alias = "start_line", alias = "from")]
    start: Option<usize>,
    #[serde(default, alias = "limit", alias = "lines", alias = "max_lines")]
    count: Option<usize>,
    #[serde(default, alias = "pattern", alias = "needle")]
    keyword: Option<String>,
    #[serde(default, alias = "line_numbers")]
    show_line_numbers: Option<bool>,
}

pub struct ReadFilesTool;

impl Tool for ReadFilesTool {
    fn name(&self) -> &'static str {
        "read_files"
    }

    crate::tool_description!("read_files");

    crate::impl_args_schema!(ReadFilesArgs);

    crate::impl_pure_read!();

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: ReadFilesArgs = parse_tool_args(call)?;

        if args.paths.is_empty() {
            return Ok(ToolResult::error(call, "paths must not be empty"));
        }
        let entries = if args.paths.len() > 8 {
            return Ok(ToolResult::error(
                call,
                format!(
                    "read_files capped at 8 paths per call; got {}. Split the request.",
                    args.paths.len()
                ),
            ));
        } else {
            args.paths
        };

        // directory entries auto-expand via gitignore-aware
        // walker. Old behavior errored on directories which forced the
        // planner to waste a turn enumerating files manually; new
        // behavior walks the dir (respecting .gitignore + hidden-file
        // rules) and inlines up to 12 files per directory entry. Total
        // expanded count surfaces in `expanded_from_dirs` for trace
        // visibility.
        let mut expanded_entries: Vec<ReadFilesEntry> = Vec::with_capacity(entries.len());
        let mut dir_expansions: Vec<Value> = Vec::new();
        for entry in &entries {
            let path = resolve_path(&ctx.cwd, entry.path());
            if path.is_dir() {
                let walk_opts = crate::WalkOptions {
                    max_files: 12,
                    absolute: true,
                    ..Default::default()
                };
                let walked = crate::walk_workspace(&path, &walk_opts);
                let count = walked.paths.len();
                for walked_path in walked.paths {
                    expanded_entries.push(ReadFilesEntry::Plain(
                        walked_path.display().to_string(),
                    ));
                }
                dir_expansions.push(json!({
                    "dir": path,
                    "files_added": count,
                    "truncated": walked.truncated,
                }));
            } else {
                expanded_entries.push(entry.clone());
            }
        }
        let entries = expanded_entries;
        if entries.is_empty() {
            return Ok(ToolResult::error(
                call,
                "no readable files after expanding directory entries (all were ignored or empty)",
            ));
        }

        let fallback_start = args.start.unwrap_or(1).max(1);
        // Per-file scaling: as we read more files in one turn, shrink each
        // file's window so total output stays bounded.
        let base_default = ctx.scaled_default(200, 60);
        let per_file_default = (base_default / entries.len().max(1)).max(40);
        let fallback_count = args.count.unwrap_or(per_file_default).clamp(1, 1000);
        let show_line_numbers = args.show_line_numbers.unwrap_or(true);

        let mut files: Vec<Value> = Vec::with_capacity(entries.len());
        let mut succeeded = 0usize;
        for entry in &entries {
            // per-file params win over uniform fallback.
            let (start, count, keyword) =
                entry.resolved(fallback_start, fallback_count, args.keyword.as_deref());
            let path = resolve_path(&ctx.cwd, entry.path());
            match read_file_window(&path, start, count, keyword.as_deref(), show_line_numbers) {
                Ok(content) => {
                    succeeded += 1;
                    crate::read_paths_guard::record(&path);
                    files.push(json!({
                        "path": path,
                        "status": "ok",
                        "content": content,
                        // Surface effective params per file so the planner
                        // sees what we actually applied — important when it
                        // sent per-file overrides.
                        "start": start,
                        "count": count,
                    }));
                }
                Err(err) => {
                    files.push(json!({
                        "path": path,
                        "status": "error",
                        "error": err.to_string(),
                    }));
                }
            }
        }
        let total = files.len();
        let mut content = json!({
            "status": if succeeded == total { "success" } else { "partial" },
            "succeeded": succeeded,
            "failed": total - succeeded,
            "default_count_per_file": fallback_count,
            "files": files,
        });
        // surface dir-expansion bookkeeping so the planner can
        // tell why it got back more files than it asked for paths.
        if !dir_expansions.is_empty() {
            content["expanded_from_dirs"] = json!(dir_expansions);
        }
        Ok(ToolResult::ok(call, content))
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct PatchFileArgs {
    // `file`/`filename` synonyms for `path`. `old`/`new` shorthand
    // synonyms for the content fields (Anthropic tool-use style).
    #[serde(alias = "file", alias = "filename", alias = "filepath")]
    path: String,
    #[serde(alias = "old", alias = "before", alias = "search")]
    old_content: String,
    #[serde(alias = "new", alias = "after", alias = "replace")]
    new_content: String,
}

pub struct PatchFileTool;

impl Tool for PatchFileTool {
    fn name(&self) -> &'static str {
        "patch_file"
    }

    crate::tool_description!("patch_file");

    crate::impl_args_schema!(PatchFileArgs);

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: PatchFileArgs = parse_tool_args(call)?;

        let path = resolve_path(&ctx.cwd, &args.path);
        if args.old_content.is_empty() {
            return Ok(ToolResult::error(call, "old_content must not be empty"));
        }
        let text = fs::read_to_string(&path).map_err(|err| ToolError::Failed(err.to_string()))?;
        let matches = text.matches(&args.old_content).count();
        if matches == 0 {
            return Ok(ToolResult::error(
                call,
                "old_content was not found; read the file again and patch a smaller exact block",
            ));
        }
        if matches > 1 {
            return Ok(ToolResult::error(
                call,
                format!("old_content matched {matches} places; provide a more specific block"),
            ));
        }
        let updated_text = text.replace(&args.old_content, &args.new_content);
        if let Some(message) = durable_write_guard(
            ctx,
            &path,
            &args.new_content,
            DurableWriteMode::Patch,
            false,
        ) {
            return Ok(ToolResult::error(call, message));
        }
        if let Some(msg) = check_read_before_write(&path) {
            return Ok(ToolResult::error(call, msg));
        }
        fs::write(&path, updated_text).map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({ "status": "success", "path": path, "matches": matches, "durable_guarded": is_durable_path(ctx, &path) }),
        ))
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct WriteFileArgs {
    // `file`/`filename` for `path`, `text`/`body`/`data` for content.
    #[serde(alias = "file", alias = "filename", alias = "filepath")]
    path: String,
    #[serde(alias = "text", alias = "body", alias = "data")]
    content: String,
    #[serde(default)]
    mode: Option<WriteMode>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WriteMode {
    Overwrite,
    Append,
    Prepend,
}

impl From<WriteMode> for DurableWriteMode {
    fn from(value: WriteMode) -> Self {
        match value {
            WriteMode::Overwrite => Self::Overwrite,
            WriteMode::Append => Self::Append,
            WriteMode::Prepend => Self::Prepend,
        }
    }
}

pub struct WriteFileTool;

impl Tool for WriteFileTool {
    fn name(&self) -> &'static str {
        "write_file"
    }

    crate::tool_description!("write_file");

    crate::impl_args_schema!(WriteFileArgs);

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: WriteFileArgs = parse_tool_args(call)?;

        let path = resolve_path(&ctx.cwd, &args.path);
        let mode = args.mode.unwrap_or(WriteMode::Overwrite);
        let existing_nonempty = fs::read_to_string(&path)
            .map(|text| !text.trim().is_empty())
            .unwrap_or(false);
        if let Some(message) = durable_write_guard(
            ctx,
            &path,
            &args.content,
            DurableWriteMode::from(mode),
            existing_nonempty,
        ) {
            return Ok(ToolResult::error(call, message));
        }
        if let Some(msg) = check_read_before_write(&path) {
            return Ok(ToolResult::error(call, msg));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| ToolError::Failed(err.to_string()))?;
        }
        match mode {
            WriteMode::Overwrite => {
                fs::write(&path, &args.content).map_err(|err| ToolError::Failed(err.to_string()))?
            }
            WriteMode::Append => {
                let mut file = fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .map_err(|err| ToolError::Failed(err.to_string()))?;
                file.write_all(args.content.as_bytes())
                    .map_err(|err| ToolError::Failed(err.to_string()))?;
            }
            WriteMode::Prepend => {
                let old = fs::read_to_string(&path).unwrap_or_default();
                fs::write(&path, format!("{}{}", args.content, old))
                    .map_err(|err| ToolError::Failed(err.to_string()))?;
            }
        }
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "path": path,
                "written_bytes": args.content.len(),
            }),
        ))
    }
}

pub(crate) fn read_file_window(
    path: &Path,
    start: usize,
    count: usize,
    keyword: Option<&str>,
    show_line_numbers: bool,
) -> anyhow::Result<String> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    let keyword = keyword.map(str::to_lowercase);

    if let Some(keyword) = keyword {
        let before_size = (count / 3).max(1);
        let mut before = VecDeque::with_capacity(before_size);
        for (idx, line) in reader.lines().enumerate().skip(start - 1) {
            let line_no = idx + 1;
            let line = line?;
            if line.to_lowercase().contains(&keyword) {
                rows.extend(before);
                rows.push((line_no, line));
                break;
            }
            if before.len() == before_size {
                before.pop_front();
            }
            before.push_back((line_no, line));
        }
    } else {
        for (idx, line) in reader.lines().enumerate().skip(start - 1).take(count) {
            rows.push((idx + 1, line?));
        }
    }

    if rows.is_empty() {
        return Ok("[FILE] no matching content".to_string());
    }

    let mut out = format!(
        "[FILE] showing {} lines from {}\n",
        rows.len(),
        path.display()
    );
    for (line_no, mut line) in rows.into_iter().take(count) {
        if line.len() > 8_000 {
            truncate_utf8(&mut line, 8_000);
            line.push_str(" ... [TRUNCATED]");
        }
        if show_line_numbers {
            out.push_str(&format!("{line_no}|{line}\n"));
        } else {
            out.push_str(&line);
            out.push('\n');
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn check_read_before_write_blocks_unread_existing_file() {
        crate::read_paths_guard::reset();
        // Use a temp file so the test is hermetic.
        let dir = std::env::temp_dir().join(format!(
            "seed-rbw-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("foo.txt");
        fs::write(&path, "original").unwrap();
        let blocked = check_read_before_write(&path);
        assert!(blocked.is_some(), "existing unread file should be blocked");
        crate::read_paths_guard::record(&path);
        let allowed = check_read_before_write(&path);
        assert!(allowed.is_none(), "after read, write should be allowed");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn check_read_before_write_allows_new_file() {
        crate::read_paths_guard::reset();
        let path = std::env::temp_dir().join(format!(
            "seed-rbw-new-{}-{}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert!(!path.exists());
        let result = check_read_before_write(&path);
        assert!(result.is_none(), "new file should be allowed");
    }

    #[test]
    fn read_file_args_accept_file_alias() {
        let v = serde_json::json!({"file": "Cargo.toml"});
        let args: ReadFileArgs = serde_json::from_value(v).unwrap();
        assert_eq!(args.path, "Cargo.toml");
    }

    #[test]
    fn read_files_args_accept_files_alias() {
        let v = serde_json::json!({"files": ["a", "b"]});
        let args: ReadFilesArgs = serde_json::from_value(v).unwrap();
        let paths: Vec<&str> = args.paths.iter().map(|e| e.path()).collect();
        assert_eq!(paths, vec!["a", "b"]);
    }

    // --- read_files per-file spec ----------------------------------

    #[test]
    fn read_files_accepts_string_array() {
        // Existing Vec<String> shape still works.
        let v = serde_json::json!({"paths": ["src/lib.rs", "Cargo.toml"]});
        let args: ReadFilesArgs = serde_json::from_value(v).unwrap();
        assert_eq!(args.paths.len(), 2);
        assert_eq!(args.paths[0].path(), "src/lib.rs");
        // Plain entries inherit caller fallback values.
        let (start, count, kw) = args.paths[0].resolved(5, 100, Some("foo"));
        assert_eq!(start, 5);
        assert_eq!(count, 100);
        assert_eq!(kw, Some("foo".to_string()));
    }

    #[test]
    fn read_files_accepts_per_file_spec_objects() {
        // The shape the planner tried in the verification re-run:
        // files=[{path, start, count}, …].
        let v = serde_json::json!({
            "files": [
                {"path": "a.rs", "start": 1, "count": 50},
                {"path": "b.rs", "start": 200, "count": 60}
            ]
        });
        let args: ReadFilesArgs = serde_json::from_value(v).unwrap();
        assert_eq!(args.paths.len(), 2);
        assert_eq!(args.paths[0].path(), "a.rs");
        // Detailed entries override fallback with their own values.
        let (start_a, count_a, _) = args.paths[0].resolved(999, 999, None);
        assert_eq!(start_a, 1);
        assert_eq!(count_a, 50);
        let (start_b, count_b, _) = args.paths[1].resolved(999, 999, None);
        assert_eq!(start_b, 200);
        assert_eq!(count_b, 60);
    }

    #[test]
    fn read_files_mixed_plain_and_detailed_in_one_array() {
        // Untagged enum should accept a mix.
        let v = serde_json::json!({
            "paths": [
                "a.rs",
                {"path": "b.rs", "count": 99}
            ]
        });
        let args: ReadFilesArgs = serde_json::from_value(v).unwrap();
        assert_eq!(args.paths.len(), 2);
        let (_, count_a, _) = args.paths[0].resolved(1, 200, None);
        let (_, count_b, _) = args.paths[1].resolved(1, 200, None);
        assert_eq!(count_a, 200); // plain → fallback
        assert_eq!(count_b, 99); // detailed → override
    }

    #[test]
    fn read_files_per_file_omitted_fields_use_fallback() {
        // Object entry with only `path` → start/count/keyword come from
        // the caller's defaults.
        let v = serde_json::json!({"paths": [{"path": "x.rs"}]});
        let args: ReadFilesArgs = serde_json::from_value(v).unwrap();
        let (start, count, kw) = args.paths[0].resolved(10, 77, Some("TODO"));
        assert_eq!(start, 10);
        assert_eq!(count, 77);
        assert_eq!(kw, Some("TODO".to_string()));
    }

    #[test]
    fn read_files_per_file_accepts_field_aliases() {
        // The same alias map as ReadFileArgs (file/start_line/from/etc.)
        // works inside the per-file object too.
        let v = serde_json::json!({
            "paths": [
                {"file": "x.rs", "start_line": 5, "lines": 30, "pattern": "foo"}
            ]
        });
        let args: ReadFilesArgs = serde_json::from_value(v).unwrap();
        assert_eq!(args.paths[0].path(), "x.rs");
        let (start, count, kw) = args.paths[0].resolved(1, 200, None);
        assert_eq!(start, 5);
        assert_eq!(count, 30);
        assert_eq!(kw, Some("foo".to_string()));
    }

    #[test]
    fn patch_file_args_accept_anthropic_style_aliases() {
        // Anthropic tool-use shape: {old, new} not {old_content, new_content}.
        let v = serde_json::json!({
            "path": "f.rs",
            "old": "foo",
            "new": "bar"
        });
        let args: PatchFileArgs = serde_json::from_value(v).unwrap();
        assert_eq!(args.old_content, "foo");
        assert_eq!(args.new_content, "bar");
    }

    #[test]
    fn write_file_args_accept_text_alias() {
        let v = serde_json::json!({"path": "out.txt", "text": "hi"});
        let args: WriteFileArgs = serde_json::from_value(v).unwrap();
        assert_eq!(args.content, "hi");
    }
}

#[cfg(test)]
mod tests_dir_expansion {
    //! : end-to-end test that read_files auto-expands a
    //! directory entry via the gitignore-aware walker.
    use super::*;
    use agent_core::{ToolContext, ToolCall, Tool};
    use std::fs;
    use std::process::Command;

    fn temp_dir(test: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "seed-readfiles-expand-{}-{}-{}",
            test,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn read_files_expands_a_directory_path() {
        let root = temp_dir("dir_expand");
        let _ = Command::new("git").args(["init", "-q"]).current_dir(&root).output();
        fs::write(root.join(".gitignore"), "noise.log\n").unwrap();
        fs::write(root.join("a.rs"), "fn a(){}").unwrap();
        fs::write(root.join("b.rs"), "fn b(){}").unwrap();
        fs::write(root.join("noise.log"), "should be filtered").unwrap();

        let ctx = ToolContext::with_cwd(&root);
        let call = ToolCall::new("read_files", serde_json::json!({"paths": ["."]}));
        let result = ReadFilesTool.execute(&ctx, &call).unwrap();
        assert!(result.ok, "expected success, got: {:?}", result.content);

        let files = result.content["files"].as_array().unwrap();
        let names: Vec<String> = files
            .iter()
            .filter_map(|f| f["path"].as_str())
            .map(|p| std::path::Path::new(p).file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"a.rs".to_string()), "got: {names:?}");
        assert!(names.contains(&"b.rs".to_string()), "got: {names:?}");
        // gitignore'd files should NOT appear
        assert!(!names.contains(&"noise.log".to_string()), "noise.log should be filtered: {names:?}");
        // Expansion record surfaces.
        assert!(result.content.get("expanded_from_dirs").is_some(), "missing expanded_from_dirs trace");

        let _ = fs::remove_dir_all(&root);
    }
}
