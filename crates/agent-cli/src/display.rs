//! Display + formatting helpers used by the planner-loop output path.
//!
//! Pulled out of `main.rs` so the bin's entry point is no longer a kitchen
//! sink. Everything here is `pub(crate)` because callers live elsewhere in the
//! same binary; nothing here is library-level reusable.

use std::cell::{Cell, RefCell};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;

// ---------------------------------------------------------------------------
// Slow-op threshold + value helpers
// ---------------------------------------------------------------------------

pub(crate) const SLOW_OP_THRESHOLD: Duration = Duration::from_secs(5);

pub(crate) fn compact_single_line_cli(input: &str, max_len: usize) -> String {
    let text = input.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.chars().count() <= max_len {
        return text;
    }
    let keep = max_len / 2;
    let head = text.chars().take(keep).collect::<String>();
    let tail = text
        .chars()
        .rev()
        .take(keep)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("{head} ...[omitted]... {tail}")
}

pub(crate) fn format_token_subtitle(chars: usize) -> String {
    if chars < 1000 {
        format!("{chars} chars")
    } else {
        format!("{:.1}k chars", chars as f64 / 1000.0)
    }
}

pub(crate) fn format_elapsed_cli(elapsed: Duration) -> String {
    let secs = elapsed.as_secs_f64();
    if secs < 1.0 {
        format!("{}ms", elapsed.as_millis())
    } else if secs < 60.0 {
        format!("{secs:.2}s")
    } else {
        let mins = (secs / 60.0).floor() as u64;
        let rem = secs - (mins as f64) * 60.0;
        format!("{mins}m{rem:.1}s")
    }
}

// ---------------------------------------------------------------------------
// Path-aware formatting (cwd elision + segment-boundary truncation)
// ---------------------------------------------------------------------------

thread_local! {
    /// Per-run cwd context for path elision. Set by run_planner_tool before
    /// formatting; cleared at the end of each tool invocation. Lets
    /// `preview_value_for_key` collapse absolute paths that live under cwd
    /// into a short relative form (`crates/agent-cli/src/main.rs`).
    pub(crate) static DISPLAY_CWD: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

pub(crate) fn set_display_cwd(cwd: &Path) {
    DISPLAY_CWD.with(|cell| *cell.borrow_mut() = Some(cwd.to_path_buf()));
}

pub(crate) fn elide_path_under_cwd(text: &str) -> String {
    DISPLAY_CWD.with(|cell| {
        let guard = cell.borrow();
        let Some(cwd) = guard.as_ref() else {
            return text.to_string();
        };
        let cwd_str = cwd.to_string_lossy();
        if cwd_str.is_empty() || cwd_str == "/" {
            return text.to_string();
        }
        let prefix = if cwd_str.ends_with('/') {
            cwd_str.to_string()
        } else {
            format!("{cwd_str}/")
        };
        if text.starts_with(&prefix) {
            let rel = &text[prefix.len()..];
            if rel.is_empty() {
                ".".to_string()
            } else {
                rel.to_string()
            }
        } else if text == cwd_str.as_ref() {
            ".".to_string()
        } else {
            text.to_string()
        }
    })
}

pub(crate) fn is_path_key(key: &str) -> bool {
    matches!(
        key,
        "path"
            | "paths"
            | "file_path"
            | "filepath"
            | "file_paths"
            | "working_dir"
            | "working_dirs"
            | "workdir"
            | "cwd"
            | "export_path"
            | "context_files"
            | "input_files"
    )
}

pub(crate) fn preview_path_value(text: &str, max_chars: usize) -> String {
    let cleaned = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.chars().count() <= max_chars {
        return cleaned;
    }
    // Snap truncation to a `/` component boundary so we never produce
    // half-words like `…/uange/WorkSpace/...`. Keep the tail components that
    // collectively fit under the budget; if even one component is too long,
    // fall back to char-level truncation on that tail only.
    let segments: Vec<&str> = cleaned.split('/').collect();
    let budget = max_chars.saturating_sub(2).max(8); // 2 for the `…/` prefix
    let mut kept: Vec<&str> = Vec::new();
    let mut used = 0usize;
    for seg in segments.iter().rev() {
        // +1 for the joining `/`
        let need = seg.chars().count() + if kept.is_empty() { 0 } else { 1 };
        if used + need > budget {
            break;
        }
        used += need;
        kept.push(seg);
    }
    kept.reverse();
    if kept.is_empty() {
        // Last-resort fallback: keep the final budget chars of the last segment.
        let last = segments.last().copied().unwrap_or("");
        let chars: Vec<char> = last.chars().collect();
        let start = chars.len().saturating_sub(budget);
        let tail: String = chars[start..].iter().collect();
        return format!("…/{tail}");
    }
    format!("…/{}", kept.join("/").trim_start_matches('/'))
}

// ---------------------------------------------------------------------------
// Argument / value preview (object inline, array inline, path-aware)
// ---------------------------------------------------------------------------

pub(crate) fn format_tool_args_cli(args: &Value, max_len: usize) -> String {
    if let Some(obj) = args.as_object() {
        if obj.is_empty() {
            return String::new();
        }
        let mut parts: Vec<String> = Vec::new();
        let mut dropped: Vec<String> = Vec::new();
        let mut used = 0usize;
        for (key, value) in obj {
            let entry = format!("{key}={}", preview_value_for_key(key, value, 60));
            let entry_len = entry.chars().count();
            let needed = used + entry_len + if parts.is_empty() { 0 } else { 1 };
            if needed > max_len && !parts.is_empty() {
                dropped.push(key.clone());
                continue;
            }
            used = needed;
            parts.push(entry);
        }
        if !dropped.is_empty() {
            parts.push(format!("+{}", dropped.join(",")));
        }
        return parts.join(" ");
    }
    compact_single_line_cli(&args.to_string(), max_len)
}

pub(crate) fn preview_value_for_key(key: &str, value: &Value, max_chars: usize) -> String {
    let looks_path = is_path_key(key);
    if looks_path {
        if let Value::String(text) = value {
            let elided = elide_path_under_cwd(text);
            return format!("\"{}\"", preview_path_value(&elided, max_chars));
        }
        if let Value::Array(items) = value
            && items.iter().all(|v| matches!(v, Value::String(_)))
        {
            return preview_array_inline(items, max_chars, |v, budget| {
                let raw = v.as_str().unwrap_or("");
                let elided = elide_path_under_cwd(raw);
                format!(
                    "\"{}\"",
                    preview_path_value(&elided, budget.saturating_sub(2))
                )
            });
        }
    }
    preview_arg_value(value, max_chars)
}

/// Try to render a JSON array inline: `[a, b, c]`. Falls back to
/// `[a, b, +N more]` if too long; final fallback is `[N items]`. Each item
/// gets rendered via the supplied closure with a per-item budget.
pub(crate) fn preview_array_inline(
    items: &[Value],
    max_chars: usize,
    item_preview: impl Fn(&Value, usize) -> String,
) -> String {
    if items.is_empty() {
        return "[]".to_string();
    }
    let separators = items.len().saturating_sub(1) * 2;
    let per_item = max_chars
        .saturating_sub(2 + separators)
        .checked_div(items.len())
        .unwrap_or(0)
        .max(16);

    let pieces: Vec<String> = items.iter().map(|v| item_preview(v, per_item)).collect();
    let full = format!("[{}]", pieces.join(", "));
    if full.chars().count() <= max_chars {
        return full;
    }
    if items.len() > 2 {
        let head1 = item_preview(&items[0], per_item);
        let head2 = item_preview(&items[1], per_item);
        let partial = format!("[{}, {}, +{} more]", head1, head2, items.len() - 2);
        if partial.chars().count() <= max_chars {
            return partial;
        }
    }
    if items.len() > 1 {
        let head = item_preview(&items[0], per_item);
        let partial = format!("[{}, +{} more]", head, items.len() - 1);
        if partial.chars().count() <= max_chars {
            return partial;
        }
    }
    format!("[{} items]", items.len())
}

pub(crate) fn preview_arg_value(value: &Value, max_chars: usize) -> String {
    match value {
        Value::String(text) => {
            let cleaned = text.split_whitespace().collect::<Vec<_>>().join(" ");
            if cleaned.chars().count() <= max_chars {
                format!("\"{cleaned}\"")
            } else {
                let head: String = cleaned.chars().take(max_chars).collect();
                format!("\"{head}…\"")
            }
        }
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::Array(items) => {
            if items.len() == 1 {
                format!("[{}]", preview_arg_value(&items[0], max_chars.saturating_sub(2)))
            } else if items.len() <= 4 {
                preview_array_inline(items, max_chars, preview_arg_value)
            } else {
                format!("[{} items]", items.len())
            }
        }
        Value::Object(fields) => {
            if fields.is_empty() {
                "{}".to_string()
            } else if fields.len() <= 3 {
                let inner: Vec<String> = fields
                    .iter()
                    .map(|(k, v)| {
                        format!("{k}={}", preview_arg_value(v, max_chars.saturating_sub(4).max(8)))
                    })
                    .collect();
                let joined = inner.join(",");
                if joined.chars().count() + 2 <= max_chars {
                    format!("{{{joined}}}")
                } else {
                    format!("{{{} keys}}", fields.len())
                }
            } else {
                format!("{{{} keys}}", fields.len())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Wrapper-aware tool label + args display
// ---------------------------------------------------------------------------

/// Wrapper-aware display: tools like `repoprompt_call` take a nested
/// `{tool, args}` envelope. Returns (Option<inner_tool_name>, args_display).
/// The caller composes the visible label as `wrapper::inner_tool` when the
/// inner name is present, so the most important fact ("which underlying tool
/// is actually firing") leads the line instead of being buried at the end.
pub(crate) fn format_call_args_for_display(
    name: &str,
    args: &Value,
    max_len: usize,
) -> (Option<String>, String) {
    if name == "repoprompt_call"
        && let Some(obj) = args.as_object()
    {
        let inner_tool = obj
            .get("tool")
            .or_else(|| obj.get("tool_name"))
            .or_else(|| obj.get("name"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let inner_args = obj.get("args").or_else(|| obj.get("params"));
        if let Some(tool) = inner_tool {
            let mut display = serde_json::Map::new();
            if let Some(args_value) = inner_args
                && let Some(args_obj) = args_value.as_object()
            {
                for (k, v) in args_obj {
                    display.insert(k.clone(), v.clone());
                }
            }
            return (
                Some(tool),
                format_tool_args_cli(&Value::Object(display), max_len),
            );
        }
    }
    if name == "repoprompt_exec"
        && let Some(obj) = args.as_object()
        && let Some(cmd) = obj
            .get("command")
            .or_else(|| obj.get("cmd"))
            .and_then(Value::as_str)
    {
        let limit = max_len.saturating_sub(4).max(20);
        return (None, format!("\"{}\"", compact_single_line_cli(cmd, limit)));
    }
    (None, format_tool_args_cli(args, max_len))
}

pub(crate) fn compose_tool_label(name: &str, inner: Option<&str>) -> String {
    match inner {
        Some(inner_name) if !inner_name.is_empty() => format!("{name}::{inner_name}"),
        _ => name.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Failure-reason extraction for `✗` tool lines
// ---------------------------------------------------------------------------

/// Extract a short error reason from a ToolResult content blob and prefix it
/// with `→` so it visually attaches to the status line. Falls back to a
/// generic label when no `message` field is present.
pub(crate) fn short_failure_reason(content: &Value) -> String {
    let raw = content
        .get("message")
        .or_else(|| content.get("error"))
        .or_else(|| content.get("reason"))
        .and_then(Value::as_str)
        .unwrap_or("");
    short_failure_text(raw)
}

pub(crate) fn short_failure_text(raw: &str) -> String {
    let cleaned = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.is_empty() {
        return "→ failed".to_string();
    }
    let max_chars = 90;
    if cleaned.chars().count() <= max_chars {
        format!("→ {cleaned}")
    } else {
        let head: String = cleaned.chars().take(max_chars).collect();
        format!("→ {head}…")
    }
}

// ---------------------------------------------------------------------------
// Phase divider (auto-grouping consecutive tools by category)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolPhase {
    Explore,
    Memory,
    Plan,
    Verify,
    Execute,
    Other,
}

impl ToolPhase {
    pub(crate) fn classify(name: &str) -> Self {
        // `compose_tool_label` may have already joined wrapper::inner; classify
        // on the wrapper name so repoprompt_call::get_file_tree still lands in
        // Explore rather than Other.
        let outer = name.split("::").next().unwrap_or(name);
        match outer {
            "read_file"
            | "read_files"
            | "repoprompt_call"
            | "repoprompt_exec"
            | "repoprompt_tools"
            | "run_shell"
            | "skill_list"
            | "skill_search"
            | "skill_fetch" => ToolPhase::Explore,
            "memory_search"
            | "memory_fetch"
            | "update_working_checkpoint"
            | "start_long_term_update"
            | "complete_long_term_update" => ToolPhase::Memory,
            "plan_create"
            | "plan_create_from_repoprompt"
            | "plan_create_via_repoprompt"
            | "plan_refine_via_repoprompt"
            | "plan_next"
            | "plan_status"
            | "plan_list"
            | "plan_complete"
            | "plan_record_artifact"
            | "plan_record_handoff" => ToolPhase::Plan,
            "plan_verify" => ToolPhase::Verify,
            "patch_file" | "write_file" | "spawn_subagent" | "spawn_subagent_map"
            | "subagent_nudge" => ToolPhase::Execute,
            _ => ToolPhase::Other,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            ToolPhase::Explore => "explore",
            ToolPhase::Memory => "memory",
            ToolPhase::Plan => "plan",
            ToolPhase::Verify => "verify",
            ToolPhase::Execute => "execute",
            ToolPhase::Other => "other",
        }
    }
}

thread_local! {
    static LAST_PHASE: Cell<Option<ToolPhase>> = const { Cell::new(None) };
}

pub(crate) fn reset_phase_tracker() {
    LAST_PHASE.with(|cell| cell.set(None));
}

fn maybe_emit_phase_divider(err: &mut impl Write, name: &str) {
    let phase = ToolPhase::classify(name);
    let prev = LAST_PHASE.with(|cell| cell.get());
    if prev != Some(phase) {
        if prev.is_some() {
            let _ = writeln!(err);
        }
        let _ = writeln!(err, "{}", agent_core::tui::phase_divider(phase.label(), 70));
        LAST_PHASE.with(|cell| cell.set(Some(phase)));
    }
}

// ---------------------------------------------------------------------------
// The `seed → ...` tool line itself
// ---------------------------------------------------------------------------

pub(crate) fn emit_tool_line(
    spinner: Option<&agent_core::tui::Spinner>,
    name: &str,
    args: &str,
    status: agent_core::tui::Status,
    elapsed: Option<Duration>,
    note: &str,
) {
    if let Some(s) = spinner {
        s.pause();
    }
    let mut err = io::stderr().lock();
    // Defensive prefix: even with synchronous pause(), a tiny window exists
    // where the spinner thread can paint between the caller's send and ack.
    // \r\x1b[2K resets the cursor + clears the line so the message is never
    // concatenated onto a residual spinner frame (e.g. `14.7sseed → ...`).
    let _ = write!(err, "\r\x1b[2K");
    maybe_emit_phase_divider(&mut err, name);

    let marker = agent_core::tui::status_marker(status);
    let styled_name = agent_core::tui::tool_name(name);
    let elapsed_chunk = match elapsed {
        Some(d) if d >= SLOW_OP_THRESHOLD => format!(
            " {}",
            agent_core::tui::slow_elapsed(&agent_core::tui::format_elapsed(d))
        ),
        Some(d) => format!(
            " {}",
            agent_core::tui::fast_elapsed(&agent_core::tui::format_elapsed(d))
        ),
        None => String::new(),
    };
    let note_chunk = if note.is_empty() {
        String::new()
    } else {
        format!(" {}", agent_core::tui::dim_text(note))
    };
    let body = if args.is_empty() {
        format!("{marker} {styled_name}{elapsed_chunk}{note_chunk}")
    } else {
        format!("{marker} {styled_name}  {args}{elapsed_chunk}{note_chunk}")
    };
    let _ = writeln!(err, "{body}");
    let _ = err.flush();
    if let Some(s) = spinner {
        s.resume();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn format_call_args_unwraps_repoprompt_call_envelope() {
        let envelope = json!({
            "tool": "get_file_tree",
            "args": {"depth": 3, "path": "."},
        });
        let (inner, args_text) = format_call_args_for_display("repoprompt_call", &envelope, 120);
        assert_eq!(inner.as_deref(), Some("get_file_tree"));
        assert!(args_text.contains("depth=3"), "got: {args_text}");
        assert!(args_text.contains("path=\".\""), "got: {args_text}");
        assert!(!args_text.contains("args={"), "got: {args_text}");
        assert_eq!(
            compose_tool_label("repoprompt_call", inner.as_deref()),
            "repoprompt_call::get_file_tree"
        );
    }

    #[test]
    fn format_call_args_handles_tool_name_field_alias() {
        let envelope = json!({
            "tool_name": "file_search",
            "args": {"pattern": "TODO"},
        });
        let (inner, args_text) = format_call_args_for_display("repoprompt_call", &envelope, 120);
        assert_eq!(inner.as_deref(), Some("file_search"));
        assert!(args_text.contains("pattern=\"TODO\""), "got: {args_text}");
    }

    #[test]
    fn format_call_args_inlines_repoprompt_exec_command() {
        let envelope = json!({ "command": "tree --mode folders" });
        let (inner, args_text) = format_call_args_for_display("repoprompt_exec", &envelope, 120);
        assert!(inner.is_none());
        assert!(args_text.contains("\"tree --mode folders\""), "got: {args_text}");
    }

    #[test]
    fn format_call_args_falls_back_for_unknown_tool() {
        let args = json!({ "a": 1, "b": "two" });
        let (inner, args_text) = format_call_args_for_display("read_file", &args, 120);
        assert!(inner.is_none());
        assert!(args_text.contains("a=1"));
        assert!(args_text.contains("b=\"two\""));
    }

    #[test]
    fn elide_path_under_cwd_makes_relative_when_inside_cwd() {
        set_display_cwd(Path::new("/Users/me/repo"));
        assert_eq!(
            elide_path_under_cwd("/Users/me/repo/crates/agent-cli/src/main.rs"),
            "crates/agent-cli/src/main.rs"
        );
        assert_eq!(elide_path_under_cwd("/Users/me/repo"), ".");
        assert_eq!(
            elide_path_under_cwd("/Users/other/elsewhere"),
            "/Users/other/elsewhere"
        );
        DISPLAY_CWD.with(|cell| cell.borrow_mut().take());
    }

    #[test]
    fn preview_value_for_key_uses_relative_path_when_under_cwd() {
        set_display_cwd(Path::new("/Users/me/repo"));
        let value = json!("/Users/me/repo/crates/agent-tools/src/lib.rs");
        let preview = preview_value_for_key("path", &value, 80);
        assert_eq!(preview, "\"crates/agent-tools/src/lib.rs\"");
        DISPLAY_CWD.with(|cell| cell.borrow_mut().take());
    }

    #[test]
    fn preview_value_for_key_inlines_multi_path_array_under_cwd() {
        set_display_cwd(Path::new("/Users/me/repo"));
        let value = json!([
            "/Users/me/repo/crates/agent-cli/src/main.rs",
            "/Users/me/repo/crates/agent-tools/src/lib.rs",
        ]);
        let preview = preview_value_for_key("paths", &value, 120);
        assert!(preview.contains("crates/agent-cli/src/main.rs"), "got: {preview}");
        assert!(preview.contains("crates/agent-tools/src/lib.rs"), "got: {preview}");
        assert!(!preview.contains("items]"), "got: {preview}");
        DISPLAY_CWD.with(|cell| cell.borrow_mut().take());
    }

    #[test]
    fn preview_value_for_key_falls_back_to_count_when_paths_too_long() {
        set_display_cwd(Path::new("/Users/me/repo"));
        let value = json!([
            "/Users/me/repo/very/long/path/segment/a.rs",
            "/Users/me/repo/very/long/path/segment/b.rs",
            "/Users/me/repo/very/long/path/segment/c.rs",
            "/Users/me/repo/very/long/path/segment/d.rs",
        ]);
        let preview = preview_value_for_key("paths", &value, 30);
        assert!(
            preview.contains("more") || preview.contains("items]") || preview.chars().count() <= 32,
            "got: {preview}"
        );
        DISPLAY_CWD.with(|cell| cell.borrow_mut().take());
    }

    #[test]
    fn preview_value_for_key_folds_single_element_path_array() {
        let path_array = json!(["/Users/duange/WorkSpace/seed-agent-rs/crates/agent-runtime/src/lib.rs"]);
        let preview = preview_value_for_key("paths", &path_array, 60);
        assert!(preview.starts_with("[\"…/"), "got: {preview}");
        assert!(preview.ends_with("lib.rs\"]"), "got: {preview}");
    }

    #[test]
    fn preview_path_value_keeps_tail_when_too_long() {
        let path = "/Users/duange/WorkSpace/seed-agent-rs/crates/agent-runtime/src/lib.rs";
        let preview = preview_path_value(path, 40);
        assert!(preview.starts_with("…/") && preview.ends_with("lib.rs"), "got: {preview}");
    }

    #[test]
    fn preview_path_value_snaps_to_component_boundary() {
        let path = "/Users/duange/WorkSpace/seed-agent-rs/crates/agent-cli/src/main.rs";
        let preview = preview_path_value(path, 50);
        assert!(preview.starts_with("…/"));
        let tail = preview.trim_start_matches("…/");
        let first_segment = tail.split('/').next().unwrap_or("");
        assert!(
            path.split('/').any(|seg| seg == first_segment),
            "first segment {first_segment:?} is not a complete original component (preview: {preview})"
        );
    }

    #[test]
    fn preview_arg_value_inlines_tiny_nested_objects() {
        let value = json!({"include_globs": ["*.rs"], "max_results": 80});
        let preview = preview_arg_value(&value, 60);
        assert!(preview.starts_with('{') && preview.ends_with('}'));
        assert!(preview.contains("include_globs="));
        assert!(preview.contains("max_results=80"));
        assert!(!preview.contains("keys}"), "got: {preview}");
    }

    #[test]
    fn preview_arg_value_keeps_opaque_summary_for_big_objects() {
        let mut value = serde_json::Map::new();
        for i in 0..6 {
            value.insert(format!("k{i}"), json!(i));
        }
        let preview = preview_arg_value(&Value::Object(value), 60);
        assert!(preview.contains("keys}"), "got: {preview}");
    }

    #[test]
    fn preview_arg_value_inlines_short_generic_arrays() {
        let value = json!(["alpha", "beta", "gamma"]);
        let preview = preview_arg_value(&value, 60);
        assert!(preview.contains("\"alpha\""), "got: {preview}");
        assert!(preview.contains("\"gamma\""), "got: {preview}");
        assert!(!preview.contains("items]"), "got: {preview}");
    }

    #[test]
    fn preview_arg_value_uses_count_for_long_arrays() {
        let value = json!(["a", "b", "c", "d", "e"]);
        let preview = preview_arg_value(&value, 60);
        assert!(preview.contains("items]"), "got: {preview}");
    }

    #[test]
    fn preview_arg_value_expands_single_element_arrays() {
        assert_eq!(preview_arg_value(&json!(["hello"]), 60), "[\"hello\"]");
        assert_eq!(
            preview_arg_value(&json!(["a", "b", "c"]), 60),
            "[\"a\", \"b\", \"c\"]"
        );
    }

    #[test]
    fn format_tool_args_drops_overflow_with_marker() {
        let mut value = serde_json::Map::new();
        for index in 0..12 {
            value.insert(format!("key_{index}"), json!(format!("value_{index}")));
        }
        let label = format_tool_args_cli(&Value::Object(value), 60);
        assert!(label.contains("+"), "got: {label}");
        assert!(label.contains("key_"), "got: {label}");
    }

    #[test]
    fn format_tool_args_elides_long_strings_per_field() {
        let args = json!({
            "command": "rg --files -g 'README*' -g 'docs/**' -g 'crates/**/src/**'",
            "timeout": 10000,
            "workdir": "/Users/duange/WorkSpace/seed-agent-rs",
        });
        let label = format_tool_args_cli(&args, 140);
        assert!(label.contains("command=\""));
        assert!(label.contains("timeout=10000"));
        assert!(label.contains("workdir=\"/Users/duange/WorkSpace/seed-agent-rs\""));
        assert!(!label.contains("[omitted]"));
    }

    #[test]
    fn format_tool_args_lists_dropped_field_names() {
        let args = json!({
            "command": "rg --files -g 'README*' -g 'docs/**' -g 'crates/**/src/**'",
            "timeout_ms": 30000,
            "workdir": "/Users/duange/WorkSpace/seed-agent-rs",
        });
        let label = format_tool_args_cli(&args, 60);
        assert!(label.contains("+"), "expected `+name` marker, got: {label}");
    }

    #[test]
    fn format_token_subtitle_collapses_thousands() {
        assert_eq!(format_token_subtitle(0), "0 chars");
        assert_eq!(format_token_subtitle(999), "999 chars");
        assert_eq!(format_token_subtitle(1234), "1.2k chars");
        assert_eq!(format_token_subtitle(42000), "42.0k chars");
    }

    #[test]
    fn format_elapsed_cli_picks_unit_by_magnitude() {
        assert_eq!(format_elapsed_cli(Duration::from_millis(250)), "250ms");
        assert_eq!(format_elapsed_cli(Duration::from_millis(1240)), "1.24s");
        assert_eq!(
            format_elapsed_cli(Duration::from_secs(65) + Duration::from_millis(300)),
            "1m5.3s"
        );
    }

    #[test]
    fn tool_phase_classify_handles_wrapper_inner_naming() {
        assert_eq!(
            ToolPhase::classify("repoprompt_call::get_file_tree"),
            ToolPhase::Explore
        );
        assert_eq!(
            ToolPhase::classify("repoprompt_call::apply_edits"),
            ToolPhase::Explore
        );
        assert_eq!(ToolPhase::classify("read_files"), ToolPhase::Explore);
        assert_eq!(ToolPhase::classify("plan_verify"), ToolPhase::Verify);
        assert_eq!(
            ToolPhase::classify("update_working_checkpoint"),
            ToolPhase::Memory
        );
        assert_eq!(ToolPhase::classify("some_future_tool"), ToolPhase::Other);
    }

    #[test]
    fn short_failure_reason_pulls_message_field() {
        let content = json!({"status": "error", "message": "paths must not be empty"});
        let note = short_failure_reason(&content);
        assert_eq!(note, "→ paths must not be empty");
    }

    #[test]
    fn short_failure_reason_truncates_long_messages() {
        let long = "A".repeat(200);
        let content = json!({"status": "error", "message": long});
        let note = short_failure_reason(&content);
        assert!(note.ends_with('…'), "got: {note}");
        assert!(note.chars().count() <= 95, "got len: {}", note.chars().count());
    }

    #[test]
    fn short_failure_reason_falls_back_to_label_when_no_message() {
        let content = json!({"status": "error"});
        assert_eq!(short_failure_reason(&content), "→ failed");
    }
}
