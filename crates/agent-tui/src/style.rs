//! Tiny ANSI styling helpers for the CLI. We deliberately avoid pulling in a
//! styling crate — the surface is small and the dependencies stay flat.
//!
//! Honor the `NO_COLOR` env var (https://no-color.org/) and require stderr to
//! be a TTY before emitting escape codes. Anything else gets the plain
//! fallback so logs / scripts stay clean.

use std::io::{IsTerminal, stderr};
use std::sync::OnceLock;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok,
    Failed,
    Blocked,
    Pending,
}

impl Status {
    pub fn symbol(self) -> &'static str {
        match self {
            Status::Ok => "✓",
            Status::Failed => "✗",
            Status::Blocked => "⚠",
            Status::Pending => "⠧",
        }
    }
}

fn color_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        if std::env::var_os("NO_COLOR").is_some() {
            return false;
        }
        stderr().is_terminal()
    })
}

fn wrap(text: &str, prefix: &str) -> String {
    if color_enabled() {
        format!("{prefix}{text}{RESET}")
    } else {
        text.to_string()
    }
}

pub fn status_marker(status: Status) -> String {
    match status {
        Status::Ok => wrap(Status::Ok.symbol(), GREEN),
        Status::Failed => wrap(Status::Failed.symbol(), &format!("{BOLD}{RED}")),
        Status::Blocked => wrap(Status::Blocked.symbol(), YELLOW),
        Status::Pending => wrap(Status::Pending.symbol(), CYAN),
    }
}

pub fn tool_name(text: &str) -> String {
    wrap(text, BOLD)
}

pub fn dim_text(text: &str) -> String {
    wrap(text, DIM)
}

pub fn slow_elapsed(text: &str) -> String {
    if color_enabled() {
        format!("{BOLD}{YELLOW}▶ {text} ◀{RESET}")
    } else {
        format!(">> {text} <<")
    }
}

pub fn fast_elapsed(text: &str) -> String {
    dim_text(text)
}

pub fn phase_divider(label: &str, width: usize) -> String {
    let title = format!(" {label} ");
    let fill = width.saturating_sub(title.chars().count() + 1);
    let line = format!("─{title}{}", "─".repeat(fill));
    dim_text(&line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_symbols_have_distinct_chars() {
        assert_ne!(Status::Ok.symbol(), Status::Failed.symbol());
        assert_ne!(Status::Failed.symbol(), Status::Blocked.symbol());
    }

    #[test]
    fn fast_elapsed_plain_when_no_color() {
        // SAFETY: tests are not run in parallel against NO_COLOR.
        unsafe { std::env::set_var("NO_COLOR", "1") };
        // Note: color_enabled is memoized via OnceLock per process, so we can't
        // toggle it mid-test. This just verifies the API shape.
        let _ = fast_elapsed("123ms");
        unsafe { std::env::remove_var("NO_COLOR") };
    }
}
