//! Process-wide state shared across tool calls within one run.
//!
//! Three thread-local guards: [`skill_tools_guard`] for skill-driven
//! tool catalog narrowing, [`run_mode_guard`] for the active [`RunMode`]
//! (read by `ShellTool` to refuse writes in read-only), and
//! [`repoprompt_sync`] for RepoPrompt coordination (pending skill
//! override, bound-window cache, sticky-cwd queue).

use agent_core::RunMode;

pub mod skill_tools_guard {
    use std::cell::RefCell;
    use std::collections::HashSet;

    thread_local! {
        static STATE: RefCell<Option<HashSet<String>>> = const { RefCell::new(None) };
    }

    /// Empty list clears the restriction.
    pub fn set(allowed: Vec<String>) {
        STATE.with(|cell| {
            *cell.borrow_mut() = if allowed.is_empty() {
                None
            } else {
                Some(allowed.into_iter().collect())
            };
        });
    }

    pub fn reset() {
        STATE.with(|cell| *cell.borrow_mut() = None);
    }

    /// Returns `true` when no restriction is active.
    pub fn permits(tool_name: &str) -> bool {
        STATE.with(|cell| match cell.borrow().as_ref() {
            None => true,
            Some(set) => set.contains(tool_name),
        })
    }

    pub fn current() -> Option<Vec<String>> {
        STATE.with(|cell| {
            cell.borrow().as_ref().map(|s| {
                let mut v: Vec<String> = s.iter().cloned().collect();
                v.sort();
                v
            })
        })
    }
}

pub mod run_mode_guard {
    use super::RunMode;
    use std::cell::Cell;

    thread_local! {
        static STATE: Cell<RunMode> = const { Cell::new(RunMode::Implementation) };
    }

    pub fn set(mode: RunMode) {
        STATE.with(|cell| cell.set(mode));
    }

    /// Defaults to `Implementation` if the guard has never been set on this thread.
    pub fn current() -> RunMode {
        STATE.with(|cell| cell.get())
    }

    pub fn reset() {
        set(RunMode::Implementation);
    }
}

/// Tracks which absolute paths have been read this run. Write tools
/// consult this to refuse "hallucinated" edits to files the planner
/// never inspected (Cline / Cursor convention). New-file creation is
/// always allowed — only writes to files that ALREADY EXIST on disk
/// and weren't read this run are blocked.
///
/// Escape hatch: set `SEED_DISABLE_READ_BEFORE_WRITE=1` to disable
/// (useful for migration scripts or codegen tools that legitimately
/// write without reading first).
pub mod read_paths_guard {
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};

    thread_local! {
        static STATE: RefCell<HashSet<PathBuf>> = RefCell::new(HashSet::new());
    }

    pub fn record(path: &Path) {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        STATE.with(|cell| {
            cell.borrow_mut().insert(canonical);
        });
    }

    /// True iff `path` has been read this run (after canonicalization).
    pub fn was_read(path: &Path) -> bool {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        STATE.with(|cell| cell.borrow().contains(&canonical))
    }

    pub fn reset() {
        STATE.with(|cell| cell.borrow_mut().clear());
    }

    /// True when the env-var escape hatch is active.
    pub fn is_disabled() -> bool {
        std::env::var("SEED_DISABLE_READ_BEFORE_WRITE")
            .ok()
            .filter(|v| !v.is_empty() && v != "0")
            .is_some()
    }
}

/// Carries one-shot suggestions from `skill_fetch` to the next
/// `repoprompt_*` tool call without changing the cross-tool API. Reset
/// at the top of every `run_goal` to prevent leakage across runs.
pub mod repoprompt_sync {
    use std::cell::RefCell;
    use std::path::PathBuf;

    #[derive(Debug, Default)]
    struct SyncState {
        /// `None` = fall back to `[ctx.cwd]`. Consumed (taken) by
        /// `default_repoprompt_working_dirs`.
        pending_override: Option<Vec<PathBuf>>,
        /// Cached `(working_dirs, window_id)` from the last successful
        /// `bind_context`. Lets matching rp calls skip one rp-cli roundtrip
        /// (~70ms). Invalidated by `reset`, by consuming a `pending_override`,
        /// and by `clear_bound_window`.
        bound: Option<BoundWindow>,
        /// Opt-in sticky cwd from a skill whose frontmatter set
        /// `sticky_cwd: true`. Polled by `run_goal` between turns.
        pending_sticky_cwd: Option<PathBuf>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct BoundWindow {
        pub working_dirs: Vec<PathBuf>,
        pub window_id: u32,
    }

    thread_local! {
        static STATE: RefCell<SyncState> = RefCell::new(SyncState::default());
    }

    fn with_state<R>(f: impl FnOnce(&mut SyncState) -> R) -> R {
        STATE.with(|cell| f(&mut cell.borrow_mut()))
    }

    fn read_state<R>(f: impl FnOnce(&SyncState) -> R) -> R {
        STATE.with(|cell| f(&cell.borrow()))
    }

    pub fn reset() {
        with_state(|st| {
            st.pending_override = None;
            st.bound = None;
            st.pending_sticky_cwd = None;
        });
    }

    pub fn set_pending_sticky_cwd(path: PathBuf) {
        with_state(|st| st.pending_sticky_cwd = Some(path));
    }

    pub fn take_pending_sticky_cwd() -> Option<PathBuf> {
        with_state(|st| st.pending_sticky_cwd.take())
    }

    pub fn peek_pending_sticky_cwd() -> Option<PathBuf> {
        read_state(|st| st.pending_sticky_cwd.clone())
    }

    /// Also clears the bound-window cache — we're about to switch dirs.
    pub fn set_pending_override(working_dirs: Vec<PathBuf>) {
        with_state(|st| {
            st.pending_override = Some(working_dirs);
            st.bound = None;
        });
    }

    pub fn take_pending_override() -> Option<Vec<PathBuf>> {
        with_state(|st| st.pending_override.take())
    }

    pub(crate) fn cached_window_id_for(working_dirs: &[PathBuf]) -> Option<u32> {
        read_state(|st| {
            st.bound
                .as_ref()
                .filter(|b| b.working_dirs == working_dirs)
                .map(|b| b.window_id)
        })
    }

    pub fn record_bound_window(working_dirs: Vec<PathBuf>, window_id: u32) {
        with_state(|st| {
            st.bound = Some(BoundWindow { working_dirs, window_id });
        });
    }

    /// Used when `/cd` changes the workspace cwd — the cached window is for the old cwd.
    pub fn clear_bound_window() {
        with_state(|st| st.bound = None);
    }

    pub fn peek_pending_override() -> Option<Vec<PathBuf>> {
        read_state(|st| st.pending_override.clone())
    }

    pub fn peek_bound_window() -> Option<(Vec<PathBuf>, u32)> {
        read_state(|st| {
            st.bound
                .as_ref()
                .map(|b| (b.working_dirs.clone(), b.window_id))
        })
    }
}
