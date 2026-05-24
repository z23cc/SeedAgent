//! RF40-A1: process-wide state that lives across tool calls within one run.
//!
//! Three submodules, all thread-local (RF40-A2):
//!
//! - [`skill_tools_guard`] — narrowed tool catalog when a skill's
//!   `allowed-tools` frontmatter is active.
//! - [`run_mode_guard`] — active [`RunMode`] (read-only vs implementation),
//!   read by `ShellTool` to refuse write-shaped commands in read-only.
//! - [`repoprompt_sync`] — RepoPrompt coordination state: pending skill
//!   override, bound-window cache, sticky-cwd queue.
//!
//! Each module's public surface is preserved verbatim from the pre-RF40
//! shape so callers and tests don't need to change. The split exists only
//! to keep `lib.rs` navigable.

use agent_core::RunMode;

/// RF40-A2: skill-driven tool narrow set, now thread-local. The previous
/// `OnceLock<Mutex<...>>` shape forced 3 separate test mutexes to
/// serialize state-mutating tests, and would have stomped if anyone ever
/// ran two `run_goal` calls in parallel. `thread_local!` gives us
/// per-thread state — tests get independent state for free (each test
/// runs on its own thread), and the architectural concurrent-run
/// safety is achieved at zero API cost.
pub mod skill_tools_guard {
    use std::cell::RefCell;
    use std::collections::HashSet;

    thread_local! {
        static STATE: RefCell<Option<HashSet<String>>> = const { RefCell::new(None) };
    }

    /// Replace the active narrow set. Empty list clears (no restriction).
    pub fn set(allowed: Vec<String>) {
        STATE.with(|cell| {
            *cell.borrow_mut() = if allowed.is_empty() {
                None
            } else {
                Some(allowed.into_iter().collect())
            };
        });
    }

    /// Clear restriction (back to no skill narrowing).
    pub fn reset() {
        STATE.with(|cell| *cell.borrow_mut() = None);
    }

    /// Check whether a tool name is allowed under the current restriction.
    /// Returns `true` when no restriction is active (most of the time).
    pub fn permits(tool_name: &str) -> bool {
        STATE.with(|cell| match cell.borrow().as_ref() {
            None => true,
            Some(set) => set.contains(tool_name),
        })
    }

    /// Inspect the current narrow set for tests / doctor display.
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

/// RF40-A2: active `RunMode` for the current thread. Same migration story
/// as `skill_tools_guard` above — `thread_local!` removes the need for
/// per-test serialization and makes concurrent `run_goal` safe by
/// construction.
pub mod run_mode_guard {
    use super::RunMode;
    use std::cell::Cell;

    thread_local! {
        static STATE: Cell<RunMode> = const { Cell::new(RunMode::Implementation) };
    }

    /// Set the active mode. Called by `run_goal` once at the top of each run.
    pub fn set(mode: RunMode) {
        STATE.with(|cell| cell.set(mode));
    }

    /// Read the active mode. Defaults to `Implementation` if the guard has
    /// never been set on this thread.
    pub fn current() -> RunMode {
        STATE.with(|cell| cell.get())
    }

    /// Reset to the default. Tests can call this; runtime code calls
    /// `set(...)` directly with the resolved mode.
    pub fn reset() {
        set(RunMode::Implementation);
    }
}

/// Process-wide RepoPrompt sync state. Tiny — only used to carry "one-shot
/// working_dirs override" suggestions from `skill_fetch` to the next
/// `repoprompt_*` tool call without changing the cross-tool API surface.
///
/// `repoprompt_client()` is built fresh per call from `ToolContext` + routing
/// args, so there's no natural place to hang per-run state. We use a static
/// guarded by a Mutex; `repoprompt_sync::reset()` is called at the top of
/// each `run_goal` to prevent leakage across runs.
pub mod repoprompt_sync {
    use std::cell::RefCell;
    use std::path::PathBuf;

    #[derive(Debug, Default)]
    struct SyncState {
        /// Working_dirs the next RepoPrompt bind should use *instead of*
        /// `[ctx.cwd]`. Consumed (taken) by `default_repoprompt_working_dirs`.
        /// `None` means "no skill override pending — fall back to ctx.cwd".
        pending_override: Option<Vec<PathBuf>>,
        /// RF25-2: cached `(working_dirs, window_id)` from the most recent
        /// successful `bind_context`. Subsequent rp calls with matching
        /// working_dirs can pre-set `cfg.window_id` from this cache,
        /// which short-circuits `resolve_repoprompt_window` and avoids one
        /// rp-cli subprocess (~70ms each). Invalidated by `reset()`, by
        /// consuming a `pending_override` (next call is intentionally
        /// targeting different dirs), and by `clear_bound_window`.
        bound: Option<BoundWindow>,
        /// RF33-2: opt-in sticky cwd request from a skill whose frontmatter
        /// set `sticky_cwd: true`. Polled by `run_goal` between turns; when
        /// present, it applies workspace.set_cwd(path) (and pushes the new
        /// cwd into the cached Codex client if live). Consumed on poll
        /// regardless of whether the apply succeeded.
        pending_sticky_cwd: Option<PathBuf>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct BoundWindow {
        pub working_dirs: Vec<PathBuf>,
        pub window_id: u32,
    }

    // RF40-A2: thread-local instead of OnceLock<Mutex>. Same behavior at
    // runtime (REPL is single-threaded; each run resets at start), but
    // tests get independent state without serialization mutexes and
    // any future concurrent-run embed-as-library use is safe by
    // construction.
    thread_local! {
        static STATE: RefCell<SyncState> = RefCell::new(SyncState::default());
    }

    fn with_state<R>(f: impl FnOnce(&mut SyncState) -> R) -> R {
        STATE.with(|cell| f(&mut cell.borrow_mut()))
    }

    fn read_state<R>(f: impl FnOnce(&SyncState) -> R) -> R {
        STATE.with(|cell| f(&cell.borrow()))
    }

    /// Clear all sync state. Call once at the start of a fresh run so stale
    /// pending overrides + window caches + sticky-cwd requests from a prior
    /// REPL turn / aborted run don't bleed in.
    pub fn reset() {
        with_state(|st| {
            st.pending_override = None;
            st.bound = None;
            st.pending_sticky_cwd = None;
        });
    }

    /// RF33-2: queue a sticky-cwd request. Called by `skill_fetch` when the
    /// fetched skill has `sticky_cwd: true` + a non-empty working_dirs list.
    /// The very next planner-loop iteration polls and applies it via
    /// `take_pending_sticky_cwd`.
    pub fn set_pending_sticky_cwd(path: PathBuf) {
        with_state(|st| st.pending_sticky_cwd = Some(path));
    }

    /// Consume the queued sticky-cwd request. Returns `None` if no skill
    /// requested one. Caller (`run_goal`) is responsible for applying it
    /// to `workspace.cwd` and (if live) `codex_session.client.set_cwd`.
    pub fn take_pending_sticky_cwd() -> Option<PathBuf> {
        with_state(|st| st.pending_sticky_cwd.take())
    }

    /// Inspect without consuming. For doctor / tests.
    pub fn peek_pending_sticky_cwd() -> Option<PathBuf> {
        read_state(|st| st.pending_sticky_cwd.clone())
    }

    /// Queue a one-shot override. The next RepoPrompt call that would
    /// otherwise default to `[ctx.cwd]` will use these dirs and then
    /// the override is consumed. Also clears the bound-window cache —
    /// we're about to switch dirs, so the cached window won't match.
    pub fn set_pending_override(working_dirs: Vec<PathBuf>) {
        with_state(|st| {
            st.pending_override = Some(working_dirs);
            st.bound = None;
        });
    }

    /// Atomically take and return the pending override, leaving the slot
    /// empty for the call after this one.
    pub fn take_pending_override() -> Option<Vec<PathBuf>> {
        with_state(|st| st.pending_override.take())
    }

    /// RF25-2: look up a previously-bound window_id matching `working_dirs`.
    /// Returns `None` if the cache is empty or the cached binding is for
    /// a different dir set.
    pub(crate) fn cached_window_id_for(working_dirs: &[PathBuf]) -> Option<u32> {
        read_state(|st| {
            st.bound
                .as_ref()
                .filter(|b| b.working_dirs == working_dirs)
                .map(|b| b.window_id)
        })
    }

    /// Record a successful bind so future calls with the same working_dirs
    /// can skip the bind_context CLI roundtrip.
    pub fn record_bound_window(working_dirs: Vec<PathBuf>, window_id: u32) {
        with_state(|st| {
            st.bound = Some(BoundWindow { working_dirs, window_id });
        });
    }

    /// Drop the cached bound window without touching pending_override.
    /// Used when /cd changes the workspace cwd — the cached window is for
    /// the old cwd.
    pub fn clear_bound_window() {
        with_state(|st| st.bound = None);
    }

    /// Inspect without consuming. Used by `/doctor` and by other crates'
    /// tests that want to prime/peek state without spinning up a real RP CLI.
    pub fn peek_pending_override() -> Option<Vec<PathBuf>> {
        read_state(|st| st.pending_override.clone())
    }

    /// As above, for the bound-window cache. Returns `(working_dirs, window_id)`
    /// or `None` if no bind has been recorded since the last invalidation.
    pub fn peek_bound_window() -> Option<(Vec<PathBuf>, u32)> {
        read_state(|st| {
            st.bound
                .as_ref()
                .map(|b| (b.working_dirs.clone(), b.window_id))
        })
    }
}
