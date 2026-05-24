//! REPL-lifetime cache for a [`CodexAppServerClient`].
//!
//! Every `run_goal` previously spawned a fresh `codex app-server` subprocess
//! (~300ms + `initialize` handshake). With a long REPL session that adds up
//! to tens of seconds of pure startup. The Codex protocol allows mutating
//! per-turn config (cwd / model / effort / sandbox / approval) without
//! restart, so as long as `plugins_enabled` and `mcp_policy` and the
//! `command`/`args`/`experimental_api` triple don't change, we can keep the
//! same subprocess alive across calls.
//!
//! That decision lives in `CodexLaunchFingerprint` (agent-delegate); this
//! file just owns the `Option<CodexAppServerClient>` and the cached
//! fingerprint, and does the "restart vs hot-swap" routing.

use agent_delegate::{CodexAppServerClient, CodexAppServerConfig, CodexLaunchFingerprint};
use anyhow::Result;

#[derive(Default)]
pub(crate) struct CodexSession {
    inner: Option<CodexAppServerClient>,
    fingerprint: Option<CodexLaunchFingerprint>,
}

impl CodexSession {
    /// Return a `&mut` to a client whose config matches `cfg`. If we have a
    /// live client with a matching launch fingerprint, hot-swap the
    /// per-turn fields and reuse. Otherwise drop the old client (its `Drop`
    /// kills the subprocess) and start a fresh one.
    ///
    /// Note: we do not call `ensure_ready` here — `run_prompt[_streaming]`
    /// already does that lazily, and lifting it would force the caller to
    /// handle errors before they have a session writer ready.
    pub(crate) fn ensure(&mut self, cfg: CodexAppServerConfig) -> Result<&mut CodexAppServerClient> {
        let new_fp = CodexLaunchFingerprint::from(&cfg);
        let need_restart = self.fingerprint.as_ref() != Some(&new_fp);
        if need_restart {
            self.shutdown();
            self.inner = Some(CodexAppServerClient::new(cfg));
            self.fingerprint = Some(new_fp);
            return Ok(self.inner.as_mut().expect("just inserted"));
        }
        // Reuse path: hot-swap fields known to be per-turn-mutable.
        // We deliberately do NOT replace the entire cfg, because some fields
        // (client_name/title/version, request_timeout_secs, turn_timeout_secs)
        // are only meaningful at launch time and silently changing them
        // would mislead future maintainers reading the cached config.
        let client = self.inner.as_mut().expect("fingerprint match implies live client");
        client.set_cwd_opt(cfg.cwd);
        client.set_model(cfg.model);
        client.set_effort(cfg.reasoning_effort);
        client.set_sandbox(cfg.sandbox);
        client.set_approval_policy(cfg.approval_policy);
        client.set_approval_mode(cfg.approval_mode);
        Ok(client)
    }

    /// Tear down the cached client (its `Drop` impl kills the subprocess
    /// and joins the reader thread). Called by `/new` so a fresh prompt
    /// starts with a clean Codex slate, and at REPL exit to be tidy.
    pub(crate) fn shutdown(&mut self) {
        self.inner = None;
        self.fingerprint = None;
    }

    /// True iff a client is currently held (subprocess may be alive — we
    /// don't probe it). Used by `/doctor` to report session state.
    pub(crate) fn is_live(&self) -> bool {
        self.inner.is_some()
    }

    /// `cfg.cwd` of the currently-held client, if any. Used by `/doctor`'s
    /// cwd health check to compare against `workspace.cwd`.
    pub(crate) fn client_cwd(&self) -> Option<std::path::PathBuf> {
        self.inner.as_ref().and_then(|c| c.cwd().cloned())
    }

    /// Mutable accessor for the inner client. Returns `None` if no client
    /// is held. Currently only used by `/sync` to push a new cwd into the
    /// live client without going through `ensure()` (which would have to
    /// re-check the fingerprint).
    pub(crate) fn client_mut(&mut self) -> Option<&mut CodexAppServerClient> {
        self.inner.as_mut()
    }

    #[cfg(test)]
    pub(crate) fn fingerprint(&self) -> Option<&CodexLaunchFingerprint> {
        self.fingerprint.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_delegate::{ApprovalMode, McpPolicy};
    use std::path::PathBuf;

    fn base_cfg() -> CodexAppServerConfig {
        CodexAppServerConfig::default()
    }

    #[test]
    fn ensure_starts_a_client_on_first_call() {
        // We can't actually spawn `codex app-server` in tests (would need
        // the binary on PATH and an actual launch). But `ensure` only calls
        // `CodexAppServerClient::new`, which is pure struct construction —
        // the subprocess starts lazily inside `run_prompt`. So this test
        // verifies the bookkeeping, not the spawn.
        let mut session = CodexSession::default();
        assert!(!session.is_live());
        let _ = session.ensure(base_cfg()).unwrap();
        assert!(session.is_live());
        assert!(session.fingerprint().is_some());
    }

    #[test]
    fn ensure_reuses_when_fingerprint_matches() {
        let mut session = CodexSession::default();
        let _ = session.ensure(base_cfg()).unwrap();
        let fp_before = session.fingerprint().cloned().unwrap();

        // Change only per-turn fields: cwd, model, effort, sandbox, approval.
        let mut next = base_cfg();
        next.cwd = Some(PathBuf::from("/different"));
        next.model = Some("new-model".to_string());
        next.reasoning_effort = Some("high".to_string());
        next.sandbox = "read-only".to_string();
        next.approval_policy = "never".to_string();
        next.approval_mode = ApprovalMode::AcceptForSession;

        let _ = session.ensure(next).unwrap();
        let fp_after = session.fingerprint().cloned().unwrap();
        assert_eq!(fp_before, fp_after, "fingerprint must be unchanged");
        // And the cwd hot-swap actually landed.
        assert_eq!(session.client_cwd(), Some(PathBuf::from("/different")));
    }

    #[test]
    fn ensure_restarts_when_plugins_change() {
        let mut session = CodexSession::default();
        let _ = session.ensure(base_cfg()).unwrap();
        let fp_before = session.fingerprint().cloned().unwrap();
        let mut next = base_cfg();
        next.plugins_enabled = true;
        let _ = session.ensure(next).unwrap();
        let fp_after = session.fingerprint().cloned().unwrap();
        assert_ne!(fp_before, fp_after, "plugins_enabled change must restart");
        assert_eq!(fp_after.plugins_enabled, true);
    }

    #[test]
    fn ensure_restarts_when_mcp_policy_changes() {
        let mut session = CodexSession::default();
        let _ = session.ensure(base_cfg()).unwrap();
        let fp_before = session.fingerprint().cloned().unwrap();
        let mut next = base_cfg();
        next.mcp_policy = McpPolicy::All;
        let _ = session.ensure(next).unwrap();
        let fp_after = session.fingerprint().cloned().unwrap();
        assert_ne!(fp_before, fp_after, "mcp_policy change must restart");
        assert_eq!(fp_after.mcp_policy, McpPolicy::All);
    }

    #[test]
    fn shutdown_clears_state() {
        let mut session = CodexSession::default();
        let _ = session.ensure(base_cfg()).unwrap();
        assert!(session.is_live());
        session.shutdown();
        assert!(!session.is_live());
        assert!(session.fingerprint().is_none());
    }

    #[test]
    fn ensure_after_shutdown_starts_fresh() {
        let mut session = CodexSession::default();
        let _ = session.ensure(base_cfg()).unwrap();
        session.shutdown();
        let _ = session.ensure(base_cfg()).unwrap();
        assert!(session.is_live());
    }
}
