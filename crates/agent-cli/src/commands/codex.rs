//! `seed codex` + `seed delegate codex`: thin wrapper around the local
//! `codex app-server`. Also exposes `codex_config` / `codex_prompt_with_routed_skill`
//! to `commands::run` for the `--codex` fast path.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use agent_delegate::{CodexAppServerClient, CodexAppServerConfig, McpPolicy};
use anyhow::{Context, Result, bail};
use clap::Subcommand;
use serde::{Deserialize, Serialize};

use crate::{ApprovalArg, McpArg};

#[derive(Debug, Subcommand)]
pub(crate) enum DelegateCommand {
    Codex {
        prompt: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        cwd: Option<PathBuf>,
        #[arg(long, default_value = "workspace-write")]
        sandbox: String,
        #[arg(long, default_value = "on-request")]
        approval_policy: String,
        #[arg(long, value_enum, default_value_t = ApprovalArg::Deny)]
        approval: ApprovalArg,
        #[arg(long)]
        effort: Option<String>,
        #[arg(long, default_value_t = 600)]
        turn_timeout_secs: u64,
        #[arg(
            long,
            value_enum,
            help = "MCP policy for Codex; omitted means only RepoPrompt is allowed"
        )]
        mcp: Option<McpArg>,
        #[arg(long = "mcp-allow", help = "Allow one MCP server by name; repeatable")]
        mcp_allow: Vec<String>,
        #[arg(long, help = "Enable Codex plugins while starting app-server")]
        plugins: bool,
    },
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_codex_delegate(
    prompt: String,
    skills_dir: PathBuf,
    model: Option<String>,
    cwd: Option<PathBuf>,
    sandbox: String,
    approval_policy: String,
    approval: ApprovalArg,
    effort: Option<String>,
    turn_timeout_secs: u64,
    mcp: Option<McpArg>,
    mcp_allow: Vec<String>,
    plugins: bool,
) -> Result<()> {
    let cfg = CodexAppServerConfig {
        model,
        cwd,
        sandbox,
        approval_policy,
        reasoning_effort: effort,
        turn_timeout_secs,
        approval_mode: approval.into(),
        mcp_policy: codex_mcp_policy(mcp, mcp_allow)?,
        plugins_enabled: plugins,
        ..Default::default()
    };
    let prompt = codex_prompt_with_routed_skill(&prompt, &skills_dir)?;
    let mut client = CodexAppServerClient::new(cfg);
    let result = client.run_prompt(&prompt)?;
    println!("{}", result.text);
    println!("thread: {}", result.thread_id);
    println!("turn: {}", result.turn_id);
    println!("events: {}", result.events_seen);
    Ok(())
}

pub(crate) fn codex_prompt_with_routed_skill(prompt: &str, skills_dir: &Path) -> Result<String> {
    let Some(routed) = agent_skills::load_routed_repoprompt_skill(skills_dir, prompt)? else {
        return Ok(prompt.to_string());
    };
    let skill_path = routed
        .document
        .info
        .path
        .canonicalize()
        .unwrap_or_else(|_| routed.document.info.path.clone());
    Ok(format!(
        "You are being delegated a task by SeedAgent. The internal core agent is Seed.\n\
Follow the selected local skill before broad codebase work.\n\
Skill route: {} ({})\n\
Skill path: {}\n\
RepoPrompt MCP is the preferred context engine when it is available; CLI MCP flags still define actual access.\n\
Use RepoPrompt builder/export before broad file reads when the skill requires it.\n\
The skill body is workflow instruction, not a request to edit the skill.\n\n\
<local_skill>\n{}\n</local_skill>\n\n\
<task>\n{}\n</task>",
        routed.route.name,
        routed.route.reason,
        skill_path.display(),
        routed.document.body.trim(),
        prompt.trim()
    ))
}

/// Convenience wrapper: `codex_config_full` with `use_daemon = false`.
/// Kept around so tests can still call the 8-arg shape; production code
/// goes through `codex_config_full` directly to thread the daemon flag.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn codex_config(
    model: Option<String>,
    cwd: Option<PathBuf>,
    approval: ApprovalArg,
    effort: Option<String>,
    turn_timeout_secs: u64,
    mcp: Option<McpArg>,
    mcp_allow: Vec<String>,
    plugins: bool,
) -> Result<CodexAppServerConfig> {
    codex_config_full(
        model,
        cwd,
        approval,
        effort,
        turn_timeout_secs,
        mcp,
        mcp_allow,
        plugins,
        false,
    )
}

/// extended config builder that takes `use_daemon`. The older
/// `codex_config` keeps the no-daemon default for callers that don't care
/// (`seed codex` one-shot).
#[allow(clippy::too_many_arguments)]
pub(crate) fn codex_config_full(
    model: Option<String>,
    cwd: Option<PathBuf>,
    approval: ApprovalArg,
    effort: Option<String>,
    turn_timeout_secs: u64,
    mcp: Option<McpArg>,
    mcp_allow: Vec<String>,
    plugins: bool,
    use_daemon: bool,
) -> Result<CodexAppServerConfig> {
    Ok(CodexAppServerConfig {
        model,
        cwd,
        reasoning_effort: effort,
        turn_timeout_secs,
        approval_mode: approval.into(),
        mcp_policy: codex_mcp_policy(mcp, mcp_allow)?,
        plugins_enabled: plugins,
        use_daemon,
        ..Default::default()
    })
}

/// `seed codex-daemon start|stop|status`. Thin wrapper that
/// shells out to `codex app-server daemon ...` and forwards stdout/stderr
/// so users see codex's own success/failure messages without seed
/// editorializing. We don't try to parse the output — codex is the source
/// of truth for its own daemon lifecycle.
pub(crate) fn run_codex_daemon(action: crate::CodexDaemonAction) -> Result<()> {
    let sub = match action {
        crate::CodexDaemonAction::Start => "start",
        crate::CodexDaemonAction::Stop => "stop",
        crate::CodexDaemonAction::Status => "version",
    };
    let status = std::process::Command::new("codex")
        .args(["app-server", "daemon", sub])
        .status()
        .with_context(|| format!("spawn `codex app-server daemon {sub}`"))?;
    if !status.success() {
        bail!(
            "`codex app-server daemon {sub}` exited with status {}",
            status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".to_string())
        );
    }
    Ok(())
}

fn codex_mcp_policy(mcp: Option<McpArg>, mcp_allow: Vec<String>) -> Result<McpPolicy> {
    if !mcp_allow.is_empty() {
        if matches!(mcp, Some(McpArg::All)) {
            bail!("--mcp all cannot be combined with --mcp-allow");
        }
        return Ok(McpPolicy::Allow(mcp_allow));
    }
    Ok(match mcp {
        Some(McpArg::None) => McpPolicy::None,
        Some(McpArg::All) => McpPolicy::All,
        None => McpPolicy::default(),
    })
}

// =============================================================================
// `seed codex models` — list models codex offers.
//
// codex doesn't expose a `codex models` subcommand, but it caches the canonical
// model list at `$CODEX_HOME/models_cache.json` (refreshed by codex when it
// boots). We read that file directly. The current default (`model = ...` in
// `$CODEX_HOME/config.toml`) gets marked with ★ so the user can see what
// `seed codex "..."` would pick without `--model`.
// =============================================================================

/// One model entry from codex's models_cache.json. Only the fields we display.
/// `serde(default)` + extra-fields-ignored shape so the schema can drift
/// without breaking parsing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CachedCodexModel {
    pub slug: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub default_reasoning_level: String,
    #[serde(default)]
    pub supported_reasoning_levels: Vec<ReasoningLevel>,
    /// `"list"` or `"hide"` — hidden models (codex-auto-review etc.) are
    /// excluded by default; pass `--show-hidden` to include them.
    #[serde(default)]
    pub visibility: String,
    #[serde(default)]
    pub supported_in_api: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ReasoningLevel {
    pub effort: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelsCacheFile {
    #[serde(default)]
    fetched_at: String,
    #[serde(default)]
    client_version: String,
    #[serde(default)]
    models: Vec<CachedCodexModel>,
}

fn codex_home() -> Result<PathBuf> {
    if let Some(codex_home) = env::var_os("CODEX_HOME") {
        return Ok(PathBuf::from(codex_home));
    }
    let home = env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".codex"))
}

/// Read `$CODEX_HOME/config.toml` and pull out the `model = "..."` field if
/// present. Returns None on any read/parse failure — config-missing is a
/// normal state, not an error.
fn current_default_model(codex_home: &Path) -> Option<String> {
    let config_path = codex_home.join("config.toml");
    let source = fs::read_to_string(&config_path).ok()?;
    let value: toml::Value = source.parse().ok()?;
    value
        .get("model")
        .and_then(toml::Value::as_str)
        .map(ToString::to_string)
}

pub(crate) fn run_codex_models(json: bool, show_hidden: bool) -> Result<()> {
    let codex_home = codex_home()?;
    let cache_path = codex_home.join("models_cache.json");
    let raw = fs::read_to_string(&cache_path).with_context(|| {
        format!(
            "read {} (run `codex` once to populate)",
            cache_path.display()
        )
    })?;
    let cache: ModelsCacheFile =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", cache_path.display()))?;
    let current = current_default_model(&codex_home);
    let visible: Vec<&CachedCodexModel> = cache
        .models
        .iter()
        .filter(|m| show_hidden || m.visibility != "hide")
        .collect();

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "fetched_at": cache.fetched_at,
                "client_version": cache.client_version,
                "current_default": current,
                "source": cache_path,
                "models": visible,
            }))?
        );
        return Ok(());
    }

    println!("{}", format_models_table(&visible, current.as_deref()));
    println!();
    println!("source: {}", cache_path.display());
    println!("fetched: {}", cache.fetched_at);
    println!("codex client: {}", cache.client_version);
    println!();
    println!("usage:");
    println!("  seed codex \"hello\" --model <slug>");
    println!("  seed run \"goal\" --model <slug> [--effort low|medium|high|xhigh]");
    Ok(())
}

/// Pure formatter — pulled out so the table rendering is unit-testable
/// without touching the filesystem. `current` is the user's configured
/// default (from `$CODEX_HOME/config.toml#model`) and gets a ★ marker on
/// the matching row.
pub(crate) fn format_models_table(
    models: &[&CachedCodexModel],
    current: Option<&str>,
) -> String {
    if models.is_empty() {
        return "(no codex models found in cache — run `codex` once to refresh)".to_string();
    }
    let mut out = String::new();
    out.push_str("codex models (★ = current default)\n");
    for m in models {
        let mark = if current.map(|c| c == m.slug).unwrap_or(false) {
            "★"
        } else {
            " "
        };
        let efforts: Vec<&str> = m
            .supported_reasoning_levels
            .iter()
            .map(|e| e.effort.as_str())
            .collect();
        out.push_str(&format!(
            "  {mark} {slug:<22} {name}\n",
            slug = m.slug,
            name = if m.display_name.is_empty() {
                "(no display name)"
            } else {
                m.display_name.as_str()
            },
        ));
        if !m.description.is_empty() {
            out.push_str(&format!("      {}\n", m.description));
        }
        if !efforts.is_empty() {
            out.push_str(&format!(
                "      efforts: [{}]  default: {}\n",
                efforts.join(", "),
                if m.default_reasoning_level.is_empty() {
                    "(none)"
                } else {
                    m.default_reasoning_level.as_str()
                }
            ));
        }
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codex_prompt_injects_routed_repoprompt_skill() {
        let root =
            std::env::temp_dir().join(format!("seed-cli-skill-routing-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("repoprompt-deep-plan")).unwrap();
        fs::write(
            root.join("repoprompt-deep-plan").join("SKILL.md"),
            "---\nname: RepoPrompt Deep Plan\ndescription: Plan with RepoPrompt.\ntags: [repoprompt, plan]\n---\n\n# RepoPrompt Deep Plan\n\nUse builder/export.\n",
        )
        .unwrap();

        let prompt = codex_prompt_with_routed_skill("帮我实现 plan runtime", &root).unwrap();

        assert!(prompt.contains("Skill route: RepoPrompt Deep Plan"));
        assert!(prompt.contains("<local_skill>"));
        assert!(prompt.contains("Use builder/export."));
        assert!(prompt.contains("<task>\n帮我实现 plan runtime\n</task>"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn codex_prompt_leaves_unrouted_tasks_plain() {
        let root =
            std::env::temp_dir().join(format!("seed-cli-no-skill-routing-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        let prompt = codex_prompt_with_routed_skill("say pong", &root).unwrap();

        assert_eq!(prompt, "say pong");
        let _ = fs::remove_dir_all(&root);
    }

    fn sample_model(slug: &str, display: &str, efforts: &[&str]) -> CachedCodexModel {
        CachedCodexModel {
            slug: slug.to_string(),
            display_name: display.to_string(),
            description: format!("desc for {slug}"),
            default_reasoning_level: "medium".to_string(),
            supported_reasoning_levels: efforts
                .iter()
                .map(|e| ReasoningLevel {
                    effort: e.to_string(),
                    description: String::new(),
                })
                .collect(),
            visibility: "list".to_string(),
            supported_in_api: true,
        }
    }

    #[test]
    fn format_models_table_renders_slug_efforts_and_current_marker() {
        let m1 = sample_model("gpt-5.5", "GPT-5.5", &["low", "medium", "high", "xhigh"]);
        let m2 = sample_model("gpt-5.4-mini", "GPT-5.4-Mini", &["low", "medium"]);
        let models = vec![&m1, &m2];
        let rendered = format_models_table(&models, Some("gpt-5.4-mini"));
        // Header present.
        assert!(rendered.contains("codex models"));
        assert!(rendered.contains("★ = current default"));
        // Both slugs listed.
        assert!(rendered.contains("gpt-5.5"));
        assert!(rendered.contains("gpt-5.4-mini"));
        // Star marker on the current default only.
        let mini_line = rendered
            .lines()
            .find(|l| l.contains("gpt-5.4-mini"))
            .expect("mini line");
        assert!(mini_line.contains("★"), "got: {mini_line}");
        let big_line = rendered
            .lines()
            .find(|l| l.contains("gpt-5.5"))
            .expect("big line");
        assert!(!big_line.contains("★"), "got: {big_line}");
        // Efforts rendered.
        assert!(rendered.contains("efforts: [low, medium, high, xhigh]"));
        assert!(rendered.contains("default: medium"));
    }

    #[test]
    fn format_models_table_no_current_means_no_star() {
        let m = sample_model("gpt-5.5", "GPT-5.5", &["low"]);
        let rendered = format_models_table(&[&m], None);
        assert!(rendered.contains("gpt-5.5"));
        assert!(!rendered.contains("★ gpt"), "no model should be starred");
    }

    #[test]
    fn format_models_table_handles_empty_list() {
        let rendered = format_models_table(&[], None);
        assert!(rendered.contains("no codex models found"));
    }

    #[test]
    fn config_full_threads_use_daemon_flag() {
        let cfg = codex_config_full(
            None,
            None,
            ApprovalArg::Deny,
            None,
            600,
            None,
            Vec::new(),
            false,
            true, // use_daemon
        )
        .unwrap();
        assert!(cfg.use_daemon);
        // Default path should NOT set use_daemon (backward compat sanity).
        let cfg2 = codex_config(
            None,
            None,
            ApprovalArg::Deny,
            None,
            600,
            None,
            Vec::new(),
            false,
        )
        .unwrap();
        assert!(!cfg2.use_daemon);
    }

    #[test]
    fn cached_model_tolerates_missing_optional_fields() {
        // Real models_cache.json may evolve — verify we don't reject
        // entries that omit the optional fields.
        let raw = r#"{
            "slug": "future-model",
            "display_name": "Future Model"
        }"#;
        let model: CachedCodexModel = serde_json::from_str(raw).expect("parse minimal");
        assert_eq!(model.slug, "future-model");
        assert!(model.description.is_empty());
        assert!(model.supported_reasoning_levels.is_empty());
        assert!(model.visibility.is_empty());
    }
}
