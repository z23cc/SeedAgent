//! RepoPrompt-bridge tools: `repoprompt_tools`, `repoprompt_exec`,
//! `repoprompt_call`.
//!
//! extracted from `lib.rs`. The three tools share helpers
//! that stay in `lib.rs` as `pub(crate)` since the plan tools also
//! consume them (`repoprompt_client`, `repoprompt_output_json`,
//! `attach_repoprompt_protocol_hint`, `repoprompt_ledger_prompt_for_*`,
//! `default_cwd_for_repoprompt_*`). The shared `RepoPromptRoutingArgs`
//! struct moved here and is re-exported from `lib.rs` so existing
//! call sites in `plan.rs` keep compiling.

use std::path::PathBuf;

use agent_core::{Tool, ToolCall, ToolContext, ToolError, ToolResult};
use serde::Deserialize;
use serde_json::json;

use crate::{
    attach_repoprompt_protocol_hint, default_cwd_for_repoprompt_exec,
    default_cwd_for_repoprompt_tool, repoprompt_client, repoprompt_ledger_prompt_for_exec,
    repoprompt_ledger_prompt_for_tool, repoprompt_output_json,
};

pub struct RepoPromptToolsTool;

impl Tool for RepoPromptToolsTool {
    fn name(&self) -> &'static str {
        "repoprompt_tools"
    }

    crate::tool_description!("repoprompt_tools");

    crate::impl_pure_read!();

    fn execute(&self, _ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "tools": agent_repoprompt::known_tools(),
            }),
        ))
    }
}

#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub(crate) struct RepoPromptRoutingArgs {
    pub(crate) cli_path: Option<PathBuf>,
    pub(crate) timeout_secs: Option<u64>,
    pub(crate) window_id: Option<u32>,
    pub(crate) tab: Option<String>,
    pub(crate) context_id: Option<String>,
    pub(crate) working_dirs: Option<Vec<PathBuf>>,
    pub(crate) raw_json: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RepoPromptExecArgs {
    #[serde(alias = "cmd")]
    command: String,
    #[serde(flatten)]
    routing: RepoPromptRoutingArgs,
}

pub struct RepoPromptExecTool;

impl Tool for RepoPromptExecTool {
    fn name(&self) -> &'static str {
        "repoprompt_exec"
    }

    crate::tool_description!("repoprompt_exec");

    crate::impl_args_schema!(RepoPromptExecArgs);

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: RepoPromptExecArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let default_cwd = default_cwd_for_repoprompt_exec(&args.command);
        let client = repoprompt_client(ctx, args.routing, default_cwd)?;
        let output = client
            .exec(&args.command)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let mut content = repoprompt_output_json(output);
        attach_repoprompt_protocol_hint(
            &mut content,
            repoprompt_ledger_prompt_for_exec(&args.command),
            client.config(),
        );
        Ok(ToolResult::ok(call, content))
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RepoPromptCallArgs {
    // The planner frequently bleeds its own `PlannedAction.tool_name` field
    // into the inner envelope — accept both spellings so we don't waste a
    // retry turn on what is just a naming conflict.
    #[serde(alias = "tool_name", alias = "name")]
    tool: String,
    #[serde(default, alias = "args_json", alias = "params")]
    args: serde_json::Value,
    #[serde(flatten)]
    routing: RepoPromptRoutingArgs,
}

pub struct RepoPromptCallTool;

impl Tool for RepoPromptCallTool {
    fn name(&self) -> &'static str {
        "repoprompt_call"
    }

    crate::tool_description!("repoprompt_call");

    crate::impl_args_schema!(RepoPromptCallArgs);

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: RepoPromptCallArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let tool = args
            .tool
            .parse::<agent_repoprompt::RepoPromptTool>()
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let mut routing = args.routing;
        routing.raw_json = Some(routing.raw_json.unwrap_or(true));
        let client = repoprompt_client(ctx, routing, default_cwd_for_repoprompt_tool(tool))?;
        let output = client
            .call_tool(tool, &args.args)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let mut content = repoprompt_output_json(output);
        attach_repoprompt_protocol_hint(
            &mut content,
            repoprompt_ledger_prompt_for_tool(tool),
            client.config(),
        );
        Ok(ToolResult::ok(call, content))
    }
}

// Typed wrappers around three frequently-used RP tools. The planner CAN
// reach them via `repoprompt_call`, but a named wrapper makes the choice
// obvious in the catalog and avoids string-typed `tool: "..."` args.

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RepoPromptCodemapArgs {
    /// Files or directories to render code structure for. Empty = whole workspace.
    #[serde(default, alias = "path", alias = "paths")]
    targets: Vec<String>,
    /// Optional language filter (e.g. "rust", "python").
    #[serde(default)]
    language: Option<String>,
    #[serde(flatten)]
    routing: RepoPromptRoutingArgs,
}

pub struct RepoPromptCodemapTool;

impl Tool for RepoPromptCodemapTool {
    fn name(&self) -> &'static str {
        "repoprompt_codemap"
    }

    crate::tool_description!("repoprompt_codemap");

    crate::impl_args_schema!(RepoPromptCodemapArgs);

    crate::impl_pure_read!();

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: RepoPromptCodemapArgs = crate::parse_tool_args(call)?;
        let mut payload = serde_json::Map::new();
        if !args.targets.is_empty() {
            payload.insert("paths".to_string(), json!(args.targets));
        }
        if let Some(lang) = args.language {
            payload.insert("language".to_string(), json!(lang));
        }
        let mut routing = args.routing;
        routing.raw_json = Some(routing.raw_json.unwrap_or(true));
        let client = repoprompt_client(
            ctx,
            routing,
            default_cwd_for_repoprompt_tool(agent_repoprompt::RepoPromptTool::GetCodeStructure),
        )?;
        let output = client
            .call_tool(
                agent_repoprompt::RepoPromptTool::GetCodeStructure,
                &serde_json::Value::Object(payload),
            )
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let mut content = repoprompt_output_json(output);
        attach_repoprompt_protocol_hint(
            &mut content,
            repoprompt_ledger_prompt_for_tool(agent_repoprompt::RepoPromptTool::GetCodeStructure),
            client.config(),
        );
        Ok(ToolResult::ok(call, content))
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RepoPromptFileSearchArgs {
    /// Pattern to search. Can be a path glob or content regex depending on `mode`.
    #[serde(alias = "query", alias = "q")]
    pattern: String,
    /// `path` (default) matches file paths; `content` matches file contents.
    #[serde(default)]
    mode: Option<String>,
    /// Lines of surrounding context for content matches.
    #[serde(default, alias = "context")]
    context_lines: Option<u32>,
    /// Optional file-glob filter (e.g. `*.rs`).
    #[serde(default, alias = "glob")]
    include: Option<String>,
    #[serde(flatten)]
    routing: RepoPromptRoutingArgs,
}

pub struct RepoPromptFileSearchTool;

impl Tool for RepoPromptFileSearchTool {
    fn name(&self) -> &'static str {
        "repoprompt_file_search"
    }

    crate::tool_description!("repoprompt_file_search");

    crate::impl_args_schema!(RepoPromptFileSearchArgs);

    crate::impl_pure_read!();

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: RepoPromptFileSearchArgs = crate::parse_tool_args(call)?;
        let mut payload = serde_json::Map::new();
        payload.insert("pattern".to_string(), json!(args.pattern));
        if let Some(mode) = args.mode {
            payload.insert("mode".to_string(), json!(mode));
        }
        if let Some(ctx_lines) = args.context_lines {
            payload.insert("context_lines".to_string(), json!(ctx_lines));
        }
        if let Some(include) = args.include {
            payload.insert("include".to_string(), json!(include));
        }
        let mut routing = args.routing;
        routing.raw_json = Some(routing.raw_json.unwrap_or(true));
        let client = repoprompt_client(
            ctx,
            routing,
            default_cwd_for_repoprompt_tool(agent_repoprompt::RepoPromptTool::FileSearch),
        )?;
        let output = client
            .call_tool(
                agent_repoprompt::RepoPromptTool::FileSearch,
                &serde_json::Value::Object(payload),
            )
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let mut content = repoprompt_output_json(output);
        attach_repoprompt_protocol_hint(
            &mut content,
            repoprompt_ledger_prompt_for_tool(agent_repoprompt::RepoPromptTool::FileSearch),
            client.config(),
        );
        Ok(ToolResult::ok(call, content))
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RepoPromptGitArgs {
    /// Git op: `status` | `diff` | `log` | `show` | `blame`.
    #[serde(alias = "cmd", alias = "command")]
    op: String,
    /// Path argument when relevant (e.g. for `diff`/`show`/`blame`).
    #[serde(default)]
    path: Option<String>,
    /// Revision/commit argument when relevant (e.g. for `show`).
    #[serde(default, alias = "rev", alias = "commit")]
    revision: Option<String>,
    /// Cap on lines returned (RP applies its own default if omitted).
    #[serde(default)]
    max_lines: Option<u32>,
    #[serde(flatten)]
    routing: RepoPromptRoutingArgs,
}

pub struct RepoPromptGitTool;

impl Tool for RepoPromptGitTool {
    fn name(&self) -> &'static str {
        "repoprompt_git"
    }

    crate::tool_description!("repoprompt_git");

    crate::impl_args_schema!(RepoPromptGitArgs);

    crate::impl_pure_read!();

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: RepoPromptGitArgs = crate::parse_tool_args(call)?;
        let mut payload = serde_json::Map::new();
        payload.insert("op".to_string(), json!(args.op));
        if let Some(path) = args.path {
            payload.insert("path".to_string(), json!(path));
        }
        if let Some(rev) = args.revision {
            payload.insert("revision".to_string(), json!(rev));
        }
        if let Some(max) = args.max_lines {
            payload.insert("max_lines".to_string(), json!(max));
        }
        let mut routing = args.routing;
        routing.raw_json = Some(routing.raw_json.unwrap_or(true));
        let client = repoprompt_client(
            ctx,
            routing,
            default_cwd_for_repoprompt_tool(agent_repoprompt::RepoPromptTool::Git),
        )?;
        let output = client
            .call_tool(
                agent_repoprompt::RepoPromptTool::Git,
                &serde_json::Value::Object(payload),
            )
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let mut content = repoprompt_output_json(output);
        attach_repoprompt_protocol_hint(
            &mut content,
            repoprompt_ledger_prompt_for_tool(agent_repoprompt::RepoPromptTool::Git),
            client.config(),
        );
        Ok(ToolResult::ok(call, content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn repoprompt_exec_accepts_cmd_alias() {
        let args: RepoPromptExecArgs = serde_json::from_value(json!({
            "cmd": "tree --mode folders"
        }))
        .unwrap();

        assert_eq!(args.command, "tree --mode folders");
    }

    #[test]
    fn codemap_args_accept_path_alias() {
        let args: RepoPromptCodemapArgs = serde_json::from_value(json!({
            "path": ["crates/agent-tools/src/lib.rs"]
        }))
        .unwrap();
        assert_eq!(args.targets, vec!["crates/agent-tools/src/lib.rs"]);
    }

    #[test]
    fn file_search_args_accept_q_alias() {
        let args: RepoPromptFileSearchArgs = serde_json::from_value(json!({
            "q": "TODO",
            "glob": "*.rs"
        }))
        .unwrap();
        assert_eq!(args.pattern, "TODO");
        assert_eq!(args.include.as_deref(), Some("*.rs"));
    }

    #[test]
    fn git_args_accept_cmd_alias() {
        let args: RepoPromptGitArgs = serde_json::from_value(json!({
            "cmd": "status"
        }))
        .unwrap();
        assert_eq!(args.op, "status");
    }
}
