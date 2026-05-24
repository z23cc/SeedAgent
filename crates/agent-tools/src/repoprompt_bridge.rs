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
}
