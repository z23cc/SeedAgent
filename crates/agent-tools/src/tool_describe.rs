//! `tool_describe` tool: planner-callable lookup for a forgotten tool's
//! description.
//!
//! extracted from `lib.rs`. Pairs with 's "send full
//! descriptions on turns 1–4, names-only after that" prompt economy —
//! if the planner needs a forgotten description on a late turn, it
//! calls `tool_describe {name: "..."}` and gets the full text back.
//!
//! Reads from the process-cached `seed_registry()` so the
//! lookup is `O(n)` over ~31 tools without re-allocating the registry.

use agent_core::{Tool, ToolCall, ToolContext, ToolError, ToolResult};
use serde::Deserialize;
use serde_json::json;

use crate::seed_registry;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ToolDescribeArgs {
    name: String,
}

pub struct ToolDescribeTool;

impl Tool for ToolDescribeTool {
    fn name(&self) -> &'static str {
        "tool_describe"
    }

    crate::tool_description!("tool_describe");

    crate::impl_args_schema!(ToolDescribeArgs);

    crate::impl_pure_read!();

    fn execute(&self, _ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: ToolDescribeArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let registry = seed_registry();
        let info = registry
            .infos()
            .into_iter()
            .find(|info| info.name == args.name);
        match info {
            Some(info) => Ok(ToolResult::ok(
                call,
                json!({
                    "status": "success",
                    "name": info.name,
                    "description": info.description,
                }),
            )),
            None => Ok(ToolResult::error(
                call,
                format!("unknown tool: {}", args.name),
            )),
        }
    }
}
