//! Memory-index tools: `memory_search` and `memory_fetch`.
//!
//! extracted from `lib.rs`. The `memory_paths` helper that
//! both tools share moved with them since nothing else uses it (the
//! memory_protocol tools build paths directly from `ctx.memory_dir`).

use agent_core::{Tool, ToolCall, ToolContext, ToolError, ToolResult};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct MemorySearchArgs {
    // planners often use `q` for query and `top_k`/`n` for limit.
    #[serde(alias = "q", alias = "search", alias = "term", alias = "needle")]
    query: String,
    #[serde(default, alias = "top_k", alias = "n", alias = "max_results")]
    limit: Option<usize>,
}

pub struct MemorySearchTool;

impl Tool for MemorySearchTool {
    fn name(&self) -> &'static str {
        "memory_search"
    }

    crate::tool_description!("memory_search");

    crate::impl_args_schema!(MemorySearchArgs);

    crate::impl_pure_read!();

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: MemorySearchArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let paths = memory_paths(ctx);
        let index = agent_memory::rebuild_index(&paths)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let hits = agent_memory::search_index(&index, &args.query, args.limit.unwrap_or(10));
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "query": args.query,
                "index_path": paths.index_path(),
                "results": hits,
            }),
        ))
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct MemoryFetchArgs {
    // `name`/`key`/`path` synonyms — planners conflate them.
    #[serde(alias = "name", alias = "key", alias = "path", alias = "memory_id")]
    id: String,
    #[serde(default, alias = "max_size", alias = "byte_limit")]
    max_bytes: Option<usize>,
}

pub struct MemoryFetchTool;

impl Tool for MemoryFetchTool {
    fn name(&self) -> &'static str {
        "memory_fetch"
    }

    crate::tool_description!("memory_fetch");

    crate::impl_args_schema!(MemoryFetchArgs);

    crate::impl_pure_read!();

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: MemoryFetchArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let default_bytes = ctx.scaled_default(16_000, 4_000);
        let doc = agent_memory::fetch_memory(
            &memory_paths(ctx),
            &args.id,
            args.max_bytes.unwrap_or(default_bytes),
        )
        .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "entry": doc.entry,
                "body": doc.body,
            }),
        ))
    }
}

fn memory_paths(ctx: &ToolContext) -> agent_memory::MemoryPaths {
    agent_memory::MemoryPaths::new(
        ctx.memory_dir.clone(),
        ctx.skills_dir.clone(),
        ctx.sessions_dir.clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_search_args_accept_q_alias() {
        let v = serde_json::json!({"q": "foo", "top_k": 5});
        let args: MemorySearchArgs = serde_json::from_value(v).unwrap();
        assert_eq!(args.query, "foo");
        assert_eq!(args.limit, Some(5));
    }
}
