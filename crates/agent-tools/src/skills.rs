//! Skill-discovery tools: `skill_list`, `skill_search`, `skill_fetch`.
//!
//! extracted from `lib.rs`. The region uses only
//! `parse_tool_args` plus the `skill_tools_guard` / `repoprompt_sync`
//! re-exports from the `sync` module — no shared internal helpers
//! beyond that, so the move is purely mechanical.

use agent_core::{Tool, ToolCall, ToolContext, ToolError, ToolResult};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{parse_tool_args, repoprompt_sync, skill_tools_guard};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SkillListArgs {
    limit: Option<usize>,
}

pub struct SkillListTool;

impl Tool for SkillListTool {
    fn name(&self) -> &'static str {
        "skill_list"
    }

    crate::tool_description!("skill_list");

    crate::impl_args_schema!(SkillListArgs);

    crate::impl_pure_read!();

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: SkillListArgs =
            serde_json::from_value(call.args.clone()).unwrap_or(SkillListArgs { limit: None });
        let limit = args.limit.unwrap_or(50).clamp(1, 200);
        let skills = agent_skills::list_skill_infos(&ctx.skills_dir)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "skills_dir": ctx.skills_dir.display().to_string(),
                "skills": skills.into_iter().take(limit).collect::<Vec<_>>(),
            }),
        ))
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SkillSearchArgs {
    #[serde(alias = "q", alias = "search", alias = "term", alias = "needle")]
    query: String,
    #[serde(default, alias = "top_k", alias = "n", alias = "max_results")]
    limit: Option<usize>,
}

pub struct SkillSearchTool;

impl Tool for SkillSearchTool {
    fn name(&self) -> &'static str {
        "skill_search"
    }

    crate::tool_description!("skill_search");

    crate::impl_args_schema!(SkillSearchArgs);

    crate::impl_pure_read!();

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: SkillSearchArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let skills = agent_skills::search_skill_infos(
            &ctx.skills_dir,
            &args.query,
            args.limit.unwrap_or(10),
        )
        .map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "query": args.query,
                "skills": skills,
            }),
        ))
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SkillFetchArgs {
    name: String,
}

pub struct SkillFetchTool;

impl Tool for SkillFetchTool {
    fn name(&self) -> &'static str {
        "skill_fetch"
    }

    crate::tool_description!("skill_fetch");

    crate::impl_args_schema!(SkillFetchArgs);

    crate::impl_pure_read!();

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: SkillFetchArgs = parse_tool_args(call)?;

        let document = agent_skills::fetch_skill(&ctx.skills_dir, &args.name)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let auto_bind = document
            .info
            .repoprompt
            .as_ref()
            .map(queue_skill_repoprompt_binding);
        // if the skill declares `allowed-tools:`, narrow the
        // planner's tool catalog for subsequent turns. Empty list = no
        // restriction (the default). Last-skill-wins semantics — fetching
        // a different skill replaces (doesn't intersect).
        let narrowed_tools = if document.info.allowed_tools.is_empty() {
            skill_tools_guard::reset();
            None
        } else {
            skill_tools_guard::set(document.info.allowed_tools.clone());
            Some(document.info.allowed_tools.clone())
        };
        let mut content = json!({
            "status": "success",
            "skill": document.info,
            "body": document.body,
        });
        if let Some(outcome) = auto_bind {
            content["repoprompt_autobind"] = outcome;
        }
        if let Some(tools) = narrowed_tools {
            content["narrowed_tools"] = json!({
                "active": true,
                "applies_to": "subsequent planner turns until /new or skill_fetch of another skill",
                "tools": tools,
            });
        }
        Ok(ToolResult::ok(call, content))
    }
}

/// Skills with `repoprompt_*` frontmatter no longer call
/// `bind_context` eagerly. We just queue the binding as a one-shot override
/// in `repoprompt_sync::set_pending_override` — the very next rp tool call
/// consumes it via `default_repoprompt_working_dirs`. After that, RP binds
/// fall back to `ctx.cwd` (mutated by the user via `/cd`), so skill
/// suggestions remain transient and the user's workspace stays primary.
///
/// `context_id` is not queue-able the same way (it's a stable RP context
/// reference, not a per-cwd thing), so we just surface it in the result
/// for the planner to use explicitly if it wants.
pub(crate) fn queue_skill_repoprompt_binding(
    binding: &agent_skills::RepoPromptBinding,
) -> Value {
    if binding.working_dirs.is_empty() {
        return json!({
            "status": "noop",
            "reason": "skill has repoprompt frontmatter but no working_dirs",
            "context_id": binding.context_id,
        });
    }
    repoprompt_sync::set_pending_override(binding.working_dirs.clone());
    // opt-in sticky_cwd. When the skill frontmatter requests it,
    // also queue a workspace.cwd change that run_goal will poll between
    // turns. We use working_dirs[0] (canonical first entry) as the cwd
    // target — for multi-root skills the user should pick a primary in
    // the skill design rather than relying on heuristics here.
    let sticky_target = if binding.sticky_cwd {
        let target = binding.working_dirs[0].clone();
        repoprompt_sync::set_pending_sticky_cwd(target.clone());
        Some(target)
    } else {
        None
    };
    json!({
        "status": if binding.sticky_cwd { "queued_sticky" } else { "queued_transient" },
        "applies_to": if binding.sticky_cwd {
            "next rp call + workspace.cwd after current turn"
        } else {
            "next repoprompt_* tool call only"
        },
        "working_dirs": binding.working_dirs,
        "context_id": binding.context_id,
        "sticky_cwd_target": sticky_target,
    })
}
