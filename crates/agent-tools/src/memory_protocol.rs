//! Memory-protocol tools: `update_working_checkpoint`,
//! `start_long_term_update`, `complete_long_term_update`.
//!
//! extracted from `lib.rs`. This region is genuinely
//! self-contained — the three tools share no internal helpers with the
//! rest of the file beyond `parse_tool_args`, so the move is purely
//! mechanical. The bundled SOP text (`DEFAULT_MEMORY_SOP`) is the
//! fallback the planner sees when `memory/memory_management_sop.md`
//! doesn't exist on disk.

use std::fs;

use agent_core::{Tool, ToolCall, ToolContext, ToolError, ToolResult};
use serde::Deserialize;
use serde_json::json;

use crate::parse_tool_args;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct CheckpointArgs {
    key_info: String,
    related_skill: Option<String>,
}

pub struct WorkingCheckpointTool;

impl Tool for WorkingCheckpointTool {
    fn name(&self) -> &'static str {
        "update_working_checkpoint"
    }

    crate::tool_description!("update_working_checkpoint");

    crate::impl_args_schema!(CheckpointArgs);

    fn execute(&self, _ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        checkpoint_result(call)
    }
}

fn checkpoint_result(call: &ToolCall) -> Result<ToolResult, ToolError> {
    let args: CheckpointArgs = parse_tool_args(call)?;

    Ok(ToolResult::ok(
        call,
        json!({
            "status": "success",
            "key_info": args.key_info,
            "related_skill": args.related_skill,
        }),
    ))
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LongTermUpdateArgs {
    reason: String,
    evidence: Option<String>,
}

pub struct LongTermUpdateTool;

impl Tool for LongTermUpdateTool {
    fn name(&self) -> &'static str {
        "start_long_term_update"
    }

    crate::tool_description!("start_long_term_update");

    crate::impl_args_schema!(LongTermUpdateArgs);

    fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: LongTermUpdateArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        let sop_path = ctx.memory_dir.join("memory_management_sop.md");
        let sop = fs::read_to_string(&sop_path).unwrap_or_else(|_| DEFAULT_MEMORY_SOP.to_string());
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "phase": "long_term_memory_settlement",
                "reason": args.reason,
                "evidence": args.evidence,
                "sop_path": sop_path,
                "sop": sop,
                "next_prompt": "LONG_TERM_MEMORY_SETTLEMENT: choose exactly one branch now. Branch A update_l2_global_facts: use only for stable environment facts, durable user preferences, paths, or configuration; first read/fetch memory/global_facts.md, then make the smallest patch/write. Branch B update_l3_skill: use for reusable workflows or troubleshooting patterns; first memory_search for an existing skill, then memory_fetch/read the selected existing SKILL.md, then patch that existing SKILL.md when appropriate; do not create duplicate skills. Branch C skip: if evidence is unverified, temporary, generic, duplicate, or not future-useful. After the write or skip decision, call complete_long_term_update with decision, target, reason, evidence, and changed. Never write secrets.",
                "settlement_branches": [
                    {
                        "decision": "update_l2_global_facts",
                        "when": "The evidence is a stable environment fact, path, configuration, or durable user preference.",
                        "first_step": "memory_fetch id=global-facts or read_file path=memory/global_facts.md",
                        "write_step": "patch_file/write_file with the smallest local update",
                        "complete_step": "complete_long_term_update decision=update_l2_global_facts changed=true"
                    },
                    {
                        "decision": "update_l3_skill",
                        "when": "The evidence is a reusable workflow, prerequisite, failure mode, or troubleshooting pattern.",
                        "first_step": "memory_search for an existing skill, then memory_fetch/read the selected SKILL.md",
                        "write_step": "patch_file the existing skill; avoid duplicate skills",
                        "complete_step": "complete_long_term_update decision=update_l3_skill changed=true"
                    },
                    {
                        "decision": "skip",
                        "when": "The evidence is unverified, temporary, generic, duplicated, secret, or not likely future-useful.",
                        "complete_step": "complete_long_term_update decision=skip changed=false",
                        "finish_step": "finish with 'Skipped long-term memory update: <reason>'"
                    }
                ],
            }),
        ))
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct CompleteLongTermUpdateArgs {
    decision: String,
    target: Option<String>,
    reason: String,
    evidence: Option<String>,
    changed: bool,
}

pub struct CompleteLongTermUpdateTool;

impl Tool for CompleteLongTermUpdateTool {
    fn name(&self) -> &'static str {
        "complete_long_term_update"
    }

    crate::tool_description!("complete_long_term_update");

    crate::impl_args_schema!(CompleteLongTermUpdateArgs);

    fn execute(&self, _ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: CompleteLongTermUpdateArgs =
            serde_json::from_value(call.args.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: call.name.clone(),
                    source,
                }
            })?;
        if !matches!(
            args.decision.as_str(),
            "update_l2_global_facts" | "update_l3_skill" | "skip"
        ) {
            return Ok(ToolResult::error(
                call,
                "decision must be update_l2_global_facts, update_l3_skill, or skip",
            ));
        }
        if args.decision != "skip" && args.target.as_deref().unwrap_or_default().is_empty() {
            return Ok(ToolResult::error(
                call,
                "target is required when decision changes memory",
            ));
        }
        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "decision": args.decision,
                "target": args.target,
                "reason": args.reason,
                "evidence": args.evidence,
                "changed": args.changed,
            }),
        ))
    }
}

const DEFAULT_MEMORY_SOP: &str = r#"# Memory Management SOP

Only write long-term memory when the fact is verified by a successful tool call and likely useful in future tasks.

- L2 global facts: stable paths, config, environment constraints, durable user preferences.
- L3 skills: reusable workflows, exact prerequisites, common failure modes, verification commands.
- L4 session archive: completed traces kept as evidence; crystallize with `run --learn`.

Rules:
1. Read the current target first.
2. Make the smallest local update.
3. Skip memory writes for guesses, temporary variables, generic advice, or one-off outputs.
4. Prefer updating a skill over creating a duplicate when the behavior is the same.
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_term_update_returns_sop_without_writing_memory() {
        let call = ToolCall::new(
            "start_long_term_update",
            json!({
                "reason": "remember verified shell workflow",
                "evidence": "run_shell exited 0"
            }),
        );
        let ctx = ToolContext::with_cwd(".");
        let result = LongTermUpdateTool.execute(&ctx, &call).unwrap();

        assert!(result.ok);
        assert_eq!(result.content["status"], "success");
        assert_eq!(result.content["phase"], "long_term_memory_settlement");
        assert!(result.content["sop"].as_str().unwrap().contains("verified"));
        assert!(result.content["settlement_branches"].is_array());
        assert!(
            result.content["next_prompt"]
                .as_str()
                .unwrap()
                .contains("complete_long_term_update")
        );
    }

    #[test]
    fn complete_long_term_update_requires_valid_decision_and_target() {
        let valid = ToolCall::new(
            "complete_long_term_update",
            json!({
                "decision": "update_l3_skill",
                "target": "skills/demo/SKILL.md",
                "reason": "captured reusable workflow",
                "evidence": "verified by test",
                "changed": true
            }),
        );
        let result = CompleteLongTermUpdateTool
            .execute(&ToolContext::with_cwd("."), &valid)
            .unwrap();
        assert!(result.ok);
        assert_eq!(result.content["decision"], "update_l3_skill");

        let missing_target = ToolCall::new(
            "complete_long_term_update",
            json!({
                "decision": "update_l2_global_facts",
                "reason": "needs a target",
                "changed": true
            }),
        );
        let result = CompleteLongTermUpdateTool
            .execute(&ToolContext::with_cwd("."), &missing_target)
            .unwrap();
        assert!(!result.ok);
    }
}
