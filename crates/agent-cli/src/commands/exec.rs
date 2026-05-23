//! Tool-call execution kernel: shared between `seed tool` (one-shot) and
//! `seed run` (in-loop). Owns the session-event side-effects so the JSONL
//! trace is replayable end-to-end, and rebuilds the L1/index whenever the
//! call mutated `memory/` or `skills/`.

use std::path::{Path, PathBuf};

use agent_core::{AgentEvent, ToolCall, ToolContext, ToolResult};
use agent_session::SessionWriter;
use anyhow::Result;

pub(crate) fn execute_call(
    session: &mut SessionWriter,
    cwd: &Path,
    skills_dir: &Path,
    sessions_dir: &Path,
    call: ToolCall,
) -> Result<ToolResult> {
    execute_call_with_turn(session, cwd, skills_dir, sessions_dir, 0, call)
}

pub(crate) fn execute_call_with_turn(
    session: &mut SessionWriter,
    cwd: &Path,
    skills_dir: &Path,
    sessions_dir: &Path,
    current_turn: usize,
    call: ToolCall,
) -> Result<ToolResult> {
    let registry = agent_tools::seed_registry();
    let ctx = ToolContext::with_paths(
        cwd.to_path_buf(),
        skills_dir.to_path_buf(),
        cwd.join("memory"),
        sessions_dir.to_path_buf(),
    )
    .with_turn(current_turn);
    session.append(AgentEvent::ToolStarted { call: call.clone() })?;
    let result = match registry.execute(&ctx, &call) {
        Ok(result) => result,
        Err(err) => ToolResult::error(&call, err.to_string()),
    };
    if call.name == "update_working_checkpoint" && result.ok {
        let key_info = result
            .content
            .get("key_info")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        let related_skill = result
            .content
            .get("related_skill")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        session.append(AgentEvent::CheckpointUpdated {
            key_info,
            related_skill,
        })?;
    }
    if call.name == "start_long_term_update" && result.ok {
        let reason = result
            .content
            .get("reason")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        let evidence = result
            .content
            .get("evidence")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        session.append(AgentEvent::LongTermUpdateStarted { reason, evidence })?;
    }
    if call.name == "complete_long_term_update" && result.ok {
        let decision = result
            .content
            .get("decision")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        let target = result
            .content
            .get("target")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let reason = result
            .content
            .get("reason")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        let evidence = result
            .content
            .get("evidence")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let changed = result
            .content
            .get("changed")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        session.append(AgentEvent::LongTermUpdateSettled {
            decision,
            target,
            reason,
            evidence,
            changed,
        })?;
    }
    maybe_rebuild_memory_index_after_tool(&ctx, &call, &result)?;
    session.append(AgentEvent::ToolFinished {
        result: result.clone(),
    })?;
    Ok(result)
}

fn maybe_rebuild_memory_index_after_tool(
    ctx: &ToolContext,
    call: &ToolCall,
    result: &ToolResult,
) -> Result<()> {
    if !result.ok || !matches!(call.name.as_str(), "patch_file" | "write_file") {
        return Ok(());
    }
    let Some(path) = result.content.get("path").and_then(|value| value.as_str()) else {
        return Ok(());
    };
    let path = PathBuf::from(path);
    if path.starts_with(&ctx.memory_dir) || path.starts_with(&ctx.skills_dir) {
        let paths = agent_memory::MemoryPaths::new(
            ctx.memory_dir.clone(),
            ctx.skills_dir.clone(),
            ctx.sessions_dir.clone(),
        );
        agent_memory::rebuild_index(&paths)?;
    }
    Ok(())
}
