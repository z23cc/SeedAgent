//! `ask_user` tool: prompt the human operator via stdin for clarification.
//!
//! extracted from `lib.rs`. Fully self-contained — only depends
//! on `parse_tool_args` from the crate root.

use std::io::{BufRead, IsTerminal, Write};

use agent_core::{Tool, ToolCall, ToolContext, ToolError, ToolResult};
use serde::Deserialize;
use serde_json::json;

use crate::parse_tool_args;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct AskUserArgs {
    question: String,
    #[serde(default)]
    candidates: Vec<String>,
}

pub struct AskUserTool;

impl Tool for AskUserTool {
    fn name(&self) -> &'static str {
        "ask_user"
    }

    crate::tool_description!("ask_user");

    crate::impl_args_schema!(AskUserArgs);

    fn execute(&self, _ctx: &ToolContext, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let args: AskUserArgs = parse_tool_args(call)?;

        let stdin = std::io::stdin();
        if !stdin.is_terminal() {
            return Ok(ToolResult::error(
                call,
                "stdin is not a terminal; ask_user cannot collect a response in non-interactive mode",
            ));
        }
        let mut stderr = std::io::stderr().lock();
        let _ = writeln!(stderr, "\nseed asks: {}", args.question);
        if !args.candidates.is_empty() {
            for (idx, candidate) in args.candidates.iter().enumerate() {
                let _ = writeln!(stderr, "  {}) {}", idx + 1, candidate);
            }
            let _ = writeln!(stderr, "  reply with a number or your own answer.");
        }
        let _ = write!(stderr, "> ");
        let _ = stderr.flush();

        let mut line = String::new();
        stdin
            .lock()
            .read_line(&mut line)
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        let trimmed = line.trim().to_string();
        if trimmed.is_empty() {
            return Ok(ToolResult::error(call, "user replied with an empty line"));
        }

        let resolved = if !args.candidates.is_empty()
            && let Ok(idx) = trimmed.parse::<usize>()
            && idx >= 1
            && idx <= args.candidates.len()
        {
            args.candidates[idx - 1].clone()
        } else {
            trimmed.clone()
        };

        Ok(ToolResult::ok(
            call,
            json!({
                "status": "success",
                "question": args.question,
                "answer": resolved,
                "raw_input": trimmed,
            }),
        ))
    }
}
