---
name: RepoPrompt Deep Plan
description: Use RepoPrompt builder/export to create a grounded implementation plan before broad code changes.
tags: [repoprompt, plan, codex, context]
---

# RepoPrompt Deep Plan

Use this skill for complex implementation, architecture, refactor, or multi-file planning tasks.

## Trigger

- The task asks to implement, refactor, design, optimize, or plan a codebase change.
- The task needs repository-wide context or careful file selection.
- The task will hand work to Codex after planning.

## Preconditions

- RepoPrompt app is running and its MCP server or CLI is available.
- Preferred CLI path: `repoprompt_cli`
- Current Codex MCP policy should allow `RepoPrompt` unless the user requested a different policy.

## Phases

1. Bind or identify the active RepoPrompt workspace/window for the target repository.
2. Use RepoPrompt tree, structure, search, and sliced reads to locate load-bearing files.
3. Build context through RepoPrompt selection rather than broad raw file dumps.
4. Ask RepoPrompt builder for a plan and export the response.
5. Treat the export path as durable evidence for the execution handoff.
6. Turn the export into a concrete plan artifact before implementation when the change spans multiple steps.

## RepoPrompt Commands

```bash
repoprompt_cli -e 'workspace list'
repoprompt_cli -e 'tree --mode folders'
repoprompt_cli -e 'structure crates/ --max-results 50'
repoprompt_cli -e 'search "target symbol" --context-lines 3'
repoprompt_cli -e 'builder "create an implementation plan for: <task>" --response-type plan --export'
```

Use long timeouts for builder or oracle work; 2700 seconds is the expected upper bound.

## Handoff

- Pass the RepoPrompt export path to Codex or downstream agents.
- Keep RepoPrompt selection discipline: after builder, adjust with add/remove/slices instead of clearing context casually.
- Ask Codex to execute the first concrete item, run focused verification, then update the plan state.

## Anti-Patterns

- Manually reading a large repository before asking RepoPrompt to curate context.
- Treating a builder answer as a chat note instead of an exported artifact.
- Starting implementation while ambiguities that change architecture remain open.
- Letting generated plans live only in the model context.

## Verification

- Confirm the plan names the files or modules it depends on.
- Confirm acceptance checks are executable.
- Before finishing, run the narrow test or build command tied to the changed surface.

## Memory Rule

Only carry forward facts that were verified by successful tool calls. Do not store guesses, volatile state, or one-off command output as durable memory.
