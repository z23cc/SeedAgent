---
name: RepoPrompt Investigate
description: Use RepoPrompt to investigate repository structure, execution flow, and design boundaries with compact context.
tags: [repoprompt, investigate, codex, context]
task_type: investigation
capabilities: [context-building, code-archaeology, synthesis]
required_tools: [RepoPrompt]
preferred_backend: repoprompt
autonomous_safe: true
blast_radius: low
---

# RepoPrompt Investigate

Use this skill when the user asks to understand, analyze, trace, or summarize a codebase.

## Trigger

- The task asks how a feature works, where behavior lives, or what design to borrow.
- The task needs code archaeology, call flow, or repository map context.
- The task should produce a report or recommendation before implementation.

## Phases

1. Bind or identify the RepoPrompt workspace/window for the target repository.
2. Start with tree and structure to map the likely area.
3. Use search with context lines to find symbols, entry points, docs, and tests.
4. Read narrow slices for definitions and key call paths.
5. Use RepoPrompt builder/oracle for synthesis when the answer spans multiple files.
6. Save or reference the exported synthesis when it becomes input to planning or execution.

## RepoPrompt Commands

```bash
repoprompt_cli -e 'workspace list'
repoprompt_cli -e 'tree --mode folders'
repoprompt_cli -e 'structure . --max-results 80'
repoprompt_cli -e 'search "entry point" --context-lines 3'
repoprompt_cli -e 'builder "investigate and summarize: <question>" --response-type plan --export'
```

Use long timeouts for builder or oracle work; 2700 seconds is the expected upper bound.

## Output Shape

- Observed facts first.
- Inferences clearly labeled as inferences.
- Exact files or exported RepoPrompt paths for important claims.
- A concise next-step recommendation when the user is deciding what to build.

## Anti-Patterns

- Dumping broad file contents into chat before using RepoPrompt selection.
- Losing the export path when the investigation should feed a later plan.
- Mixing source facts and design guesses without labels.

## Verification

- Confirm entry points against source slices or codemap output.
- Check at least one caller or test when explaining behavior.
- Name what remains unknown when the evidence is partial.

## Memory Rule

Only carry forward facts that were verified by successful tool calls. Do not store guesses, volatile state, or one-off command output as durable memory.
