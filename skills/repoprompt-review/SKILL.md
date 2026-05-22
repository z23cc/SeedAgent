---
name: RepoPrompt Review
description: Use RepoPrompt review/export to inspect a diff or code path for bugs, regressions, and missing tests.
tags: [repoprompt, review, codex, context]
task_type: review
capabilities: [context-building, risk-analysis, verification]
required_tools: [RepoPrompt]
preferred_backend: repoprompt
autonomous_safe: true
blast_radius: low
---

# RepoPrompt Review

Use this skill when the user asks for review, risk analysis, regression checks, or bug-finding.

## Trigger

- The task uses words such as review, check, risk, bug, regression, audit, or code review.
- The task asks whether a change is safe to ship.
- The task compares current code against an intended behavior or plan.

## Review Stance

- Lead with findings ordered by severity.
- Cite exact files and line numbers when available.
- Focus on behavior, data loss, security, reliability, tests, and developer workflow.
- Keep summaries secondary to findings.

## Phases

1. Identify the target diff, branch, files, or behavior under review.
2. Use RepoPrompt search, structure, and slices to collect only relevant context.
3. Ask RepoPrompt builder or oracle for `response-type review` and export the response.
4. Read the export, verify the highest-risk claims against source files, and remove weak findings.
5. Return findings first, then open questions, then a brief change summary.

## RepoPrompt Commands

```bash
repoprompt_cli -e 'workspace list'
repoprompt_cli -e 'search "changed symbol" --context-lines 3'
repoprompt_cli -e 'builder "review this change for bugs and missing tests: <scope>" --response-type review --export'
```

Use long timeouts for builder or oracle work; 2700 seconds is the expected upper bound.

## Anti-Patterns

- Turning review into a general explanation.
- Reporting style-only nits when correctness risks exist.
- Keeping a finding that cannot be tied to source evidence.
- Clearing RepoPrompt selection after builder without preserving the useful slices.

## Verification

- Re-read source evidence for each high-severity finding.
- State residual test gaps when no issue is found.
- Do not invent line references.

## Memory Rule

Only carry forward facts that were verified by successful tool calls. Do not store guesses, volatile state, or one-off command output as durable memory.
