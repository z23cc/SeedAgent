# Memory Management SOP

Use this SOP when `start_long_term_update` is called.

Only write long-term memory when a fact is verified by a successful tool call and likely useful in future tasks.

## Layers

- L1 insight: pointer-only navigation in `memory/l1_insight.md`; generated from L2/L3/L4, never edited directly.
- L2 global facts: stable local paths, configuration constraints, durable preferences, credential references without secrets.
- L3 skills: reusable workflows, exact prerequisites, common failure modes, verification commands.
- L4 session archive: JSONL traces in `sessions/`; crystallize successful traces with `run --learn`.

## Rules

1. Read the current target before editing it.
2. Make the smallest local update.
3. Skip memory writes for guesses, temporary variables, generic advice, or one-off outputs.
4. Prefer updating an existing skill over creating a duplicate when the workflow is the same.
5. Never store raw secrets.
6. Let the runtime rebuild L1 after L2/L3 writes; do not patch generated `l1_insight.md` or `index.json`.
