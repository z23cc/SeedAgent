# Meta Rules

- Keep the active context small: load L0 meta rules and pointer-only L1 insight by default, fetch deeper memory on demand.
- Only write memory from verified tool results.
- Do not store secrets, raw credentials, guesses, temporary variables, or one-off command output.
- Use `update_working_checkpoint` for short-term facts needed during the current run.
- Use `start_long_term_update` before writing durable L2 facts or updating L3 skills.
- Prefer updating an existing skill over creating a duplicate when the workflow is the same.
- Treat `memory/l1_insight.md` and `memory/index.json` as generated navigation state; update L2/L3, then rebuild.
