# SeedAgent

A minimal Rust kernel for self-bootstrapping LLM agents. The binary
is `seed`. The planner picks one typed tool per turn, records every
side-effect into a JSONL session, and updates a structured memory
tree on disk.

The kernel deliberately stays small — capabilities ship as crates,
not as kernel changes. Today's tool surface is ~32 typed tools
(read/write/patch files, shell, plan state, skill discovery,
RepoPrompt bridge, subagent spawning, memory protocol) across 12
workspace crates.

## Install

```bash
# Build + install the `seed` binary
cargo install --path crates/agent-cli --force

# Verify
seed doctor
```

Toolchain is pinned in `rust-toolchain.toml` (currently 1.95). First
`cargo` invocation auto-installs it.

## Quick start

```bash
# One-shot LLM run
seed run --llm --provider codex --mode read "What does this project do?"

# Interactive REPL (Ctrl-D to exit)
seed chat

# Run an eval suite against a backend (regex + LLM-as-judge grading)
seed eval --provider codex

# Micro-benchmarks for perf claims (RF42 / RF45 / RF46)
cargo bench -p agent-bench
```

## Planner backends

`--provider <id>` picks the LLM that drives the planner loop. Four
families, 9 ids:

| `--provider` | Routes via | Best for | Needs |
|---|---|---|---|
| `codex` (default) | Local `codex app-server` over stdio JSON-RPC | Most tasks; fast; uses Codex login | Codex CLI installed |
| `repoprompt_oracle` (or `repoprompt`) | RepoPrompt `ask_oracle` (one-shot Q&A) | Quick reasoning with RepoPrompt's curated context | RepoPrompt running |
| `repoprompt_agent` (or `rp_agent` / `rp-agent`) | RepoPrompt `agent_run` (full Agent Mode) | Multi-turn tasks needing rich workspace context | RepoPrompt running |
| `openai` / `anthropic` / `google` / `openai_compatible` / `openai_responses_compatible` / `opencode` | HTTP, with SSE streaming | Direct API access; CI without local backends | Matching `*_API_KEY` env var |

Pick a role/model via `--model`:
- For `codex`: any Codex model id (`gpt-5.1` etc.)
- For `repoprompt_oracle`: oracle mode (`chat | plan | edit | review`)
- For `repoprompt_agent`: role label (`explore | engineer | pair | design`)
- For HTTP providers: provider-specific model id

`--mode read|write|auto` toggles the read-only tool catalog. `auto`
classifies by goal keywords (default).

## What it does each turn

```
goal → resolve mode (read/write/auto)
     → build planner prompt (system + memory + tool catalog + observations)
     → planner.plan() → PlannedAction { tool | finish }
     → if Tool: execute via registry, record observation, loop
     → if Finish: write Reflection + RunFinished events
```

Every observable side-effect lands in `sessions/<uuid>.jsonl` as an
`AgentEvent`. Use `seed reflect` to summarize and `seed replay` to
walk through a recorded session.

## Memory model (L0–L4)

`memory/` and `skills/` form a layered store the planner can search
and pull from. Writes are guarded — generated indexes (L1) are
read-only from the tool layer; the long-term update protocol forces
the planner to declare intent (`start_long_term_update`) and audit
the write (`complete_long_term_update`).

| Layer | What | Location |
|---|---|---|
| L0 | Meta-rules + memory SOP | `memory/meta_rules.md`, `memory/memory_management_sop.md` |
| L1 | Generated pointer index | `memory/l1_insight.md`, `memory/index.json` |
| L2 | Stable global facts | `memory/global_facts.md` |
| L3 | Skills (reusable workflows) | `skills/<slug>/SKILL.md` |
| L4 | Session archive | `memory/session_archive.jsonl` + `sessions/*.jsonl` |

The planner sees L0 + L1 every turn (compact). L2/L3 are loaded on
demand via `memory_search` → `memory_fetch`.

## Plan protocol

For non-trivial implementation goals, the planner can create a
durable plan under `plans/<id>/`:

```bash
# CLI mirror of the planner's plan_create tool
seed plan create --task "..." --steps "step 1" --steps "step 2"
seed plan list
seed plan status <id>
seed plan next <id>
```

A `[VERIFY]` checkbox is always added — `plan_verify` runs an
independent verifier (currently RepoPrompt `agent_run`) before the
plan can be marked complete. Items with `[D]` are delegated;
items with `[P]` can run in parallel.

## Skill system

Skills are reusable workflow recipes that ship with the project.
The planner can `skill_search` → `skill_fetch` to load one at any
time. `seed run --learn` consolidates a successful run into a new
skill (or updates a similar existing one).

```bash
seed skill list
seed skill search "memory"
seed skill fetch repoprompt-deep-plan
```

Skills with `repoprompt_*` frontmatter auto-bind the next RepoPrompt
call to the skill's workspace dirs.

## Eval suite

`evals/*.toml` defines goals + grading rules. Two grade kinds:

- `kind = "regex"` — answer must match the pattern
- `kind = "judge"` — separate backend judges PASS/FAIL against a rubric

```bash
# Run all evals against codex
seed eval --provider codex

# Run in-process (reuses CodexSession across evals; faster)
seed eval --provider codex --in-process

# Validate that `--learn` actually produces useful skills
seed eval-learn evals/01_count_crates.toml --provider codex
```

The eval-learn subcommand runs the same task 3× (baseline, with
--learn, post-learn) and reports whether the learned skill reduced
turn count or upgraded the grade.

## Benchmarks

`cargo bench -p agent-bench` runs criterion micro-benchmarks for
the perf claims accumulated through the project's history:
ToolRegistry caching (RF42-A3), schemars schema generation
(RF45-Phase1), JSON repair (RF46-A), memoize key (RF35-2),
AgentLoopState construction. Baseline numbers live in `CLAUDE.md`'s
RF49-B section.

## Codex integration knobs

- `--codex` — fast-path: hand the whole goal to `codex app-server`
  in one turn (skip the planner loop entirely)
- `--use-daemon` — connect via `codex app-server proxy` (running
  daemon) instead of spawning a fresh app-server per goal
- `--approval deny|accept-once|accept-for-session` — what to do
  when Codex requests tool approval
- `--mcp none|all` + `--mcp-allow <name>` — which MCP servers
  Codex discovers (default: only RepoPrompt allowed)

The Codex client is REPL-lifetime-cached (`CodexSession`) — hot-swap
cwd/model/effort between turns without restart, only respawn when
the `CodexLaunchFingerprint` (plugins / MCP policy / launch args)
actually changes.

## REPL slash commands

`seed chat` enters an interactive REPL. Slash commands:

| Group | Commands |
|---|---|
| View | `/help` `/doctor` `/skills` `/tools` `/memory` `/plan` `/plans` `/dump` `/providers` |
| Configure | `/cd` `/mode` `/provider` `/model` `/effort` |
| Operate | `/new` `/retry` `/compact` `/sync` |
| Exit | `/exit` (or `/quit` / `:q`) |

`!<cmd>` runs a shell command and prints the output without
involving the planner.

## Architecture (one paragraph)

12 crates in `crates/`. `agent-core` defines the shared
`Tool`/`ToolRegistry`/`ToolCall`/`AgentEvent` types + the TUI / session
sub-modules. `agent-runtime` owns the planner loop (`AgentLoopState`,
`PlannedAction`, retries). `agent-tools` registers the 32 typed tools
(split into per-family modules: `files.rs`, `plan.rs`,
`memory_protocol.rs`, etc.). `agent-cli` is the `seed` binary
(subcommand routing + REPL + `run_goal`). `agent-llm` is the HTTP
provider registry; `agent-delegate` wraps `codex app-server`;
`agent-repoprompt` wraps the RepoPrompt CLI. `agent-memory` /
`agent-plan` / `agent-skills` own the L0-L4 / plan / skill state
machines respectively. `agent-bench` is the criterion harness.

For the full architectural decision log, see `CLAUDE.md` (it's the
agent-readable design doc — RF24 through RF52, ~1k lines).

## Status

- 369 unit + integration tests; basic GitHub Actions CI gating
  build + test
- 4 planner backend families, 9 provider ids
- 8 evals across 4 categories (regex + judge grading)
- 10 micro-benchmarks
- Schema-driven tool args (planner sees real input shapes, not
  guesses)
- Markdown-file tool descriptions (`crates/agent-tools/descriptions/`)

What's intentionally **not** here:
- `ToolCatalog` enum dispatch (forge's pattern) — defer until a
  native function-calling protocol consumer arrives (see
  CLAUDE.md RF50-C)
- LLM-as-judge across all evals — only opt-in per eval; regex
  grading covers the simple cases
- Production observability (metrics export, structured logging
  beyond JSONL sessions)

## Contributing

Standard cargo. Add a new tool: write a struct + `impl Tool`,
declare its args struct with `#[derive(schemars::JsonSchema)]`,
add `crate::impl_args_schema!(MyArgs)` + `crate::impl_pure_read!()`
(if read-only) to the impl, write the description to
`crates/agent-tools/descriptions/<name>.md`, register in
`seed_registry()`. Run `cargo test --workspace` before opening a PR;
clippy + rustfmt are advisory in CI.

## License

MIT.
