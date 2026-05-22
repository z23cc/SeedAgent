# SeedAgent

SeedAgent is a minimal Rust seed for a self-bootstrapping agent. The CLI is `seed`, and the
internal core agent is `Seed`.

The first version intentionally keeps the kernel small:

- typed tool calls
- exact-match file patching
- bounded file reads
- shell execution with timeout
- JSONL session traces
- per-turn working summaries inspired by GenericAgent checkpoints
- GenericAgent-style memory tools: `update_working_checkpoint`, `start_long_term_update`,
  and `complete_long_term_update`
- L0/L1/L2/L3/L4 memory skeleton with a pointer-only L1 insight file plus a JSON index
- GenericAgent-style plan mode with durable `plan.md`, `state.json`, and verification gate
- reflection into `SKILL.md` drafts
- lightweight memory and skill search/fetch tools
- ForgeCode-style LLM provider routing
- first real LLM call path through OpenAI Responses
- multi-turn LLM planner loop for choosing local tools
- Codex app-server as the preferred execution backend
- RepoPrompt skill routing for Codex delegate tasks, with searchable execution metadata

The workspace is split so future features can grow without reshaping the core:

```text
crates/
  agent-core      # tool protocol, events, registry
  agent-llm       # provider ids, model ids, routes, request transforms
  agent-memory    # L0-L4 memory layout, L1 insight/index, memory fetch/search
  agent-plan      # durable plan.md/state.json state machine and verification context
  agent-repoprompt # RepoPrompt CLI/MCP backend wrapper
  agent-delegate  # external agent backends such as Codex app-server
  agent-tools     # shell/read/write/patch/checkpoint tools
  agent-session   # JSONL sessions and replay
  agent-skills    # session reflection and skill draft creation
  agent-cli       # clap command surface
  agent-tui       # reserved for ratatui UI
skills/           # generated skill drafts
memory/           # meta rules, L1 insight/index, global facts, session archive
sessions/         # JSONL execution traces
```

Try it after installing the local CLI:

```bash
cargo install --path crates/agent-cli --force
seed doctor
seed run "shell: pwd"
seed reflect
seed skill create --name first-shell
seed skill list
seed tool memory-search shell
seed tool memory-fetch meta-rules
seed tool skill-search shell
seed tool update-working-checkpoint "Repo root verified at $PWD"
seed tool start-long-term-update "verified reusable shell workflow" --evidence "run_shell exited 0"
seed tool complete-long-term-update skip --reason "not durable" --evidence "one-off"
seed run "Find a shell-related skill, fetch it, then summarize it." --llm --max-turns 4
seed run "Find a shell-related skill, fetch it, then summarize it." --llm --learn --max-turns 4
seed providers --provider opencode --model gpt-5.1-codex
seed llm ask "Say pong." --model gpt-5.1 --max-output-tokens 64
seed codex "Summarize this repo." --approval deny
seed codex "Summarize without any MCP." --mcp none
seed run "Summarize this repo." --codex
seed skill search repoprompt
seed rp status
seed rp tools
seed rp windows
seed rp bind --create-if-missing
seed rp call file_search --args '{"pattern":"TODO","max_results":5}'
seed tool repoprompt-tools
seed tool repoprompt-call bind_context --args '{"op":"list"}'
seed plan create --title "Demo" --task "Implement demo" --step "Inspect context" --step "Run tests"
seed plan next
seed plan complete --item 1 --note "context inspected"
seed plan record-artifact --kind context-export --path prompt-exports/context.md --note "RepoPrompt selected context"
seed plan record-handoff --backend repoprompt --role engineer --run-id agent-run-123 --artifact-path prompt-exports/context.md --status completed --summary "implemented the selected step"
seed plan verify --dry-run
seed tool plan-create --title "Tool Demo" --task "Planner-visible plan" --step "Do one thing"
seed tool plan-record-artifact --kind verification-report --path reports/verify.md --note "independent verifier output"
```

`run --llm` defaults to the local Codex app-server planner, so it uses your existing Codex login
instead of `OPENAI_API_KEY`. Pass `--provider openai` or another provider only when you explicitly
want the HTTP provider path. `llm ask` is still a raw provider smoke command and reads
`OPENAI_API_KEY` for the built-in OpenAI provider. Compatible endpoints can use `OPENAI_BASE_URL`
with the `openai_responses_compatible` provider.

Every planner turn asks for a short `summary`, stores it in the JSONL session, and feeds a
GenericAgent-style `### [WORKING MEMORY]` anchor into the next planner prompt. The anchor includes
recent turn history, `update_working_checkpoint` key facts, related skills, and loop guard hints
when the agent is repeating a failed action or approaching the turn limit. Passing `run --learn`
after a successful LLM run consolidates that trace into the skill tree: it updates a similar
existing `SKILL.md` under `## Learned Updates`, or creates a new skill only when no sufficiently
similar skill exists. The self-bootstrapping loop is: run, summarize, consolidate, reuse.

The memory tools mirror GenericAgent's split between short-term and durable memory. Use
`update_working_checkpoint` for verified context needed during the current task. Use
`start_long_term_update` to begin a distillation pass that reads `memory/memory_management_sop.md`
and only writes durable facts or reusable SOPs when there is successful evidence. Once
`start_long_term_update` succeeds, the next planner turn enters phase 2 settlement and must choose
exactly one branch: update L2 `global_facts.md`, update an existing L3 skill, or skip with a reason.
After the branch writes or skips, the planner calls `complete_long_term_update` so the session has
an auditable settlement event. Memory and skill writes automatically rebuild the L1 insight and
machine index. Tool writes to `memory/` and `skills/` are guarded: generated L1/index files cannot
be edited directly, existing durable memory cannot be overwritten wholesale, and secret-like or
unverified text is rejected before it lands.

The durable memory skeleton is intentionally simple:

- L0: `memory/meta_rules.md` and `memory/memory_management_sop.md`
- L1: `memory/l1_insight.md` for pointer-only navigation, plus `memory/index.json` for machine search
- L2: `memory/global_facts.md`
- L3: `skills/<slug>/SKILL.md`
- L4: `sessions/*.jsonl` plus `memory/session_archive.jsonl`

`run --llm` injects L0, `memory/l1_insight.md`, and a compact machine index by default. Deeper
bodies stay out of the prompt until the planner calls `memory_search` and then `memory_fetch`.

Plan mode mirrors GenericAgent's external plan-state pattern. `plan_create` writes
`plans/<id>/plan.md`, `state.json`, and a mandatory `[VERIFY]` checkbox. The planner can then call
`plan_next`, do exactly the next unchecked item, and call `plan_complete`. Once all non-verify items
are complete, `plan_verify` writes `verify_context.json` and can launch an independent RepoPrompt
`agent_run` verifier with the `pair` Codex role by default. `VERDICT: PASS` marks the plan
verified; `VERDICT: FAIL` appends a `[FIX]` item and keeps the plan from finishing.

Each plan also carries an orchestration ledger for RepoPrompt-led execution. `plan_record_artifact`
records selected context exports, RepoPrompt exports, verification contexts, and verifier reports.
`plan_record_handoff` records who executed a step, which backend or role ran it, any run/thread id,
the artifact used, and the outcome summary. This keeps Seed as the durable state machine while
RepoPrompt does the context building, handoff, review, and verification work.
Planner prompts and `plan_next` now treat this as protocol, not optional bookkeeping: RepoPrompt
exports should be followed by `plan_record_artifact`, and RepoPrompt `agent_run` or Codex delegated
work should be followed by `plan_record_handoff`. `plan_verify` records its RepoPrompt verifier
handoff automatically.

`codex` and `run --codex` use the local `codex app-server` transport and your existing Codex
login/config. Passing `--model` is optional; when omitted, Codex chooses from local config. By
default the delegate starts Codex with plugins disabled and only `RepoPrompt` allowed. Use
`--mcp none` to disable every configured MCP server, `--mcp all` to keep Codex's full MCP surface,
or repeat `--mcp-allow <name>` to keep a custom allowlist such as `RepoPrompt` plus `semgrep`.
With plugins disabled, MCP discovery reads `~/.codex/config.toml` directly and only falls back to
`codex mcp list` if local config parsing fails.

Before `codex` and `run --codex` send a prompt to Codex, the CLI now routes broad repository tasks
to a local RepoPrompt SOP skill and injects that skill body into the delegated prompt. Planning or
implementation prompts route to `skills/repoprompt-deep-plan/SKILL.md`; review prompts route to
`skills/repoprompt-review/SKILL.md`; investigation prompts route to
`skills/repoprompt-investigate/SKILL.md`. This makes RepoPrompt usage explicit instead of relying
on Codex to infer the right skill from MCP availability alone. Skill discovery also indexes
frontmatter fields such as `task_type`, `capabilities`, `required_tools`, `preferred_backend`,
`autonomous_safe`, and `blast_radius`, so planners can choose skills by execution fit rather than
name alone.

RepoPrompt is also available as a first-class SeedAgent backend. The `agent-repoprompt` crate wraps
the RepoPrompt CLI and knows the full 18-tool MCP surface: exploration, context selection, oracle
conversations, editing, git, workspace routing, settings, and agent control. It uses
`REPOPROMPT_CLI` when set, otherwise `$HOME/RepoPrompt/repoprompt_cli` when present, otherwise
`repoprompt_cli` from `PATH`. The local planner can call `repoprompt_tools` to inspect the surface,
`repoprompt_exec` for RepoPrompt shorthand commands, and `repoprompt_call` for any specific
RepoPrompt MCP tool with JSON args. Workspace-scoped RepoPrompt calls default to the current cwd
when no window, context id, or working directory is supplied; discovery calls such as window or
workspace listing stay unbound. The CLI mirrors this through `seed rp ...`, and `seed rp bind`
defaults to binding the current directory when no `--working-dir` is given.
