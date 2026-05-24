# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, test, run

Rust workspace, edition 2024, MSRV `1.95`, resolver 3. The published binary is `seed` (defined in `crates/agent-cli`).

```bash
cargo build --workspace
cargo test  --workspace
cargo test  -p agent-plan                       # one crate
cargo test  -p agent-runtime is_read_only       # one test by name substring
cargo run   -p agent-cli -- doctor              # dev-run without installing
cargo install --path crates/agent-cli --force   # install `seed` to PATH
seed doctor                                     # smoke: lists tools, providers, RepoPrompt path
```

There is no separate lint/format config beyond stock `cargo fmt` and `cargo clippy --workspace --all-targets`. CI is not configured in-tree.

## Architecture

SeedAgent is a self-bootstrapping agent kernel. The CLI `seed` drives a small planner loop that picks one typed tool per turn, records the result in a JSONL session, and updates a structured memory tree. The kernel deliberately keeps the agent surface tiny so future capabilities can be added as crates.

### Crate map (workspace members in `Cargo.toml`)

- `agent-core` — `Tool` trait, `ToolRegistry`, `ToolCall`/`ToolResult`, `ToolContext` (binds `cwd`, `skills_dir`, `memory_dir`, `sessions_dir`), the `RunMode`/`ModeSource` enums (RF27), and the `AgentEvent` enum that every other crate emits into sessions. RF34-2 added `AgentEvent::PlannerRetry { turn, attempt, of, backoff_ms, kind, reason }` so transient parse/transport retries are visible in session JSONLs + spinner subtitles, not silent. **RF36-1**: `TurnTimings` carries `prompt_chars` (input side) alongside `planner_chars` (output side) so users can see context-input growth — surfaced in the run footer as `prompt chars: 320k / planner chars: 18k`.
- `agent-runtime` — the planner loop. Defines `PlannedAction` (the JSON contract the LLM must return: `tool` or `finish`, optional `summary`), `WorkingMemory` (with `earlier_summary` for >30-turn runs — see RF16), `AgentLoopState`, prompt assembly (`planner_prompt_*`), the read-only vs implementation goal classifier (`is_read_only_analysis_goal`), and `run_agent_loop_*` entry points. `parse_planned_action` runs a `sanity_check` pass that rejects obviously-broken actions (empty `tool_name`, `args: null`, blank `finish.answer`) as `InvalidPlannerJson` so the retry path nudges the planner instead of dispatching garbage. Errors split into retryable `RuntimeError::Planner(_)` (network/stdio blip — backoff + retry) vs non-retryable `RuntimeError::PlannerFatal(_)` (auth rejection, ToolErrorTracker abort, model said no) — runtime only re-arms the former.
- `agent-llm` — provider registry and HTTP routing. Built-in IDs: `openai`, `openai_compatible`, `openai_responses_compatible`, `anthropic`, `google`, `opencode`, `codex`, `repoprompt_oracle`. Two are not HTTP: `codex` goes through the local Codex app-server (see `agent-delegate`) and uses your existing Codex login instead of `OPENAI_API_KEY`; `repoprompt_oracle` opts the planner into RepoPrompt's `ask_oracle` so prompts inherit RepoPrompt's curated context (see `agent-repoprompt::send_oracle`).
- `agent-delegate` — `CodexAppServerClient`: spawns `codex app-server --listen stdio://`, speaks JSON-RPC, handles approval callbacks (`ApprovalMode::{Deny,AcceptOnce,AcceptForSession}`), and applies an `McpPolicy` (`None`/`All`/`Allow(list)`; default = only `RepoPrompt`). MCP discovery reads `~/.codex/config.toml` directly when plugins are disabled, falling back to `codex mcp list --json`. Per-turn config (cwd/model/effort/sandbox/approval) is hot-swappable via `set_cwd`/`set_model`/`set_effort`/`set_sandbox`/`set_approval_*` — `start_turn` reads `self.cfg` at request build time so no restart needed. `CodexLaunchFingerprint` captures the subset (`plugins_enabled`, `mcp_policy`, `command`, `args`, `experimental_api`) that DOES require a fresh subprocess; `commands::codex_session::CodexSession` (agent-cli) uses the fingerprint to decide reuse-vs-restart and powers REPL-lifetime Codex client caching for both the `--codex` fast-path AND (RF29-1) the planner-loop `--provider codex` backend (`CodexPlanner<'a>` borrows `&'a mut CodexAppServerClient` from the session via `Box<dyn Planner + 'a>`). **RF35-1**: `stream_turn` captures `thread/tokenUsageUpdated` notifications via `parse_thread_token_usage`; `CodexRunResult.tokens: Option<TokenUsage>` carries the per-turn breakdown (input / cached_input / output / reasoning_output / total). The `--codex` fast-path prints a dim "(tokens: …)" line after the answer when usage was reported.
- `agent-repoprompt` — wraps the RepoPrompt CLI (`REPOPROMPT_CLI` env, then `$HOME/RepoPrompt/repoprompt_cli`, then `repoprompt_cli` on PATH). Knows the full 18-tool MCP surface; exposed to the planner via the `repoprompt_tools`/`repoprompt_exec`/`repoprompt_call` tools and to humans via `seed rp ...`. Workspace-scoped calls default to the current cwd unless a window/context/working-dir is provided.
- `agent-plan` — durable plan state machine. A plan is a directory under `plans/<id>/` containing `plan.md`, `state.json`, an always-present `[VERIFY]` checkbox, and `verify_context.json`. Tracks `PlanOrchestration` (preferred backend, artifacts, handoffs, verification records). `plan_verify` is a gate: `VERDICT: PASS` → `Verified`; `VERDICT: FAIL` appends a `[FIX]` item and blocks completion. Items support `[D]` (delegate via `spawn_subagent`) and `[P]` (parallel) markers. RepoPrompt integration: `import_repoprompt_plan` parses a builder export into `ImportedPlan { title, task, steps }` with markers auto-applied via keyword heuristics; `parse_plan_review` extracts a `## Recommended Fixes` section from an oracle review; `PlanStore::append_items` inserts new numbered `[FIX]` rows before the `[VERIFY]` gate. Surfaced as three tools: `plan_create_from_repoprompt` (import an existing export), `plan_create_via_repoprompt` (one-shot via `context_builder`), `plan_refine_via_repoprompt` (oracle review → `[FIX]` items + reviewer handoff). CLI mirrors: `seed plan import --path X.md`, `seed plan build --task "..."`, `seed plan refine [id] [--focus "..."]`.
- `agent-memory` — the L0–L4 memory skeleton. L0: `memory/meta_rules.md` + `memory/memory_management_sop.md`. L1: generated `memory/l1_insight.md` (pointer-only) + `memory/index.json` (machine index). L2: `memory/global_facts.md`. L3: `skills/<slug>/SKILL.md`. L4: `sessions/*.jsonl` plus `memory/session_archive.jsonl`. Provides `rebuild_index`, `search_index`, `fetch_memory`. **L1 and the index are generated** — never edit them directly; rewriting L2/L3 then rebuilding is the only supported path. **RF29-2**: `append_session_archive_record` enforces a FIFO cap (`DEFAULT_SESSION_ARCHIVE_CAP = 500`, override via `SEED_SESSION_ARCHIVE_CAP` env) so the archive can't grow unbounded; cap enforcement runs after the append so a failure there doesn't lose the freshly-recorded session.
- `agent-tools` — the registered planner tools (`seed_registry()` in `crates/agent-tools/src/lib.rs`). Covers memory (`memory_search`, `memory_fetch`), skills (`skill_*`), plans (`plan_*`), RepoPrompt bridge tools, filesystem (`read_file`, `patch_file`, `write_file`, bounded), `shell` (timed-out subprocess), `tool_describe` (RF33-3 description-recovery), and the GenericAgent-style memory tools `update_working_checkpoint`, `start_long_term_update`, `complete_long_term_update`. Writes to `memory/` and `skills/` are guarded: generated L1/index files are read-only, durable memory can't be wholesale overwritten, and secret-like/unverified text is rejected. **RF34-1**: `parse_tool_args(call)` + `repair_tool_args(value)` normalize planner-supplied args before deserializing — unwraps stringified JSON, treats `null` as `{}`, peels sole `{"args": {...}}` envelopes — so common planner deviations don't crash the tool layer. Every Tool impl uses `parse_tool_args` instead of inline `serde_json::from_value`. **RF34-3**: `truncate_middle_with_stats` returns `(text, TruncationStats { was_truncated, original_bytes })`; shell + repoprompt outputs surface those as `*_truncated` + `*_original_bytes` JSON fields so the planner can tell when it's only seeing a slice of the real output. **RF39**: `ReadFilesArgs.paths` is `Vec<ReadFilesEntry>` where each entry is an untagged-enum of `String | { path, start?, count?, keyword? }`. Per-file fields override the uniform top-level defaults; aliases (`file`/`start_line`/`lines`/`pattern`/…) work inside the object too. Lets a planner say "read main.rs:1-50 and lib.rs:200-260 in one call" without splitting. Effective per-file `start`/`count` are echoed in the result so the planner sees what we applied. **RF38-2**: arg structs carry `#[serde(alias = "...")]` for common synonyms (`cmd` → `command`, `file`/`filename` → `path`, `q` → `query`, `old`/`new` → `old_content`/`new_content`, `text`/`body` → `content`, `timeout_ms` → secs via `resolved_timeout_secs()`). Closes the "planner used wrong field name → wasted a turn" class of failures. **RF38-1**: implicit RP `bind_context` defaults to `create_if_missing: true` so first-time use of seed in a new repo auto-registers the workspace instead of hard-failing. **RF38-3**: `humanize_rp_bind_failure(raw) -> &'static str` translates rp-cli error tails into one-line recovery hints. Exposes three process-wide guards: (1) `pub mod repoprompt_sync` — RF24-4's skill-override queue + RF25-2's `(working_dirs, window_id)` bind cache (`/cd` calls `clear_bound_window()` to invalidate the cache without dropping a pending skill override); (2) `pub mod run_mode_guard` — RF27 stores the active `RunMode` so `ShellTool` can refuse write-shaped commands when set to `ReadOnly`; (3) **RF37 `pub mod skill_tools_guard`** — process-global narrow set populated when `skill_fetch` loads a skill with non-empty `allowed-tools` frontmatter. `planner_tool_infos_for_mode` intersects with the narrow set before the RunMode filter, so a skill declaring `allowed-tools: [read_file, run_shell]` makes the planner only see those tools until the next `/new` or different skill_fetch. `shell_command_intent()` (also pub) is the classifier: redirect operators / mutating subcommands (`rm`, `mv`, `git commit`, `cargo build`, `sed -i`, …) → `Write`; obvious readers (`ls`, `cat`, `git status`, …) → `Read`; everything else → `Ambiguous` (allowed). `run_goal` must invoke `repoprompt_sync::reset()` AND `run_mode_guard::set(mode)` at the start of every run so prior state doesn't leak.
- `agent-session` — `SessionStore`/`SessionWriter` append `AgentEvent`s to `sessions/<uuid>.jsonl`. `last_session_path()` powers `seed reflect` and `seed replay`.
- `agent-skills` — session reflection (`seed reflect`) and skill drafting (`seed skill create|search|fetch|list`). Skill frontmatter is indexed (`task_type`, `capabilities`, `required_tools`, `preferred_backend`, `autonomous_safe`, `blast_radius`, **RF37 `allowed-tools`/`allowed_tools` for skill-driven tool catalog narrowing**) plus optional RepoPrompt binding (`repoprompt_working_dirs`, `repoprompt_context_id`, `repoprompt_oracle_mode`, `repoprompt_workspace_name`). When a skill has the binding, `skill_fetch` **queues** it via `agent_tools::repoprompt_sync::set_pending_override` (RF24-4) — it does NOT eagerly call `bind_context`. The very next `repoprompt_*` tool call consumes the queued override (via `default_repoprompt_working_dirs`) and then RP rebinds; after that, future rp calls default back to `ctx.cwd`, so skill bindings stay transient and the user's workspace (mutated via `/cd`) remains primary. `seed skill create` (and `run --learn`) embed the current binding into the new skill's frontmatter, closing the loop. Also owns `query_current_repoprompt_binding()` and `parse_repoprompt_status` (moved from agent-cli in RF8 so the producer-type pair stays together).
- `agent-tui` — `reedline`-based REPL used by `seed chat`. Owns the `SLASH_COMMANDS` table (`(/cmd, description)` pairs that drive the completer + `print_help`) and the `SHELL_ESCAPE_PREFIX` constant. The CLI host (`agent-cli::commands::interactive`) is the one that actually dispatches: built-in commands are `/help`, `/doctor`, `/providers`, `/provider`, `/skills`, `/tools`, `/model`, `/effort`, `/memory`, `/plan`, `/plans`, `/dump`, `/compact`, `/new`, `/retry`, `/cd`, `/sync`, `/mode`, `/exit`, plus `!<cmd>` shell-escape. Adding a slash command means appending one row to `SLASH_COMMANDS` AND adding a `match` arm in `commands/interactive.rs::handle_interactive_command` — the table alone is metadata; it does not dispatch. `commands/interactive.rs` also owns `SeedWorkspace { cwd }` (RF24-2) and the REPL-lifetime `CodexSession` (RF25-1): `/cd <path>` mutates `workspace.cwd` (and clears the RP bound-window cache), and the next `run_goal` propagates the new cwd to Codex (via `CodexSession::ensure` → hot-swap of `cfg.cwd`) and to RepoPrompt (via the lazy override consumption in `default_repoprompt_working_dirs`). `/new` calls `CodexSession::shutdown()` so the next prompt starts with a clean Codex slate.
- **Within-run tool memoization (RF35-2)**: `run_goal` wraps every planner tool call in a per-run cache. `is_memoizable_tool(name)` allowlists pure-read tools (`read_file`, `read_files`, `memory_*`, `skill_*`, `plan_status`/`plan_next`/`plan_list`, `tool_describe`, `repoprompt_tools`); side-effect tools always re-run. Cache key is `(name, canonical_json(args))`; a cache hit returns the previous `ToolResult` and emits a "(cached)" trace line. Saves real time when the planner re-reads the same file across turns.

- `agent-cli` — the `seed` binary; `clap` subcommands (`chat`, `run`, `doctor`, `tool`, `plan`, `reflect`, `replay`, `skill`, `providers`, `llm`, `codex`, `delegate`, `rp`). After RF1-RF9 the file structure is `main.rs` (486 lines: top-level `Cli`/`Command` enum + dispatch) + `commands/<verb>.rs` (one module per subcommand, each owns its own clap enum + `run_*` fn) + sibling modules `display.rs` (ANSI/format), `doctor.rs`, `plan_repoprompt.rs`. The planner loop lives in `commands/run.rs` and uses a `Planner` trait with `OraclePlanner` / `CodexPlanner` / `HttpPlanner` impls + a single `drive_planner_loop` helper — adding a new provider is "implement the trait + add a `PlannerProvider` variant". Run-control knobs live in `RunPolicy` (`max_turns`, `turn_timeout_secs`, `max_consecutive_failures` — RF13 borrowed forge's `ToolErrorTracker` so back-to-back tool failures abort the loop instead of burning the turn budget).

### Control flow (the part you need to understand to change behavior)

1. **Goal classification → RunMode (RF27).** `agent_runtime::is_read_only_analysis_goal` is still the keyword primitive (looks for analyze/summarize/explain incl. Chinese variants without implementation verbs); `agent_runtime::classify_run_mode(goal)` wraps it into `agent_core::RunMode::{ReadOnly,Implementation}`. `run_goal` resolves the effective mode from `RunGoalArgs.mode: ModeArg` (`Auto` → classifier, `Read`/`Write` → pinned) and stores `(mode, source)` in the `RunStarted` event and on `agent_tools::run_mode_guard` (process-singleton). Read-only runs (a) get a pared-down tool catalog via `planner_tool_infos_for_mode`, (b) auto-default Codex `reasoning_effort` to `"low"` when the user didn't pass `--effort` (RF26-2), (c) cause `ShellTool` to refuse write-shaped commands (RF27-2). (The previous synthesis-skip path was deleted in RF44-#6 — the schema lives in the planner prompt directly, no post-hoc rewrite needed.) User entry points: `--mode auto|read|write` CLI flag (Run + Chat) and `/mode auto|read|write` REPL slash. The trace prints `mode: read-only (auto-classified from goal)` (or `… (explicit via --mode)`) at run start so the user can tell which toolset the planner has access to.
2. **Planner prompt.** The runtime builds a prompt that always carries: L0 meta rules, the pointer-only `l1_insight.md`, a compact machine index, the registered `ToolInfo` set, and a GenericAgent-style `### [WORKING MEMORY]` anchor (recent turn summaries, `update_working_checkpoint` key facts, related skills, loop-guard hints when the planner is repeating itself or running out of turns). Deeper memory bodies are loaded only when the planner calls `memory_search` then `memory_fetch`.
3. **One typed call per turn.** The planner must reply with a single `PlannedAction` JSON object (`tool`+`args` or `finish`+`answer`, always with a `summary`). `parse_planned_action` enforces the contract AND runs a `sanity_check` pass (RF17) — blank `tool_name`, `args: null`, placeholder values (`TODO`/`TBD`), or empty `finish.answer` are rejected as `InvalidPlannerJson` with a specific reason so the retry path nudges the model to fix the shape instead of dispatching a doomed turn. The tool result and the summary become the next turn's working memory.
4. **Two-phase long-term memory.** `start_long_term_update` reads the SOP and shifts the planner into "settlement" — the *next* turn must call one of: a `patch_file`/`write_file` to L2 `global_facts.md`, a write to an existing L3 skill, or `complete_long_term_update skip`. After that branch, `complete_long_term_update` writes an auditable settlement event and L1/index are rebuilt automatically.
5. **Plan mode.** `plan_create` writes `plans/<id>/{plan.md,state.json}` plus a mandatory `[VERIFY]` checkbox. The loop is `plan_next` → do the step → `plan_complete`. RepoPrompt exports must be followed by `plan_record_artifact`; `agent_run`/Codex delegations must be followed by `plan_record_handoff` — the planner prompt treats this as protocol, and `plan_verify` records its verifier handoff automatically.
6. **Codex delegate.** `seed codex` and `seed run --codex` talk to `codex app-server` via stdio JSON-RPC. By default plugins are disabled, only the `RepoPrompt` MCP is allowed, and approvals are denied. Before sending, the CLI routes the prompt to one of the bundled RepoPrompt SOP skills (`repoprompt-deep-plan` for implement/refactor/plan, `repoprompt-review` for review, `repoprompt-investigate` for investigation) and inlines that skill body into the delegated prompt — RepoPrompt usage is explicit, not inferred from MCP availability.
7. **Learning.** `seed run --learn` consolidates a successful session into the skill tree: it appends `## Learned Updates` to the most-similar existing `SKILL.md` and only creates a new skill when no sufficiently similar one exists.
8. **~~Synthesis pass (RF26-1)~~ → removed in RF44-#6.** Previously a read-only `Finished` run fired one extra Codex turn (~60s) to rewrite the draft answer into the FINISH ANSWER SCHEMA. That schema **already lives in `planner_goal_guidance`** (agent-runtime/src/lib.rs:336-378), so the extra turn was a workaround for the planner ignoring its own system prompt — not a real semantic step. Deleted entirely: ~300 LOC + ~60s per read-only run. If quality regresses on a specific provider, the fix is to strengthen the prompt or add a finish-answer validator at the loop boundary, not to bring back a post-hoc rewrite.

### Conventions worth respecting when editing

- **Memory invariants.** Don't write to `memory/l1_insight.md` or `memory/index.json` from code paths other than `agent_memory::rebuild_index`. The tool layer enforces this; matching it in new code keeps the rebuild deterministic.
- **`AgentEvent` is the wire format.** Every observable side-effect (tool started/finished, turn summary, checkpoint, long-term-update lifecycle, reflection, run finish) goes through `AgentEvent` so it lands in the JSONL session and is replayable. Adding a new observable concept means extending the enum, not printing.
- **Planner JSON contract.** Any new tool needs a `ToolInfo` description that's specific enough for the planner to choose it correctly, and its `args` schema must be tolerant of the planner's habit of supplying string-encoded JSON (existing tools use `serde_json::from_value` with `ToolError::InvalidArguments` on failure).
- **Provider routing vs. delegate.** `run --llm` defaults to `provider = codex` (the local app-server) and does **not** require `OPENAI_API_KEY`. Only switch to an HTTP provider (`--provider openai|opencode|...`) when you explicitly want that path. `llm ask` is the raw HTTP smoke command and does read `OPENAI_API_KEY` / `OPENAI_BASE_URL`.
- **Approval & MCP defaults are intentional.** Codex is launched with `--disable plugins` and a minimal allowlist; changing the default to `--mcp all` or `AcceptForSession` materially expands blast radius.
- **Library crates expose typed errors** (RF10): `agent-plan`, `agent-memory`, `agent-session`, `agent-skills`, `agent-repoprompt`, plus the pre-existing `agent-core`/`agent-llm`/`agent-runtime`/`agent-delegate`. Public fns return `XxxResult<T> = Result<T, XxxError>`; each `XxxError` carries a small set of pattern-matchable variants (e.g. `PlanError::ItemNotFound { index }`, `SkillError::NotFound { name, skills_dir }`) plus an `Other(#[from] anyhow::Error)` escape hatch so internal `?` keeps working. Borrowed from forge's `forge_domain::Error` pattern — when adding a new library error, name the variants callers will want to switch on, leave everything else as `Other`.
- **Adding planner errors** (RF14): use `RuntimeError::Planner(_)` only for transient failures (network/transport/stdio hiccup that retrying might fix). Use `RuntimeError::PlannerFatal(_)` (or `RuntimeError::planner_fatal(msg)`) for permanent failures (auth rejection, model returned non-success response, runtime decided to give up). The runtime's retry loop only re-arms on `Planner` + `InvalidPlannerJson`.

### Architecture polish (RF40)

- **Thread-local sync state (A2)**: the three process-wide guards in
  `agent-tools::sync` (`skill_tools_guard`, `run_mode_guard`,
  `repoprompt_sync`) migrated from `OnceLock<Mutex<...>>` to
  `thread_local!{}`. Each test thread gets independent state; concurrent
  `run_goal` calls (e.g. embedded-as-library) are safe by construction.
  API surface unchanged.
- **Sync module extracted (A1 partial)**: the 3 sync submodules live in
  `agent-tools/src/sync.rs` instead of inline in `lib.rs`. lib.rs
  dropped 248 lines; tests + call sites unchanged via re-exports.
- **Planner extracted (A3 partial)**: `Planner` trait + Oracle / Codex /
  Http impls + `build_planner` live in
  `agent-cli/src/commands/run_planners.rs`. `run.rs` dropped 282 lines;
  MockPlanner stays in run.rs because it's a test fixture for the loop
  driver, not the trait.
- **Slash command groups (B3)**: `SLASH_COMMANDS` table carries a
  `SlashCategory` (`View`/`Configure`/`Operate`/`Exit`); `print_help`
  groups by category instead of one alphabetical wall. `/providers`
  (plural) is an alias for `/provider list` so muscle memory from
  pre-RF28 still works.

### Deferred (architectural debt, needs dedicated effort)

- **Per-tool file split in `agent-tools` (A1 deeper)** → **complete**.
  Nine modules now sit under `crates/agent-tools/src/`:
  - **RF43-A1a (files.rs)**: `read_file` / `read_files` / `patch_file` /
    `write_file`.
  - **RF43-A1b (memory_protocol.rs)**: `update_working_checkpoint` /
    `start_long_term_update` / `complete_long_term_update` + bundled
    SOP fallback.
  - **RF43-A1c (skills.rs)**: `skill_list` / `skill_search` /
    `skill_fetch` + the `queue_skill_repoprompt_binding` auto-bind
    helper.
  - **RF43-A1d (plan.rs)**: all 11 `plan_*` tools + plan-store /
    plan-mode-next-prompt / plan-ledger-summary helpers.
  - **RF43-A1e (ask_user.rs)**: interactive `ask_user` stdin prompt.
  - **RF43-A1f (tool_describe.rs)**: late-turn description-recovery.
  - **RF43-A1g (memory.rs)**: `memory_search` / `memory_fetch` + the
    `memory_paths` helper they share.
  - **RF43-A1h (shell.rs)**: `run_shell` + `shell_command_intent`
    classifier + `ShellIntent` enum + `read_pipe` helper.
  - **RF43-A1i (repoprompt_bridge.rs)**: `repoprompt_tools` /
    `repoprompt_exec` / `repoprompt_call` + `RepoPromptRoutingArgs`
    (re-exported from lib.rs as `pub(crate)` so plan.rs keeps its
    import path).

  Net: `lib.rs` 4410 → 1981 (-55%, -2429 lines). What remains in
  `lib.rs` is genuine shared infrastructure: `parse_tool_args` /
  `repair_tool_args`, the registry builder, the durable-write guard
  machinery (`DurableWriteMode`, `durable_write_guard`, `durable_target`,
  `is_durable_path`, …), the truncation utilities, RP-helpers cluster
  (`repoprompt_client`, `repoprompt_output_*`, `find_*_by_key`,
  `attach_repoprompt_protocol_hint`, `default_cwd_for_repoprompt_*`,
  `humanize_rp_bind_failure`) shared by both `plan.rs` and
  `repoprompt_bridge.rs`, and cross-module tests. Further splitting
  the RP-helpers cluster into its own module is possible (~600 LOC
  could move) but provides little code-locality benefit and adds
  import churn — the cluster is purely shared utilities, not a
  conceptual unit on its own.
- **Run phase split (A3 deeper)** → **substantially complete**.
  `run_goal` went 575 → 300 lines (-48%) via targeted helpers:
  - `resolve_and_announce_run_mode` (mode classification + guard +
    trace print, ~30 LOC).
  - `record_codex_fast_path_outcome` (codex `Result<CodexRunResult>`
    → session events + stdout + token-usage dim line, ~45 LOC).
  - `finalize_llm_run_outcome` + `FinalizeInputs<'a>` (post-loop
    TurnSummary/TurnTimings flush + Finished/MaxTurnsExceeded
    dispatch + archive append + `run --learn` consolidation,
    ~95 LOC). Returns `loop_result.turns` for the footer counter.
  - (The previous `apply_synthesis_pass_if_eligible` extraction
    became moot when RF44-#6 deleted the synthesis pass entirely.)

  What remains in `run_goal` (~340 lines): the args destructure,
  setup phase (cwd/effort/guards/memory paths/session start), the
  three-way provider dispatch (`use_codex` / `use_llm` / record-only),
  and the post-finalize footer (timing stats + run_turns header).
  These are now reasonably-sized chunks that read top-to-bottom as
  the high-level orchestration — further splitting would be
  cosmetic at best, since the remaining nesting is the genuine
  dispatch shape, not bookkeeping.
- ~~**Crate collapse 14→7 (B1)**~~ → **RF41-B1 shipped the easy half**:
  agent-tui + agent-session collapsed into `agent-core` (12 crates now).
  agent-tui's submodules moved to `agent-core/src/tui/`; agent-session
  became `agent-core/src/session.rs`. The provider-crate collapse
  (agent-llm + agent-delegate + agent-repoprompt → agent-providers) was
  evaluated and declined — the 3 crates have meaningfully different
  external dep footprints and merging them would force
  `agent_providers::codex::X` style verbose paths without real compile
  savings.

### RF53 — README, workflow evals, dogfood

The post-RF52 polish round. Three threads — one user-facing, one
infrastructural, one empirical:

- **RF53-A (shipped)**: `README.md` rewritten as the project's front
  door. 4 backend families × 9 provider IDs in a single table; install
  / quick start / memory model (L0–L4) / plan / skill / eval / bench
  / Codex knobs / REPL slash / architecture summary / status. Calls
  out what is intentionally **not** built (ToolCatalog enum, judge
  across all evals, prod observability) so future readers know which
  omissions are deliberate vs missing.
- **RF53-B (shipped)**: 3 new workflow evals + 2 turn-capture bug
  fixes in `eval-learn`. The new evals
  (`09_workflow_find_panics.toml`, `10_workflow_list_serde_structs`,
  `11_workflow_describe_tool_pattern`) target "find all X across the
  codebase" / "list things matching attribute Y" / "describe pattern
  Z" shapes — the kind of reusable recipe a learned skill could
  capture. **Bugs found+fixed in `last_session_turn_count`**:
    1. `serde_json::from_str(line).ok()?` returned `None` on the
       first malformed/empty line, making T1=T2=T3=0 even when the
       run finished — flipped to `let Ok(v) = ... else { continue }`.
    2. The function reads `store.last_session_path()` AFTER the
       judge has run; the judge shells out to `seed run` which writes
       its OWN session into the same `sessions/` dir, shadowing the
       eval's session. Fix: capture turn count INSIDE
       `run_one_eval_in_process` BEFORE `grade_answer` is invoked
       and stash it on `EvalOutcome.turns`. `run_eval_learn` now
       reads `outcome.turns` instead of re-querying the store.

  First real workflow data point:
  ```
  T1=2 T2=2 T3=2  (workflow_find_panics, codex backend)
  verdict: skill NEUTRAL
  ```
  Still neutral on 2-turn tasks — consistent with RF52-C's finding
  that `consolidate_run_skill` doesn't fire for trivial workflows.
  The shell-out eval path (`run_one_eval`) intentionally leaves
  `turns: None` because the subprocess uses its own session store
  the parent can't easily introspect — the in-process path is where
  eval-learn lives anyway.
- **RF53-C (shipped)**: dogfood run. `seed run --llm --provider
  codex --mode read "Find any TODO or FIXME comments in the agent-*
  crates and report them."` → 2 turns, 10.5s, answer "No TODO or
  FIXME comments found." Independent `rg` confirms 0 real markers
  (the 3 matches in the codebase are inside test fixture strings,
  not actual TODO/FIXME comments). **Friction observed**: the
  planner picked `--glob 'crates/agent-*/*'` which only matches one
  directory level deep — it got the right answer by luck (no test
  fixtures live at that depth either). A user-facing planner hint
  about recursive globs would help, but the dogfood task itself
  completed cleanly without intervention.

### RF52 — eval pool expansion + judge grading + --learn validation

After the project hit "architecture clean" status at RF51, the next
investment was to make the eval infrastructure good enough to actually
answer "is this backend / this prompt change / this feature an
improvement?" Three shipped pieces:

- **RF52-A (shipped)**: eval pool went 3 → 8 across 4 categories:
  factual lookup (regex), multi-step reasoning (regex), code citation
  (regex), free-form synthesis (judge), workflow choice (judge).
  Each `.toml` declares `kind = "regex"` or `kind = "judge"` so the
  grader picks the right path automatically.
- **RF52-B (shipped)**: LLM-as-judge grading. New `GradeSpec::Judge`
  variant: hands the agent's answer + a rubric to a separate backend
  (`--judge-provider`, default `repoprompt_oracle`) and parses
  `PASS`/`FAIL` from the first line. Lets evals grade free-form
  answers where regex would be either too lax or too strict. Same
  `grade_answer` helper used by both shell-out and in-process eval
  paths, so adding judge grading didn't fork the eval runner.
- **RF52-C (shipped)**: new `seed eval-learn <eval>` subcommand
  runs ONE eval 3× (baseline → with-learn → post-learn), parses
  turn counts from each session's `RunFinished` event, prints a
  comparison with a verdict (helped / hurt / neutral). First real
  data point on `--learn`:

  ```
  T1=2 T2=2 T3=2  (count_crates eval, codex backend)
  verdict: skill NEUTRAL
  ```

  Translation: for trivial tasks, `consolidate_run_skill` does NOT
  produce a skill (no new file appeared in `skills/`). So `--learn`
  is conservative-by-design — it won't generate noise for simple
  workflows. To actually validate whether `--learn` HELPS, evals
  need to target reusable multi-step workflows where the planner
  benefits from "here's the recipe" guidance. **Future work**:
  design 3-4 such evals (e.g. "refactor module X following pattern
  Y") and re-run `eval-learn` to see if T3 < T1 on those.

  Current `seed eval-learn` doesn't auto-clean up created skills —
  the trace tells the user which slug to `rm -rf`. CI integration
  would want a sandboxed `skills/` dir per run; v1 shares the
  project's real skills tree which is fine for local dev.

### RF51 — actionable findings round 2 (after 4th deep analysis)

- **RF51-#1 (shipped)**: `is_memoizable_tool`'s hardcoded 11-name
  match in run.rs moved onto the `Tool` trait as `is_pure_read() ->
  bool`. Adding a new read-tool now requires only
  `crate::impl_pure_read!()` in the tool's impl block — the property
  travels with the tool definition (mirrors the RF45-Phase1
  `impl_args_schema!` pattern). `ToolInfo` carries the new
  `is_pure_read: bool` field; run_goal's exec_tool closure builds
  the allowlist once via `pure_read_tool_names()` (registry lookup)
  and consults a BTreeSet per call. Closes the "OCP violation" from
  the analysis.
- **RF51-#3 (shipped)**: in-process eval runner. `seed eval
  --in-process` calls `run_goal` directly within the eval process
  instead of shelling out. Lets RF25-1's CodexSession reuse kick in
  (no per-eval Codex cold-start). Reads the final answer from the
  session JSONL (`run_finished` or `reflection` event). Default mode
  is still shell-out — `--in-process` is opt-in for speed.
  Smoke: 3-eval Codex suite went 24s (shell-out) → ~15s (in-process).
- **RF51-#2 (shipped, scope-reduced)**: SLASH_COMMANDS table /
  dispatch sync. The full forge-style trait-based dispatch would
  rewrite 19 match arms into 19 trait impls — substantial boilerplate
  for the small benefit of "compile-time prevents you from forgetting
  to dispatch a slash". Picked the smaller fix instead: added
  `HANDLED_SLASH_COMMANDS: &[&str]` const next to the dispatcher
  + two cross-check tests (`slash_table_and_dispatch_stay_in_sync`
  and `handled_slash_entries_are_table_entries_or_aliases`). Adding
  a slash command without updating both `SLASH_COMMANDS` (in
  agent-core::tui) and `HANDLED_SLASH_COMMANDS` (here) now fails
  `cargo test`. Net: ~30 LOC + 2 tests vs ~300 LOC for the full
  trait conversion, same bug-prevention property.

### RF50 — gitignore walker + eval suite + ToolCatalog deferral

- **RF50-A (shipped)**: gitignore-aware workspace walker via the
  `ignore` crate (same engine as ripgrep/fd). `agent_tools::walk_workspace`
  honors `.gitignore` / global gitignore / `.ignore` + skips hidden
  dirs + caps total files. **Integrated into `read_files`**: when an
  entry resolves to a directory, the tool expands via the walker (up
  to 12 files per dir, surfaces `expanded_from_dirs` in the response).
  Closes the "planner asks for `src/` and either errors out or gets
  `node_modules/*` noise" class. Old error-on-directory behavior is
  gone — directories silently expand.
- **RF50-B (shipped)**: minimal eval suite. `evals/*.toml` defines
  goals + regex grades; `seed eval --provider <id> --evals-dir evals`
  shells out to `seed run` for each, grades the stdout answer,
  prints PASS/FAIL + per-eval timing, exits non-zero on any failure.
  Three starter evals in `evals/`: `count_crates`, `find_msrv`,
  `naming_convention` (all read-only repo-shape questions).

  **First real backend comparison numbers** (mid-2026):

  | provider | result | total time | per-eval avg |
  |---|---|---|---|
  | codex (local app-server) | 3/3 PASS | 24 s | 8 s |
  | repoprompt_agent (explore role, default) | 3/3 PASS | 53 s | 18 s |

  Both backends correctly answer these simple questions; Codex is
  ~2× faster on questions whose ground truth is in the repo (no
  network round-trip beyond the LLM itself). RP Agent's curated
  context didn't help here because the eval questions are answerable
  in one local tool call — the curated context premium pays off
  more for questions that need cross-file synthesis.

  **Design**: shell-out, not in-process. Tradeoffs:
  - Pro: zero coupling to `run_goal` internals; each eval is an
    isolated process so thread-local state can't leak.
  - Pro: stdout-grading matches what a user sees.
  - Con: 100-200ms spawn overhead per eval; acceptable for a CI suite.

  **v1 grading**: regex match only. Future: `kind = "rubric"` or
  `kind = "llm_judge"` for free-form answers. v1 is enough to catch
  "did this PR break the existing answer shape" regressions.
- **RF50-C (documented, not built)**: Forge's `ToolCatalog` single-enum
  + `#[serde(tag = "name", content = "arguments")]` dispatch is a real
  architectural improvement but only pays off when you wire the tool
  catalog to a native function-calling protocol (OpenAI `tool_calls`,
  Anthropic `tool_use`). SeedAgent's current dispatch goes through
  `PlannedAction { Tool { tool_name, args } | Finish { answer } }`
  which the planner emits as plain JSON — the LLM doesn't need to know
  the schema is a tagged enum. Collapsing 32 trait impls into a
  single enum would touch every dispatch site for no behavior change.
  **Defer until**: we add a backend that consumes
  `tools: [{name, parameters_schema}, …]` as a native API parameter
  (i.e., GPT/Claude native tool-use mode). At that point, having the
  catalog be one serde-tagged enum makes the round-trip
  `LLM → tool_calls → ToolCatalog::deserialize → Tool::execute`
  type-safe end-to-end.

### RF49 — actionable findings from deep analysis

After tracing the full `seed run --llm` flow end-to-end, three real
issues surfaced and were addressed (one was a false alarm, kept the
note so future readers don't re-investigate):

- **RF49-D (shipped, real bug)**: in `run_goal`'s `use_llm` branch,
  the cache-hit path called `failure_streak.set(0)`. That meant a
  planner stuck in `"fail-fail-fail-recheck-cached-fail"` could
  cloak its lack of progress with cached reads and never trip
  `max_consecutive_failures` abort. Fix: cache hits no longer touch
  `failure_streak`. Regression test
  `cache_hits_do_not_reset_failure_streak` in run.rs verifies the
  property with an interleaved fail/cache/fail pattern.
- **RF49-A (false alarm)**: I suspected `RepoPromptAgent` would
  stay bound to the original cwd after `/cd` in the REPL — turned
  out the REPL rebuilds the planner from scratch on every
  `run_goal` invocation (only Codex needs `set_cwd` because
  `CodexSession` is REPL-shared; RepoPromptAgent has no shared
  client). Closed as not-a-bug; kept this note so a future reviewer
  who has the same suspicion doesn't re-litigate.
- **RF49-B (shipped)**: added `crates/agent-bench` — a criterion
  harness for the perf claims accumulated across RF33/35/42/45/46.
  Run with `cargo bench -p agent-bench`. Without numbers, every
  "optimization" was faith-based and the next refactor that broke
  one of them would have gone unnoticed.

  **Baseline numbers** (M-series Mac, mid-2026):

  | bench | time | what it confirms |
  |---|---|---|
  | `seed_registry/cached_access` | 64 ns | RF42-A3 OnceLock is essentially a pointer deref |
  | `schema/small_args` (2 fields) | 1.3 µs | schemars derive cost real but negligible (~30µs/turn at 32 tools) |
  | `schema/large_args` (10 fields) | 5.0 µs | linear in field count |
  | `json_repair/clean_passthrough` | 218 ns | RF46-A common-case overhead <1% |
  | `json_repair/strip_fence` | 258 ns | ` ```json ` strip |
  | `json_repair/strip_trailing_comma` | 159 ns | `,}` / `,]` strip |
  | `memoize_key/canonical_json` | 75 ns | RF35-2 lookup-key build is free |
  | `loop_state/from_1_observations` | 460 ns | |
  | `loop_state/from_5_observations` | 2.2 µs | linear |
  | `loop_state/from_20_observations` | 8.8 µs | scales linearly; still negligible vs network |

  **Implication**: every "this is cheap" claim in RF33-RF46 holds.
  No hidden hotspots in the metadata-handling layers. The dominant
  cost in any real `seed run` is network/LLM time (seconds to minutes
  per turn), not these microsecond-scale internal steps. A future
  refactor that, say, accidentally drops the OnceLock cache and goes
  back to 31 Box allocations per call would show up as
  `seed_registry/cached_access` jumping from 64ns to ~50µs+ —
  catchable with one bench re-run.

### RF48 — RepoPrompt agent_run as planner backend

Adds `PlannerProvider::RepoPromptAgent` — uses RepoPrompt's `agent_run`
tool (full Agent Mode) as a planner backend, alongside the existing
`Oracle` (`ask_oracle` one-shot Q&A) path.

**Usage**: `seed run --llm --provider repoprompt_agent --model engineer
"your task"`. Aliases: `repoprompt_agent` / `rp_agent` / `rp-agent`.
Role labels selected via `--model`: `explore` (fast read), `engineer`
(balanced impl), `pair` (highest tier, default), `design` (writes
review markdown). RepoPrompt's `model_id` parameter resolves these
through the global role-default mapping.

**Key implementation detail**: `agent_run` REQUIRES a persistent
`bind_context` with the `window_id` pinned — `--working-dir` flags
work for `ask_oracle` but not for `agent_run` (RP rejects with
"Multiple RepoPrompt windows detected"). The planner constructor
does a bind→extract window_id→pin pattern, mirroring
`agent-tools::resolve_repoprompt_window` inline (can't reuse — that
helper is `pub(crate)` to agent-tools).

**Session continuity**: first turn → `agent_run start` saves
`session_id`, subsequent turns → `steer` on the same session. The
agent keeps its own conversation context across turns.

**Smoke verified**: "crates 目录里有几个 rust crate" → role=explore
answered correctly in 1 turn / 12.67s with no tool calls — RP's
curated workspace context had the answer pre-loaded.

### RF47 — tool descriptions in markdown files

All 32 tool descriptions live in `crates/agent-tools/descriptions/<name>.md`
instead of inline `fn description() { "..." }` strings. Wiring:

```rust
impl Tool for ReadFileTool {
    fn name(&self) -> &'static str { "read_file" }
    crate::tool_description!("read_file");  // include_str! at compile time
    crate::impl_args_schema!(ReadFileArgs);
    fn execute(&self, ctx, call) -> ... { ... }
}
```

**Why a `macro_rules!` instead of forge's `#[tool_description_file]`
proc-macro**: we don't ship a `ToolDescription` trait that needs
auto-deriving — just want the file-on-disk editability. `include_str!`
inside a small macro is enough, zero new deps.

**Convention**: `.md` files MUST NOT end with a trailing newline.
A test in `lib.rs` (`every_tool_has_a_clean_description`) catches
violations + missing-file + stub-length issues at `cargo test` time.

**Why this matters**: prompt-engineering iteration on tool descriptions
no longer requires scrolling through 32 trait impls. Reviewers diff
the `.md` files in isolation. A typo fix is a 2-character edit in a
3-line file, not a 200-char string-literal edit in a 60-line impl
block.

### RF46 — engineering-hygiene maturity

Three smaller-scope additions that close real maturity gaps vs forge:

- **RF46-A (shipped)**: planner JSON repair pre-pass. `parse_planned_action`
  now tries strict-parse first, then on failure runs `repair_planner_json`
  and retries. The repair handles three common LLM glitches:
  (a) markdown code fences (` ```json … ``` `), (b) trailing commas
  before `}`/`]`, (c) `//` line comments. String-safe (won't touch
  commas inside string literals; tracks backslash-aware `in_string`
  state). 10 unit tests + 2 integration tests prove the repair
  shape. Real LLM output that previously cost a retry turn now
  parses on the first try.
- **RF46-B (shipped)**: `rust-toolchain.toml` (pins `channel = "1.95"`),
  `rustfmt.toml` (mirrors forge's style settings), `clippy.toml`
  (slightly raised cognitive-complexity + arg-count thresholds to
  match real seed code). Closes the "every dev's setup is different"
  gap. The toolchain pin is enforced — auto-installs 1.95 on first
  `cargo` invocation.
- **RF46-C (shipped)**: `.github/workflows/ci.yml`. Runs build + test
  + clippy (advisory) + rustfmt (advisory) on push/PR to `main`.
  Caches `~/.cargo` + `target` keyed on `Cargo.lock`. Stops PRs
  that break the workspace from merging — until now, "did this
  PR break tests" had to be manually re-checked. Clippy + rustfmt
  are advisory because the codebase has accumulated style warnings
  through 45 RFs; flipping them to blocking is a separate dedicated
  cleanup pass.

### RF45 — forge-parity maturity upgrade

After studying `ref/forgecode/` for "what makes forge mature", three
gaps stood out and were closed in three phases:

- **RF45-Phase1 (shipped)**: schema-driven tool dispatch. Every
  `*Args` struct in `agent-tools` now derives `schemars::JsonSchema`;
  `ToolInfo` carries an optional `args_schema: serde_json::Value`;
  `planner_request_with_state_and_memory` renders a compact `{path!:
  string, start?: integer}`-style summary inline in the tool catalog.
  A `impl_args_schema!` macro at the crate root makes per-tool wiring
  a single line. **Forge parity**: matches forge's
  `ToolDefinition.input_schema: schemars::Schema` shape. **Knock-on
  effect**: the RF38-2 serde-alias workaround class (planner guessing
  field names) is now redundant — the schema names every accepted
  field including aliases, so planners that read schemas will pick
  correct names on the first try.
- **RF45-Phase2 (shipped)**: `Planner` trait decoupled from `Spinner`.
  Previously the trait took `&agent_core::tui::Spinner` as a parameter
  on `plan()` AND had an `on_turn_start(spinner)` method — each impl
  reached into the UI layer directly. Now the trait emits
  `ProgressEvent::{StaticSubtitle, StreamingTokens}` via an
  `on_progress: &mut dyn FnMut(ProgressEvent)` callback; the driver
  (`drive_planner_loop`) translates events into spinner calls. The
  `last_prompt_chars()` accessor is gone — `prompt_chars` is now a
  field on the new `PlanOutput` return type. Each Planner impl is
  now pure backend logic, testable without a spinner.
- **RF45-Phase3 (shipped)**: pruned `_with_binding` accretion in
  `agent-skills`. Deleted `create_skill` (0 callers) and
  `consolidate_skill` (only 2 test callers, migrated to
  `_with_binding(.., None)`). Same accretion pattern as RF44-#1's
  `agent-runtime` cleanup — solved the same way.

### RF44 — design-review cleanups

After a deep architectural review (mid-2026), 6 issues were identified.
Final disposition after attempting each:

- **RF44-#6 (shipped)**: deleted the entire synthesis pass. ~300 LOC +
  ~60s per read-only run gone. The FINISH ANSWER SCHEMA was already in
  `planner_goal_guidance` — the post-hoc rewrite was a workaround for
  the planner ignoring its own system prompt, not a real semantic
  step. If quality regresses on a specific provider, the fix is to
  add a finish-answer validator at the loop boundary, not to bring
  back a post-hoc rewrite.
- **RF44-#1 (shipped)**: pruned 8 dead/redundant public fns from
  `agent-runtime` (`run_agent_loop`, `run_agent_loop_with_planner`,
  `plan_one_tool_call`, `plan_next_action`,
  `plan_next_action_with_observations_and_memory`,
  `plan_next_action_with_state`, `plan_next_action_with_state_and_memory`,
  `planner_request`, `planner_prompt_with_observations`). Demoted 4
  test-convenience wrappers to `pub(crate)`. Net: 2501 → 2390 lines.
  External surface now matches the 3 fns actually called from
  `agent-cli`: `planner_prompt_with_state_and_memory`,
  `planner_request_with_state_and_memory`,
  `run_agent_loop_with_state_planner_observed`.
- **RF44-#3 (moot)**: `Planner::send_freeform` was proposed to support
  the synthesis pass's per-provider re-dispatch. Synthesis pass is
  gone (RF44-#6), so the method has no consumer.
- **RF44-#2 (partial)**: deleted dead `ToolContext::new` constructor (0
  callers; only `with_cwd` and `with_paths` are used). Adding
  `ctx.memory_paths()` would force a new agent-memory→agent-core
  dependency for a 10-line clone-elimination — declined as net
  negative. `scaled_default` stays on `ToolContext` since its 3
  callers live in different modules (would otherwise need a shared
  helper crate).
- **RF44-#4 (declined)**: layering `AgentEvent` into
  `Lifecycle/Planner/Tool/Memory` sub-enums would break the JSONL
  wire format. No external consumer (no replay-as-a-service, no
  external SDK) is asking for the typed grouping. Cost (breaking
  stability) exceeds value (replay/reflect get cleaner match arms).
- **RF44-#5 (declined on re-examination)**: the original critique
  claimed `WorkingMemory.history` / `guard_hints` / `earlier_summary`
  duplicate `observations`. On closer reading, the dual storage is
  intentional: a session JSONL snapshot includes the full working
  memory state so `seed replay` can render it directly without
  re-running `from_observations`. Splitting derived-vs-owned fields
  in the type would force replay to do the recomputation. Keeping
  current shape.

### Recently shipped (formerly deferred)

All items originally considered in RF24–RF29 are now implemented (RF24–RF42).
The deferred-by-design list is empty — what used to be there:

- **`ToolRegistry` rebuilt 31 boxes per invocation** → RF42-A3:
  `seed_registry()` now returns `&'static ToolRegistry` backed by a
  `OnceLock`. Built once per process, shared across the planner loop,
  `seed exec`, REPL `/tools`, and the `tool_describe` planner tool. All
  consumer methods (`names`, `infos`, `execute`) already took `&self`,
  so call sites were return-type-only migrations.
- **Production planner path paid for `Box<dyn Planner>` indirection** →
  RF42-A1: `build_planner` now returns a concrete `PlannerKind<'a>`
  enum (`Oracle | Codex | Http`) that implements `Planner` via match
  dispatch. The trait still exists for `MockPlanner` test substitution
  — the indirection moved off the hot path. `&mut planner` coerces to
  `&mut dyn Planner` for `drive_planner_loop`, which is unchanged.

- **Codex daemon mode** → RF33-4: `--use-daemon` flag + `seed codex-daemon
  start|stop|status` subcommand. Launches via `codex app-server proxy`
  (connecting to running daemon) instead of stdio app-server.
  `CodexLaunchFingerprint` includes `use_daemon` so session reuse splits
  correctly when the flag flips.
- **Skill autobind sticky** → RF33-2: opt-in `repoprompt_sticky_cwd: true`
  in skill frontmatter. When set, `queue_skill_repoprompt_binding` ALSO
  queues a sticky cwd change via `repoprompt_sync::set_pending_sticky_cwd`;
  REPL polls between turns via `poll_sticky_cwd_into_workspace` and
  applies to `workspace.cwd` + the cached Codex client. Default false
  preserves the transient (RF24-4) behavior.
- **Per-turn tool-description culling** → RF33-3: `planner_request_with_state_and_memory`
  ships full descriptions on turns 1–4, names-only on turn 5+ with a hint
  pointing at the new `tool_describe` planner tool. The planner can call
  `tool_describe {name: "..."}` to recover any description it forgot.
- **Cross-turn memory-context cache** → RF33-1: `agent_memory::planner_memory_context`
  caches its output keyed on a tuple of mtimes
  (meta_rules/l1_insight/global_facts/index/skills_dir). Mtime change →
  cache miss → rebuild. `reset_planner_memory_cache` is exposed for tests.
- **HttpPlanner streaming** → RF32: `ProviderClient::chat_streaming` posts
  with `stream: true` and parses OpenAI Responses SSE events. HttpPlanner
  uses it with the same spinner subtitle callback as Codex.

### Environment variables

- `OPENAI_API_KEY` — required only for HTTP OpenAI provider paths (`llm ask`, `run --llm --provider openai`).
- `OPENAI_BASE_URL` — substituted into the `openai_compatible` / `openai_responses_compatible` endpoint templates.
- `REPOPROMPT_CLI` — override the RepoPrompt CLI path.
- `CODEX_HOME` — override `~/.codex` for MCP config discovery.
- `SEED_AGENT_CODEX_STDERR` — when set, forwards the Codex app-server's stderr to the terminal (useful when debugging delegate launches).

### Where things live at runtime

- `sessions/` — per-run JSONL traces; `seed reflect` / `seed replay` operate on the most recent unless one is named.
- `skills/<slug>/SKILL.md` — L3 skills, including the three bundled `repoprompt-*` SOPs that the Codex delegate inlines.
- `memory/` — L0/L1/L2 plus the rebuilt `index.json` and the long-term `session_archive.jsonl`.
- `plans/<id>/` — durable plan state (created by `plan_create`, never hand-edited; mutate via the `plan_*` tools so timestamps and orchestration ledgers stay consistent).
