//! Micro-benchmarks for performance-sensitive helpers. Run with
//! `cargo bench -p agent-bench`. HTML reports land in
//! `target/criterion/`. These measure CPU cost in isolation, not
//! end-to-end planner-turn latency (which is dominated by
//! network/LLM time).

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

// ToolRegistry OnceLock cache: <10ns per call after first invocation.
fn bench_seed_registry_cached(c: &mut Criterion) {
    // Prime the OnceLock so we measure cached access, not init.
    let _ = agent_tools::seed_registry();
    c.bench_function("seed_registry/cached_access", |b| {
        b.iter(|| {
            let registry = agent_tools::seed_registry();
            std::hint::black_box(registry.names().len())
        });
    });
}

// schemars-derived args schema runs at every infos() call (no per-type
// cache). Establish whether memoizing it would be worth the complexity.
fn bench_args_schema_generation(c: &mut Criterion) {
    use schemars::JsonSchema;

    #[derive(serde::Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct SmallArgs {
        path: String,
        start: Option<usize>,
    }

    #[derive(serde::Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct LargeArgs {
        paths: Vec<String>,
        start: Option<usize>,
        count: Option<usize>,
        keyword: Option<String>,
        show_line_numbers: Option<bool>,
        recursive: Option<bool>,
        max_depth: Option<usize>,
        follow_symlinks: Option<bool>,
        extra_field_1: Option<String>,
        extra_field_2: Option<String>,
    }

    c.bench_function("schema/small_args", |b| {
        b.iter(|| {
            let schema = agent_core::tool_args_schema::<SmallArgs>();
            std::hint::black_box(schema);
        });
    });

    c.bench_function("schema/large_args", |b| {
        b.iter(|| {
            let schema = agent_core::tool_args_schema::<LargeArgs>();
            std::hint::black_box(schema);
        });
    });
}

// JSON repair runs only when strict parse fails — measure both the
// clean passthrough (common case) and the dirty-input cost.
fn bench_json_repair(c: &mut Criterion) {
    let clean = r#"{"summary":"investigated Cargo.toml","action":"tool","tool_name":"read_file","args":{"path":"Cargo.toml"}}"#;
    let fenced = format!("```json\n{clean}\n```");
    let trailing_comma = r#"{"summary":"x","action":"finish","answer":"y",}"#;

    let mut group = c.benchmark_group("json_repair");
    group.bench_function("clean_passthrough", |b| {
        b.iter(|| {
            let out = agent_runtime::repair_planner_json(std::hint::black_box(clean));
            std::hint::black_box(out);
        });
    });
    group.bench_function("strip_fence", |b| {
        b.iter(|| {
            let out = agent_runtime::repair_planner_json(std::hint::black_box(&fenced));
            std::hint::black_box(out);
        });
    });
    group.bench_function("strip_trailing_comma", |b| {
        b.iter(|| {
            let out = agent_runtime::repair_planner_json(std::hint::black_box(trailing_comma));
            std::hint::black_box(out);
        });
    });
    group.finish();
}

// run.rs's `memoize_key` is private; bench the underlying
// canonicalization via serde_json. Significant divergence from the
// real `memoize_key` is worth investigating.
fn bench_memoize_key_shape(c: &mut Criterion) {
    let args = serde_json::json!({
        "path": "crates/agent-tools/src/lib.rs",
        "start": 100,
        "count": 50,
        "keyword": "memoize",
        "show_line_numbers": true,
    });

    c.bench_function("memoize_key/canonical_json", |b| {
        b.iter(|| {
            let s = serde_json::to_string(std::hint::black_box(&args)).unwrap();
            std::hint::black_box(s);
        });
    });
}

// AgentLoopState::from_observations is O(observations) and runs each
// turn — verify it stays cheap on 15+ turn runs.
fn bench_loop_state_prepare(c: &mut Criterion) {
    use agent_core::{ToolCall, ToolResult};
    use agent_runtime::{AgentLoopState, ToolObservation};

    let mk_obs = |i: usize| ToolObservation {
        turn: i,
        summary: format!("turn {i} did some work and recorded a finding worth ~80 chars"),
        call: ToolCall::new("read_file", serde_json::json!({"path": format!("file_{i}.rs")})),
        result: ToolResult::ok(
            &ToolCall::new("read_file", serde_json::json!({})),
            serde_json::json!({"status": "success", "content": "x".repeat(500)}),
        ),
    };

    for n in [1usize, 5, 20] {
        let observations: Vec<ToolObservation> = (1..=n).map(mk_obs).collect();
        c.bench_function(&format!("loop_state/from_{n}_observations"), |b| {
            b.iter(|| {
                let state = AgentLoopState::from_observations(
                    std::hint::black_box(&observations),
                    n + 1,
                    24,
                );
                std::hint::black_box(state);
            });
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_secs(3))
        .warm_up_time(Duration::from_secs(1));
    targets =
        bench_seed_registry_cached,
        bench_args_schema_generation,
        bench_json_repair,
        bench_memoize_key_shape,
        bench_loop_state_prepare,
}
criterion_main!(benches);
