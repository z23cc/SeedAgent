//! CLI command implementations. Each submodule contains one `seed <verb>`
//! subcommand's logic; `main.rs` keeps only argument parsing and dispatch.

pub(crate) mod codex;
pub(crate) mod exec;
pub(crate) mod interactive;
pub(crate) mod llm;
pub(crate) mod plan;
pub(crate) mod replay;
pub(crate) mod rp;
pub(crate) mod run;
pub(crate) mod skill;
pub(crate) mod tool;
