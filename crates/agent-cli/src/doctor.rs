//! `seed doctor` + `seed providers` output. Both are read-only diagnostics —
//! no LLM, no state mutation. Lives in its own module so the main binary
//! entry point stays focused on dispatch.

use std::env;
use std::path::Path;

use agent_llm::{ModelId, ProviderRouter};
use agent_session::SessionStore;
use anyhow::Result;

pub(crate) fn doctor(skills_dir: &Path, store: &SessionStore) -> Result<()> {
    let registry = agent_tools::seed_registry();
    println!("seed doctor");
    println!("- cwd: {}", env::current_dir()?.display());
    println!("- sessions: {}", store.root().display());
    println!("- skills: {}", skills_dir.display());
    println!("- tui: {}", agent_tui::status());
    println!(
        "- repoprompt: {}",
        agent_repoprompt::default_cli_path().display()
    );
    println!("- tools: {}", registry.names().join(", "));
    println!(
        "- providers: {}",
        agent_llm::built_in_providers()
            .iter()
            .map(|provider| provider.id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("- delegates: codex-app-server, repoprompt-oracle");
    Ok(())
}

pub(crate) fn show_providers(
    provider_id: &str,
    model: Option<&str>,
    as_json: bool,
) -> Result<()> {
    let providers = agent_llm::built_in_providers();
    if as_json {
        println!("{}", serde_json::to_string_pretty(&providers)?);
        return Ok(());
    }

    println!("providers");
    println!("- codex local-app-server (default planner; uses local Codex login, no API key)");
    println!(
        "- repoprompt_oracle (opt-in: --provider repoprompt_oracle; planner goes through RepoPrompt ask_oracle so prompts inherit RepoPrompt's curated context; --model selects oracle mode: chat|plan|edit|review)"
    );
    for provider in &providers {
        let models = provider
            .models
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>();
        println!(
            "- {} {:?} {}",
            provider.id.as_str(),
            provider.response,
            if models.is_empty() {
                "(no built-in models)".to_string()
            } else {
                models.join(", ")
            }
        );
    }

    let Some(provider) = providers
        .iter()
        .find(|provider| provider.id.as_str() == provider_id)
    else {
        if provider_id == agent_llm::ProviderId::REPOPROMPT_ORACLE
            || provider_id == "repoprompt"
        {
            println!("route: repoprompt_oracle uses the local RepoPrompt CLI (no HTTP route);");
            println!("       seed run --provider repoprompt_oracle sends each planner turn via");
            println!("       `repoprompt_cli --call oracle_send` and keeps chat_id across turns.");
        } else if provider_id == agent_llm::ProviderId::CODEX {
            println!("route: codex uses the local app-server transport (no HTTP route).");
        } else {
            println!("route: provider {provider_id} not found");
        }
        return Ok(());
    };
    let model = model
        .map(ModelId::from)
        .or_else(|| provider.models.first().map(|model| model.id.clone()))
        .unwrap_or_else(|| ModelId::from("gpt-5.1"));
    let route = ProviderRouter.route(provider, &model);
    let transformed = agent_llm::default_pipeline()
        .transform(provider, agent_llm::ChatRequest::user(model, ""));

    println!("route");
    println!("- provider: {}", provider.id.as_str());
    println!("- backend: {:?}", route.response);
    println!("- endpoint: {}", route.endpoint);
    println!(
        "- transforms: {}",
        agent_llm::default_pipeline().names().join(", ")
    );
    println!(
        "- options: {}",
        serde_json::to_string(&transformed.options)?
    );
    Ok(())
}
