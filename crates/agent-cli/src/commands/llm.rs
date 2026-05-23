//! `seed llm ask`: raw HTTP smoke command. Reads OPENAI_API_KEY /
//! OPENAI_BASE_URL via the provider router; no session state.

use agent_llm::{ChatRequest, ProviderClient};
use anyhow::Result;
use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub(crate) enum LlmCommand {
    Ask {
        prompt: String,
        #[arg(long, default_value = "openai")]
        provider: String,
        #[arg(long, default_value = "gpt-5.1")]
        model: String,
        #[arg(long)]
        system: Option<String>,
        #[arg(long)]
        effort: Option<String>,
        #[arg(long)]
        max_output_tokens: Option<u32>,
        #[arg(long)]
        raw: bool,
    },
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_llm_ask(
    prompt: String,
    provider_id: String,
    model: String,
    system: Option<String>,
    effort: Option<String>,
    max_output_tokens: Option<u32>,
    raw: bool,
) -> Result<()> {
    let provider = agent_llm::find_provider(&provider_id)
        .ok_or_else(|| anyhow::anyhow!("provider not found: {provider_id}"))?;
    let mut request = ChatRequest::user(model, prompt);
    if let Some(system) = system {
        request
            .messages
            .insert(0, agent_llm::ChatMessage::system(system));
    }
    request.reasoning_effort = effort;
    request.max_output_tokens = max_output_tokens;

    let response = ProviderClient::new().chat(provider, request)?;
    if raw {
        println!("{}", serde_json::to_string_pretty(&response.raw)?);
    } else {
        println!("{}", response.text);
    }
    Ok(())
}
