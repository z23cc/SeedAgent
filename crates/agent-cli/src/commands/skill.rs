//! `seed skill {create,list,search,fetch}`: durable skill tree CRUD around
//! `agent_skills`. Skill creation captures the current RepoPrompt binding so
//! `skill_fetch` can re-bind the workspace later — see
//! `query_current_repoprompt_binding`.

use std::path::Path;

use agent_session::SessionStore;
use anyhow::Result;
use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub(crate) enum SkillCommand {
    Create {
        #[arg(long)]
        name: String,
        session: Option<String>,
    },
    List {
        #[arg(long)]
        json: bool,
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    Search {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    Fetch {
        name: String,
    },
}

pub(crate) fn run_skill(
    command: SkillCommand,
    skills_dir: &Path,
    store: &SessionStore,
) -> Result<()> {
    match command {
        SkillCommand::Create { name, session } => {
            let records = store.read(session.as_deref())?;
            let binding = agent_skills::query_current_repoprompt_binding();
            let path = agent_skills::create_skill_with_binding(
                skills_dir,
                &name,
                &records,
                binding.as_ref(),
            )?;
            println!("created skill: {}", path.display());
            Ok(())
        }
        SkillCommand::List { json, limit } => {
            let skills = agent_skills::list_skill_infos(skills_dir)?;
            let skills = skills.into_iter().take(limit).collect::<Vec<_>>();
            if json {
                println!("{}", serde_json::to_string_pretty(&skills)?);
            } else {
                print_skill_infos(&skills);
            }
            Ok(())
        }
        SkillCommand::Search { query, limit, json } => {
            let skills = agent_skills::search_skill_infos(skills_dir, &query, limit)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&skills)?);
            } else {
                print_skill_infos(&skills);
            }
            Ok(())
        }
        SkillCommand::Fetch { name } => {
            let skill = agent_skills::fetch_skill(skills_dir, &name)?;
            println!("{}", skill.body);
            Ok(())
        }
    }
}

pub(crate) fn print_skill_infos(skills: &[agent_skills::SkillInfo]) {
    if skills.is_empty() {
        println!("skills: none");
        return;
    }
    println!("skills");
    for skill in skills {
        println!(
            "- {} [{}] {}",
            skill.name,
            skill.tags.join(", "),
            skill.description
        );
        println!("  path: {}", skill.path.display());
    }
}
