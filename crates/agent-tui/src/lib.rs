use std::borrow::Cow;
use std::path::PathBuf;

use anyhow::Result;
use reedline::{
    Completer, DefaultHinter, EditCommand, Emacs, FileBackedHistory, KeyCode, KeyModifiers, Prompt,
    PromptEditMode, PromptHistorySearch, PromptHistorySearchStatus, Reedline, ReedlineEvent,
    Signal, Span, Suggestion, default_emacs_keybindings,
};

mod info;
mod spinner;
mod style;

pub use info::Info;
pub use spinner::{Spinner, format_elapsed};
pub use style::{
    Status, dim_text, fast_elapsed, phase_divider, slow_elapsed, status_marker, tool_name,
};

const HISTORY_CAPACITY: usize = 10_000;
const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/help", "show commands"),
    ("/doctor", "print diagnostics"),
    ("/providers", "list provider routes"),
    ("/skills", "list local skills"),
    ("/tools", "list registered planner tools"),
    ("/model", "view or switch model (try /model list)"),
    ("/effort", "view or switch reasoning effort (low|medium|high|none)"),
    ("/memory", "search local memory index (/memory <query>)"),
    ("/plan", "show the active plan's next item + progress"),
    ("/plans", "list known plans"),
    ("/dump", "print path to the most recent session JSONL"),
    ("/compact", "rebuild the memory index from current L2/L3/L4 state"),
    ("/new", "start a fresh session for the next prompt"),
    ("/retry", "re-run the previous goal in a new session"),
    ("/cd", "change REPL workspace cwd (Codex + RepoPrompt follow)"),
    ("/exit", "leave interactive mode"),
];

/// Lines starting with `!` are shell-escapes (e.g. `!git status`), not slash
/// commands. Exposed so the REPL host can branch on it before the slash
/// dispatcher runs and so the completer can short-circuit cleanly.
pub const SHELL_ESCAPE_PREFIX: char = '!';

pub fn status() -> &'static str {
    "agent-tui reedline repl"
}

#[derive(Debug, Clone)]
pub struct PromptState {
    pub cwd: PathBuf,
    pub mode: String,
    pub provider: String,
    pub model: Option<String>,
}

impl PromptState {
    pub fn new(
        cwd: PathBuf,
        mode: impl Into<String>,
        provider: impl Into<String>,
        model: Option<String>,
    ) -> Self {
        Self {
            cwd,
            mode: mode.into(),
            provider: provider.into(),
            model,
        }
    }
}

impl Prompt for PromptState {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        let dir = self
            .cwd
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .unwrap_or(".");
        let mut header = format!("seed {dir} · {} · {}", self.mode, self.provider);
        if let Some(model) = self.model.as_deref() {
            header.push(' ');
            header.push_str(model);
        }
        header.push_str("\n> ");
        Cow::Owned(header)
    }

    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _prompt_mode: PromptEditMode) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        Cow::Borrowed("... ")
    }

    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> Cow<'_, str> {
        let status = match history_search.status {
            PromptHistorySearchStatus::Passing => "history",
            PromptHistorySearchStatus::Failing => "history?",
        };
        if history_search.term.is_empty() {
            Cow::Owned(format!("({status}) "))
        } else {
            Cow::Owned(format!("({status}: {}) ", history_search.term))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplInput {
    Line(String),
    Empty,
    Continue,
    Exit,
}

pub struct Repl {
    editor: Reedline,
}

impl Repl {
    pub fn new(history_path: PathBuf) -> Self {
        let mut keybindings = default_emacs_keybindings();
        keybindings.add_binding(
            KeyModifiers::CONTROL,
            KeyCode::Char('k'),
            ReedlineEvent::ClearScreen,
        );
        keybindings.add_binding(
            KeyModifiers::ALT,
            KeyCode::Enter,
            ReedlineEvent::Edit(vec![EditCommand::InsertNewline]),
        );

        let history = Box::new(
            FileBackedHistory::with_file(HISTORY_CAPACITY, history_path).unwrap_or_default(),
        );

        let editor = Reedline::create()
            .with_history(history)
            .with_completer(Box::new(SlashCommandCompleter))
            .with_hinter(Box::new(DefaultHinter::default()))
            .with_edit_mode(Box::new(Emacs::new(keybindings)))
            .with_ansi_colors(true)
            .use_bracketed_paste(true);

        Self { editor }
    }

    pub fn read(&mut self, prompt: &PromptState) -> Result<ReplInput> {
        Ok(self.editor.read_line(prompt).map(ReplInput::from)?)
    }
}

impl From<Signal> for ReplInput {
    fn from(signal: Signal) -> Self {
        match signal {
            Signal::Success(buffer) | Signal::ExternalBreak(buffer) => {
                let trimmed = buffer.trim();
                if trimmed.is_empty() {
                    ReplInput::Empty
                } else {
                    ReplInput::Line(trimmed.to_string())
                }
            }
            Signal::CtrlC => ReplInput::Continue,
            Signal::CtrlD => ReplInput::Exit,
            _ => ReplInput::Continue,
        }
    }
}

struct SlashCommandCompleter;

impl Completer for SlashCommandCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        if !line.starts_with('/') || line.contains(' ') {
            return Vec::new();
        }

        SLASH_COMMANDS
            .iter()
            .filter(|(command, _)| command.starts_with(line))
            .map(|(command, description)| Suggestion {
                value: (*command).to_string(),
                description: Some((*description).to_string()),
                style: None,
                extra: None,
                span: Span::new(0, pos),
                append_whitespace: true,
                match_indices: None,
                display_override: None,
            })
            .collect()
    }
}

pub fn print_banner() {
    println!("seed interactive · type a goal · /help for commands · Ctrl-D exits");
}

pub fn print_help() {
    println!("commands");
    // Driven off the SLASH_COMMANDS table so adding a new entry above
    // automatically shows up here too — prevents the "added /model in
    // table but forgot to update help" drift that bit us before.
    let max_cmd_len = SLASH_COMMANDS
        .iter()
        .map(|(cmd, _)| cmd.len())
        .max()
        .unwrap_or(0);
    for (cmd, desc) in SLASH_COMMANDS {
        println!("- {cmd:<width$}  {desc}", width = max_cmd_len);
    }
}

pub fn print_error(error: impl std::fmt::Display) {
    eprintln!("error: {error}");
}

#[cfg(test)]
mod slash_command_tests {
    use super::SLASH_COMMANDS;
    use std::collections::HashSet;

    #[test]
    fn slash_commands_have_unique_names() {
        let mut seen = HashSet::new();
        for (cmd, _) in SLASH_COMMANDS {
            assert!(seen.insert(*cmd), "duplicate slash command in table: {cmd}");
        }
    }

    #[test]
    fn slash_commands_all_start_with_slash() {
        for (cmd, _) in SLASH_COMMANDS {
            assert!(cmd.starts_with('/'), "slash command missing leading /: {cmd}");
        }
    }

    #[test]
    fn slash_commands_have_descriptions() {
        for (cmd, desc) in SLASH_COMMANDS {
            assert!(!desc.is_empty(), "missing description for {cmd}");
        }
    }
}
