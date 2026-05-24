use std::fmt::Write;

#[derive(Debug, Default, Clone)]
pub struct Info {
    sections: Vec<Section>,
}

#[derive(Debug, Clone)]
struct Section {
    title: Option<String>,
    pairs: Vec<(String, String)>,
}

impl Info {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn section(mut self, title: impl Into<String>) -> Self {
        self.sections.push(Section {
            title: Some(title.into()),
            pairs: Vec::new(),
        });
        self
    }

    pub fn pair(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if self.sections.is_empty() {
            self.sections.push(Section {
                title: None,
                pairs: Vec::new(),
            });
        }
        let last = self.sections.last_mut().expect("section just pushed");
        last.pairs.push((key.into(), value.into()));
        self
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        for (index, section) in self.sections.iter().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            if let Some(title) = &section.title
                && !title.is_empty()
            {
                let _ = writeln!(out, "{}", title.to_uppercase());
            }
            let key_width = section
                .pairs
                .iter()
                .map(|(key, _)| key.chars().count())
                .max()
                .unwrap_or(0);
            for (key, value) in &section.pairs {
                let pad = key_width.saturating_sub(key.chars().count());
                let _ = writeln!(out, "  {key}{:pad$}  {value}", "", pad = pad);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_right_aligned_keys_within_section() {
        let info = Info::new()
            .section("Summary")
            .pair("turns", "6")
            .pair("elapsed", "1m41s")
            .pair("session", "sessions/abc.jsonl");
        let rendered = info.render();
        assert!(rendered.contains("SUMMARY\n"));
        assert!(rendered.contains("  turns    6\n"), "got: {rendered:?}");
        assert!(
            rendered.contains("  elapsed  1m41s\n"),
            "got: {rendered:?}"
        );
        assert!(
            rendered.contains("  session  sessions/abc.jsonl\n"),
            "got: {rendered:?}"
        );
    }

    #[test]
    fn empty_section_title_is_skipped() {
        let info = Info::new().pair("k", "v");
        let rendered = info.render();
        assert_eq!(rendered, "  k  v\n");
    }

    #[test]
    fn multiple_sections_separated_by_blank_line() {
        let info = Info::new()
            .section("A")
            .pair("k", "v")
            .section("B")
            .pair("kk", "vv");
        let rendered = info.render();
        assert_eq!(rendered, "A\n  k  v\n\nB\n  kk  vv\n");
    }
}
