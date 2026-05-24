//! Gitignore-aware workspace walker built on the `ignore` crate
//! (ripgrep/fd's engine). Read-only, doesn't follow symlinks. The
//! `max_files` cap stops a planner from pulling a whole monorepo into
//! one prompt by asking for `read_files paths=["./"]`.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct WalkOptions {
    /// Hard cap on returned paths. Beyond this, `truncated=true`.
    pub max_files: usize,
    /// Absolute paths if true; relative to `root` if false.
    pub absolute: bool,
    pub include_hidden: bool,
    /// Files larger than this byte count are skipped.
    pub max_filesize: u64,
}

impl Default for WalkOptions {
    fn default() -> Self {
        Self {
            max_files: 200,
            absolute: false,
            include_hidden: false,
            max_filesize: 1_048_576,
        }
    }
}

/// `paths` is in depth-first walker order. `truncated` is true when
/// `max_files` was hit.
#[derive(Debug, Clone)]
pub struct WalkResult {
    pub paths: Vec<PathBuf>,
    pub truncated: bool,
}

/// `root` may be a file or a directory. Per-entry errors are skipped
/// rather than propagated, so a partial result is still useful.
pub fn walk_workspace(root: &Path, opts: &WalkOptions) -> WalkResult {
    let mut builder = ignore::WalkBuilder::new(root);
    builder
        .hidden(!opts.include_hidden) // hidden(true) means SKIP hidden
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .ignore(true)
        .parents(true)
        .follow_links(false);
    if opts.max_filesize > 0 {
        builder.max_filesize(Some(opts.max_filesize));
    }
    let walker = builder.build();

    let mut paths = Vec::new();
    let mut truncated = false;
    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if entry.file_type().is_none_or(|t| !t.is_file()) {
            continue;
        }
        let path = entry.path();
        let out_path = if opts.absolute {
            path.to_path_buf()
        } else {
            path.strip_prefix(root)
                .unwrap_or(path)
                .to_path_buf()
        };
        paths.push(out_path);
        if paths.len() >= opts.max_files {
            truncated = true;
            break;
        }
    }
    WalkResult { paths, truncated }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process::Command;

    fn temp_dir(test: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "seed-walk-{}-{}-{}",
            test,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn walks_a_single_file_root() {
        let root = temp_dir("single_file");
        let file = root.join("hello.txt");
        fs::write(&file, "hi").unwrap();
        let result = walk_workspace(&file, &WalkOptions::default());
        assert_eq!(result.paths.len(), 1);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn respects_gitignore_in_subdir() {
        let root = temp_dir("gitignore_subdir");
        // Needs a real .git boundary for the ignore crate to find gitignore.
        let _ = Command::new("git")
            .args(["init", "-q"])
            .current_dir(&root)
            .output();
        fs::write(root.join(".gitignore"), "target/\nignored.log\n").unwrap();
        fs::write(root.join("keep.rs"), "fn main() {}").unwrap();
        fs::write(root.join("ignored.log"), "noise").unwrap();
        fs::create_dir_all(root.join("target")).unwrap();
        fs::write(root.join("target/build.txt"), "should not appear").unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/lib.rs"), "// lib").unwrap();

        let result = walk_workspace(&root, &WalkOptions::default());
        let names: Vec<String> = result
            .paths
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        assert!(names.iter().any(|n| n.ends_with("keep.rs")), "got: {names:?}");
        assert!(names.iter().any(|n| n.ends_with("lib.rs")), "got: {names:?}");
        assert!(
            !names.iter().any(|n| n.contains("ignored.log")),
            "ignored.log should be filtered: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.contains("target")),
            "target/ should be filtered: {names:?}"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn truncates_at_max_files() {
        let root = temp_dir("truncate");
        for i in 0..10 {
            fs::write(root.join(format!("file_{i}.txt")), "x").unwrap();
        }
        let opts = WalkOptions {
            max_files: 3,
            ..Default::default()
        };
        let result = walk_workspace(&root, &opts);
        assert_eq!(result.paths.len(), 3);
        assert!(result.truncated);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn skips_hidden_dirs_by_default() {
        let root = temp_dir("hidden");
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::write(root.join(".git/HEAD"), "ref: refs/heads/main").unwrap();
        fs::write(root.join("real.rs"), "fn main() {}").unwrap();
        let result = walk_workspace(&root, &WalkOptions::default());
        let names: Vec<String> = result
            .paths
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        assert!(names.iter().any(|n| n.ends_with("real.rs")));
        assert!(
            !names.iter().any(|n| n.contains(".git")),
            "hidden .git should be skipped: {names:?}"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn skips_files_over_max_filesize() {
        let root = temp_dir("filesize");
        fs::write(root.join("small.txt"), "ok").unwrap();
        fs::write(root.join("big.bin"), vec![0u8; 2 * 1024 * 1024]).unwrap();
        let result = walk_workspace(&root, &WalkOptions::default());
        let names: Vec<String> = result
            .paths
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        assert!(names.iter().any(|n| n.ends_with("small.txt")));
        assert!(
            !names.iter().any(|n| n.ends_with("big.bin")),
            "files over max_filesize should be skipped: {names:?}"
        );
        let _ = fs::remove_dir_all(&root);
    }
}
