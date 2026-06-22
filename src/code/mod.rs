//! Tree-sitter code-symbol indexing.
//!
//! Pure parsing layer: turns source files into [`ParsedFile`]s (definitions and
//! references) and provides a gitignore-aware directory walk. It has no
//! knowledge of the graph store — [`crate::actions::code`] maps the parsed
//! output onto entities/relations and handles incremental hashing.

pub mod lang;

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicUsize;

pub use lang::{Lang, detect};

/// Files larger than this are skipped (parsing huge generated/minified files
/// is slow and rarely useful for a symbol map).
pub const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

/// Signatures and doc lines are capped to keep observations compact.
const MAX_SIGNATURE_CHARS: usize = 240;
const MAX_DOC_CHARS: usize = 240;

/// A symbol definition extracted from a file.
#[derive(Debug, Clone)]
pub struct Def {
    /// Normalized kind: `function`, `method`, `class`, `module`, `constant`, …
    pub kind: String,
    /// Bare symbol name.
    pub name: String,
    /// 1-based inclusive line range of the definition.
    pub line_start: usize,
    pub line_end: usize,
    /// First (declaration) line of the definition, trimmed.
    pub signature: String,
    /// First line of the associated doc comment, if any.
    pub doc: Option<String>,
}

/// A reference (call / type use) extracted from a file.
#[derive(Debug, Clone)]
pub struct Ref {
    /// Reference kind from the grammar's tags query, e.g. `call`, `type`.
    pub kind: String,
    /// Bare name referenced.
    pub name: String,
    /// 1-based line of the reference.
    pub line: usize,
}

/// Parsed symbols for a single file.
#[derive(Debug, Clone, Default)]
pub struct ParsedFile {
    pub defs: Vec<Def>,
    pub refs: Vec<Ref>,
}

/// BLAKE3 content hash (hex) used for incremental change detection.
pub fn hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// Normalize a grammar's definition kind into our small entity vocabulary.
fn normalize_def_kind(raw: &str) -> &str {
    match raw {
        "function" | "macro" => "function",
        "method" => "method",
        "class" | "interface" | "struct" | "type" | "enum" | "trait" => "class",
        "module" | "namespace" => "module",
        "constant" => "constant",
        other => other,
    }
}

/// Extract the trimmed first line of `source` starting at byte offset `start`.
fn first_line(source: &[u8], start: usize) -> String {
    let end = source[start..]
        .iter()
        .position(|&b| b == b'\n')
        .map(|p| start + p)
        .unwrap_or(source.len());
    let mut s = String::from_utf8_lossy(&source[start..end]).trim().to_string();
    if s.chars().count() > MAX_SIGNATURE_CHARS {
        s = s.chars().take(MAX_SIGNATURE_CHARS).collect::<String>() + "…";
    }
    s
}

fn clamp_doc(doc: &str) -> Option<String> {
    let line = doc.lines().find(|l| !l.trim().is_empty())?.trim();
    if line.is_empty() {
        return None;
    }
    let s = if line.chars().count() > MAX_DOC_CHARS {
        line.chars().take(MAX_DOC_CHARS).collect::<String>() + "…"
    } else {
        line.to_string()
    };
    Some(s)
}

/// Parse one in-memory source buffer into defs/refs. Returns an empty result
/// for unsupported languages or unbuildable tag configs.
pub fn parse_source(lang: Lang, source: &[u8]) -> ParsedFile {
    let Some(config) = lang::config(lang) else {
        return ParsedFile::default();
    };

    let mut ctx = tree_sitter_tags::TagsContext::new();
    let cancel = AtomicUsize::new(0);
    let (tags, _failed) = match ctx.generate_tags(config, source, Some(&cancel)) {
        Ok(v) => v,
        Err(_) => return ParsedFile::default(),
    };

    // Byte offset of the start of each line, for O(log n) byte→line lookups.
    // `tag.range` spans the whole definition node (body included), while
    // `tag.span` is only the name; we derive the def's line range from `range`.
    let line_starts: Vec<usize> = std::iter::once(0)
        .chain(source.iter().enumerate().filter(|&(_, &b)| b == b'\n').map(|(i, _)| i + 1))
        .collect();
    let line_of = |byte: usize| line_starts.partition_point(|&s| s <= byte).max(1);

    let mut out = ParsedFile::default();
    for tag in tags.flatten() {
        let name = String::from_utf8_lossy(&source[tag.name_range.clone()]).to_string();
        if name.is_empty() {
            continue;
        }
        let kind = config.syntax_type_name(tag.syntax_type_id).to_string();
        if tag.is_definition {
            let end_byte = tag.range.end.saturating_sub(1).max(tag.range.start);
            out.defs.push(Def {
                kind: normalize_def_kind(&kind).to_string(),
                name,
                line_start: line_of(tag.range.start),
                line_end: line_of(end_byte),
                signature: first_line(source, tag.range.start),
                doc: tag.docs.as_deref().and_then(clamp_doc),
            });
        } else {
            out.refs.push(Ref {
                kind,
                name,
                line: tag.span.start.row + 1,
            });
        }
    }
    out
}

/// Walk `root` (a file or directory) and collect indexable source files,
/// honoring `.gitignore`/hidden-file rules and skipping oversized files.
pub fn walk(root: &Path, max_bytes: u64) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if root.is_file() {
        if detect(root).is_some()
            && std::fs::metadata(root).map(|m| m.len() <= max_bytes).unwrap_or(false)
        {
            files.push(root.to_path_buf());
        }
        return files;
    }

    let walker = ignore::WalkBuilder::new(root)
        .standard_filters(true)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .require_git(false)
        .filter_entry(|e| {
            // Belt-and-suspenders: skip common build/vendor dirs even when no
            // .gitignore is present.
            let name = e.file_name().to_string_lossy();
            !matches!(name.as_ref(), "target" | "node_modules" | ".git" | "dist" | "build")
        })
        .build();

    for entry in walker.flatten() {
        let path = entry.path();
        if !path.is_file() || detect(path).is_none() {
            continue;
        }
        if std::fs::metadata(path).map(|m| m.len() > max_bytes).unwrap_or(true) {
            continue;
        }
        files.push(path.to_path_buf());
    }
    files
}
