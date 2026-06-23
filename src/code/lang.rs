//! Language registry for code-symbol indexing.
//!
//! Maps file extensions to a [`Lang`] and lazily builds the per-language
//! [`TagsConfiguration`] (from each grammar's bundled `tags.scm`). Building a
//! tags configuration compiles a tree-sitter query, so the configs are cached
//! in a process-wide `OnceLock` and shared across every parsed file.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use tree_sitter_tags::TagsConfiguration;

/// A source language we can extract symbols from.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Lang {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Tsx,
    Go,
    Java,
    C,
    Cpp,
    Ruby,
    Php,
}

impl Lang {
    /// Stable lowercase identifier stored in the graph (`lang:` observation).
    pub const fn name(self) -> &'static str {
        match self {
            Lang::Rust => "rust",
            Lang::Python => "python",
            Lang::JavaScript => "javascript",
            Lang::TypeScript => "typescript",
            Lang::Tsx => "tsx",
            Lang::Go => "go",
            Lang::Java => "java",
            Lang::C => "c",
            Lang::Cpp => "cpp",
            Lang::Ruby => "ruby",
            Lang::Php => "php",
        }
    }

    fn language(self) -> tree_sitter::Language {
        match self {
            Lang::Rust => tree_sitter_rust::LANGUAGE.into(),
            Lang::Python => tree_sitter_python::LANGUAGE.into(),
            Lang::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Lang::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Lang::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Lang::Go => tree_sitter_go::LANGUAGE.into(),
            Lang::Java => tree_sitter_java::LANGUAGE.into(),
            Lang::C => tree_sitter_c::LANGUAGE.into(),
            Lang::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Lang::Ruby => tree_sitter_ruby::LANGUAGE.into(),
            Lang::Php => tree_sitter_php::LANGUAGE_PHP.into(),
        }
    }

    const fn tags_query(self) -> &'static str {
        match self {
            Lang::Rust => tree_sitter_rust::TAGS_QUERY,
            Lang::Python => tree_sitter_python::TAGS_QUERY,
            Lang::JavaScript => tree_sitter_javascript::TAGS_QUERY,
            Lang::TypeScript | Lang::Tsx => tree_sitter_typescript::TAGS_QUERY,
            Lang::Go => tree_sitter_go::TAGS_QUERY,
            Lang::Java => tree_sitter_java::TAGS_QUERY,
            Lang::C => tree_sitter_c::TAGS_QUERY,
            Lang::Cpp => tree_sitter_cpp::TAGS_QUERY,
            Lang::Ruby => tree_sitter_ruby::TAGS_QUERY,
            Lang::Php => tree_sitter_php::TAGS_QUERY,
        }
    }

    pub(crate) const fn all() -> [Lang; 11] {
        [
            Lang::Rust,
            Lang::Python,
            Lang::JavaScript,
            Lang::TypeScript,
            Lang::Tsx,
            Lang::Go,
            Lang::Java,
            Lang::C,
            Lang::Cpp,
            Lang::Ruby,
            Lang::Php,
        ]
    }
}

/// Resolve a path's extension to a [`Lang`], or `None` if unsupported.
pub fn detect(path: &Path) -> Option<Lang> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "rs" => Lang::Rust,
        "py" | "pyi" => Lang::Python,
        "js" | "jsx" | "mjs" | "cjs" => Lang::JavaScript,
        "ts" | "mts" | "cts" => Lang::TypeScript,
        "tsx" => Lang::Tsx,
        "go" => Lang::Go,
        "java" => Lang::Java,
        "c" | "h" => Lang::C,
        "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => Lang::Cpp,
        "rb" => Lang::Ruby,
        "php" | "phtml" | "php3" | "php4" | "php5" => Lang::Php,
        _ => return None,
    })
}

/// Process-wide cache of compiled tags configurations, built on first use.
/// Languages whose query fails to compile are simply absent (and skipped).
fn configs() -> &'static HashMap<Lang, TagsConfiguration> {
    static CONFIGS: OnceLock<HashMap<Lang, TagsConfiguration>> = OnceLock::new();
    CONFIGS.get_or_init(|| {
        let mut m = HashMap::new();
        for lang in Lang::all() {
            match TagsConfiguration::new(lang.language(), lang.tags_query(), "") {
                Ok(cfg) => {
                    m.insert(lang, cfg);
                }
                Err(e) => {
                    tracing::warn!("code: tags config for {} failed: {e}", lang.name());
                }
            }
        }
        m
    })
}

/// The compiled tags configuration for a language, if available.
pub fn config(lang: Lang) -> Option<&'static TagsConfiguration> {
    configs().get(&lang)
}
