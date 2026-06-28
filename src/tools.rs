//! Tool registry and category gating.
//!
//! Tools are grouped into [`ToolCategory`] banners. **No tool is exposed by
//! default** — each category must be explicitly enabled at startup with the
//! matching `--enable-<slug>` flag (or `--enable-all`). A tool whose category is
//! disabled is hidden from `tools/list` and rejected from `tools/call`.
//!
//! The knowledge-graph tools below carry a `write` flag that also selects their
//! category: read-only queries are [`ToolCategory::GraphRead`], mutations are
//! [`ToolCategory::GraphWrite`]. The vector and code tools live in separate
//! JSON manifests (`vector_tools.json`, `code_tools.json`); their names are
//! enumerated here so [`category_of`] can classify them uniformly.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Coarse capability groups used to selectively expose tools at startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolCategory {
    /// Read-only knowledge-graph queries.
    GraphRead,
    /// Knowledge-graph mutations (create/delete/merge/compact/upsert).
    GraphWrite,
    /// Vector / semantic search (`vector_*` + `hybrid_search`).
    Vectors,
    /// Tree-sitter code-symbol indexing (`code_*`).
    Code,
}

impl ToolCategory {
    pub const ALL: &'static [ToolCategory] = &[
        ToolCategory::GraphRead,
        ToolCategory::GraphWrite,
        ToolCategory::Vectors,
        ToolCategory::Code,
    ];

    pub const fn slug(self) -> &'static str {
        match self {
            ToolCategory::GraphRead => "graph-read",
            ToolCategory::GraphWrite => "graph-write",
            ToolCategory::Vectors => "vectors",
            ToolCategory::Code => "code",
        }
    }
}

impl fmt::Display for ToolCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

impl FromStr for ToolCategory {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_lowercase().replace('_', "-").as_str() {
            "graph-read" => Ok(ToolCategory::GraphRead),
            "graph-write" => Ok(ToolCategory::GraphWrite),
            "vectors" => Ok(ToolCategory::Vectors),
            "code" => Ok(ToolCategory::Code),
            _ => Err(format!("Unknown tool category: {s}")),
        }
    }
}

pub struct ToolMeta {
    pub name: &'static str,
    pub write: bool,
}

impl ToolMeta {
    /// Category of a knowledge-graph tool: writes are `GraphWrite`, the rest
    /// `GraphRead`.
    pub const fn category(&self) -> ToolCategory {
        if self.write {
            ToolCategory::GraphWrite
        } else {
            ToolCategory::GraphRead
        }
    }
}

pub const ALL_TOOLS: &[ToolMeta] = &[
    ToolMeta { name: "create_entities",    write: true  },
    ToolMeta { name: "create_relations",   write: true  },
    ToolMeta { name: "add_observations",   write: true  },
    ToolMeta { name: "delete_entities",    write: true  },
    ToolMeta { name: "delete_observations",write: true  },
    ToolMeta { name: "delete_relations",   write: true  },
    ToolMeta { name: "read_graph",         write: false },
    ToolMeta { name: "search_nodes",       write: false },
    ToolMeta { name: "open_nodes",         write: false },
    ToolMeta { name: "get_entity",         write: false },
    ToolMeta { name: "graph_stats",        write: false },
    ToolMeta { name: "search_relations",   write: false },
    ToolMeta { name: "find_path",          write: false },
    ToolMeta { name: "compact",            write: true  },
    ToolMeta { name: "get_neighbors",      write: false },
    ToolMeta { name: "describe_entity",    write: false },
    ToolMeta { name: "list_entity_types",  write: false },
    ToolMeta { name: "list_relation_types",write: false },
    ToolMeta { name: "upsert_entities",    write: true  },
    ToolMeta { name: "export_graph",       write: false },
    ToolMeta { name: "merge_entities",    write: true  },
    ToolMeta { name: "extract_subgraph",  write: false },
    ToolMeta { name: "batch_get_entities",write: false },
    ToolMeta { name: "find_all_paths",    write: false },
    ToolMeta { name: "entity_exists",     write: false },
    ToolMeta { name: "degree",            write: false },
];

/// Names of the vector-search tools (manifest: `vector_tools.json`).
pub const VECTOR_TOOL_NAMES: &[&str] = &[
    "vector_upsert_embedding",
    "vector_search_entities",
    "vector_delete_embedding",
    "hybrid_search",
    "vector_refresh_graph_cache",
    "vector_store_stats",
    "vector_batch_upsert",
    "vector_get_embedding",
    "vector_search_by_entity",
    "vector_recommend",
    "vector_mmr_search",
    "vector_reindex",
];

/// Names of the tree-sitter code tools (manifest: `code_tools.json`).
pub const CODE_TOOL_NAMES: &[&str] = &[
    "code_index",
    "code_outline",
    "code_search",
    "code_get_symbol",
    "code_watch",
    "code_embed",
    "code_semantic_search",
];

#[inline]
pub fn tool_exists(name: &str) -> bool {
    ALL_TOOLS.iter().any(|t| t.name == name)
}

#[inline]
pub fn is_write_tool(name: &str) -> bool {
    ALL_TOOLS.iter().find(|t| t.name == name).map(|t| t.write).unwrap_or(false)
}

/// `true` for the vector-specific tool names (`vector_*` plus `hybrid_search`).
#[inline]
pub fn is_vector_tool_name(name: &str) -> bool {
    VECTOR_TOOL_NAMES.contains(&name)
}

/// `true` for the tree-sitter code tool names.
#[inline]
pub fn is_code_tool_name(name: &str) -> bool {
    CODE_TOOL_NAMES.contains(&name)
}

/// The category a tool belongs to, or `None` if the name is unknown.
#[inline]
pub fn category_of(name: &str) -> Option<ToolCategory> {
    if let Some(t) = ALL_TOOLS.iter().find(|t| t.name == name) {
        return Some(t.category());
    }
    if is_vector_tool_name(name) {
        return Some(ToolCategory::Vectors);
    }
    if is_code_tool_name(name) {
        return Some(ToolCategory::Code);
    }
    None
}

/// Whether a tool is callable given the set of enabled categories. A tool is
/// available only if it is a known name *and* its category is enabled. (The
/// vector and code subsystems impose additional runtime gating — a store/flag
/// must also be present — handled at the call site.)
#[inline]
pub fn is_tool_available(name: &str, enabled: &[ToolCategory]) -> bool {
    category_of(name).is_some_and(|c| enabled.contains(&c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_exists() {
        assert!(tool_exists("create_entities"));
        assert!(tool_exists("read_graph"));
        assert!(!tool_exists("nope"));
    }

    #[test]
    fn test_is_write_tool() {
        assert!(is_write_tool("create_entities"));
        assert!(!is_write_tool("read_graph"));
    }

    #[test]
    fn test_categories() {
        assert_eq!(category_of("read_graph"), Some(ToolCategory::GraphRead));
        assert_eq!(category_of("create_entities"), Some(ToolCategory::GraphWrite));
        assert_eq!(category_of("hybrid_search"), Some(ToolCategory::Vectors));
        assert_eq!(category_of("code_index"), Some(ToolCategory::Code));
        assert_eq!(category_of("nope"), None);
    }

    #[test]
    fn test_is_tool_available_gating() {
        assert!(!is_tool_available("read_graph", &[]));
        assert!(is_tool_available("read_graph", &[ToolCategory::GraphRead]));
        assert!(!is_tool_available("create_entities", &[ToolCategory::GraphRead]));
        assert!(is_tool_available("create_entities", &[ToolCategory::GraphWrite]));
        assert!(!is_tool_available("nope", ToolCategory::ALL));
    }

    #[test]
    fn test_category_slug_roundtrip() {
        for &c in ToolCategory::ALL {
            assert_eq!(c.slug().parse::<ToolCategory>().unwrap(), c);
        }
        assert!("bogus".parse::<ToolCategory>().is_err());
        assert!(ToolCategory::ALL.len() <= 10);
    }
}
