pub struct ToolMeta {
    pub name: &'static str,
    pub write: bool,
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

#[inline]
pub fn tool_exists(name: &str) -> bool {
    ALL_TOOLS.iter().any(|t| t.name == name)
}

#[inline]
pub fn is_write_tool(name: &str) -> bool {
    ALL_TOOLS.iter().find(|t| t.name == name).map(|t| t.write).unwrap_or(false)
}


