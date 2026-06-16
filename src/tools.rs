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
];

#[inline]
pub fn tool_exists(name: &str) -> bool {
    ALL_TOOLS.iter().any(|t| t.name == name)
}

#[inline]
pub fn is_write_tool(name: &str) -> bool {
    ALL_TOOLS.iter().find(|t| t.name == name).map(|t| t.write).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_exists_known() {
        assert!(tool_exists("create_entities"));
        assert!(tool_exists("read_graph"));
        assert!(tool_exists("search_nodes"));
    }

    #[test]
    fn test_tool_exists_unknown() {
        assert!(!tool_exists("nonexistent_tool"));
    }

    #[test]
    fn test_is_write_tool() {
        assert!(is_write_tool("create_entities"));
        assert!(is_write_tool("delete_entities"));
        assert!(!is_write_tool("read_graph"));
        assert!(!is_write_tool("search_nodes"));
    }

    #[test]
    fn test_all_tools_unique() {
        let mut names: Vec<&str> = ALL_TOOLS.iter().map(|t| t.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), ALL_TOOLS.len(), "Duplicate tool names");
    }
}
