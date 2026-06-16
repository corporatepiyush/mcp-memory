# mcp-memory

A [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server that gives
LLM agents a persistent **knowledge graph memory** — entities, relations, and
observations stored in a compact custom binary log with write-ahead durability.

It speaks MCP over stdio, so it plugs directly into Claude Desktop, Claude Code, and
any other MCP-compatible client.

## Features

- **Knowledge graph model** — entities (typed, with free-form observations) connected
  by directed relations.
- **Durable binary log** — every mutation is written ahead to an append-only log and
  `fsync`ed; state is replayed on startup. `compact` rewrites the log to its minimal
  form via an atomic rename.
- **Fast lookups** — string interning, an open-addressing name table, and an inverted
  search index keep CRUD, substring search, and BFS path-finding cheap.
- **14 MCP tools** covering reads, writes, search, and graph traversal.

## Installation

```sh
cargo install mcp-memory
```

Or build from source:

```sh
git clone https://github.com/corporatepiyush/mcp-memory
cd mcp-memory
cargo build --release
```

## Usage

The server runs in stdio mode:

```sh
mcp-memory --memory-file ./memory.mcpmem
```

The memory file path is resolved in this order:

1. `--memory-file` / `-f` flag
2. `MEMORY_FILE_PATH` environment variable
3. Default: `memory.mcpmem` in the working directory

### Claude Desktop / Claude Code config

```json
{
  "mcpServers": {
    "memory": {
      "command": "mcp-memory",
      "args": ["--memory-file", "/absolute/path/to/memory.mcpmem"]
    }
  }
}
```

## Tools

| Tool | Kind | Description |
| --- | --- | --- |
| `create_entities` | write | Create new entities (deduplicated by name). |
| `create_relations` | write | Create directed relations between entities. |
| `add_observations` | write | Append observations to an existing entity. |
| `delete_entities` | write | Delete entities; relations touching them are cascaded away. |
| `delete_observations` | write | Remove specific observations from an entity. |
| `delete_relations` | write | Remove exact `(from, to, type)` relations. |
| `compact` | write | Rewrite the log to its minimal form. |
| `read_graph` | read | Return the full graph. |
| `search_nodes` | read | Substring search over names/types/observations. |
| `open_nodes` | read | Fetch a specific set of entities and their relations. |
| `get_entity` | read | Fetch a single entity by name. |
| `graph_stats` | read | Entity/relation counts and other stats. |
| `search_relations` | read | Filter relations by `from`, `to`, and/or `type`. |
| `find_path` | read | Shortest path (BFS, undirected) between two entities. |

## Development

```sh
cargo test            # unit + integration + fuzzy tests
cargo clippy --all-targets
cargo build --release
```

The test suite includes property-style **fuzzy tests** (`tests/fuzzy.rs`) that drive
randomized CRUD sequences against an in-memory model and assert graph invariants
(uniqueness, cascade deletes, durability across reopen, search/path correctness, and
Unicode/large-string handling).

## License

MIT
