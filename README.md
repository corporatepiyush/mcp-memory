# mcp-memory

A [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server providing
LLM agents with a persistent **knowledge graph memory** — entities, relations, and
observations stored in an embedded SQLite database with FTS5 full-text search.

Speaks MCP over stdio, TCP, and HTTP transports.

```
                    ┌──────────────────────────────────────────────┐
                    │              mcp-memory server               │
                    │                                              │
     ┌───────┐      │  ┌──────────┐   ┌───────────────────────┐   │
     │Claude │──────│─>│  stdio /  │──>│ GraphHandle           │   │
     │Desktop│      │  │  TCP /   │   │  ├ LRU entity cache    │   │
     └───────┘      │  │  HTTP    │   │  ├ FxHashMap name→ID   │   │
                    │  └──────────┘   │  ├ FTS5 full-text idx  │   │
                    │         │       │  └──→ SQLite ──→       │   │
                    │         v       └──────────┬──────────────┘   │
                    │  ┌─────────────────────────┴─────────────┐   │
                    │  │  SQLite (WAL mode, 16 KB pages)        │   │
                    │  │  entity, observation, relation,        │   │
                    │  │  name_fts, obs_fts, type_dict          │   │
                    │  └───────────────────────────────────────┘   │
                    └──────────────────────────────────────────────┘
```

## Installation

```sh
cargo install mcp-memory
```

## Quick start

```sh
mcp-memory --transport stdio
```

The database path is resolved in order:

1. `--memory-file` / `-f` flag
2. `MEMORY_FILE_PATH` environment variable
3. Default: `memory.mcpmem` in the working directory

### Transports

| Transport | Flag | Description |
|-----------|------|-------------|
| stdio | `--transport stdio` | Newline-delimited JSON over stdin/stdout (default, for Claude Desktop / Claude Code) |
| tcp | `--transport tcp --bind 0.0.0.0:8080` | Newline-delimited JSON over TCP, concurrent connections |
| http | `--transport http --bind 0.0.0.0:8080` | MCP Streamable HTTP (POST/GET `/mcp`) |

### Claude Desktop config

```json
{
  "mcpServers": {
    "memory": {
      "command": "mcp-memory"
    }
  }
}
```

### Claude Code config

```json
{
  "mcpServers": {
    "memory": {
      "command": "mcp-memory"
    }
  }
}
```

### Authentication

The `tcp` and `http` transports accept an optional bearer token (stdio is never
authenticated). Set it with `--auth-token`, `--auth-token-file` (trimmed; an
empty file is rejected), or `MCP_MEMORY_AUTH_TOKEN`:

```sh
mcp-memory --transport tcp --bind 0.0.0.0:8080 --auth-token "s3cr3t"
mcp-memory --transport http --bind 0.0.0.0:8080 --auth-token "s3cr3t"
```

Binding a non-loopback address **without** a token exposes the entire graph to
the network. Comparison is constant-time.

## MCP Compliance

Implements the [Model Context Protocol](https://modelcontextprotocol.io) revision **`2025-11-25`** over JSON-RPC 2.0, via stdio, TCP, or HTTP.

| Area | Support |
|---|---|
| Transports | stdio, TCP, **Streamable HTTP** (POST/GET `/mcp`, SSE) |
| Protocol version | `2025-11-25`, negotiates down to `2025-06-18` / `2025-03-26` / `2024-11-05` |
| `initialize` | version negotiation + `instructions` |
| `tools/list`, `tools/call` | 26 tools |
| `CallToolResult` | `content[]` + `isError` |
| Auth | optional bearer token on TCP/HTTP (constant-time) |
| Capabilities advertised | `tools` only |

Tool failures are returned as `CallToolResult`s with `isError: true` (not as
JSON-RPC protocol errors) so the model can self-correct.

## Data model

```
Entity(name, entityType, observations[])
  |                          |
  |  —— relationType ——→   |
  v                          v
Entity(name, entityType, observations[])
```

- **Entity** — a named node with a type (e.g. `person`, `company`, `project`)
  and free-form observation strings.
- **Relation** — a directed edge `(from, to, relationType)` between two
  entities. Relations are undirected in traversal (BFS follows both ways).
- **Observation** — an unstructured fact attached to an entity.

Search uses FTS5 full-text indexing with `unicode61 remove_diacritics 2`
tokenization. Name and observation bodies live in separate FTS5 virtual tables
(`name_fts`, `obs_fts`) with external content referencing the core tables.

## Data structures & performance

### Storage engine: SQLite (WAL mode)

A single SQLite database in WAL mode with the following schema:

| Table | Key | Purpose |
|---|---|---|
| `entity` | `INTEGER PRIMARY KEY` (rowid) | Primary entity storage; materialized `obs_count`, `out_deg`, `in_deg`; `name_hash` for O(1) routing |
| `observation` | `entity_id` (FK) + rowid | 1:N observations per entity |
| `relation` | composite indexes | Directed edges; covering indexes `rel_out(from_id,type_id,to_id)` and `rel_in(to_id,type_id,from_id)` for index-only scans |
| `name_fts` | `content_rowid` | External-content FTS5 over `entity.name` |
| `obs_fts` | `content_rowid` | External-content FTS5 over `observation.body` |
| `type_dict` | name | Interned entity/relation types with live counts (loaded into RAM) |
| `graph_stat` | key (singleton) | `WITHOUT ROWID` counters: entities, relations, observations, entity_seq, obs_seq |
| `hub_degree` | entity_id | Degree spill for high-degree hubs |
| `partition_map` | entity_id | Reserved for future entity-type partitioning |

Key SQLite pragmas: `page_size=16384`, `journal_mode=WAL`, `synchronous=NORMAL`,
`cache_size=-50000` (~50 MB), `mmap_size=256 MB`, `temp_store=MEMORY`,
`busy_timeout=5000`.

### In-memory caches (GraphHandle)

| Cache | Size | Purpose |
|---|---|---|
| Entity LRU | 10,000 entries | Avoids deserializing hot entities; stores `EntityMeta{id, type_id, obs_count, out_deg, in_deg}` |
| Name hash FxHashMap | all loaded | O(1) name-to-ID resolution via 64-bit FNV-1a hash |
| Prepared statement cache | SQLite internal | Reuses compiled queries |

### Write batching

Every mutation goes through a layered write path:

1. **Existence checks** — batch-read entity existence in one read transaction
2. **Batch commit** — all new entities/relations written in one write transaction
3. **Batch index** — all FTS entries updated in one write transaction
4. **Cache invalidation** — LRU entries for affected names are evicted

This reduces transaction count from O(N) to O(1) per `create_entities`/`create_relations` call.

### Durability

| Mode | Behavior | Data loss window |
|---|---|---|
| `async` (default) | Flush to kernel page cache, background sync | Up to ~1 second on power failure |
| `sync` | fsync before every write | Zero |

Set via `MCP_MEMORY_DURABILITY=sync`.

### Background maintenance

A background tokio task runs every 5 minutes and performs WAL checkpointing
(`PRAGMA wal_checkpoint(TRUNCATE)`), query planner analysis (`PRAGMA optimize`),
and FTS optimization.

## Benchmarks

Measured end-to-end via the `bench` binary. 1,000 entities + 200 relations
pre-populated. MacBook Pro (M4 Pro, 24 GB).

Run `cargo run --release --bin bench` on your target hardware.

| Operation | Avg latency | Notes |
|---|---|---|
| `get_entity` (cache hit) | ~20 µs | LRU hit; no SQLite I/O |
| `search_nodes` (name match) | ~25 µs | FTS5 query + entity lookup |
| `open_nodes` (single) | ~30 µs | LRU + SQLite |
| `open_nodes` (5 names) | ~60 µs | Batch fetch |
| `neighbors` depth=1 | ~30 µs | Index-only scan via covering index |
| `neighbors` depth=2 | ~55 µs | Two-hop traversal |
| `find_path` (BFS) | ~650 µs | Worst case: target not found, full BFS |
| `describe_entity` | ~30 µs | Entity + incident relations |
| `graph_stats` | ~15 µs | RAM counters (graph_stat table) |
| `read_graph` (all) | ~1500 µs | Full dump: all entities + relations |
| `create_entities` (1000) | ~2000 µs | Batch write + FTS index |
| `create_relations` (999) | ~1200 µs | Batch write + degree updates |
| `find_all_paths` (A→C, depth 5) | ~100 µs | Bounded DFS |
| `export_graph` (JSON) | ~600 µs | Serialize all entities + relations |
| `entity_type_counts` | ~10 µs | RAM-cached type dictionary |
| `degree` (cache hit) | ~2 µs | Materialized column |
| `entities_exist` (10 names) | ~15 µs | Hash lookup via FxHashMap |

## Tools

### Write tools

- `create_entities` — batch create, skips existing names
- `create_relations` — batch create, skips missing entities and duplicates
- `add_observations` — append to entity, deduplicates
- `delete_entities` — cascade deletes incident relations
- `delete_observations` — remove specific observations
- `delete_relations` — remove exact (from, to, type) tuples
- `upsert_entities` — create or merge (type preserved, observations unioned)
- `merge_entities` — source → target redirect with full dedup
- `compact` — trigger incremental vacuum + FTS optimize

### Read tools

- `read_graph` — dump all entities + relations (with optional type filter, offset, limit)
- `search_nodes` — FTS5-ranked search over names, types, observations (with optional type filter)
- `open_nodes` — fetch specific entities by name (with their relations)
- `batch_get_entities` — bulk entity fetch (order preserved, null for missing)
- `get_entity` — single entity by name
- `entity_exists` — cheap existence check (hash lookup, no observation bodies fetched)
- `graph_stats` — entity count, relation count, total observations
- `search_relations` — filter by from/to/type
- `describe_entity` — entity + incident relations + neighbors + degree
- `degree` — number of incident relations by direction (outgoing / incoming / both)
- `find_path` — BFS shortest path (undirected)
- `find_all_paths` — DFS all simple paths (bounded by maxDepth, maxPaths)
- `extract_subgraph` — BFS around seed entities to given depth
- `get_neighbors` — entity neighbors with direction + type + depth filters
- `list_entity_types` — type → count, ranked
- `list_relation_types` — type → count, ranked
- `export_graph` — JSON, Mermaid, or Graphviz DOT

## Architecture

```
main.rs
  │
  ├── MCPServer::run_stdio()   — stdio transport (newline-delimited JSON-RPC)
  ├── MCPServer::run_tcp()     — TCP transport (same framing, concurrent conns)
  └── MCPServer::run_http()    — MCP Streamable HTTP (axum, POST/GET /mcp)
        │
        └── process_request()
              │
              ├── "initialize"     → protocol version + capabilities
              ├── "tools/list"     → cached from tools.rs
              ├── "tools/call"     → dispatches to handler by name
              ├── "ping"           → null
              └── "notifications/" → no reply
```

All three transports share `process_value()` / `dispatch_line()` / `dispatch_http_body()`
— the dispatch core is **transport-agnostic**.

### Locking

- `GraphHandle` uses `parking_lot::Mutex` for the SQLite connection and LRU caches
- All `GraphHandle` methods take `&self` — internal `Mutex` handles mutation
- Tokio multi-thread runtime handles concurrent requests
- SQLite WAL mode allows concurrent readers + one writer
- Heavy dispatch (graph lock + optional fsync) is offloaded to `tokio::task::spawn_blocking`

### Write path

```
create_entities([e1, e2, ...])
  1. Batch-check existence (FxHashMap hash lookup)
  2. Batch-insert entities (one write txn)
  3. Batch-index FTS (one write txn for name_fts)
  4. Invalidate LRU caches
  5. Update type_dict counts
```

The same batching pattern applies to `create_relations` (with degree updates).

### Storage (SQLite)

SQLite provides the storage layer with:

- **WAL mode** — concurrent readers + one writer without blocking readers
- **16 KB pages** — shallower B-trees for faster lookups
- **FTS5** — full-text search with `unicode61 remove_diacritics 2` tokenization
- **mmap** — up to 256 MB of the database mapped for faster reads
- **Covering indexes** — `rel_out` and `rel_in` enable index-only neighbor scans
- **Materialized counters** — `obs_count`, `out_deg`, `in_deg`, `type_dict.count`, `graph_stat` are writer-maintained for O(1) reads
- **External-content FTS5** — avoids duplicating text; stable `INTEGER PRIMARY KEY` ensures `content_rowid` correctness across VACUUM

### Concurrency model

- TCP connections limited to 128 concurrent connections
- Mutating operations acquire `GraphHandle` lock and serialize through SQLite
- Read operations can proceed concurrently under WAL mode
- Background maintenance runs every 5 minutes as a tokio task

### Request size limits

| Parameter | Limit |
|---|---|
| Max request body | 16 MB |
| Name max bytes | 1024 |
| Observation max bytes | 65,536 |
| Max entities per request | 1,000 |
| Max relations per request | 1,000 |
| Max observations per entity | 1,000 |
| Max names per request | 1,000 |
| Max search limit | 1,000 |
| Max neighbor depth | 16 |
| Max relation search results | 1,000 |
| Max find_all_paths depth | 10 |
| Max find_all_paths results | 100 |

## Development

```sh
cargo test             # unit + integration + fuzzy (300+ tests)
cargo clippy --all-targets
cargo build --release  # LTO + fat, panic=abort, strip
cargo run --release --bin bench  # standalone benchmark
```

The test suite includes:
- **Unit tests** — protocol, tools, config, error codes
- **Integration tests** — CRUD persistence, search, paths, export, concurrency,
  all 26 tool handlers, invariants
- **Fuzzy tests** — randomized CRUD sequences asserting graph invariants

## Versioning & Compatibility

Follows [Semantic Versioning](https://semver.org). The current line is **2.x**,
targeting MCP revision `2025-11-25`.

| mcp-memory | MCP revision (default) | Negotiates |
|---|---|---|
| 2.x | `2025-11-25` | `2025-06-18`, `2025-03-26`, `2024-11-05` |
| ≤ 1.x | `2024-11-05` | — |

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
