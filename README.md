# mcp-memory

A [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server providing
LLM agents with a persistent **knowledge graph memory** — entities, relations, and
observations stored in an LSM-tree-backed embedded database.

Speaks MCP over stdio, TCP, and HTTP transports.

```
                   ┌──────────────────────────────────────────────┐
                   │              mcp-memory server               │
                   │                                              │
    ┌───────┐      │  ┌──────────┐   ┌───────────────────────┐   │
    │Claude │──────│─>│  stdio /  │──>│ GraphHandle           │   │
    │Desktop│      │  │  TCP /   │   │  ├ LRU entity cache    │   │
    └───────┘      │  │  HTTP    │   │  ├ LRU adj cache       │   │
                   │  └──────────┘   │  ├ BM25 search index   │   │
                   │         │       │  └──→ TidesStore ──→   │   │
                   │         v       └──────────┬──────────────┘   │
                   │  ┌─────────────────────────┴─────────────┐   │
                   │  │  TidesDB (LSM-tree, 6 column families) │   │
                   │  │  entities, rel_out/in, search/inv,    │   │
                   │  │  metadata                              │   │
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
| `tools/list`, `tools/call` | 24 tools |
| `CallToolResult` | `content[]` + `isError` |
| Auth | optional bearer token on TCP/HTTP (constant-time) |
| Capabilities advertised | `tools` only |

Tool failures are returned as `CallToolResult`s with `isError: true` (not as
JSON-RPC protocol errors) so the model can self-correct.

## Data model

```
Entity(name, entityType, observations[])
  |                          |
  |  ——— relationType ———→   |
  v                          v
Entity(name, entityType, observations[])
```

- **Entity** — a named node with a type (e.g. `person`, `company`, `project`)
  and free-form observation strings.
- **Relation** — a directed edge `(from, to, relationType)` between two
  entities. Relations are undirected in traversal (BFS follows both ways).
- **Observation** — an unstructured fact attached to an entity.

Search uses [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) tokenization via
the `bm25` crate. The inverted index maps token IDs to entity names with BM25
weights, stored in the `search_inv` column family.

## Data structures & performance

### Storage engine: TidesDB (embedded LSM-tree)

Six column families in a single TidesDB database:

| Column family | Key | Value | Purpose |
|---|---|---|---|
| `entities` | entity name (UTF-8) | bincode-serialized Entity | Primary entity storage |
| `rel_out` | `from\|to\|type` | `[0u8]` (1 byte) | Outgoing relation index |
| `rel_in` | `to\|from\|type` | `[0u8]` (1 byte) | Incoming relation index |
| `search` | entity name (UTF-8) | bincode (token indices + weights) | BM25 embedding storage |
| `search_inv` | `{:010}\|name` (padded token idx) | f32 LE bytes | Inverted token → entity index |
| `metadata` | — | — | Reserved |

### In-memory caches (GraphHandle)

| Cache | Size | Purpose |
|---|---|---|
| Entity LRU | 10,000 entries | Avoids deserializing hot entities |
| Adjacency LRU | 5,000 entries | Avoids re-reading relation iterators |
| Token invert cache | 10,000 entries | Avoids re-reading search_inv rows |

### Write batching

Every mutation goes through a layered write path:

1. **Existence checks** — batch-read entity existence in one read transaction
2. **Batch commit** — all new entities written in one write transaction
3. **Batch index** — all BM25 embeddings written in one write transaction
4. **Cache invalidation** — LRU entries for affected names are evicted

This reduces transaction count from O(N) to O(1) per `create_entities`/`create_relations` call.

## Benchmarks

Measured end-to-end via stdio (spawn server subprocess, JSON-RPC round-trip).
1,000 entities + 200 relations pre-populated. MacBook Pro (M4 Pro, 24 GB).

Run `cargo run --release --bin bench_stdio` on your target hardware.

| Operation | Avg latency | Notes |
|---|---|---|
| `get_entity` | 20 µs | LRU cache hit hot path |
| `search_nodes` | 30 µs | Query token → invert index → entity lookup |
| `open_nodes` (10 names) | 25 µs | Batch get via LRU + store |
| `neighbors` depth=1 | 30 µs | Outgoing relation scan |
| `find_path` (BFS) | 670 µs | Worst case: target not found, full BFS |
| `describe_entity` | 30 µs | Entity + incident relations |
| `graph_stats` | 135 µs | Entity count + obs count + relation count |
| `read_graph` | 900 µs | Full dump: all entities + all relations |
| `graph_stats` (throughput, pipelined) | — | **8,200 req/s** |
| `create_entities` (10 new) | 35 µs | Batch existence check + batch put + batch index |
| `create_relations` (10 new) | 47 µs | Batch entity checks + batch dup check + batch put |

### How writes scale

| Batch size | `create_entities` | `create_relations` |
|---|---|---|
| 1 entity | ~15 µs | — |
| 10 entities | **35 µs** (0.29 µs/entity overhead) | — |
| 10 relations | — | **47 µs** (0.47 µs/relation overhead) |

Per-element overhead comes from BM25 embedding computation (Rust-side, not I/O bound).

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
- `compact` — compact all column families in TidesDB

### Read tools

- `read_graph` — dump all entities + relations
- `search_nodes` — BM25-ranked search over names, types, observations
- `open_nodes` — fetch specific entities by name
- `batch_get_entities` — bulk entity fetch (order preserved, null for missing)
- `get_entity` — single entity by name
- `graph_stats` — entity count, relation count, total observations
- `search_relations` — filter by from/to/type
- `describe_entity` — entity + incident relations + neighbors + degree
- `find_path` — BFS shortest path (undirected)
- `find_all_paths` — DFS all simple paths (bounded by maxDepth, maxPaths)
- `extract_subgraph` — BFS around seed entities to given depth
- `neighbors` — entity neighbors with direction + type + depth filters
- `entity_type_counts` — type → count, ranked
- `relation_type_counts` — type → count, ranked
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
              ├── "tools/list"     → cached from tools.json
              ├── "tools/call"     → dispatches to handler by name
              ├── "ping"           → null
              └── "notifications/" → no reply
```

All three transports share `process_value()` / `dispatch_line()` / `dispatch_http_body()`
— the dispatch core is **transport-agnostic**.

### Locking

- `GraphHandle` uses `parking_lot::RwLock` for LRU caches (entity_cache, adj_cache)
- All `GraphHandle` methods take `&self` — internal `RwLock` handles mutation
- Tokio multi-thread runtime handles concurrent requests
- TidesDB handles its own internal locking (MVCC + latches)

### Write path

```
create_entities([e1, e2, ...])
  1. Batch-check existence: entities_exist_batch (one read txn)
  2. Batch-put: put_entities_batch (one write txn)
  3. Batch-index: search.index_entities_batch (BM25 embed + one write txn)
  4. Invalidate LRU caches
```

The same batching pattern applies to `create_relations`.

### Storage (TidesDB)

TidesDB is an embedded LSM-tree database with:

- **Memtable** — in-memory write buffer (WAL-backed)
- **SSTables** — sorted immutable files on disk (LZ4-compressed blocks)
- **Bloom filters** — per-SSTable, ~1/128 false positive rate
- **Compaction** — background level merge to maintain read performance
- **WAL** — write-ahead log for crash recovery

TidesDB uses `Async` durability by default (flush to kernel page cache, background
sync). This gives sub-millisecond write latencies with at-most-1-second data loss
on power failure. Set `MEMORY_MEMORY_DURABILITY=sync` for fsync-on-every-write.

## Development

```sh
cargo test             # unit + integration + fuzzy (128+ tests)
cargo clippy --all-targets
cargo build --release  # LTO + panic=abort
cargo run --release --bin bench_stdio  # stdio round-trip benchmarks
```

The test suite includes:
- **Unit tests** — protocol, tools, config, error codes
- **Integration tests** — CRUD persistence, search, paths, export, concurrency,
  all 24 tool handlers, invariants
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
