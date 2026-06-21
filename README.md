# mcp-memory

A [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server that gives
LLM agents a persistent **knowledge graph memory** — entities, relations, and
observations stored in an embedded SQLite database with FTS5 full-text search.

It is **one unified server** with an opt-in vector subsystem:

| Invocation | What you get | Tools |
|---|---|---|
| `mcp-memory` | The knowledge-graph server | 26 |
| `mcp-memory --vectors` | Everything above **plus** vector embeddings and semantic / hybrid / MMR search (usearch HNSW **or** IVF-Flat) | 38 |
| `mcp-memory-vec` | Backward-compatible alias for `mcp-memory --vectors` | 38 |

> **v4 note:** the former separate `mcp-memory-vec` server has been merged into
> `mcp-memory`. Vectors are now enabled with the `--vectors` flag; `mcp-memory-vec`
> remains as a thin alias that turns the flag on, so existing configs keep working.

It speaks MCP over **stdio, TCP, and HTTP** (with optional bearer-token auth and TLS).

```
                    ┌────────────────────────────────────────────────┐
                    │      mcp-memory  (+ --vectors / -vec alias)     │
                    │                                                │
     ┌───────┐      │  ┌──────────┐   ┌─────────────────────────┐   │
     │Claude │──────│─>│  stdio / │──>│ GraphHandle             │   │
     │ / LLM │      │  │  TCP /   │   │  ├ LRU entity cache      │   │
     └───────┘      │  │  HTTP    │   │  ├ FxHashMap name→ID     │   │
                    │  └────┬─────┘   │  └ FTS5 full-text index  │   │
                    │       │         └───────────┬─────────────┘   │
                    │       │     (--vectors only) │                 │
                    │       v         ┌───────────┴─────────────┐   │
                    │  ┌─────────┐    │ VectorStore             │   │
                    │  │ dispatch│───>│  ├ ANN: HNSW *or* IVF    │   │
                    │  └─────────┘    │  └ petgraph adjacency    │   │
                    │       │         └───────────┬─────────────┘   │
                    │       v                     v                  │
                    │  ┌──────────────────────────────────────────┐ │
                    │  │ SQLite (WAL, 4 KB pages, auto_vacuum)     │ │
                    │  │ entity, observation, relation, *_fts,     │ │
                    │  │ type_dict, vector_embedding               │ │
                    │  └──────────────────────────────────────────┘ │
                    └────────────────────────────────────────────────┘
```

## Installation

```sh
cargo install mcp-memory
```

This installs both `mcp-memory` and `mcp-memory-vec`.

## Quick start

```sh
# Knowledge-graph server
mcp-memory --transport stdio

# Knowledge-graph + vector search
mcp-memory --vectors --transport stdio --embedding-dims 384

# Equivalent backward-compatible alias
mcp-memory-vec --transport stdio --embedding-dims 384
```

The database path is resolved in order:

1. `--memory-file` / `-f` flag
2. `MEMORY_FILE_PATH` environment variable
3. Default: `memory.mcpmem` in the working directory

The same SQLite file works with or without `--vectors`, so you can populate the
graph plain and later serve it with vectors enabled. With `--vectors` off, the
vector tools are neither advertised in `tools/list` nor served.

### Transports

| Transport | Flag | Description |
|-----------|------|-------------|
| stdio | `--transport stdio` | Newline-delimited JSON over stdin/stdout (default, for Claude Desktop / Claude Code) |
| tcp | `--transport tcp --bind 0.0.0.0:8080` | Newline-delimited JSON over TCP, concurrent connections |
| http | `--transport http --bind 0.0.0.0:8080` | MCP Streamable HTTP (POST/GET `/mcp`, SSE) |

### Claude Desktop / Claude Code config

```json
{
  "mcpServers": {
    "memory": {
      "command": "mcp-memory"
    }
  }
}
```

Add `"args": ["--vectors", "--embedding-dims", "384"]` to enable vector search
(or use `"command": "mcp-memory-vec"`).

### Authentication

The `tcp` and `http` transports accept an optional bearer token (stdio is never
authenticated). Set it with `--auth-token` or `--auth-token-file` (trimmed; an
empty file is rejected), or the `MCP_MEMORY_AUTH_TOKEN` environment variable.

```sh
mcp-memory --transport http --bind 0.0.0.0:8080 --auth-token "s3cr3t"
mcp-memory --vectors --transport http --bind 0.0.0.0:8080 --auth-token "s3cr3t"
```

On HTTP the token is sent as `Authorization: Bearer <token>`; on TCP it is the
first line of the connection. Comparison is constant-time. Binding a non-loopback
address **without** a token exposes the entire graph to the network.

### TLS (HTTPS)

The `http` transport can be served over TLS (rustls, `ring` provider). Provide a
PEM certificate chain and private key via `--tls-cert` / `--tls-key`; both must be
supplied together or startup is refused. The `MCP_TLS_CERT` / `MCP_TLS_KEY`
environment variables are accepted as fallbacks. When neither is set the transport
stays plaintext (the default).

```sh
mcp-memory --transport http --bind 0.0.0.0:8080 \
  --tls-cert ./cert.pem --tls-key ./key.pem
```

## Vector search (`--vectors`)

With `--vectors`, the server layers a vector store on top of the knowledge graph.
Each embedding is attached to an **existing** entity (by name), indexed in an
in-memory ANN index, and persisted as a blob in the `vector_embedding` SQLite
table. On startup the index is rebuilt from those blobs.

- **Bring your own embeddings.** The server stores and searches vectors; it does
  not call an embedding model. Compute embeddings client-side (e.g. with an
  embedding API) and pass them in. All vectors must match `--embedding-dims`.
- **Semantic search** — `vector_search_entities` returns the nearest entities by
  cosine similarity (configurable), optionally filtered by entity type.
- **More-like-this & recommendations** — `vector_search_by_entity` finds entities
  similar to a given entity's own embedding; `vector_recommend` builds a query from
  positive (minus negative) example entities.
- **MMR diversification** — `vector_mmr_search` returns results that balance
  relevance against novelty (Maximal Marginal Relevance), a common RAG
  context-selection step that suppresses near-duplicate hits.
- **Batch ingestion** — `vector_batch_upsert` upserts up to 1,024 embeddings per
  call, reporting per-item failures instead of aborting.
- **Hybrid search** — `hybrid_search` runs vector search and FTS5 text search in
  parallel and fuses the two rankings with Reciprocal Rank Fusion (RRF, constant
  60), then optionally boosts results by graph centrality from an in-memory
  petgraph adjacency cache.

### Index backends: HNSW vs IVF-Flat

Two ANN backends are available via `--vec-index`:

| Backend | When to use | Notes |
|---|---|---|
| `hnsw` *(default)* | Best recall/latency for most workloads | [usearch](https://github.com/unum-cloud/usearch) graph index; supports `f16`/`bf16`/`i8` quantization |
| `ivf` | Large, batch-ingested, periodically-rebuilt corpora | k-means partitioned (IVF-Flat); cheaper to build, lighter memory. **Exact (brute-force) until trained**, so results are always correct |

The IVF index trains automatically when a populated database is opened. After a
large batch ingestion into a fresh database, call `vector_reindex` to (re)run
k-means and keep recall high (no-op for HNSW).

### Vector configuration

The index is tunable from the command line (all require `--vectors`):

| Flag | Default | Meaning |
|---|---|---|
| `--embedding-dims` | `384` | Vector dimension; all embeddings must match |
| `--vec-index` | `hnsw` | ANN backend: `hnsw` or `ivf` |
| `--vec-metric` | `cos` | Distance metric: `cos`, `ip` (dot product), or `l2sq` |
| `--vec-quantization` | `f32` | HNSW scalar storage: `f32`, `f16`, `bf16`, or `i8` (lower = less memory) |
| `--vec-connectivity` | `16` | HNSW graph degree `M` (higher = better recall, more memory) |
| `--vec-expansion-add` | `200` | HNSW `efConstruction` (higher = better index quality, slower inserts) |
| `--vec-expansion-search` | `50` | HNSW `efSearch` (higher = better recall, slower queries) |
| `--ivf-nlist` | `256` | IVF number of Voronoi cells / centroids |
| `--ivf-nprobe` | `8` | IVF cells probed per query (higher = better recall, slower) |

```sh
# HNSW with half-precision storage
mcp-memory --vectors --transport http --bind 0.0.0.0:8080 \
  --embedding-dims 768 --vec-metric cos --vec-quantization f16 \
  --vec-connectivity 32 --vec-expansion-search 128

# IVF-Flat for a large corpus
mcp-memory --vectors --embedding-dims 768 \
  --vec-index ivf --ivf-nlist 1024 --ivf-nprobe 16
```

The petgraph adjacency cache used for the hybrid-search centrality boost is built
lazily; call `vector_refresh_graph_cache` after mutating relations to refresh it.

## MCP compliance

Implements the [Model Context Protocol](https://modelcontextprotocol.io) revision
**`2025-11-25`** over JSON-RPC 2.0, via stdio, TCP, or HTTP.

| Area | Support |
|---|---|
| Transports | stdio, TCP, **Streamable HTTP** (POST/GET `/mcp`, SSE) |
| Protocol version | `2025-11-25`, negotiates down to `2025-06-18` / `2025-03-26` / `2024-11-05` |
| `initialize` | version negotiation + `instructions` |
| `tools/list`, `tools/call` | 26 tools (KG only) / 38 tools (with `--vectors`) |
| `CallToolResult` | `content[]` + `isError` |
| Auth | optional bearer token on TCP/HTTP (constant-time) |
| Capabilities advertised | `tools` only |

Tool failures are returned as `CallToolResult`s with `isError: true` (not as
JSON-RPC protocol errors) so the model can self-correct.

## Data model

```
Entity(name, entityType, observations[])   ──relationType──▶   Entity(...)
```

- **Entity** — a named node with a type (e.g. `person`, `company`, `project`) and
  free-form observation strings. Names are unique and case-sensitive.
- **Relation** — a directed edge `(from, to, relationType)`. Traversal is
  undirected (BFS/DFS follow both directions).
- **Observation** — an unstructured fact attached to an entity.
- **Embedding** *(`--vectors`)* — a fixed-dimension `f32` vector attached to an
  entity, plus an optional model identifier.

Search uses FTS5 full-text indexing with `unicode61 remove_diacritics 2`
tokenization. Names and observation bodies live in separate external-content FTS5
tables (`name_fts`, `obs_fts`).

## Storage & performance

### SQLite (WAL mode)

A single SQLite database in WAL mode:

| Table | Key | Purpose |
|---|---|---|
| `entity` | `INTEGER PRIMARY KEY` (rowid) | Primary entity storage; materialized `obs_count`, `out_deg`, `in_deg`; `name_hash` for O(1) routing |
| `observation` | `entity_id` (FK) + rowid | 1:N observations per entity |
| `relation` | composite indexes | Directed edges; covering indexes `rel_out(from_id,type_id,to_id)` and `rel_in(to_id,type_id,from_id)` for index-only scans |
| `name_fts` | `content_rowid` | External-content FTS5 over `entity.name` |
| `obs_fts` | `content_rowid` | External-content FTS5 over `observation.body` |
| `type_dict` | name | Interned entity/relation types with live counts (loaded into RAM) |
| `graph_stat` | key (singleton) | `WITHOUT ROWID` counters: entities, relations, observations, sequences |
| `vector_embedding` | `entity_id` | *(`--vectors`)* `dims`, `blob` (f32 vector), `model`, `created_us` |

Key pragmas (defaults, all tunable via flags): `page_size=4096`,
`journal_mode=WAL`, `auto_vacuum=INCREMENTAL`, `synchronous=NORMAL`,
`cache_size=-50000` (~50 MB, `--cache-size-mb`), `mmap_size=256 MB`
(`--mmap-size`), `temp_store=MEMORY`, `busy_timeout=5000` (`--busy-timeout-ms`).
A background `wal_checkpoint(PASSIVE)` runs every `--wal-flush-ms` (default 250 ms)
to bound the async durability window.

### In-memory caches

| Cache | Purpose |
|---|---|
| Entity LRU (10,000 entries) | Avoids deserializing hot entities; stores `EntityMeta{id, type_id, obs_count, out_deg, in_deg}` |
| Name-hash map | O(1) name-to-ID resolution via 64-bit hash |
| Prepared-statement cache | Reuses compiled SQLite queries |
| ANN index *(`--vectors`)* | In-memory HNSW or IVF-Flat index, rebuilt from `vector_embedding` on startup |
| petgraph adjacency *(`--vectors`)* | Directed graph cache for the hybrid-search centrality boost |

### Write batching

Every mutation goes through a layered write path that collapses transaction count
from O(N) to O(1) per `create_entities` / `create_relations` call:

1. Batch existence checks in one read transaction
2. Batch commit of all new entities/relations in one write transaction
3. Batch FTS index updates in one write transaction
4. Cache invalidation for affected names

### Durability

| Mode | Behavior | Data-loss window |
|---|---|---|
| `async` (default) | Flush to kernel page cache, background sync | Up to ~1 s on power failure |
| `sync` | fsync before every write | Zero |

Set via the `MCP_MEMORY_DURABILITY=sync` environment variable (applies whether or
not `--vectors` is on).

### Background maintenance

A background tokio task runs every 5 minutes: WAL checkpoint
(`PRAGMA wal_checkpoint(TRUNCATE)`), planner analysis (`PRAGMA optimize`), and FTS
optimization.

## Benchmarks

Measured end-to-end via the `bench` binary, 1,000 entities (5 observations each) +
999 relations pre-populated, on a **MacBook Pro (Apple M1 Pro, 32 GB)**. Numbers
are averages and will vary by hardware — run `cargo run --release --bin bench` on
your own target.

| Operation | Avg latency | Notes |
|---|---|---|
| `degree` (cache hit) | ~44 ns | Materialized column |
| `relation_type_counts` | ~2.3 µs | RAM-cached type dictionary |
| `get_entity_count` | ~3.0 µs | RAM counter |
| `entity_type_counts` | ~4.5 µs | RAM-cached type dictionary |
| `get_entity` (cache hit) | ~5.4 µs | LRU hit; no SQLite I/O |
| `describe_entity` | ~5.4 µs | Entity + incident relations |
| `search_relations` (from / from+type) | ~6.3 µs | Covering index scan |
| `delete_observations` (1) | ~11 µs | |
| `find_all_paths` (A→C, depth 5) | ~12 µs | Bounded DFS |
| `upsert_entities` (type change + obs) | ~27 µs | |
| `entities_exist` (10 names) | ~38 µs | Hash lookups |
| `batch_get_entities` (10) | ~42 µs | Batch fetch |
| `neighbors` (depth 1 / depth 2) | ~50 µs | Index-only covering scan |
| `open_nodes` (single / 5 names) | ~53–77 µs | LRU + SQLite |
| `search_nodes` (name match) | ~96 µs | FTS5 query + entity lookup |
| `add_observations` (2) | ~163 µs | Append + FTS index |
| `search_nodes` (obs match) | ~161 µs | FTS5 over observation bodies |
| `find_path` (BFS) | ~453 µs | Worst case: full BFS |
| `search_nodes` (filtered) | ~623 µs | FTS5 + type filter |
| `export` (JSON) | ~2.5 ms | Serialize all entities + relations |
| `read_graph` (all) | ~3.4 ms | Full dump |
| `create_relations` (999) | ~10 ms | Batch write + degree updates |
| `create_entities` (1000) | ~41 ms | Batch write + FTS index |

## Tools

### Knowledge-graph tools (always available)

**Write:** `create_entities`, `create_relations`, `add_observations`,
`delete_entities`, `delete_observations`, `delete_relations`, `upsert_entities`,
`merge_entities`, `compact`.

**Read:** `read_graph`, `search_nodes`, `open_nodes`, `batch_get_entities`,
`get_entity`, `entity_exists`, `graph_stats`, `search_relations`,
`describe_entity`, `degree`, `find_path`, `find_all_paths`, `extract_subgraph`,
`get_neighbors`, `list_entity_types`, `list_relation_types`, `export_graph`.

### Vector tools (`--vectors` only)

- `vector_upsert_embedding` — attach/replace an embedding on an existing entity
- `vector_batch_upsert` — bulk-upsert up to 1,024 embeddings; per-item error reporting
- `vector_get_embedding` — fetch the stored embedding (and model) for an entity
- `vector_search_entities` — top-K nearest entities by vector similarity (optional type filter)
- `vector_search_by_entity` — "more like this": nearest to an entity's own embedding
- `vector_recommend` — example-based recommendation from positive/negative entities
- `vector_mmr_search` — diversified retrieval via Maximal Marginal Relevance (`lambda`)
- `hybrid_search` — vector + FTS5 fused by RRF, optional graph-centrality boost
- `vector_delete_embedding` — remove an entity's embedding (entity is kept)
- `vector_reindex` — retrain the IVF index over current vectors (no-op for HNSW)
- `vector_refresh_graph_cache` — rebuild the petgraph adjacency cache from relations
- `vector_store_stats` — embedding count, dimension, backend kind, index/graph sizes

## Architecture

```
main.rs / vec_main.rs → MCPServer { kg, vs: Option<VectorStore> }
  ├── run_stdio()  — newline-delimited JSON-RPC over stdio
  ├── run_tcp()    — same framing, concurrent connections
  └── run_http()   — MCP Streamable HTTP (axum, POST/GET /mcp)
        └── process_request()
              ├── "initialize"      → protocol version + capabilities
              ├── "tools/list"      → cached tool list
              ├── "tools/call"      → dispatch to handler by name
              ├── "ping"            → null
              └── "notifications/…" → no reply
```

All transports share the transport-agnostic dispatch core
(`dispatch_line()` / `dispatch_http_body()`).

### Concurrency & locking

- `GraphHandle` uses `parking_lot::Mutex` for the writer connection and caches; a
  read-only connection pool serves concurrent reads under WAL.
- The `VectorStore` uses `DashMap` for name↔ID maps and an `RwLock` over the
  petgraph cache; the HNSW index is internally synchronized, the IVF index behind
  its own `RwLock`. Vector tools are gated behind `--vectors`; a pure-KG server
  carries no vector state.
- Heavy dispatch (graph lock + optional fsync) is offloaded to
  `tokio::task::spawn_blocking` to keep the reactor responsive.
- TCP connections are capped at 128 concurrent.

### Request size limits

| Parameter | Limit |
|---|---|
| Max request body | 16 MB |
| Name max bytes | 1,024 |
| Observation max bytes | 65,536 |
| Max entities / relations / observations / names per request | 1,000 |
| Max search limit | 1,000 |
| Max neighbor depth | 16 |
| Max `find_all_paths` depth / results | 10 / 100 |
| Max embedding dimensions *(`--vectors`)* | 4,096 |
| Max `topK` *(`--vectors`)* | 100 |
| Max items per `vector_batch_upsert` | 1,024 |

## Development

```sh
cargo test                       # 100+ unit + integration tests
cargo clippy                     # lint (lib + binaries)
cargo build --release            # LTO + fat, opt-level 3
cargo run --release --bin bench  # standalone benchmark
```

The test suite covers protocol handling, all tool handlers, CRUD/search/path
persistence, concurrency, fuzzy invariant checks, and — for the vector subsystem —
the IVF-Flat index (training, probe search, upsert/remove, metrics), both ANN
backends end-to-end, the modern retrieval tools (batch upsert, more-like-this,
recommend, MMR), vector gating when `--vectors` is off, input validation, the
tunable index config, and HTTP bearer-token authentication.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
