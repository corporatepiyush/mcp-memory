# mcp-memory

**Persistent memory, a knowledge graph, code intelligence, and semantic search for LLM agents — in a single ~Rust binary backed by one embedded SQLite file.**

[![crates.io](https://img.shields.io/crates/v/mcp-memory.svg)](https://crates.io/crates/mcp-memory)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-2025--11--25-purple.svg)](https://modelcontextprotocol.io)

`mcp-memory` is a [Model Context Protocol](https://modelcontextprotocol.io) server that gives
your agent a long-term brain. It remembers **entities, relations, and observations** in a
queryable **knowledge graph**, indexes your **codebase** with tree-sitter, and serves
**vector / hybrid semantic search** — all from one file, with no database to run, no service to
deploy, and no telemetry.

Drop it into Claude Desktop, Claude Code, or any MCP client and your agent stops forgetting.

---

## Why mcp-memory

- 🧠 **Real memory, not a scratchpad.** A typed knowledge graph — entities, directed relations,
  and free-form observations — with FTS5 full-text search and graph traversal (paths, neighbors,
  subgraphs, centrality). Survives restarts; portable as a single file.
- ⚡ **Fast and embedded.** Pure Rust on SQLite in WAL mode. Sub-microsecond cache hits,
  microsecond reads, batched writes. No external services, no network round-trips, no daemons.
- 🔎 **Semantic + hybrid search.** Bring your own embeddings; the server indexes them in a
  [usearch](https://github.com/unum-cloud/usearch) **HNSW** (or IVF-Flat) index and fuses vector
  similarity with full-text relevance and graph centrality — RAG retrieval, more-like-this,
  recommendations, and MMR diversification included.
- 🗺️ **Code intelligence built in.** Point it at a repo and it parses **10 languages** with
  tree-sitter into a searchable symbol + call graph — then optionally embed symbols for
  meaning-based code search. A live, incremental, token-cheap map of your codebase.
- 🔌 **MCP-native and safe by default.** Speaks MCP `2025-11-25` over **stdio** and
  **Streamable HTTP** (with bearer-token auth and TLS). Tools are **opt-in by category**, so the
  server only ever exposes what you turn on.

---

```
                    ┌────────────────────────────────────────────────────┐
                    │   mcp-memory   (one binary · one SQLite file)        │
                    │                                                      │
     ┌────────┐     │  ┌──────────┐    ┌─────────────────────────────┐    │
     │ Claude │─────┼─▶│  stdio   │───▶│ GraphHandle                 │    │
     │  / LLM │     │  │   or     │    │  ├ LRU entity cache         │    │
     │  agent │     │  │  HTTP    │    │  ├ FxHashMap  name → id     │    │
     └────────┘     │  └────┬─────┘    │  └ FTS5 full-text index     │    │
                    │       │          └──────────────┬──────────────┘    │
                    │       ▼      (--enable-vectors)  │                   │
                    │  ┌─────────┐   ┌────────────────┴──────────────┐    │
                    │  │ dispatch│──▶│ VectorStore                   │    │
                    │  └─────────┘   │  ├ ANN: HNSW *or* IVF-Flat    │    │
                    │       │        │  └ petgraph adjacency cache   │    │
                    │       ▼        └────────────────┬──────────────┘    │
                    │  ┌─────────────────────────────────────────────┐    │
                    │  │ SQLite (WAL · 4 KB pages · auto_vacuum)      │    │
                    │  │ entity · observation · relation · *_fts ·    │    │
                    │  │ type_dict · vector_embedding                 │    │
                    │  └─────────────────────────────────────────────┘    │
                    └────────────────────────────────────────────────────┘
```

## Installation

```sh
cargo install mcp-memory
```

This installs the `mcp-memory` binary. (The `code` feature is on by default; build with
`--no-default-features` for a lean pure-memory binary without tree-sitter grammars.)

## Quick start

```sh
# Knowledge-graph memory (read + write)
mcp-memory --transport stdio --enable-graph-read --enable-graph-write

# Memory + semantic vector search
mcp-memory --transport stdio --enable-graph-read --enable-graph-write \
  --enable-vectors --embedding-dims 384

# Everything on — memory + vectors + code intelligence
mcp-memory --transport stdio --enable-all
```

### Use it from Claude Desktop / Claude Code

```json
{
  "mcpServers": {
    "memory": {
      "command": "mcp-memory",
      "args": ["--enable-all"]
    }
  }
}
```

That's it — your agent now has persistent memory and can index code. Trim `--enable-all` to just
the categories you want (see below).

## Tools are opt-in by category

**Nothing is exposed until you enable its category.** Disabled tools are hidden from `tools/list`
and rejected from `tools/call` as if they never existed — least privilege by default.

| Flag | Category | Tools |
|------|----------|-------|
| `--enable-graph-read` | **graph-read** | `read_graph`, `search_nodes`, `open_nodes`, `get_entity`, `graph_stats`, `search_relations`, `find_path`/`find_all_paths`, `get_neighbors`, `describe_entity`, `list_entity_types`, `list_relation_types`, `export_graph`, `extract_subgraph`, `batch_get_entities`, `entity_exists`, `degree` |
| `--enable-graph-write` | **graph-write** | `create_entities`, `create_relations`, `add_observations`, `delete_entities`, `delete_observations`, `delete_relations`, `upsert_entities`, `merge_entities`, `compact` |
| `--enable-vectors` | **vectors** | `vector_*` + `hybrid_search` (usearch HNSW or IVF-Flat) |
| `--enable-code` | **code** | `code_index`, `code_outline`, `code_search`, `code_get_symbol`, `code_watch`, `code_embed`, `code_semantic_search` |
| `--enable-all` | *(all)* | Every category. Overrides the individual flags. |

The database path is resolved in order:

1. `--memory-file` / `-f` flag
2. `MEMORY_FILE_PATH` environment variable
3. Default: `memory.mcpmem` in the working directory

The same SQLite file works with or without `--enable-vectors`, so you can populate the graph
plain and later serve it with vectors enabled.

### Transports

| Transport | Flag | Description |
|-----------|------|-------------|
| stdio | `--transport stdio` | Newline-delimited JSON-RPC over stdin/stdout (default; for Claude Desktop / Claude Code) |
| http | `--transport http --bind 0.0.0.0:8080` | MCP Streamable HTTP (POST/GET `/mcp`, SSE) |

The stdio transport dispatches up to `--stdio-concurrency` requests in parallel (default 8), so
clients that pipeline requests get concurrent execution; responses are correlated by JSON-RPC id
and may arrive in completion order. Set `--stdio-concurrency 1` for strict request/response
ordering (e.g. when pipelining order-dependent writes without awaiting each response).

### Authentication

The `http` transport accepts an optional bearer token (stdio is never authenticated). Set it with
`--auth-token`, `--auth-token-file` (trimmed; an empty file is rejected), or the
`MCP_MEMORY_AUTH_TOKEN` environment variable.

```sh
mcp-memory --enable-all --transport http --bind 0.0.0.0:8080 --auth-token "s3cr3t"
```

On HTTP the token is sent as `Authorization: Bearer <token>`; comparison is constant-time.
Binding a non-loopback address **without** a token exposes the entire graph to the network.

### TLS (HTTPS)

The `http` transport can be served over TLS (rustls, `ring` provider). Provide a PEM certificate
chain and private key via `--tls-cert` / `--tls-key` (both required together, or startup is
refused); the `MCP_TLS_CERT` / `MCP_TLS_KEY` environment variables are accepted as fallbacks.

```sh
mcp-memory --enable-all --transport http --bind 0.0.0.0:8080 \
  --tls-cert ./cert.pem --tls-key ./key.pem
```

### Web UI (graph viewer)

The `http` transport serves a **Neo4j-Browser-style knowledge-graph viewer** — open
[`http://<bind>/ui`](http://127.0.0.1:8080/ui) in any browser to explore the graph interactively:

- A **force-directed** layout with pan / zoom (scroll or the on-canvas ＋ / − / ⤢ controls) and
  drag-to-pin nodes.
- **Captioned circular nodes** coloured by entity type (the Neo4j categorical palette), a live
  **legend**, and curved multi-edges with **relationship-type labels + arrowheads**.
- **Double-click a node to expand its relationships** — incremental graph traversal that pulls the
  node's neighbourhood from the server and merges it into the view (start small, expand outward).
- **Paginated browse + full-text search.** Page through the graph with Prev / Next, or search all
  entities (FTS5, prefix / search-as-you-type) — both paginated, so large graphs stay responsive.
- A **node inspector** (type, observations, relationships — click a relationship to jump), plus
  **Isolate** / **Dismiss** actions, a label filter, and Esc-to-deselect.

It is served as three static assets — `index.html`, `graph.css`, `graph.js` — with **no external
dependencies** (no CDNs, no telemetry; everything renders locally on a `<canvas>`). The viewer is a
distinct browser front-end: it talks only to the `/ui/*` HTTP routes below and adds **no MCP tools**
and no stdio behaviour.

| Route | Purpose |
|-------|---------|
| `GET /ui` | The viewer page (app shell + `/ui/graph.css` + `/ui/graph.js`; carries no graph data, so it needs no auth). |
| `GET /ui/graph` | A page of the graph: `{ entities, relations, entityTypes, stats, page }`. Query params: `entityType` (filter), `offset`, `limit` (≤ 1,000), `token`. |
| `GET /ui/search` | A page of FTS5 matches (matched nodes only): same shape as `/ui/graph`. Query params: `q` (prefix-matched), `entityType`, `offset`, `limit` (≤ 1,000), `token`. |
| `GET /ui/expand` | One node's neighbourhood `{ entities, relations }` for double-click traversal. Query params: `name` (required), `depth` (1–3), `direction` (`outgoing`/`incoming`/`both`), `token`. |

Every data response carries a `page` cursor — `{ offset, limit, returned, hasMore }` — that drives
the Prev / Next controls without a second round-trip.

The viewer reads the graph, so `/ui/graph`, `/ui/search`, and `/ui/expand` require
**`--enable-graph-read`** (or `--enable-all`); without it they return `403` and the page says so.
They honor the same bearer token
as the MCP endpoints: pass it as `Authorization: Bearer <token>`, as a `?token=` query parameter, or
open `http://<bind>/ui#token=<token>` — the `#`-fragment stays client-side (never sent to the server
or written to logs) and the page forwards it as a header.

```sh
mcp-memory --enable-graph-read --transport http --bind 127.0.0.1:8080
# then open http://127.0.0.1:8080/ui in a browser
```

## Code intelligence (`--enable-code`)

Point the server at a source tree and it parses it with **tree-sitter** into a persistent,
searchable **code map** — symbols, signatures, and a call graph — turning the memory server into a
token-cheap navigator for terminal coding agents (Claude Code, opencode, codex, …). Because
symbols are ordinary graph entities, every graph tool (`search_nodes`, `extract_subgraph`,
`get_neighbors`, `find_path`, and `hybrid_search`) works on code for free.

- **What it stores.** Functions, classes, methods, modules, and constants become entities named
  `relpath::symbol` with type `code:<kind>`. Metadata (file, line range, signature, first doc
  line, language) lives in observations. Edges: `defines` (file→symbol) and `calls`/`references`
  (caller→callee). Bodies are **not** stored by default — only signatures and line ranges, so an
  agent reads the exact lines on demand (far fewer tokens than grep-then-read-whole-file).
- **Semantic code search.** Pass `code_index {"snippets": true}` to also store each symbol's
  bounded body text; embed those with your model via `code_embed`, then `code_semantic_search`
  does ANN (usearch **HNSW**) lookup to find code *by meaning*. Embeddings live in the same
  per-project database, keyed by symbol id; dimension defaults to **768** (`--code-embedding-dims`).
- **Incremental & live.** Each file's content hash is stored, so re-indexing only re-parses what
  changed. `code_watch` keeps the map fresh automatically, re-indexing on save (debounced).
- **Honest edges.** A `calls` edge is created only when the callee name resolves to exactly one
  definition; ambiguous references are dropped rather than recorded as false edges. Call edges are
  most complete after indexing the whole repo root in one pass.
- **Project isolation.** Each project is a dedicated, independent database — index many repos
  without collisions.
- **10 languages.** Rust, Python, JavaScript, TypeScript/TSX, Go, Java, C, C++, Ruby, PHP. Header
  files are indexed alongside sources. The walk honors `.gitignore` and skips
  `target`/`node_modules`/`dist`/`build` and oversized files.

| Tool | Purpose |
| --- | --- |
| `code_index` | Parse a file/dir into the graph (incremental; `force` to re-parse all, `snippets` to store bodies). |
| `code_outline` | List the symbols defined in one file (kind, lines, signature). |
| `code_search` | Full-text search over symbols → compact location rows (filter by `kind`/`lang`). |
| `code_get_symbol` | A symbol's metadata plus its callers and callees. |
| `code_watch` | Index a directory and re-index changed files on save (debounced). |
| `code_embed` | Attach client-computed embeddings to indexed symbols (batch). |
| `code_semantic_search` | ANN (HNSW) search over embedded symbols by a query vector. |

```bash
mcp-memory --enable-code --transport stdio
# then, over MCP:  code_index {"path": "src", "project": "my-repo"}
```

## Semantic & hybrid search (`--enable-vectors`)

Layer a vector store on top of the knowledge graph. Each embedding attaches to an existing entity
by name, is indexed in an in-memory ANN index, and persists as a blob in SQLite — rebuilt on
startup.

- **Bring your own embeddings.** The server stores and searches vectors; it does not call an
  embedding model. Compute embeddings client-side and pass them in (all must match
  `--embedding-dims`).
- **Semantic search** — `vector_search_entities` returns nearest entities by cosine similarity
  (configurable), optionally filtered by type.
- **More-like-this & recommendations** — `vector_search_by_entity` finds entities similar to a
  given one; `vector_recommend` builds a query from positive (minus negative) examples.
- **MMR diversification** — `vector_mmr_search` balances relevance against novelty (Maximal
  Marginal Relevance), suppressing near-duplicate hits during RAG context selection.
- **Batch ingestion** — `vector_batch_upsert` upserts up to 1,024 embeddings per call with
  per-item error reporting.
- **Hybrid search** — `hybrid_search` runs vector and FTS5 search in parallel, fuses them with
  Reciprocal Rank Fusion, and optionally boosts by graph centrality.

### HNSW vs IVF-Flat vs TurboQuant

| Backend | When to use | Notes |
|---|---|---|
| `hnsw` *(default)* | Best recall/latency for most workloads | usearch graph index; `f16`/`bf16`/`i8` quantization |
| `ivf` | Large, batch-ingested, periodically-rebuilt corpora | k-means partitioned; cheaper to build, lighter memory. **Exact until trained**, so results are always correct |
| `turbo` | Memory-bound corpora; online ingestion | [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research): data-oblivious quantization to `--tq-bits` bits/coordinate (~8× smaller than `f32` at 4 bits) with **unbiased** inner-product estimates and near-optimal distortion. Zero training/indexing time; brute-force scan over compact codes. Requires `--embedding-dims` 384–1536 |

The IVF index trains automatically when a populated database is opened; after a large batch
ingestion into a fresh database, call `vector_reindex` to keep recall high (no-op for HNSW and
TurboQuant — the latter is data-oblivious, so there is never anything to train).

### Tuning

All require `--enable-vectors`:

| Flag | Default | Meaning |
|---|---|---|
| `--embedding-dims` | `384` | Vector dimension; all embeddings must match |
| `--vec-index` | `hnsw` | ANN backend: `hnsw`, `ivf`, or `turbo` |
| `--vec-metric` | `cos` | Distance metric: `cos`, `ip` (dot product), or `l2sq` |
| `--vec-quantization` | `f32` | HNSW scalar storage: `f32`, `f16`, `bf16`, or `i8` |
| `--vec-connectivity` | `16` | HNSW graph degree `M` (higher = better recall, more memory) |
| `--vec-expansion-add` | `200` | HNSW `efConstruction` (higher = better quality, slower inserts) |
| `--vec-expansion-search` | `50` | HNSW `efSearch` (higher = better recall, slower queries) |
| `--ivf-nlist` | `256` | IVF number of Voronoi cells / centroids |
| `--ivf-nprobe` | `8` | IVF cells probed per query (higher = better recall, slower) |
| `--tq-bits` | `4` | TurboQuant bits per coordinate, 1–8 (higher = better recall, more memory). TurboQuant requires `--embedding-dims` in 384–1536 |

```sh
# HNSW with half-precision storage
mcp-memory --enable-vectors --transport http --bind 0.0.0.0:8080 \
  --embedding-dims 768 --vec-metric cos --vec-quantization f16 \
  --vec-connectivity 32 --vec-expansion-search 128

# IVF-Flat for a large corpus
mcp-memory --enable-vectors --embedding-dims 768 \
  --vec-index ivf --ivf-nlist 1024 --ivf-nprobe 16

# TurboQuant: ~8x memory reduction with unbiased inner-product scoring
mcp-memory --enable-vectors --embedding-dims 768 \
  --vec-index turbo --tq-bits 4
```

## MCP compliance

Implements [MCP](https://modelcontextprotocol.io) revision **`2025-11-25`** over JSON-RPC 2.0,
via stdio or HTTP.

| Area | Support |
|---|---|
| Transports | stdio, **Streamable HTTP** (POST/GET `/mcp`, SSE) |
| Protocol version | `2025-11-25`, negotiates down to `2025-06-18` / `2025-03-26` / `2024-11-05` |
| `initialize` | version negotiation + `instructions` |
| `tools/list`, `tools/call` | opt-in by category (`--enable-*`) |
| `CallToolResult` | `content[]` + `isError` |
| Auth | optional bearer token on HTTP (constant-time) |
| Capabilities | `tools` |

Tool failures are returned as `CallToolResult`s with `isError: true` (not as JSON-RPC protocol
errors) so the model can read the message and self-correct.

## Data model

```
Entity(name, entityType, observations[])   ──relationType──▶   Entity(...)
```

- **Entity** — a named node with a type (e.g. `person`, `company`, `project`) and free-form
  observation strings. Names are unique and case-sensitive.
- **Relation** — a directed edge `(from, to, relationType)`. Traversal is undirected (BFS/DFS
  follow both directions).
- **Observation** — an unstructured fact attached to an entity.
- **Embedding** *(`--enable-vectors`)* — a fixed-dimension `f32` vector attached to an entity, plus
  an optional model identifier.

Search uses FTS5 with `unicode61 remove_diacritics 2` tokenization. Names and observation bodies
live in separate external-content FTS5 tables (`name_fts`, `obs_fts`).

## Storage & performance

### SQLite (WAL mode)

| Table | Key | Purpose |
|---|---|---|
| `entity` | rowid | Primary storage; materialized `obs_count`/`out_deg`/`in_deg`; `name_hash` for O(1) routing |
| `observation` | `entity_id` (FK) | 1:N observations per entity |
| `relation` | composite indexes | Directed edges; covering indexes `rel_out`/`rel_in` for index-only scans |
| `name_fts` / `obs_fts` | `content_rowid` | External-content FTS5 over names / observation bodies |
| `type_dict` | name | Interned entity/relation types with live counts (RAM-loaded) |
| `graph_stat` | key | `WITHOUT ROWID` counters: entities, relations, observations, sequences |
| `vector_embedding` | `entity_id` | *(`--enable-vectors`)* `dims`, `blob`, `model`, `created_us` |

Key pragmas (defaults, all tunable): `page_size=4096`, `journal_mode=WAL`,
`auto_vacuum=INCREMENTAL`, `synchronous=NORMAL`, `cache_size=-50000` (~50 MB), `mmap_size=256 MB`,
`temp_store=MEMORY`, `busy_timeout=5000`. A background `wal_checkpoint(PASSIVE)` runs every
`--wal-flush-ms` to bound the async durability window.

### In-memory caches

| Cache | Purpose |
|---|---|
| Entity LRU (10,000) | Avoids deserializing hot entities (`EntityMeta`) |
| Name-hash map | O(1) name→ID resolution via 64-bit hash |
| Prepared-statement cache | Reuses compiled SQLite queries |
| ANN index *(vectors)* | In-memory HNSW or IVF-Flat, rebuilt from `vector_embedding` on startup |
| petgraph adjacency *(vectors)* | Directed graph cache for the hybrid-search centrality boost |

### Write batching

Mutations go through a layered write path that collapses transaction count from O(N) to O(1) per
`create_entities` / `create_relations` call: batch existence checks → batch commit → batch FTS
updates → cache invalidation.

### Durability

| Mode | Behavior | Data-loss window |
|---|---|---|
| `async` (default) | Flush to kernel page cache, background sync | Up to ~1 s on power failure |
| `sync` | fsync before every write | Zero |

Set via `MCP_MEMORY_DURABILITY=sync`. A background task also runs every 5 minutes: WAL checkpoint
(TRUNCATE), planner analysis (`PRAGMA optimize`), and FTS optimization.

## Benchmarks

Measured end-to-end via the `bench` binary — 1,000 entities (5 observations each) + 999 relations
pre-populated, on a **MacBook Pro (Apple M1 Pro, 32 GB)**. Averages; run
`cargo run --release --bin bench` on your own hardware.

| Operation | Avg latency | Notes |
|---|---|---|
| `degree` (cache hit) | ~44 ns | Materialized column |
| `get_entity` (cache hit) | ~5.4 µs | LRU hit; no SQLite I/O |
| `search_relations` | ~6.3 µs | Covering index scan |
| `find_all_paths` (depth 5) | ~12 µs | Bounded DFS |
| `neighbors` (depth 1–2) | ~50 µs | Index-only covering scan |
| `search_nodes` (name match) | ~96 µs | FTS5 query + entity lookup |
| `find_path` (BFS) | ~453 µs | Worst case: full BFS |
| `read_graph` (all) | ~3.4 ms | Full dump |
| `create_relations` (999) | ~10 ms | Batch write + degree updates |
| `create_entities` (1000) | ~41 ms | Batch write + FTS index |

## Tools

### Knowledge-graph

**Write:** `create_entities`, `create_relations`, `add_observations`, `delete_entities`,
`delete_observations`, `delete_relations`, `upsert_entities`, `merge_entities`, `compact`.

**Read:** `read_graph`, `search_nodes`, `open_nodes`, `batch_get_entities`, `get_entity`,
`entity_exists`, `graph_stats`, `search_relations`, `describe_entity`, `degree`, `find_path`,
`find_all_paths`, `extract_subgraph`, `get_neighbors`, `list_entity_types`, `list_relation_types`,
`export_graph`.

### Vector (`--enable-vectors`)

`vector_upsert_embedding`, `vector_batch_upsert`, `vector_get_embedding`, `vector_search_entities`,
`vector_search_by_entity`, `vector_recommend`, `vector_mmr_search`, `hybrid_search`,
`vector_delete_embedding`, `vector_reindex`, `vector_refresh_graph_cache`, `vector_store_stats`.

### Code (`--enable-code`)

`code_index`, `code_outline`, `code_search`, `code_get_symbol`, `code_watch`, `code_embed`,
`code_semantic_search`.

## Architecture

```
main.rs → MCPServer { kg, vs: Option<VectorStore> }
  ├── run_stdio()  — newline-delimited JSON-RPC over stdio
  └── run_http()   — MCP Streamable HTTP (axum, POST/GET /mcp)
        ├── GET /ui        — graph viewer shell + /ui/graph.css + /ui/graph.js (static)
        ├── GET /ui/graph  — a paged view of the graph for the viewer (gated by graph-read)
        ├── GET /ui/search — paged FTS5 search for the viewer (gated)
        ├── GET /ui/expand — a node's neighbourhood for double-click traversal (gated)
        └── process_request()
              ├── "initialize"      → protocol version + capabilities
              ├── "tools/list"      → tool list (filtered by enabled categories)
              ├── "tools/call"      → dispatch to handler by name
              ├── "ping"            → null
              └── "notifications/…" → no reply
```

All transports share one transport-agnostic dispatch core (`dispatch_line()` /
`dispatch_http_body()`).

- **Concurrency.** `GraphHandle` uses a `parking_lot::Mutex` writer connection plus a read-only
  connection pool for concurrent reads under WAL. `VectorStore` uses `DashMap` for name↔ID and an
  `RwLock` over the petgraph cache; HNSW/IVF indexes are internally synchronized. Heavy dispatch
  (graph lock + optional fsync) is offloaded to `tokio::task::spawn_blocking` to keep the reactor
  responsive.

### Limits

| Parameter | Limit |
|---|---|
| Max request body | 16 MB |
| Name max bytes | 1,024 |
| Observation max bytes | 65,536 |
| Max entities/relations/observations/names per request | 1,000 |
| Max search limit | 1,000 |
| Max neighbor depth | 16 |
| Max `find_all_paths` depth / results | 10 / 100 |
| Max embedding dimensions *(vectors)* | 4,096 |
| Max `topK` *(vectors)* | 100 |
| Max items per `vector_batch_upsert` | 1,024 |

## Development

```sh
cargo test                       # unit + integration tests
cargo clippy --all-targets       # lint
cargo build --release            # LTO + fat, opt-level 3
cargo run --release --bin bench  # standalone benchmark
```

The suite covers protocol handling, every tool handler, CRUD/search/path persistence,
concurrency, fuzzy invariant checks, both ANN backends end-to-end, the retrieval tools (batch
upsert, more-like-this, recommend, MMR), category gating, code indexing across all 10 languages,
and HTTP bearer-token authentication.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
