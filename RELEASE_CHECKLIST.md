# Release Checklist

## Safety — unbounded growth

Before every release, audit every public-facing handler for unbounded
memory / output growth. Check each of these:

### Arrays from user input
- [ ] `handle_open_nodes` — `names` truncated at `MAX_NAMES_PER_REQUEST` (1000)
- [ ] `handle_entity_exists` — `names` truncated at `MAX_NAMES_PER_REQUEST` (1000)
- [ ] `handle_batch_get_entities` — `names` truncated at `MAX_NAMES_PER_REQUEST` (1000)
- [ ] `handle_delete_entities` — `entityNames` capped
- [ ] `handle_delete_observations` — `deletions` array capped

### Limit / offset parameters
- [ ] `handle_read_graph` — `limit` clamped at `MAX_SEARCH_LIMIT` (1000)
- [ ] `handle_search_nodes` — `limit` clamped at `MAX_SEARCH_LIMIT` (1000)
- [ ] `handle_search_relations` — results truncated at `MAX_RELATION_SEARCH_RESULTS` (1000)

### Path traversal depth
- [ ] `handle_find_all_paths` — `maxDepth` clamped at `MAX_FIND_ALL_PATHS_DEPTH` (10)
- [ ] `handle_find_all_paths` — `maxPaths` clamped at `MAX_FIND_ALL_PATHS_RESULTS` (100)
- [ ] `handle_get_neighbors` — `depth` capped at `MAX_NEIGHBOR_DEPTH` (16)
- [ ] `handle_extract_subgraph` — `depth` capped at `MAX_NEIGHBOR_DEPTH` (16)

### Cumulative DB growth (no handler fix — document)
- [ ] Observations per entity — no cumulative cap (1000 per request, unlimited requests)
- [ ] `export` / `read_graph` with no filter — serialises large portions of the DB

### Internal
- [ ] LRU cache bounded (`--lru-cache-size`, default 10000)
- [ ] SQLite page cache bounded (`PRAGMA cache_size = -50000`)
- [ ] Request body capped at `MAX_REQUEST_BYTES` (16 MB)
- [ ] `out` Vec in `serve_line_conn` cleared per iteration

## Procedure
1. Run `cargo clippy` and fix all warnings.
2. Run `cargo test` — all 40 unit + 8 e2e tests must pass.
3. Verify tool descriptions in `tools.json` match actual behaviour.
4. Bump version in `Cargo.toml`.
5. Run `cargo build --release`.
