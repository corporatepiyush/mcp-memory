-- mcp-memory v3 — SQLite knowledge-graph schema
-- See docs/sqlite-architecture.md for the full rationale, pathway analysis,
-- page_size/zstd discussion, and the maintenance daemon.
--
-- DESIGN RULES IN FORCE
--   * INTEGER PRIMARY KEY everywhere it's the natural key (DECIDED: PKs help).
--     It IS the rowid: zero extra bytes in the row body, fastest access path,
--     and the only integer id stable across VACUUM. A stable rowid is also
--     REQUIRED for external-content FTS (content_rowid) to keep pointing at the
--     right row.
--   * NO foreign keys — the single writer enforces integrity in-transaction.
--   * `name_hash` (xxh3 of name, computed in Rust) is the UUID-like logical
--     handle for O(1) routing/dedup; it is an indexed column, never the key.
--   * Base data in ~3NF; the denormalized counters (obs_count, out_deg, in_deg,
--     type_dict.count, graph_stat) are explicitly-maintained materialized
--     aggregates for O(1) reads — maintained by the writer in the same txn.
--
-- STORAGE-MODEL NOTES (why there are no "smaller/bigger" int types to pick)
--   SQLite stores each *value* with a minimal "serial type": integers take
--   1/2/3/4/6/8 bytes by magnitude, and the constants 0 and 1 take ZERO bytes.
--   So `INTEGER` is already optimal per value — you never choose int widths.
--   `flags`/`kind`/small counts cost ~1 byte; an INTEGER PRIMARY KEY costs 0
--   bytes in the body (it's the B-tree key). REAL=8B float, BLOB=raw, TEXT=UTF-8.
--   STRICT tables enforce the declared affinity at ~no cost (catches bugs).

------------------------------------------------------------------------------
-- PRAGMAs (reference; applied by the Rust connection layer).
--   page_size MUST be set before the first table is created.
------------------------------------------------------------------------------
-- PRAGMA page_size    = 16384;        -- see §"page_size & zstd": big pages suit
--                                        large (compressed) observation bodies and
--                                        give shallower B-trees; hot reads are RAM,
--                                        so cold read-amplification is tolerable.
-- PRAGMA journal_mode = WAL;          -- many readers + one writer, no blocking
-- PRAGMA synchronous  = NORMAL;       -- fsync on checkpoint, not every commit
-- PRAGMA cache_size   = -50000;       -- ~50 MB page cache per connection (tune)
-- PRAGMA mmap_size    = 30000000000;  -- zero-copy of *uncompressed* pages (blocking pool)
-- PRAGMA temp_store   = MEMORY;
-- PRAGMA busy_timeout = 5000;
-- PRAGMA foreign_keys = OFF;          -- explicit: integrity enforced in app

------------------------------------------------------------------------------
-- CORE TABLES
------------------------------------------------------------------------------

-- Entities. id = rowid (physical key, 0 body bytes). name_hash = xxh3(name).
-- obs_count/out_deg/in_deg are materialized aggregates (writer-maintained) so the
-- MetaCache answers existence/type/degree/obs_count without touching child tables.
CREATE TABLE entity (
    id          INTEGER PRIMARY KEY,
    name_hash   INTEGER NOT NULL,
    name        TEXT    NOT NULL,
    type_id     INTEGER NOT NULL,
    obs_count   INTEGER NOT NULL DEFAULT 0,
    out_deg     INTEGER NOT NULL DEFAULT 0,
    in_deg      INTEGER NOT NULL DEFAULT 0,
    created_us  INTEGER NOT NULL,
    updated_us  INTEGER NOT NULL,
    flags       INTEGER NOT NULL DEFAULT 0   -- bit0 = soft-deleted
) STRICT;

-- name -> id resolution AND MetaCache warm-up. Covering + partial: includes the
-- columns the cache needs (rowid is implicit in every index), so both the
-- per-lookup seek and the full warm-up scan are INDEX-ONLY (no table reads), and
-- tombstones are skipped. (EXPLAIN QUERY PLAN: "COVERING INDEX entity_by_hash".)
CREATE INDEX entity_by_hash
    ON entity(name_hash, type_id, obs_count, out_deg, in_deg)
    WHERE flags = 0;

-- Optional case-insensitive / prefix name lookup (expression + partial index).
CREATE INDEX entity_name_ci ON entity(lower(name)) WHERE flags = 0;

-- Observations: clean 1:N child (3NF). id PK gives a stable rowid for the
-- external-content FTS below. `body` is the large column → the zstd target.
CREATE TABLE observation (
    id          INTEGER PRIMARY KEY,
    entity_id   INTEGER NOT NULL,
    idx         INTEGER NOT NULL,          -- position within the entity
    body        TEXT    NOT NULL,
    created_us  INTEGER NOT NULL
) STRICT;
CREATE INDEX obs_by_entity ON observation(entity_id, idx);

-- Relations: directed, typed. The two covering indexes ARE sorted adjacency
-- lists (SQLite's B-tree = CSR equivalent): index-only neighbor scans with O(1)
-- cursor pagination. (EQP: "COVERING INDEX rel_out (from_id=? AND type_id=? AND
-- to_id>?)".)
CREATE TABLE relation (
    from_id     INTEGER NOT NULL,
    to_id       INTEGER NOT NULL,
    type_id     INTEGER NOT NULL,
    created_us  INTEGER NOT NULL
) STRICT;
CREATE INDEX rel_out ON relation(from_id, type_id, to_id);   -- out-neighbors / dedup
CREATE INDEX rel_in  ON relation(to_id,   type_id, from_id); -- in-neighbors

------------------------------------------------------------------------------
-- FULL-TEXT SEARCH (FTS5, EXTERNAL-CONTENT) — write-cheap, no duplicated text.
-- External content stores only the index and reads/needs the source row by
-- content_rowid; the writer syncs it in-txn (insert / 'delete' commands). This
-- requires the stable INTEGER PRIMARY KEY on the content tables (above).
--   search: SELECT rowid, bm25(obs_fts) FROM obs_fts WHERE obs_fts MATCH ?
--           ORDER BY bm25(obs_fts) LIMIT :N;   -- rowid = observation.id
--   then join observation.id -> entity_id, merge with name_fts, rerank in Rust.
-- bm25() is NEGATIVE (more negative = more relevant); ORDER BY ascending = best first.
------------------------------------------------------------------------------
CREATE VIRTUAL TABLE obs_fts  USING fts5(body, content='observation', content_rowid='id',
                                         tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE name_fts USING fts5(name, content='entity',      content_rowid='id',
                                         tokenize='unicode61 remove_diacritics 2');

------------------------------------------------------------------------------
-- METADATA TABLES (mapped into RAM caches by the Rust layer; writer-maintained).
------------------------------------------------------------------------------

-- Interned entity/relation types + live counts. Loaded FULLY into RAM.
CREATE TABLE type_dict (
    id     INTEGER PRIMARY KEY,
    kind   INTEGER NOT NULL,               -- 0 = entity type, 1 = relation type
    name   TEXT    NOT NULL,
    count  INTEGER NOT NULL DEFAULT 0
) STRICT;
CREATE INDEX type_by_name ON type_dict(kind, name);

-- Singleton counters. WITHOUT ROWID = the row lives in the PK B-tree (no rowid
-- indirection) — ideal for a tiny text-keyed lookup table. Loaded into RAM atomics;
-- persisted so restart and graph_stats are O(1), never a scan.
CREATE TABLE graph_stat (
    key    TEXT NOT NULL PRIMARY KEY,       -- 'entities','relations','observations','entity_seq','obs_seq'
    value  INTEGER NOT NULL
) STRICT, WITHOUT ROWID;

-- Degree spill for TRUE hubs only (writer inserts past a threshold); backs the
-- hot hub-degree cache. Low-degree nodes are served from entity.out_deg/in_deg.
CREATE TABLE hub_degree (
    entity_id INTEGER PRIMARY KEY,
    out_deg   INTEGER NOT NULL,
    in_deg    INTEGER NOT NULL
) STRICT;

-- Partition map — populated only if we ever partition by entity_type (§5). The
-- TableRouter loads it to know which physical tables exist for query planning.
CREATE TABLE partition_map (
    table_name TEXT NOT NULL PRIMARY KEY,
    role       INTEGER NOT NULL,           -- 0=entity, 1=relation, 2=fts
    type_id    INTEGER,
    row_count  INTEGER NOT NULL DEFAULT 0
) STRICT, WITHOUT ROWID;

------------------------------------------------------------------------------
-- ZSTD (sqlite-zstd, transparent row-level compression) — apply at runtime, only
-- to the large cold column. Decompression is paid per row read, so keep it OFF
-- the hot path (metadata, indexes, names) and ON observation bodies:
--   SELECT zstd_enable_transparent('{"table":"observation","column":"body",
--          "compression_level":19,"dict_chooser":"''entity_'' || (entity_id % 64)"}');
-- Indexes are never compressed (so keep them lean). FTS shadow tables hold the
-- inverted index, not the bodies, so they are unaffected. See §"page_size & zstd".
------------------------------------------------------------------------------

-- Seed counters so reads never special-case "missing".
INSERT INTO graph_stat(key, value) VALUES
    ('entities', 0), ('relations', 0), ('observations', 0),
    ('entity_seq', 0), ('obs_seq', 0);
