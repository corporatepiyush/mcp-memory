# mcp-memory v3 — SQLite engine + Rust metadata cache

Status: **design / RFC**. Supersedes the custom‑mmap engine plan
(`scaling-architecture.md`) for the storage layer. The v2 doc's *analysis* still
stands (latency physics, the abstraction‑cost audit, the tiering); what changes is
the **engine**: instead of building a storage/WAL/recovery/lock‑free engine from
scratch, we let **SQLite** be the durable query engine and put a **Rust in‑memory
metadata cache** in front of it for the sub‑100 µs hot path.

This is a deliberate risk trade. We give up theoretical peak (single writer; cold
reads in the ms range, not µs) and gain: proven durability + crash recovery, WAL
MVCC, FTS5/BM25 for free, mature query planning, and *far* less code to get wrong.
For an agent‑memory workload (read‑dominated, write‑bursty, correctness‑critical)
that is the right trade.

The rest of this doc **comes clean** about where SQLite's behavior contradicts the
v3 README's instincts (especially partitioning and keys), then lays out a schema
— core tables *and* the metadata tables that map directly into RAM caches.

---

## 0. SQLite truths that constrain everything (read this first)

Design that ignores these is fiction. Each is load‑bearing.

1. **One writer, many readers (WAL).** In WAL mode readers run concurrently and
   never block the writer, and the writer never blocks readers — but there is
   **exactly one writer at a time**. Consequence: we do **not** fight for a
   multi‑writer design. We funnel all writes through **one writer thread** that
   **batches** them into transactions (group commit). Reads scale across a
   **read‑connection pool**. This is simpler and safer than v2's per‑shard write
   locks, and it fits a read‑dominated workload.

2. **SQLite is a blocking library.** Every SQLite call can touch disk and block
   the calling thread. Therefore SQLite calls run on a **blocking pool / the
   writer thread**, never on the async reactor. This *dissolves the v2 R1
   "mmap‑fault‑stalls‑the‑reactor" problem*: blocking is expected and isolated to
   pool threads. The reactor only ever touches the **RAM cache** (no SQLite call)
   on the hot path.

3. **`INTEGER PRIMARY KEY` is the rowid — the single fastest key in SQLite, not a
   bottleneck.** What is slow is a **TEXT primary key**, a **random‑UUID key**
   (destroys insert locality, bloats every index), and **FK constraint
   enforcement** (per‑write checks). See §2 — this is where we engage the "no
   PK/FK" directive head‑on.

4. **Partitioning into many tables is usually a *pessimization* in SQLite.** One
   writer ⇒ no write‑parallelism to gain. A table is one B‑tree; splitting 10M
   rows into 256 tables saves ~1 B‑tree level (negligible) while making every
   cross‑partition query a `UNION ALL` over 256 cursors (planner + cursor
   overhead), and every empty partition still costs a root page. See §5 — we
   **reject hash‑prefix partitioning** and keep partitioning to the few cases that
   actually pay.

5. **`VACUUM` renumbers plain rowids.** A bare `rowid` (table with no
   `INTEGER PRIMARY KEY`) can be **renumbered by `VACUUM`** — which would silently
   corrupt every `from_id`/`to_id`/`entity_id` reference we store. A stable
   integer id that survives VACUUM **requires `INTEGER PRIMARY KEY`**. This is the
   decisive technical reason §2 recommends `INTEGER PRIMARY KEY` for the physical
   id.

6. **FTS5 with `bm25()` exists and is good.** We do not hand‑roll an inverted
   index unless a benchmark forces us to. See §6.

---

## 1. Architecture at a glance

```
                         async reactor (tokio)
   10k clients ───► parse (borrow-only) ───► route
                                  │
              ┌───────────────────┴─────────────────────┐
              │  RAM hot path (NO SQLite call)           │   Tier A  < 100 µs
              │  • MetaCache:  name_hash → EntityMeta     │
              │  • TypeInterner, counters, degree(hubs)   │
              └───────────────────┬─────────────────────┘
                       hit ◄──────┘   miss / needs bodies / search
                                  │
              ┌───────────────────▼─────────────────────┐
              │  blocking pool                            │
              │  • READ pool: N WAL read connections      │   Tier B  ~ms
              │      - rowid fetch (fastest read)         │
              │      - covering-index scans               │
              │      - FTS5 MATCH + bm25()                │
              │  • WRITER: 1 connection, MPSC-fed,        │
              │      batched transactions (group commit)  │
              └───────────────────┬─────────────────────┘
                                  │ WAL
                              SQLite file  (page cache + optional mmap)
```

- **Hot reads never call SQLite.** The MetaCache answers existence, type, degree,
  obs_count, and rowid from RAM. Only fetching observation *bodies* or running a
  search drops to the read pool.
- **All writes are one thread.** It owns the writer connection, drains an MPSC
  queue, batches ops into transactions, then updates the RAM caches to match —
  single‑writer means **no atomic contention on the maintained counters/degree**.

---

## 2. Identity & keys — engaging the "no PK / no FK" directive

> **Your note, kept verbatim as a constraint:** *don't use primary keys and
> foreign keys in SQLite to avoid perf bottlenecks; use UUIDs or similar
> alternatives.*

Here is the honest decomposition, because "no keys" hides two very different
things:

- **Foreign keys → DROP them. Agreed, unconditionally.** FK enforcement is off by
  default in SQLite and, when on, adds a per‑write existence check against the
  referenced table. We enforce referential integrity in the **single writer**
  (e.g., deleting an entity also deletes its observations/relations/FTS row in the
  *same transaction*). No `FOREIGN KEY` clauses anywhere. ✔ matches your intent.

- **Primary keys → the nuance that matters.** The bottleneck you've heard about is
  real but specific: it is a **TEXT primary key** (entity name as the key) or a
  **random‑UUID key**. Both force large, randomly‑distributed keys → page splits,
  poor cache locality, fat secondary indexes. **But `INTEGER PRIMARY KEY` is not a
  separate key at all — it is the table's `rowid`, the fastest access path SQLite
  has, and it's the *only* way to get an integer id that survives `VACUUM`**
  (truth #5). Avoiding it doesn't remove a cost; it removes the fast path and
  risks rowid renumbering.

**Reconciliation (recommended, please veto if you disagree):**

| Identity | What | Why |
| :-- | :-- | :-- |
| **Physical id** | `id INTEGER PRIMARY KEY` (= stable rowid) | fastest reads/joins; stable across VACUUM; locality of monotonic insert |
| **Logical/dedup id** | `name_hash INTEGER` (xxh3 of name, indexed, *not* unique‑constrained) | O(1) name→id routing & dedup, computed in Rust, no central allocator |
| **External id (optional)** | a `uuid BLOB(16)` column if clients need a stable public handle | stored + indexed *as a column*, **never** as the physical key |

So: **no FKs, no TEXT key, no UUID‑as‑physical‑key** (your perf concern, honored),
**`INTEGER PRIMARY KEY` = rowid** as the physical id (the fast path), and a
**`name_hash` integer** as the UUID‑like logical handle. Relations and
observations reference the integer `id`. **DECIDED (per your call): keep the PK.**
A `name_hash` collision is disambiguated by an equality on `name` during the row
verify — exact and ~free.

> A stable `INTEGER PRIMARY KEY` is not just *nice* here, it is **required** by two
> things: (a) external‑content FTS5 keys its index by `content_rowid`, which must
> not move under `VACUUM`; (b) relations/observations store the id as a soft
> reference. A bare rowid renumbered by VACUUM would corrupt both. PK it is.

Collision handling: `name_hash` is not unique (64‑bit hash, rare collisions). A
name lookup is `WHERE name_hash = ? AND name = ?` — the index narrows to ~1 row,
the equality on `name` disambiguates. Exact, cheap.

---

## 3. Schema — core tables (no FK, implicit referential integrity)

Full DDL lives in [`schema.sql`](./schema.sql); the shape and rationale:

```sql
-- Entities. Physical id = rowid (INTEGER PRIMARY KEY); logical handle = name_hash.
CREATE TABLE entity (
    id          INTEGER PRIMARY KEY,   -- rowid; stable across VACUUM; the join key
    name_hash   INTEGER NOT NULL,      -- xxh3(name); routing & dedup
    name        TEXT    NOT NULL,
    type_id     INTEGER NOT NULL,      -- interned (type_dict.id)
    obs_count   INTEGER NOT NULL DEFAULT 0,
    out_deg     INTEGER NOT NULL DEFAULT 0,   -- maintained by writer (no triggers)
    in_deg      INTEGER NOT NULL DEFAULT 0,
    created_us  INTEGER NOT NULL,
    updated_us  INTEGER NOT NULL,
    flags       INTEGER NOT NULL DEFAULT 0     -- bit0=deleted (soft), ...
) STRICT;

-- name → id (+ enough columns to *cover* MetaCache warm-up without table reads).
CREATE INDEX entity_by_hash ON entity(name_hash, type_id, obs_count, out_deg, in_deg);

-- Observations: 1:N, keyed by entity id. No FK; writer cascades on delete.
CREATE TABLE observation (
    entity_id   INTEGER NOT NULL,
    idx         INTEGER NOT NULL,      -- position within the entity
    body        TEXT    NOT NULL,
    created_us  INTEGER NOT NULL
) STRICT;
CREATE INDEX obs_by_entity ON observation(entity_id, idx);

-- Relations: directed, typed. Two covering indexes = a sorted adjacency list
-- (SQLite's B-tree IS the CSR equivalent), giving O(log n + degree) neighbor scans
-- that paginate by cursor.
CREATE TABLE relation (
    from_id     INTEGER NOT NULL,
    to_id       INTEGER NOT NULL,
    type_id     INTEGER NOT NULL,
    created_us  INTEGER NOT NULL
) STRICT;
CREATE INDEX rel_out ON relation(from_id, type_id, to_id);   -- out-neighbors / dedup
CREATE INDEX rel_in  ON relation(to_id,   type_id, from_id); -- in-neighbors
```

Notes:
- `STRICT` tables enforce column types (catches bugs) at ~no cost.
- The `relation` indexes are the graph engine: `WHERE from_id=? [AND type_id=?]`
  is an index range scan = a sorted neighbor run, and `... AND (type_id,to_id) >
  (?,?) LIMIT k` is **O(1) cursor pagination** for hubs — the v2 "CSR + cursor"
  property, for free, from a B‑tree.
- Soft delete via `flags` bit keeps deletes cheap and lets a partial index skip
  tombstones (§4); a background sweep hard‑deletes.

---

## 4. Schema — metadata tables (built to live in RAM)

This is the heart of your ask: **tables designed so the Rust layer maps them into
caches that collapse lookup latency.** Each metadata table has a direct RAM
counterpart, a defined load‑at‑startup path, and a maintenance rule (updated by
the single writer in the *same transaction* as the data change, so they never
drift and recovery is O(metadata), not O(data)).

```sql
-- (1) Interned types + live counts. Tiny; loaded FULLY into RAM.
CREATE TABLE type_dict (
    id     INTEGER PRIMARY KEY,
    kind   INTEGER NOT NULL,   -- 0 = entity type, 1 = relation type
    name   TEXT    NOT NULL,
    count  INTEGER NOT NULL DEFAULT 0
) STRICT;
CREATE INDEX type_by_name ON type_dict(kind, name);

-- (2) Singleton graph counters. Loaded into RAM atomics; persisted so restart and
--     graph_stats are O(1) (no scan ever).
CREATE TABLE graph_stat (
    key    TEXT PRIMARY KEY,   -- 'entities','relations','observations'
    value  INTEGER NOT NULL
) STRICT, WITHOUT ROWID;       -- tiny keyed table; WITHOUT ROWID is ideal here

-- (3) Hub degrees — materialized ONLY for high-degree nodes (partial), so the
--     degree cache and hub-pagination decisions are O(1) without indexing 10M
--     low-degree nodes. (Low-degree `degree` is answered from entity.out_deg/in_deg
--     already in MetaCache; this table is the spill for true hubs if we want a
--     separate hot structure.)
CREATE TABLE hub_degree (
    entity_id INTEGER PRIMARY KEY,
    out_deg   INTEGER NOT NULL,
    in_deg    INTEGER NOT NULL
) STRICT;
-- (populated by the writer when out_deg+in_deg crosses a threshold)

-- (4) Partition map — only if/when we partition (see §5). The TableRouter loads
--     it to know which physical tables exist and their sizes for planning.
CREATE TABLE partition_map (
    table_name TEXT PRIMARY KEY,
    role       INTEGER NOT NULL,   -- 0=entity,1=relation,2=fts
    type_id    INTEGER,            -- nullable
    row_count  INTEGER NOT NULL DEFAULT 0
) STRICT, WITHOUT ROWID;
```

**RAM counterparts and how the metadata tables feed them:**

| Metadata table | RAM structure | Loaded at startup by | Maintained by | Serves |
| :-- | :-- | :-- | :-- | :-- |
| `type_dict` | `TypeInterner` (name⇆u32) + `type_counts` | full table scan (tiny) | writer, in‑txn | `list_*_types`, type filters, stats breakdown |
| `graph_stat` | counters (atomics / seqlock) | 3 point reads | writer, in‑txn | `graph_stats` in O(1), no scan |
| `entity_by_hash` (covering index) | **MetaCache** `name_hash → EntityMeta{id,type_id,obs_count,out_deg,in_deg,flags}` | **index‑only scan** (no table reads) | writer, post‑commit | `get_entity`/`entity_exists`/`degree` hot path |
| `hub_degree` | hub degree cache | partial scan | writer, in‑txn | hub‑pagination decisions |
| `partition_map` | `TableRouter` | full scan (small) | writer | query routing (if partitioned) |

Key property: **the MetaCache is built from a covering index, so warm‑up is an
index‑only scan** — at 10M entities that reads ~tens of MB of index pages, not the
full table. And because `graph_stat`/`type_dict` are persisted and maintained
transactionally, **a restart rebuilds RAM from metadata in milliseconds**, never a
full‑graph scan.

**MetaCache sizing / the 10 MB → 1 TB axis:**
- Small/medium graphs (≤ ~50–100 M entities): MetaCache is a **full resident
  index** — every entity's metadata in RAM, every existence/type/degree query is
  Tier A. ~32–40 B/entity in a custom open‑addressing table keyed by the u64
  `name_hash` (the hash *is* the key — no rehashing, no `HashMap<String>`):
  10 M → ~0.4 GB, 100 M → ~4 GB.
- Huge graphs (≫ RAM): MetaCache becomes a **bounded hot cache** (CLOCK/second‑
  chance, sized to a RAM budget). Misses fall to the `entity_by_hash` index in
  SQLite (Tier B, ~ms) and populate the cache. So RAM stays fixed while the graph
  grows to 1 TB — the same "RAM ∝ working set, not data volume" property as v2,
  but with SQLite (not a custom engine) holding the cold mass.

---

## 4A. SQLite storage model — there is no "int width" to pick (and that's good)

You asked: *only INTEGER and TEXT? no bigger/smaller types? no optimizations?* The
honest answer is that SQLite's optimization is **per‑value and automatic**, so
there is nothing to hand‑tune at the column level — which is *why* schema shape
(not column types) is where the performance lives.

- **Five storage classes:** `NULL`, `INTEGER`, `REAL` (8‑byte float), `TEXT`
  (UTF‑8), `BLOB` (raw). That's the whole value universe.
- **Per‑value "serial types" = automatic width.** Inside a row record each value
  carries a varint serial type that picks the *minimal* encoding: integers use
  **1, 2, 3, 4, 6, or 8 bytes by magnitude**, and the constants **0 and 1 take
  ZERO bytes**. So a `flags`/`kind`/small‑count column costs ~1 byte; you never
  declare `int8/int16/...` because SQLite already stores the smallest form that
  holds each value. There is no smaller type to choose and no win in trying.
- **`INTEGER PRIMARY KEY` costs 0 bytes in the body** — it *is* the B‑tree rowid,
  not a stored column. This is the cheapest possible key. (Confirmed by EQP:
  lookups read it straight from the index/rowid.)
- **`WITHOUT ROWID`** stores the row *in* its PK B‑tree (no rowid→row hop) — ideal
  for tiny keyed tables (`graph_stat`, `partition_map`). For the big tables the
  rowid form is faster, so we keep rowid there.
- **`STRICT`** enforces the declared affinity (catches type bugs) at ~no cost.
- **Large values overflow:** a value larger than ~`page_size` spills to an
  overflow page chain — which is exactly where `page_size` and zstd interact (§C).

Takeaway: model columns as plain `INTEGER`/`TEXT`/`BLOB`, let serial types size
them, and spend the optimization budget on **table shape + indexes + the cache** —
which §4B does pathway by pathway.

## 4B. Normalization & every read/write/delete pathway

Base data is **~3NF**: `entity`, `observation` (clean 1:N child), `relation`
(typed edge), and `type_dict` (types interned, not repeated as text — kills
redundancy). The only deliberate departures are **materialized aggregates**
(`obs_count`, `out_deg`, `in_deg`, `type_dict.count`, `graph_stat`): a normal form
would `COUNT(*)` on demand (correct but an index scan per call); we instead keep
writer‑maintained counters so the hot reads are O(1). That is the one place we
trade NF for latency, and it's explicit and writer‑owned.

Each pathway below names the **index used** and the **cost tier**. The plans are
verified with `EXPLAIN QUERY PLAN` (see commit notes); none falls back to a table
scan on the hot path.

**Reads**

| Tool | Plan | Index | Tier |
| :-- | :-- | :-- | :-- |
| `entity_exists` | RAM MetaCache; misses → `WHERE name_hash IN(..) AND flags=0` | `entity_by_hash` (index‑only) | A (RAM) / B |
| `get_entity` no bodies | MetaCache → `{id,type,obs_count,deg}` | — | A |
| `get_entity` + bodies | cache → rowid; `WHERE entity_id=? ORDER BY idx` | rowid + `obs_by_entity` | B |
| `degree` | MetaCache `out_deg/in_deg`; hubs from `hub_degree` | — | A |
| `get_neighbors` (page) | `WHERE from_id=? [AND type_id=?] AND to_id>? ORDER BY to_id LIMIT k` | `rel_out`/`rel_in` (index‑only, **O(1) cursor**) | B |
| `read_graph` (page) | `WHERE id>:cursor AND flags=0 ORDER BY id LIMIT k` | rowid range (**O(1) cursor**) | B |
| `search_nodes` | `obs_fts`/`name_fts` MATCH → ids → Rust rerank | FTS5 + bm25() | B |
| `graph_stats`, `list_*_types` | RAM counters / `type_dict` | — | A |
| `find_path`/`find_all_paths` | repeated neighbor scans | `rel_out` | B/C |

**Writes (all funneled to the single writer; batched into one txn = group commit)**

| Tool | Touches | Notes |
| :-- | :-- | :-- |
| `create_entities` | INSERT `entity`; per‑obs INSERT `observation`+`obs_fts`; INSERT `name_fts`; `type_dict.count`++, `graph_stat`++ | id from `entity_seq`; dedup via `name_hash` probe |
| `create_relations` | INSERT `relation` (rel_out+rel_in); `entity.out_deg`/`in_deg`++ for both ends; `graph_stat.relations`++ | dedup via `rel_out` probe; degree update = 2 row writes/edge (the price of O(1) `degree`) |
| `add_observations` | INSERT `observation`+`obs_fts`; `entity.obs_count`+=n, `updated_us` | append‑cheap — FTS is per‑row, no entity re‑index |
| `upsert_entities` | UPSERT `entity`; diff observations | invalidate MetaCache entry |
| `merge_entities` | re‑point `relation` rows; move observations; delete source | one txn; updates both endpoints' degrees |

**Deletes**

- **Default = transactional cascade** (correctness): DELETE `entity`, its
  `observation`s (+ `obs_fts` deletes), its `relation`s via `rel_out`+`rel_in`
  scans (+ neighbor `in_deg`/`out_deg`--), `name_fts` delete, counters/`type_dict`
  decrement — **one WAL txn**. Heavy only for hub deletes (rare).
- **Optional = soft delete** (latency): set `flags` bit + remove from FTS +
  decrement counters + evict from MetaCache (cheap, bounded); the **maintenance
  daemon (§D)** does the cascade + space reclamation later. Trade‑off: a window
  where a neighbor scan could surface an edge to a soft‑deleted node, repaired by
  the daemon. Partial indexes (`WHERE flags=0`) already hide soft‑deleted entities
  from `entity_by_hash`/`name_ci`/warm‑up.

The single writer makes every one of these atomic *without* FKs, and keeps the
materialized aggregates exact across crashes (persisted in the same txn).

## 5. Partitioning — all permutations, and the honest verdict

The README proposes 256 hash‑prefix partitions per type. Evaluated against truth
#4, that is the wrong pattern *for SQLite*. Here is the full matrix:

| Scheme | Real benefit in SQLite | Real cost | Verdict |
| :-- | :-- | :-- | :-- |
| **Single table + covering indexes** | one B‑tree; cross‑cutting queries trivial; planner happy | none beyond index upkeep | **DEFAULT** |
| **Hash‑prefix (256/type)** | ~1 fewer B‑tree level (negligible); *no* write parallelism (single writer) | every cross‑partition query = `UNION ALL` over 256 cursors; 256 root pages/type even when empty; huge schema slows planning | **REJECT** — this is a multi‑node sharding pattern, not a single‑file‑embedded one |
| **By `entity_type` (dozens)** | per‑type **FTS5** index; cheap "drop a whole type"; type‑scoped scans skip other types | cross‑type queries = `UNION ALL` over types; more schema | **OPTIONAL** — only if type‑scoped queries/FTS dominate |
| **By time (hot/recent vs archive)** | keeps the hot table small; cheap retention/archival/drop | walks/searches that cross the boundary need `UNION ALL` | **OPTIONAL** — for retention policies |
| **Hot/cold by access** | — | the RAM MetaCache *is already* the hot tier | **REDUNDANT** — let RAM be the hot tier |

**Recommendation:** start with **single `entity`/`observation`/`relation` tables +
covering, partial, and expression indexes.** Partition **only** by `entity_type`
**and only if** measurements show type‑scoped queries or per‑type FTS dominate;
keep `partition_map` + `TableRouter` in the design so partitioning is an additive
change, not a rewrite. Never hash‑partition.

**Where SQLite's index richness actually wins (use these instead of partitioning):**
- **Partial index** — index only live, hot, or non‑trivial rows:
  `CREATE INDEX rel_out_live ON relation(from_id,type_id,to_id) WHERE flags=0;`
  (skip tombstones), or index observations only `WHERE length(body)>0`.
- **Expression / functional index** — `CREATE INDEX ent_name_ci ON entity(lower(name));`
  for case‑insensitive lookup without a separate column.
- **Covering index** — the `entity_by_hash` index above answers MetaCache warm‑up
  and name→id *without touching the table* (index‑only scan).
- **Generated columns** — for derived filters you query often.

These give the "make the hot path fastest" goal the README wanted from
partitioning, without the `UNION ALL` tax.

---

## 6. BM25 / search — permutations and the split

| Option | How | Pros | Cons | Verdict |
| :-- | :-- | :-- | :-- | :-- |
| **A. FTS5 + `bm25()` in SQL** | `entity_fts` virtual table; `MATCH ... ORDER BY bm25() LIMIT k` | least code; mature; disk‑scaling; tokenizers built in | FTS5's scoring/tokenizer; weights limited; FTS5 is per‑table | **DEFAULT** |
| **B. FTS5 candidate‑gen + Rust rerank** | FTS5 `MATCH` to fetch top‑N ids + raw stats, then score in Rust (field boosts, recency, type weighting) on the small N | flexible scoring; SQL does the heavy index scan; Rust does cheap rerank | a second pass over N (small) | **when custom ranking needed** |
| **C. Custom inverted index in tables + Rust scoring** | tables `postings(term_id, entity_id, weight)` + Rust BM25 | full control | reinvents FTS5; more writes; no clear win | **avoid** |
| **D. Rust scalar fn registered in SQLite** | `create_scalar_function("bm25x", …)` called inside SQL | score in SQL without materializing | per‑row VM call overhead; awkward | **niche only** |

**The split we recommend:** **FTS5 (option A/B) does term matching + candidate
generation and bounds the set with its own `bm25()` ordering (`LIMIT N`); Rust
does the final, flexible scoring/rerank on those N** — IDF/weights, an entity‑name
boost, type filter, recency. Heavy inverted‑index scan stays in optimized C; the
policy‑rich part stays in Rust on a tiny candidate set.

**Use external‑content FTS5, not contentless** (validated). External content
stores *only* the inverted index and reads source rows back by `content_rowid` —
no duplicated text (the bodies live once in `observation`, compressed by zstd),
and the `content_rowid` is the stable `INTEGER PRIMARY KEY`. Two small FTS tables:

```sql
-- bodies live in `observation`; names in `entity`. No text is duplicated.
CREATE VIRTUAL TABLE obs_fts  USING fts5(body, content='observation', content_rowid='id', ...);
CREATE VIRTUAL TABLE name_fts USING fts5(name, content='entity',      content_rowid='id', ...);
```

Write path is cheap: one FTS insert per observation / per name (no re‑indexing the
whole entity on append). The writer syncs in‑txn with the FTS `'delete'`/insert
commands (no triggers). Search:

```sql
SELECT rowid, bm25(obs_fts) AS s FROM obs_fts  WHERE obs_fts  MATCH ? ORDER BY s LIMIT :N;  -- rowid = observation.id → entity_id
SELECT rowid, bm25(name_fts) AS s FROM name_fts WHERE name_fts MATCH ? ORDER BY s LIMIT :N;  -- rowid = entity.id
```

Rust merges the two candidate streams, dedups to `entity_id`, applies the name
boost, type filter, and final ranking. (`bm25()` is negative — more negative = more
relevant — so `ORDER BY bm25() ASC` is best‑first. Do the dedup/group‑by‑entity in
Rust, not SQL, to avoid the temp B‑trees `GROUP BY`/`ORDER BY` would add over the
candidate set.) If we partition by type, there is one `obs_fts_<type>` per type and
search `UNION ALL`s the relevant ones — another reason to partition by type *only
when search is type‑scoped*.

---

## 7. Concurrency & request path

- **Read pool:** `N` read‑only WAL connections (N ≈ physical cores) on a blocking
  pool. WAL ⇒ they run truly concurrently and never block the writer. A read tool
  that misses the MetaCache grabs a pooled connection, runs an indexed query
  (rowid fetch / covering scan / FTS), returns, and populates the cache.
- **Writer:** exactly one connection on one thread, fed by an MPSC channel. It
  **drains a batch** of queued ops and commits them in **one transaction** (group
  commit) — turning bursty writes into few fsyncs and high throughput. After
  commit it updates the RAM caches (it's the only writer, so counter/degree
  updates need no atomics among writers; readers see them via atomics/seqlock).
- **Hot path bypasses SQLite entirely:** existence, type, degree, obs_count,
  rowid, and stats are answered from RAM with **no SQLite call and no blocking
  hop** → Tier A < 100 µs. The reactor only dispatches to the blocking pool when
  it must read bodies or search.
- **Cache fill is concurrent‑safe:** MetaCache is a sharded, read‑mostly
  open‑addressing map (lock‑free read, per‑shard lock on fill) keyed by the u64
  `name_hash`.

This is strictly simpler than v2's lock‑free engine and gets correctness from
SQLite + WAL rather than from epoch reclamation we'd have to verify.

---

## 8. Latency tiering (honest, matches the README targets)

| Op | Path | Target |
| :-- | :-- | :-- |
| `entity_exists`, `degree`, `graph_stats`, hot `get_entity` (no bodies) | RAM cache, no SQLite | **< 100 µs** |
| `get_entity` with observations (hot pages) | MetaCache → SQLite rowid + `obs_by_entity` | ~0.1–1 ms |
| `get_neighbors` (one page) | `rel_out`/`rel_in` index range | ~0.5–2 ms |
| `search_nodes` | FTS5 `MATCH` + Rust rerank | ~1–10 ms |
| cold `get_entity` (cache miss, cold pages) | `entity_by_hash` index + disk | ~1–5 ms |
| writes | batched into the writer's transaction | ~ms amortized, high throughput |

Sub‑100 µs is a **RAM‑cache** SLO (as it must be — a cold SQLite page read is a
disk read). The metadata tables exist precisely to make the hot tier as wide as
RAM allows.

---

## 9. Crash consistency without FKs

- The single writer enforces integrity **in the transaction**: deleting an entity
  deletes its observations, its in/out relations, its FTS row, decrements
  `type_dict.count`, and updates `graph_stat` — **atomically** (one WAL txn). No
  FK needed; no dangling rows.
- Counters/degrees are persisted **in the same transaction** as the data, so after
  a crash they match the data and the RAM caches rebuild from metadata in ms.
- We **avoid SQLite triggers** for this maintenance: triggers run per‑row in the
  VM and add overhead/debugging surface, and we already have a single controlled
  writer. (Triggers remain a fallback if a second write path ever appears.)

---

## 10. RAM budget

| Structure | Bytes/entry | 10 M entities |
| :-- | --: | --: |
| MetaCache (`name_hash`→meta, open‑addr, load 0.7) | ~40 | ~0.4 GB |
| TypeInterner + counts | ~32 / type | < 1 MB |
| graph_stat counters | — | negligible |
| hub_degree cache (hubs only) | ~24 / hub | small |
| SQLite page cache (`cache_size`) | — | configurable (e.g., 100–500 MB) |
| SQLite mmap (`mmap_size`, optional) | — | maps file, not extra RSS |
| **Total (10 M)** | | **~0.5–1 GB**, tunable down |

RAM ∝ entity *count* (MetaCache) + a configurable SQLite cache; value/observation
*volume* costs only page cache. Same scaling property as v2, far less code.

---

## C. `page_size` & zstd (sqlite‑zstd)

`page_size` is a **single, database‑wide** choice (power of two, 512–65536, set
before the first write). It trades:

- **Larger pages (16 KB+):** shallower B‑trees (fewer faults per lookup), fewer
  overflow pages for big values, better sequential/scan throughput. **Cost:** read
  amplification (a small random row still faults a whole page) and WAL write
  amplification (a 1‑byte change writes a full‑page WAL frame).
- **Smaller pages (4 KB):** less amplification for scattered small writes/reads.
  **Cost:** deeper trees, more overflow for big blobs.

For *this* design the usual "small page for point reads" instinct is weakened,
because **the hot point reads are served from the RAM MetaCache, not SQLite** — so
cold read‑amplification is a Tier‑B concern, and write amplification is absorbed by
**batched group‑commit + append‑mostly (monotonic rowid) inserts**. Meanwhile the
observation bodies are large and benefit from big pages. So **`page_size = 16384`
is the right default here** (matches your spec), for three reasons: shallower
trees for cold lookups, fewer overflow pages per (compressed) body, and amortized
WAL frames under batching.

**zstd ([sqlite‑zstd], transparent *row‑level* compression):**

- It compresses the **values of one column**, transparently, using **trained
  dictionaries** (grouped via a `dict_chooser` expression). It is *not* page‑level
  (that's the commercial ZIPVFS); it does not touch indexes or FTS shadow tables.
- **Apply it only to `observation.body`** — the large, repetitive, cold column.
  Expect ~3–5× on natural‑language observations. Group the dictionary by something
  with shared vocabulary (e.g. `entity_id % 64`, or by `type_id`) so each dict is
  well‑trained.
- **Keep it OFF the hot path.** Decompression is paid **per row read** (zstd
  decompresses at ~GB/s, so µs for a body, but non‑zero). Metadata, names,
  counters, indexes, and `name_hash` are never compressed — those are what the
  < 100 µs tier touches (and they're in RAM anyway). Compressing bodies pushes
  `get_entity`‑with‑bodies slightly deeper into Tier B; acceptable.
- **Interaction with `page_size`:** compressed bodies are smaller, so **more rows
  pack per page and fewer overflow pages are needed** — a strict win for cold scan
  density. Big pages + zstd compound here: a 16 KB page holds many compressed
  bodies, and a body that *would* have overflowed at 4 KB often fits inline once
  compressed. (Net: 16 KB + zstd on `body` is a good pairing.)
- **Interaction with `mmap`:** `mmap_size` gives zero‑copy of *raw* pages —
  uncompressed metadata/index pages benefit directly; compressed bodies must be
  decompressed in userspace regardless, so mmap gives them no zero‑copy benefit
  (only saves the read syscall). Fine — bodies are Tier B.
- **Operational cost:** sqlite‑zstd works via a virtual‑table/loadable extension
  and dictionary training; compression and dictionary (re)training are **deferred
  to the maintenance daemon (§D)**, not done inline on the writer's hot commit.
  New rows can be written uncompressed and compressed in the background, or
  compressed at write time at the cost of writer CPU — a tunable.

[sqlite‑zstd]: https://github.com/phiresky/sqlite-zstd

## D. Maintenance daemon (Rust, background)

A dedicated background task (its own thread / tokio task, *separate* from the hot
writer) performs the periodic "learning, reindexing, cleaning" the engine needs.
It either holds the writer connection during a maintenance window or submits its
ops through the same single‑writer queue (so it never races the foreground
writer). Responsibilities, each with a trigger:

| Job | What | Trigger |
| :-- | :-- | :-- |
| **WAL checkpoint** | `PRAGMA wal_checkpoint(TRUNCATE)` to bound WAL growth | WAL size threshold / interval |
| **`ANALYZE` (planner stats)** | refresh `sqlite_stat1` so the query planner keeps picking the right index as data grows — *this is the "learning"* | after N writes / row‑count drift |
| **`PRAGMA optimize`** | SQLite's built‑in "analyze what changed" | on idle / interval |
| **FTS5 `'optimize'` / `'merge'`** | compact the FTS inverted index for faster MATCH | after M FTS writes |
| **Tombstone sweep** | hard‑delete `flags`‑soft‑deleted entities: cascade observations/relations/FTS, fix neighbor degrees | backlog threshold |
| **`incremental_vacuum`** | reclaim freed pages without a full `VACUUM` stall (DB in `auto_vacuum=INCREMENTAL`) | free‑page threshold |
| **zstd compress / retrain** | compress newly‑written `observation.body` rows; periodically **retrain dictionaries** as vocabulary drifts and recompress | size / drift threshold |
| **Counter/aggregate audit** | recompute `graph_stat`/`type_dict.count`/degrees and reconcile if a crash left drift; rebuild the RAM caches | startup + periodic |
| **Cache warming** | pre‑load the MetaCache covering‑index scan after restart | startup |
| **Integrity check** | `PRAGMA integrity_check` / FTS `'integrity-check'` (validated) on a sampled cadence | low‑traffic window |

Design rules for the daemon:
- **Never block the hot path.** It runs in low‑traffic windows or in small bounded
  batches; WAL means its reads/most work don't block readers, and its writes go
  through the same single‑writer discipline so there's no write‑write conflict.
- **Idempotent + resumable.** Every job can be interrupted and resumed (it works
  off persisted state: `flags`, free‑list, `sqlite_stat1`, a small
  `maintenance_log` table if needed), so a crash mid‑maintenance is safe.
- **Avoid full `VACUUM`.** A full `VACUUM` rewrites the whole file (long stall and,
  on a non‑PK table, rowid renumbering — which we sidestep by using
  `INTEGER PRIMARY KEY`). Prefer `auto_vacuum=INCREMENTAL` + `incremental_vacuum`.
- **`ANALYZE` is the "learning."** As the graph grows and the type/degree
  distribution shifts, refreshed planner statistics are what keep
  `search_nodes`/neighbor/`read_graph` on the right indexes without us hand‑tuning.

## 11. Phasing & migration

- **P0.** Add the SQLite schema + `schema.sql`; bring up `rusqlite` with the
  pragmas (WAL, `synchronous=NORMAL`, `cache_size`, `page_size=16384`,
  `temp_store=MEMORY`, `busy_timeout`, optional `mmap_size`). Read pool + writer
  thread skeleton. *Exit:* CRUD parity with current behavior on the existing test
  suite, SQLite‑backed.
- **P1.** MetaCache + TypeInterner + counters from the metadata tables; route
  `entity_exists`/`degree`/`graph_stats`/hot `get_entity` to RAM. *Exit:* Tier A
  < 100 µs on the bench; stats/types O(1).
- **P2.** FTS5 + Rust rerank for `search_nodes`. *Exit:* search correctness parity
  + Tier‑B latency.
- **P3.** Cursor pagination for `read_graph`/`get_neighbors` via index ranges; the
  `tools.json` projection/cursor additions from the v2 doc. *Exit:* hubs paginate
  O(1); no unbounded payloads.
- **P4 — maintenance daemon (§D).** WAL checkpointing, `ANALYZE`/`optimize`, FTS
  `optimize`, tombstone sweep, `incremental_vacuum`, counter audit + cache warming.
  *Exit:* WAL bounded, planner stats fresh, soft‑deletes reclaimed, restart warms
  caches in ms.
- **P5 — zstd on `observation.body` (§C).** Background compression + dictionary
  training via the daemon; `page_size=16384`. *Exit:* measured ≥3× on bodies with
  Tier‑B latency intact.
- **P6 (optional, measured).** `entity_type` partitioning + `TableRouter` *iff*
  the bench shows type‑scoped queries dominate. Bounded MetaCache (CLOCK) for
  ≫‑RAM graphs.

The existing semantic tests (`integration.rs`, `fuzzy.rs`, `redesign_tests.rs`)
are the parity gate at every phase: the engine swap must be behavior‑preserving.

---

## 12. Risks & come‑clean

- **Single writer caps write throughput.** Mitigated by batching/group commit;
  acceptable for a read‑dominated memory server. If write throughput becomes the
  binding constraint, SQLite is the wrong engine — revisit v2.
- **Cold reads are ms, not µs.** Intrinsic: a cold page is a disk read. The < 100 µs
  promise is a RAM‑cache promise; the metadata tables maximize how much is RAM‑hot.
- **`mmap_size` reintroduces the fault question** — but on *blocking* pool threads,
  not the reactor, so a fault stalls one pooled reader, not the event loop.
  Acceptable; tune `mmap_size` vs `cache_size` by benchmark.
- **No FK means app bugs can dangle rows.** Mitigated by funneling *all* writes
  through one writer with in‑txn cascade + a periodic integrity sweep + tests.
- **`INTEGER PRIMARY KEY`** is now settled (you approved keeping the PK): it's the
  fast path, the only VACUUM‑stable integer id, and required by external‑content
  FTS `content_rowid`. UUID/hash remains the *logical* handle (`name_hash`), never
  the physical key.
- **zstd adds per‑row decompression on body reads** and an operational dictionary‑
  training/recompression job — confined to `observation.body` and the daemon, off
  the hot path. If observations are tiny, zstd may not pay; gate it by measured
  ratio.
- **The maintenance daemon shares the single‑writer discipline** — a runaway job
  (e.g. a large tombstone sweep) could starve foreground writes; bound its batch
  sizes and run in low‑traffic windows.

---

## 13. One‑screen summary

- **SQLite is the durable query engine; a Rust RAM metadata cache is the < 100 µs
  hot path.** Hot reads never call SQLite.
- **No FKs** (writer enforces integrity in‑txn). **`INTEGER PRIMARY KEY` = rowid**
  is the physical id (fast, VACUUM‑stable); **`name_hash`** is the UUID‑like
  logical handle; **no TEXT/UUID physical key**.
- **Metadata tables** (`type_dict`, `graph_stat`, covering `entity_by_hash`,
  `hub_degree`, `partition_map`) each map to a RAM cache, are maintained in‑txn by
  the single writer, and make restart O(metadata).
- **Reject hash‑prefix partitioning** (SQLite anti‑pattern); win with covering /
  partial / expression indexes; partition by `entity_type` only if measured.
- **Search = external‑content FTS5 (`obs_fts`+`name_fts`) candidate‑gen + Rust
  rerank.** No duplicated text; write‑cheap per‑row indexing.
- **One writer thread (batched group commit) + a WAL read‑connection pool;** the
  RAM cache absorbs the hot reads so the reactor never blocks.
- **Storage model:** don't pick int widths — SQLite's serial types size every
  value automatically (0/1 cost 0 bytes; `INTEGER PRIMARY KEY` costs 0 body bytes).
  Spend the budget on **table shape + indexes**, validated pathway by pathway
  (§4B) with `EXPLAIN QUERY PLAN` — all hot reads are index‑only or O(1)‑cursor.
- **`page_size=16384` + zstd on `observation.body` only** (3–5×); decompression is
  per‑row so it stays off the RAM hot path; big pages + compression pack cold rows
  densely.
- **Rust maintenance daemon** (§D) does the "learning": `ANALYZE`/`optimize`, WAL
  checkpoint, FTS optimize, tombstone sweep, `incremental_vacuum`, zstd
  (re)training, counter audit + cache warming — bounded, idempotent, off the hot
  path; **never full `VACUUM`** (incremental only).
