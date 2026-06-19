# Scaling mcp-memory: 10 MB → 1 TB, 10 k clients, < 100 µs

Status: **design / RFC**. Nothing here is implemented yet. This document exists
to be argued with before a line of engine code is written. It questions the
current stack (TidesDB + bincode + serde_json + global write gate), states the
*physics* the target implies, and proposes a concrete, staged architecture with
byte layouts, RAM math, a concurrency model, a catalog of custom data
structures, and the `tools.json` changes that actually move throughput.

The two success metrics, per the brief, are: **(1) the `tools.json` surface**
and **(2) throughput per unit of hardware**. Everything below is justified
against those two, not against internal elegance.

---

## 0. The target, stated honestly

> Scale from 10 MB to 1000 GB, with as little RAM as possible, as much
> concurrency as possible, ~10,000 clients, < 100 µs latency.

Before designing anything, pin down what 100 µs *buys*. These are order‑of‑
magnitude costs on a modern server (Zen4/SPR class, NVMe):

| Operation | Cost | # that fit in 100 µs |
| :-- | --: | --: |
| L1/L2 hit | 1–4 ns | ~30,000 |
| RAM random access (cache miss) | 60–100 ns | ~1,000 |
| Hash probe (2–3 misses + compare) | 150–300 ns | ~400 |
| Uncontended mutex lock/unlock | 15–25 ns | ~4,000 |
| **Contended** mutex (10 k writers) | 1–100 µs | 1–100 |
| Syscall (gettimeofday-class) | 0.3–3 µs | ~30–300 |
| Tokio `spawn_blocking` hop (thread handoff) | 1–10 µs | ~10–100 |
| mmap **minor** fault (page resident) | 0.5–2 µs | ~50–200 |
| serde_json parse of a small envelope | 0.5–3 µs | ~30–200 |
| **NVMe random read (cold page)** | **10–100 µs** | **1–10** |
| SATA SSD random read | 100–500 µs | 0 |
| HDD seek | 5–10 ms | 0 (disqualifying) |

**Three conclusions fall straight out of the table and constrain the whole
design:**

1. **A single cold disk read can consume the entire budget.** Therefore data
   served in < 100 µs *must already be in RAM* — either in a compact in‑process
   index or in the OS page cache. Disk is for cold/overflow data and cannot be on
   the sub‑100‑µs path. This is not a tuning detail; it's the load‑bearing wall.

2. **Contended locks and thread handoffs are budget‑sized.** A global write
   mutex, an LRU that takes a write lock on every *read* (to bump recency), or a
   `spawn_blocking` hop on every request each cost 1–100 µs under load. At 10 k
   clients these dominate. The hot path must be **lock‑free for reads** and must
   **not hop threads for RAM‑resident work**.

3. **RAM must scale with entity *count*, not data *volume*.** 1 TB of
   observations cannot live in RAM "as little as possible." The only way to hold
   1 TB and stay small is: keep **compact indexes** in RAM (bytes per *entity*,
   independent of how fat each entity is) and let the **OS page cache** hold the
   hot slice of the 1 TB of values. RAM ≈ f(N_entities), not f(bytes).

### 0.1 Honest SLO tiering (don't promise the impossible)

< 100 µs is achievable *for the working set*, not for arbitrary cold queries over
1 TB, and not for unbounded graph walks. We define tiers and hold each to a
realistic SLO. The `tools.json` design (Section 7) exists largely to keep callers
on the fast tiers and to make the slow tier explicitly paginated.

| Tier | Operations | SLO (working‑set resident) |
| :-- | :-- | :-- |
| **A — point** | `get_entity` (projected), `entity_exists`, `degree`, `add_observations` (1 entity) | **p99 < 100 µs** |
| **B — bounded fan‑out** | `search_nodes` (top‑k), `get_neighbors` depth 1 (one page), `open_nodes` (small batch) | p99 < 1 ms |
| **C — unbounded / cold** | multi‑hop `find_*`, `read_graph`, `export_graph`, `extract_subgraph`, cold‑page reads | best‑effort, **cursor‑paginated**, never blocks A/B |

If a reviewer disagrees with anything in this doc, this is the table to argue
about first: it decides which operations get the engineering and which get a
cursor and a shrug.

---

## 1. Workload model (what the graph is actually asked to do)

A KG memory for agents is **read‑dominated, point‑and‑small‑fan‑out**, write
in bursts. Empirically the call mix is roughly:

- **Point reads / existence checks** (`get_entity`, dedup before insert): the
  bulk of traffic. Latency‑critical (Tier A).
- **Search** (`search_nodes`): frequent, top‑k, bounded (Tier B).
- **1‑hop neighborhood** (`get_neighbors` depth 1): frequent (Tier B).
- **Writes** (`create_*`, `add_observations`, `upsert`): bursty, a minority of
  ops, tolerant of slightly higher latency, but must be durable and must not
  stall readers.
- **Analytics / walks** (`find_path`, `read_graph`, `extract_subgraph`): rare,
  expensive, Tier C.

Design consequence: **optimize reads to the floor; make writes not interfere
with reads; make walks paginated and off the hot path.** A symmetric "everything
is a transaction" engine (today's TidesDB usage) spends the same heavy machinery
on a hot point read as on a write — that symmetry is the core abstraction cost.

Key structural facts we will exploit:
- Entity **names are the identity** and are reused constantly (as relation
  endpoints, observation targets). They should be **interned to dense integer
  IDs** once and referenced by ID everywhere internal; strings reappear only at
  the JSON boundary.
- Entities are **immutable‑ish**: observations append far more than they mutate.
  An append/log‑structured value store fits perfectly and gives zero‑copy reads.
- The graph is **sparse and power‑law**: most nodes have low degree, a few hubs
  have enormous degree. Adjacency must be O(degree) to read *and* paginate hubs.

---

## 2. Why the current stack cannot hit the target — abstraction‑cost audit

The brief says: *"recursively look for each abstraction cost."* Here is the audit
of the current hot read path, HTTP → disk, with the cost each layer imposes per
operation and whether it survives the redesign.

| Layer (today) | Per‑op cost | Why it fails the target | Verdict |
| :-- | :-- | :-- | :-- |
| `spawn_blocking` on **every** request | 1–10 µs thread handoff; blocking pool (512) caps concurrency | RAM‑resident reads don't need a thread hop; pool becomes the bottleneck at 10 k clients | **Remove for Tier A/B (RAM hits); keep only for cold disk** |
| `serde_json::Value` parse of the whole request | allocates a DOM (multiple allocs) | We only need 2–3 fields; DOM build is pure waste | **Replace with borrow‑only field extraction** |
| TidesDB `begin_transaction()` per read | FFI crossing + C‑side txn object alloc + isolation bookkeeping | A point read needs none of this; it's transaction machinery on a read‑only lookup | **Eliminate (lock‑free mmap read path)** |
| TidesDB `get` returns `Vec<u8>` | heap alloc + memcpy of the value out of the C side, per read | At 10 k reads/100 µs that's allocator + bandwidth pressure | **Eliminate (return `&[u8]` into mmap)** |
| LSM read amplification | a point read may probe memtable + N SSTable levels + blooms | multiple lookups/faults per read → tail latency; fights p99 < 100 µs | **Replace hot path with direct index → offset → mmap** |
| `bincode::deserialize::<Entity>` | allocates `String`×(2+obs), `Vec` | full owned struct built even to read one field or just the name | **Replace with zero‑copy `RecordRef` (lazy field slices)** |
| `String` keys everywhere (`HashMap<String,_>`, cache keys, BFS `visited`) | hashing variable‑length bytes, allocs, clones | identity should be a 4/8‑byte ID; string ops are 10–50× the cost | **Intern to `u32`/`u64` IDs; strings only at the edge** |
| `entity_cache: RwLock<LruCache>` — `get` takes a **write** lock | serializes all reads through one lock to bump LRU recency | the single worst contention point under 10 k readers | **Delete app cache; use page cache + lock‑free index** |
| `KnowledgeGraphOut { Vec<Entity> }` owned outputs | clones every returned entity (name+type+all obs) before serializing | doubles allocation and copy for every read result | **Stream JSON directly from mmap; never materialize owned structs** |
| `write_gate: Mutex<()>` — one global write lock | every write serializes globally | writes can't scale past 1 core; long write stalls a reader behind the same lock if reused | **Replace with per‑shard write locks** |
| Stats: full‑CF deserializing scan (mitigated to lazy snapshot) | O(N) rebuild after any write | still O(N) when reads and writes interleave | **Striped counters updated O(1) per write** |

**The pattern:** today the system pays *write‑grade, transactional, owned,
string‑keyed, FFI* costs on *read‑grade* operations. The redesign's thesis is to
make the read path **lock‑free, zero‑copy, integer‑keyed, allocation‑free, and
syscall‑free on a cache hit** — and to confine the heavy machinery to writes and
to cold disk.

### 2.1 Should TidesDB stay?

TidesDB is a competent general LSM, but three of its properties are intrinsic
mismatches for this target, not tuning knobs: **FFI per op**, **transaction +
value‑copy per read**, and **LSM read amplification**. For the *cold tier* (Tier
C scans, the 1 TB that doesn't fit in the page‑cache working set) an LSM is fine.
For the *hot tier* it is the wrong tool. The plan therefore **does not rip out
TidesDB on day one**; it introduces a native zero‑copy read path in front of /
instead of it, staged and measured (Section 8), and retires TidesDB from the hot
path only once the native engine beats it on the bench.

---

## 3. Target architecture

```
                       ┌──────────────────────────────────────────────┐
   10k clients         │              mcp-memory process               │
  (HTTP/2, TCP) ─────► │                                               │
                       │  Transport: tokio, per‑conn buffers, no       │
                       │  spawn_blocking for RAM hits                   │
                       │      │                                         │
                       │      ▼                                         │
                       │  Dispatch: borrow‑only JSON field extraction, │
                       │  classify fast(A/B) vs slow(C)                 │
                       │      │                                         │
                       │      ▼                                         │
                       │  Graph API (integer‑ID domain)                │
                       │   ├─ Interner        name ⇆ u64 id  (sharded)  │
                       │   ├─ EntityIndex     id → value offset (RAM)   │
                       │   ├─ Adjacency       CSR (cold) + delta (hot)  │
                       │   ├─ SearchIndex     term dict (RAM)+postings  │
                       │   └─ Striped stats / seqlock snapshots         │
                       │      │                 │                       │
                       │  lock‑free reads        │ per‑shard write locks│
                       │      ▼                 ▼                        │
                       │  ValueLog: mmap immutable segments (zero‑copy) │
                       │  WAL: append + group commit (durability)       │
                       └───────────────┬───────────────┬───────────────┘
                                       │ page cache     │ fsync
                                       ▼                ▼
                                   OS page cache  ──►  NVMe (segments, WAL)
```

The RAM/disk split is the whole game: **indexes in RAM (small, ∝ entity count),
values on disk via mmap (large, page‑cached working set).**

### 3.1 Identity: the Interner (name ⇆ id)

All internal code speaks `EntityId(u64)` and `TypeId(u32)` / `RelTypeId(u32)`,
never `String`. The interner is the only place strings are hashed.

**Logical IDs are decoupled from physical location** (see Revision R2 — this is
the fix for the compaction paradox). An `EntityId` is a **dense, monotonic
logical integer that never changes for the life of the entity**. Physical
location is reached through one extra indirection:

- **name → logical id**: a sharded, lock‑free open‑addressing table. Each slot
  stores an 8‑byte name **fingerprint** (xxh3) + the 5‑byte logical id; the full
  name is *not* in RAM — it lives in the value record. Lookup: hash name → shard →
  probe by fingerprint → on fingerprint match, verify against the on‑disk name
  (one page‑cache/io_uring read, only on a probe hit). 64‑bit fp + verify ⇒ exact.
- **logical id → physical location**: a single dense, mlock‑resident array
  `loc[]` of 8‑byte `PhysicalRef { segment: u24, offset: u40 }`. `loc[id]` is a
  direct index — one L1/L2 access. **This is the only structure compaction
  mutates when it moves a record: a single atomic 64‑bit store to `loc[id]`.**
  Every other structure (CSR edges, postings, in/out adjacency) references the
  *logical id* and is therefore untouched by record movement — no cascading
  rewrites, no write amplification, compaction stays background. The read path
  pays exactly one extra resident‑array dereference (`id → loc[id] → bytes`).
- **types** (`entityType`, `relationType`): a tiny separate interner; usually
  < 10⁴ distinct types, a tiny RAM map; types compare as `u32`.

Deletion retires a logical id (tombstone `loc[id] = NULL`); ids are monotonic so
the array only grows, and a rare *id‑space* compaction (distinct from value
compaction) reclaims a sparse `loc[]` by renumbering — the one operation that does
rewrite references, scheduled rarely and offline‑style under the per‑shard locks.

RAM cost: see Section 4. This removes string hashing/allocation/cloning *and*
makes physical compaction an O(1)‑per‑moved‑record pointer flip.

### 3.2 Value storage: log‑structured ValueLog + explicit async I/O

> **Residency discipline (Revision R1).** `mmap` is a trap on an async reactor:
> a **major** page fault (cold page → NVMe) parks the *OS thread*, stalling every
> task multiplexed onto that tokio worker for 10–100 µs. You cannot know a page is
> cold without touching it, and touching it on the reactor *is* the fault. So the
> rule is: **only slice memory we can prove is resident on the async thread;
> everything else goes through explicit async I/O.**

Two classes of bytes, two policies:

- **Indexes are `mlock`‑resident and sliceable inline.** The interner shards, the
  `loc[]` translation array, the term dictionary, and CSR `row_ptr` are small
  (Section 4), pinned with `mlock`, and therefore *guaranteed* not to fault. The
  reactor may touch them directly — that's the lock‑free, syscall‑free hot path.
- **Values, postings, and CSR neighbor runs use explicit `io_uring` + `O_DIRECT`,
  not transparent `mmap` faults.** Reads are submitted to the ring and complete
  asynchronously; the reactor **never blocks** on disk. We manage a **userspace
  buffer/slab cache** (our own page cache) for the hot value set, so a hot read is
  a cache hit (pure memory, no syscall) and a cold read is an async ring
  completion (10–100 µs, but *non‑blocking* — the worker keeps serving other
  tasks). This also gives us **precise control of the cache size** ("as little RAM
  as possible" becomes a number we set, not a kernel heuristic) at the cost of
  losing the free, shared OS page cache.

Why not the simpler `mmap` + `mincore()` guard (check residency, punt cold reads
to `spawn_blocking`)? `mincore` is a per‑read syscall (budget) **and** racy: a
page can be evicted between the check and the access (TOCTOU), reintroducing the
exact stall it was meant to prevent. We therefore treat `mmap`+`mincore`+offload
as a *second‑class fallback mode* for the cold tier only, and make `io_uring`/
`O_DIRECT` with a userspace cache the primary value path. (On platforms without
io_uring, the fallback is a bounded `spawn_blocking` pool sized to the device's
queue depth — slower tail, same correctness.)

- A read of a *resident* value returns a **`RecordRef<'a>`** — a borrowed view over
  cache‑pinned bytes — and decodes *lazily*: reading just the name or type touches
  only those bytes; observations are sliced, not allocated, and streamed straight
  into the JSON output buffer. **Zero deserialization, zero owned `Entity`, zero
  copy.** A non‑resident value yields an async fetch that pins it into the cache,
  then the same zero‑copy decode.
- Record layout (little‑endian, varint where marked):

  ```
  RecordHeader (fixed 16 B): u32 total_len | u32 type_id | u32 n_obs | u32 flags
  name:     varint len, then bytes
  obs[i]:   varint len, then bytes        (× n_obs)
  ```

  Fixed header first so `total_len`, `type_id`, `n_obs` are readable without
  scanning. Field offsets within a record are computed by walking varints — cheap
  and branch‑predictable, and only walked for the fields actually requested.

- **Updates** (add/delete observation, upsert) append a *new* record and flip the
  index offset (single 8‑byte atomic store, Section 5). The old record becomes
  garbage reclaimed by compaction. Readers mid‑flight keep using the old mapping
  safely (epoch reclamation). This is copy‑on‑write at record granularity and is
  why reads need no lock against writes.
- **Compaction** rewrites live records into fresh segments, drops tombstones,
  rebuilds CSR adjacency and recomputes search `avgdl`; runs in the background,
  bounded, and never blocks Tier A/B.

### 3.3 Adjacency: CSR (cold) + delta overlay (hot), paginated

Graph reads need O(degree) neighbor enumeration with one sequential access and
the ability to **paginate hubs** (a node with 10⁶ edges must not return 10⁶ rows).

- **CSR (Compressed Sparse Row)** for the compacted majority: a per‑node offset
  array `row_ptr[id] → start` into a flat `neighbors[]` array of
  `(neighbor_id, rel_type_id)` sorted by neighbor. Enumerate = one contiguous,
  cache‑friendly scan; **a cursor is just an index** into the run → O(1) resume,
  no O(offset) skipping.
- **Delta overlay** for edges added since last compaction: a small per‑shard
  append structure (`id → Vec<(neighbor_id, rel_type_id)>` in an arena), merged
  with CSR at read time. Compaction folds the delta into a new CSR.
- Both directions: maintain CSR for out‑edges and in‑edges (mirror), as today's
  `rel_out`/`rel_in` split, but as integer CSR not string keys.
- Edge *existence* (dedup on `create_relations`) = binary search in the CSR run +
  delta check, integer comparisons only.

### 3.4 Search: RAM term dictionary + mmap postings, true BM25

Keep the BM25 model already implemented, but split RAM vs disk on the scaling
boundary:

- **Term dictionary** in RAM: `term → (df, postings_offset)`. Vocabulary ≪ N, so
  this is small. `df` and corpus `N`, `avgdl` live here for IDF.
- **Postings** per term stored as **Roaring bitmaps** (for the id set) plus a
  parallel weight run, fetched via the async value path (§3.2), not transparent
  mmap. Roaring is compressed (less RAM/disk and fewer bytes moved) and has
  SIMD‑accelerated AND/OR — ideal for multi‑term queries and for `entity_exists`‑
  style membership. Weight runs are delta + **Stream‑VByte** packed so decode is
  vectorizable.
- **Vectorized scoring (Revision R3).** Scoring iterates query terms and
  accumulates `idf · weight` into a **dense score array indexed by a compact local
  id** (the candidate set is dense‑ranked first), processed with **AVX2/AVX‑512**:
  8–16 accumulations per cycle instead of a scalar loop, with a portable scalar
  fallback selected by runtime CPU feature detection. Set‑intersection style work
  (multi‑term postings, `extract_subgraph` edge filtering, CSR neighbor
  intersection) uses SIMD merge/galloping over the *sorted* id runs. This is the
  lever that moves the Tier‑B `search_nodes` p99, and it is gated behind a
  microbench proving the SIMD path beats scalar for representative k.
- Top‑k via a bounded **min‑heap of size k**, never sort‑the‑world.
- `entity_id`‑keyed throughout; the name is resolved (interner → value path) only
  for the k results actually returned.

> CSR neighbor runs (§3.3) are likewise kept **sorted by neighbor id** precisely
> so that intersection/merge can be vectorized and cursors are O(1) — the layout
> choice and the SIMD choice reinforce each other.

### 3.5 Durability: WAL + group commit

- Mutations append to a **WAL** (logical: op + ids + bytes) before the index
  flip; `Durability::Sync` ⇒ fsync per group‑commit batch, `Async` ⇒ fsync on a
  ~ms timer. **Group commit** batches concurrent writers' fsyncs into one — the
  standard way writes stay cheap under concurrency.
- Recovery replays the WAL tail into the index on open. Segments are immutable so
  they need no per‑record recovery; only the index + delta are rebuilt from WAL.

---

## 4. RAM budget — the "minimal RAM" contract, with numbers

RAM is dominated by the in‑process indexes, and each is **bytes per entity/edge,
independent of value size**. Let `N` = entity count, `E` = edge count, `V` =
distinct terms.

| Structure | Bytes/unit | Note |
| :-- | --: | :-- |
| Interner slot (fp 8 B + logical id 5 B) ÷ load 0.7 | ~19 B / entity | could drop to ~6 B/entity with a minimal‑perfect‑hash + dynamic overlay (Section 6.8) |
| `loc[]` logical→physical (mlock‑resident) | 8 B / entity | the R2 indirection; the only thing compaction mutates |
| Adjacency `row_ptr` (out + in) | ~12 B / entity | offsets only; neighbor runs are on disk |
| CSR neighbor entry on disk | (8 B / edge, **not RAM**) | in the userspace value cache when hot |
| Search term dict | ~24 B / term | V ≪ N |
| Striped counters / stats | O(shards) | negligible |

So **resident index ≈ ~40 B / entity** (open‑addressing) and the value bytes cost
**only the userspace cache size you configure** (§3.2) — not the data volume.

**Worked examples:**

- **10 MB dataset** (say 10⁴ entities): indexes ≈ 10⁴ × ~40 B ≈ **~400 KB RAM**.
  Everything trivially resident; sub‑µs.
- **1 TB dataset, fat entities** (10⁸ entities × ~10 KB each): indexes ≈ 10⁸ ×
  ~40 B ≈ **~4 GB RAM** (mlock‑resident) + whatever value cache you grant. The
  headline: **1 TB served from ~4 GB of pinned index + a value cache you size.**
- **1 TB dataset, many small entities** (10⁹ entities × ~1 KB): indexes ≈ **~40
  GB RAM** open‑addressing, or **~14 GB** with the MPHF interner variant. This is
  the genuine floor for *random* sub‑100‑µs access to 10⁹ keys — a property of key
  count, not design slack. The cold tail can spill the interner to disk at a
  latency cost (Tier C), trading RAM for latency explicitly.

The "minimal RAM" guarantee is therefore precise: **resident RAM ≈ N × ~14–40 B
(indexes) + V × ~24 B (term dict) + a value cache you choose; value volume itself
costs no RAM beyond that cache.** RAM scales with *entity count*, not bytes.

---

## 5. Concurrency model — lock‑free reads, sharded writes

Goal: read throughput scales with cores; writes scale with shard count; no single
contended cache line; no global lock; no thread hop on a cache hit.

### 5.1 Reads are lock‑free

- Value segments are **immutable** → reading bytes needs no lock, ever.
- The only mutable read‑path state is the **index slot** (id → current offset). A
  read does a single **atomic load** of the offset (`Acquire`), then reads
  immutable bytes. A concurrent writer publishes a new offset with a single
  atomic store (`Release`). Reclamation of the superseded record/segment uses
  **epoch‑based reclamation** (crossbeam‑epoch): a reader pins an epoch (a thread‑
  local bump, no contention), so memory it's reading is never freed underneath it.
- Net: a hot point read = `hash → atomic load → mmap slice → stream to buffer`.
  **No mutex, no allocation, no syscall** (page resident). That is the < 100 µs
  path.

### 5.2 Writes are per‑shard

- Keyspace is split into `S` shards by `id` (S = next_pow2(≈4×cores), e.g. 256).
  Each shard owns its slice of the interner table, its delta adjacency, and a
  **`parking_lot::Mutex` (or futex) write lock**. A write to entity X locks only
  `shard(X)`. Throughput scales to ~S concurrent writers.
- **Multi‑entity atomic ops** (`create_relations`, `merge_entities`) lock the
  *two* shards involved **in ascending shard‑index order** → deadlock‑free. Same‑
  shard endpoints lock once.
- This replaces the global `write_gate`. Per‑entity read‑modify‑write atomicity
  (the correctness property we fixed earlier) is preserved at shard granularity.

### 5.3 No contended counters

- Stats (`entities`, `relations`, `observations`, per‑type counts) use **striped
  counters**: one cache‑line‑padded counter per shard; a write bumps its own
  shard's counter (no cross‑core traffic); a read sums S counters. `graph_stats`
  becomes O(S), allocation‑free, and never scans. Per‑type counts use a small
  per‑shard map merged on read.
- Read‑mostly scalars (`avgdl`, `N` for IDF) use a **seqlock**: writers bump an
  even→odd→even version around the update; readers read‑version, read‑value, re‑
  read‑version and retry on mismatch — **zero reader locks**.

### 5.4 Transport / runtime

- **Do not `spawn_blocking` RAM‑resident ops.** Classify at dispatch: Tier A/B
  served inline on the async worker (lock‑free, no fault expected) → no thread
  hop. Only Tier C and confirmed cold‑page reads go to a blocking pool (or
  **io_uring** via `tokio-uring`/`glommio` for async disk without a thread hop).
- Per‑connection **reusable read/write buffers** (already partly done) → no per‑
  request envelope allocation.
- **Optional, advanced:** a **thread‑per‑core, share‑nothing** runtime
  (`glommio`/`monoio` + io_uring) pins shards to cores and removes cross‑core
  contention entirely for partitioned ops. The cost is cross‑partition graph
  walks become message‑passing. Recommendation: **stay on tokio + sharded‑lock +
  io_uring for disk** first (keeps traversal simple), keep thread‑per‑core as a
  measured future option, not a day‑one bet.

### 5.5 Folding the CSR delta into the immutable base without reader spikes

This is the load‑bearing concurrency question: writes are bursty and sharded, the
adjacency is `CSR_base` (immutable, sorted) + `delta` (recent edges); periodically
the delta must fold into a new base. A naïve "lock, rebuild O(E), swap" stalls
readers for the rebuild — a latency cliff. The protocol below makes a reader's
worst case during a fold *"load one atomic pointer and merge a few small sorted
runs,"* and a writer's worst case *"a sub‑µs pointer rotation on its own shard."*

Five mechanisms compose:

**(1) RCU snapshot via `ArcSwap` — readers never lock, never wait.**
Per shard, the adjacency is one immutable snapshot behind an `ArcSwap`:

```
AdjSnapshot { base: Arc<Csr>, frozen: Option<Arc<Delta>>, active: Arc<DeltaLog> }
```

A reader does **one `arc_swap.load()`** (an atomic load + epoch‑cheap refcount) to
get a *coherent* tuple, then merges. Publishing a new snapshot is **one atomic
store**. No reader ever blocks on a fold; the cost a reader pays is the `load()`
(tens of ns) plus merging small runs.

**(2) Generational double‑buffered delta — no lost writes, no big lock.**
The fold cannot "snapshot then clear" the delta, because writes land during the
rebuild. Instead we **rotate generations** under the per‑shard write lock — a
*pointer swap only*, microseconds, blocking *writers to that one shard* but **not
readers**:

1. `frozen ← active`; install a fresh empty `active`; publish the new snapshot
   `{base, Some(frozen), active'}`. (One lock, one `ArcSwap::store`.)
2. Background: build `base' = merge(base, frozen)` off to the side — **no lock
   held against readers or writers**. Writers keep appending to `active'`; readers
   merge `base + frozen + active'` (a three‑way merge of sorted runs, frozen and
   active are small).
3. Publish `{base', None, active'}` with one `ArcSwap::store`; hand `base` and
   `frozen` to epoch reclamation. New readers merge `base' + active'` only.

At every instant the reader's loaded snapshot is internally consistent, and no
edge written during the rebuild is lost (it's in `active'`, carried forward).

**(3) The `active` delta is a lock‑free append‑only log — wait‑free reads.**
`active` is shared *mutably with writers* while readers iterate it, so it can't be
a plain `Vec`. It is a **segmented append‑only log**: fixed‑size chunks that are
never reallocated (so a reader's pointer into a chunk never dangles), with an
**atomic length**. A writer appends an (immutable) `(neighbor_id, rel_type)` entry
then `Release`‑stores `len+1`; a reader `Acquire`‑loads `len` and scans `[0,len)`.
Entries never mutate or move ⇒ **no torn reads, no reader lock**. (Append ordering
within a shard is serialized by that shard's write lock, which writers already
hold; the read side is wait‑free regardless.)

**(4) Per‑shard, size‑tiered bases — folds are small, local, and amortized.**
A single monolithic CSR would make every fold O(E_shard). Instead each shard keeps
**size‑tiered base segments** (LSM‑style levels for adjacency): a frozen delta
compacts into a small L0 base; like‑sized bases later merge upward. A fold merges
only similar‑sized runs ⇒ **bounded per‑fold work**, amortized O(log) merges per
edge, never a stop‑the‑world rebuild. Readers k‑way‑merge the few sorted runs
(SIMD‑friendly, §3.4). Bursts are absorbed by `active` and folded incrementally;
a *hub* node whose `active` run grows hot triggers a *targeted* fold of just its
rows.

**(5) Epoch‑deferred, batched reclamation — the swap itself can't spike.**
Freeing a just‑retired large `base`/`frozen` inline (a big `drop`/`munmap`) inside
a reader's critical section would itself be a latency spike. So reclamation is
**deferred via crossbeam‑epoch and batched on a background thread**: the `ArcSwap`
store just drops a reference; actual free happens later, off the hot path, once no
pinned reader can observe it. No inline large free, ever.

**Backpressure (bursty writes outrunning folds).** If ingest outpaces folding,
`active` grows and read‑merge cost creeps up. Bounded by: (a) a delta high‑water
mark per shard that *raises fold priority*; (b) above a hard ceiling, **per‑shard
write admission control** — new writes to that shard briefly queue/throttle so
read latency stays inside SLO. We shed *write* latency, never *read* latency —
consistent with the read‑dominated workload and the Tier‑A contract.

**Net guarantee.** Reader cost during a fold = `arc_swap.load()` + k‑way merge of
a handful of small sorted runs. No reader lock, no reader wait, no inline large
free. Writer cost during the swap = a sub‑µs pointer rotation on its own shard.
That is how the swap avoids latency spikes.

**Cross‑shard traversal note.** A multi‑hop walk loads each shard's snapshot
independently, so it may observe shard A at generation N and shard B at N+1. For
agent memory this per‑shard snapshot consistency is acceptable; a globally
consistent walk would pin epochs across all touched shards for its duration
(holding memory longer) and is offered only if a tool explicitly needs it.

---

## 6. Custom data structures (the "small functions" catalog)

The brief asks for purpose‑built structures even where a stdlib type would "do,"
because each generic type carries an abstraction cost on the hot path. Each entry
is a small, testable module with a tight surface. None should be added without a
microbenchmark proving it beats the stdlib alternative *for this workload*.

1. **`Interner`** — sharded lock‑free name⇆id (Section 3.1). Replaces
   `HashMap<String, _>` identity maps and all string cloning of keys.
2. **`RecordRef<'a>` + `RecordWriter`** — zero‑copy entity record over mmap bytes
   (Section 3.2). Replaces `bincode` + owned `Entity`.
3. **`Csr` + `DeltaAdj`** — integer adjacency with O(1) cursor resume
   (Section 3.3). Replaces string‑keyed relation scans and `HashSet<(String,…)>`
   dedup.
4. **`Varint`** — LEB128 encode/decode for lengths/offsets/ids; shrinks records
   and postings, cutting bytes moved (bytes moved ∝ latency at these sizes).
5. **`IdBitset`** — dense bitset for BFS/DFS `visited` keyed by dense id. Replaces
   `HashSet<String>` (alloc + string hash per node) with a bit test.
6. **`ScoreAccumulator`** — search scoring into a `Vec<f32>` indexed by a compact
   *local* id (dense‑rank the candidate set), plus a size‑k min‑heap for top‑k.
   Replaces `HashMap<String,f32>` + full sort.
7. **`SeqLock<T>`** — read‑mostly snapshot for `avgdl`/`N`/config (Section 5.3).
8. **`StripedCounter`** — padded per‑shard counters for stats (Section 5.3).
9. **`Arena` / bump allocator** — per‑request scratch (visited buffers, candidate
   lists), reset at end of request → **zero per‑allocation free cost**, no
   allocator contention across 10 k requests.
10. **`MmapSegment`** — owns an mmap, hands out `&[u8]`; epoch‑guarded lifetime.
11. **`EpochGuard`** — thin wrapper over crossbeam‑epoch for read‑side pinning.
12. **`JsonScan`** — borrow‑only extraction of the 2–3 fields we need from a
    JSON‑RPC envelope without building a `Value` DOM (extends the existing
    `push_json_str` writer to the *input* side). Replaces `serde_json::from_value`.
13. **`group_commit`** — batches concurrent fsyncs (Section 3.5).
14. **`LocTable`** — dense, `mlock`‑resident `logical id → PhysicalRef` array;
    compaction updates one atomic slot per moved record (R2). The indirection that
    decouples stable ids from moving bytes.
15. **`AdjSnapshot` + `ArcSwap`** — RCU‑published immutable adjacency snapshot;
    lock‑free reader `load()`, single‑store publish (§5.5).
16. **`DeltaLog`** — segmented, append‑only, atomic‑length adjacency overlay;
    wait‑free reads, never reallocates (§5.5 mechanism 3).
17. **`RoaringPostings` + `StreamVByte`** — compressed, SIMD‑friendly postings and
    weight runs (R3); fewer bytes moved, vectorized AND/OR and decode.
18. **`simd` kernels** — AVX2/AVX‑512 score accumulation and sorted‑run
    intersection with runtime feature detection + scalar fallback (R3).
19. **`ValueCache` + `io_uring`** — userspace slab cache and async, non‑blocking
    value/postings I/O over `O_DIRECT` (R1); the reactor never faults on disk.
20. **`mlock` resident region** — pins the small indexes so the reactor may slice
    them inline without fault risk (R1).

> Note on discipline: items 4–8 are tiny but each kills a named cost in Section 2.
> Items 1–3 and 12 are the structural wins. The plan ships them behind the *same*
> `GraphHandle` API so tools and tests don't churn while the engine is swapped.

---

## 7. `tools.json` changes — the surface that drives throughput

`tools.json` is a stated success metric. The biggest *protocol‑level* throughput
lever is **not shipping bytes the caller didn't ask for** and **never forcing an
O(offset) or unbounded scan**. Changes below are grouped by impact.

### 7.1 Field projection (largest single throughput win)

Most read tools currently return full entities — name + type + **all
observations**. An agent checking existence or listing names pays to serialize and
transfer kilobytes it discards. Add an optional projection to every entity‑
returning tool:

- `get_entity`, `open_nodes`, `batch_get_entities`, `search_nodes`,
  `read_graph`, `get_neighbors`, `describe_entity`, `extract_subgraph`:
  add `view: "id" | "summary" | "full"` (default `summary`) and/or
  `includeObservations: bool` + `maxObservations: int`.
  - `summary` = name + type + observation **count** (no observation bodies).
  - `full` = today's behavior.
  This maps directly onto the zero‑copy `RecordRef`: `summary` reads only the
  header + name and never touches observation bytes → less disk, less CPU, less
  wire. For the hot dedup/existence pattern it can cut payload by 10–100×.

### 7.2 Cursor pagination (kills O(offset) at scale)

`offset`/`limit` is O(offset): page 100,000 of `read_graph` re‑scans 100,000
rows. Replace with **opaque cursors** that resume in O(1):

- `read_graph`, `search_nodes`: accept `cursor` + `limit`, return `nextCursor`.
  (Keep `offset` accepted for small pages / back‑compat, but document it Tier C.)
- The cursor encodes a segment+offset (entities) or a heap position (search); CSR
  and segment scans resume from it directly (Section 3.3).

### 7.3 New point/cheap tools (keep callers on Tier A)

- **`entity_exists(names: [string]) → [bool]`** — existence without fetching
  bodies. Today agents call `get_entity` (or `open_nodes`) just to check, paying
  full record cost. This is a pure index probe (no value read) → the cheapest Tier
  A op. High practical value (dedup before insert).
- **`degree(name, direction?) → int`** — node degree without enumerating
  neighbors. O(1) from CSR `row_ptr` diff + delta count. Lets agents decide
  whether to paginate a hub before pulling it.
- **`get_neighbors` → add `cursor` + `limit`** — a hub with 10⁶ edges must
  paginate; today `depth`/`relationType` can return unbounded rows. This is the
  single biggest tail‑latency risk in the current surface.
- **`get_observations(name, cursor?, limit?)`** — page a single entity's
  observations when it has thousands; pairs with `summary` projection.

### 7.4 Bulk ingest (throughput for loads)

- **`bulk_upsert(entities, relations)`** — one call that batches into one WAL
  group‑commit and one index pass, instead of N round‑trips. Annotated non‑
  destructive/idempotent. This is the high‑throughput path for migrations and
  agent memory dumps; it amortizes per‑request overhead across thousands of items.
- **Binary ingress (Revision R3).** JSON parsing — even borrow‑only — is
  branch‑heavy at ingest scale. For the highest‑throughput loaders, offer an
  optional **binary framing** for `bulk_upsert`: the client ships length‑prefixed
  packed records (and may pre‑compute name fingerprints), carried either as a
  base64 blob inside the JSON arg (one decode + struct cast, no per‑field parse)
  or over a separate non‑MCP admin socket. This is explicitly *outside* strict MCP
  and offered only for bulk loads; the normal per‑call API stays JSON. The win is
  turning per‑field JSON tokenization into a single bounded memcpy + validate.

### 7.5 Annotation correctness (lets clients parallelize/cache)

MCP clients use annotations to decide what they can run concurrently or cache.
Two are wrong/missing today:

- `find_path` and `find_all_paths` are **read‑only** queries; `find_path` is
  currently `readOnlyHint: false`. Set both `readOnlyHint: true`. (A client that
  trusts `readOnlyHint` can fan these out without ordering them against writes.)
- Add `openWorldHint: false` to the read tools (the graph is a closed world the
  server owns) so clients don't assume external side effects.

### 7.6 Deliberately *not* adding

- Subscriptions / change feeds (`listChanged`): real value but a different
  consistency/transport story; out of scope for the throughput target.
- Vector / embedding search: the BM25 path is the committed model; a vector tier
  is a separate project, not a `tools.json` tweak.

Every change above is justified by **fewer bytes** and/or **bounded work per
call** — the two things that actually move throughput at the protocol layer.

---

## 8. Phased plan (no big bang; each phase is independently shippable & measured)

Each phase has an **exit criterion measured on the bench harness (Section 9)**.
We do not start a phase until the prior phase's number is in hand. The public
`GraphHandle` API and `tools.json` semantics stay stable across phases so tests
don't churn (new tools/params are additive).

- **P0 — Instrument & baseline.** Build the load generator + latency‑histogram
  harness (Section 9). Record today's p50/p99/throughput and RAM at 10 MB, 1 GB,
  10 GB. *Exit:* reproducible numbers we can regress against. **No engine change.**

- **P1 — Kill the cheap abstraction costs around the current engine.** Remove
  `spawn_blocking` for reads; borrow‑only JSON envelope extraction; reuse buffers;
  add `entity_exists`/`degree`/projection/cursor *tools* (served by the existing
  engine first). *Exit:* p99 down measurably with **zero storage change**; proves
  how much was transport/serialization vs storage.

- **P2 — Integer domain.** Introduce the `Interner` and make the graph API speak
  `EntityId` internally (TidesDB still the value store, now keyed by id). *Exit:*
  string hashing/cloning gone from hot path; search/BFS use `IdBitset` +
  `ScoreAccumulator`.

- **P3 — Native zero‑copy read path.** mmap `ValueLog` + `RecordRef`; reads served
  from the native path (lock‑free), writes still dual‑write to TidesDB as the
  durable log during transition. *Exit:* Tier A p99 < 100 µs on working set; native
  read beats TidesDB read on the bench.

- **P4 — Native write path + WAL/group commit; per‑shard locks; striped stats.**
  Retire the global write gate and TidesDB from the hot path; TidesDB optionally
  remains the cold tier or is dropped. *Exit:* write throughput scales with shards;
  reads unaffected by write load.

- **P5 — Adjacency CSR + delta; compaction; cursored hubs.** Native graph
  traversal; `get_neighbors`/`read_graph` cursors backed by CSR. *Exit:* hub
  pagination O(1) resume; Tier C never stalls Tier A/B under mixed load.

- **P6 — Cold tier & RAM‑floor options.** MPHF interner variant; optional on‑disk
  interner spill for the cold tail; tune page‑cache working set. *Exit:* documented
  RAM‑vs‑latency curve at 100 GB / 1 TB.

If we stop after P1–P2 we already have a markedly faster, lower‑allocation server
on the existing storage — the plan front‑loads the cheap wins.

---

## 9. Benchmark & test strategy (how we *prove* it, files to add/remove)

Throughput/latency is the metric, so the harness is a first‑class deliverable, not
an afterthought.

**Add:**
- `benches/load_gen.rs` — a closed‑loop + open‑loop load generator: N virtual
  clients (configurable to 10 k) over TCP/HTTP, a configurable op mix matching
  Section 1, producing **latency histograms (HdrHistogram), p50/p90/p99/p999, and
  throughput**, plus RSS sampling. Open‑loop (fixed arrival rate) is essential —
  closed‑loop hides queueing and flatters tail latency.
- `benches/dataset_gen.rs` — deterministic synthetic corpora at 10 MB / 1 GB /
  10 GB / (CI‑gated) 100 GB, power‑law degree distribution and a few hubs, so hub
  pagination and tail latency are actually exercised.
- `tests/concurrency.rs` — loom‑ or stress‑based: concurrent readers + writers
  asserting no torn reads, no lost updates (per‑shard atomicity), correct epoch
  reclamation (no use‑after‑free under miri/ASan where feasible).
- `tests/recovery.rs` — kill‑and‑reopen: WAL replay correctness, segment integrity
  after simulated crash mid‑write, `Durability::Sync` vs `Async` semantics.
- Microbenchmarks per custom structure (interner, CSR, varint, score
  accumulator) proving each beats its stdlib alternative — the gate for keeping
  the structure.

**Keep & extend:** `tests/redesign_tests.rs`, `tests/integration.rs`,
`tests/fuzzy.rs` — they pin the *semantics* and must pass unchanged across every
phase (the engine swap must be behavior‑preserving). Extend with projection/cursor
cases as the tools land.

**Bench acceptance gates (the contract):**
- Tier A p99 < 100 µs at the largest working‑set‑resident dataset, while a write
  load runs concurrently.
- Read throughput scales ≥ linearly to ~physical cores (lock‑free reads).
- RAM(RSS) tracks the Section‑4 model within a small constant; flat as value
  volume grows with entity count fixed.
- Tier C under load does not regress Tier A/B p99 (isolation).

**Possibly remove:** `src/bin/bench_stdio.rs` is single‑client and round‑trip‑
latency only; it stays useful for smoke but is **not** the scaling harness. Don't
rely on it for the target numbers.

---

## 10. Risks, limits, and the things we are explicitly trading

- **< 100 µs is a working‑set SLO, not a universal one.** Cold reads over 1 TB hit
  disk and will exceed it; the architecture confines that to Tier C and makes it
  explicit via cursors. We will not pretend otherwise.
- **RAM floor at huge N is real.** 10⁹ keys cost ~10–30 GB of index RAM for fast
  random access. The MPHF + cold‑spill options trade RAM for latency; the curve is
  documented, not hidden.
- **A custom engine is the riskiest part.** Mitigation: it is the *last* phases,
  gated behind benches, with TidesDB as the durable fallback during transition;
  P1–P2 deliver value with no engine change.
- **Epoch reclamation / lock‑free correctness is subtle.** Mitigation: loom +
  miri/ASan stress tests are a P3 gate, not an afterthought.
- **Thread‑per‑core is tempting but complicates graph walks.** Deferred to a
  measured option; not on the critical path.
- **Compaction is a background system with its own failure modes** (space
  amplification, write stalls). Bounded, scheduled, and load‑shed under pressure;
  covered by `recovery.rs` and soak tests.

---

## 11. One‑screen summary

- Physics first: < 100 µs ⇒ hot path is **RAM‑resident, lock‑free, zero‑copy,
  integer‑keyed, allocation‑ and syscall‑free on a hit**; disk is Tier C only.
- RAM ∝ **entity count**, not data volume: **compact indexes in RAM, values in
  mmap/page cache** → 1 TB from a few GB of index.
- Concurrency: **lock‑free reads (atomic offset + epoch), per‑shard write locks,
  striped counters, no global gate, no thread hop on a hit.**
- Custom structures replace each named abstraction cost (interner, RecordRef, CSR,
  varint, bitset, score accumulator, seqlock, striped counter, arena, JSON scan).
- `tools.json`: **projection + cursors + `entity_exists`/`degree`/neighbor
  pagination + bulk ingest + annotation fixes** — fewer bytes, bounded work.
- Staged P0→P6, each measured; cheap wins (P1–P2) first, custom engine last,
  behavior preserved throughout, TidesDB retired from the hot path only once beaten
  on the bench.

---

## 12. Revision log (adversarial review)

The first draft survived three rounds of stress‑testing; each found a real flaw,
folded back into the sections above.

- **R1 — `mmap` page‑fault trap in async.** Draft leaned on transparent `mmap` for
  the cold tier. A *major* fault parks the tokio worker (10–100 µs), stalling every
  task on it. `mincore`‑guarding is racy (TOCTOU) and adds a per‑read syscall. Fix
  (§3.2): `mlock`‑resident indexes are sliceable inline; values/postings use
  `io_uring`+`O_DIRECT` with a userspace cache so the reactor never faults on disk.
  Trade‑off accepted: we manage our own cache instead of the free OS page cache —
  which also makes "minimal RAM" a number we set.

- **R2 — physical‑offset / compaction paradox.** Draft both packed location into
  the id *and* moved records during compaction — contradictory, since moving a
  record would cascade rewrites into every edge/posting/interner slot. Fix (§3.1):
  stable monotonic **logical ids** + a dense `loc[]` indirection array;
  compaction flips exactly one atomic slot per moved record. One extra L1
  dereference on read buys background, write‑amplification‑free compaction.

- **R3 — leaving the hardware on the table.** Scalar loops over postings/CSR and
  branch‑heavy JSON ingest underuse the CPU. Fix: Roaring/Stream‑VByte postings
  with **AVX2/AVX‑512** scoring and sorted‑run intersection (§3.4); optional
  **binary ingress** for `bulk_upsert` (§7.4). SIMD is a Tier‑B percentile lever,
  gated behind microbenchmarks with scalar fallback — not a correctness dependency.

- **Open question answered — CSR delta fold without reader spikes (§5.5):**
  RCU/`ArcSwap` snapshots (lock‑free reads) + generational double‑buffered delta
  (sub‑µs pointer rotation, no lost writes) + lock‑free append‑only `active` log
  (wait‑free reads) + per‑shard size‑tiered bases (bounded, amortized folds) +
  epoch‑deferred batched reclamation (no inline large free) + per‑shard write
  admission control (shed write latency, never read latency).
