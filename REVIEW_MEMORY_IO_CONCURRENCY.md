# Harsh Review — Memory, IO, Concurrency

Date: 2026-06-18 · Scope: `src/kg.rs`, `src/store.rs`, `src/intern.rs`,
`src/search.rs`, `src/server.rs`, `src/http.rs`, `src/actions/memory.rs`.

## Verdict

The micro-optimizations are real (ctrl-byte tables, prefetch, hand-rolled JSON,
borrowing serialize views). But they sit on top of three architecture-level
defects that make every one of them moot at scale:

1. **Reads deep-copy most of the graph.** `GraphHandle::read()` clones the
   interner arena, name table, adjacency map, and search index on *every read*.
2. **Writes deep-copy the entire graph.** `snapshot()` rebuilds everything on
   *every write*, inside the write lock.
3. **A correctness bug** (`merge_entities` corrupts the adjacency index) and a
   **durability bug** (background `fsync` targets a stale fd after `compact`).

So the README's "fast vs JS server" table is measuring small graphs. On a large
graph the O(N)-per-operation copies dominate and the prefetch/ctrl-byte work is
noise. Fix the snapshot model first; everything else is secondary.

Severity legend: **[CRIT]** data loss / wrong results / O(N) per op ·
**[HIGH]** scaling cliff or availability · **[MED]** waste / latent ·
**[LOW]** cleanup.

---

## Concurrency

### [CRIT] C1 — `read()` deep-clones the graph on every read
`kg.rs:2894` `(**self.snapshot.load()).clone()`.

`ArcSwap::load()` already hands back a cheap `Arc<ReadSnapshot>`. The trailing
`.clone()` then deep-copies the `ReadSnapshot`. `entity_slots`/`relations` are
`Arc<[_]>` (cheap), **but** `interner: StringInterner`, `name_table:
ShardedNameTable`, `adjacency: AHashMap`, `search: SearchIndex`, and
`free_slots: Vec<u32>` are *owned* and clone deeply. That is the whole string
arena + the entire inverted index + the entire adjacency map copied for
`get_entity("Alice")`.

Result: every read tool is O(N) time and O(N) transient memory. N concurrent
reads = N full copies simultaneously → OOM risk.

**Fix:** `read()` returns `Arc<ReadSnapshot>` (or an `arc_swap::Guard`). Make the
owned sub-structures `Arc<...>` inside `ReadSnapshot` so the type is trivially
shareable. Readers hold an `Arc`; no copy. This is the single highest-leverage
change in the codebase.

### [CRIT] C2 — `snapshot()` rebuilds everything on every write
`kg.rs:2682`. Called from `WriteGuard::publish()` (`kg.rs:2763`) on every write,
under the lock: `interner.clone()`, `name_table.clone()`, `adjacency.clone()`,
`search.clone()`, plus `Arc::from_iter(entity_slots…cloned())` and
`Arc::from_iter(relations…cloned())` (full materializations, not Arc shares).

A one-observation `add_observations` on a 1M-entity graph copies the whole
multi-hundred-MB structure. Peak memory is 2–3× resident (old snapshot still
held by in-flight readers + new snapshot + live graph).

**Fix (incremental):** move the sub-structures behind `Arc` and rebuild only what
a given mutation changed (copy-on-write per structure). E.g. `add_observations`
only needs a new `search` + new `entity_slots`; it can `Arc::clone` the
interner, name_table, adjacency. Longer term, consider an MVCC/epoch scheme so
writers don't rebuild snapshots at all.

### [HIGH] C3 — stdio/TCP run blocking work on async worker threads
`server.rs:248` `serve_line_conn` calls `dispatch_line` *inline* on the tokio
task. That path takes the `parking_lot::Mutex` and does blocking `write()`
syscalls (and, today, the O(N) clones above). HTTP correctly offloads via
`spawn_blocking` (`http.rs:64`); stdio and TCP do not. On the multi-thread
runtime (`main.rs:8` `Runtime::new()`), a slow write on one TCP connection
blocks a worker thread and starves other connections.

**Fix:** wrap the dispatch in `tokio::task::spawn_blocking` for TCP (and stdio),
or run a dedicated blocking dispatch pool, mirroring the HTTP path.

### [MED] C4 — unbounded connection concurrency on TCP
`server.rs:226` spawns a task per accepted socket with no cap. Trivial DoS / fd
exhaustion. Add a `Semaphore` (or a connection limit) around `accept`.

### [LOW] C5 — `read_graph_cached` double-build race
`kg.rs:2883`: two first-time concurrent readers both build the JSON then both
`store`. Benign (same bytes, last wins) but wasted work. A `OnceCell`-style
single-flight or just accept it. Note it also goes through the O(N) `read()` of
C1, so it inherits that cost on cache miss.

---

## IO / Durability

### [HIGH] D1 — background `fsync` targets a stale fd after `compact`
`GraphHandle::new` captures `sync_file = Arc::clone(&kg.store.sync_file)` once
(`kg.rs:2821`) and the sync thread fsyncs *that* handle forever (`kg.rs:2856`).
`compact()` does `*self = KnowledgeGraph::new(&path)` (`kg.rs:1614`), which
builds a **new** `BinaryStore` with a **new** `sync_file`. After any compaction,
the background thread keeps fsyncing the old (renamed-over / unlinked) inode, and
**new writes are never fsynced** — they reach the kernel page cache via
`publish()`'s flush but are not forced to disk. Acknowledged writes after a
compact are lost on power failure. Directly contradicts the durability story.

**Fix:** the sync thread must reference the *current* file. Options: store the
sync target in an `ArcSwap<File>` the thread reloads each iteration and have
`compact`/`reopen_truncated` update it; or have the sync thread fsync via the
`GraphHandle` (re-fetch the fd under a short lock); or fsync inside the write
path for compacted state. Add a regression test: write → compact → write →
assert the post-compact record is on disk after dropping the handle.

### [HIGH] D2 — a torn record mid-stream bricks the whole database
`store.rs` replay tolerates `UnexpectedEof` only on the **length** read
(`store.rs:144`). The **kind** and **payload** reads use `?` (`store.rs:157`,
`store.rs:163`), so a crash that wrote a length prefix but only part of the
payload makes replay return `Err` → `KnowledgeGraph::new` fails → the server
refuses to start. The existing `test_truncated_log_handling` only cuts after the
MAGIC, so it never exercises this. A power loss mid-`write_record` can render the
store unopenable.

**Fix:** treat `UnexpectedEof` on the kind/payload reads as a torn tail — stop
replay and return `Ok` (optionally truncate the file back to the last good
record). Add CRC32 per record so a torn-but-length-plausible record is detected
rather than silently applied. Add a test that writes a valid record then appends
a partial record and asserts clean recovery of the first.

### [MED] D3 — write before durability; no strong-durability mode
`publish()` flushes to the kernel and returns; the client gets its response
before the background `fsync` runs. Fine for a best-effort memory server, but
there is no opt-in "fsync before ack" mode and the window is undocumented at the
API level. Combined with D1 this is worse than it looks. Make the durability
policy explicit and configurable (`async` default, `sync` optional).

### [MED] D4 — partial batch persisted but reported as total failure
In `create_entities` (`kg.rs:1621`) each entity is logged then applied in a loop.
If entity *k*'s `write_record` fails, entities `0..k` are already logged **and**
applied (and will be flushed on the next publish), but the function returns
`Err`, so the client sees a failure for an operation that partially succeeded.
Memory and log stay consistent, but the client's view does not. Either make the
batch atomic (wrap in `TxnBegin/TxnCommit` like `merge_entities`) or return a
partial-success result.

### [LOW] D5 — no file lock; `u32` length cast can wrap
Two processes on the same memory file will interleave appends and corrupt it; add
an advisory `flock`. And `write_record`'s `total_len as u32` (`store.rs:83`) can
truncate before the bound check on a pathological >4 GB payload (currently
unreachable via request caps, but make it a `u64` check for safety).

---

## Memory

### [CRIT] M1 — same as C1/C2
The dominant memory problem *is* the per-read and per-write deep copies. Nothing
else matters until those are Arc-shared. (See C1, C2.)

### [HIGH] M2 — `merge_entities` leaves the adjacency index stale → wrong reads
*Correctness bug, surfaced here because it rides the snapshot.* `merge_entities`
(`kg.rs:2409`) applies its records through `Self::apply_record`
(`kg.rs:2503`), and `apply_record`'s `CreateRelation` arm (`kg.rs:1250`) and
`replay_delete_entity` (`kg.rs:1352`) **do not touch `self.adjacency`** — those
static replay helpers were written for load time, when adjacency is rebuilt from
scratch afterward (`kg.rs:1207`). So after a live merge: the source's edges
linger in `adjacency` and the redirected edges are missing. Every adjacency
consumer — `find_path`, `neighbors(depth≥2)`, `extract_subgraph`,
`find_all_paths` — returns wrong results until the next restart/compact rebuilds
the index. The stale map is then frozen into the published snapshot, so reads are
wrong too.

**Fix:** maintain `adjacency` in `merge_entities` (mirror what
`create_relations`/`delete_entities` do), or route the merge through the public
mutators instead of the raw replay helpers. Add a test: create A,B,C with edges,
`merge_entities(A→B)`, assert `find_path`/`neighbors` reflect the merge.

### [MED] M3 — `delete_entities` adjacency cleanup is quadratic
`kg.rs:1819-1825`: for each deleted id, iterate **all** adjacency values and
`retain`. O(deleted × nodes × degree). Bulk deletes blow up. Collect the deleted
set once and do a single pass over `adjacency.values_mut()`.

### [MED] M4 — interner never reclaims between compactions
`intern()` only appends to the arena (`intern.rs:106`); deleted/edited strings
live until `compact`. Documented, but there is no size/ratio-based auto-compaction
trigger, so a long-running server with churn grows unbounded. Add a heuristic
(e.g. compact when dead-bytes ratio exceeds a threshold) or expose it.

### [MED] M5 — adjacency stores both directions (2× edges) and is cloned per snapshot
`create_relations` pushes `(to,type)` under `from` **and** `(from,type)` under
`to` (`kg.rs:1716-1717`). That doubles the index, and the whole map is deep-copied
on every read/write today (C1/C2). After the Arc fix it's shared, but consider
whether both directions are needed or whether a single CSR-style structure is
cheaper.

### [LOW] M6 — vestigial `state`/`is_live` machinery
`StoredEntity.state` is only ever set to `ENTITY_SLOT_LIVE` (`kg.rs:1311`,
`1662`); tombstoning is done via `Option = None`. No `Some(entity)` is ever
non-live, so `is_live()` is always true and the `ENTITY_SLOT_LIVE` constant +
checks are dead weight scattered across the file. Either remove it or actually use
in-place soft-delete (which would also avoid the `Option` niche dance).

### [LOW] M7 — `ReadSnapshot.free_slots` is cloned for nothing
`kg.rs:458` is `#[allow(dead_code)]` yet deep-cloned on every snapshot. Drop it
from `ReadSnapshot`.

### [LOW] M8 — `read_graph_json` capacity uses `entity_slots.len()`
`kg.rs:565` sizes the buffer by the slot count (includes tombstones). Over-
allocates after churn. Use a live count or accept the over-estimate.

---

## Recommended order of work

**Phase 0 — correctness/durability (ship first, small diffs):**
- M2: fix `merge_entities` adjacency maintenance (+ test). *Wrong results today.*
- D1: fix stale `fsync` fd after compact (+ test). *Silent data loss today.*
- D2: tolerate torn tail on kind/payload reads; add per-record CRC (+ test).
  *Unopenable DB after a crash today.*

**Phase 1 — the snapshot model (the real win):**
- C1: `read()` returns `Arc<ReadSnapshot>`; make sub-structures `Arc`. Update the
  ~15 read handlers in `actions/memory.rs` to take the shared snapshot.
- C2: copy-on-write `snapshot()` — rebuild only changed structures.
- Re-run `benches/graph_bench.rs` on a large graph (≥100k entities) and refresh
  the README table; the current numbers are small-graph numbers.

**Phase 2 — runtime hygiene:**
- C3: `spawn_blocking` for stdio/TCP dispatch.
- C4: bound TCP connection concurrency.
- D4: make batch writes atomic or report partial success.
- D3: configurable durability (`async`/`sync`).

**Phase 3 — cleanup:**
- M3 (quadratic delete), M4 (auto-compaction), M5 (adjacency layout), M6–M8.

## Test gaps to close
- Crash recovery from a **mid-record** torn write (not just post-MAGIC).
- write → `compact` → write → reopen: assert the post-compact write survives.
- `merge_entities` followed by `find_path`/`neighbors`/`extract_subgraph`.
- Concurrent readers during a writer: assert snapshot isolation and no O(N) blowup
  (a memory-ceiling assertion under load).
- Bulk `delete_entities` performance regression guard.
