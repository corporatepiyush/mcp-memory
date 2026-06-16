# mcp-memory — Harsh Review & Optimization Plan

A review of CPU, memory, I/O, and security/durability trade-offs for the
knowledge-graph MCP server. Findings are ordered by severity. Each item lists
the **location**, **why it's wrong**, **impact**, and a **concrete fix**.

The short version: the code spends enormous effort on micro-optimizations
(hand-rolled SwissTable, software prefetch, ctrl-byte probing) while the actual
hot paths are **O(n) or O(n²)**, the durability story is **broken**, and the
server is **remotely crashable**. The fancy parts optimize the cheap things;
the expensive things were left naive.

---

## 0. Critical correctness & durability bugs (fix first — these lose data or crash)

### C1. Log writes silently ignore errors → memory/disk divergence, silent data loss
**Where:** `src/kg.rs:899-951` — every `write_*_log` does `let _ = store.write_record(...)`.
**Problem:** In-memory state is mutated *first*, then the log write result is
discarded. If the write fails (record > `MAX_RECORD_BYTES` = 1 MiB, disk full,
I/O error), the graph in RAM diverges from the log on disk. On restart the
change is gone — with no error ever returned to the client.
**Impact:** Silent, unrecoverable data loss. The "durable log" is not durable.
**Fix:**
- Propagate the `io::Result` out of `write_*_log` and up through the public
  mutating methods (`create_entities`, etc.). Return an MCP error on failure.
- Write the log record **before** committing to in-memory state (write-ahead),
  or at minimum roll back the in-memory mutation if the write fails.
- Decide a single source of truth: log-first (WAL semantics) is the correct
  model here.

### C2. Oversized strings panic → remote denial-of-service
**Where:** `src/store.rs:165` — `encode_str` does `assert!(len <= u16::MAX ...)`.
Combined with `panic = "abort"` (`Cargo.toml:29`) and the 16 MiB request cap
(`src/server.rs:98`).
**Problem:** A client can send an entity name or observation longer than 65 535
bytes (the request limit allows 16 MiB). `encode_str` asserts, panics, and
because `panic = "abort"` the **entire server process dies**, dropping all
in-flight work for every client.
**Impact:** Trivial remote crash / DoS. One bad tool call kills the server.
**Fix:**
- Replace the `assert!` with a returned `io::Error` (or a validation error).
- Enforce explicit, documented per-field limits (e.g. name ≤ 1 KiB,
  observation ≤ 64 KiB) at the API boundary in `actions/memory.rs`, rejecting
  with `InvalidParams` *before* mutating anything.
- Reconsider `panic = "abort"` for a long-running server; `unwind` lets a
  poisoned request fail without taking the process down (the input parsing path
  in particular should never abort the daemon).

### C3. `compact()` is not crash-safe — a crash mid-compaction destroys the log
**Where:** `src/kg.rs:610-649`, `src/store.rs:145-157` (`reopen_truncated`).
**Problem:** `compact` **truncates the live log first** (`reopen_truncated`),
then rewrites records one by one. A crash (or any `write_*_log` failure, see C1)
between truncate and full rewrite leaves a truncated/partial log. The
authoritative on-disk state is gone.
**Impact:** Data loss on crash during maintenance.
**Fix:** Write the compacted log to a temp file (`<path>.tmp`), `flush` +
`sync_all`, then atomically `rename` over the original (atomic on POSIX). Only
then swap the writer. Never destroy the old log until the new one is durable.

### C4. Read/search operations mutate state and leak memory permanently
**Where:** `src/kg.rs:481, 521, 543` — `get_entity`, `search_relations`,
`find_path` all take `&mut self` and call `self.interner.intern(...)` on
**caller-supplied query strings**.
**Problem:** The interner arena is append-only (`src/intern.rs:108-109`). Every
*unique* query/filter string is interned and **never freed**. A read-only
search permanently grows memory. An attacker (or just normal varied querying)
can grow the arena without bound by issuing distinct queries.
**Impact:** Unbounded memory growth driven by *reads*; reads should not write.
**Fix:**
- Add a non-interning lookup path: hash the query with the same hasher and probe
  the dedup table **without inserting** (return `Option<StrId>`; `None` ⇒ no
  match, short-circuit the query).
- `search_relations`/`get_entity`/`find_path` should become `&self` once they no
  longer intern. `SearchIndex::search` already avoids interning the query — model
  the others on it.

### C5. No `fsync` — "persisted" data is only in the OS page cache
**Where:** `src/store.rs` — `write_record` calls `flush()` (line 72) but never
`File::sync_data`/`sync_all`. `close`/`reopen_truncated` likewise only `flush`.
**Problem:** `flush` on a `BufWriter` only pushes bytes to the kernel, not to
stable storage. On power loss / kernel panic, acknowledged writes are lost.
**Impact:** The system flushes on *every* record (slow — see I1) yet is still
**not crash-durable** — the worst of both worlds.
**Fix:** Decide an explicit durability policy and document it:
- Durable mode: `sync_data()` once per request/batch (not per record).
- Throughput mode: buffer and flush+sync on a timer or on batch boundary.
Either way, stop paying per-record flush cost while getting no durability.

---

## 1. CPU / algorithmic complexity

### P1. "Inverted index" search is a full linear substring scan — O(total_tokens × query_len)
**Where:** `src/search.rs:60-82` (`SearchIndex::search`).
**Problem:** The struct comment claims a sorted inverted index "for
cache-friendly lookups via binary search," but `search` **iterates every entry**
and runs `windows(qlen).any(...)` substring matching on each token. The sort is
never used. This is the single most expensive operation in the server and it is
linear in the entire corpus per query.
**Impact:** Search latency grows with total data size, regardless of result
count. Defeats the entire point of an index.
**Fix:** Pick one and commit:
- **Exact / prefix token match:** `binary_search` the sorted `entries` for the
  token range → O(log n + matches). This is what the data structure is *already
  built for*.
- **Substring match:** if substring search is a real requirement, the flat
  sorted vec is the wrong structure — build a real inverted index
  `HashMap<token_StrId, Vec<entity_idx>>`, or a trigram/suffix structure. A
  sorted `Vec<(StrId,u32)>` cannot accelerate substrings.

### P2. Index construction is O(n²) — mid-vector inserts
**Where:** `src/search.rs:107-114` (`insert_entry`) — `Vec::insert` at a
binary-searched position shifts the tail on every token.
**Problem:** Inserting T tokens one at a time into a sorted vec is O(T²) from the
memmoves. `index_entity` is called on every create and on **every**
`add_observations`/`delete_observations` (full re-index).
**Impact:** Bulk loads and edits degrade quadratically.
**Fix:** Batch-build: collect all `(token, idx)` pairs, then `sort_unstable` +
`dedup` once (O(T log T)). For incremental updates, append to a per-entity
posting list (`HashMap<token, SmallVec<u32>>`) instead of a global sorted vec.

### P3. `add_observations` / `delete_observations` re-index the entire entity every call
**Where:** `src/kg.rs:736-738, 781-783` → `search.remove_entity` +
`search.index_entity`.
**Problem:** `remove_entity` (`src/search.rs:55`) does `entries.retain(...)` —
a full O(index_size) scan — then the entity is fully re-tokenized and
re-inserted (each insert O(n), per P2). Adding one observation costs
O(index_size).
**Impact:** Observation edits scale with total corpus size, not with the change.
**Fix:** Incrementally index only the *new* tokens on add, and remove only the
*dropped* tokens on delete. Keep a per-entity posting list so removal is O(tokens
of that entity), not O(everything).

### P4. `find_path` BFS is O(V × E) — rescans all relations per node
**Where:** `src/kg.rs:567-583`.
**Problem:** For every dequeued node it loops over the **entire** `relations`
vector to find neighbors. Proper BFS is O(V + E); this is O(V × E).
**Impact:** Path queries blow up on graphs with many relations.
**Fix:** Build an adjacency list once (`HashMap<StrId, Vec<(StrId, edge)>>`,
maintained incrementally on relation create/delete) and traverse that. For a
one-shot query, build it once at the top of `find_path` (O(E)) rather than per
node.

### P5. Relation dup-check and deletions are O(n) / O(n²)
**Where:**
- `src/kg.rs:693-696` — `create_relations` linear `any(...)` dup check per
  relation ⇒ O(n²) for a batch.
- `src/kg.rs:758-760, 788-800` — `delete_entities`/`delete_relations` use
  `Vec::contains` inside `retain` ⇒ O(deleted × relations).
**Fix:** Maintain a `HashSet<(StrId, StrId, StrId)>` of relations for O(1) dup
checks, and collect deletion targets into a `HashSet` so `retain` predicates are
O(1) per relation.

### P6. `search_nodes` / `open_nodes` relation filtering is O(entities × relations)
**Where:** `src/kg.rs:842, 871` — `entity_names.contains(&r.from)` (linear) for
every relation.
**Fix:** Put matched names in a `HashSet<StrId>` and probe in O(1).

### P7. Observation dedup is O(obs²) per batch
**Where:** `src/kg.rs:730, 411` — `entity.observations.contains(&oid)` for each
new observation; `delete_observations` `remove_ids.contains` similarly
(`src/kg.rs:460, 780`).
**Fix:** For batches above a small threshold, build a `HashSet<StrId>` of
existing/removal ids. Below the threshold the linear scan is fine.

### P8. Over-engineered hash tables vs. the real bottleneck
**Where:** `src/kg.rs:18-263` (custom sharded SwissTable + software prefetch),
`src/intern.rs` (second hand-rolled ctrl-byte table).
**Problem:** Two bespoke open-addressing hash tables, ctrl-byte stamping, and
`_mm_prefetch` 4 slots ahead — for tables that are typically hundreds to a few
thousand entries, while `search`/`find_path` remain O(n)/O(n²). This is effort
spent in the wrong place, plus a large `unsafe` surface (`get_unchecked`, raw
pointer probing) to maintain.
**Recommendation:** Replace both with `hashbrown`/`ahash`-backed `HashMap`
unless a benchmark proves the custom tables matter. Spend the saved complexity
budget on P1–P4, which dominate real cost. Sharding a single-threaded,
mutex-guarded map (`NAME_TABLE_SHARDS = 4`) buys nothing — there is no
concurrent access.

---

## 2. Memory

### M1. Append-only interner never reclaims — deletes and edits leak
**Where:** `src/intern.rs:94-131` (arena + offsets + hashes are append-only).
**Problem:** Deleting entities/observations or replacing observations leaves the
old strings in the arena forever. `compact()` rewrites the *log* but does **not**
rebuild the interner or entity slots, so RAM is never reclaimed. Combined with
C4 (queries interned forever), long-running memory grows monotonically.
**Impact:** Memory leak under any churn; `compact` doesn't help RAM at all.
**Fix:** Make `compact` rebuild from scratch into a fresh `KnowledgeGraph`
(new interner, compacted slots, rebuilt index) and swap it in. That reclaims
arena, tombstoned slots, and stale index entries in one pass.

### M2. Dead entity slots are never reclaimed
**Where:** `src/kg.rs:752` (`entity_slots[slot] = None`) vs `:667`
(`create_entities` always pushes `entity_slots.len()`).
**Problem:** Slots are tombstoned, never reused. `entity_slots` grows forever
under create/delete churn. `read_graph`/`graph_stats` scan all tombstones too.
**Fix:** Maintain a free-list of dead slot indices and reuse them on create
(careful to also reset the search index for the reused slot). Or rely on M1's
rebuild-on-compact to reset.

### M3. `enc_buf.clone()` on every single log write defeats the reuse buffer
**Where:** `src/kg.rs:902, 911, 920, 929, 938, 947`.
**Problem:** `enc_buf` exists to avoid per-write allocation, but each writer
clones it into a fresh `Vec` before writing — a heap allocation + copy per
record, exactly what the buffer was meant to prevent. (The clone exists only to
dodge a borrow conflict with `self.store`.)
**Fix:** Restructure so the encode buffer and the store don't both borrow
`self` — e.g. move the store out from behind the redundant inner mutex (see X2),
or encode into a stack/local buffer passed by reference. Write directly from the
buffer; no clone.

### M4. Redundant per-string hash storage
**Where:** `src/intern.rs:56` (`hashes: Vec<u64>`) and `src/kg.rs:74`
(`NameTableShard.hashes`, commented "used only during grow/rehash").
**Problem:** 8 bytes/string in the interner purely to serve `get_hash`, plus
8 bytes/slot in every name-table shard purely for rehash. On rehash the hash can
be recomputed from the interned bytes; `get_hash` exists only to feed the name
table, which only needs it because it stores `StrId` instead of the string.
**Fix:** Drop `NameTableShard.hashes` and recompute during `grow` (grow is
rare). Evaluate whether `interner.hashes` is needed at all once the name table
stops requiring an external hash.

### M5. `obs_ids.clone()` on entity create/replay
**Where:** `src/kg.rs:389, 672`.
**Problem:** The `Vec<StrId>` is cloned to hand one copy to storage and one to
the search index.
**Fix:** Index first using the borrowed slice, then move the vec into the slot
(or vice versa) — no clone needed.

### M6. Full-graph string materialization on read
**Where:** `src/kg.rs:807-824` (`read_graph`), `entity_to_output:885-895`, then
`serde_json::to_string_pretty` in handlers.
**Problem:** `read_graph` allocates a fresh `String` for every name/type/
observation, builds owned `Entity`/`Relation` vecs, then `to_string_pretty`
serializes (another full copy) — and pretty-printing inflates payload size.
**Impact:** Peak memory ≈ 2–3× the graph for a single `read_graph`.
**Fix:** Serialize directly from interned `&str` via a borrowing
`Serialize` view (no intermediate owned structs). Use compact JSON, not
`to_string_pretty`, for machine-to-machine MCP responses (pretty everywhere is
pure waste — see I3).

---

## 3. I/O

### I1. `flush()` on every record kills batching
**Where:** `src/store.rs:72` (`write_record` ends with `self.writer.flush()?`).
**Problem:** The 64 KiB `BufWriter` (`src/store.rs:48`) is rendered useless: a
batch of 1 000 `create_entities` triggers 1 000 flushes (kernel writes). The
buffer never accumulates.
**Impact:** Write throughput is syscall-bound; bulk inserts are far slower than
necessary, with no durability benefit (see C5).
**Fix:** Remove the per-record flush. Flush once per request/batch boundary
(in the server loop after a `tools/call` completes), and pair with a single
`sync_data` if durable mode is selected.

### I2. Unnecessary buffer zeroing on replay
**Where:** `src/store.rs:126-127` — `payload_buf.resize(payload_len, 0)` zeros
the buffer before `read_exact` overwrites it.
**Fix:** Reserve capacity and read into an uninitialized tail (`read_buf`, or
`spare_capacity_mut` + `set_len`), or read directly into a slice you immediately
overwrite. Avoids memset on every record during startup replay.

### I3. Pretty-printing every response
**Where:** all handlers in `src/actions/memory.rs` use
`serde_json::to_string_pretty`.
**Problem:** MCP responses are consumed by a program, not a human. Pretty output
adds whitespace bytes (larger payloads, more write work) and is slower to
serialize.
**Fix:** Use `serde_json::to_string` (compact). Saves CPU and I/O on every call.

### I4. `tools.json` re-parsed on every `tools/list`
**Where:** `src/server.rs:189-193` — `serde_json::from_str` of the embedded
~8 KB file on each call.
**Fix:** Parse once into a `OnceLock<Value>` (or precompute the response string)
and clone/reference thereafter.

---

## 4. Concurrency / architecture

### X1. Blocking file I/O under a std `Mutex` on a tokio worker
**Where:** `src/server.rs:156-226` runs `process_request` (which locks the KG
and performs blocking `write` + `flush` syscalls) directly on the async runtime.
**Problem:** The synchronous, disk-touching critical section runs on a tokio
worker thread, blocking the executor. With a global lock, the server is
effectively single-threaded *plus* async overhead — no concurrency is gained,
and one slow disk write stalls a worker.
**Fix:** Either (a) run the whole thing as a plain synchronous stdio loop (this
workload has no concurrency to exploit — drop tokio), or (b) move KG mutations
to a dedicated writer thread / `spawn_blocking` and keep the async loop for I/O
only. Given the design, (a) is simpler and removes the tokio dependency surface.

### X2. Redundant double locking
**Where:** `src/kg.rs:274` (`store: Mutex<BinaryStore>`) inside a `KnowledgeGraph`
that is itself always accessed behind `Arc<Mutex<KnowledgeGraph>>`
(`src/server.rs:77`).
**Problem:** Every method already holds `&mut self` exclusively, so the inner
store mutex is never contended — it's pure lock/unlock overhead and forces the
`enc_buf.clone()` dance in M3.
**Fix:** Remove the inner `Mutex`; store `BinaryStore` directly. The outer lock
already serializes access. (The replay closure can take `&store` directly.)

### X3. Unsafe raw-pointer aliasing during replay is fragile
**Where:** `src/kg.rs:293-362` — five `*mut` pointers to distinct fields are
dereferenced inside the replay closure to dodge the borrow checker.
**Problem:** Sound only because the fields are disjoint and `store` is a separate
field; any future refactor that aliases them is instant UB. Large, unnecessary
`unsafe` block.
**Fix:** Replay into local owned collections, then assign into `self` after the
closure; or take `&mut` to the individual fields via a small helper struct
borrowed disjointly. No raw pointers needed.

---

## 5. Security / robustness (smaller, but real)

- **S1. No input size/count limits.** Beyond C2, there is no cap on the number
  of entities, relations, or observations per request, nor total graph size.
  A client can exhaust memory. Add configurable limits, enforced in
  `actions/memory.rs` before mutation.
- **S2. `from_utf8_lossy` silently corrupts input.** `src/server.rs:35,46`
  replaces invalid UTF-8 with U+FFFD, so a malformed request becomes a
  *different* valid-looking request rather than a clean parse error. Prefer
  rejecting non-UTF-8 input explicitly.
- **S3. `get_unchecked` lookups trust caller-provided `StrId`.** `src/intern.rs:124,140`
  index the arena with no bounds check. Safe only if every `StrId` originates
  from this interner. A stray/cross-interner id is UB. Add `debug_assert`s, or
  bounds-check in non-release builds; document the invariant loudly.
- **S4. Misleading file naming/format.** Default path is `memory.jsonl`
  (`src/config.rs:14,23`) and the crate description says "persisted via JSONL,"
  but the format is a custom binary log (`MCPMEMV1`). Anyone inspecting/backing
  up the `.jsonl` file gets binary garbage. Rename default to `.mcpmem`/`.bin`
  and fix the description.
- **S5. Unknown record kinds are silently skipped on replay.** `src/kg.rs:132`
  (`if let Some(kind) = RecordKind::from_u8`) — a corrupt/forward-version byte is
  dropped without error, silently losing that record while continuing. At least
  log a warning; consider failing closed on unknown kinds within a known-version
  file.

---

## 6. Dead code / hygiene

- `tools::is_write_tool` (`src/tools.rs:31`) and the `idempotent`/`destructive`
  fields are never used. Either wire them into permission gating or remove.
- `KnowledgeGraphOut::empty()` (`src/types.rs:26`) is unused.
- `RecordKind::from_u8` is `pub` but only used internally.
- Lints are extensive in `Cargo.toml` but several real issues above
  (e.g. redundant clones in M3/M5) would be caught by `redundant_clone` /
  `implicit_clone` — confirm clippy actually runs in CI.

---

## 7. Suggested execution order

Highest value per unit of effort, roughly:

1. **C2** (stop the remote crash) and **C1** (stop silent data loss) — these are
   showstoppers and small to fix.
2. **C4 + C5 + C3** — durability and the read-path memory leak.
3. **P1** (real search) and **P4** (real BFS) — the biggest CPU wins; user-visible.
4. **I1 + I3** — remove per-record flush and pretty-printing; cheap, broad wins.
5. **M1/M2 via rebuild-on-compact** — bound memory growth.
6. **X1 + X2 + M3** — simplify the concurrency/store architecture; removes the
   redundant lock and the per-write clone together.
7. **P2/P3/P5/P6/P7** — finish the algorithmic cleanup.
8. **P8 / M4 / X3** — only after benchmarks justify keeping (or removing) the
   bespoke hash tables and unsafe replay.

### Guardrails before changing perf code
There are no benchmarks. Before/after any P-item, add a Criterion bench (bulk
create, search, find_path on a 10k-entity / 50k-relation graph) so the
optimizations are measured, not assumed — which is precisely the trap the
current SwissTable/prefetch code fell into.
