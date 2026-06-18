use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex as StdMutex};

use ahash::{AHashMap, AHashSet};
use arc_swap::ArcSwap;

use serde::ser::{Serialize, SerializeSeq, SerializeStruct, Serializer};

use crate::errors::{MCSError, Result};
use crate::intern::{StrId, StringInterner};
use crate::types::{Entity, Relation, KnowledgeGraphOut};
use crate::search::SearchIndex;
use crate::store::{self as store_enc, BinaryStore, RecordKind};

const ENTITY_SLOT_LIVE: u8 = 1;
const NAME_TABLE_SHARDS: usize = 4;

// ---------------------------------------------------------------------------
// Prefetch helper – issues a non-binding software prefetch hint to pull a
// cache-line into L1/L2 while we finish probing the current entry.
// ---------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prefetch_addr(addr: *const u8) {
    // _MM_HINT_T0 = 3  (temporal prefetch to all cache levels)
    std::arch::x86_64::_mm_prefetch::<3>(addr);
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
const unsafe fn prefetch_addr(_addr: *const u8) {}

/// fsync the directory containing `path` so a rename/create inside it is durable.
/// On platforms where a directory cannot be opened/synced this is a no-op.
fn sync_parent_dir(path: &Path) -> std::io::Result<()> {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    let dir = match dir {
        Some(d) => d,
        None => Path::new("."),
    };
    match std::fs::File::open(dir) {
        Ok(f) => match f.sync_all() {
            Ok(()) => Ok(()),
            // Some filesystems disallow fsync on a directory handle; tolerate it.
            Err(e) if e.kind() == std::io::ErrorKind::InvalidInput => Ok(()),
            Err(e) => Err(e),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------------
// StoredEntity / StoredRelation – internal representations using StrId.
// ---------------------------------------------------------------------------
// Default layout: 40 B / align 8 (Rust packs `state` into the Vec's padding and
// the `Option` niche is free). With `cache_align`, the slot is rounded to a full
// 64-byte line so a point lookup/mutation (name_table -> slot index) touches
// exactly one cache line instead of occasionally straddling two. Costs +60%
// memory and a wider stride on bulk scans — measure before enabling.
#[derive(Clone)]
#[cfg_attr(feature = "cache_align", repr(align(64)))]
pub(crate) struct StoredEntity {
    state: u8,
    pub(crate) name: StrId,
    pub(crate) entity_type: StrId,
    pub(crate) observations: Vec<StrId>,
}

impl StoredEntity {
    pub(crate) const fn is_live(&self) -> bool {
        self.state == ENTITY_SLOT_LIVE
    }
}

// Default layout: 12 B / align 4 → ~1 in 5 records straddles a 64-byte line.
// With `cache_align`, align(16) rounds the size to 16 B so 4 records fill a line
// exactly (no straddle, AVX2-load-friendly) for +33% memory.
#[derive(Clone)]
#[cfg_attr(feature = "cache_align", repr(align(16)))]
pub(crate) struct StoredRelation {
    pub(crate) from: StrId,
    pub(crate) to: StrId,
    pub(crate) relation_type: StrId,
}

// ---------------------------------------------------------------------------
// Borrowing serialization views (M6).
//
// Read tools used to build owned `Entity`/`Relation` vecs (a fresh `String`
// per name/type/observation) and *then* serialize them — roughly 2-3x the
// graph resident at once. These views instead hold references to the selected
// stored records and emit their interned `&str` directly during
// serialization, with no intermediate owned strings. The emitted JSON is
// byte-for-byte identical to serializing `KnowledgeGraphOut`.
// ---------------------------------------------------------------------------

/// A borrowing view over a selected slice of the graph. Serializes to
/// `{"entities":[...],"relations":[...]}`.
pub struct GraphView<'a> {
    kg: &'a KnowledgeGraph,
    entities: Vec<&'a StoredEntity>,
    relations: Vec<&'a StoredRelation>,
}

impl GraphView<'_> {
    /// Materialize into the owned [`KnowledgeGraphOut`]. Used by the direct
    /// (non-serializing) callers and tests; the server's read handlers
    /// serialize the view directly instead.
    pub fn to_owned_out(&self) -> KnowledgeGraphOut {
        KnowledgeGraphOut {
            entities: self.entities.iter().map(|e| self.kg.entity_to_output(e)).collect(),
            relations: self.relations.iter().map(|r| self.kg.relation_to_output(r)).collect(),
        }
    }
}

impl Serialize for GraphView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut st = s.serialize_struct("KnowledgeGraphOut", 2)?;
        st.serialize_field("entities", &EntityListRef { kg: self.kg, items: &self.entities })?;
        st.serialize_field("relations", &RelationListRef { kg: self.kg, items: &self.relations })?;
        st.end()
    }
}

struct EntityListRef<'a> { kg: &'a KnowledgeGraph, items: &'a [&'a StoredEntity] }
impl Serialize for EntityListRef<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.items.len()))?;
        for &e in self.items {
            seq.serialize_element(&EntityRef { kg: self.kg, e })?;
        }
        seq.end()
    }
}

struct RelationListRef<'a> { kg: &'a KnowledgeGraph, items: &'a [&'a StoredRelation] }
impl Serialize for RelationListRef<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.items.len()))?;
        for &r in self.items {
            seq.serialize_element(&RelationRef { kg: self.kg, r })?;
        }
        seq.end()
    }
}

struct EntityRef<'a> { kg: &'a KnowledgeGraph, e: &'a StoredEntity }
impl Serialize for EntityRef<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut st = s.serialize_struct("Entity", 3)?;
        st.serialize_field("name", self.kg.interner.lookup(self.e.name))?;
        st.serialize_field("entityType", self.kg.interner.lookup(self.e.entity_type))?;
        st.serialize_field("observations", &ObsRef { kg: self.kg, obs: &self.e.observations })?;
        st.end()
    }
}

struct ObsRef<'a> { kg: &'a KnowledgeGraph, obs: &'a [StrId] }
impl Serialize for ObsRef<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.obs.len()))?;
        for &o in self.obs {
            seq.serialize_element(self.kg.interner.lookup(o))?;
        }
        seq.end()
    }
}

struct RelationRef<'a> { kg: &'a KnowledgeGraph, r: &'a StoredRelation }
impl Serialize for RelationRef<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut st = s.serialize_struct("Relation", 3)?;
        st.serialize_field("from", self.kg.interner.lookup(self.r.from))?;
        st.serialize_field("to", self.kg.interner.lookup(self.r.to))?;
        st.serialize_field("relationType", self.kg.interner.lookup(self.r.relation_type))?;
        st.end()
    }
}

/// Edge-following direction for neighborhood queries.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    /// Follow `from -> to` (outgoing edges).
    Out,
    /// Follow `to -> from` (incoming edges).
    In,
    /// Follow edges regardless of orientation.
    Both,
}

impl Direction {
    /// Parse a direction string; anything other than `"out"`/`"in"` is `Both`.
    pub fn parse(s: Option<&str>) -> Self {
        match s {
            Some("out") => Direction::Out,
            Some("in") => Direction::In,
            _ => Direction::Both,
        }
    }
}

/// Escape a string for embedding inside a Mermaid/DOT quoted label.
fn sanitize_label(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push('\''),
            '\n' | '\r' => out.push(' '),
            _ => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// ShardedNameTable – open-addressing hash map split into N independent shards.
//
// Each shard uses **ctrl-byte bucket** approach: a 1-byte metadata array
// stores the 7-bit hash stamp (h2) for each slot, with `0xFF` = EMPTY.
// On probe, the first memory access is a single byte (ctrl). The full key
// (StrId) is only compared when the stamp matches — ~127/128 of probe steps
// touch nothing but the ctrl byte.  See also SwissTable / hashbrown.
// ---------------------------------------------------------------------------
const EMPTY_SLOT: u8 = 0xFF;

#[inline(always)]
const fn h2(hash: u64) -> u8 {
    (hash & 0x7F) as u8
}

#[inline(always)]
const fn h1(hash: u64, mask: usize) -> usize {
    ((hash >> 7) as usize) & mask
}

#[derive(Clone)]
struct NameTableShard {
    ctrl: Vec<u8>,      // 0xFF = empty; 0x00-0x7F = h2 stamp (bit 7 always clear)
    names: Vec<StrId>,
    slots: Vec<u32>,
    mask: usize,
    count: usize,
}

impl NameTableShard {
    fn new(capacity: usize) -> Self {
        let cap = capacity.next_power_of_two().max(16);
        Self {
            ctrl: vec![EMPTY_SLOT; cap],
            names: vec![StrId::EMPTY; cap],
            slots: vec![u32::MAX; cap],
            mask: cap - 1,
            count: 0,
        }
    }

    #[inline(always)]
    fn lookup(&self, hash: u64, name: StrId) -> Option<u32> {
        let stamp = h2(hash);
        let mask = self.mask;
        let mut idx = h1(hash, mask);
        let ctrl = self.ctrl.as_ptr();
        let names = self.names.as_ptr();
        let slots = self.slots.as_ptr();
        let len = self.ctrl.len();

        for _ in 0..len {
            // Prefetch the ctrl byte 4 slots ahead — overlaps memory latency.
            let prefetch_idx = idx.wrapping_add(4) & mask;
            unsafe { prefetch_addr(ctrl.add(prefetch_idx)) };

            // SAFETY: idx always < len because of &mask on each iteration.
            unsafe {
                let c = *ctrl.add(idx);
                // Bit 7 set → EMPTY → key not present.
                if c & 0x80 != 0 {
                    return None;
                }
                // Stamp match → compare full key (rare: ~1/128 probes).
                if c == stamp && *names.add(idx) == name {
                    return Some(*slots.add(idx));
                }
            }
            idx = (idx + 1) & mask;
        }
        None
    }

    fn insert(&mut self, interner: &StringInterner, hash: u64, name: StrId, slot: u32) {
        if self.count * 4 > self.ctrl.len() * 3 {
            self.grow(interner);
        }
        let stamp = h2(hash);
        let mask = self.mask;
        let mut idx = h1(hash, mask);
        loop {
            // SAFETY: idx & mask always < len for power-of-two capacity.
            unsafe {
                if *self.ctrl.get_unchecked(idx) & 0x80 != 0 {
                    *self.ctrl.get_unchecked_mut(idx) = stamp;
                    *self.names.get_unchecked_mut(idx) = name;
                    *self.slots.get_unchecked_mut(idx) = slot;
                    self.count += 1;
                    return;
                }
            }
            idx = (idx + 1) & mask;
        }
    }

    fn remove(&mut self, interner: &StringInterner, hash: u64, name: StrId) {
        let stamp = h2(hash);
        let mask = self.mask;
        let mut idx = h1(hash, mask);
        let len = self.ctrl.len();
        for _ in 0..len {
            if self.ctrl[idx] & 0x80 != 0 {
                return;
            }
            if self.ctrl[idx] == stamp && self.names[idx] == name {
                // Found — remove with shift-back to preserve probe chains.
                self.ctrl[idx] = EMPTY_SLOT;
                self.names[idx] = StrId::EMPTY;
                self.slots[idx] = u32::MAX;
                self.count -= 1;

                let mut next = (idx + 1) & mask;
                while self.ctrl[next] & 0x80 == 0 {
                    let nn = self.names[next];
                    let ns = self.slots[next];
                    // Hash is no longer stored (M4) — recompute it from the
                    // interned name to find the entry's ideal bucket.
                    let nh = interner.get_hash(nn);
                    self.ctrl[next] = EMPTY_SLOT;
                    self.names[next] = StrId::EMPTY;
                    self.slots[next] = u32::MAX;
                    self.count -= 1;

                    // Re-insert at its ideal bucket.
                    let nstamp = h2(nh);
                    let mut re_idx = h1(nh, mask);
                    while self.ctrl[re_idx] & 0x80 == 0 {
                        re_idx = (re_idx + 1) & mask;
                    }
                    self.ctrl[re_idx] = nstamp;
                    self.names[re_idx] = nn;
                    self.slots[re_idx] = ns;
                    self.count += 1;

                    next = (next + 1) & mask;
                }
                return;
            }
            idx = (idx + 1) & mask;
        }
    }

    fn grow(&mut self, interner: &StringInterner) {
        let new_cap = self.ctrl.len() * 2;
        let new_mask = new_cap - 1;
        let mut new_ctrl = vec![EMPTY_SLOT; new_cap];
        let mut new_names = vec![StrId::EMPTY; new_cap];
        let mut new_slots = vec![u32::MAX; new_cap];

        for i in 0..self.ctrl.len() {
            if self.ctrl[i] & 0x80 == 0 {
                // Recompute the hash from the interned name (M4: not stored).
                let name = self.names[i];
                let hash = interner.get_hash(name);
                let stamp = h2(hash);
                let mut idx = h1(hash, new_mask);
                while new_ctrl[idx] & 0x80 == 0 {
                    idx = (idx + 1) & new_mask;
                }
                new_ctrl[idx] = stamp;
                new_names[idx] = name;
                new_slots[idx] = self.slots[i];
            }
        }

        self.ctrl = new_ctrl;
        self.names = new_names;
        self.slots = new_slots;
        self.mask = new_mask;
    }
}

#[derive(Clone)]
struct ShardedNameTable {
    shards: [NameTableShard; NAME_TABLE_SHARDS],
}

impl ShardedNameTable {
    fn new(capacity_per_shard: usize) -> Self {
        Self {
            shards: [
                NameTableShard::new(capacity_per_shard),
                NameTableShard::new(capacity_per_shard),
                NameTableShard::new(capacity_per_shard),
                NameTableShard::new(capacity_per_shard),
            ],
        }
    }

    #[inline(always)]
    const fn shard(hash: u64) -> usize {
        (hash as usize) & (NAME_TABLE_SHARDS - 1)
    }

    #[inline(always)]
    fn lookup(&self, hash: u64, name: StrId) -> Option<u32> {
        self.shards[Self::shard(hash)].lookup(hash, name)
    }

    #[inline(always)]
    fn insert(&mut self, interner: &StringInterner, hash: u64, name: StrId, slot: u32) {
        self.shards[Self::shard(hash)].insert(interner, hash, name, slot);
    }

    #[inline(always)]
    fn remove(&mut self, interner: &StringInterner, hash: u64, name: StrId) {
        self.shards[Self::shard(hash)].remove(interner, hash, name);
    }
}

// ---------------------------------------------------------------------------
// KnowledgeGraph – the central type.
// ---------------------------------------------------------------------------
pub struct KnowledgeGraph {
    interner: StringInterner,
    entity_slots: Vec<Option<StoredEntity>>,
    /// Tombstoned slot indices available for reuse on the next create (M2),
    /// so create/delete churn doesn't grow `entity_slots` without bound.
    free_slots: Vec<u32>,
    name_table: ShardedNameTable,
    relations: Vec<StoredRelation>,
    /// Incremental adjacency index: StrId → outgoing (to, type) pairs.
    /// Updated on every create_relations/delete_relations/delete_entities.
    /// Traversals use this instead of rebuilding from scratch (item 3 in plan).
    adjacency: AHashMap<StrId, Vec<(StrId, StrId)>>,
    search: SearchIndex,
    store: BinaryStore,
}

// ---------------------------------------------------------------------------
// ReadSnapshot – wait-free, lock-free frozen view of the graph for readers.
// Created by KnowledgeGraph::snapshot() after each write transaction.
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct ReadSnapshot {
    pub(crate) interner: StringInterner,
    pub(crate) entity_slots: Arc<[Option<StoredEntity>]>,
    #[allow(dead_code)]
    free_slots: Vec<u32>,
    name_table: ShardedNameTable,
    pub(crate) relations: Arc<[StoredRelation]>,
    adjacency: AHashMap<StrId, Vec<(StrId, StrId)>>,
    search: SearchIndex,
}

// Fast JSON string escaper — writes escaped string into the buffer without
// allocating an intermediate string for the escape pass.
pub(crate) fn push_json_str(buf: &mut String, s: &str) {
    buf.push('"');
    for c in s.chars() {
        match c {
            '"' => buf.push_str("\\\""),
            '\\' => buf.push_str("\\\\"),
            '\n' => buf.push_str("\\n"),
            '\r' => buf.push_str("\\r"),
            '\t' => buf.push_str("\\t"),
            c if c.is_control() => {
                use std::fmt::Write;
                write!(buf, "\\u{:04x}", c as u32).unwrap();
            }
            c => buf.push(c),
        }
    }
    buf.push('"');
}

/// A borrowing view over a selection of entities/relations from a ReadSnapshot.
/// Serializes without intermediate owned String allocations.
pub struct ReadGraphView<'a> {
    snap: &'a ReadSnapshot,
    entities: Vec<&'a StoredEntity>,
    relations: Vec<&'a StoredRelation>,
}

impl Serialize for ReadGraphView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut st = s.serialize_struct("KnowledgeGraphOut", 2)?;
        st.serialize_field("entities", &ReadEntityListRef { snap: self.snap, items: &self.entities })?;
        st.serialize_field("relations", &ReadRelationListRef { snap: self.snap, items: &self.relations })?;
        st.end()
    }
}

struct ReadEntityListRef<'a> { snap: &'a ReadSnapshot, items: &'a [&'a StoredEntity] }
impl Serialize for ReadEntityListRef<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.items.len()))?;
        for &e in self.items {
            seq.serialize_element(&ReadEntityRef { snap: self.snap, e })?;
        }
        seq.end()
    }
}

struct ReadRelationListRef<'a> { snap: &'a ReadSnapshot, items: &'a [&'a StoredRelation] }
impl Serialize for ReadRelationListRef<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.items.len()))?;
        for &r in self.items {
            seq.serialize_element(&ReadRelationRef { snap: self.snap, r })?;
        }
        seq.end()
    }
}

struct ReadEntityRef<'a> { snap: &'a ReadSnapshot, e: &'a StoredEntity }
impl Serialize for ReadEntityRef<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut st = s.serialize_struct("Entity", 3)?;
        st.serialize_field("name", self.snap.interner.lookup(self.e.name))?;
        st.serialize_field("entityType", self.snap.interner.lookup(self.e.entity_type))?;
        st.serialize_field("observations", &ReadObsRef { snap: self.snap, obs: &self.e.observations })?;
        st.end()
    }
}

struct ReadObsRef<'a> { snap: &'a ReadSnapshot, obs: &'a [StrId] }
impl Serialize for ReadObsRef<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.obs.len()))?;
        for &o in self.obs {
            seq.serialize_element(self.snap.interner.lookup(o))?;
        }
        seq.end()
    }
}

struct ReadRelationRef<'a> { snap: &'a ReadSnapshot, r: &'a StoredRelation }
impl Serialize for ReadRelationRef<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut st = s.serialize_struct("Relation", 3)?;
        st.serialize_field("from", self.snap.interner.lookup(self.r.from))?;
        st.serialize_field("to", self.snap.interner.lookup(self.r.to))?;
        st.serialize_field("relationType", self.snap.interner.lookup(self.r.relation_type))?;
        st.end()
    }
}

// --- ReadSnapshot helpers (mirrors KnowledgeGraph helpers) ---
impl ReadSnapshot {

    /// Serialize the full graph directly to a JSON string, avoiding intermediate
    /// owned `Entity`/`Relation` allocations. This is the fast path for handlers.
    pub fn read_graph_json(&self) -> String {
        // Rough capacity: 40K entities × ~60 B + 120K relations × ~55 B = ~9 MB
        let cap = self.entity_slots.len() * 64 + self.relations.len() * 60 + 128;
        let mut buf = String::with_capacity(cap);

        // entities array
        buf.push_str(r#"{"entities":["#);
        let mut first = true;
        for slot in self.entity_slots.iter() {
            let Some(e) = slot.as_ref().filter(|e| e.is_live()) else { continue };
            if first { first = false } else { buf.push(',') }
            buf.push('{');
            // name
            buf.push_str(r#""name":"#);
            push_json_str(&mut buf, self.interner.lookup(e.name));
            buf.push(',');
            // entityType
            buf.push_str(r#""entityType":"#);
            push_json_str(&mut buf, self.interner.lookup(e.entity_type));
            buf.push(',');
            // observations
            buf.push_str(r#""observations":["#);
            for (oi, o) in e.observations.iter().enumerate() {
                if oi > 0 { buf.push(',') }
                push_json_str(&mut buf, self.interner.lookup(*o));
            }
            buf.push_str("]}");
        }

        // relations array
        buf.push_str(r#"],"relations":["#);
        first = true;
        for r in self.relations.iter() {
            if first { first = false } else { buf.push(',') }
            buf.push('{');
            buf.push_str(r#""from":"#);
            push_json_str(&mut buf, self.interner.lookup(r.from));
            buf.push(',');
            buf.push_str(r#""to":"#);
            push_json_str(&mut buf, self.interner.lookup(r.to));
            buf.push(',');
            buf.push_str(r#""relationType":"#);
            push_json_str(&mut buf, self.interner.lookup(r.relation_type));
            buf.push('}');
        }
        buf.push_str("]}");

        buf
    }

    /// Borrowing view of the full graph, suitable for serde-serialization without
    /// intermediate owned String allocations.
    pub fn read_graph_view(&self) -> ReadGraphView<'_> {
        let entities: Vec<&StoredEntity> = self
            .entity_slots
            .iter()
            .filter_map(|s| s.as_ref().filter(|e| e.is_live()))
            .collect();
        let relations: Vec<&StoredRelation> = self.relations.iter().collect();
        ReadGraphView { snap: self, entities, relations }
    }

    fn lookup_live_slot(&self, name: &str) -> Option<u32> {
        let name_id = self.interner.get_optional(name)?;
        let hash = self.interner.get_hash(name_id);
        let slot = self.name_table.lookup(hash, name_id)?;
        self.entity_slots
            .get(slot as usize)
            .and_then(|s| s.as_ref())
            .filter(|e| e.is_live())?;
        Some(slot)
    }

    fn entity_by_name_id(&self, name_id: StrId) -> Option<Entity> {
        let hash = self.interner.get_hash(name_id);
        let slot = self.name_table.lookup(hash, name_id)?;
        let e = self.entity_slots.get(slot as usize)?.as_ref()?;
        Some(self.entity_to_output(e))
    }

    pub(crate) fn entity_to_output(&self, e: &StoredEntity) -> Entity {
        Entity {
            name: self.interner.lookup(e.name).to_string(),
            entity_type: self.interner.lookup(e.entity_type).to_string(),
            observations: e
                .observations
                .iter()
                .map(|o| self.interner.lookup(*o).to_string())
                .collect(),
        }
    }

    pub(crate) fn relation_to_output(&self, r: &StoredRelation) -> Relation {
        Relation {
            from: self.interner.lookup(r.from).to_string(),
            to: self.interner.lookup(r.to).to_string(),
            relation_type: self.interner.lookup(r.relation_type).to_string(),
        }
    }

    /// Open named entities + incident relations.
    pub fn open_nodes(&self, names: &[String]) -> KnowledgeGraphOut {
        let name_ids: std::collections::HashSet<StrId> = names
            .iter()
            .filter_map(|n| self.interner.get_optional(n))
            .collect();
        let entities: Vec<Entity> = self
            .entity_slots
            .iter()
            .filter_map(|s| {
                let e = s.as_ref()?;
                if e.is_live() && name_ids.contains(&e.name) {
                    Some(self.entity_to_output(e))
                } else {
                    None
                }
            })
            .collect();
        let matched: std::collections::HashSet<StrId> = entities.iter()
            .filter_map(|e| self.interner.get_optional(&e.name))
            .collect();
        let relations: Vec<Relation> = self
            .relations
            .iter()
            .filter(|r| matched.contains(&r.from) || matched.contains(&r.to))
            .map(|r| self.relation_to_output(r))
            .collect();
        KnowledgeGraphOut { entities, relations }
    }

    /// Full graph read. Returns all entities and relations.
    pub fn read_graph(&self) -> KnowledgeGraphOut {
        let entities: Vec<Entity> = self
            .entity_slots
            .iter()
            .filter_map(|s| s.as_ref().filter(|e| e.is_live()))
            .map(|e| self.entity_to_output(e))
            .collect();
        let relations: Vec<Relation> = self
            .relations
            .iter()
            .map(|r| self.relation_to_output(r))
            .collect();
        KnowledgeGraphOut { entities, relations }
    }

    /// Get a single entity by name.
    pub fn get_entity(&self, name: &str) -> Option<Entity> {
        self.lookup_live_slot(name)?;
        let name_id = self.interner.get_optional(name)?;
        self.entity_by_name_id(name_id)
    }

    /// Neighborhood expansion (same logic as KnowledgeGraph::neighbors but read-only).
    pub fn neighbors(
        &self,
        name: &str,
        direction: Direction,
        rtype: Option<&str>,
        depth: u32,
    ) -> Result<KnowledgeGraphOut> {
        self.lookup_live_slot(name)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{name}' not found")))?;
        let start = self.interner.get_optional(name).unwrap();

        let rtype_id = match rtype {
            Some(r) => match self.interner.get_optional(r) {
                Some(id) => Some(id),
                None => {
                    let entities = self.entity_by_name_id(start).into_iter().collect();
                    return Ok(KnowledgeGraphOut { entities, relations: Vec::new() });
                }
            },
            None => None,
        };

        let mut visited: AHashSet<StrId> = AHashSet::new();
        visited.insert(start);

        let type_ok = |r: &StoredRelation, rt: Option<StrId>| rt.is_none_or(|rt_id| r.relation_type == rt_id);

        if depth == 1 {
            for r in self.relations.iter().filter(|r| type_ok(r, rtype_id)) {
                match direction {
                    Direction::Out => {
                        if r.from == start { visited.insert(r.to); }
                    }
                    Direction::In => {
                        if r.to == start { visited.insert(r.from); }
                    }
                    Direction::Both => {
                        if r.from == start { visited.insert(r.to); }
                        else if r.to == start { visited.insert(r.from); }
                    }
                }
            }
        } else if depth >= 2 {
            let mut adj: AHashMap<StrId, Vec<StrId>> = AHashMap::new();
            match direction {
                Direction::Both => {
                    for (&node, edges) in &self.adjacency {
                        for &(nb, rt) in edges {
                            if rtype_id.is_none_or(|rt_id| rt == rt_id) {
                                adj.entry(node).or_default().push(nb);
                            }
                        }
                    }
                }
                Direction::Out | Direction::In => {
                    for r in self.relations.iter().filter(|r| type_ok(r, rtype_id)) {
                        match direction {
                            Direction::Out => adj.entry(r.from).or_default().push(r.to),
                            Direction::In => adj.entry(r.to).or_default().push(r.from),
                            _ => unreachable!(),
                        }
                    }
                }
            }
            let mut queue: VecDeque<(StrId, u32)> = VecDeque::new();
            queue.push_back((start, 0));
            while let Some((node, d)) = queue.pop_front() {
                if d >= depth { continue; }
                if let Some(nbrs) = adj.get(&node) {
                    for &nb in nbrs {
                        if visited.insert(nb) {
                            queue.push_back((nb, d + 1));
                        }
                    }
                }
            }
        }

        let mut entities = Vec::with_capacity(visited.len());
        for &nid in &visited {
            if let Some(e) = self.entity_by_name_id(nid) {
                entities.push(e);
            }
        }
        let relations: Vec<Relation> = self
            .relations
            .iter()
            .filter(|r| type_ok(r, rtype_id) && visited.contains(&r.from) && visited.contains(&r.to))
            .map(|r| self.relation_to_output(r))
            .collect();
        Ok(KnowledgeGraphOut { entities, relations })
    }

    /// Describe an entity: entity details, incident relations, neighbors, degree.
    pub fn describe_entity(&self, name: &str) -> Result<serde_json::Value> {
        let name_id = self
            .interner
            .get_optional(name)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{name}' not found")))?;
        let entity = self
            .entity_by_name_id(name_id)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{name}' not found")))?;

        let mut incident: Vec<Relation> = Vec::new();
        let mut neighbor_seen: AHashSet<StrId> = AHashSet::new();
        let mut neighbors: Vec<&str> = Vec::new();
        for r in self.relations.iter() {
            if r.from == name_id || r.to == name_id {
                incident.push(self.relation_to_output(r));
                let other = if r.from == name_id { r.to } else { r.from };
                if other != name_id && neighbor_seen.insert(other) {
                    neighbors.push(self.interner.lookup(other));
                }
            }
        }

        Ok(serde_json::json!({
            "entity": entity,
            "relations": incident,
            "neighbors": neighbors,
            "degree": incident.len(),
        }))
    }

    /// Find the shortest path between two entities.
    pub fn find_path(&self, from: &str, to: &str) -> Result<Vec<String>> {
        let from_id = self
            .interner
            .get_optional(from)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{from}' not found")))?;
        let to_id = self
            .interner
            .get_optional(to)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{to}' not found")))?;
        if self.lookup_live_slot(from).is_none() {
            return Err(MCSError::InvalidParams(format!("Entity '{from}' not found")));
        }
        if self.lookup_live_slot(to).is_none() {
            return Err(MCSError::InvalidParams(format!("Entity '{to}' not found")));
        }

        // BFS over incremental adjacency index.
        let mut visited: AHashSet<StrId> = AHashSet::new();
        let mut parent: AHashMap<StrId, StrId> = AHashMap::new();
        let mut queue: VecDeque<StrId> = VecDeque::new();

        visited.insert(from_id);
        queue.push_back(from_id);

        while let Some(current) = queue.pop_front() {
            if current == to_id { break; }
            if let Some(neighbors) = self.adjacency.get(&current) {
                for &(neighbor, _) in neighbors {
                    if visited.insert(neighbor) {
                        parent.insert(neighbor, current);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if !visited.contains(&to_id) {
            return Err(MCSError::MemoryError(format!(
                "No path found between '{from}' and '{to}'"
            )));
        }

        let mut path = Vec::new();
        let mut cur = to_id;
        path.push(self.interner.lookup(cur).to_string());
        while let Some(&p) = parent.get(&cur) {
            path.push(self.interner.lookup(p).to_string());
            cur = p;
        }
        path.reverse();
        Ok(path)
    }

    /// Extract a subgraph around the given entities.
    pub fn extract_subgraph(&self, names: &[String], depth: u32) -> Result<KnowledgeGraphOut> {
        if names.is_empty() {
            return Ok(KnowledgeGraphOut { entities: Vec::new(), relations: Vec::new() });
        }
        let mut visited: AHashSet<StrId> = AHashSet::new();
        let mut queue: VecDeque<(StrId, u32)> = VecDeque::new();
        for name in names {
            if let Some(id) = self.interner.get_optional(name)
                && visited.insert(id)
            {
                queue.push_back((id, 0));
            }
        }
        let mut adj: AHashMap<StrId, Vec<StrId>> = AHashMap::new();
        for (&node, edges) in &self.adjacency {
            let nbrs: Vec<StrId> = edges.iter().map(|(to, _)| *to).collect();
            adj.insert(node, nbrs);
        }
        while let Some((node, d)) = queue.pop_front() {
            if d >= depth { continue; }
            if let Some(nbrs) = adj.get(&node) {
                for &nb in nbrs {
                    if visited.insert(nb) {
                        queue.push_back((nb, d + 1));
                    }
                }
            }
        }
        let mut entities: Vec<Entity> = Vec::with_capacity(visited.len());
        for &nid in &visited {
            if let Some(e) = self.entity_by_name_id(nid) {
                entities.push(e);
            }
        }
        let relations: Vec<Relation> = self
            .relations
            .iter()
            .filter(|r| visited.contains(&r.from) && visited.contains(&r.to))
            .map(|r| self.relation_to_output(r))
            .collect();
        Ok(KnowledgeGraphOut { entities, relations })
    }

    /// Batch get entities.
    pub fn batch_get_entities(&self, names: &[String]) -> Vec<Option<Entity>> {
        names.iter().map(|n| self.get_entity(n)).collect()
    }

    /// Graph statistics.
    pub fn graph_stats(&self) -> serde_json::Value {
        let entity_count = self
            .entity_slots
            .iter()
            .filter(|s| s.as_ref().is_some_and(|e| e.is_live()))
            .count();
        let relation_count = self.relations.len();
        let type_counts = self.entity_type_counts();
        let relation_type_counts = self.relation_type_counts();
        serde_json::json!({
            "entities": entity_count,
            "relations": relation_count,
            "entityTypes": type_counts,
            "relationTypes": relation_type_counts,
        })
    }

    /// Search relations by from/to/type filters.
    pub fn search_relations(&self, from: Option<&str>, to: Option<&str>, rtype: Option<&str>) -> Vec<Relation> {
        let from_id = from.and_then(|n| self.interner.get_optional(n));
        let to_id = to.and_then(|n| self.interner.get_optional(n));
        let rtype_id = rtype.and_then(|n| self.interner.get_optional(n));
        self.relations
            .iter()
            .filter(|r| {
                from_id.is_none_or(|id| r.from == id)
                    && to_id.is_none_or(|id| r.to == id)
                    && rtype_id.is_none_or(|id| r.relation_type == id)
            })
            .map(|r| self.relation_to_output(r))
            .collect()
    }

    /// Count entities per type.
    pub fn entity_type_counts(&self) -> Vec<(String, usize)> {
        let mut counts: AHashMap<StrId, usize> = AHashMap::new();
        for slot in self.entity_slots.iter() {
            if let Some(e) = slot.as_ref().filter(|e| e.is_live()) {
                *counts.entry(e.entity_type).or_default() += 1;
            }
        }
        let mut result: Vec<(String, usize)> = counts
            .into_iter()
            .map(|(id, c)| (self.interner.lookup(id).to_string(), c))
            .collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    /// Count relations per type.
    pub fn relation_type_counts(&self) -> Vec<(String, usize)> {
        let mut counts: AHashMap<StrId, usize> = AHashMap::new();
        for r in self.relations.iter() {
            *counts.entry(r.relation_type).or_default() += 1;
        }
        let mut result: Vec<(String, usize)> = counts
            .into_iter()
            .map(|(id, c)| (self.interner.lookup(id).to_string(), c))
            .collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    /// Export the graph in one of: json, mermaid, dot.
    pub fn export(&self, format: &str) -> Result<String> {
        match format {
            "json" => serde_json::to_string(&self.read_graph()).map_err(MCSError::JsonError),
            "mermaid" => Ok(self.export_mermaid()),
            "dot" => Ok(self.export_dot()),
            other => Err(MCSError::InvalidParams(format!(
                "Unknown export format '{other}' (expected json|mermaid|dot)"
            ))),
        }
    }

    fn export_mermaid(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str("graph LR\n");
        for r in self.relations.iter() {
            let from = sanitize_label(self.interner.lookup(r.from));
            let to = sanitize_label(self.interner.lookup(r.to));
            let rt = sanitize_label(self.interner.lookup(r.relation_type));
            out.push_str(&format!("    {} -- \"{}\" --> {}\n", from, rt, to));
        }
        out
    }

    fn export_dot(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str("digraph KG {\n");
        out.push_str("    rankdir=LR;\n");
        for slot in self.entity_slots.iter() {
            if let Some(e) = slot.as_ref().filter(|e| e.is_live()) {
                let name = sanitize_label(self.interner.lookup(e.name));
                let etype = sanitize_label(self.interner.lookup(e.entity_type));
                out.push_str(&format!("    \"{}\" [label=\"{}\n({})\"];\n", name, name, etype));
            }
        }
        for r in self.relations.iter() {
            let from = sanitize_label(self.interner.lookup(r.from));
            let to = sanitize_label(self.interner.lookup(r.to));
            let rt = sanitize_label(self.interner.lookup(r.relation_type));
            out.push_str(&format!("    \"{}\" -> \"{}\" [label=\"{}\"];\n", from, to, rt));
        }
        out.push_str("}\n");
        out
    }

    /// Find all paths between two entities.
    pub fn find_all_paths(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
        max_paths: usize,
    ) -> Result<Vec<Vec<String>>> {
        let from_id = self
            .interner
            .get_optional(from)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{from}' not found")))?;
        let to_id = self
            .interner
            .get_optional(to)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{to}' not found")))?;
        if self.lookup_live_slot(from).is_none() {
            return Err(MCSError::InvalidParams(format!("Entity '{from}' not found")));
        }
        if self.lookup_live_slot(to).is_none() {
            return Err(MCSError::InvalidParams(format!("Entity '{to}' not found")));
        }
        if from_id == to_id {
            return Ok(vec![vec![from.to_string()]]);
        }
        let mut adj: AHashMap<StrId, Vec<StrId>> = AHashMap::with_capacity(self.adjacency.len());
        for (&node, edges) in &self.adjacency {
            let nbrs: Vec<StrId> = edges.iter().map(|(to, _)| *to).collect();
            adj.insert(node, nbrs);
        }
        let mut all_paths: Vec<Vec<StrId>> = Vec::new();
        let mut current_path = Vec::new();
        let mut visited: AHashSet<StrId> = AHashSet::new();
        visited.insert(from_id);
        current_path.push(from_id);
        Self::dfs_all_paths(
            &adj,
            from_id,
            to_id,
            max_depth,
            max_paths,
            &mut visited,
            &mut current_path,
            &mut all_paths,
        );
        if all_paths.is_empty() {
            return Err(MCSError::MemoryError(format!(
                "No path found between '{from}' and '{to}'"
            )));
        }
        let result: Vec<Vec<String>> = all_paths
            .into_iter()
            .map(|path| {
                path.into_iter()
                    .map(|id| self.interner.lookup(id).to_string())
                    .collect()
            })
            .collect();
        Ok(result)
    }

    fn dfs_all_paths(
        adj: &AHashMap<StrId, Vec<StrId>>,
        current: StrId,
        target: StrId,
        max_depth: usize,
        max_paths: usize,
        visited: &mut AHashSet<StrId>,
        current_path: &mut Vec<StrId>,
        all_paths: &mut Vec<Vec<StrId>>,
    ) {
        if all_paths.len() >= max_paths { return; }
        if current == target && current_path.len() > 1 {
            all_paths.push(current_path.clone());
            return;
        }
        if current_path.len() > max_depth { return; }
        if let Some(neighbors) = adj.get(&current) {
            for &nb in neighbors {
                if !visited.contains(&nb) {
                    visited.insert(nb);
                    current_path.push(nb);
                    Self::dfs_all_paths(adj, nb, target, max_depth, max_paths, visited, current_path, all_paths);
                    current_path.pop();
                    visited.remove(&nb);
                }
            }
        }
    }

    /// Search for entities whose name/type/observations contain `query`.
    pub fn search_entities(&self, query: &str) -> Result<Vec<Entity>> {
        let token = query.to_lowercase();
        let matching = self.search.search(&token, &self.interner);
        Ok(matching
            .iter()
            .filter_map(|idx| {
                self.entity_slots
                    .get(*idx as usize)?
                    .as_ref()
                    .filter(|e| e.is_live())
                    .map(|e| self.entity_to_output(e))
            })
            .collect())
    }
}

impl KnowledgeGraph {
    pub fn new(path: &Path) -> std::io::Result<Self> {
        let store = BinaryStore::new(path)?;

        // Replay into local collections, then assign into self — no raw pointers needed (X3).
        let mut interner = StringInterner::with_capacity(65536, 1024);
        let mut entity_slots: Vec<Option<StoredEntity>> = Vec::with_capacity(256);
        let mut name_table = ShardedNameTable::new(64);
        let mut relations: Vec<StoredRelation> = Vec::with_capacity(64);
        let mut search = SearchIndex::new();

        // Transaction buffer: while `Some`, records are accumulated and only
        // applied on `TxnCommit`. An unclosed transaction at EOF is discarded,
        // which is what makes multi-record operations (e.g. `merge_entities`)
        // crash-atomic.
        let mut pending: Option<Vec<(RecordKind, Vec<u8>)>> = None;
        store.replay(|kind, data| {
            match kind {
                RecordKind::TxnBegin => pending = Some(Vec::new()),
                RecordKind::TxnCommit => {
                    if let Some(buffered) = pending.take() {
                        for (k, d) in &buffered {
                            Self::apply_record(
                                *k, d, &mut interner, &mut entity_slots, &mut search,
                                &mut name_table, &mut relations,
                            );
                        }
                    }
                }
                other => match pending.as_mut() {
                    Some(buffered) => buffered.push((other, data.to_vec())),
                    None => Self::apply_record(
                        other, data, &mut interner, &mut entity_slots, &mut search,
                        &mut name_table, &mut relations,
                    ),
                },
            }
        })?;

        // Slots tombstoned by deletes during replay are available for reuse (M2).
        let free_slots: Vec<u32> = entity_slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_none())
            .map(|(i, _)| i as u32)
            .collect();

        let mut adjacency: AHashMap<StrId, Vec<(StrId, StrId)>> = AHashMap::new();
        for rel in &relations {
            adjacency.entry(rel.from).or_default().push((rel.to, rel.relation_type));
            adjacency.entry(rel.to).or_default().push((rel.from, rel.relation_type));
        }

        Ok(Self {
            interner,
            entity_slots,
            free_slots,
            name_table,
            relations,
            adjacency,
            search,
            store,
        })
    }

    // -----------------------------------------------------------------------
    // Replay helpers (static to avoid borrow issues in the closure)
    // -----------------------------------------------------------------------

    /// Apply one already-decoded log record to the in-memory collections.
    /// Shared by direct replay and by transaction commit. `TxnBegin`/`TxnCommit`
    /// are handled by the caller and are no-ops here.
    #[allow(clippy::too_many_arguments)]
    fn apply_record(
        kind: RecordKind,
        data: &[u8],
        interner: &mut StringInterner,
        entity_slots: &mut Vec<Option<StoredEntity>>,
        search: &mut SearchIndex,
        name_table: &mut ShardedNameTable,
        relations: &mut Vec<StoredRelation>,
    ) {
        match kind {
            RecordKind::CreateEntity => {
                if let Some((name, etype, obs)) = store_enc::decode_create_entity(data) {
                    Self::replay_create_entity(
                        interner, entity_slots, search, name_table, name, etype, &obs,
                    );
                }
            }
            RecordKind::CreateRelation => {
                if let Some((from, to, rtype)) = store_enc::decode_create_relation(data) {
                    let from_id = interner.intern(from);
                    let to_id = interner.intern(to);
                    let type_id = interner.intern(rtype);
                    relations.push(StoredRelation {
                        from: from_id,
                        to: to_id,
                        relation_type: type_id,
                    });
                }
            }
            RecordKind::AddObservations => {
                if let Some((name, obs)) = store_enc::decode_add_observations(data) {
                    Self::replay_add_observations(
                        interner, entity_slots, search, name_table, name, &obs,
                    );
                }
            }
            RecordKind::DeleteEntity => {
                if let Some(name) = store_enc::decode_delete_entity(data) {
                    Self::replay_delete_entity(
                        interner, entity_slots, relations, search, name_table, name,
                    );
                }
            }
            RecordKind::DeleteObservations => {
                if let Some((name, obs)) = store_enc::decode_delete_observations(data) {
                    Self::replay_delete_observations(
                        interner, entity_slots, search, name_table, name, &obs,
                    );
                }
            }
            RecordKind::DeleteRelation => {
                if let Some((from, to, rtype)) = store_enc::decode_delete_relation(data) {
                    let from_id = interner.intern(from);
                    let to_id = interner.intern(to);
                    let type_id = interner.intern(rtype);
                    relations.retain(|r| {
                        !(r.from == from_id && r.to == to_id && r.relation_type == type_id)
                    });
                }
            }
            RecordKind::TxnBegin | RecordKind::TxnCommit => {}
        }
    }

    #[allow(clippy::ptr_arg)]
    fn replay_create_entity(
        interner: &mut StringInterner,
        entities: &mut Vec<Option<StoredEntity>>,
        search: &mut SearchIndex,
        name_table: &mut ShardedNameTable,
        name: &str,
        etype: &str,
        observations: &[&str],
    ) {
        let name_id = interner.intern(name);
        let type_id = interner.intern(etype);
        let obs_ids: Vec<StrId> = observations.iter().map(|o| interner.intern(o)).collect();
        let slot = entities.len() as u32;
        entities.push(Some(StoredEntity {
            state: ENTITY_SLOT_LIVE,
            name: name_id,
            entity_type: type_id,
            observations: obs_ids.clone(),
        }));
        let hash = interner.get_hash(name_id);
        name_table.insert(&*interner, hash, name_id, slot);
        search.index_entity(interner, slot, name_id, type_id, &obs_ids);
    }

    fn replay_add_observations(
        interner: &mut StringInterner,
        entities: &mut [Option<StoredEntity>],
        search: &mut SearchIndex,
        name_table: &mut ShardedNameTable,
        name: &str,
        observations: &[&str],
    ) {
        let name_id = interner.intern(name);
        let hash = interner.get_hash(name_id);
        if let Some(slot) = name_table.lookup(hash, name_id)
            && let Some(Some(entity)) = entities.get_mut(slot as usize)
        {
            for &o in observations {
                let oid = interner.intern(o);
                if !entity.observations.contains(&oid) {
                    entity.observations.push(oid);
                }
            }
            search.remove_entity(slot);
            search.index_entity(
                interner,
                slot,
                entity.name,
                entity.entity_type,
                &entity.observations,
            );
        }
    }

    fn replay_delete_entity(
        interner: &mut StringInterner,
        entities: &mut [Option<StoredEntity>],
        rels: &mut Vec<StoredRelation>,
        search: &mut SearchIndex,
        name_table: &mut ShardedNameTable,
        name: &str,
    ) {
        let name_id = interner.intern(name);
        let hash = interner.get_hash(name_id);
        if let Some(slot) = name_table.lookup(hash, name_id)
            && let Some(Some(_)) = entities.get(slot as usize)
        {
            entities[slot as usize] = None;
            search.remove_entity(slot);
            name_table.remove(&*interner, hash, name_id);
        }
        rels.retain(|r| r.from != name_id && r.to != name_id);
    }

    fn replay_delete_observations(
        interner: &mut StringInterner,
        entities: &mut [Option<StoredEntity>],
        search: &mut SearchIndex,
        name_table: &mut ShardedNameTable,
        name: &str,
        observations: &[&str],
    ) {
        let name_id = interner.intern(name);
        let hash = interner.get_hash(name_id);
        if let Some(slot) = name_table.lookup(hash, name_id)
            && let Some(Some(entity)) = entities.get_mut(slot as usize)
        {
            let remove_ids: Vec<StrId> = observations.iter().map(|o| interner.intern(o)).collect();
            entity.observations.retain(|o| !remove_ids.contains(o));
            search.remove_entity(slot);
            search.index_entity(
                interner,
                slot,
                entity.name,
                entity.entity_type,
                &entity.observations,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    pub const fn interner(&self) -> &StringInterner {
        &self.interner
    }

    /// Return a single entity by exact name match.
    pub fn get_entity(&self, name: &str) -> Option<Entity> {
        let name_id = self.interner.get_optional(name)?;
        let hash = self.interner.get_hash(name_id);
        let slot = self.name_table.lookup(hash, name_id)?;
        let stored = self.entity_slots.get(slot as usize)?.as_ref()?;
        if !stored.is_live() {
            return None;
        }
        Some(self.entity_to_output(stored))
    }

    /// Return aggregate statistics about the graph.
    pub fn graph_stats(&self) -> serde_json::Value {
        let live_entities = self
            .entity_slots
            .iter()
            .filter(|s| s.as_ref().is_some_and(|e| e.is_live()))
            .count();
        let total_relations = self.relations.len();
        let index_entries = self.search.len();
        let total_obs: usize = self
            .entity_slots
            .iter()
            .filter_map(|s| s.as_ref())
            .filter(|e| e.is_live())
            .map(|e| e.observations.len())
            .sum();

        serde_json::json!({
            "entities": live_entities,
            "relations": total_relations,
            "totalObservations": total_obs,
            "searchIndexEntries": index_entries,
            "internedStrings": self.interner.len(),
            "internedBytes": self.interner.total_bytes(),
        })
    }

    /// Search relations by optional filters: `from`, `to`, `relationType`.
    /// Any filter that is absent matches everything. A filter value that does
    /// not exist in the graph returns empty results.
    pub fn search_relations(&self, from: Option<&str>, to: Option<&str>, rtype: Option<&str>) -> Vec<Relation> {
        let from_id = match from {
            Some(f) => match self.interner.get_optional(f) {
                Some(id) => Some(id),
                None => return Vec::new(),
            },
            None => None,
        };
        let to_id = match to {
            Some(t) => match self.interner.get_optional(t) {
                Some(id) => Some(id),
                None => return Vec::new(),
            },
            None => None,
        };
        let rtype_id = match rtype {
            Some(r) => match self.interner.get_optional(r) {
                Some(id) => Some(id),
                None => return Vec::new(),
            },
            None => None,
        };

        self.relations
            .iter()
            .filter(|r| {
                from_id.is_none_or(|f| r.from == f)
                    && to_id.is_none_or(|t| r.to == t)
                    && rtype_id.is_none_or(|rt| r.relation_type == rt)
            })
            .map(|r| Relation {
                from: self.interner.lookup(r.from).to_string(),
                to: self.interner.lookup(r.to).to_string(),
                relation_type: self.interner.lookup(r.relation_type).to_string(),
            })
            .collect()
    }

    /// BFS shortest-path between two entity names. Returns the sequence of
    /// entity names along the path (inclusive of both endpoints).
    pub fn find_path(&self, from: &str, to: &str) -> Result<Vec<String>> {
        let from_id = self.interner.get_optional(from)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{from}' not found")))?;
        let to_id = self.interner.get_optional(to)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{to}' not found")))?;
        let hash_from = self.interner.get_hash(from_id);
        let hash_to = self.interner.get_hash(to_id);

        if self.name_table.lookup(hash_from, from_id).is_none() {
            return Err(MCSError::InvalidParams(format!("Entity '{from}' not found")));
        }
        if self.name_table.lookup(hash_to, to_id).is_none() {
            return Err(MCSError::InvalidParams(format!("Entity '{to}' not found")));
        }
        if from_id == to_id {
            return Ok(vec![from.to_string()]);
        }

        // Use incremental adjacency index — O(degree) per hop, no rebuild.
        let mut visited: AHashSet<StrId> = AHashSet::new();
        let mut parent: AHashMap<StrId, StrId> = AHashMap::new();
        let mut queue: VecDeque<StrId> = VecDeque::new();

        visited.insert(from_id);
        queue.push_back(from_id);

        while let Some(current) = queue.pop_front() {
            if current == to_id {
                break;
            }

            if let Some(neighbors) = self.adjacency.get(&current) {
                for &(neighbor, _) in neighbors {
                    if visited.insert(neighbor) {
                        parent.insert(neighbor, current);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if !parent.contains_key(&to_id) && from_id != to_id {
            return Err(MCSError::MemoryError(format!(
                "No path found between '{from}' and '{to}'"
            )));
        }

        // Reconstruct path
        let mut path: Vec<String> = Vec::new();
        let mut cur = to_id;
        loop {
            path.push(self.interner.lookup(cur).to_string());
            if cur == from_id {
                break;
            }
            cur = *parent.get(&cur).ok_or_else(|| {
                MCSError::MemoryError("Path reconstruction failed".into())
            })?;
        }
        path.reverse();
        Ok(path)
    }

    /// Rewrite the binary log from the current in-memory state.
    /// After compaction the log contains only the minimal set of records
    /// needed to reconstruct the graph (all creates, no deletes).
    /// Crash-safe: writes to a temp file, then atomically renames (C3).
    pub fn compact(&mut self) -> Result<()> {
        // 1. Collect current state as create-records
        let mut create_entities: Vec<Entity> = Vec::new();
        let mut create_relations: Vec<Relation> = Vec::new();

        for slot in &self.entity_slots {
            if let Some(stored) = slot.as_ref().filter(|e| e.is_live()) {
                create_entities.push(self.entity_to_output(stored));
            }
        }
        for rel in &self.relations {
            create_relations.push(Relation {
                from: self.interner.lookup(rel.from).to_string(),
                to: self.interner.lookup(rel.to).to_string(),
                relation_type: self.interner.lookup(rel.relation_type).to_string(),
            });
        }

        // 2. Write to a temp file first.
        //    Remove any stale temp left by a previously-interrupted compact:
        //    `BinaryStore::new` opens in append mode and only writes the MAGIC
        //    header when the file does not already exist, so appending to a
        //    leftover temp would produce a duplicated, header-corrupted log
        //    once renamed over the real one (C1).
        let tmp_path = self.store.path().with_extension("tmp");
        if let Err(e) = std::fs::remove_file(&tmp_path)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            return Err(MCSError::IoError(e));
        }
        let mut tmp_store = BinaryStore::new(&tmp_path).map_err(MCSError::IoError)?;
        for entity in &create_entities {
            let mut buf = Vec::new();
            store_enc::encode_create_entity(&mut buf, &entity.name, &entity.entity_type, &entity.observations)
                .map_err(MCSError::IoError)?;
            tmp_store.write_record(RecordKind::CreateEntity, &buf).map_err(MCSError::IoError)?;
        }
        for relation in &create_relations {
            let mut buf = Vec::new();
            store_enc::encode_create_relation(&mut buf, &relation.from, &relation.to, &relation.relation_type)
                .map_err(MCSError::IoError)?;
            tmp_store.write_record(RecordKind::CreateRelation, &buf).map_err(MCSError::IoError)?;
        }
        tmp_store.flush_and_sync().map_err(MCSError::IoError)?;
        drop(tmp_store);

        // 3. Atomically rename over the original (atomic on POSIX), then fsync
        //    the containing directory so the rename itself is durable across a
        //    crash — content swap is atomic, but the directory entry update is
        //    not durable until the dir is synced (C2).
        std::fs::rename(&tmp_path, self.store.path()).map_err(MCSError::IoError)?;
        sync_parent_dir(self.store.path()).map_err(MCSError::IoError)?;

        // 4. Rebuild the entire in-memory graph from the compacted log (M1/M2).
        //    Replaying into fresh structures reclaims the interner arena (stale
        //    strings from deleted/edited entities), tombstoned entity slots, and
        //    stale search-index entries — none of which the old reopen-only path
        //    reclaimed.
        let path = self.store.path().clone();
        *self = KnowledgeGraph::new(&path).map_err(MCSError::IoError)?;

        Ok(())
    }

    // ---- Public API with write-ahead log (C1) and error propagation ----

    pub fn create_entities(&mut self, entities: &[Entity]) -> Result<Vec<Entity>> {
        // Validate up front so an invalid entity never produces partial writes.
        for entity in entities {
            if entity.name.is_empty() {
                return Err(MCSError::InvalidParams(
                    "Entity name must not be empty".into(),
                ));
            }
        }
        let mut created = Vec::new();
        for entity in entities {
            // Check dedup before writing (using non-interning lookup)
            let existing = self.interner.get_optional(&entity.name)
                .and_then(|id| {
                    let hash = self.interner.get_hash(id);
                    self.name_table.lookup(hash, id)
                });
            if existing.is_some() {
                continue;
            }
            // Write-ahead: encode and log before mutating state
            let mut buf = Vec::new();
            store_enc::encode_create_entity(&mut buf, &entity.name, &entity.entity_type, &entity.observations)
                .map_err(MCSError::IoError)?;
            self.store.write_record(RecordKind::CreateEntity, &buf)
                .map_err(MCSError::IoError)?;

            let name_id = self.interner.intern(&entity.name);
            let hash = self.interner.get_hash(name_id);
            let type_id = self.interner.intern(&entity.entity_type);
            let obs_ids: Vec<StrId> = entity
                .observations
                .iter()
                .map(|o| self.interner.intern(o))
                .collect();
            // Reuse a tombstoned slot if one is free (M2); its old search-index
            // entries were cleared on delete, so the slot starts clean.
            let reused = self.free_slots.pop();
            let slot = reused.unwrap_or(self.entity_slots.len() as u32);
            self.search
                .index_entity(&mut self.interner, slot, name_id, type_id, &obs_ids);
            let stored = Some(StoredEntity {
                state: ENTITY_SLOT_LIVE,
                name: name_id,
                entity_type: type_id,
                observations: obs_ids,
            });
            match reused {
                Some(s) => self.entity_slots[s as usize] = stored,
                None => self.entity_slots.push(stored),
            }
            self.name_table.insert(&self.interner, hash, name_id, slot);
            created.push(Entity {
                name: entity.name.clone(),
                entity_type: entity.entity_type.clone(),
                observations: entity.observations.clone(),
            });
        }
        Ok(created)
    }

    pub fn create_relations(&mut self, relations: &[Relation]) -> Result<Vec<Relation>> {
        // Validate up front so an invalid relation never produces partial writes.
        for relation in relations {
            if relation.from.is_empty() || relation.to.is_empty() {
                return Err(MCSError::InvalidParams(
                    "Relation endpoints must not be empty".into(),
                ));
            }
        }
        let mut created = Vec::new();
        // Build a dedup set for O(1) duplicate checks (P5)
        let mut rel_set: AHashSet<(StrId, StrId, StrId)> = AHashSet::new();
        for rel in &self.relations {
            rel_set.insert((rel.from, rel.to, rel.relation_type));
        }
        for relation in relations {
            let from_id = self.interner.intern(&relation.from);
            let to_id = self.interner.intern(&relation.to);
            let type_id = self.interner.intern(&relation.relation_type);
            if !rel_set.insert((from_id, to_id, type_id)) {
                continue;
            }
            // Write-ahead: log before mutation
            let mut buf = Vec::new();
            store_enc::encode_create_relation(&mut buf, &relation.from, &relation.to, &relation.relation_type)
                .map_err(MCSError::IoError)?;
            self.store.write_record(RecordKind::CreateRelation, &buf)
                .map_err(MCSError::IoError)?;

            self.relations.push(StoredRelation {
                from: from_id,
                to: to_id,
                relation_type: type_id,
            });
            self.adjacency.entry(from_id).or_default().push((to_id, type_id));
            self.adjacency.entry(to_id).or_default().push((from_id, type_id));
            created.push(Relation {
                from: relation.from.clone(),
                to: relation.to.clone(),
                relation_type: relation.relation_type.clone(),
            });
        }
        Ok(created)
    }

    pub fn add_observations(&mut self, entity_name: &str, contents: &[String]) -> Result<Vec<String>> {
        let name_id = self.interner.get_optional(entity_name)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{entity_name}' not found")))?;
        let hash = self.interner.get_hash(name_id);
        let slot = self
            .name_table
            .lookup(hash, name_id)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{entity_name}' not found")))?;
        // Snapshot the current observations so we can compute the deduplicated
        // additions *without* mutating in-memory state yet.
        let existing: AHashSet<StrId> = self
            .entity_slots
            .get(slot as usize)
            .and_then(|e| e.as_ref())
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{entity_name}' not found")))?
            .observations
            .iter()
            .copied()
            .collect();

        // Deduplicate against existing observations *and* within this batch, so
        // the live result matches what replay (which dedups one-by-one) rebuilds.
        let mut added = Vec::new();
        let mut interned_added = Vec::new();
        let mut seen: AHashSet<StrId> = AHashSet::new();
        for content in contents {
            let cid = self.interner.intern(content);
            if existing.contains(&cid) || !seen.insert(cid) {
                continue;
            }
            interned_added.push(cid);
            added.push(content.clone());
        }
        if added.is_empty() {
            return Ok(added);
        }

        // Write-ahead: the record must hit the log *before* any in-memory
        // mutation, so a failed write leaves memory and disk in agreement (C3).
        let mut buf = Vec::new();
        store_enc::encode_add_observations(&mut buf, entity_name, &added)
            .map_err(MCSError::IoError)?;
        self.store.write_record(RecordKind::AddObservations, &buf)
            .map_err(MCSError::IoError)?;

        // Logged — now apply to in-memory state.
        let stored = self
            .entity_slots
            .get_mut(slot as usize)
            .and_then(|e| e.as_mut())
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{entity_name}' not found")))?;
        stored.observations.extend_from_slice(&interned_added);

        // Incrementally index only the new observation tokens (P3) — no
        // full remove + re-index of the whole entity.
        self.search
            .index_additional(&mut self.interner, slot, &interned_added);
        Ok(added)
    }

    pub fn delete_entities(&mut self, entity_names: &[String]) -> Result<()> {
        let mut deleted_names = Vec::new();
        for name in entity_names {
            let name_id_opt = self.interner.get_optional(name);
            if let Some(name_id) = name_id_opt {
                let hash = self.interner.get_hash(name_id);
                if let Some(slot) = self.name_table.lookup(hash, name_id)
                    && let Some(Some(_)) = self.entity_slots.get(slot as usize)
                {
                    // Write-ahead: log before mutation
                    let mut buf = Vec::new();
                    store_enc::encode_delete_entity(&mut buf, name)
                        .map_err(MCSError::IoError)?;
                    self.store.write_record(RecordKind::DeleteEntity, &buf)
                        .map_err(MCSError::IoError)?;

                    self.entity_slots[slot as usize] = None;
                    self.free_slots.push(slot);
                    self.search.remove_entity(slot);
                    self.name_table.remove(&self.interner, hash, name_id);
                    deleted_names.push(name.clone());
                }
            }
        }
        if !deleted_names.is_empty() {
            // Use a AHashSet for O(1) retain checks (P5)
            let deleted_ids: AHashSet<StrId> = deleted_names.iter()
                .map(|n| self.interner.intern(n))
                .collect();
            self.relations
                .retain(|r| !deleted_ids.contains(&r.from) && !deleted_ids.contains(&r.to));
            // Clean adjacency index
            for id in &deleted_ids {
                self.adjacency.remove(id);
                // Remove references from other entities' adjacency lists
                for list in self.adjacency.values_mut() {
                    list.retain(|(to, _)| !deleted_ids.contains(to));
                }
            }
        }
        Ok(())
    }

    pub fn delete_observations(&mut self, entity_name: &str, observations: &[String]) -> Result<()> {
        let name_id = self.interner.get_optional(entity_name)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{entity_name}' not found")))?;
        let hash = self.interner.get_hash(name_id);
        let slot = self
            .name_table
            .lookup(hash, name_id)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{entity_name}' not found")))?;
        // Confirm the slot is live before logging.
        self.entity_slots
            .get(slot as usize)
            .and_then(|e| e.as_ref())
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{entity_name}' not found")))?;
        let remove_ids: AHashSet<StrId> = observations.iter().map(|o| self.interner.intern(o)).collect();

        // Write-ahead: log before touching in-memory state (C3).
        let mut buf = Vec::new();
        store_enc::encode_delete_observations(&mut buf, entity_name, observations)
            .map_err(MCSError::IoError)?;
        self.store.write_record(RecordKind::DeleteObservations, &buf)
            .map_err(MCSError::IoError)?;

        // Logged — now apply.
        let stored = self
            .entity_slots
            .get_mut(slot as usize)
            .and_then(|e| e.as_mut())
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{entity_name}' not found")))?;
        stored.observations.retain(|o| !remove_ids.contains(o));
        self.search.remove_entity(slot);
        self.search
            .index_entity(&mut self.interner, slot, stored.name, stored.entity_type, &stored.observations);
        Ok(())
    }

    pub fn delete_relations(&mut self, relations: &[Relation]) -> Result<()> {
        // Collect targets into a AHashSet for O(1) retain checks (P5)
        let rels: AHashSet<(StrId, StrId, StrId)> = relations
            .iter()
            .map(|r| {
                (
                    self.interner.intern(&r.from),
                    self.interner.intern(&r.to),
                    self.interner.intern(&r.relation_type),
                )
            })
            .collect();
        // Write-ahead: log every deletion before mutating in-memory state (C3),
        // so a failed write can't leave memory ahead of the log.
        for relation in relations {
            let mut buf = Vec::new();
            store_enc::encode_delete_relation(&mut buf, &relation.from, &relation.to, &relation.relation_type)
                .map_err(MCSError::IoError)?;
            self.store.write_record(RecordKind::DeleteRelation, &buf)
                .map_err(MCSError::IoError)?;
        }
        self.relations
            .retain(|r| !rels.contains(&(r.from, r.to, r.relation_type)));
        // Clean adjacency index
        for (f, t, rt) in &rels {
            if let Some(edges) = self.adjacency.get_mut(f) {
                edges.retain(|(to, rtype)| to != t || rtype != rt);
                if edges.is_empty() {
                    self.adjacency.remove(f);
                }
            }
            if let Some(edges) = self.adjacency.get_mut(t) {
                edges.retain(|(to, rtype)| to != f || rtype != rt);
                if edges.is_empty() {
                    self.adjacency.remove(t);
                }
            }
        }
        Ok(())
    }

    pub fn read_graph(&self) -> KnowledgeGraphOut {
        self.read_graph_view().to_owned_out()
    }

    /// Borrowing, allocation-light view of the full graph (M6). Serializing it
    /// streams interned `&str` directly instead of materializing a `String`
    /// per name/type/observation.
    pub fn read_graph_view(&self) -> GraphView<'_> {
        let entities: Vec<&StoredEntity> = self
            .entity_slots
            .iter()
            .filter_map(|s| s.as_ref().filter(|e| e.is_live()))
            .collect();
        let relations: Vec<&StoredRelation> = self.relations.iter().collect();
        GraphView { kg: self, entities, relations }
    }

    /// Relevance-ranked substring search returning all matches (no pagination).
    /// Equivalent to `search_nodes_filtered(query, None, 0, usize::MAX)`.
    pub fn search_nodes(&self, query: &str) -> KnowledgeGraphOut {
        self.search_nodes_filtered(query, None, 0, usize::MAX)
    }

    pub fn open_nodes(&self, names: &[String]) -> KnowledgeGraphOut {
        self.open_nodes_view(names).to_owned_out()
    }

    /// Borrowing view variant of [`open_nodes`] (M6).
    pub fn open_nodes_view(&self, names: &[String]) -> GraphView<'_> {
        let name_ids: AHashSet<StrId> = names.iter()
            .filter_map(|n| self.interner.get_optional(n))
            .collect();
        let entities: Vec<&StoredEntity> = self
            .entity_slots
            .iter()
            .filter_map(|s| {
                s.as_ref()
                    .filter(|stored| stored.is_live() && name_ids.contains(&stored.name))
            })
            .collect();
        let matched_names: AHashSet<StrId> = entities.iter().map(|e| e.name).collect();
        let relations: Vec<&StoredRelation> = self
            .relations
            .iter()
            .filter(|r| matched_names.contains(&r.from) || matched_names.contains(&r.to))
            .collect();
        GraphView { kg: self, entities, relations }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn entity_to_output(&self, stored: &StoredEntity) -> Entity {
        Entity {
            name: self.interner.lookup(stored.name).to_string(),
            entity_type: self.interner.lookup(stored.entity_type).to_string(),
            observations: stored
                .observations
                .iter()
                .map(|o| self.interner.lookup(*o).to_string())
                .collect(),
        }
    }

    #[inline]
    fn relation_to_output(&self, r: &StoredRelation) -> Relation {
        Relation {
            from: self.interner.lookup(r.from).to_string(),
            to: self.interner.lookup(r.to).to_string(),
            relation_type: self.interner.lookup(r.relation_type).to_string(),
        }
    }

    /// Resolve a name to its live entity slot, or `None` if absent/deleted.
    fn lookup_live_slot(&self, name: &str) -> Option<u32> {
        let name_id = self.interner.get_optional(name)?;
        let hash = self.interner.get_hash(name_id);
        let slot = self.name_table.lookup(hash, name_id)?;
        let stored = self.entity_slots.get(slot as usize)?.as_ref()?;
        stored.is_live().then_some(slot)
    }

    /// Materialize a live entity from its interned name id.
    fn entity_by_name_id(&self, name_id: StrId) -> Option<Entity> {
        let hash = self.interner.get_hash(name_id);
        let slot = self.name_table.lookup(hash, name_id)?;
        let stored = self.entity_slots.get(slot as usize)?.as_ref()?;
        stored.is_live().then(|| self.entity_to_output(stored))
    }

    /// Tally distinct entity types and their live-entity counts, ranked by
    /// count descending (ties broken by name). One linear pass over the dense
    /// slot vec; only the final names are allocated.
    pub fn entity_type_counts(&self) -> Vec<(String, usize)> {
        let mut counts: AHashMap<StrId, usize> = AHashMap::new();
        for st in self
            .entity_slots
            .iter()
            .filter_map(|s| s.as_ref())
            .filter(|e| e.is_live())
        {
            *counts.entry(st.entity_type).or_insert(0) += 1;
        }
        self.rank_counts(counts)
    }

    /// Tally distinct relation types and their counts, ranked by count desc.
    pub fn relation_type_counts(&self) -> Vec<(String, usize)> {
        let mut counts: AHashMap<StrId, usize> = AHashMap::new();
        for r in &self.relations {
            *counts.entry(r.relation_type).or_insert(0) += 1;
        }
        self.rank_counts(counts)
    }

    fn rank_counts(&self, counts: AHashMap<StrId, usize>) -> Vec<(String, usize)> {
        let mut out: Vec<(String, usize)> = counts
            .into_iter()
            .map(|(id, c)| (self.interner.lookup(id).to_string(), c))
            .collect();
        out.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// Relevance-ranked, optionally type-filtered, paginated node search.
    /// Entities come back best-match-first (see [`SearchIndex::search_ranked`]).
    /// Relations touching any returned entity (either endpoint) are included.
    pub fn search_nodes_filtered(
        &self,
        query: &str,
        entity_type: Option<&str>,
        offset: usize,
        limit: usize,
    ) -> KnowledgeGraphOut {
        self.search_nodes_view(query, entity_type, offset, limit).to_owned_out()
    }

    /// Borrowing view variant of [`search_nodes_filtered`] (M6).
    pub fn search_nodes_view(
        &self,
        query: &str,
        entity_type: Option<&str>,
        offset: usize,
        limit: usize,
    ) -> GraphView<'_> {
        let type_id = match entity_type {
            Some(t) => match self.interner.get_optional(t) {
                Some(id) => Some(id),
                None => return GraphView { kg: self, entities: Vec::new(), relations: Vec::new() },
            },
            None => None,
        };

        let ranked = self.search.search_ranked(query, &self.interner);
        let mut selected: AHashSet<StrId> = AHashSet::new();
        let mut entities: Vec<&StoredEntity> = Vec::new();
        let mut skipped = 0usize;
        for (slot, _score) in ranked {
            let Some(st) = self
                .entity_slots
                .get(slot as usize)
                .and_then(|s| s.as_ref())
                .filter(|e| e.is_live())
            else {
                continue;
            };
            if type_id.is_some_and(|tid| st.entity_type != tid) {
                continue;
            }
            if skipped < offset {
                skipped += 1;
                continue;
            }
            if entities.len() >= limit {
                break;
            }
            selected.insert(st.name);
            entities.push(st);
        }

        let relations: Vec<&StoredRelation> = self
            .relations
            .iter()
            .filter(|r| selected.contains(&r.from) || selected.contains(&r.to))
            .collect();
        GraphView { kg: self, entities, relations }
    }

    /// Type-filtered, paginated view of the whole graph. Unlike [`read_graph`],
    /// relations are restricted to those whose **both** endpoints fall in the
    /// returned entity page, so the slice is internally consistent.
    pub fn read_graph_filtered(
        &self,
        entity_type: Option<&str>,
        offset: usize,
        limit: usize,
    ) -> KnowledgeGraphOut {
        self.read_graph_filtered_view(entity_type, offset, limit).to_owned_out()
    }

    /// Borrowing view variant of [`read_graph_filtered`] (M6).
    pub fn read_graph_filtered_view(
        &self,
        entity_type: Option<&str>,
        offset: usize,
        limit: usize,
    ) -> GraphView<'_> {
        let type_id = match entity_type {
            Some(t) => match self.interner.get_optional(t) {
                Some(id) => Some(id),
                None => return GraphView { kg: self, entities: Vec::new(), relations: Vec::new() },
            },
            None => None,
        };

        let mut selected: AHashSet<StrId> = AHashSet::new();
        let mut entities: Vec<&StoredEntity> = Vec::new();
        let mut skipped = 0usize;
        for st in self
            .entity_slots
            .iter()
            .filter_map(|s| s.as_ref())
            .filter(|e| e.is_live())
        {
            if type_id.is_some_and(|tid| st.entity_type != tid) {
                continue;
            }
            if skipped < offset {
                skipped += 1;
                continue;
            }
            if entities.len() >= limit {
                break;
            }
            selected.insert(st.name);
            entities.push(st);
        }

        let relations: Vec<&StoredRelation> = self
            .relations
            .iter()
            .filter(|r| selected.contains(&r.from) && selected.contains(&r.to))
            .collect();
        GraphView { kg: self, entities, relations }
    }

    /// Neighborhood expansion around `name` out to `depth` hops, following
    /// edges in the requested [`Direction`] and (optionally) of one relation
    /// type. Returns the origin plus reached entities, and every relation
    /// (passing the type filter) whose endpoints are both inside that set.
    ///
    /// `depth == 1` (the common case) is a single linear pass over the flat
    /// relation vec; deeper queries build an adjacency map once (O(E)) and BFS.
    pub fn neighbors(
        &self,
        name: &str,
        direction: Direction,
        rtype: Option<&str>,
        depth: u32,
    ) -> Result<KnowledgeGraphOut> {
        self.lookup_live_slot(name)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{name}' not found")))?;
        // Safe: lookup_live_slot succeeded, so the name is interned.
        let start = self.interner.get_optional(name).unwrap();

        // An unknown relation-type filter can match nothing: return just origin.
        let rtype_id = match rtype {
            Some(r) => match self.interner.get_optional(r) {
                Some(id) => Some(id),
                None => {
                    let entities = self.entity_by_name_id(start).into_iter().collect();
                    return Ok(KnowledgeGraphOut { entities, relations: Vec::new() });
                }
            },
            None => None,
        };

        let mut visited: AHashSet<StrId> = AHashSet::new();
        visited.insert(start);

        let type_ok = |r: &StoredRelation| rtype_id.is_none_or(|rt| r.relation_type == rt);

        if depth == 1 {
            for r in self.relations.iter().filter(|r| type_ok(r)) {
                match direction {
                    Direction::Out => {
                        if r.from == start {
                            visited.insert(r.to);
                        }
                    }
                    Direction::In => {
                        if r.to == start {
                            visited.insert(r.from);
                        }
                    }
                    Direction::Both => {
                        if r.from == start {
                            visited.insert(r.to);
                        } else if r.to == start {
                            visited.insert(r.from);
                        }
                    }
                }
            }
        } else if depth >= 2 {
            // Build a direction-aware adjacency map once, then BFS.
            // For Direction::Both we use the incremental adjacency index;
            // for Direction::Out/In we filter relations directly.
            let mut adj: AHashMap<StrId, Vec<StrId>> = AHashMap::new();
            match direction {
                Direction::Both => {
                    for (&node, edges) in &self.adjacency {
                        for &(nb, rt) in edges {
                            if rtype_id.is_none_or(|rt_id| rt == rt_id) {
                                adj.entry(node).or_default().push(nb);
                            }
                        }
                    }
                }
                Direction::Out | Direction::In => {
                    for r in self.relations.iter().filter(|r| type_ok(r)) {
                        match direction {
                            Direction::Out => adj.entry(r.from).or_default().push(r.to),
                            Direction::In => adj.entry(r.to).or_default().push(r.from),
                            _ => unreachable!(),
                        }
                    }
                }
            }
            let mut queue: VecDeque<(StrId, u32)> = VecDeque::new();
            queue.push_back((start, 0));
            while let Some((node, d)) = queue.pop_front() {
                if d >= depth {
                    continue;
                }
                if let Some(nbrs) = adj.get(&node) {
                    for &nb in nbrs {
                        if visited.insert(nb) {
                            queue.push_back((nb, d + 1));
                        }
                    }
                }
            }
        }

        let mut entities = Vec::with_capacity(visited.len());
        for &nid in &visited {
            if let Some(e) = self.entity_by_name_id(nid) {
                entities.push(e);
            }
        }
        let relations = self
            .relations
            .iter()
            .filter(|r| type_ok(r) && visited.contains(&r.from) && visited.contains(&r.to))
            .map(|r| self.relation_to_output(r))
            .collect();
        Ok(KnowledgeGraphOut { entities, relations })
    }

    /// One-shot context bundle for a single entity: the entity itself, every
    /// incident relation, its distinct neighbor names, and its degree. Saves an
    /// agent the get_entity + two search_relations round-trips.
    pub fn describe_entity(&self, name: &str) -> Result<serde_json::Value> {
        let name_id = self
            .interner
            .get_optional(name)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{name}' not found")))?;
        let entity = self
            .entity_by_name_id(name_id)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{name}' not found")))?;

        let mut incident: Vec<Relation> = Vec::new();
        let mut neighbor_seen: AHashSet<StrId> = AHashSet::new();
        let mut neighbors: Vec<&str> = Vec::new();
        for r in &self.relations {
            if r.from == name_id || r.to == name_id {
                incident.push(self.relation_to_output(r));
                let other = if r.from == name_id { r.to } else { r.from };
                if other != name_id && neighbor_seen.insert(other) {
                    neighbors.push(self.interner.lookup(other));
                }
            }
        }

        Ok(serde_json::json!({
            "entity": entity,
            "relations": incident,
            "neighbors": neighbors,
            "degree": incident.len(),
        }))
    }

    /// Create-or-merge a batch of entities idempotently. Missing entities are
    /// created; existing ones keep their type and gain any new observations
    /// (deduplicated). Returns a per-entity outcome. The caller is responsible
    /// for flushing — every underlying op is already write-ahead logged.
    pub fn upsert_entities(&mut self, entities: &[Entity]) -> Result<Vec<serde_json::Value>> {
        for e in entities {
            if e.name.is_empty() {
                return Err(MCSError::InvalidParams(
                    "Entity name must not be empty".into(),
                ));
            }
        }
        let mut out = Vec::with_capacity(entities.len());
        for e in entities {
            if self.lookup_live_slot(&e.name).is_some() {
                let added = self.add_observations(&e.name, &e.observations)?;
                out.push(serde_json::json!({
                    "name": e.name,
                    "created": false,
                    "addedObservations": added,
                }));
            } else {
                let created = self.create_entities(std::slice::from_ref(e))?;
                out.push(serde_json::json!({
                    "name": e.name,
                    "created": !created.is_empty(),
                    "addedObservations": e.observations,
                }));
            }
        }
        Ok(out)
    }

    /// Serialize the graph in one of: `json` (read_graph), `mermaid`, `dot`.
    pub fn export(&self, format: &str) -> Result<String> {
        match format {
            "json" => serde_json::to_string(&self.read_graph()).map_err(MCSError::JsonError),
            "mermaid" => Ok(self.export_mermaid()),
            "dot" => Ok(self.export_dot()),
            other => Err(MCSError::InvalidParams(format!(
                "Unknown export format '{other}' (expected json|mermaid|dot)"
            ))),
        }
    }

    /// Assign each live entity a stable `n{k}` node id for diagram output.
    fn diagram_node_ids(&self) -> (AHashMap<StrId, usize>, Vec<(usize, StrId)>) {
        let mut ids: AHashMap<StrId, usize> = AHashMap::new();
        let mut order: Vec<(usize, StrId)> = Vec::new();
        for st in self
            .entity_slots
            .iter()
            .filter_map(|s| s.as_ref())
            .filter(|e| e.is_live())
        {
            let n = ids.len();
            ids.insert(st.name, n);
            order.push((n, st.name));
        }
        (ids, order)
    }

    fn export_mermaid(&self) -> String {
        let (ids, order) = self.diagram_node_ids();
        let mut s = String::with_capacity(64 + order.len() * 32 + self.relations.len() * 32);
        s.push_str("graph LR\n");
        for (n, name_id) in &order {
            let label = sanitize_label(self.interner.lookup(*name_id));
            s.push_str(&format!("  n{n}[\"{label}\"]\n"));
        }
        for r in &self.relations {
            if let (Some(&a), Some(&b)) = (ids.get(&r.from), ids.get(&r.to)) {
                let rel = sanitize_label(self.interner.lookup(r.relation_type));
                s.push_str(&format!("  n{a} -->|{rel}| n{b}\n"));
            }
        }
        s
    }

    fn export_dot(&self) -> String {
        let (ids, order) = self.diagram_node_ids();
        let mut s = String::with_capacity(64 + order.len() * 32 + self.relations.len() * 32);
        s.push_str("digraph G {\n");
        for (n, name_id) in &order {
            let label = sanitize_label(self.interner.lookup(*name_id));
            s.push_str(&format!("  n{n} [label=\"{label}\"];\n"));
        }
        for r in &self.relations {
            if let (Some(&a), Some(&b)) = (ids.get(&r.from), ids.get(&r.to)) {
                let rel = sanitize_label(self.interner.lookup(r.relation_type));
                s.push_str(&format!("  n{a} -> n{b} [label=\"{rel}\"];\n"));
            }
        }
        s.push_str("}\n");
        s
    }

    // ------ High-level productivity tools ------

    /// Merge `source` entity into `target` entity. All observations from
    /// source are moved to target (deduplicated), all relations involving
    /// source are redirected to target (deduplicated), and source is then
    /// deleted.
    ///
    /// The whole merge is **atomic**: every sub-record is written to the log
    /// inside a single `TxnBegin`…`TxnCommit` transaction *before* any in-memory
    /// mutation. A failed or torn write therefore leaves both memory and the
    /// log untouched (an uncommitted transaction is discarded on replay), so the
    /// graph can never observe a half-applied merge. Caller flushes.
    pub fn merge_entities(&mut self, source: &str, target: &str) -> Result<serde_json::Value> {
        if source == target {
            return Err(MCSError::InvalidParams(
                "Source and target must be different entities".into(),
            ));
        }
        self.lookup_live_slot(source).ok_or_else(|| {
            MCSError::InvalidParams(format!("Source entity '{source}' not found"))
        })?;
        let target_slot = self.lookup_live_slot(target).ok_or_else(|| {
            MCSError::InvalidParams(format!("Target entity '{target}' not found"))
        })?;

        let source_entity = self.get_entity(source).unwrap();
        let moved_obs_count = source_entity.observations.len();
        let source_id = self.interner.get_optional(source).unwrap();
        let target_id = self.interner.get_optional(target).unwrap();

        // Observations to move: dedup against target's existing set and within
        // the batch (matching what `add_observations` would have done).
        let target_existing: AHashSet<StrId> = self.entity_slots[target_slot as usize]
            .as_ref()
            .unwrap()
            .observations
            .iter()
            .copied()
            .collect();
        let mut obs_seen: AHashSet<StrId> = AHashSet::new();
        let mut obs_to_add: Vec<String> = Vec::new();
        for o in &source_entity.observations {
            if let Some(oid) = self.interner.get_optional(o)
                && !target_existing.contains(&oid)
                && obs_seen.insert(oid)
            {
                obs_to_add.push(o.clone());
            }
        }

        // Relations to redirect: replace source with target, drop self-loops,
        // and dedup against existing relations and within the batch.
        let existing_rels: AHashSet<(StrId, StrId, StrId)> =
            self.relations.iter().map(|r| (r.from, r.to, r.relation_type)).collect();
        let mut rel_seen: AHashSet<(StrId, StrId, StrId)> = AHashSet::new();
        let mut redirect: Vec<Relation> = Vec::new();
        for r in &self.relations {
            if r.from != source_id && r.to != source_id {
                continue;
            }
            let new_from = if r.from == source_id { target_id } else { r.from };
            let new_to = if r.to == source_id { target_id } else { r.to };
            if new_from == new_to {
                continue; // self-loop after redirect
            }
            let key = (new_from, new_to, r.relation_type);
            if existing_rels.contains(&key) || !rel_seen.insert(key) {
                continue;
            }
            redirect.push(Relation {
                from: self.interner.lookup(new_from).to_string(),
                to: self.interner.lookup(new_to).to_string(),
                relation_type: self.interner.lookup(r.relation_type).to_string(),
            });
        }

        let added_count = obs_to_add.len();
        let redirected = redirect.len() as u32;

        // Build every record up front so writing is the only fallible step.
        let mut records: Vec<(RecordKind, Vec<u8>)> = Vec::new();
        if !obs_to_add.is_empty() {
            let mut buf = Vec::new();
            store_enc::encode_add_observations(&mut buf, target, &obs_to_add)
                .map_err(MCSError::IoError)?;
            records.push((RecordKind::AddObservations, buf));
        }
        for r in &redirect {
            let mut buf = Vec::new();
            store_enc::encode_create_relation(&mut buf, &r.from, &r.to, &r.relation_type)
                .map_err(MCSError::IoError)?;
            records.push((RecordKind::CreateRelation, buf));
        }
        let mut del_buf = Vec::new();
        store_enc::encode_delete_entity(&mut del_buf, source).map_err(MCSError::IoError)?;
        records.push((RecordKind::DeleteEntity, del_buf));

        // Write-ahead, transactionally: begin, all records, commit.
        self.store.write_record(RecordKind::TxnBegin, &[]).map_err(MCSError::IoError)?;
        for (kind, data) in &records {
            self.store.write_record(*kind, data).map_err(MCSError::IoError)?;
        }
        self.store.write_record(RecordKind::TxnCommit, &[]).map_err(MCSError::IoError)?;

        // Logged and committed — now apply to in-memory state (no more logging).
        for (kind, data) in &records {
            Self::apply_record(
                *kind, data, &mut self.interner, &mut self.entity_slots, &mut self.search,
                &mut self.name_table, &mut self.relations,
            );
        }

        Ok(serde_json::json!({
            "source": source,
            "target": target,
            "movedObservations": moved_obs_count,
            "addedObservations": added_count,
            "redirectedRelations": redirected,
        }))
    }

    /// Extract a connected subgraph around one or more seed entity names,
    /// expanding out to `depth` hops along all relations (undirected). Returns
    /// the set of reached entities and the relations among them.
    pub fn extract_subgraph(&self, names: &[String], depth: u32) -> Result<KnowledgeGraphOut> {
        if names.is_empty() {
            return Ok(KnowledgeGraphOut {
                entities: Vec::new(),
                relations: Vec::new(),
            });
        }
        // Seed the BFS queue from any names that exist.
        let mut visited: AHashSet<StrId> = AHashSet::new();
        let mut queue: VecDeque<(StrId, u32)> = VecDeque::new();
        for name in names {
            if let Some(id) = self.interner.get_optional(name)
                && visited.insert(id)
            {
                queue.push_back((id, 0));
            }
        }
        // Build an undirected adjacency map from the incremental index.
        let mut adj: AHashMap<StrId, Vec<StrId>> = AHashMap::new();
        for (&node, edges) in &self.adjacency {
            let nb: Vec<StrId> = edges.iter().map(|(to, _)| *to).collect();
            adj.insert(node, nb);
        }
        while let Some((node, d)) = queue.pop_front() {
            if d >= depth {
                continue;
            }
            if let Some(nbrs) = adj.get(&node) {
                for &nb in nbrs {
                    if visited.insert(nb) {
                        queue.push_back((nb, d + 1));
                    }
                }
            }
        }
        let mut entities: Vec<Entity> = Vec::with_capacity(visited.len());
        for &nid in &visited {
            if let Some(e) = self.entity_by_name_id(nid) {
                entities.push(e);
            }
        }
        let relations: Vec<Relation> = self
            .relations
            .iter()
            .filter(|r| visited.contains(&r.from) && visited.contains(&r.to))
            .map(|r| self.relation_to_output(r))
            .collect();
        Ok(KnowledgeGraphOut { entities, relations })
    }

    /// Return full entities for a list of names. Missing names yield `None`.
    pub fn batch_get_entities(&self, names: &[String]) -> Vec<Option<Entity>> {
        names.iter().map(|n| self.get_entity(n)).collect()
    }

    /// Recursive DFS helper — collects every simple path from `current` to
    /// `target` up to `max_depth` hops, capped at `max_paths` results.
    #[allow(clippy::too_many_arguments)]
    fn dfs_all_paths(
        adj: &AHashMap<StrId, Vec<StrId>>,
        current: StrId,
        target: StrId,
        max_depth: usize,
        max_paths: usize,
        visited: &mut AHashSet<StrId>,
        current_path: &mut Vec<StrId>,
        all_paths: &mut Vec<Vec<StrId>>,
    ) {
        if all_paths.len() >= max_paths {
            return;
        }
        if current == target && current_path.len() > 1 {
            all_paths.push(current_path.clone());
            return;
        }
        if current_path.len() > max_depth {
            return;
        }
        if let Some(neighbors) = adj.get(&current) {
            for &nb in neighbors {
                if visited.insert(nb) {
                    current_path.push(nb);
                    Self::dfs_all_paths(
                        adj, nb, target, max_depth, max_paths, visited, current_path, all_paths,
                    );
                    current_path.pop();
                    visited.remove(&nb);
                }
            }
        }
    }

    /// Find all simple paths between `from` and `to` up to `max_depth` hops,
    /// returning at most `max_paths` results. Paths are found via DFS with
    /// backtracking and include both endpoints.
    pub fn find_all_paths(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
        max_paths: usize,
    ) -> Result<Vec<Vec<String>>> {
        let from_id = self
            .interner
            .get_optional(from)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{from}' not found")))?;
        let to_id = self
            .interner
            .get_optional(to)
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{to}' not found")))?;
        // Verify both are live.
        if self.lookup_live_slot(from).is_none() {
            return Err(MCSError::InvalidParams(format!("Entity '{from}' not found")));
        }
        if self.lookup_live_slot(to).is_none() {
            return Err(MCSError::InvalidParams(format!("Entity '{to}' not found")));
        }
        if from_id == to_id {
            return Ok(vec![vec![from.to_string()]]);
        }
        // Build undirected adjacency from the incremental index.
        let mut adj: AHashMap<StrId, Vec<StrId>> = AHashMap::with_capacity(self.adjacency.len());
        for (&node, edges) in &self.adjacency {
            let nbrs: Vec<StrId> = edges.iter().map(|(to, _)| *to).collect();
            adj.insert(node, nbrs);
        }
        let mut all_paths: Vec<Vec<StrId>> = Vec::new();
        let mut current_path = Vec::new();
        let mut visited: AHashSet<StrId> = AHashSet::new();
        visited.insert(from_id);
        current_path.push(from_id);
        Self::dfs_all_paths(
            &adj,
            from_id,
            to_id,
            max_depth,
            max_paths,
            &mut visited,
            &mut current_path,
            &mut all_paths,
        );
        if all_paths.is_empty() {
            return Err(MCSError::MemoryError(format!(
                "No path found between '{from}' and '{to}'"
            )));
        }
        let result: Vec<Vec<String>> = all_paths
            .into_iter()
            .map(|path| {
                path.into_iter()
                    .map(|id| self.interner.lookup(id).to_string())
                    .collect()
            })
            .collect();
        Ok(result)
    }

    // --- Snapshot ---

    /// Create a wait-free read snapshot (item 2 in plan).
    /// Freezes entity_slots and relations into `Arc<[_]>` and clones the rest.
    pub fn snapshot(&self) -> ReadSnapshot {
        ReadSnapshot {
            interner: self.interner.clone(),
            entity_slots: Arc::from_iter(self.entity_slots.iter().cloned()),
            free_slots: self.free_slots.clone(),
            name_table: self.name_table.clone(),
            relations: Arc::from_iter(self.relations.iter().cloned()),
            adjacency: self.adjacency.clone(),
            search: self.search.clone(),
        }
    }

    // --- Flush & sync ---

    /// Flush the `BufWriter` to the kernel buffer (process-crash safe).
    pub fn flush(&mut self) -> Result<()> {
        self.store.flush().map_err(MCSError::IoError)
    }

    /// `fsync` the log to disk (OS-crash safe). Called by the background sync
    /// thread in [`GraphHandle`]; most callers should use `flush()` instead.
    pub fn sync(&mut self) -> Result<()> {
        self.store.sync().map_err(MCSError::IoError)
    }

    /// Flush + fsync (legacy; prefer [`flush`](Self::flush) for production use).
    pub fn flush_and_sync(&mut self) -> Result<()> {
        self.store.flush_and_sync().map_err(MCSError::IoError)
    }
}



// ---------------------------------------------------------------------------
// GraphHandle – wait-free read / serialized-write handle.
// ---------------------------------------------------------------------------

/// Wait-free read / serialized-write handle to the graph.
///
/// Readers load a frozen [`ReadSnapshot`] via [`read`](GraphHandle::read)
/// (lock-free via `ArcSwap`). Writers take a [`Mutex`] lock, mutate the
/// underlying [`KnowledgeGraph`], and publish a fresh snapshot on unlock
/// via the [`WriteGuard`] drop glue.
///
/// A background thread calls `fsync` on the WAL file every 1 second so that
/// write handlers never block on disk I/O. The thread is stopped on `Drop`.
///
/// The sync thread uses its own `Arc<File>` handle (cloned from the WAL file)
/// so that `fsync` never contends with the graph mutex. A [`Condvar`] notifies
/// the thread immediately after every write, ensuring low-latency sync without
/// polling.
pub struct GraphHandle {
    inner: Arc<parking_lot::Mutex<KnowledgeGraph>>,
    snapshot: ArcSwap<ReadSnapshot>,
    /// Cached JSON of the full `read_graph` output. Invalidated on every write.
    read_cache: ArcSwap<Option<Arc<str>>>,
    /// Notifies the background sync thread when a write has flushed data to the
    /// kernel buffer. The thread also wakes on a 1-second timeout as a fallback.
    /// The `bool` is `true` when there is pending data to sync.
    sync_notify: Arc<(StdMutex<bool>, Condvar)>,
    /// Signal the background sync thread to stop. Set on `Drop`.
    stop_sync: Arc<AtomicBool>,
}

/// RAII guard that publishes a fresh [`ReadSnapshot`] on drop.
pub struct WriteGuard<'a> {
    guard: parking_lot::MutexGuard<'a, KnowledgeGraph>,
    snapshot: &'a ArcSwap<ReadSnapshot>,
    read_cache: &'a ArcSwap<Option<Arc<str>>>,
    sync_notify: &'a (StdMutex<bool>, Condvar),
    did_publish: bool,
}

impl WriteGuard<'_> {
    /// Publish a snapshot now (eager, before drop). Also called by Drop.
    /// Invalidates the serialized read cache. Flushes the WAL to kernel buffer
    /// (the background sync thread in [`GraphHandle`] handles the actual `fsync`).
    pub fn publish(&mut self) {
        if let Err(e) = self.guard.flush() {
            tracing::error!("WAL flush failed: {e}");
        }
        let snap = Arc::new(self.guard.snapshot());
        self.snapshot.store(snap);
        self.read_cache.store(Arc::new(None));
        self.did_publish = true;
        // Wake the sync thread — data is in the kernel buffer waiting for fsync.
        let (lock, cvar) = self.sync_notify;
        let mut pending = lock.lock().unwrap_or_else(|e| e.into_inner());
        *pending = true;
        cvar.notify_one();
    }

    /// Access the underlying `KnowledgeGraph` for mutation.
    pub fn graph(&mut self) -> &mut KnowledgeGraph {
        &mut self.guard
    }
}

impl std::ops::Deref for WriteGuard<'_> {
    type Target = KnowledgeGraph;
    fn deref(&self) -> &KnowledgeGraph {
        &self.guard
    }
}

impl std::ops::DerefMut for WriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut KnowledgeGraph {
        &mut self.guard
    }
}

impl Drop for WriteGuard<'_> {
    fn drop(&mut self) {
        if !self.did_publish {
            self.publish();
        }
    }
}

impl Drop for GraphHandle {
    fn drop(&mut self) {
        self.stop_sync.store(true, Ordering::Relaxed);
        // Wake the sync thread by setting pending=true so the
        // wait_timeout_while(|p| !*p) condition breaks immediately.
        let (lock, cvar) = &*self.sync_notify;
        let mut pending = lock.lock().unwrap_or_else(|e| e.into_inner());
        *pending = true;
        cvar.notify_one();
    }
}

impl GraphHandle {
    /// Open or create the graph at `path`, seeding the initial snapshot.
    /// Spawns a background thread that `fsync`s the WAL so that write handlers
    /// never block on disk I/O.
    pub fn new(path: &Path) -> std::io::Result<Self> {
        let kg = KnowledgeGraph::new(path)?;
        let snapshot = Arc::new(kg.snapshot());
        // Clone the sync file handle before moving `kg` into the Mutex.
        let sync_file = Arc::clone(&kg.store.sync_file);
        let inner = Arc::new(parking_lot::Mutex::new(kg));

        let sync_notify: Arc<(StdMutex<bool>, Condvar)> =
            Arc::new((StdMutex::new(false), Condvar::new()));
        let notify = Arc::clone(&sync_notify);
        let stop_sync = Arc::new(AtomicBool::new(false));

        // Background sync thread — calls fsync on a dedicated `Arc<File>`
        // handle, never touching the graph mutex. Woken by Condvar after every
        // write; falls back to a 1-second timeout.
        let sync_stop = Arc::clone(&stop_sync);
        std::thread::Builder::new()
            .name("mcp-memory-sync".into())
            .spawn(move || {
                let (lock, cvar) = &*notify;
                loop {
                    // Wait while there is nothing to sync. Woken by publish()
                    // or by the 1-second timeout.
                    let mut guard = cvar
                        .wait_timeout_while(
                            lock.lock().unwrap_or_else(|e| e.into_inner()),
                            std::time::Duration::from_secs(1),
                            |p| !*p,
                        )
                        .unwrap_or_else(|e| e.into_inner())
                        .0;

                    let should_sync = *guard;
                    *guard = false;
                    // Release the StdMutex before fsync so publish() is not
                    // blocked on setting the next pending flag.
                    drop(guard);

                    if should_sync {
                        if let Err(e) = sync_file.sync_data() {
                            tracing::error!("WAL fsync failed: {e}");
                        }
                    }

                    if sync_stop.load(Ordering::Relaxed) {
                        // One final fsync before exiting.
                        if let Err(e) = sync_file.sync_data() {
                            tracing::error!("WAL final fsync failed: {e}");
                        }
                        break;
                    }
                }
            })
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        Ok(Self {
            inner,
            snapshot: ArcSwap::new(snapshot),
            read_cache: ArcSwap::new(Arc::new(None)),
            sync_notify,
            stop_sync,
        })
    }

    /// Return the cached full-graph JSON, or build and cache it on first call.
    /// Invalidated by any write (see [`WriteGuard::publish`]).
    pub fn read_graph_cached(&self) -> Arc<str> {
        if let Some(cached) = self.read_cache.load().as_ref() {
            return cached.clone();
        }
        let graph = self.read();
        let json: Arc<str> = Arc::from(graph.read_graph_json().into_boxed_str());
        self.read_cache.store(Arc::new(Some(json.clone())));
        json
    }

    /// Lock-free read snapshot. Holds an `Arc` reference to the frozen graph data.
    pub fn read(&self) -> ReadSnapshot {
        (**self.snapshot.load()).clone()
    }

    /// Serialised write access. Returns a guard that publishes a fresh snapshot
    /// when dropped (or when [`WriteGuard::publish`] is called eagerly).
    pub fn write(&self) -> WriteGuard<'_> {
        WriteGuard {
            guard: self.inner.lock(),
            snapshot: &self.snapshot,
            read_cache: &self.read_cache,
            sync_notify: &self.sync_notify,
            did_publish: false,
        }
    }
}


