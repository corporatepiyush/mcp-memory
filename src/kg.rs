use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

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

// ---------------------------------------------------------------------------
// StoredEntity / StoredRelation – internal representations using StrId.
// ---------------------------------------------------------------------------
struct StoredEntity {
    state: u8,
    name: StrId,
    entity_type: StrId,
    observations: Vec<StrId>,
}

impl StoredEntity {
    const fn is_live(&self) -> bool {
        self.state == ENTITY_SLOT_LIVE
    }
}

struct StoredRelation {
    from: StrId,
    to: StrId,
    relation_type: StrId,
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

struct NameTableShard {
    ctrl: Vec<u8>,      // 0xFF = empty; 0x00-0x7F = h2 stamp (bit 7 always clear)
    hashes: Vec<u64>,   // full 64-bit hash (used only during grow/rehash)
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
            hashes: vec![0; cap],
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

    fn insert(&mut self, hash: u64, name: StrId, slot: u32) {
        if self.count * 4 > self.ctrl.len() * 3 {
            self.grow();
        }
        let stamp = h2(hash);
        let mask = self.mask;
        let mut idx = h1(hash, mask);
        loop {
            // SAFETY: idx & mask always < len for power-of-two capacity.
            unsafe {
                if *self.ctrl.get_unchecked(idx) & 0x80 != 0 {
                    *self.ctrl.get_unchecked_mut(idx) = stamp;
                    *self.hashes.get_unchecked_mut(idx) = hash;
                    *self.names.get_unchecked_mut(idx) = name;
                    *self.slots.get_unchecked_mut(idx) = slot;
                    self.count += 1;
                    return;
                }
            }
            idx = (idx + 1) & mask;
        }
    }

    fn remove(&mut self, hash: u64, name: StrId) {
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
                self.hashes[idx] = 0;
                self.names[idx] = StrId::EMPTY;
                self.slots[idx] = u32::MAX;
                self.count -= 1;

                let mut next = (idx + 1) & mask;
                while self.ctrl[next] & 0x80 == 0 {
                    let nh = self.hashes[next];
                    let nn = self.names[next];
                    let ns = self.slots[next];
                    self.ctrl[next] = EMPTY_SLOT;
                    self.hashes[next] = 0;
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
                    self.hashes[re_idx] = nh;
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

    fn grow(&mut self) {
        let new_cap = self.ctrl.len() * 2;
        let new_mask = new_cap - 1;
        let mut new_ctrl = vec![EMPTY_SLOT; new_cap];
        let mut new_hashes = vec![0u64; new_cap];
        let mut new_names = vec![StrId::EMPTY; new_cap];
        let mut new_slots = vec![u32::MAX; new_cap];

        for i in 0..self.ctrl.len() {
            if self.ctrl[i] & 0x80 == 0 {
                let hash = self.hashes[i];
                let stamp = h2(hash);
                let mut idx = h1(hash, new_mask);
                while new_ctrl[idx] & 0x80 == 0 {
                    idx = (idx + 1) & new_mask;
                }
                new_ctrl[idx] = stamp;
                new_hashes[idx] = hash;
                new_names[idx] = self.names[i];
                new_slots[idx] = self.slots[i];
            }
        }

        self.ctrl = new_ctrl;
        self.hashes = new_hashes;
        self.names = new_names;
        self.slots = new_slots;
        self.mask = new_mask;
    }
}

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
    fn insert(&mut self, hash: u64, name: StrId, slot: u32) {
        self.shards[Self::shard(hash)].insert(hash, name, slot);
    }

    #[inline(always)]
    fn remove(&mut self, hash: u64, name: StrId) {
        self.shards[Self::shard(hash)].remove(hash, name);
    }
}

// ---------------------------------------------------------------------------
// KnowledgeGraph – the central type.
// ---------------------------------------------------------------------------
pub struct KnowledgeGraph {
    interner: StringInterner,
    entity_slots: Vec<Option<StoredEntity>>,
    name_table: ShardedNameTable,
    relations: Vec<StoredRelation>,
    search: SearchIndex,
    store: BinaryStore,
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

        store.replay(|kind, data| {
            match kind {
                RecordKind::CreateEntity => {
                    if let Some((name, etype, obs)) = store_enc::decode_create_entity(data) {
                        Self::replay_create_entity(
                            &mut interner, &mut entity_slots, &mut search, &mut name_table, name, etype, &obs,
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
                            &mut interner, &mut entity_slots, &mut search, &mut name_table, name, &obs,
                        );
                    }
                }
                RecordKind::DeleteEntity => {
                    if let Some(name) = store_enc::decode_delete_entity(data) {
                        Self::replay_delete_entity(
                            &mut interner, &mut entity_slots, &mut relations, &mut search, &mut name_table, name,
                        );
                    }
                }
                RecordKind::DeleteObservations => {
                    if let Some((name, obs)) = store_enc::decode_delete_observations(data) {
                        Self::replay_delete_observations(
                            &mut interner, &mut entity_slots, &mut search, &mut name_table, name, &obs,
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
            }
        })?;

        Ok(Self {
            interner,
            entity_slots,
            name_table,
            relations,
            search,
            store,
        })
    }

    // -----------------------------------------------------------------------
    // Replay helpers (static to avoid borrow issues in the closure)
    // -----------------------------------------------------------------------

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
        name_table.insert(hash, name_id, slot);
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
            name_table.remove(hash, name_id);
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

        // Build adjacency list (P4) — O(E) once, not O(V×E).
        let mut adj: HashMap<StrId, Vec<(StrId, StrId)>> = HashMap::new();
        for rel in &self.relations {
            adj.entry(rel.from).or_default().push((rel.to, rel.relation_type));
            adj.entry(rel.to).or_default().push((rel.from, rel.relation_type));
        }

        // BFS over adjacency list
        let mut visited: HashSet<StrId> = HashSet::new();
        let mut parent: HashMap<StrId, StrId> = HashMap::new();
        let mut queue: VecDeque<StrId> = VecDeque::new();

        visited.insert(from_id);
        queue.push_back(from_id);

        while let Some(current) = queue.pop_front() {
            if current == to_id {
                break;
            }

            if let Some(neighbors) = adj.get(&current) {
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

        // 2. Write to a temp file first
        let tmp_path = self.store.path().with_extension("tmp");
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

        // 3. Atomically rename over the original (atomic on POSIX)
        std::fs::rename(&tmp_path, self.store.path()).map_err(MCSError::IoError)?;

        // 4. Reopen the store with the new file
        self.store = BinaryStore::new(self.store.path()).map_err(MCSError::IoError)?;

        Ok(())
    }

    // ---- Public API with write-ahead log (C1) and error propagation ----

    pub fn create_entities(&mut self, entities: &[Entity]) -> Result<Vec<Entity>> {
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
            let slot = self.entity_slots.len() as u32;
            self.search
                .index_entity(&mut self.interner, slot, name_id, type_id, &obs_ids);
            self.entity_slots.push(Some(StoredEntity {
                state: ENTITY_SLOT_LIVE,
                name: name_id,
                entity_type: type_id,
                observations: obs_ids,
            }));
            self.name_table.insert(hash, name_id, slot);
            created.push(Entity {
                name: entity.name.clone(),
                entity_type: entity.entity_type.clone(),
                observations: entity.observations.clone(),
            });
        }
        Ok(created)
    }

    pub fn create_relations(&mut self, relations: &[Relation]) -> Result<Vec<Relation>> {
        let mut created = Vec::new();
        // Build a dedup set for O(1) duplicate checks (P5)
        let mut rel_set: HashSet<(StrId, StrId, StrId)> = HashSet::new();
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
        let stored = self
            .entity_slots
            .get_mut(slot as usize)
            .and_then(|e| e.as_mut())
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{entity_name}' not found")))?;

        // Deduplicate new observations (P7) — use HashSet for O(1) lookups
        let existing: HashSet<StrId> = stored.observations.iter().copied().collect();
        let mut added = Vec::new();
        let mut interned_added = Vec::new();
        for content in contents {
            let cid = self.interner.intern(content);
            if existing.contains(&cid) {
                continue;
            }
            stored.observations.push(cid);
            interned_added.push(cid);
            added.push(content.clone());
        }
        if !added.is_empty() {
            // Write-ahead: log before re-indexing
            let mut buf = Vec::new();
            store_enc::encode_add_observations(&mut buf, entity_name, &added)
                .map_err(MCSError::IoError)?;
            self.store.write_record(RecordKind::AddObservations, &buf)
                .map_err(MCSError::IoError)?;

            self.search.remove_entity(slot);
            self.search
                .index_entity(&mut self.interner, slot, stored.name, stored.entity_type, &stored.observations);
        }
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
                    self.search.remove_entity(slot);
                    self.name_table.remove(hash, name_id);
                    deleted_names.push(name.clone());
                }
            }
        }
        if !deleted_names.is_empty() {
            // Use a HashSet for O(1) retain checks (P5)
            let deleted_ids: HashSet<StrId> = deleted_names.iter()
                .map(|n| self.interner.intern(n))
                .collect();
            self.relations
                .retain(|r| !deleted_ids.contains(&r.from) && !deleted_ids.contains(&r.to));
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
        let stored = self
            .entity_slots
            .get_mut(slot as usize)
            .and_then(|e| e.as_mut())
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{entity_name}' not found")))?;
        let remove_ids: HashSet<StrId> = observations.iter().map(|o| self.interner.intern(o)).collect();
        stored.observations.retain(|o| !remove_ids.contains(o));
        // Write-ahead: log before re-indexing
        let mut buf = Vec::new();
        store_enc::encode_delete_observations(&mut buf, entity_name, observations)
            .map_err(MCSError::IoError)?;
        self.store.write_record(RecordKind::DeleteObservations, &buf)
            .map_err(MCSError::IoError)?;

        self.search.remove_entity(slot);
        self.search
            .index_entity(&mut self.interner, slot, stored.name, stored.entity_type, &stored.observations);
        Ok(())
    }

    pub fn delete_relations(&mut self, relations: &[Relation]) -> Result<()> {
        // Collect targets into a HashSet for O(1) retain checks (P5)
        let rels: HashSet<(StrId, StrId, StrId)> = relations
            .iter()
            .map(|r| {
                (
                    self.interner.intern(&r.from),
                    self.interner.intern(&r.to),
                    self.interner.intern(&r.relation_type),
                )
            })
            .collect();
        self.relations
            .retain(|r| !rels.contains(&(r.from, r.to, r.relation_type)));
        for relation in relations {
            let mut buf = Vec::new();
            store_enc::encode_delete_relation(&mut buf, &relation.from, &relation.to, &relation.relation_type)
                .map_err(MCSError::IoError)?;
            self.store.write_record(RecordKind::DeleteRelation, &buf)
                .map_err(MCSError::IoError)?;
        }
        Ok(())
    }

    pub fn read_graph(&self) -> KnowledgeGraphOut {
        let entities: Vec<Entity> = self
            .entity_slots
            .iter()
            .filter_map(|s| s.as_ref().filter(|e| e.is_live()))
            .map(|stored| self.entity_to_output(stored))
            .collect();
        let rels: Vec<Relation> = self
            .relations
            .iter()
            .map(|r| Relation {
                from: self.interner.lookup(r.from).to_string(),
                to: self.interner.lookup(r.to).to_string(),
                relation_type: self.interner.lookup(r.relation_type).to_string(),
            })
            .collect();
        KnowledgeGraphOut { entities, relations: rels }
    }

    pub fn search_nodes(&self, query: &str) -> KnowledgeGraphOut {
        let matched = self.search.search(query, &self.interner);
        let entities: Vec<Entity> = matched
            .iter()
            .filter_map(|&slot| {
                self.entity_slots
                    .get(slot as usize)?
                    .as_ref()
                    .filter(|e| e.is_live())
                    .map(|stored| self.entity_to_output(stored))
            })
            .collect();
        let entity_names: HashSet<StrId> = entities.iter()
            .filter_map(|e| self.interner.get_optional(&e.name))
            .collect();
        let rels: Vec<Relation> = self
            .relations
            .iter()
            .filter(|r| entity_names.contains(&r.from) || entity_names.contains(&r.to))
            .map(|r| Relation {
                from: self.interner.lookup(r.from).to_string(),
                to: self.interner.lookup(r.to).to_string(),
                relation_type: self.interner.lookup(r.relation_type).to_string(),
            })
            .collect();
        KnowledgeGraphOut { entities, relations: rels }
    }

    pub fn open_nodes(&self, names: &[String]) -> KnowledgeGraphOut {
        let name_ids: HashSet<StrId> = names.iter()
            .filter_map(|n| self.interner.get_optional(n))
            .collect();
        let entities: Vec<Entity> = self
            .entity_slots
            .iter()
            .filter_map(|s| {
                s.as_ref().and_then(|stored| {
                    if stored.is_live() && name_ids.contains(&stored.name) {
                        Some(self.entity_to_output(stored))
                    } else {
                        None
                    }
                })
            })
            .collect();
        let matched_names: HashSet<StrId> = entities.iter()
            .filter_map(|e| self.interner.get_optional(&e.name))
            .collect();
        let rels: Vec<Relation> = self
            .relations
            .iter()
            .filter(|r| matched_names.contains(&r.from) || matched_names.contains(&r.to))
            .map(|r| Relation {
                from: self.interner.lookup(r.from).to_string(),
                to: self.interner.lookup(r.to).to_string(),
                relation_type: self.interner.lookup(r.relation_type).to_string(),
            })
            .collect();
        KnowledgeGraphOut { entities, relations: rels }
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

    // --- Flush & sync ---

    /// Flush and fsync the log to stable storage.
    pub fn flush_and_sync(&mut self) -> Result<()> {
        self.store.flush_and_sync().map_err(MCSError::IoError)
    }
}
