use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use petgraph::Directed;
use rusqlite::{params, Connection};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::errors::{MCSError, Result};
use crate::kg::push_json_str;

pub type EntityId = i64;

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct BlobHeader {
    dims: u32,
}

/// Tunable parameters for the usearch HNSW index. Built from CLI flags in the
/// `mcp-memory-vec` binary; [`VectorConfig::new`] supplies the defaults used by
/// tests and any caller that only cares about the embedding dimension.
#[derive(Clone, Copy, Debug)]
pub struct VectorConfig {
    /// Embedding dimension. All upserted/queried vectors must match this.
    pub dims: u32,
    /// Distance metric used by the index.
    pub metric: MetricKind,
    /// On-disk/in-index scalar representation (enables quantization).
    pub quantization: ScalarKind,
    /// HNSW graph degree (`M`). Higher = better recall, more memory.
    pub connectivity: usize,
    /// HNSW `efConstruction`. Higher = better index quality, slower inserts.
    pub expansion_add: usize,
    /// HNSW `efSearch`. Higher = better recall, slower queries.
    pub expansion_search: usize,
}

impl VectorConfig {
    /// Default HNSW configuration for the given embedding dimension.
    pub const fn new(dims: u32) -> Self {
        Self {
            dims,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: 16,
            expansion_add: 200,
            expansion_search: 50,
        }
    }
}

pub struct VectorStore {
    pub name_to_id: Arc<DashMap<String, EntityId>>,
    pub id_to_name: Arc<DashMap<EntityId, String>>,

    pub(crate) graph: Arc<RwLock<StableGraph<EntityId, (), Directed, u32>>>,
    pub(crate) node_map: Arc<DashMap<EntityId, NodeIndex<u32>>>,

    pub index: Arc<Index>,
    pub(crate) db: Mutex<Connection>,

    pub dims: u32,
    pub count: AtomicUsize,

    pub db_path: std::path::PathBuf,
}

fn sqlite_err(e: rusqlite::Error) -> MCSError {
    MCSError::IoError(std::io::Error::other(e))
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

pub fn with_scratch<R>(f: impl FnOnce(&mut Vec<f32>) -> R) -> R {
    SCRATCH.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        f(&mut buf)
    })
}

fn serialize_embedding(emb: &[f32]) -> Vec<u8> {
    let header = BlobHeader {
        dims: emb.len() as u32,
    };
    let f32_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(emb.as_ptr() as *const u8, emb.len() * 4)
    };
    let mut bytes = Vec::with_capacity(4 + f32_bytes.len());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(f32_bytes);
    bytes
}

fn parse_embedding_blob(blob: &[u8]) -> Result<&[f32]> {
    let (header, rest) = BlobHeader::ref_from_prefix(blob)
        .map_err(|_| MCSError::MemoryError("Invalid blob header".into()))?;
    let count = header.dims as usize;
    let bytes = rest
        .get(..count * 4)
        .ok_or_else(|| MCSError::MemoryError("Blob data too short".into()))?;
    let emb = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, count) };
    Ok(emb)
}

impl VectorStore {
    /// Open a store with the default HNSW configuration for `dims`.
    pub fn new(db_path: &Path, dims: u32) -> Result<Self> {
        Self::with_config(db_path, &VectorConfig::new(dims))
    }

    /// Open a store with an explicit HNSW configuration.
    pub fn with_config(db_path: &Path, cfg: &VectorConfig) -> Result<Self> {
        let dims = cfg.dims;
        let conn = Connection::open(db_path).map_err(sqlite_err)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))
            .map_err(sqlite_err)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA temp_store = MEMORY;
             CREATE TABLE IF NOT EXISTS vector_embedding (
                 entity_id INTEGER PRIMARY KEY,
                 dims      INTEGER NOT NULL,
                 blob      BLOB    NOT NULL,
                 model     TEXT    NOT NULL DEFAULT '',
                 created_us INTEGER NOT NULL
             );",
        )
        .map_err(sqlite_err)?;

        let index_opts = IndexOptions {
            dimensions: dims as usize,
            metric: cfg.metric,
            quantization: cfg.quantization,
            connectivity: cfg.connectivity,
            expansion_add: cfg.expansion_add,
            expansion_search: cfg.expansion_search,
            multi: false,
        };
        let index = Index::new(&index_opts)
            .map_err(|e| MCSError::MemoryError(format!("usearch init: {e}")))?;
        let index = Arc::new(index);

        let name_to_id = Arc::new(DashMap::new());
        let id_to_name = Arc::new(DashMap::new());
        let graph = Arc::new(RwLock::new(StableGraph::<EntityId, (), Directed, u32>::new()));
        let node_map = Arc::new(DashMap::new());
        let db = Mutex::new(conn);

        let store = Self {
            name_to_id,
            id_to_name,
            graph,
            node_map,
            index,
            db,
            dims,
            count: AtomicUsize::new(0),
            db_path: db_path.to_path_buf(),
        };
        store.load_existing()?;

        Ok(store)
    }

    fn load_existing(&self) -> Result<()> {
        let conn = self.db.lock();
        let count: usize = conn
            .query_row("SELECT COUNT(*) FROM vector_embedding", [], |r| {
                r.get::<_, i64>(0)
            })
            .map_err(sqlite_err)?
            as usize;

        if count == 0 {
            return Ok(());
        }

        self.index
            .reserve_capacity_and_threads(count, 1)
            .map_err(|e| MCSError::MemoryError(format!("usearch reserve: {e}")))?;

        let mut stmt = conn
            .prepare("SELECT entity_id, dims, blob, model FROM vector_embedding")
            .map_err(sqlite_err)?;

        let rows = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let dims: i64 = row.get(1)?;
                let blob: Vec<u8> = row.get(2)?;
                let model: String = row.get(3)?;
                Ok((id, dims, blob, model))
            })
            .map_err(sqlite_err)?;

        for row in rows {
            let (id, _row_dims, blob, _model) = row.map_err(sqlite_err)?;
            let emb = parse_embedding_blob(&blob)?;
            self.index
                .add(id as u64, emb)
                .map_err(|e| MCSError::MemoryError(format!("usearch add: {e}")))?;
            self.count.fetch_add(1, Ordering::Relaxed);
        }

        if count > 0 {
            self.load_names_from_entity_table(&conn)?;
        }
        Ok(())
    }

    fn load_names_from_entity_table(&self, conn: &Connection) -> Result<()> {
        let mut stmt = conn
            .prepare("SELECT id, name FROM entity WHERE flags = 0")
            .map_err(sqlite_err)?;
        let rows = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let name: String = row.get(1)?;
                Ok((id, name))
            })
            .map_err(sqlite_err)?;

        self.name_to_id.clear();
        self.id_to_name.clear();

        for row in rows {
            let (id, name) = row.map_err(sqlite_err)?;
            self.name_to_id.insert(name.clone(), id);
            self.id_to_name.insert(id, name);
        }
        Ok(())
    }

    fn get_entity_id_and_name(&self, conn: &Connection, entity_name: &str) -> Result<Option<(EntityId, String)>> {
        if let Some(entry) = self.name_to_id.get(entity_name) {
            let id = *entry;
            let name = entity_name.to_string();
            return Ok(Some((id, name)));
        }
        let h = crate::kg::name_hash(entity_name);
        let mut stmt = conn
            .prepare_cached(
                "SELECT id, name FROM entity WHERE name_hash = ?1 AND name = ?2 AND flags = 0",
            )
            .map_err(sqlite_err)?;
        match stmt.query_row(params![h, entity_name], |row| {
            let id: i64 = row.get(0)?;
            let name: String = row.get(1)?;
            Ok((id, name))
        }) {
            Ok(tup) => {
                self.name_to_id.insert(tup.1.clone(), tup.0);
                self.id_to_name.insert(tup.0, tup.1.clone());
                Ok(Some(tup))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(sqlite_err(e)),
        }
    }

    pub fn upsert_embedding(&self, entity_name: &str, embedding: &[f32], model: &str) -> Result<()> {
        if embedding.len() != self.dims as usize {
            return Err(MCSError::InvalidParams(format!(
                "Embedding dimension mismatch: got {}, expected {}",
                embedding.len(),
                self.dims
            )));
        }

        let conn = self.db.lock();
        let entity = self
            .get_entity_id_and_name(&conn, entity_name)?
            .ok_or_else(|| {
                MCSError::InvalidParams(format!("Entity '{entity_name}' not found in KG"))
            })?;
        let entity_id = entity.0;

        let total = self.count.load(Ordering::Relaxed);
        self.index
            .reserve_capacity_and_threads(total.saturating_add(1), 1)
            .map_err(|e| MCSError::MemoryError(format!("usearch reserve: {e}")))?;
        let existed = self
            .index
            .remove(entity_id as u64)
            .unwrap_or(0) > 0;
        self.index
            .add(entity_id as u64, embedding)
            .map_err(|e| MCSError::MemoryError(format!("usearch add: {e}")))?;

        self.name_to_id
            .insert(entity_name.to_string(), entity_id);
        self.id_to_name.insert(entity_id, entity_name.to_string());

        let blob = serialize_embedding(embedding);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as i64;

        conn.execute(
            "INSERT OR REPLACE INTO vector_embedding (entity_id, dims, blob, model, created_us) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![entity_id, self.dims, blob, model, now],
        )
        .map_err(sqlite_err)?;

        if !existed {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    pub fn delete_embedding(&self, entity_name: &str) -> Result<bool> {
        let conn = self.db.lock();
        let entity_id = match self.name_to_id.get(entity_name) {
            Some(entry) => *entry,
            None => {
                return Ok(false);
            }
        };

        self.index
            .remove(entity_id as u64)
            .map_err(|e| MCSError::MemoryError(format!("usearch remove: {e}")))?;

        self.name_to_id.remove(entity_name);
        self.id_to_name.remove(&entity_id);

        conn.execute(
            "DELETE FROM vector_embedding WHERE entity_id = ?1",
            params![entity_id],
        )
        .map_err(sqlite_err)?;

        {
            let mut g = self.graph.write();
            if let Some(nx) = self.node_map.get(&entity_id) {
                g.remove_node(*nx);
                self.node_map.remove(&entity_id);
            }
        }

        self.count.fetch_sub(1, Ordering::Relaxed);
        Ok(true)
    }

    pub fn search_embeddings(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(EntityId, f32)>> {
        if self.count.load(Ordering::Relaxed) == 0 {
            return Ok(Vec::new());
        }
        let top_k = top_k.clamp(1, 100);
        let matches = self
            .index
            .search(query, top_k)
            .map_err(|e| MCSError::MemoryError(format!("usearch search: {e}")))?;

        let cap = matches.keys.len().min(matches.distances.len());
        let mut results = Vec::with_capacity(cap);
        for i in 0..cap {
            let id = matches.keys[i] as EntityId;
            let dist = matches.distances[i];
            results.push((id, dist));
        }
        Ok(results)
    }

    pub fn search_entities_json(
        &self,
        query: &[f32],
        top_k: usize,
        entity_type_filter: Option<&str>,
    ) -> Result<String> {
        let results = self.search_embeddings(query, top_k)?;
        if results.is_empty() {
            return Ok(r#"{"results":[],"count":0}"#.to_string());
        }

        let conn = self.db.lock();
        let mut out = String::with_capacity(128 + results.len() * 64);
        out.push_str(r#"{"results":["#);
        let mut first = true;
        let mut actual_count = 0usize;

        for &(id, dist) in &results {
            let name = self
                .id_to_name
                .get(&id)
                .map(|r| r.value().clone())
                .or_else(|| {
                    conn.query_row(
                        "SELECT name FROM entity WHERE id = ?1 AND flags = 0",
                        params![id],
                        |row| row.get::<_, String>(0),
                    )
                    .ok()
                });

            let name = match name {
                Some(n) => n,
                None => continue,
            };

            if let Some(filter_type) = entity_type_filter {
                let actual_type: Option<String> = conn
                    .query_row(
                        "SELECT t.name FROM entity e JOIN type_dict t ON t.id = e.type_id WHERE e.id = ?1 AND e.flags = 0",
                        params![id],
                        |row| row.get(0),
                    )
                    .ok();
                match actual_type {
                    Some(t) if t == filter_type => {}
                    _ => continue,
                }
            }

            if !first {
                out.push(',');
            }
            first = false;

            let etype: String = conn
                .query_row(
                    "SELECT t.name FROM entity e JOIN type_dict t ON t.id = e.type_id WHERE e.id = ?1 AND e.flags = 0",
                    params![id],
                    |row| row.get(0),
                )
                .unwrap_or_default();

            out.push_str(r#"{"name":"#);
            push_json_str(&mut out, &name);
            out.push_str(r#","entityType":"#);
            push_json_str(&mut out, &etype);
            write_f32(&mut out, dist);
            out.push('}');
            actual_count += 1;
        }

        out.push_str(r#"],"count":"#);
        out.push_str(&actual_count.to_string());
        out.push('}');
        Ok(out)
    }

    pub fn build_search_response_json(&self, results: &[(EntityId, f32)]) -> String {
        let mut out = String::with_capacity(128 + results.len() * 64);
        out.push_str(r#"{"results":["#);
        for (i, &(id, dist)) in results.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(r#"{"entityId":"#);
            out.push_str(&id.to_string());
            out.push_str(r#","distance":"#);
            write_f32(&mut out, dist);
            out.push('}');
        }
        out.push_str(r#"],"count":"#);
        out.push_str(&results.len().to_string());
        out.push('}');
        out
    }

    pub fn rebuild_graph_cache(&self) -> Result<()> {
        let conn = self.db.lock();

        let mut ent_stmt = conn
            .prepare("SELECT entity_id FROM vector_embedding")
            .map_err(sqlite_err)?;
        let ids: Vec<EntityId> = ent_stmt
            .query_map([], |r| r.get::<_, i64>(0))
            .map_err(sqlite_err)?
            .filter_map(|r| r.ok())
            .collect();

        let mut g = StableGraph::<EntityId, (), Directed, u32>::with_capacity(ids.len(), 0);
        let nm = DashMap::new();

        for &id in &ids {
            let nx = g.add_node(id);
            nm.insert(id, nx);
        }

        if !ids.is_empty() {
            let placeholders: Vec<String> = ids.iter().map(|_| "?".to_string()).collect();
            let sql = format!(
                "SELECT from_id, to_id FROM relation WHERE from_id IN ({}) AND to_id IN ({})",
                placeholders.join(","),
                placeholders.join(",")
            );
            let mut rel_stmt = conn.prepare(&sql).map_err(sqlite_err)?;

            let mut param_values: Vec<&dyn rusqlite::types::ToSql> = Vec::with_capacity(ids.len() * 2);
            for id in &ids {
                param_values.push(id as &dyn rusqlite::types::ToSql);
            }
            for id in &ids {
                param_values.push(id as &dyn rusqlite::types::ToSql);
            }

            let rel_rows = rel_stmt
                .query_map(param_values.as_slice(), |row| {
                    let from: i64 = row.get(0)?;
                    let to: i64 = row.get(1)?;
                    Ok((from, to))
                })
                .map_err(sqlite_err)?;

            for rel in rel_rows {
                let (from, to) = rel.map_err(sqlite_err)?;
                if let (Some(f_nx), Some(t_nx)) = (nm.get(&from), nm.get(&to))
                    && g.find_edge(*f_nx, *t_nx).is_none()
                {
                    g.add_edge(*f_nx, *t_nx, ());
                }
            }
        }

        *self.graph.write() = g;
        self.node_map.clear();
        for entry in nm.iter() {
            self.node_map.insert(*entry.key(), *entry.value());
        }

        Ok(())
    }

    pub fn graph_node_count(&self) -> usize {
        self.node_map.len()
    }

    pub fn graph_edge_count(&self) -> usize {
        self.graph.read().edge_count()
    }

    pub fn get_entity_type(&self, entity_id: EntityId) -> Result<Option<String>> {
        let conn = self.db.lock();
        let etype = conn
            .query_row(
                "SELECT t.name FROM entity e JOIN type_dict t ON t.id = e.type_id WHERE e.id = ?1 AND e.flags = 0",
                params![entity_id],
                |row| row.get(0),
            )
            .ok();
        Ok(etype)
    }

    pub fn count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    pub const fn dims(&self) -> u32 {
        self.dims
    }

    pub fn name_to_id(&self) -> &DashMap<String, EntityId> {
        &self.name_to_id
    }

    pub fn id_to_name(&self) -> &DashMap<EntityId, String> {
        &self.id_to_name
    }
}

fn write_f32(buf: &mut String, val: f32) {
    use std::fmt::Write;
    write!(buf, r#","score":{:.6}"#, val).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kg::GraphHandle;
    use crate::config::Durability;
    use crate::types::Entity;
    use std::num::NonZeroUsize;

    struct TestEnv {
        kg: GraphHandle,
        vs: VectorStore,
        _dir: tempfile::TempDir,
    }

    fn setup(dims: u32) -> TestEnv {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("test.db");
        let lru = NonZeroUsize::new(10000).unwrap();
        let kg = GraphHandle::new(&db_path, Durability::Async, 268435456, lru, 4).unwrap();
        let vs = VectorStore::new(&db_path, dims).unwrap();
        TestEnv {
            kg,
            vs,
            _dir: dir,
        }
    }

    fn create_test_entity(kg: &GraphHandle, name: &str, etype: &str) {
        kg.create_entities(&[Entity {
            name: name.into(),
            entity_type: etype.into(),
            observations: vec!["test observation".into()],
        }])
        .unwrap();
    }

    fn make_embedding(dims: u32, value: f32) -> Vec<f32> {
        vec![value; dims as usize]
    }

    #[test]
    fn test_vector_upsert_and_search() {
        let env = setup(4);
        create_test_entity(&env.kg, "alice", "person");
        create_test_entity(&env.kg, "bob", "person");

        let emb_a = make_embedding(4, 1.0);
        let emb_b = make_embedding(4, 0.1);
        env.vs.upsert_embedding("alice", &emb_a, "test-model").unwrap();
        env.vs.upsert_embedding("bob", &emb_b, "test-model").unwrap();

        let query = make_embedding(4, 1.0);
        let results = env.vs.search_embeddings(&query, 10).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].1 < results[1].1);
    }

    #[test]
    fn test_vector_delete_embedding() {
        let env = setup(4);
        create_test_entity(&env.kg, "alice", "person");
        env.vs.upsert_embedding("alice", &make_embedding(4, 1.0), "").unwrap();
        assert_eq!(env.vs.count(), 1);

        let deleted = env.vs.delete_embedding("alice").unwrap();
        assert!(deleted);
        assert_eq!(env.vs.count(), 0);

        let results = env.vs.search_embeddings(&make_embedding(4, 1.0), 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_vector_upsert_nonexistent_entity() {
        let env = setup(4);
        let err = env.vs.upsert_embedding("nonexistent", &make_embedding(4, 1.0), "");
        assert!(err.is_err());
    }

    #[test]
    fn test_vector_dimension_mismatch() {
        let env = setup(4);
        create_test_entity(&env.kg, "alice", "person");
        let err = env.vs.upsert_embedding("alice", &make_embedding(8, 1.0), "");
        assert!(err.is_err());
    }

    #[test]
    fn test_vector_search_top_k() {
        let env = setup(4);
        for i in 0..5 {
            create_test_entity(&env.kg, &format!("e{i}"), "test");
            env.vs.upsert_embedding(&format!("e{i}"), &make_embedding(4, i as f32 * 0.2), "")
                .unwrap();
        }
        let results = env.vs.search_embeddings(&make_embedding(4, 0.0), 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_vector_search_type_filter() {
        let env = setup(4);
        create_test_entity(&env.kg, "alice", "person");
        create_test_entity(&env.kg, "acme", "organization");
        env.vs.upsert_embedding("alice", &make_embedding(4, 1.0), "").unwrap();
        env.vs.upsert_embedding("acme", &make_embedding(4, 0.95), "").unwrap();

        let json = env.vs.search_entities_json(&make_embedding(4, 1.0), 10, Some("person")).unwrap();
        assert!(json.contains("alice"));
        assert!(!json.contains("acme"));
    }

    #[test]
    fn test_vector_blob_roundtrip() {
        let emb: Vec<f32> = vec![1.0, 2.5, -3.0, 0.0];
        let blob = serialize_embedding(&emb);
        let parsed = parse_embedding_blob(&blob).unwrap();
        assert_eq!(parsed.len(), emb.len());
        for (a, b) in parsed.iter().zip(emb.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_vector_scratch_buffer() {
        with_scratch(|buf| {
            buf.push(1.0);
            buf.push(2.0);
            assert_eq!(buf.len(), 2);
        });
        with_scratch(|buf| {
            assert!(buf.is_empty());
            buf.extend_from_slice(&[3.0, 4.0, 5.0]);
            assert_eq!(buf.len(), 3);
        });
    }

    #[test]
    fn test_vector_rebuild_graph_cache() {
        let env = setup(4);
        create_test_entity(&env.kg, "alice", "person");
        create_test_entity(&env.kg, "bob", "person");
        create_test_entity(&env.kg, "charlie", "person");

        env.vs.upsert_embedding("alice", &make_embedding(4, 1.0), "").unwrap();
        env.vs.upsert_embedding("bob", &make_embedding(4, 0.5), "").unwrap();
        env.vs.upsert_embedding("charlie", &make_embedding(4, 0.0), "").unwrap();

        env.kg
            .create_relations(&[crate::types::Relation {
                from: "alice".into(),
                to: "bob".into(),
                relation_type: "knows".into(),
            }])
            .unwrap();

        env.vs.rebuild_graph_cache().unwrap();
        assert_eq!(env.vs.graph_node_count(), 3);
        assert_eq!(env.vs.graph_edge_count(), 1);
    }

    #[test]
    fn test_vector_upsert_replace() {
        let env = setup(4);
        create_test_entity(&env.kg, "alice", "person");
        env.vs.upsert_embedding("alice", &make_embedding(4, 1.0), "").unwrap();
        env.vs.upsert_embedding("alice", &make_embedding(4, 0.5), "").unwrap();
        assert_eq!(env.vs.count(), 1);

        let results = env.vs.search_embeddings(&make_embedding(4, 0.5), 10).unwrap();
        assert_eq!(results.len(), 1);
        let name = env.vs.id_to_name.get(&results[0].0).map(|r| r.value().clone());
        assert_eq!(name.as_deref(), Some("alice"));
    }

    #[test]
    fn test_vector_empty_store_search() {
        let env = setup(4);
        let json = env.vs.search_entities_json(&make_embedding(4, 1.0), 10, None).unwrap();
        assert_eq!(json, r#"{"results":[],"count":0}"#);
    }

    #[test]
    fn test_vector_persistence_across_reopen() {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("persist.db");
        let lru = NonZeroUsize::new(10000).unwrap();

        let kg = GraphHandle::new(&db_path, Durability::Async, 268435456, lru, 4).unwrap();
        kg.create_entities(&[Entity {
            name: "alice".into(),
            entity_type: "person".into(),
            observations: vec![],
        }])
        .unwrap();

        let vs1 = VectorStore::new(&db_path, 4).unwrap();
        vs1.upsert_embedding("alice", &make_embedding(4, 1.0), "").unwrap();
        assert_eq!(vs1.count(), 1);
        drop(vs1);
        drop(kg);

        let kg2 = GraphHandle::new(&db_path, Durability::Async, 268435456, lru, 4).unwrap();
        let vs2 = VectorStore::new(&db_path, 4).unwrap();
        assert_eq!(vs2.count(), 1);

        let results = vs2.search_embeddings(&make_embedding(4, 1.0), 10).unwrap();
        assert_eq!(results.len(), 1);
        drop(vs2);
        drop(kg2);
    }

    #[test]
    fn test_vector_search_json_format() {
        let env = setup(4);
        create_test_entity(&env.kg, "alice", "person");
        env.vs.upsert_embedding("alice", &make_embedding(4, 1.0), "").unwrap();

        let json = env.vs.search_entities_json(&make_embedding(4, 1.0), 10, None).unwrap();
        assert!(json.contains("alice"));
        assert!(json.contains("person"));
        assert!(json.contains("score"));
        assert!(json.contains("count"));
    }

    #[test]
    fn test_vector_concurrent_upsert() {
        let env = setup(8);
        let vs = Arc::new(env.vs);

        let mut threads = Vec::new();
        for i in 0..4 {
            let vs = Arc::clone(&vs);
            threads.push(std::thread::spawn(move || {
                let name = format!("thread_{i}");
                // entity creation happens through GraphHandle - shared
                vs.upsert_embedding(&name, &make_embedding(8, i as f32 * 0.25), "")
                    .ok();
            }));
        }

        create_test_entity(&env.kg, "thread_0", "t");
        create_test_entity(&env.kg, "thread_1", "t");
        create_test_entity(&env.kg, "thread_2", "t");
        create_test_entity(&env.kg, "thread_3", "t");

        for t in threads {
            t.join().unwrap();
        }
    }
}
