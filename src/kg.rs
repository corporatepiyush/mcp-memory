use rustc_hash::FxHashMap;
use std::collections::{HashSet, VecDeque};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use lru::LruCache;
use parking_lot::Mutex;
use rusqlite::{params, types::ToSql, Connection};

use crate::config::Durability;
use crate::errors::{MCSError, Result};
use crate::types::{Entity, Relation};

fn sqlite_err(e: rusqlite::Error) -> MCSError {
    MCSError::IoError(std::io::Error::other(e.to_string()))
}

fn is_not_found(e: &rusqlite::Error) -> bool {
    matches!(e, rusqlite::Error::QueryReturnedNoRows)
}

#[inline(always)]
fn now_us() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as i64
}

#[inline(always)]
pub(crate) fn name_hash(name: &str) -> i64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h as i64
}

fn load_observations(conn: &Connection, entity_id: i64) -> Result<Vec<String>> {
    let mut stmt = conn
        .prepare_cached("SELECT body FROM observation WHERE entity_id = ?1 ORDER BY idx")
        .map_err(sqlite_err)?;
    let rows = stmt
        .query_map(params![entity_id], |row| row.get::<_, String>(0))
        .map_err(sqlite_err)?
        .filter_map(|r| r.ok())
        .collect::<Vec<_>>();
    Ok(rows)
}

fn load_observations_opt(conn: &Connection, entity_id: i64) -> Vec<String> {
    load_observations(conn, entity_id).unwrap_or_default()
}

fn entity_name_lookup(conn: &Connection, name: &str) -> Result<Option<i64>> {
    let h = name_hash(name);
    let mut stmt = conn
        .prepare_cached(
            "SELECT id FROM entity WHERE name_hash = ?1 AND name = ?2 AND flags = 0",
        )
        .map_err(sqlite_err)?;
    match stmt.query_row(params![h, name], |row| row.get::<_, i64>(0)) {
        Ok(id) => Ok(Some(id)),
        Err(e) if is_not_found(&e) => Ok(None),
        Err(e) => Err(sqlite_err(e)),
    }
}

fn get_type_id(conn: &Connection, type_name: &str, kind: i64) -> Result<i64> {
    let mut sel = conn
        .prepare_cached("SELECT id FROM type_dict WHERE kind = ?1 AND name = ?2")
        .map_err(sqlite_err)?;
    if let Ok(id) = sel.query_row(params![kind, type_name], |row| row.get::<_, i64>(0)) {
        return Ok(id);
    }
    conn.execute(
        "INSERT INTO type_dict (kind, name, count) VALUES (?1, ?2, 0)",
        params![kind, type_name],
    )
    .map_err(sqlite_err)?;
    Ok(conn.last_insert_rowid())
}

fn inc_type_count(conn: &Connection, type_id: i64, delta: i64) -> Result<()> {
    conn.execute(
        "UPDATE type_dict SET count = count + ?1 WHERE id = ?2",
        params![delta, type_id],
    )
    .map_err(sqlite_err)?;
    Ok(())
}

fn inc_graph_stat(conn: &Connection, key: &str, delta: i64) -> Result<()> {
    conn.execute(
        "UPDATE graph_stat SET value = value + ?1 WHERE key = ?2",
        params![delta, key],
    )
    .map_err(sqlite_err)?;
    Ok(())
}

fn read_graph_stat(conn: &Connection, key: &str) -> Result<i64> {
    conn.query_row(
        "SELECT value FROM graph_stat WHERE key = ?1",
        params![key],
        |row| row.get(0),
    )
    .map_err(sqlite_err)
}

fn name_of_type(conn: &Connection, type_id: i64) -> Result<String> {
    conn.query_row(
        "SELECT name FROM type_dict WHERE id = ?1",
        params![type_id],
        |row| row.get(0),
    )
    .map_err(sqlite_err)
}

fn select_all_types(conn: &Connection, kind: i64) -> Result<Vec<(String, usize)>> {
    let mut stmt = conn
        .prepare_cached(
            "SELECT name, count FROM type_dict WHERE kind = ?1 AND count > 0 ORDER BY count DESC",
        )
        .map_err(sqlite_err)?;
    let rows = stmt
        .query_map(params![kind], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)? as usize,
            ))
        })
        .map_err(sqlite_err)?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}

fn entity_by_id(conn: &Connection, id: i64) -> Result<Entity> {
    let mut stmt = conn
        .prepare_cached(
            "SELECT e.name, t.name,
               COALESCE((SELECT json_group_array(o.body ORDER BY o.idx) FROM observation o WHERE o.entity_id = e.id), '[]')
             FROM entity e JOIN type_dict t ON t.id = e.type_id WHERE e.id = ?1 AND e.flags = 0",
        )
        .map_err(sqlite_err)?;
    let (name, etype, obs_json): (String, String, String) = stmt
        .query_row(params![id], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .map_err(sqlite_err)?;
    let observations: Vec<String> = serde_json::from_str(&obs_json).unwrap_or_default();
    Ok(Entity {
        name,
        entity_type: etype,
        observations,
    })
}

/// Direction of relation traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Outgoing,
    Incoming,
    Both,
}

impl Direction {
    pub fn parse(s: Option<&str>) -> Self {
        match s {
            Some("OUTGOING") => Direction::Outgoing,
            Some("INCOMING") => Direction::Incoming,
            _ => Direction::Both,
        }
    }
}

/// Escape a string for embedding in JSON, writing directly into the given buffer.
/// Avoids allocating a temporary `serde_json::Value` for the JSON-RPC wrapper.
pub fn push_json_str(buf: &mut String, raw: &str) {
    buf.push('"');
    let mut start = 0;
    let bytes = raw.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        let esc: u8 = match b {
            b'"' => b'"',
            b'\\' => b'\\',
            b'\n' => b'n',
            b'\r' => b'r',
            b'\t' => b't',
            0x08 => b'b',
            0x0C => b'f',
            0x00..=0x07 | 0x0B | 0x0E..=0x1F => continue, // escaped below
            _ => continue,
        };
        buf.push_str(&raw[start..i]);
        buf.push('\\');
        buf.push(esc as char);
        start = i + 1;
    }
    // Control chars 0x00-0x1F not handled above: escape as \u00XX
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if b <= 0x07 || b == 0x0B || (b >= 0x0E && b <= 0x1F) {
            buf.push_str(&raw[start..i]);
            write_escape_unicode(buf, b);
            start = i + 1;
        }
    }
    buf.push_str(&raw[start..]);
    buf.push('"');
}

#[inline(never)]
fn write_escape_unicode(buf: &mut String, b: u8) {
    use std::fmt::Write;
    write!(buf, "\\u{:04x}", b).unwrap();
}

// ── MetaCache ────────────────────────────────────────────────────────────

#[derive(Copy, Clone)]
struct EntityMeta {
    id: i64,
    type_id: i64,
    obs_count: i64,
    out_deg: i64,
    in_deg: i64,
}

// ── Transaction guard (RAII rollback on error) ─────────────────────────

struct TxGuard<'a> {
    conn: &'a Connection,
    done: bool,
}

impl<'a> TxGuard<'a> {
    fn begin(conn: &'a Connection) -> Result<Self> {
        conn.execute_batch("BEGIN").map_err(sqlite_err)?;
        Ok(Self { conn, done: false })
    }

    fn commit(mut self) -> Result<()> {
        self.done = true;
        self.conn.execute_batch("COMMIT").map_err(sqlite_err)
    }
}

impl Drop for TxGuard<'_> {
    fn drop(&mut self) {
        if !self.done {
            let _ = self.conn.execute_batch("ROLLBACK");
        }
    }
}

// ── JSON helpers (shared by GraphHandle methods) ─────────────────────────

fn entities_json_raw(conn: &Connection, limit_sql: i64, offset_sql: i64) -> Result<String> {
    conn.prepare_cached(
        "SELECT COALESCE(json_group_array(json_object(
            'name', e.name,
            'entityType', t.name,
            'observations', COALESCE((
                SELECT json_group_array(o.body ORDER BY o.idx)
                FROM observation o WHERE o.entity_id = e.id
            ), json('[]'))
        ) ORDER BY e.id), json('[]'))
        FROM entity e
        JOIN type_dict t ON t.id = e.type_id
        WHERE e.flags = 0
        LIMIT ?1 OFFSET ?2"
    )
    .map_err(sqlite_err)?
    .query_row(params![limit_sql, offset_sql], |row| row.get::<_, String>(0))
    .map_err(sqlite_err)
}

fn relations_json_raw(conn: &Connection) -> Result<String> {
    conn.prepare_cached(
        "SELECT COALESCE(json_group_array(json_object(
            'from', e1.name,
            'to', e2.name,
            'relationType', t.name
        )), json('[]'))
        FROM relation r
        JOIN entity e1 ON e1.id = r.from_id
        JOIN entity e2 ON e2.id = r.to_id
        JOIN type_dict t ON t.id = r.type_id
        WHERE e1.flags = 0 AND e2.flags = 0"
    )
    .map_err(sqlite_err)?
    .query_row([], |row| row.get::<_, String>(0))
    .map_err(sqlite_err)
}

// ── GraphHandle ──────────────────────────────────────────────────────────

pub struct GraphHandle {
    conn: Mutex<Connection>,
    seq_entity: AtomicI64,
    seq_obs: AtomicI64,
    cache: Mutex<LruCache<String, EntityMeta>>,
}

impl GraphHandle {
    pub fn new(path: &Path, durability: Durability, mmap_size: i64, lru_cache_size: NonZeroUsize) -> Result<Self> {
        {
            let tmp = Connection::open(path).map_err(sqlite_err)?;
            tmp.execute_batch("PRAGMA page_size = 16384;")
                .map_err(sqlite_err)?;
            drop(tmp);
        }

        let conn = Connection::open(path).map_err(sqlite_err)?;

        conn.execute_batch(
             "PRAGMA journal_mode = WAL;
             PRAGMA foreign_keys = OFF;
             PRAGMA cache_size    = -50000;
             PRAGMA temp_store    = MEMORY;
             PRAGMA busy_timeout  = 5000;
             PRAGMA synchronous   = NORMAL;
             PRAGMA journal_size_limit = 67108864;

             CREATE TABLE IF NOT EXISTS entity (
                 id          INTEGER PRIMARY KEY,
                 name_hash   INTEGER NOT NULL,
                 name        TEXT    NOT NULL,
                 type_id     INTEGER NOT NULL,
                 obs_count   INTEGER NOT NULL DEFAULT 0,
                 out_deg     INTEGER NOT NULL DEFAULT 0,
                 in_deg      INTEGER NOT NULL DEFAULT 0,
                 created_us  INTEGER NOT NULL,
                 updated_us  INTEGER NOT NULL,
                 flags       INTEGER NOT NULL DEFAULT 0
             ) STRICT;

             CREATE INDEX IF NOT EXISTS entity_by_hash
                 ON entity(name_hash, type_id, obs_count, out_deg, in_deg)
                 WHERE flags = 0;

             CREATE INDEX IF NOT EXISTS entity_name_ci
                 ON entity(lower(name))
                 WHERE flags = 0;

             CREATE TABLE IF NOT EXISTS observation (
                 id          INTEGER PRIMARY KEY,
                 entity_id   INTEGER NOT NULL,
                 idx         INTEGER NOT NULL,
                 body        TEXT    NOT NULL,
                 created_us  INTEGER NOT NULL
             ) STRICT;

             CREATE INDEX IF NOT EXISTS obs_by_entity
                 ON observation(entity_id, idx);

             CREATE TABLE IF NOT EXISTS relation (
                 from_id     INTEGER NOT NULL,
                 to_id       INTEGER NOT NULL,
                 type_id     INTEGER NOT NULL,
                 created_us  INTEGER NOT NULL
             ) STRICT;

             CREATE INDEX IF NOT EXISTS rel_out
                 ON relation(from_id, type_id, to_id);

             CREATE INDEX IF NOT EXISTS rel_in
                 ON relation(to_id, type_id, from_id);

             CREATE VIRTUAL TABLE IF NOT EXISTS name_fts
                 USING fts5(name, content='entity', content_rowid='id',
                            tokenize='unicode61 remove_diacritics 2');

             CREATE VIRTUAL TABLE IF NOT EXISTS obs_fts
                 USING fts5(body, content='observation', content_rowid='id',
                            tokenize='unicode61 remove_diacritics 2');

             CREATE TRIGGER IF NOT EXISTS obs_fts_ai AFTER INSERT ON observation BEGIN
               INSERT INTO obs_fts(rowid, body) VALUES (new.id, new.body);
             END;

             CREATE TRIGGER IF NOT EXISTS obs_fts_bd BEFORE DELETE ON observation BEGIN
               INSERT INTO obs_fts(obs_fts, rowid, body) VALUES ('delete', old.id, '');
             END;

             CREATE TABLE IF NOT EXISTS type_dict (
                 id     INTEGER PRIMARY KEY,
                 kind   INTEGER NOT NULL,
                 name   TEXT    NOT NULL,
                 count  INTEGER NOT NULL DEFAULT 0
             ) STRICT;

             CREATE INDEX IF NOT EXISTS type_by_name
                 ON type_dict(kind, name);

             CREATE TABLE IF NOT EXISTS graph_stat (
                 key    TEXT NOT NULL PRIMARY KEY,
                 value  INTEGER NOT NULL
             ) STRICT, WITHOUT ROWID;

             CREATE TABLE IF NOT EXISTS hub_degree (
                 entity_id INTEGER PRIMARY KEY,
                 out_deg   INTEGER NOT NULL,
                 in_deg    INTEGER NOT NULL
             ) STRICT;

             CREATE TABLE IF NOT EXISTS partition_map (
                 table_name TEXT NOT NULL PRIMARY KEY,
                 role       INTEGER NOT NULL,
                 type_id    INTEGER,
                 row_count  INTEGER NOT NULL DEFAULT 0
             ) STRICT, WITHOUT ROWID;",
        )
        .map_err(sqlite_err)?;

        conn.execute_batch(&format!("PRAGMA mmap_size = {mmap_size};"))
            .map_err(sqlite_err)?;

        let sync_pragma = match durability {
            Durability::Sync => "PRAGMA synchronous = FULL",
            Durability::Async => "PRAGMA synchronous = NORMAL",
        };
        conn.execute_batch(sync_pragma).map_err(sqlite_err)?;

        let has_stat: bool = conn
            .query_row("SELECT 1 FROM graph_stat LIMIT 1", [], |_| Ok(()))
            .is_ok();
        if !has_stat {
            conn.execute_batch(
                "INSERT INTO graph_stat(key, value) VALUES
                 ('entities', 0), ('relations', 0), ('observations', 0),
                 ('entity_seq', 0), ('obs_seq', 0);",
            )
            .map_err(sqlite_err)?;
        }

        conn.execute_batch("PRAGMA optimize;").map_err(sqlite_err)?;

        let seq_entity = read_graph_stat(&conn, "entity_seq").unwrap_or(0);
        let seq_obs = read_graph_stat(&conn, "obs_seq").unwrap_or(0);

        Ok(Self {
            conn: Mutex::new(conn),
            seq_entity: AtomicI64::new(seq_entity),
            seq_obs: AtomicI64::new(seq_obs),
            cache: Mutex::new(LruCache::new(lru_cache_size)),
        })
    }

    fn next_entity_id(&self) -> i64 {
        self.seq_entity.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn next_obs_id(&self) -> i64 {
        self.seq_obs.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn meta_get(&self, name: &str) -> Option<EntityMeta> {
        self.cache.lock().get(name).copied()
    }

    fn meta_set(&self, name: &str, m: EntityMeta) {
        self.cache.lock().put(name.to_string(), m);
    }

    fn meta_remove(&self, name: &str) {
        self.cache.lock().pop(name);
    }

    fn meta_update(&self, name: &str, f: impl FnOnce(&mut EntityMeta)) {
        let mut cache = self.cache.lock();
        if let Some(m) = cache.get_mut(name) {
            f(m);
        }
    }

    fn get_entity_id(&self, conn: &Connection, name: &str) -> Result<Option<(i64, i64, i64, i64)>> {
        if let Some(m) = self.meta_get(name) {
            return Ok(Some((m.id, m.type_id, m.out_deg, m.in_deg)));
        }
        let h = name_hash(name);
        let mut stmt = conn
            .prepare_cached(
                "SELECT id, type_id, obs_count, out_deg, in_deg
                 FROM entity WHERE name_hash = ?1 AND name = ?2 AND flags = 0",
            )
            .map_err(sqlite_err)?;
        match stmt.query_row(params![h, name], |row| {
            Ok(EntityMeta {
                id: row.get(0)?,
                type_id: row.get(1)?,
                obs_count: row.get(2)?,
                out_deg: row.get(3)?,
                in_deg: row.get(4)?,
            })
        }) {
            Ok(m) => {
                self.meta_set(name, m);
                Ok(Some((m.id, m.type_id, m.out_deg, m.in_deg)))
            }
            Err(e) if is_not_found(&e) => Ok(None),
            Err(e) => Err(sqlite_err(e)),
        }
    }

    fn sync_seqs(&self, conn: &Connection) -> Result<()> {
        let seq_e = self.seq_entity.load(Ordering::Relaxed);
        let seq_o = self.seq_obs.load(Ordering::Relaxed);
        conn.execute(
            "UPDATE graph_stat SET value = CASE key WHEN 'entity_seq' THEN ?1 WHEN 'obs_seq' THEN ?2 ELSE value END
             WHERE key IN ('entity_seq', 'obs_seq')",
            params![seq_e, seq_o],
        )
        .map_err(sqlite_err)?;
        Ok(())
    }

    // ── Public API ──────────────────────────────────────────────────────

    pub fn get_entity(&self, name: &str) -> Result<Option<Entity>> {
        if name.is_empty() {
            return Ok(None);
        }

        if let Some(m) = self.meta_get(name) {
            let conn = self.conn.lock();
            let etype = name_of_type(&conn, m.type_id).unwrap_or_default();
            let observations = load_observations_opt(&conn, m.id);
            return Ok(Some(Entity {
                name: name.to_string(),
                entity_type: etype,
                observations,
            }));
        }

        let conn = self.conn.lock();
        let h = name_hash(name);
        let mut stmt = conn
            .prepare_cached(
                "SELECT e.id, e.type_id, e.name, t.name,
                        e.obs_count, e.out_deg, e.in_deg
                 FROM entity e
                 JOIN type_dict t ON t.id = e.type_id
                 WHERE e.name_hash = ?1 AND e.name = ?2 AND e.flags = 0",
            )
            .map_err(sqlite_err)?;
        match stmt.query_row(params![h, name], |row| {
            let id: i64 = row.get(0)?;
            let type_id: i64 = row.get(1)?;
            let ename: String = row.get(2)?;
            let etype: String = row.get(3)?;
            let obs_count: i64 = row.get(4)?;
            let out_deg: i64 = row.get(5)?;
            let in_deg: i64 = row.get(6)?;
            Ok((id, type_id, ename, etype, obs_count, out_deg, in_deg))
        }) {
            Ok((id, type_id, ename, etype, obs_count, out_deg, in_deg)) => {
                let observations = load_observations_opt(&conn, id);
                drop(stmt);
                drop(conn);
                self.meta_set(&ename, EntityMeta { id, type_id, obs_count, out_deg, in_deg });
                Ok(Some(Entity {
                    name: ename,
                    entity_type: etype,
                    observations,
                }))
            }
            Err(e) if is_not_found(&e) => Ok(None),
            Err(e) => Err(sqlite_err(e)),
        }
    }

    pub fn create_entities(&self, entities: &[Entity]) -> Result<Vec<Entity>> {
        let conn = self.conn.lock();
        let tx = TxGuard::begin(&conn)?;

        let mut ins_ent = conn
            .prepare_cached(
                "INSERT INTO entity (id, name_hash, name, type_id, obs_count, out_deg, in_deg, created_us, updated_us, flags)
                 SELECT ?1, ?2, ?3, ?4, ?5, 0, 0, ?6, ?6, 0
                 WHERE NOT EXISTS (SELECT 1 FROM entity WHERE name_hash = ?2 AND name = ?3 AND flags = 0)",
            )
            .map_err(sqlite_err)?;

        let mut ins_fts = conn
            .prepare_cached("INSERT INTO name_fts (rowid, name) VALUES (?1, ?2)")
            .map_err(sqlite_err)?;

        let batch_ts = now_us();
        let mut type_cache: FxHashMap<String, i64> = FxHashMap::default();
        let mut type_deltas: FxHashMap<i64, i64> = FxHashMap::default();
        let mut total_entities: i64 = 0;
        let mut total_obs: i64 = 0;
        let mut created = Vec::new();
        let mut created_metas: Vec<(String, EntityMeta)> = Vec::new();
        let mut obs_sql = String::new();

        for entity in entities {
            if entity.name.is_empty() {
                continue;
            }
            let h = name_hash(&entity.name);
            let id = self.next_entity_id();
            let type_id = match type_cache.get(entity.entity_type.as_str()) {
                Some(t) => *t,
                None => {
                    let t = get_type_id(&conn, &entity.entity_type, 0)?;
                    type_cache.insert(entity.entity_type.clone(), t);
                    t
                }
            };
            let obs_count = entity.observations.len() as i64;

            let changed = ins_ent
                .execute(params![id, h, entity.name, type_id, obs_count, batch_ts])
                .map_err(sqlite_err)?;
            if changed == 0 {
                continue;
            }

            let n = entity.observations.len();
            if n > 0 {
                obs_sql.clear();

                let mut oids = Vec::with_capacity(n);
                let mut idxs = Vec::with_capacity(n);
                for _ in 0..n {
                    oids.push(self.next_obs_id());
                }
                for i in 0..n as i64 {
                    idxs.push(i);
                }

                obs_sql.push_str("INSERT INTO observation (id,entity_id,idx,body,created_us) VALUES");
                for i in 0..n {
                    if i > 0 { obs_sql.push(','); }
                    obs_sql.push_str("(?,?,?,?,?)");
                }

                let mut obs_params: Vec<&dyn ToSql> = Vec::with_capacity(n * 5);
                for (i, obs) in entity.observations.iter().enumerate() {
                    obs_params.push(&oids[i]);
                    obs_params.push(&id);
                    obs_params.push(&idxs[i]);
                    obs_params.push(obs);
                    obs_params.push(&batch_ts);
                }

                conn.execute(&obs_sql, obs_params.as_slice())
                    .map_err(sqlite_err)?;
            }

            ins_fts
                .execute(params![id, entity.name])
                .map_err(sqlite_err)?;

            *type_deltas.entry(type_id).or_insert(0) += 1;
            total_entities += 1;
            total_obs += obs_count;

            created.push(entity.clone());
            created_metas.push((entity.name.clone(), EntityMeta {
                id,
                type_id,
                obs_count,
                out_deg: 0,
                in_deg: 0,
            }));
        }

        if total_entities > 0 {
            for (type_id, delta) in &type_deltas {
                inc_type_count(&conn, *type_id, *delta)?;
            }
            inc_graph_stat(&conn, "entities", total_entities)?;
            inc_graph_stat(&conn, "observations", total_obs)?;
            self.sync_seqs(&conn)?;
        }

        tx.commit()?;

        if total_entities > 0 {
            conn.execute_batch("PRAGMA optimize(0x10000);").map_err(sqlite_err)?;
        }

        if !created_metas.is_empty() {
            let mut cache = self.cache.lock();
            for (name, meta) in &created_metas {
                cache.put(name.clone(), *meta);
            }
        }

        Ok(created)
    }

    pub fn delete_entities(&self, names: &[String]) -> Result<()> {
        if names.is_empty() {
            return Ok(());
        }
        let conn = self.conn.lock();

        // Phase 1: Resolve all names to (id, type_id).
        let mut resolved: Vec<(i64, i64, String)> = Vec::with_capacity(names.len());
        let mut sel = conn
            .prepare_cached(
                "SELECT id, type_id FROM entity WHERE name_hash = ?1 AND name = ?2 AND flags = 0",
            )
            .map_err(sqlite_err)?;
        for name in names {
            let h = name_hash(name);
            let (id, type_id) = match sel.query_row(params![h, name], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
            }) {
                Ok(v) => v,
                Err(e) if is_not_found(&e) => continue,
                Err(e) => return Err(sqlite_err(e)),
            };
            resolved.push((id, type_id, name.clone()));
        }

        if resolved.is_empty() {
            return Ok(());
        }

        let ids: Vec<i64> = resolved.iter().map(|(id, _, _)| *id).collect();
        let n = ids.len();

        // Phase 2: Batch DELETE observations.
        let obs_p: Vec<String> = (0..n).map(|i| format!("?{}", i + 1)).collect();
        let obs_sql = format!(
            "DELETE FROM observation WHERE entity_id IN ({})",
            obs_p.join(",")
        );
        let obs_refs: Vec<&dyn ToSql> = ids.iter().map(|id| id as &dyn ToSql).collect();
        let obs_deleted = conn
            .execute(&obs_sql, obs_refs.as_slice())
            .map_err(sqlite_err)? as i64;

        // Phase 3: Batch DELETE relations.
        let rel_sql = format!(
            "DELETE FROM relation WHERE from_id IN ({}) OR to_id IN ({})",
            obs_p.join(","),
            obs_p.join(",")
        );
        let rel_refs: Vec<&dyn ToSql> = ids.iter().map(|id| id as &dyn ToSql).collect();
        let rel_deleted = conn
            .execute(&rel_sql, rel_refs.as_slice())
            .map_err(sqlite_err)? as i64;

        // Phase 4: Batch FTS deletes.
        let fts_values: Vec<String> = (0..n)
            .map(|_| "('delete', ?, '')".to_string())
            .collect();
        let fts_sql = format!(
            "INSERT INTO name_fts(name_fts, rowid, name) VALUES {}",
            fts_values.join(", ")
        );
        conn.execute(&fts_sql, rusqlite::params_from_iter(&ids))
            .map_err(sqlite_err)?;

        // Aggregate type count deltas.
        let mut type_deltas: FxHashMap<i64, i64> = FxHashMap::default();
        for &(_, type_id, _) in &resolved {
            *type_deltas.entry(type_id).or_insert(0) += 1;
        }

        // Phase 5: Batch type count decrements.
        if !type_deltas.is_empty() {
            let m = type_deltas.len();
            let type_keys: Vec<i64> = type_deltas.keys().cloned().collect();
            let type_vals: Vec<i64> = type_deltas.values().map(|v| -*v).collect();
            let mut case_parts: Vec<String> = Vec::with_capacity(m);
            let mut id_parts: Vec<String> = Vec::with_capacity(m);
            for i in 0..m {
                case_parts.push(format!("WHEN ?{} THEN ?{}", i + 1, m + i + 1));
                id_parts.push(format!("?{}", i + 1));
            }
            let sql = format!(
                "UPDATE type_dict SET count = MAX(0, count + CASE id {} ELSE 0 END) WHERE id IN ({})",
                case_parts.join(" "),
                id_parts.join(","),
            );
            let mut params: Vec<Box<dyn ToSql>> = Vec::with_capacity(2 * m);
            for id in &type_keys {
                params.push(Box::new(*id));
            }
            for delta in &type_vals {
                params.push(Box::new(*delta));
            }
            let param_refs: Vec<&dyn ToSql> = params.iter().map(|p| p.as_ref()).collect();
            conn.execute(&sql, param_refs.as_slice()).map_err(sqlite_err)?;
        }

        // Phase 6: Batch DELETE entities.
        conn.execute(
            &format!("DELETE FROM entity WHERE id IN ({})", obs_p.join(",")),
            ids.iter().map(|id| id as &dyn ToSql).collect::<Vec<_>>().as_slice(),
        )
        .map_err(sqlite_err)?;

        // Phase 7: Update stats.
        inc_graph_stat(&conn, "entities", -(n as i64))?;
        inc_graph_stat(&conn, "observations", -obs_deleted)?;
        inc_graph_stat(&conn, "relations", -rel_deleted)?;

        // Phase 8: Remove from cache.
        for (_, _, name) in &resolved {
            self.meta_remove(name);
        }

        Ok(())
    }

    pub fn create_relations(&self, relations: &[Relation]) -> Result<Vec<Relation>> {
        let conn = self.conn.lock();
        let tx = TxGuard::begin(&conn)?;

        let mut ins = conn
            .prepare_cached(
                "INSERT INTO relation (from_id, to_id, type_id, created_us)
                 SELECT ?1, ?2, ?3, ?4
                 WHERE NOT EXISTS (SELECT 1 FROM relation WHERE from_id = ?1 AND to_id = ?2 AND type_id = ?3)",
            )
            .map_err(sqlite_err)?;

        let ts = now_us();
        let mut type_cache: FxHashMap<String, i64> = FxHashMap::default();
        let mut type_deltas: FxHashMap<i64, i64> = FxHashMap::default();
        let mut out_deltas: FxHashMap<i64, i64> = FxHashMap::default();
        let mut in_deltas: FxHashMap<i64, i64> = FxHashMap::default();
        let mut total_relations: i64 = 0;
        let mut created = Vec::new();

        for rel in relations {
            let (from_id, _, _, _) = match self.get_entity_id(&conn, &rel.from)? {
                Some(v) => v,
                None => continue,
            };
            let (to_id, _, _, _) = match self.get_entity_id(&conn, &rel.to)? {
                Some(v) => v,
                None => continue,
            };
            let type_id = match type_cache.get(rel.relation_type.as_str()) {
                Some(t) => *t,
                None => {
                    let t = get_type_id(&conn, &rel.relation_type, 1)?;
                    type_cache.insert(rel.relation_type.clone(), t);
                    t
                }
            };

            let changed = ins
                .execute(params![from_id, to_id, type_id, ts])
                .map_err(sqlite_err)?;
            if changed == 0 {
                continue;
            }

            *out_deltas.entry(from_id).or_insert(0) += 1;
            *in_deltas.entry(to_id).or_insert(0) += 1;
            *type_deltas.entry(type_id).or_insert(0) += 1;
            total_relations += 1;

            created.push(rel.clone());
        }

        if total_relations > 0 {
            for (id, delta) in &out_deltas {
                conn.execute(
                    "UPDATE entity SET out_deg = out_deg + ?1 WHERE id = ?2",
                    params![delta, id],
                )
                .map_err(sqlite_err)?;
            }
            for (id, delta) in in_deltas {
                conn.execute(
                    "UPDATE entity SET in_deg = in_deg + ?1 WHERE id = ?2",
                    params![delta, id],
                )
                .map_err(sqlite_err)?;
            }
            for (type_id, delta) in &type_deltas {
                inc_type_count(&conn, *type_id, *delta)?;
            }
            inc_graph_stat(&conn, "relations", total_relations)?;
        }

        tx.commit()?;

        if total_relations > 0 {
            conn.execute_batch("PRAGMA optimize(0x10000);").map_err(sqlite_err)?;
        }

        if !created.is_empty() {
            let mut cache = self.cache.lock();
            for rel in &created {
                if let Some(m) = cache.get_mut(&rel.from) {
                    m.out_deg += 1;
                }
                if let Some(m) = cache.get_mut(&rel.to) {
                    m.in_deg += 1;
                }
            }
        }

        Ok(created)
    }

    pub fn delete_relations(&self, relations: &[Relation]) -> Result<()> {
        if relations.is_empty() {
            return Ok(());
        }
        let conn = self.conn.lock();

        // Resolve names to IDs and collect valid triples.
        let mut triples: Vec<(i64, i64, i64)> = Vec::with_capacity(relations.len());
        let mut names: Vec<(String, String)> = Vec::with_capacity(relations.len());
        for rel in relations {
            let (from_id, _, _, _) = match self.get_entity_id(&conn, &rel.from)? {
                Some(v) => v,
                None => continue,
            };
            let (to_id, _, _, _) = match self.get_entity_id(&conn, &rel.to)? {
                Some(v) => v,
                None => continue,
            };
            let type_id = match get_type_id(&conn, &rel.relation_type, 1) {
                Ok(id) => id,
                Err(_) => continue,
            };
            triples.push((from_id, to_id, type_id));
            names.push((rel.from.clone(), rel.to.clone()));
        }

        if triples.is_empty() {
            return Ok(());
        }

        // Batch DELETE using VALUES subquery.
        let mut sql = String::from(
            "DELETE FROM relation WHERE (from_id, to_id, type_id) IN (",
        );
        for (i, _) in triples.iter().enumerate() {
            if i > 0 {
                sql.push_str(", ");
            }
            let base = (i * 3) + 1;
            sql.push_str(&format!("SELECT ?{b}, ?{bp1}, ?{bp2}", b = base, bp1 = base + 1, bp2 = base + 2));
        }
        sql.push(')');

        let mut param_values: Vec<Box<dyn ToSql>> = Vec::with_capacity(triples.len() * 3);
        for &(f, t, tp) in &triples {
            param_values.push(Box::new(f));
            param_values.push(Box::new(t));
            param_values.push(Box::new(tp));
        }
        let param_refs: Vec<&dyn ToSql> = param_values.iter().map(|p| p.as_ref()).collect();
        let total = conn.execute(&sql, param_refs.as_slice()).map_err(sqlite_err)?;
        if total == 0 {
            return Ok(());
        }

        // Aggregate degree and type deltas.
        let mut out_deltas: FxHashMap<i64, i64> = FxHashMap::default();
        let mut in_deltas: FxHashMap<i64, i64> = FxHashMap::default();
        let mut type_deltas: FxHashMap<i64, i64> = FxHashMap::default();
        for &(from_id, to_id, type_id) in &triples {
            *out_deltas.entry(from_id).or_insert(0) += 1;
            *in_deltas.entry(to_id).or_insert(0) += 1;
            *type_deltas.entry(type_id).or_insert(0) += 1;
        }

        // Batch out_deg updates.
        let out_keys: Vec<i64> = out_deltas.keys().cloned().collect();
        let out_vals: Vec<i64> = out_deltas.values().cloned().collect();
        if !out_keys.is_empty() {
            let m = out_keys.len();
            let mut case_parts: Vec<String> = Vec::with_capacity(m);
            let mut id_parts: Vec<String> = Vec::with_capacity(m);
            for i in 0..m {
                case_parts.push(format!("WHEN ?{} THEN ?{}", i + 1, m + i + 1));
                id_parts.push(format!("?{}", i + 1));
            }
            let sql = format!(
                "UPDATE entity SET out_deg = MAX(0, out_deg - CASE id {} ELSE 0 END) WHERE id IN ({})",
                case_parts.join(" "),
                id_parts.join(","),
            );
            let mut params: Vec<Box<dyn ToSql>> = Vec::with_capacity(2 * m);
            for id in &out_keys {
                params.push(Box::new(*id));
            }
            for delta in &out_vals {
                params.push(Box::new(*delta));
            }
            let param_refs: Vec<&dyn ToSql> = params.iter().map(|p| p.as_ref()).collect();
            conn.execute(&sql, param_refs.as_slice()).map_err(sqlite_err)?;
        }

        // Batch in_deg updates.
        let in_keys: Vec<i64> = in_deltas.keys().cloned().collect();
        let in_vals: Vec<i64> = in_deltas.values().cloned().collect();
        if !in_keys.is_empty() {
            let m = in_keys.len();
            let mut case_parts: Vec<String> = Vec::with_capacity(m);
            let mut id_parts: Vec<String> = Vec::with_capacity(m);
            for i in 0..m {
                case_parts.push(format!("WHEN ?{} THEN ?{}", i + 1, m + i + 1));
                id_parts.push(format!("?{}", i + 1));
            }
            let sql = format!(
                "UPDATE entity SET in_deg = MAX(0, in_deg - CASE id {} ELSE 0 END) WHERE id IN ({})",
                case_parts.join(" "),
                id_parts.join(","),
            );
            let mut params: Vec<Box<dyn ToSql>> = Vec::with_capacity(2 * m);
            for id in &in_keys {
                params.push(Box::new(*id));
            }
            for delta in &in_vals {
                params.push(Box::new(*delta));
            }
            let param_refs: Vec<&dyn ToSql> = params.iter().map(|p| p.as_ref()).collect();
            conn.execute(&sql, param_refs.as_slice()).map_err(sqlite_err)?;
        }

        // Batch type_dict updates.
        let type_keys: Vec<i64> = type_deltas.keys().cloned().collect();
        let type_vals: Vec<i64> = type_deltas.values().cloned().collect();
        if !type_keys.is_empty() {
            let m = type_keys.len();
            let mut case_parts: Vec<String> = Vec::with_capacity(m);
            let mut id_parts: Vec<String> = Vec::with_capacity(m);
            for i in 0..m {
                case_parts.push(format!("WHEN ?{} THEN ?{}", i + 1, m + i + 1));
                id_parts.push(format!("?{}", i + 1));
            }
            let sql = format!(
                "UPDATE type_dict SET count = MAX(0, count - CASE id {} ELSE 0 END) WHERE id IN ({})",
                case_parts.join(" "),
                id_parts.join(","),
            );
            let mut params: Vec<Box<dyn ToSql>> = Vec::with_capacity(2 * m);
            for id in &type_keys {
                params.push(Box::new(*id));
            }
            for delta in &type_vals {
                params.push(Box::new(*delta));
            }
            let param_refs: Vec<&dyn ToSql> = params.iter().map(|p| p.as_ref()).collect();
            conn.execute(&sql, param_refs.as_slice()).map_err(sqlite_err)?;
        }

        inc_graph_stat(&conn, "relations", -(total as i64))?;

        // Update cache for resolved triples (self-heals on next reload if
        // a triple happened to not match).
        for (from, to) in &names {
            self.meta_update(from, |m| m.out_deg = m.out_deg.saturating_sub(1));
            self.meta_update(to, |m| m.in_deg = m.in_deg.saturating_sub(1));
        }

        Ok(())
    }

    pub fn add_observations(&self, entity_name: &str, contents: &[String]) -> Result<Vec<String>> {
        let conn = self.conn.lock();
        let (id, _type_id, _, _) = match self.get_entity_id(&conn, entity_name)? {
            Some(v) => v,
            None => {
                return Err(MCSError::InvalidParams(format!(
                    "Entity '{entity_name}' not found"
                )))
            }
        };

        let mut max_idx: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(idx), -1) FROM observation WHERE entity_id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(sqlite_err)?;

        let ts = now_us();
        let mut ins_obs = conn
            .prepare_cached(
                "INSERT INTO observation (id, entity_id, idx, body, created_us) VALUES (?1, ?2, ?3, ?4, ?5)",
            )
            .map_err(sqlite_err)?;

        for content in contents {
            max_idx += 1;
            let oid = self.next_obs_id();
            ins_obs
                .execute(params![oid, id, max_idx, content, ts])
                .map_err(sqlite_err)?;
        }
        let added = contents.to_vec();

        let count: i64 = contents.len() as i64;
        conn.execute(
            "UPDATE entity SET obs_count = obs_count + ?1, updated_us = ?2 WHERE id = ?3",
            params![count, ts, id],
        )
        .map_err(sqlite_err)?;

        inc_graph_stat(&conn, "observations", count)?;
        self.sync_seqs(&conn)?;

        self.meta_update(entity_name, |m| m.obs_count += count);

        Ok(added)
    }

    pub fn delete_observations(&self, entity_name: &str, observations: &[String]) -> Result<()> {
        if observations.is_empty() {
            return Ok(());
        }
        let conn = self.conn.lock();
        let (id, _, _, _) = match self.get_entity_id(&conn, entity_name)? {
            Some(v) => v,
            None => {
                return Err(MCSError::InvalidParams(format!(
                    "Entity '{entity_name}' not found"
                )))
            }
        };

        let placeholders: Vec<String> = (0..observations.len())
            .map(|i| format!("?{}", i + 2))
            .collect();
        let sql = format!(
            "DELETE FROM observation WHERE entity_id = ?1 AND body IN ({})",
            placeholders.join(",")
        );

        let mut param_values: Vec<Box<dyn ToSql>> = Vec::with_capacity(1 + observations.len());
        param_values.push(Box::new(id));
        for obs in observations {
            param_values.push(Box::new(obs.as_str()));
        }
        let param_refs: Vec<&dyn ToSql> = param_values.iter().map(|p| p.as_ref()).collect();
        let removed = conn.execute(&sql, param_refs.as_slice()).map_err(sqlite_err)? as i64;

        if removed > 0 {
            conn.execute(
                "UPDATE entity SET obs_count = MAX(0, obs_count - ?1), updated_us = ?2 WHERE id = ?3",
                params![removed, now_us(), id],
            )
            .map_err(sqlite_err)?;
            inc_graph_stat(&conn, "observations", -removed)?;

            self.meta_update(entity_name, |m| m.obs_count = m.obs_count.saturating_sub(removed));
        }

        Ok(())
    }

    pub fn upsert_entities(&self, entities: &[Entity]) -> Result<Vec<Entity>> {
        let mut results = Vec::new();
        for entity in entities {
            if let Some(existing) = self.get_entity(&entity.name)? {
                // Update type if different.
                if existing.entity_type != entity.entity_type {
                    let conn = self.conn.lock();
                    let old_type_id = conn
                        .query_row(
                            "SELECT type_id FROM entity WHERE name_hash = ?1 AND name = ?2 AND flags = 0",
                            params![name_hash(&entity.name), entity.name],
                            |row| row.get::<_, i64>(0),
                        )
                        .map_err(sqlite_err)?;
                    let new_type_id = get_type_id(&conn, &entity.entity_type, 0)?;
                    inc_type_count(&conn, old_type_id, -1)?;
                    inc_type_count(&conn, new_type_id, 1)?;
                    conn.execute(
                        "UPDATE entity SET type_id = ?1, updated_us = ?2 WHERE name_hash = ?3 AND name = ?4",
                        params![new_type_id, now_us(), name_hash(&entity.name), entity.name],
                    )
                    .map_err(sqlite_err)?;
                    // Invalidate cache so subsequent get_entity reloads meta.
                    self.meta_remove(&entity.name);
                }
                // Merge observations (append new ones not already present).
                let existing_set: HashSet<&str> =
                    existing.observations.iter().map(|s| s.as_str()).collect();
                let to_add: Vec<String> = entity
                    .observations
                    .iter()
                    .filter(|o| !existing_set.contains(o.as_str()))
                    .cloned()
                    .collect();
                if !to_add.is_empty() {
                    self.add_observations(&entity.name, &to_add)?;
                }
                let updated = self
                    .get_entity(&entity.name)?
                    .unwrap_or(entity.clone());
                results.push(updated);
            } else {
                let c = self.create_entities(&[entity.clone()])?;
                if let Some(e) = c.into_iter().next() {
                    results.push(e);
                }
            }
        }
        Ok(results)
    }

    pub fn merge_entities(&self, source: &str, target: &str) -> Result<Entity> {
        let conn = self.conn.lock();
        let (src_id, _, _, _) = match self.get_entity_id(&conn, source)? {
            Some(v) => v,
            None => {
                return Err(MCSError::InvalidParams(format!(
                    "Source entity '{source}' not found"
                )))
            }
        };
        let (tgt_id, _, _, _) = match self.get_entity_id(&conn, target)? {
            Some(v) => v,
            None => {
                return Err(MCSError::InvalidParams(format!(
                    "Target entity '{target}' not found"
                )))
            }
        };

        if src_id == tgt_id {
            return self.get_entity(target)?.ok_or_else(|| {
                MCSError::InvalidParams("Target entity not found after merge".into())
            });
        }

        // Move observations from source to target.
        let mut obs_count: i64 = 0;
        {
            let mut max_idx: i64 = conn
                .query_row(
                    "SELECT COALESCE(MAX(idx), -1) FROM observation WHERE entity_id = ?1",
                    params![tgt_id],
                    |row| row.get(0),
                )
                .map_err(sqlite_err)?;
            let mut sel_obs = conn
                .prepare_cached(
                    "SELECT id, body FROM observation WHERE entity_id = ?1 ORDER BY idx",
                )
                .map_err(sqlite_err)?;
            let mut upd_obs = conn
                .prepare_cached("UPDATE observation SET entity_id = ?1, idx = ?2 WHERE id = ?3")
                .map_err(sqlite_err)?;
            let rows: Vec<(i64, String)> = sel_obs
                .query_map(params![src_id], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(sqlite_err)?
                .filter_map(|r| r.ok())
                .collect();
            for (oid, _body) in &rows {
                max_idx += 1;
                upd_obs
                    .execute(params![tgt_id, max_idx, oid])
                    .map_err(sqlite_err)?;
                obs_count += 1;
            }
        }

        // Move relations from source to target.
        conn.execute(
            "UPDATE OR IGNORE relation SET from_id = ?1 WHERE from_id = ?2",
            params![tgt_id, src_id],
        )
        .map_err(sqlite_err)?;
        conn.execute(
            "UPDATE OR IGNORE relation SET to_id = ?1 WHERE to_id = ?2",
            params![tgt_id, src_id],
        )
        .map_err(sqlite_err)?;
        // Delete orphaned relations that were updated by the above (the "OR IGNORE"
        // keeps the first, but we still have the original row with the old id? No —
        // UPDATE OR IGNORE won't remove. So we must delete any that still reference src_id.)
        conn.execute("DELETE FROM relation WHERE from_id = ?1", params![src_id])
            .map_err(sqlite_err)?;
        conn.execute("DELETE FROM relation WHERE to_id = ?1", params![src_id])
            .map_err(sqlite_err)?;

        // Update degrees on target.
        let out_add: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM relation WHERE from_id = ?1",
                params![tgt_id],
                |row| row.get(0),
            )
            .map_err(sqlite_err)?;
        let in_add: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM relation WHERE to_id = ?1",
                params![tgt_id],
                |row| row.get(0),
            )
            .map_err(sqlite_err)?;
        conn.execute(
            "UPDATE entity SET out_deg = ?1, in_deg = ?2, obs_count = obs_count + ?3, updated_us = ?4 WHERE id = ?5",
            params![out_add, in_add, obs_count, now_us(), tgt_id],
        )
        .map_err(sqlite_err)?;

        // Delete source entity.
        conn.execute(
            "INSERT INTO name_fts(name_fts, rowid, name) VALUES ('delete', ?1, '')",
            params![src_id],
        )
        .map_err(sqlite_err)?;
        conn.execute("DELETE FROM entity WHERE id = ?1", params![src_id])
            .map_err(sqlite_err)?;

        inc_graph_stat(&conn, "entities", -1)?;
        self.meta_remove(source);

        // Reload target into cache.
        if let Ok(meta) = conn.query_row(
            "SELECT id, type_id, obs_count, out_deg, in_deg FROM entity WHERE id = ?1",
            params![tgt_id],
            |row| {
                Ok(EntityMeta {
                    id: row.get(0)?,
                    type_id: row.get(1)?,
                    obs_count: row.get(2)?,
                    out_deg: row.get(3)?,
                    in_deg: row.get(4)?,
                })
            },
        ) {
            self.meta_set(target, meta);
        }

        let (name, etype): (String, String) = conn
            .query_row(
                "SELECT e.name, t.name FROM entity e JOIN type_dict t ON t.id = e.type_id WHERE e.id = ?1",
                params![tgt_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(sqlite_err)?;
        let observations = load_observations_opt(&conn, tgt_id);

        Ok(Entity {
            name,
            entity_type: etype,
            observations,
        })
    }

    pub fn search_nodes_filtered(
        &self,
        query: &str,
        filter_type: Option<&str>,
        offset: usize,
        limit: usize,
    ) -> Vec<Entity> {
        if query.is_empty() {
            return Vec::new();
        }
        let conn = self.conn.lock();

        // Single pass: collect IDs from name_fts then obs_fts, dedup by position.
        let mut entity_ids: Vec<i64> = Vec::new();

        if let Ok(mut stmt) = conn.prepare(
            "SELECT rowid FROM name_fts WHERE name_fts MATCH ?1 ORDER BY rank LIMIT ?2",
        ) {
            let limit_i64 = (limit + offset) as i64;
            if let Ok(rows) = stmt.query_map(params![query, limit_i64], |row| {
                row.get::<_, i64>(0)
            }) {
                for row in rows.flatten() {
                    entity_ids.push(row);
                }
            }
        }

        if let Ok(mut stmt) = conn.prepare(
            "SELECT entity_id FROM obs_fts JOIN observation ON obs_fts.rowid = observation.id
             WHERE obs_fts MATCH ?1
             GROUP BY entity_id
             LIMIT ?2",
        ) {
            let limit_i64 = (limit + offset) as i64;
            if let Ok(rows) = stmt.query_map(params![query, limit_i64], |row| {
                row.get::<_, i64>(0)
            }) {
                for row in rows.flatten() {
                    if !entity_ids.contains(&row) {
                        entity_ids.push(row);
                    }
                }
            }
        }

        // Apply filter_type, offset, limit.
        let mut results = Vec::new();
        let mut count: usize = 0;
        for eid in entity_ids {
            if let Ok(entity) = entity_by_id(&conn, eid) {
                if let Some(ft) = filter_type {
                    if !ft.is_empty() && entity.entity_type != ft {
                        continue;
                    }
                }
                if count < offset {
                    count += 1;
                    continue;
                }
                if results.len() >= limit {
                    break;
                }
                results.push(entity);
                count += 1;
            }
        }

        results
    }

    pub fn read_graph_filtered(
        &self,
        filter_type: Option<&str>,
        offset: usize,
        limit: usize,
    ) -> Result<String> {
        let conn = self.conn.lock();

        let limit_sql: i64 = if limit == usize::MAX {
            -1
        } else {
            limit.min(i64::MAX as usize) as i64
        };
        let offset_sql: i64 = offset as i64;

        let entities_json = if let Some(ft) = filter_type {
            if !ft.is_empty() {
                let mut stmt = conn.prepare_cached(
                    "SELECT COALESCE(json_group_array(json_object(
                        'name', e.name,
                        'entityType', t.name,
                        'observations', COALESCE((
                            SELECT json_group_array(o.body ORDER BY o.idx)
                            FROM observation o WHERE o.entity_id = e.id
                        ), json('[]'))
                    ) ORDER BY e.id), json('[]'))
                    FROM entity e
                    JOIN type_dict t ON t.id = e.type_id
                    WHERE e.type_id = (SELECT id FROM type_dict WHERE kind = 0 AND name = ?1)
                      AND e.flags = 0
                    LIMIT ?2 OFFSET ?3"
                ).map_err(sqlite_err)?;
                stmt.query_row(params![ft, limit_sql, offset_sql], |row| {
                    row.get::<_, String>(0)
                }).unwrap_or_else(|_| "[]".to_string())
            } else {
                entities_json_raw(&conn, limit_sql, offset_sql)?
            }
        } else {
            entities_json_raw(&conn, limit_sql, offset_sql)?
        };

        let relations_json = relations_json_raw(&conn)?;

        let mut out = String::with_capacity(32 + entities_json.len() + relations_json.len());
        out.push_str("{\"entities\":");
        out.push_str(&entities_json);
        out.push_str(",\"relations\":");
        out.push_str(&relations_json);
        out.push('}');
        Ok(out)
    }

    pub fn open_nodes(&self, names: &[String]) -> String {
        let conn = self.conn.lock();
        let mut entity_ids: Vec<i64> = Vec::new();

        for name in names {
            let h = name_hash(name);
            if let Ok(Some(id)) = conn
                .query_row(
                    "SELECT id FROM entity WHERE name_hash = ?1 AND name = ?2 AND flags = 0",
                    params![h, name],
                    |row| row.get::<_, i64>(0),
                )
                .map(Some)
                .or_else(|e| if is_not_found(&e) { Ok(None) } else { Err(sqlite_err(e)) })
            {
                entity_ids.push(id);
            }
        }

        if entity_ids.is_empty() {
            return r#"{"entities":[],"relations":[]}"#.to_string();
        }

        let placeholders: Vec<String> = entity_ids.iter().map(|_| "?".to_string()).collect();
        let ids_str = placeholders.join(",");

        let entities_json: String = {
            let sql = format!(
                "SELECT COALESCE(json_group_array(json_object(
                    'name', e.name,
                    'entityType', t.name,
                    'observations', COALESCE((
                        SELECT json_group_array(o.body ORDER BY o.idx)
                        FROM observation o WHERE o.entity_id = e.id
                    ), json('[]'))
                ) ORDER BY e.id), json('[]'))
                FROM entity e
                JOIN type_dict t ON t.id = e.type_id
                WHERE e.id IN ({ids_str}) AND e.flags = 0"
            );
            conn.query_row(&sql, rusqlite::params_from_iter(&entity_ids), |row| {
                row.get::<_, String>(0)
            })
            .unwrap_or_else(|_| "[]".to_string())
        };

        let relations_json: String = {
            let sql = format!(
                "SELECT COALESCE(json_group_array(json_object(
                    'from', e1.name,
                    'to', e2.name,
                    'relationType', t.name
                )), json('[]'))
                FROM relation r
                JOIN entity e1 ON e1.id = r.from_id
                JOIN entity e2 ON e2.id = r.to_id
                JOIN type_dict t ON t.id = r.type_id
                WHERE (r.from_id IN ({ids_str}) OR r.to_id IN ({ids_str}))
                  AND e1.flags = 0 AND e2.flags = 0"
            );
            let all_params: Vec<&dyn rusqlite::types::ToSql> = entity_ids
                .iter()
                .map(|id| id as &dyn rusqlite::types::ToSql)
                .chain(entity_ids.iter().map(|id| id as &dyn rusqlite::types::ToSql))
                .collect();
            let mut stmt = conn.prepare(&sql).unwrap();
            stmt.query_row(all_params.as_slice(), |row| row.get::<_, String>(0))
                .unwrap_or_else(|_| "[]".to_string())
        };

        let mut out = String::with_capacity(32 + entities_json.len() + relations_json.len());
        out.push_str("{\"entities\":");
        out.push_str(&entities_json);
        out.push_str(",\"relations\":");
        out.push_str(&relations_json);
        out.push('}');
        out
    }

    pub fn entities_exist(&self, names: &[String]) -> Result<Vec<bool>> {
        let conn = self.conn.lock();
        let mut results = Vec::with_capacity(names.len());
        for name in names {
            let h = name_hash(name);
            let exists: bool = conn
                .query_row(
                    "SELECT 1 FROM entity WHERE name_hash = ?1 AND name = ?2 AND flags = 0",
                    params![h, name],
                    |_| Ok(()),
                )
                .is_ok();
            results.push(exists);
        }
        Ok(results)
    }

    pub fn degree(&self, name: &str, direction: Direction) -> Result<usize> {
        let conn = self.conn.lock();
        let (_, _, out_d, in_d) = match self.get_entity_id(&conn, name)? {
            Some(v) => v,
            None => {
                return Err(MCSError::InvalidParams(format!(
                    "Entity '{name}' not found"
                )))
            }
        };
        Ok(match direction {
            Direction::Outgoing => out_d as usize,
            Direction::Incoming => in_d as usize,
            Direction::Both => (out_d + in_d) as usize,
        })
    }

    pub fn get_entity_count(&self) -> Result<usize> {
        let conn = self.conn.lock();
        read_graph_stat(&conn, "entities")
            .map(|v| v as usize)
            .map_err(|_| MCSError::MemoryError("Failed to read entity count".into()))
    }

    pub fn get_relation_count(&self) -> Result<usize> {
        let conn = self.conn.lock();
        read_graph_stat(&conn, "relations")
            .map(|v| v as usize)
            .map_err(|_| MCSError::MemoryError("Failed to read relation count".into()))
    }

    pub fn search_relations(
        &self,
        from: Option<&str>,
        to: Option<&str>,
        rtype: Option<&str>,
    ) -> Vec<Relation> {
        let conn = self.conn.lock();
        let mut results = Vec::new();

        let from_id = from.and_then(|f| {
            if f.is_empty() { None }
            else { entity_name_lookup(&conn, f).ok().flatten() }
        });
        let to_id = to.and_then(|t| {
            if t.is_empty() { None }
            else { entity_name_lookup(&conn, t).ok().flatten() }
        });
        let type_id = rtype.and_then(|rt| {
            if rt.is_empty() { None }
            else { get_type_id(&conn, rt, 1).ok() }
        });

        match (from_id, to_id, type_id) {
            (Some(fid), Some(tid), Some(tpid)) => {
                if let Ok(mut stmt) = conn.prepare_cached(
                    "SELECT e1.name, e2.name, t.name
                     FROM relation r
                     JOIN entity e1 ON e1.id = r.from_id
                     JOIN entity e2 ON e2.id = r.to_id
                     JOIN type_dict t ON t.id = r.type_id
                     WHERE r.from_id = ?1 AND r.to_id = ?2 AND r.type_id = ?3
                       AND e1.flags = 0 AND e2.flags = 0
                     ORDER BY r.from_id, r.to_id"
                ) {
                    if let Ok(rows) = stmt.query_map(params![fid, tid, tpid], |row| {
                        Ok(Relation { from: row.get(0)?, to: row.get(1)?, relation_type: row.get(2)? })
                    }) {
                        for row in rows.flatten() { results.push(row); }
                    }
                }
            }
            (Some(fid), Some(tid), None) => {
                if let Ok(mut stmt) = conn.prepare_cached(
                    "SELECT e1.name, e2.name, t.name
                     FROM relation r
                     JOIN entity e1 ON e1.id = r.from_id
                     JOIN entity e2 ON e2.id = r.to_id
                     JOIN type_dict t ON t.id = r.type_id
                     WHERE r.from_id = ?1 AND r.to_id = ?2
                       AND e1.flags = 0 AND e2.flags = 0
                     ORDER BY r.from_id, r.to_id"
                ) {
                    if let Ok(rows) = stmt.query_map(params![fid, tid], |row| {
                        Ok(Relation { from: row.get(0)?, to: row.get(1)?, relation_type: row.get(2)? })
                    }) {
                        for row in rows.flatten() { results.push(row); }
                    }
                }
            }
            (Some(fid), None, Some(tpid)) => {
                if let Ok(mut stmt) = conn.prepare_cached(
                    "SELECT e1.name, e2.name, t.name
                     FROM relation r
                     JOIN entity e1 ON e1.id = r.from_id
                     JOIN entity e2 ON e2.id = r.to_id
                     JOIN type_dict t ON t.id = r.type_id
                     WHERE r.from_id = ?1 AND r.type_id = ?2
                       AND e1.flags = 0 AND e2.flags = 0
                     ORDER BY r.from_id, r.to_id"
                ) {
                    if let Ok(rows) = stmt.query_map(params![fid, tpid], |row| {
                        Ok(Relation { from: row.get(0)?, to: row.get(1)?, relation_type: row.get(2)? })
                    }) {
                        for row in rows.flatten() { results.push(row); }
                    }
                }
            }
            (None, Some(tid), Some(tpid)) => {
                if let Ok(mut stmt) = conn.prepare_cached(
                    "SELECT e1.name, e2.name, t.name
                     FROM relation r
                     JOIN entity e1 ON e1.id = r.from_id
                     JOIN entity e2 ON e2.id = r.to_id
                     JOIN type_dict t ON t.id = r.type_id
                     WHERE r.to_id = ?1 AND r.type_id = ?2
                       AND e1.flags = 0 AND e2.flags = 0
                     ORDER BY r.from_id, r.to_id"
                ) {
                    if let Ok(rows) = stmt.query_map(params![tid, tpid], |row| {
                        Ok(Relation { from: row.get(0)?, to: row.get(1)?, relation_type: row.get(2)? })
                    }) {
                        for row in rows.flatten() { results.push(row); }
                    }
                }
            }
            (Some(fid), None, None) => {
                if let Ok(mut stmt) = conn.prepare_cached(
                    "SELECT e1.name, e2.name, t.name
                     FROM relation r
                     JOIN entity e1 ON e1.id = r.from_id
                     JOIN entity e2 ON e2.id = r.to_id
                     JOIN type_dict t ON t.id = r.type_id
                     WHERE r.from_id = ?1
                       AND e1.flags = 0 AND e2.flags = 0
                     ORDER BY r.from_id, r.to_id"
                ) {
                    if let Ok(rows) = stmt.query_map(params![fid], |row| {
                        Ok(Relation { from: row.get(0)?, to: row.get(1)?, relation_type: row.get(2)? })
                    }) {
                        for row in rows.flatten() { results.push(row); }
                    }
                }
            }
            (None, Some(tid), None) => {
                if let Ok(mut stmt) = conn.prepare_cached(
                    "SELECT e1.name, e2.name, t.name
                     FROM relation r
                     JOIN entity e1 ON e1.id = r.from_id
                     JOIN entity e2 ON e2.id = r.to_id
                     JOIN type_dict t ON t.id = r.type_id
                     WHERE r.to_id = ?1
                       AND e1.flags = 0 AND e2.flags = 0
                     ORDER BY r.from_id, r.to_id"
                ) {
                    if let Ok(rows) = stmt.query_map(params![tid], |row| {
                        Ok(Relation { from: row.get(0)?, to: row.get(1)?, relation_type: row.get(2)? })
                    }) {
                        for row in rows.flatten() { results.push(row); }
                    }
                }
            }
            (None, None, Some(tpid)) => {
                if let Ok(mut stmt) = conn.prepare_cached(
                    "SELECT e1.name, e2.name, t.name
                     FROM relation r
                     JOIN entity e1 ON e1.id = r.from_id
                     JOIN entity e2 ON e2.id = r.to_id
                     JOIN type_dict t ON t.id = r.type_id
                     WHERE r.type_id = ?1
                       AND e1.flags = 0 AND e2.flags = 0
                     ORDER BY r.from_id, r.to_id"
                ) {
                    if let Ok(rows) = stmt.query_map(params![tpid], |row| {
                        Ok(Relation { from: row.get(0)?, to: row.get(1)?, relation_type: row.get(2)? })
                    }) {
                        for row in rows.flatten() { results.push(row); }
                    }
                }
            }
            (None, None, None) => {
                if let Ok(mut stmt) = conn.prepare_cached(
                    "SELECT e1.name, e2.name, t.name
                     FROM relation r
                     JOIN entity e1 ON e1.id = r.from_id
                     JOIN entity e2 ON e2.id = r.to_id
                     JOIN type_dict t ON t.id = r.type_id
                     WHERE e1.flags = 0 AND e2.flags = 0
                     ORDER BY r.from_id, r.to_id"
                ) {
                    if let Ok(rows) = stmt.query_map([], |row| {
                        Ok(Relation { from: row.get(0)?, to: row.get(1)?, relation_type: row.get(2)? })
                    }) {
                        for row in rows.flatten() { results.push(row); }
                    }
                }
            }
        }
        results
    }

    pub fn find_path(&self, from: &str, to: &str) -> Result<Option<Vec<String>>> {
        let conn = self.conn.lock();
        let (from_id, _, _, _) = match self.get_entity_id(&conn, from)? {
            Some(v) => v,
            None => {
                return Err(MCSError::InvalidParams(format!(
                    "Source entity '{from}' not found"
                )))
            }
        };
        let (to_id, _, _, _) = match self.get_entity_id(&conn, to)? {
            Some(v) => v,
            None => {
                return Err(MCSError::InvalidParams(format!(
                    "Target entity '{to}' not found"
                )))
            }
        };

        if from_id == to_id {
            return Ok(Some(vec![from.to_string()]));
        }

        // BFS with adjacency from relation table.
        let mut visited = HashSet::new();
        let mut parent: FxHashMap<i64, i64> = FxHashMap::default();
        let mut queue = VecDeque::new();
        visited.insert(from_id);
        queue.push_back(from_id);

        while let Some(cur) = queue.pop_front() {
            if cur == to_id {
                break;
            }
            // Fetch out-neighbors.
            if let Ok(mut stmt) =
                conn.prepare_cached("SELECT to_id FROM relation WHERE from_id = ?1")
            {
                if let Ok(rows) = stmt.query_map(params![cur], |row| row.get::<_, i64>(0)) {
                    for row in rows.flatten() {
                        if visited.insert(row) {
                            parent.insert(row, cur);
                            queue.push_back(row);
                        }
                    }
                }
            }
            // Also check in-neighbors (undirected traversal).
            if let Ok(mut stmt) =
                conn.prepare_cached("SELECT from_id FROM relation WHERE to_id = ?1")
            {
                if let Ok(rows) = stmt.query_map(params![cur], |row| row.get::<_, i64>(0)) {
                    for row in rows.flatten() {
                        if visited.insert(row) {
                            parent.insert(row, cur);
                            queue.push_back(row);
                        }
                    }
                }
            }
        }

        if !parent.contains_key(&to_id) && to_id != from_id {
            return Ok(None);
        }

        let mut path = Vec::new();
        let mut cur = to_id;
        path.push(cur);
        while let Some(&p) = parent.get(&cur) {
            path.push(p);
            cur = p;
            if cur == from_id {
                break;
            }
        }
        path.reverse();

        let mut name_path = Vec::with_capacity(path.len());
        for id in path {
            if let Ok(name) = conn.query_row(
                "SELECT name FROM entity WHERE id = ?1",
                params![id],
                |row| row.get::<_, String>(0),
            ) {
                name_path.push(name);
            }
        }

        Ok(Some(name_path))
    }

    pub fn compact(&self) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute_batch("PRAGMA incremental_vacuum;").map_err(sqlite_err)?;
        Ok(())
    }

    pub fn neighbors(
        &self,
        name: &str,
        direction: Direction,
        rtype: Option<&str>,
        depth: u32,
    ) -> Result<String> {
        self._traverse(name, direction, rtype, depth, true)
    }

    pub fn extract_subgraph(
        &self,
        names: &[String],
        depth: u32,
    ) -> Result<String> {
        if names.is_empty() {
            return Ok(r#"{"entities":[],"relations":[]}"#.to_string());
        }

        let conn = self.conn.lock();
        let mut all_entity_ids: HashSet<i64> = HashSet::new();
        let mut frontier: HashSet<i64> = HashSet::new();
        let mut all_rel_pairs: HashSet<(i64, i64, i64)> = HashSet::new();

        // Resolve seed entities.
        for name in names {
            let h = name_hash(name);
            if let Ok(Some(id)) = conn
                .query_row(
                    "SELECT id FROM entity WHERE name_hash = ?1 AND name = ?2 AND flags = 0",
                    params![h, name],
                    |row| row.get::<_, i64>(0),
                )
                .map(Some)
                .or_else(|e| if is_not_found(&e) { Ok(None) } else { Err(sqlite_err(e)) })
            {
                all_entity_ids.insert(id);
                frontier.insert(id);
            }
        }

        let mut current_depth = 0u32;
        while current_depth < depth && !frontier.is_empty() {
            let mut next_frontier: HashSet<i64> = HashSet::new();

            // Collect relations for current frontier.
            for fid in &frontier {
                if let Ok(mut stmt) = conn.prepare_cached(
                    "SELECT from_id, to_id, type_id FROM relation WHERE from_id = ?1",
                ) {
                    if let Ok(rows) =
                        stmt.query_map(params![fid], |row| {
                            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?))
                        })
                    {
                        for row in rows.flatten() {
                            let (from_id, to_id, type_id) = row;
                            all_rel_pairs.insert((from_id, to_id, type_id));
                            if all_entity_ids.insert(to_id) {
                                next_frontier.insert(to_id);
                            }
                        }
                    }
                }
                if let Ok(mut stmt) = conn.prepare_cached(
                    "SELECT from_id, to_id, type_id FROM relation WHERE to_id = ?1",
                ) {
                    if let Ok(rows) =
                        stmt.query_map(params![fid], |row| {
                            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?))
                        })
                    {
                        for row in rows.flatten() {
                            let (from_id, to_id, type_id) = row;
                            all_rel_pairs.insert((from_id, to_id, type_id));
                            if all_entity_ids.insert(from_id) {
                                next_frontier.insert(from_id);
                            }
                        }
                    }
                }
            }
            frontier = next_frontier;
            current_depth += 1;
        }

        let entities_json: String = {
            if all_entity_ids.is_empty() {
                "[]".to_string()
            } else {
                let ids: Vec<i64> = all_entity_ids.iter().copied().collect();
                let placeholders: Vec<String> = ids.iter().map(|_| "?".to_string()).collect();
                let sql = format!(
                    "SELECT COALESCE(json_group_array(json_object(
                        'name', e.name,
                        'entityType', t.name,
                        'observations', COALESCE((
                            SELECT json_group_array(o.body ORDER BY o.idx)
                            FROM observation o WHERE o.entity_id = e.id
                        ), json('[]'))
                    ) ORDER BY e.id), json('[]'))
                    FROM entity e
                    JOIN type_dict t ON t.id = e.type_id
                    WHERE e.id IN ({}) AND e.flags = 0",
                    placeholders.join(",")
                );
                conn.query_row(&sql, rusqlite::params_from_iter(&ids), |row| {
                    row.get::<_, String>(0)
                })
                .unwrap_or_else(|_| "[]".to_string())
            }
        };

        let relations_json: String = {
            if all_rel_pairs.is_empty() {
                "[]".to_string()
            } else {
                let vals: Vec<String> = all_rel_pairs.iter().map(|_| "(?, ?, ?)".to_string()).collect();
                let sql = format!(
                    "WITH r(from_id, to_id, type_id) AS (VALUES {})
                    SELECT COALESCE(json_group_array(json_object(
                        'from', e1.name,
                        'to', e2.name,
                        'relationType', t.name
                    )), json('[]'))
                    FROM r
                    JOIN entity e1 ON e1.id = r.from_id
                    JOIN entity e2 ON e2.id = r.to_id
                    JOIN type_dict t ON t.id = r.type_id
                    WHERE e1.flags = 0 AND e2.flags = 0",
                    vals.join(", ")
                );
                let params: Vec<&dyn ToSql> = all_rel_pairs.iter()
                    .flat_map(|(f, t, tp)| {
                        vec![f as &dyn ToSql, t as &dyn ToSql, tp as &dyn ToSql]
                    })
                    .collect();
                let mut stmt = conn.prepare(&sql).map_err(sqlite_err)?;
                stmt.query_row(params.as_slice(), |row| row.get::<_, String>(0))
                    .unwrap_or_else(|_| "[]".to_string())
            }
        };

        let mut out = String::with_capacity(32 + entities_json.len() + relations_json.len());
        out.push_str("{\"entities\":");
        out.push_str(&entities_json);
        out.push_str(",\"relations\":");
        out.push_str(&relations_json);
        out.push('}');
        Ok(out)
    }

    pub fn describe_entity(&self, name: &str) -> Result<Entity> {
        self.get_entity(name)?
            .ok_or_else(|| MCSError::InvalidParams(format!("Entity '{name}' not found")))
    }

    pub fn entity_type_counts(&self) -> Vec<(String, usize)> {
        let conn = self.conn.lock();
        select_all_types(&conn, 0).unwrap_or_default()
    }

    pub fn relation_type_counts(&self) -> Vec<(String, usize)> {
        let conn = self.conn.lock();
        select_all_types(&conn, 1).unwrap_or_default()
    }

    pub fn batch_get_entities(&self, names: &[String]) -> Vec<Option<Entity>> {
        names
            .iter()
            .map(|n| self.get_entity(n).unwrap_or(None))
            .collect()
    }

    pub fn find_all_paths(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
        max_paths: usize,
    ) -> Result<Vec<Vec<String>>> {
        let conn = self.conn.lock();
        let (from_id, _, _, _) = match self.get_entity_id(&conn, from)? {
            Some(v) => v,
            None => {
                return Err(MCSError::InvalidParams(format!(
                    "Source entity '{from}' not found"
                )))
            }
        };
        let (to_id, _, _, _) = match self.get_entity_id(&conn, to)? {
            Some(v) => v,
            None => {
                return Err(MCSError::InvalidParams(format!(
                    "Target entity '{to}' not found"
                )))
            }
        };

        if from_id == to_id {
            return Ok(vec![vec![from.to_string()]]);
        }

        // BFS enumerating all paths up to max_depth.
        let mut all_paths: Vec<Vec<i64>> = Vec::new();
        let mut queue: VecDeque<(i64, Vec<i64>)> = VecDeque::new();
        queue.push_back((from_id, vec![from_id]));

        while let Some((cur, path)) = queue.pop_front() {
            if all_paths.len() >= max_paths {
                break;
            }
            if path.len() > max_depth {
                continue;
            }

            // Out-neighbors.
            if let Ok(mut stmt) =
                conn.prepare_cached("SELECT to_id FROM relation WHERE from_id = ?1")
            {
                if let Ok(rows) = stmt.query_map(params![cur], |row| row.get::<_, i64>(0)) {
                    for next_id in rows.flatten() {
                        if next_id == to_id {
                            let mut full_path = path.clone();
                            full_path.push(next_id);
                            all_paths.push(full_path);
                            if all_paths.len() >= max_paths {
                                break;
                            }
                        } else if !path.contains(&next_id) && path.len() < max_depth {
                            let mut new_path = path.clone();
                            new_path.push(next_id);
                            queue.push_back((next_id, new_path));
                        }
                    }
                }
            }

            // In-neighbors (undirected).
            if let Ok(mut stmt) =
                conn.prepare_cached("SELECT from_id FROM relation WHERE to_id = ?1")
            {
                if let Ok(rows) = stmt.query_map(params![cur], |row| row.get::<_, i64>(0)) {
                    for next_id in rows.flatten() {
                        if next_id == to_id {
                            let mut full_path = path.clone();
                            full_path.push(next_id);
                            all_paths.push(full_path);
                            if all_paths.len() >= max_paths {
                                break;
                            }
                        } else if !path.contains(&next_id) && path.len() < max_depth {
                            let mut new_path = path.clone();
                            new_path.push(next_id);
                            queue.push_back((next_id, new_path));
                        }
                    }
                }
            }
        }

        // Convert ids to names.
        let mut named_paths: Vec<Vec<String>> = Vec::new();
        for path_ids in all_paths {
            let mut named = Vec::with_capacity(path_ids.len());
            for id in path_ids {
                if let Ok(name) = conn.query_row(
                    "SELECT name FROM entity WHERE id = ?1",
                    params![id],
                    |row| row.get::<_, String>(0),
                ) {
                    named.push(name);
                }
            }
            named_paths.push(named);
        }

        Ok(named_paths)
    }

    pub fn export(&self, format: &str) -> Result<String> {
        let conn = self.conn.lock();
        match format {
            "json" | _ => {
                conn.query_row(
                    "SELECT json_object(
                        'entities', COALESCE((
                            SELECT json_group_array(json_object(
                                'name', e.name,
                                'entityType', t.name,
                                'observations', COALESCE((
                                    SELECT json_group_array(o.body ORDER BY o.idx)
                                    FROM observation o WHERE o.entity_id = e.id
                                ), json('[]'))
                            ) ORDER BY e.id)
                            FROM entity e
                            JOIN type_dict t ON t.id = e.type_id
                            WHERE e.flags = 0
                        ), json('[]')),
                        'relations', COALESCE((
                            SELECT json_group_array(json_object(
                                'from', e1.name,
                                'to', e2.name,
                                'relationType', t.name
                            ))
                            FROM relation r
                            JOIN entity e1 ON e1.id = r.from_id
                            JOIN entity e2 ON e2.id = r.to_id
                            JOIN type_dict t ON t.id = r.type_id
                            WHERE e1.flags = 0 AND e2.flags = 0
                        ), json('[]'))
                    )",
                    [],
                    |row| row.get::<_, String>(0),
                )
                .map_err(sqlite_err)
            }
        }
    }

    pub fn wipe(&self) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute_batch(
            "DELETE FROM observation;
             DELETE FROM relation;
             DELETE FROM entity;
             DELETE FROM type_dict;
             DELETE FROM name_fts;
             DELETE FROM obs_fts;
             UPDATE graph_stat SET value = 0 WHERE key IN ('entities', 'relations', 'observations');
             UPDATE graph_stat SET value = 0 WHERE key IN ('entity_seq', 'obs_seq');",
        )
        .map_err(sqlite_err)?;
        self.seq_entity.store(0, Ordering::Relaxed);
        self.seq_obs.store(0, Ordering::Relaxed);
        self.cache.lock().clear();
        Ok(())
    }

    /// Periodic database maintenance: WAL checkpoint, query planner analysis,
    /// and FTS index optimization. Call from a background timer.
    pub fn run_maintenance(&self) -> Result<()> {
        let conn = self.conn.lock();

        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
            .map_err(sqlite_err)?;

        conn.execute_batch("PRAGMA optimize(0x10000);")
            .map_err(sqlite_err)?;

        conn.execute_batch(
            "INSERT INTO name_fts(name_fts) VALUES('optimize');
             INSERT INTO obs_fts(obs_fts) VALUES('optimize');",
        )
        .map_err(sqlite_err)?;

        Ok(())
    }

    fn _traverse(
        &self,
        name: &str,
        direction: Direction,
        rtype: Option<&str>,
        depth: u32,
        // unused — we always include relations; the caller controls via depth
        _include_relations: bool,
    ) -> Result<String> {
        let conn = self.conn.lock();
        let (start_id, _, _, _) = match self.get_entity_id(&conn, name)? {
            Some(v) => v,
            None => {
                return Err(MCSError::InvalidParams(format!(
                    "Entity '{name}' not found"
                )))
            }
        };

        let mut all_ids: HashSet<i64> = HashSet::new();
        let mut all_rels: HashSet<(i64, i64, i64)> = HashSet::new();
        let mut frontier: HashSet<i64> = HashSet::new();
        all_ids.insert(start_id);
        frontier.insert(start_id);

        let type_filter = rtype.and_then(|rt| get_type_id(&conn, rt, 1).ok());

        // Pre-compile all four possible queries outside the loop.
        let mut q_out_t = conn.prepare_cached(
            "SELECT to_id, type_id FROM relation WHERE from_id = ?1 AND type_id = ?2");
        let mut q_out   = conn.prepare_cached(
            "SELECT to_id, type_id FROM relation WHERE from_id = ?1");
        let mut q_in_t  = conn.prepare_cached(
            "SELECT from_id, type_id FROM relation WHERE to_id = ?1 AND type_id = ?2");
        let mut q_in    = conn.prepare_cached(
            "SELECT from_id, type_id FROM relation WHERE to_id = ?1");

        let mut cur_depth = 0u32;
        while cur_depth < depth && !frontier.is_empty() {
            let mut next_frontier: HashSet<i64> = HashSet::new();

            for &fid in &frontier {
                if direction == Direction::Outgoing || direction == Direction::Both {
                    if let Some(tid) = type_filter {
                        if let Ok(ref mut stmt) = q_out_t {
                            if let Ok(rows) = stmt.query_map(params![fid, tid], |row| {
                                Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
                            }) {
                                for row in rows.flatten() {
                                    let (to_id, t_id) = row;
                                    all_rels.insert((fid, to_id, t_id));
                                    if all_ids.insert(to_id) { next_frontier.insert(to_id); }
                                }
                            }
                        }
                    } else if let Ok(ref mut stmt) = q_out {
                        if let Ok(rows) = stmt.query_map(params![fid], |row| {
                            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
                        }) {
                            for row in rows.flatten() {
                                let (to_id, t_id) = row;
                                all_rels.insert((fid, to_id, t_id));
                                if all_ids.insert(to_id) { next_frontier.insert(to_id); }
                            }
                        }
                    }
                }

                if direction == Direction::Incoming || direction == Direction::Both {
                    if let Some(tid) = type_filter {
                        if let Ok(ref mut stmt) = q_in_t {
                            if let Ok(rows) = stmt.query_map(params![fid, tid], |row| {
                                Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
                            }) {
                                for row in rows.flatten() {
                                    let (from_id, t_id) = row;
                                    all_rels.insert((from_id, fid, t_id));
                                    if all_ids.insert(from_id) { next_frontier.insert(from_id); }
                                }
                            }
                        }
                    } else if let Ok(ref mut stmt) = q_in {
                        if let Ok(rows) = stmt.query_map(params![fid], |row| {
                            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
                        }) {
                            for row in rows.flatten() {
                                let (from_id, t_id) = row;
                                all_rels.insert((from_id, fid, t_id));
                                if all_ids.insert(from_id) { next_frontier.insert(from_id); }
                            }
                        }
                    }
                }
            }

            frontier = next_frontier;
            cur_depth += 1;
        }

        let entities_json: String = {
            if all_ids.is_empty() {
                "[]".to_string()
            } else {
                let ids: Vec<i64> = all_ids.iter().copied().collect();
                let placeholders: Vec<String> = ids.iter().map(|_| "?".to_string()).collect();
                let sql = format!(
                    "SELECT COALESCE(json_group_array(json_object(
                        'name', e.name,
                        'entityType', t.name,
                        'observations', COALESCE((
                            SELECT json_group_array(o.body ORDER BY o.idx)
                            FROM observation o WHERE o.entity_id = e.id
                        ), json('[]'))
                    ) ORDER BY e.id), json('[]'))
                    FROM entity e
                    JOIN type_dict t ON t.id = e.type_id
                    WHERE e.id IN ({}) AND e.flags = 0",
                    placeholders.join(",")
                );
                conn.query_row(&sql, rusqlite::params_from_iter(&ids), |row| {
                    row.get::<_, String>(0)
                })
                .unwrap_or_else(|_| "[]".to_string())
            }
        };

        let relations_json: String = {
            if all_rels.is_empty() {
                "[]".to_string()
            } else {
                let vals: Vec<String> = all_rels.iter().map(|_| "(?, ?, ?)".to_string()).collect();
                let sql = format!(
                    "WITH r(from_id, to_id, type_id) AS (VALUES {})
                    SELECT COALESCE(json_group_array(json_object(
                        'from', e1.name,
                        'to', e2.name,
                        'relationType', t.name
                    )), json('[]'))
                    FROM r
                    JOIN entity e1 ON e1.id = r.from_id
                    JOIN entity e2 ON e2.id = r.to_id
                    JOIN type_dict t ON t.id = r.type_id
                    WHERE e1.flags = 0 AND e2.flags = 0",
                    vals.join(", ")
                );
                let params: Vec<&dyn ToSql> = all_rels.iter()
                    .flat_map(|(f, t, tp)| {
                        vec![f as &dyn ToSql, t as &dyn ToSql, tp as &dyn ToSql]
                    })
                    .collect();
                let mut stmt = conn.prepare(&sql).map_err(sqlite_err)?;
                stmt.query_row(params.as_slice(), |row| row.get::<_, String>(0))
                    .unwrap_or_else(|_| "[]".to_string())
            }
        };

        let mut out = String::with_capacity(32 + entities_json.len() + relations_json.len());
        out.push_str("{\"entities\":");
        out.push_str(&entities_json);
        out.push_str(",\"relations\":");
        out.push_str(&relations_json);
        out.push('}');
        Ok(out)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::ops::Deref;
    use std::path::PathBuf;

    struct TestKg(GraphHandle, PathBuf);

    impl Deref for TestKg {
        type Target = GraphHandle;
        fn deref(&self) -> &GraphHandle {
            &self.0
        }
    }

    impl Drop for TestKg {
        fn drop(&mut self) {
            cleanup_db(&self.1);
        }
    }

    fn cleanup_db(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    fn new_kg() -> TestKg {
        use std::sync::atomic::AtomicU64;
        use std::sync::atomic::Ordering;
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("kg_test_{}_{}.db", std::process::id(), n));
        cleanup_db(&path);
        let kg = GraphHandle::new(&path, Durability::Async, 268435456, NonZeroUsize::new(10000).unwrap()).expect("create KG");
        TestKg(kg, path)
    }

    #[test]
    fn test_create_and_get_entity() {
        let kg = new_kg();
        let entities = vec![Entity {
            name: "test".into(),
            entity_type: "person".into(),
            observations: vec!["obs1".into(), "obs2".into()],
        }];
        let created = kg.create_entities(&entities).unwrap();
        assert_eq!(created.len(), 1);

        let got = kg.get_entity("test").unwrap().unwrap();
        assert_eq!(got.name, "test");
        assert_eq!(got.entity_type, "person");
        assert_eq!(got.observations, vec!["obs1", "obs2"]);
    }

    #[test]
    fn test_get_nonexistent() {
        let kg = new_kg();
        assert!(kg.get_entity("nonexistent").unwrap().is_none());
    }

    #[test]
    fn test_delete_entity() {
        let kg = new_kg();
        kg.create_entities(&[Entity {
            name: "del".into(),
            entity_type: "t".into(),
            observations: vec![],
        }])
        .unwrap();
        assert!(kg.get_entity("del").unwrap().is_some());
        kg.delete_entities(&["del".to_string()]).unwrap();
        assert!(kg.get_entity("del").unwrap().is_none());
    }

    #[test]
    fn test_add_and_delete_observations() {
        let kg = new_kg();
        kg.create_entities(&[Entity {
            name: "obs_test".into(),
            entity_type: "t".into(),
            observations: vec!["a".into()],
        }])
        .unwrap();

        let added = kg.add_observations("obs_test", &["b".into(), "c".into()]).unwrap();
        assert_eq!(added.len(), 2);

        let ent = kg.get_entity("obs_test").unwrap().unwrap();
        assert!(ent.observations.contains(&"b".into()));
        assert!(ent.observations.contains(&"c".into()));

        kg.delete_observations("obs_test", &["b".into()]).unwrap();
        let ent = kg.get_entity("obs_test").unwrap().unwrap();
        assert!(!ent.observations.contains(&"b".into()));
        assert!(ent.observations.contains(&"c".into()));
        assert!(ent.observations.contains(&"a".into()));
    }

    #[test]
    fn test_create_relations() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity {
                name: "A".into(),
                entity_type: "node".into(),
                observations: vec![],
            },
            Entity {
                name: "B".into(),
                entity_type: "node".into(),
                observations: vec![],
            },
        ])
        .unwrap();

        let rels = kg
            .create_relations(&[Relation {
                from: "A".into(),
                to: "B".into(),
                relation_type: "edge".into(),
            }])
            .unwrap();
        assert_eq!(rels.len(), 1);

        assert_eq!(kg.get_entity_count().unwrap(), 2);
        assert_eq!(kg.get_relation_count().unwrap(), 1);
    }

    #[test]
    fn test_search_nodes() {
        let kg = new_kg();
        kg.create_entities(&[Entity {
            name: "Einstein".into(),
            entity_type: "scientist".into(),
            observations: vec!["physics".into(), "relativity".into()],
        }])
        .unwrap();

        let results = kg.search_nodes_filtered("physics", None, 0, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Einstein");

        let results = kg.search_nodes_filtered("physics", Some("scientist"), 0, 10);
        assert_eq!(results.len(), 1);

        let results = kg.search_nodes_filtered("physics", Some("nonexistent"), 0, 10);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_find_path() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity { name: "A".into(), entity_type: "n".into(), observations: vec![] },
            Entity { name: "B".into(), entity_type: "n".into(), observations: vec![] },
            Entity { name: "C".into(), entity_type: "n".into(), observations: vec![] },
        ]).unwrap();

        kg.create_relations(&[
            Relation { from: "A".into(), to: "B".into(), relation_type: "e".into() },
            Relation { from: "B".into(), to: "C".into(), relation_type: "e".into() },
        ]).unwrap();

        let path = kg.find_path("A", "C").unwrap().unwrap();
        assert_eq!(path, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_degree() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity { name: "A".into(), entity_type: "n".into(), observations: vec![] },
            Entity { name: "B".into(), entity_type: "n".into(), observations: vec![] },
            Entity { name: "C".into(), entity_type: "n".into(), observations: vec![] },
        ]).unwrap();

        kg.create_relations(&[
            Relation { from: "A".into(), to: "B".into(), relation_type: "e".into() },
            Relation { from: "A".into(), to: "C".into(), relation_type: "e".into() },
        ]).unwrap();

        assert_eq!(kg.degree("A", Direction::Outgoing).unwrap(), 2);
        assert_eq!(kg.degree("A", Direction::Incoming).unwrap(), 0);
        assert_eq!(kg.degree("B", Direction::Incoming).unwrap(), 1);
    }

    #[test]
    fn test_neighbors() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity { name: "A".into(), entity_type: "n".into(), observations: vec![] },
            Entity { name: "B".into(), entity_type: "n".into(), observations: vec![] },
        ]).unwrap();

        kg.create_relations(&[Relation {
            from: "A".into(), to: "B".into(), relation_type: "e".into(),
        }]).unwrap();

        let result = kg.neighbors("A", Direction::Outgoing, None, 1).unwrap();
        let v: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["entities"].as_array().unwrap().len(), 2);
        assert_eq!(v["relations"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_open_nodes() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity { name: "X".into(), entity_type: "n".into(), observations: vec!["obs_x".into()] },
            Entity { name: "Y".into(), entity_type: "n".into(), observations: vec!["obs_y".into()] },
        ]).unwrap();

        kg.create_relations(&[Relation {
            from: "X".into(), to: "Y".into(), relation_type: "e".into(),
        }]).unwrap();

        let result = kg.open_nodes(&["X".into()]);
        let v: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["entities"].as_array().unwrap().len(), 1);
        assert_eq!(v["relations"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_entities_exist() {
        let kg = new_kg();
        kg.create_entities(&[Entity {
            name: "exists".into(), entity_type: "t".into(), observations: vec![],
        }]).unwrap();

        let res = kg.entities_exist(&["exists".into(), "missing".into()]).unwrap();
        assert_eq!(res, vec![true, false]);
    }

    #[test]
    fn test_describe_entity() {
        let kg = new_kg();
        kg.create_entities(&[Entity {
            name: "desc".into(), entity_type: "t".into(), observations: vec!["o".into()],
        }]).unwrap();

        let entity = kg.describe_entity("desc").unwrap();
        assert_eq!(entity.name, "desc");
    }

    #[test]
    fn test_entity_type_counts() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity { name: "a".into(), entity_type: "person".into(), observations: vec![] },
            Entity { name: "b".into(), entity_type: "person".into(), observations: vec![] },
            Entity { name: "c".into(), entity_type: "place".into(), observations: vec![] },
        ]).unwrap();

        let counts = kg.entity_type_counts();
        let map: FxHashMap<_, _> = counts.into_iter().collect();
        assert_eq!(map.get("person"), Some(&2));
        assert_eq!(map.get("place"), Some(&1));
    }

    #[test]
    fn test_relation_type_counts() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity { name: "a".into(), entity_type: "n".into(), observations: vec![] },
            Entity { name: "b".into(), entity_type: "n".into(), observations: vec![] },
            Entity { name: "c".into(), entity_type: "n".into(), observations: vec![] },
        ]).unwrap();

        kg.create_relations(&[
            Relation { from: "a".into(), to: "b".into(), relation_type: "knows".into() },
            Relation { from: "a".into(), to: "c".into(), relation_type: "knows".into() },
        ]).unwrap();

        let counts = kg.relation_type_counts();
        let map: FxHashMap<_, _> = counts.into_iter().collect();
        assert_eq!(map.get("knows"), Some(&2));
    }

    #[test]
    fn test_upsert_entities() {
        let kg = new_kg();
        kg.create_entities(&[Entity {
            name: "u".into(), entity_type: "old".into(), observations: vec!["existing".into()],
        }]).unwrap();

        // Upsert with new type and additional observation.
        kg.upsert_entities(&[Entity {
            name: "u".into(), entity_type: "new".into(), observations: vec!["existing".into(), "added".into()],
        }]).unwrap();

        let ent = kg.get_entity("u").unwrap().unwrap();
        assert_eq!(ent.entity_type, "new");
        assert!(ent.observations.contains(&"added".into()));
        assert!(ent.observations.contains(&"existing".into()));
    }

    #[test]
    fn test_merge_entities() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity { name: "source".into(), entity_type: "t".into(), observations: vec!["src_obs".into()] },
            Entity { name: "target".into(), entity_type: "t".into(), observations: vec!["tgt_obs".into()] },
        ]).unwrap();

        kg.create_relations(&[Relation {
            from: "source".into(), to: "target".into(), relation_type: "e".into(),
        }]).unwrap();

        let merged = kg.merge_entities("source", "target").unwrap();
        assert_eq!(merged.name, "target");
        assert_eq!(kg.get_entity("source").unwrap().is_none(), true);
    }

    #[test]
    fn test_find_all_paths() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity { name: "A".into(), entity_type: "n".into(), observations: vec![] },
            Entity { name: "B".into(), entity_type: "n".into(), observations: vec![] },
            Entity { name: "C".into(), entity_type: "n".into(), observations: vec![] },
        ]).unwrap();

        kg.create_relations(&[
            Relation { from: "A".into(), to: "B".into(), relation_type: "e".into() },
            Relation { from: "B".into(), to: "C".into(), relation_type: "e".into() },
            Relation { from: "A".into(), to: "C".into(), relation_type: "e".into() },
        ]).unwrap();

        let paths = kg.find_all_paths("A", "C", 5, 10).unwrap();
        assert!(paths.len() >= 2);
    }

    #[test]
    fn test_batch_get_entities() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity { name: "a".into(), entity_type: "t".into(), observations: vec![] },
            Entity { name: "b".into(), entity_type: "t".into(), observations: vec![] },
        ]).unwrap();

        let results = kg.batch_get_entities(&["a".into(), "missing".into(), "b".into()]);
        assert_eq!(results.len(), 3);
        assert!(results[0].is_some());
        assert!(results[1].is_none());
        assert!(results[2].is_some());
    }

    #[test]
    fn test_export_graph() {
        let kg = new_kg();
        kg.create_entities(&[Entity {
            name: "exp".into(), entity_type: "t".into(), observations: vec!["o".into()],
        }]).unwrap();

        let exported = kg.export("json").unwrap();
        assert!(exported.contains("exp"));
        assert!(exported.contains("o"));
    }

    #[test]
    fn test_graph_stats() {
        let kg = new_kg();
        assert_eq!(kg.get_entity_count().unwrap(), 0);
        assert_eq!(kg.get_relation_count().unwrap(), 0);

        kg.create_entities(&[Entity {
            name: "s".into(), entity_type: "t".into(), observations: vec![],
        }]).unwrap();

        assert_eq!(kg.get_entity_count().unwrap(), 1);
    }

    #[test]
    fn test_read_graph_filtered() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity { name: "p1".into(), entity_type: "person".into(), observations: vec![] },
            Entity { name: "p2".into(), entity_type: "place".into(), observations: vec![] },
        ]).unwrap();

        let out = kg.read_graph_filtered(Some("person"), 0, 10).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["entities"].as_array().unwrap().len(), 1);
        assert_eq!(v["entities"][0]["name"], "p1");
    }

    #[test]
    fn test_wipe() {
        let kg = new_kg();
        kg.create_entities(&[Entity {
            name: "w".into(), entity_type: "t".into(), observations: vec!["o".into()],
        }]).unwrap();
        assert_eq!(kg.get_entity_count().unwrap(), 1);

        kg.wipe().unwrap();
        assert_eq!(kg.get_entity_count().unwrap(), 0);
    }

    #[test]
    fn test_push_json_str() {
        let mut buf = String::new();
        push_json_str(&mut buf, "hello");
        assert_eq!(buf, "\"hello\"");
        let mut buf = String::new();
        push_json_str(&mut buf, "he\"llo");
        assert_eq!(buf, "\"he\\\"llo\"");
    }

    // ── create_entities edge cases ────────────────────────────────────

    #[test]
    fn test_create_entities_empty_input() {
        let kg = new_kg();
        let created = kg.create_entities(&[]).unwrap();
        assert!(created.is_empty());
    }

    #[test]
    fn test_create_entities_skip_empty_name() {
        let kg = new_kg();
        let created = kg.create_entities(&[Entity {
            name: "".into(),
            entity_type: "t".into(),
            observations: vec![],
        }])
        .unwrap();
        assert!(created.is_empty());
        assert_eq!(kg.get_entity_count().unwrap(), 0);
    }

    #[test]
    fn test_create_entities_duplicate_names() {
        let kg = new_kg();
        let e = Entity {
            name: "dup".into(),
            entity_type: "t".into(),
            observations: vec!["obs".into()],
        };
        let first = kg.create_entities(&[e.clone()]).unwrap();
        assert_eq!(first.len(), 1);
        let second = kg.create_entities(&[e.clone()]).unwrap();
        assert!(second.is_empty());
        assert_eq!(kg.get_entity_count().unwrap(), 1);
    }

    #[test]
    fn test_create_entities_partial_duplicates() {
        let kg = new_kg();
        let created = kg.create_entities(&[
            Entity { name: "a".into(), entity_type: "t".into(), observations: vec![] },
            Entity { name: "b".into(), entity_type: "t".into(), observations: vec![] },
        ]).unwrap();
        assert_eq!(created.len(), 2);

        let second = kg.create_entities(&[
            Entity { name: "b".into(), entity_type: "t".into(), observations: vec![] },
            Entity { name: "c".into(), entity_type: "t".into(), observations: vec![] },
        ]).unwrap();
        assert_eq!(second.len(), 1); // only c created
        assert_eq!(second[0].name, "c");
        assert_eq!(kg.get_entity_count().unwrap(), 3);
    }

    #[test]
    fn test_create_entities_mixed_empty_and_valid() {
        let kg = new_kg();
        let created = kg.create_entities(&[
            Entity { name: "".into(), entity_type: "t".into(), observations: vec![] },
            Entity { name: "valid".into(), entity_type: "t".into(), observations: vec![] },
            Entity { name: "".into(), entity_type: "t".into(), observations: vec![] },
        ]).unwrap();
        assert_eq!(created.len(), 1);
        assert_eq!(created[0].name, "valid");
        assert_eq!(kg.get_entity_count().unwrap(), 1);
    }

    #[test]
    fn test_create_entities_same_name_in_batch() {
        let kg = new_kg();
        let created = kg.create_entities(&[
            Entity { name: "dup_in_batch".into(), entity_type: "t".into(), observations: vec![] },
            Entity { name: "dup_in_batch".into(), entity_type: "t".into(), observations: vec![] },
        ]).unwrap();
        assert_eq!(created.len(), 1);
        assert_eq!(kg.get_entity_count().unwrap(), 1);
    }

    // ── create_relations edge cases ───────────────────────────────────

    #[test]
    fn test_create_relations_empty_input() {
        let kg = new_kg();
        let rels = kg.create_relations(&[]).unwrap();
        assert!(rels.is_empty());
    }

    #[test]
    fn test_create_relations_nonexistent_from() {
        let kg = new_kg();
        kg.create_entities(&[Entity {
            name: "B".into(), entity_type: "t".into(), observations: vec![],
        }]).unwrap();

        let rels = kg.create_relations(&[Relation {
            from: "A".into(), to: "B".into(), relation_type: "e".into(),
        }]).unwrap();
        assert!(rels.is_empty());
        assert_eq!(kg.get_relation_count().unwrap(), 0);
    }

    #[test]
    fn test_create_relations_nonexistent_to() {
        let kg = new_kg();
        kg.create_entities(&[Entity {
            name: "A".into(), entity_type: "t".into(), observations: vec![],
        }]).unwrap();

        let rels = kg.create_relations(&[Relation {
            from: "A".into(), to: "B".into(), relation_type: "e".into(),
        }]).unwrap();
        assert!(rels.is_empty());
        assert_eq!(kg.get_relation_count().unwrap(), 0);
    }

    #[test]
    fn test_create_relations_both_nonexistent() {
        let kg = new_kg();
        let rels = kg.create_relations(&[Relation {
            from: "A".into(), to: "B".into(), relation_type: "e".into(),
        }]).unwrap();
        assert!(rels.is_empty());
    }

    #[test]
    fn test_create_relations_self_loop() {
        let kg = new_kg();
        kg.create_entities(&[Entity {
            name: "self".into(), entity_type: "t".into(), observations: vec![],
        }]).unwrap();

        let rels = kg.create_relations(&[Relation {
            from: "self".into(), to: "self".into(), relation_type: "loop".into(),
        }]).unwrap();
        assert_eq!(rels.len(), 1);
        assert_eq!(kg.get_relation_count().unwrap(), 1);
        assert_eq!(kg.degree("self", Direction::Outgoing).unwrap(), 1);
        assert_eq!(kg.degree("self", Direction::Incoming).unwrap(), 1);
    }

    #[test]
    fn test_create_relations_duplicate() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity { name: "A".into(), entity_type: "t".into(), observations: vec![] },
            Entity { name: "B".into(), entity_type: "t".into(), observations: vec![] },
        ]).unwrap();

        let r = Relation {
            from: "A".into(), to: "B".into(), relation_type: "e".into(),
        };
        let first = kg.create_relations(&[r.clone()]).unwrap();
        assert_eq!(first.len(), 1);

        let second = kg.create_relations(&[r.clone()]).unwrap();
        assert!(second.is_empty());
        assert_eq!(kg.get_relation_count().unwrap(), 1);
    }

    #[test]
    fn test_create_relations_new_type_auto_created() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity { name: "A".into(), entity_type: "t".into(), observations: vec![] },
            Entity { name: "B".into(), entity_type: "t".into(), observations: vec![] },
        ]).unwrap();

        let rels = kg.create_relations(&[Relation {
            from: "A".into(), to: "B".into(), relation_type: "brand_new_type".into(),
        }]).unwrap();
        assert_eq!(rels.len(), 1);

        let counts = kg.relation_type_counts();
        let map: FxHashMap<_, _> = counts.into_iter().collect();
        assert_eq!(map.get("brand_new_type"), Some(&1));
    }

    #[test]
    fn test_create_relations_degree_updates() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity { name: "A".into(), entity_type: "t".into(), observations: vec![] },
            Entity { name: "B".into(), entity_type: "t".into(), observations: vec![] },
            Entity { name: "C".into(), entity_type: "t".into(), observations: vec![] },
        ]).unwrap();

        kg.create_relations(&[
            Relation { from: "A".into(), to: "B".into(), relation_type: "e".into() },
            Relation { from: "A".into(), to: "C".into(), relation_type: "e".into() },
        ]).unwrap();

        assert_eq!(kg.degree("A", Direction::Outgoing).unwrap(), 2);
        assert_eq!(kg.degree("A", Direction::Incoming).unwrap(), 0);
        assert_eq!(kg.degree("B", Direction::Incoming).unwrap(), 1);
        assert_eq!(kg.degree("C", Direction::Incoming).unwrap(), 1);
        assert_eq!(kg.degree("A", Direction::Both).unwrap(), 2);
    }

    #[test]
    fn test_create_relations_delete_and_recreate() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity { name: "A".into(), entity_type: "t".into(), observations: vec![] },
            Entity { name: "B".into(), entity_type: "t".into(), observations: vec![] },
        ]).unwrap();

        let r = Relation {
            from: "A".into(), to: "B".into(), relation_type: "e".into(),
        };
        kg.create_relations(&[r.clone()]).unwrap();
        assert_eq!(kg.get_relation_count().unwrap(), 1);

        kg.delete_relations(&[r.clone()]).unwrap();
        assert_eq!(kg.get_relation_count().unwrap(), 0);

        // Recreate after delete
        let re = kg.create_relations(&[r.clone()]).unwrap();
        assert_eq!(re.len(), 1);
        assert_eq!(kg.get_relation_count().unwrap(), 1);
    }

    // ── Integration edge cases ────────────────────────────────────────

    #[test]
    fn test_create_entities_then_relations_then_delete_entity_with_relations() {
        let kg = new_kg();
        kg.create_entities(&[
            Entity { name: "A".into(), entity_type: "t".into(), observations: vec![] },
            Entity { name: "B".into(), entity_type: "t".into(), observations: vec![] },
        ]).unwrap();
        kg.create_relations(&[
            Relation { from: "A".into(), to: "B".into(), relation_type: "e".into() },
        ]).unwrap();

        assert_eq!(kg.get_relation_count().unwrap(), 1);

        // Deleting entity A should also delete the relation
        kg.delete_entities(&["A".into()]).unwrap();
        assert_eq!(kg.get_entity("A").unwrap().is_none(), true);
        assert_eq!(kg.get_relation_count().unwrap(), 0);
    }

    #[test]
    fn test_graph_stats_after_entity_with_observations() {
        let kg = new_kg();
        kg.create_entities(&[Entity {
            name: "stat".into(), entity_type: "t".into(),
            observations: vec!["o1".into(), "o2".into(), "o3".into()],
        }]).unwrap();

        let ecount = kg.get_entity_count().unwrap();
        // graph_stat for observations is tracked but there's no public getter for it
        assert_eq!(ecount, 1);

        // delete reverts stats
        kg.delete_entities(&["stat".into()]).unwrap();
        assert_eq!(kg.get_entity_count().unwrap(), 0);
    }
}
