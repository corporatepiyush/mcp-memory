use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arc_swap::ArcSwap;

const MAGIC: &[u8; 8] = b"MCPMEMV1";
const MAGIC_CRC: &[u8; 8] = b"MCPMEMV2";
const MAX_RECORD_BYTES: u32 = 1 << 20;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordKind {
    CreateEntity = 0,
    CreateRelation = 1,
    AddObservations = 2,
    DeleteEntity = 3,
    DeleteObservations = 4,
    DeleteRelation = 5,
    /// Opens a transaction: records that follow are buffered on replay and only
    /// applied once a matching [`RecordKind::TxnCommit`] is seen. An unclosed
    /// transaction (no commit before EOF) is discarded — this is how
    /// multi-record operations like `merge_entities` stay crash-atomic.
    TxnBegin = 6,
    /// Closes a transaction opened by [`RecordKind::TxnBegin`].
    TxnCommit = 7,
}

impl RecordKind {
    #[inline]
    pub const fn from_u8(v: u8) -> Option<RecordKind> {
        Some(match v {
            0 => RecordKind::CreateEntity,
            1 => RecordKind::CreateRelation,
            2 => RecordKind::AddObservations,
            3 => RecordKind::DeleteEntity,
            4 => RecordKind::DeleteObservations,
            5 => RecordKind::DeleteRelation,
            6 => RecordKind::TxnBegin,
            7 => RecordKind::TxnCommit,
            _ => return None,
        })
    }
}

pub struct BinaryStore {
    writer: BufWriter<File>,
    path: PathBuf,
    /// Whether this store writes CRC32 footers on each record.
    /// `true` for new files (magic `MCPMEMV2`); `false` for legacy V1 files.
    has_crc: bool,
    /// Shared cell holding the *current* file handle for the background sync
    /// thread to `fsync`, without holding any lock. It is updated every time the
    /// underlying file is (re)opened — notably by `compact`/`reopen_truncated`,
    /// which swap in a fresh inode — so the sync thread never keeps fsyncing a
    /// stale fd that points at a renamed-away/unlinked inode (D1).
    pub(crate) sync_slot: Arc<ArcSwap<File>>,
}

impl BinaryStore {
    pub const fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn new(path: &Path) -> std::io::Result<Self> {
        Self::new_with_slot(path, None)
    }

    /// Open (or create) the log. When `slot` is `Some`, the freshly opened file
    /// handle is published into that existing shared cell instead of a new one —
    /// this is how `compact` keeps the background sync thread pointed at the
    /// post-compaction file rather than the renamed-away original (D1).
    pub fn new_with_slot(
        path: &Path,
        slot: Option<Arc<ArcSwap<File>>>,
    ) -> std::io::Result<Self> {
        let exists = path.exists();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(path)?;

        let handle = Arc::new(file.try_clone()?);
        let sync_slot = match slot {
            Some(s) => {
                s.store(handle);
                s
            }
            None => Arc::new(ArcSwap::new(handle)),
        };

        // Determine CRC support and open the write handle.
        let (has_crc, file) = if !exists {
            let f = OpenOptions::new()
                .create(true)
                .append(true)
                .read(false)
                .open(path)?;
            let mut w = BufWriter::with_capacity(65536, f);
            w.write_all(MAGIC_CRC)?;
            w.flush()?;
            (true, w.into_inner().map_err(|e| e.into_error())?)
        } else {
            // Probe the existing file's magic to determine CRC support.
            let probe_file = OpenOptions::new().read(true).open(path)?;
            let mut probe = [0u8; 8];
            let has_crc = match std::io::BufReader::new(&probe_file).read_exact(&mut probe) {
                Ok(()) => &probe == MAGIC_CRC,
                _ => false,
            };
            drop(probe_file);
            let f = OpenOptions::new()
                .create(true)
                .append(true)
                .read(false)
                .open(path)?;
            (has_crc, f)
        };

        let writer = BufWriter::with_capacity(65536, file);

        Ok(Self {
            writer,
            path: path.to_path_buf(),
            has_crc,
            sync_slot,
        })
    }

    pub fn write_record(&mut self, kind: RecordKind, payload: &[u8]) -> std::io::Result<()> {
        let crc_len: usize = if self.has_crc { 4 } else { 0 };
        let total_len = 4 + 1 + payload.len() + crc_len;
        if total_len as u32 > MAX_RECORD_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Record too large",
            ));
        }
        self.writer.write_all(&(total_len as u32).to_le_bytes())?;
        self.writer.write_all(&[kind as u8])?;
        self.writer.write_all(payload)?;
        if self.has_crc {
            let crc = crc32fast::hash(payload);
            self.writer.write_all(&crc.to_le_bytes())?;
        }
        Ok(())
    }

    /// Flush the `BufWriter` to the kernel buffer (no `fsync`).
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }

    /// `fsync` the underlying file (kernel buffer → disk).
    pub fn sync(&mut self) -> std::io::Result<()> {
        self.writer.get_ref().sync_data()
    }

    pub fn flush_and_sync(&mut self) -> std::io::Result<()> {
        self.flush()?;
        self.sync()
    }

    pub fn replay<F>(&self, mut callback: F) -> std::io::Result<()>
    where
        F: FnMut(RecordKind, &[u8]),
    {
        let file = match OpenOptions::new().read(true).open(&self.path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e),
        };

        let meta = file.metadata()?;
        if meta.len() == 0 {
            return Ok(());
        }

        let mut reader = BufReader::with_capacity(65536, file);
        let mut magic = [0u8; 8];

        match reader.read_exact(&mut magic) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        }

        let has_crc = if &magic == MAGIC_CRC {
            true
        } else if &magic == MAGIC {
            false
        } else {
            return Ok(());
        };

        let mut payload_buf = Vec::with_capacity(4096);

        loop {
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(e) => return Err(e),
            }
            let total_len = u32::from_le_bytes(len_buf) as usize;
            if total_len < 5 || total_len > MAX_RECORD_BYTES as usize {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid record length: {total_len}"),
                ));
            }
            let payload_len = if has_crc {
                total_len.checked_sub(5 + 4).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "Record too short for CRC")
                })?
            } else {
                total_len - 5
            };

            // A crash can leave a record's length prefix written but its body
            // only partially flushed. Treat a short read on the kind/payload as
            // a torn tail (stop cleanly) rather than a hard error — otherwise a
            // single interrupted write would make the whole log unopenable.
            let mut kind_buf = [0u8; 1];
            match reader.read_exact(&mut kind_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(e) => return Err(e),
            }
            let kind_val = kind_buf[0];

            payload_buf.clear();
            payload_buf.resize(payload_len, 0);
            if payload_len > 0 {
                match reader.read_exact(&mut payload_buf) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
                    Err(e) => return Err(e),
                }
            }

            // Verify CRC32 for V2 records. A mismatch is treated as a torn
            // tail (stop cleanly) rather than a hard corruption error.
            if has_crc {
                let mut crc_buf = [0u8; 4];
                match reader.read_exact(&mut crc_buf) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
                    Err(e) => return Err(e),
                }
                let expected = u32::from_le_bytes(crc_buf);
                if crc32fast::hash(&payload_buf) != expected {
                    tracing::warn!("CRC mismatch at offset — torn tail detected, stopping replay");
                    return Ok(());
                }
            }

            if let Some(kind) = RecordKind::from_u8(kind_val) {
                callback(kind, &payload_buf);
            } else {
                tracing::warn!("Unknown record kind byte {kind_val}, skipping");
            }
        }
    }

    pub fn close(&mut self) -> std::io::Result<()> {
        self.flush_and_sync()
    }

    /// Reopen the file with truncation — discards all existing records.
    /// Used by `compact` to rewrite a fresh log from in-memory state.
    pub fn reopen_truncated(&mut self) -> std::io::Result<()> {
        self.writer.flush()?;
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        // Publish the new handle so the background sync thread fsyncs this file,
        // not the truncated-away one (D1).
        self.sync_slot.store(Arc::new(file.try_clone()?));
        let mut writer = BufWriter::with_capacity(65536, file);
        writer.write_all(MAGIC_CRC)?;
        writer.flush()?;
        self.writer = writer;
        self.has_crc = true;
        Ok(())
    }
}

// --- Binary encoding helpers ---

fn encode_str(buf: &mut Vec<u8>, s: &str) -> std::io::Result<()> {
    let bytes = s.as_bytes();
    let len = bytes.len();
    if len > u16::MAX as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("string too long (max {} bytes, got {len})", u16::MAX),
        ));
    }
    buf.extend_from_slice(&(len as u16).to_le_bytes());
    buf.extend_from_slice(bytes);
    Ok(())
}

fn decode_str<'a>(data: &'a [u8], offset: &mut usize) -> Option<&'a str> {
    if *offset + 2 > data.len() {
        return None;
    }
    let len = u16::from_le_bytes([data[*offset], data[*offset + 1]]) as usize;
    *offset += 2;
    if *offset + len > data.len() {
        return None;
    }
    let s = std::str::from_utf8(&data[*offset..*offset + len]).ok()?;
    *offset += len;
    Some(s)
}

fn decode_count(data: &[u8], offset: &mut usize) -> Option<usize> {
    if *offset + 4 > data.len() {
        return None;
    }
    let count = u32::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ]) as usize;
    *offset += 4;
    Some(count)
}

pub fn encode_create_entity(buf: &mut Vec<u8>, name: &str, entity_type: &str, observations: &[String]) -> std::io::Result<()> {
    encode_str(buf, name)?;
    encode_str(buf, entity_type)?;
    buf.extend_from_slice(&(observations.len() as u32).to_le_bytes());
    for obs in observations {
        encode_str(buf, obs)?;
    }
    Ok(())
}

pub fn decode_create_entity(data: &[u8]) -> Option<(&str, &str, Vec<&str>)> {
    let mut offset = 0;
    let name = decode_str(data, &mut offset)?;
    let entity_type = decode_str(data, &mut offset)?;
    let count = decode_count(data, &mut offset)?;
    let mut observations = Vec::with_capacity(count);
    for _ in 0..count {
        observations.push(decode_str(data, &mut offset)?);
    }
    Some((name, entity_type, observations))
}

pub fn encode_create_relation(buf: &mut Vec<u8>, from: &str, to: &str, relation_type: &str) -> std::io::Result<()> {
    encode_str(buf, from)?;
    encode_str(buf, to)?;
    encode_str(buf, relation_type)
}

pub fn decode_create_relation(data: &[u8]) -> Option<(&str, &str, &str)> {
    let mut offset = 0;
    let from = decode_str(data, &mut offset)?;
    let to = decode_str(data, &mut offset)?;
    let relation_type = decode_str(data, &mut offset)?;
    Some((from, to, relation_type))
}

pub fn encode_add_observations(buf: &mut Vec<u8>, name: &str, observations: &[String]) -> std::io::Result<()> {
    encode_str(buf, name)?;
    buf.extend_from_slice(&(observations.len() as u32).to_le_bytes());
    for obs in observations {
        encode_str(buf, obs)?;
    }
    Ok(())
}

pub fn decode_add_observations(data: &[u8]) -> Option<(&str, Vec<&str>)> {
    let mut offset = 0;
    let name = decode_str(data, &mut offset)?;
    let count = decode_count(data, &mut offset)?;
    let mut observations = Vec::with_capacity(count);
    for _ in 0..count {
        observations.push(decode_str(data, &mut offset)?);
    }
    Some((name, observations))
}

pub fn encode_delete_entity(buf: &mut Vec<u8>, name: &str) -> std::io::Result<()> {
    encode_str(buf, name)
}

pub fn decode_delete_entity(data: &[u8]) -> Option<&str> {
    let mut offset = 0;
    decode_str(data, &mut offset)
}

pub fn encode_delete_observations(buf: &mut Vec<u8>, name: &str, observations: &[String]) -> std::io::Result<()> {
    encode_str(buf, name)?;
    buf.extend_from_slice(&(observations.len() as u32).to_le_bytes());
    for obs in observations {
        encode_str(buf, obs)?;
    }
    Ok(())
}

pub fn decode_delete_observations(data: &[u8]) -> Option<(&str, Vec<&str>)> {
    decode_add_observations(data)
}

pub fn encode_delete_relation(buf: &mut Vec<u8>, from: &str, to: &str, relation_type: &str) -> std::io::Result<()> {
    encode_str(buf, from)?;
    encode_str(buf, to)?;
    encode_str(buf, relation_type)
}

pub fn decode_delete_relation(data: &[u8]) -> Option<(&str, &str, &str)> {
    decode_create_relation(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn tmp_path() -> PathBuf {
        let pid = std::process::id();
        let seq = COUNTER.fetch_add(1, Ordering::SeqCst);
        std::env::temp_dir().join(format!("mcp_store_test_{pid}_{seq}.bin"))
    }

    #[test]
    fn test_write_and_replay() {
        let path = tmp_path();
        let mut store = BinaryStore::new(&path).unwrap();

        let mut buf = Vec::new();
        encode_create_entity(&mut buf, "Alice", "person", &["likes coffee".into()]).unwrap();
        store.write_record(RecordKind::CreateEntity, &buf).unwrap();

        buf.clear();
        encode_create_entity(&mut buf, "Bob", "person", &[]).unwrap();
        store.write_record(RecordKind::CreateEntity, &buf).unwrap();

        drop(store);

        let mut replayed: Vec<(RecordKind, Vec<u8>)> = Vec::new();
        let replay_store = BinaryStore::new(&path).unwrap();
        replay_store
            .replay(|kind, data| {
                replayed.push((kind, data.to_vec()));
            })
            .unwrap();

        assert_eq!(replayed.len(), 2);
        assert_eq!(replayed[0].0, RecordKind::CreateEntity);
        assert_eq!(
            decode_create_entity(&replayed[0].1).unwrap().0,
            "Alice"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut buf = Vec::new();
        encode_create_entity(
            &mut buf,
            "TestEntity",
            "test_type",
            &["obs1".into(), "obs2".into()],
        )
        .unwrap();
        let (name, etype, obs) = decode_create_entity(&buf).unwrap();
        assert_eq!(name, "TestEntity");
        assert_eq!(etype, "test_type");
        assert_eq!(obs, vec!["obs1", "obs2"]);
    }

    #[test]
    fn test_empty_file() {
        let path = tmp_path();
        let store = BinaryStore::new(&path).unwrap();
        drop(store);

        let mut count = 0;
        let replay_store = BinaryStore::new(&path).unwrap();
        replay_store.replay(|_, _| count += 1).unwrap();
        assert_eq!(count, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_all_record_kinds() {
        let path = tmp_path();
        let mut store = BinaryStore::new(&path).unwrap();
        let mut buf = Vec::new();

        // Write one of each record kind.
        encode_create_entity(&mut buf, "E1", "t1", &["o1".into()]).unwrap();
        store.write_record(RecordKind::CreateEntity, &buf).unwrap();

        buf.clear();
        encode_create_relation(&mut buf, "E1", "E2", "knows").unwrap();
        store.write_record(RecordKind::CreateRelation, &buf).unwrap();

        buf.clear();
        encode_add_observations(&mut buf, "E1", &["o2".into()]).unwrap();
        store.write_record(RecordKind::AddObservations, &buf).unwrap();

        buf.clear();
        encode_delete_entity(&mut buf, "E1").unwrap();
        store.write_record(RecordKind::DeleteEntity, &buf).unwrap();

        buf.clear();
        encode_delete_observations(&mut buf, "E1", &["o1".into()]).unwrap();
        store.write_record(RecordKind::DeleteObservations, &buf).unwrap();

        buf.clear();
        encode_delete_relation(&mut buf, "E1", "E2", "knows").unwrap();
        store.write_record(RecordKind::DeleteRelation, &buf).unwrap();

        drop(store);

        let mut kinds = Vec::new();
        let replay_store = BinaryStore::new(&path).unwrap();
        replay_store
            .replay(|kind, _| {
                kinds.push(kind);
            })
            .unwrap();

        assert_eq!(kinds.len(), 6);
        assert_eq!(kinds[0], RecordKind::CreateEntity);
        assert_eq!(kinds[1], RecordKind::CreateRelation);
        assert_eq!(kinds[2], RecordKind::AddObservations);
        assert_eq!(kinds[3], RecordKind::DeleteEntity);
        assert_eq!(kinds[4], RecordKind::DeleteObservations);
        assert_eq!(kinds[5], RecordKind::DeleteRelation);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_reopen_truncated() {
        let path = tmp_path();
        let mut store = BinaryStore::new(&path).unwrap();
        let mut buf = Vec::new();
        encode_create_entity(&mut buf, "E1", "t1", &[]).unwrap();
        store.write_record(RecordKind::CreateEntity, &buf).unwrap();
        drop(store);

        // Reopen with truncation.
        let mut store2 = BinaryStore::new(&path).unwrap();
        store2.reopen_truncated().unwrap();

        let mut buf2 = Vec::new();
        encode_create_entity(&mut buf2, "E2", "t2", &[]).unwrap();
        store2.write_record(RecordKind::CreateEntity, &buf2).unwrap();
        drop(store2);

        let mut names = Vec::new();
        let replay_store = BinaryStore::new(&path).unwrap();
        replay_store
            .replay(|_, data| {
                if let Some((name, _, _)) = decode_create_entity(data) {
                    names.push(name.to_string());
                }
            })
            .unwrap();

        // Only E2 should remain — E1 was truncated away.
        assert_eq!(names, vec!["E2"]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_encode_decode_add_observations() {
        let mut buf = Vec::new();
        encode_add_observations(&mut buf, "Alice", &["obs1".into(), "obs2".into()]).unwrap();
        let (name, obs) = decode_add_observations(&buf).unwrap();
        assert_eq!(name, "Alice");
        assert_eq!(obs, vec!["obs1", "obs2"]);
    }

    #[test]
    fn test_encode_decode_delete_entity() {
        let mut buf = Vec::new();
        encode_delete_entity(&mut buf, "ToDelete").unwrap();
        let name = decode_delete_entity(&buf).unwrap();
        assert_eq!(name, "ToDelete");
    }

    #[test]
    fn test_encode_decode_delete_observations() {
        let mut buf = Vec::new();
        encode_delete_observations(&mut buf, "Alice", &["o1".into()]).unwrap();
        let (name, obs) = decode_delete_observations(&buf).unwrap();
        assert_eq!(name, "Alice");
        assert_eq!(obs, vec!["o1"]);
    }

    #[test]
    fn test_encode_decode_delete_relation() {
        let mut buf = Vec::new();
        encode_delete_relation(&mut buf, "A", "B", "knows").unwrap();
        let (from, to, rtype) = decode_delete_relation(&buf).unwrap();
        assert_eq!(from, "A");
        assert_eq!(to, "B");
        assert_eq!(rtype, "knows");
    }

    #[test]
    fn test_sync_slot_follows_reopen_truncated() {
        // The background sync thread fsyncs through the shared slot; after a
        // reopen it must observe the *new* file handle, not the old one (D1).
        let path = tmp_path();
        let mut store = BinaryStore::new(&path).unwrap();
        let slot = Arc::clone(&store.sync_slot);
        let before = Arc::as_ptr(&slot.load_full());
        store.reopen_truncated().unwrap();
        let after = Arc::as_ptr(&slot.load_full());
        assert_ne!(before, after, "reopen must publish the new handle into the slot");
        assert!(Arc::ptr_eq(&slot, &store.sync_slot), "slot identity must be stable");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_new_with_slot_reuses_shared_cell() {
        // compact() reopens via new_with_slot(.., Some(existing_slot)) so the
        // sync thread keeps tracking the same cell across the swap (D1).
        let path = tmp_path();
        let store1 = BinaryStore::new(&path).unwrap();
        let slot = Arc::clone(&store1.sync_slot);
        let before = Arc::as_ptr(&slot.load_full());
        drop(store1);

        let store2 = BinaryStore::new_with_slot(&path, Some(Arc::clone(&slot))).unwrap();
        assert!(Arc::ptr_eq(&slot, &store2.sync_slot), "must reuse the passed slot");
        let after = Arc::as_ptr(&slot.load_full());
        assert_ne!(before, after, "reopened handle must be published into the slot");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_record_too_large() {
        let path = tmp_path();
        let mut store = BinaryStore::new(&path).unwrap();
        let huge = vec![0u8; (1 << 20) + 1];
        let result = store.write_record(RecordKind::CreateEntity, &huge);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_multiple_writes_and_replay() {
        let path = tmp_path();
        let mut store = BinaryStore::new(&path).unwrap();
        for i in 0..100 {
            let mut buf = Vec::new();
            encode_create_entity(&mut buf, &format!("E{i}"), "type", &[]).unwrap();
            store.write_record(RecordKind::CreateEntity, &buf).unwrap();
        }
        drop(store);

        let mut count = 0;
        let replay_store = BinaryStore::new(&path).unwrap();
        replay_store
            .replay(|kind, _| {
                assert_eq!(kind, RecordKind::CreateEntity);
                count += 1;
            })
            .unwrap();
        assert_eq!(count, 100);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_truncated_log_handling() {
        let path = tmp_path();
        let mut store = BinaryStore::new(&path).unwrap();
        let mut buf = Vec::new();
        encode_create_entity(&mut buf, "Alice", "person", &[]).unwrap();
        store.write_record(RecordKind::CreateEntity, &buf).unwrap();
        drop(store);

        // Truncate the file manually (simulate crash during write).
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(10).unwrap(); // cut off after MAGIC
        drop(file);

        // Replay should handle gracefully.
        let replay_store = BinaryStore::new(&path).unwrap();
        let mut count = 0;
        replay_store.replay(|_, _| count += 1).unwrap();
        assert_eq!(count, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_v1_format_backward_compat() {
        // Open an existing V1 file (MCPMEMV1 magic, no CRC), append a new record,
        // and verify all records survive replay — the CRC footer must NOT be
        // written for V1 files, and the payload_len calculation must not absorb
        // non-existent CRC bytes into the payload (D2 regression guard).
        let path = tmp_path();

        // Manually craft a V1 file: MAGIC + V1-format records (no CRC).
        let mut raw = Vec::new();
        raw.extend_from_slice(b"MCPMEMV1");

        let mut p1 = Vec::new();
        encode_create_entity(&mut p1, "Alice", "person", &[]).unwrap();
        let len1: u32 = 4 + 1 + p1.len() as u32;
        raw.extend_from_slice(&len1.to_le_bytes());
        raw.extend_from_slice(&[RecordKind::CreateEntity as u8]);
        raw.extend_from_slice(&p1);

        let mut p2 = Vec::new();
        encode_create_entity(&mut p2, "Bob", "person", &[]).unwrap();
        let len2: u32 = 4 + 1 + p2.len() as u32;
        raw.extend_from_slice(&len2.to_le_bytes());
        raw.extend_from_slice(&[RecordKind::CreateEntity as u8]);
        raw.extend_from_slice(&p2);

        std::fs::write(&path, &raw).unwrap();

        // Open with BinaryStore — must detect V1 magic, set has_crc=false.
        let mut store = BinaryStore::new(&path).unwrap();

        // Append a third record — must NOT add CRC footer.
        let mut p3 = Vec::new();
        encode_create_entity(&mut p3, "Charlie", "person", &[]).unwrap();
        store.write_record(RecordKind::CreateEntity, &p3).unwrap();
        store.flush().unwrap();
        drop(store);

        // File size: V1 records have no CRC, so each takes 5 + payload bytes.
        let expected_size = raw.len() as u64 + (5 + p3.len()) as u64;
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            expected_size,
            "V1 file must not grow by CRC bytes after write"
        );

        // Replay must decode all three records correctly.
        let replay_store = BinaryStore::new(&path).unwrap();
        let mut names = Vec::new();
        replay_store
            .replay(|_, data| {
                if let Some((name, _, _)) = decode_create_entity(data) {
                    names.push(name.to_string());
                }
            })
            .unwrap();
        assert_eq!(names, vec!["Alice", "Bob", "Charlie"]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_crc_detects_corrupted_payload() {
        // Corrupt a byte in a V2 file's payload and verify replay detects the
        // CRC mismatch and stops — the corrupted record must NOT be returned
        // to the callback (D2).
        let path = tmp_path();
        let mut store = BinaryStore::new(&path).unwrap();

        let mut buf = Vec::new();
        encode_create_entity(&mut buf, "Alice", "person", &["likes coffee".into()]).unwrap();
        store.write_record(RecordKind::CreateEntity, &buf).unwrap();
        store.flush_and_sync().unwrap();
        drop(store);

        // Read the raw file, corrupt one byte inside the payload.
        let mut data = std::fs::read(&path).unwrap();
        // Layout: MAGIC(8) + Len(4) + Kind(1) + payload(N) + CRC(4)
        // Flip a bit at the payload midpoint.
        let payload_start = 8 + 4 + 1;
        let corrupt_pos = payload_start + (data.len() - payload_start - 4) / 2;
        data[corrupt_pos] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        // Replay must detect the CRC mismatch and stop before the callback.
        let replay_store = BinaryStore::new(&path).unwrap();
        let mut count = 0;
        replay_store
            .replay(|_, _| count += 1)
            .expect("CRC mismatch must return Ok (torn-tail semantics)");
        assert_eq!(count, 0, "corrupted record must not reach callback");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_crc_detects_corrupted_middle_record() {
        // With two valid records, corrupting the second record's payload must
        // preserve the first record and stop cleanly before the second (D2).
        let path = tmp_path();
        let mut store = BinaryStore::new(&path).unwrap();

        let mut buf1 = Vec::new();
        encode_create_entity(&mut buf1, "Alice", "person", &[]).unwrap();
        store.write_record(RecordKind::CreateEntity, &buf1).unwrap();

        let mut buf2 = Vec::new();
        encode_create_entity(&mut buf2, "Bob", "person", &[]).unwrap();
        store.write_record(RecordKind::CreateEntity, &buf2).unwrap();

        store.flush_and_sync().unwrap();
        drop(store);

        // Corrupt a byte in the second record's payload.
        let mut data = std::fs::read(&path).unwrap();
        // First record ends at: 8 + 4 + 1 + payload1.len() + 4
        let rec1_end = 8 + 4 + 1 + buf1.len() + 4;
        // Second record payload starts at: rec1_end + 4 (len) + 1 (kind)
        let rec2_payload_start = rec1_end + 4 + 1;
        data[rec2_payload_start + 2] ^= 0xFF; // corrupt 3rd byte of payload
        std::fs::write(&path, &data).unwrap();

        let replay_store = BinaryStore::new(&path).unwrap();
        let mut names = Vec::new();
        replay_store
            .replay(|_, data| {
                if let Some((name, _, _)) = decode_create_entity(data) {
                    names.push(name.to_string());
                }
            })
            .expect("CRC mismatch of middle record must not hard-error");
        // Alice survives; Bob is discarded.
        assert_eq!(names, vec!["Alice"]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_torn_record_mid_stream_recovers_prefix() {
        // A crash that writes a record's length prefix but only part of its
        // body must not brick the log: replay should return the records written
        // before the torn one and stop cleanly (D2).
        let path = tmp_path();
        let mut store = BinaryStore::new(&path).unwrap();
        let mut buf = Vec::new();
        encode_create_entity(&mut buf, "Alice", "person", &["likes coffee".into()]).unwrap();
        store.write_record(RecordKind::CreateEntity, &buf).unwrap();
        store.flush_and_sync().unwrap();
        let good_len = std::fs::metadata(&path).unwrap().len();

        // Append a second record, then chop it in half to simulate a torn write
        // (length prefix present, payload incomplete).
        buf.clear();
        encode_create_entity(&mut buf, "Bob", "person", &["drinks tea".into()]).unwrap();
        store.write_record(RecordKind::CreateEntity, &buf).unwrap();
        store.flush_and_sync().unwrap();
        drop(store);

        let full_len = std::fs::metadata(&path).unwrap().len();
        // Cut somewhere inside the second record's body.
        let torn_len = good_len + (full_len - good_len) / 2;
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(torn_len).unwrap();
        drop(file);

        let replay_store = BinaryStore::new(&path).unwrap();
        let mut names = Vec::new();
        replay_store
            .replay(|_, data| {
                if let Some((name, _, _)) = decode_create_entity(data) {
                    names.push(name.to_string());
                }
            })
            .expect("torn tail must not be a hard error");
        // Only the fully-written first record survives.
        assert_eq!(names, vec!["Alice"]);
        let _ = std::fs::remove_file(&path);
    }
}
