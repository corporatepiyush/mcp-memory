use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"MCPMEMV1";
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
            _ => return None,
        })
    }
}

pub struct BinaryStore {
    writer: BufWriter<File>,
    path: PathBuf,
}

impl BinaryStore {
    pub const fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn new(path: &Path) -> std::io::Result<Self> {
        let exists = path.exists();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(false)
            .open(path)?;

        let mut writer = BufWriter::with_capacity(65536, file);

        if !exists {
            writer.write_all(MAGIC)?;
            writer.flush()?;
        }

        Ok(Self {
            writer,
            path: path.to_path_buf(),
        })
    }

    pub fn write_record(&mut self, kind: RecordKind, payload: &[u8]) -> std::io::Result<()> {
        let total_len = 4 + 1 + payload.len();
        if total_len as u32 > MAX_RECORD_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Record too large",
            ));
        }
        self.writer.write_all(&(total_len as u32).to_le_bytes())?;
        self.writer.write_all(&[kind as u8])?;
        self.writer.write_all(payload)?;
        Ok(())
    }

    pub fn flush_and_sync(&mut self) -> std::io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_data()
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
            Ok(()) => {
                if &magic != MAGIC {
                    return Ok(());
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        }

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
            let payload_len = total_len - 5;

            let mut kind_buf = [0u8; 1];
            reader.read_exact(&mut kind_buf)?;
            let kind_val = kind_buf[0];

            payload_buf.clear();
            payload_buf.resize(payload_len, 0);
            if payload_len > 0 {
                reader.read_exact(&mut payload_buf)?;
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
        let mut writer = BufWriter::with_capacity(65536, file);
        writer.write_all(MAGIC)?;
        writer.flush()?;
        self.writer = writer;
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
}
