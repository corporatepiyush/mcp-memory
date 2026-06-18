use ahash::RandomState;
use std::fmt;

// Ctrl-byte sentinels for the dedup hash table.
const EMPTY: u8 = 0xFF;
// Stored h2 values are 0x00-0x7F — never collide with 0xFF.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct StrId(u32);

impl StrId {
    pub const EMPTY: StrId = StrId(u32::MAX);

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == u32::MAX
    }

    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Display for StrId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "<empty>")
        } else {
            write!(f, "StrId({})", self.0)
        }
    }
}

/// A 7-bit hash stamp extracted from the full 64-bit hash.
/// Always has bit 7 clear, so it never collides with `EMPTY` (0xFF).
#[inline(always)]
const fn h2(hash: u64) -> u8 {
    (hash & 0x7F) as u8
}

/// Starting bucket index derived from the upper bits of the hash.
#[inline(always)]
const fn h1(hash: u64, mask: usize) -> usize {
    ((hash >> 7) as usize) & mask
}

#[derive(Clone)]
pub struct StringInterner {
    arena: String,
    offsets: Vec<u32>,
    // Dedup hash table — ctrl-byte buckets with parallel arrays.
    ctrl: Vec<u8>,
    table_hashes: Vec<u64>,
    table_ids: Vec<StrId>,
    table_mask: usize,
    count: usize,
    hasher: RandomState,
}

impl StringInterner {
    pub fn new() -> Self {
        const CAP: usize = 256;
        Self {
            arena: String::with_capacity(8192),
            offsets: vec![0],
            ctrl: vec![EMPTY; CAP],
            table_hashes: vec![0; CAP],
            table_ids: vec![StrId::EMPTY; CAP],
            table_mask: CAP - 1,
            count: 0,
            hasher: RandomState::new(),
        }
    }

    pub fn with_capacity(string_capacity: usize, estimated_strings: usize) -> Self {
        let cap = estimated_strings.next_power_of_two().max(64);
        Self {
            arena: String::with_capacity(string_capacity),
            offsets: vec![0],
            ctrl: vec![EMPTY; cap],
            table_hashes: vec![0; cap],
            table_ids: vec![StrId::EMPTY; cap],
            table_mask: cap - 1,
            count: 0,
            hasher: RandomState::new(),
        }
    }

    #[inline]
    pub fn intern(&mut self, s: &str) -> StrId {
        if s.is_empty() {
            return StrId::EMPTY;
        }
        let hash = self.hasher.hash_one(s);
        let stamp = h2(hash);
        let mask = self.table_mask;
        let mut idx = h1(hash, mask);

        loop {
            let c = &self.ctrl[idx];
            if *c & 0x80 != 0 {
                // Empty slot → insert new string.
                let id = self.offsets.len() as u32 - 1;
                self.arena.push_str(s);
                self.offsets.push(self.arena.len() as u32);
                self.ctrl[idx] = stamp;
                self.table_hashes[idx] = hash;
                self.table_ids[idx] = StrId(id);
                self.count += 1;
                if self.count * 4 > self.ctrl.len() * 3 {
                    self.grow();
                }
                return StrId(id);
            }
            if *c == stamp && self.table_hashes[idx] == hash {
                let existing = self.table_ids[idx].0;
                let start = self.offsets[existing as usize] as usize;
                let end = self.offsets[existing as usize + 1] as usize;
                let existing_str = unsafe { self.arena.get_unchecked(start..end) };
                if existing_str == s {
                    return StrId(existing);
                }
            }
            idx = (idx + 1) & mask;
        }
    }

    /// Look up a string in the dedup table without inserting.
    /// Returns `StrId` if the string already exists, `None` otherwise.
    #[inline]
    pub fn get_optional(&self, s: &str) -> Option<StrId> {
        if s.is_empty() {
            return Some(StrId::EMPTY);
        }
        let hash = self.hasher.hash_one(s);
        let stamp = h2(hash);
        let mask = self.table_mask;
        let mut idx = h1(hash, mask);

        for _ in 0..self.ctrl.len() {
            let c = self.ctrl[idx];
            if c & 0x80 != 0 {
                return None;
            }
            if c == stamp && self.table_hashes[idx] == hash {
                let existing = self.table_ids[idx].0;
                let start = self.offsets[existing as usize] as usize;
                let end = self.offsets[existing as usize + 1] as usize;
                if unsafe { self.arena.get_unchecked(start..end) == s } {
                    return Some(StrId(existing));
                }
            }
            idx = (idx + 1) & mask;
        }
        None
    }

    #[inline]
    pub fn lookup(&self, id: StrId) -> &str {
        if id.is_empty() {
            return "";
        }
        let start = self.offsets[id.0 as usize] as usize;
        let end = self.offsets[id.0 as usize + 1] as usize;
        unsafe { self.arena.get_unchecked(start..end) }
    }

    /// Recompute the hash of an interned string on demand. The per-string hash
    /// is no longer stored (M4) — it is cheaply re-derived from the interned
    /// bytes with the same hasher, saving 8 bytes per unique string. Used to
    /// key the (separate) entity name table.
    #[inline]
    pub fn get_hash(&self, id: StrId) -> u64 {
        if id.is_empty() {
            return 0;
        }
        self.hasher.hash_one(self.lookup(id))
    }

    pub const fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub const fn total_bytes(&self) -> usize {
        self.arena.len()
    }

    fn grow(&mut self) {
        let new_size = self.ctrl.len() * 2;
        let new_mask = new_size - 1;
        let mut new_ctrl = vec![EMPTY; new_size];
        let mut new_hashes = vec![0u64; new_size];
        let mut new_ids = vec![StrId::EMPTY; new_size];

        for i in 0..self.ctrl.len() {
            if self.ctrl[i] & 0x80 == 0 {
                let hash = self.table_hashes[i];
                let stamp = h2(hash);
                let mut idx = h1(hash, new_mask);
                while new_ctrl[idx] & 0x80 == 0 {
                    idx = (idx + 1) & new_mask;
                }
                new_ctrl[idx] = stamp;
                new_hashes[idx] = hash;
                new_ids[idx] = self.table_ids[i];
            }
        }

        self.ctrl = new_ctrl;
        self.table_hashes = new_hashes;
        self.table_ids = new_ids;
        self.table_mask = new_mask;
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_empty() {
        let mut interner = StringInterner::new();
        assert!(interner.intern("").is_empty());
    }

    #[test]
    fn test_intern_dedup() {
        let mut interner = StringInterner::new();
        let a = interner.intern("hello");
        let b = interner.intern("hello");
        assert_eq!(a, b);
    }

    #[test]
    fn test_intern_unique() {
        let mut interner = StringInterner::new();
        let a = interner.intern("hello");
        let b = interner.intern("world");
        assert_ne!(a, b);
    }

    #[test]
    fn test_lookup() {
        let mut interner = StringInterner::new();
        let id = interner.intern("hello world");
        assert_eq!(interner.lookup(id), "hello world");
    }

    #[test]
    fn test_large_intern() {
        let mut interner = StringInterner::new();
        let mut ids = Vec::new();
        for i in 0..1000 {
            let s = format!("string_{i}");
            ids.push(interner.intern(&s));
        }
        for (i, &id) in ids.iter().enumerate() {
            let expected = format!("string_{i}");
            assert_eq!(interner.lookup(id), expected);
        }
        assert_eq!(interner.len(), 1000);
    }

    #[test]
    fn test_lookup_empty_id() {
        let interner = StringInterner::new();
        assert_eq!(interner.lookup(StrId::EMPTY), "");
    }

    #[test]
    fn test_get_hash_empty_id() {
        let interner = StringInterner::new();
        assert_eq!(interner.get_hash(StrId::EMPTY), 0);
    }

    #[test]
    fn test_get_hash_consistency() {
        let mut interner = StringInterner::new();
        let id = interner.intern("consistent");
        let hash1 = interner.get_hash(id);
        let id2 = interner.intern("consistent");
        assert_eq!(id, id2);
        let hash2 = interner.get_hash(id2);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_total_bytes() {
        let mut interner = StringInterner::new();
        assert_eq!(interner.total_bytes(), 0);
        interner.intern("abc");
        assert_eq!(interner.total_bytes(), 3);
        interner.intern("defg");
        assert_eq!(interner.total_bytes(), 7);
        interner.intern("abc"); // dedup, no new bytes
        assert_eq!(interner.total_bytes(), 7);
    }

    #[test]
    fn test_intern_empty_via_lookup() {
        let mut interner = StringInterner::new();
        let e = interner.intern("");
        assert!(e.is_empty());
        // Interning empty again should still return EMPTY.
        let e2 = interner.intern("");
        assert!(e2.is_empty());
    }

    #[test]
    fn test_grow_triggers() {
        let mut interner = StringInterner::with_capacity(4096, 16);
        // Insert enough strings to force a grow.
        for i in 0..100 {
            interner.intern(&format!("grow_test_{i}"));
        }
        assert_eq!(interner.len(), 100);
        // Verify all strings are still accessible.
        for i in 0..100 {
            let id = interner.intern(&format!("grow_test_{i}"));
            assert_eq!(interner.lookup(id), format!("grow_test_{i}"));
        }
    }

    #[test]
    fn test_many_dedup_same_string() {
        let mut interner = StringInterner::new();
        let id = interner.intern("same");
        for _ in 0..1000 {
            let new_id = interner.intern("same");
            assert_eq!(new_id, id);
        }
        assert_eq!(interner.len(), 1);
    }

    #[test]
    fn test_interner_with_capacity() {
        let mut interner = StringInterner::with_capacity(1024, 50);
        assert_eq!(interner.len(), 0);
        for i in 0..50 {
            interner.intern(&format!("cap_test_{i}"));
        }
        assert_eq!(interner.len(), 50);
    }

    #[test]
    fn test_case_sensitive_dedup() {
        let mut interner = StringInterner::new();
        let a = interner.intern("Hello");
        let b = interner.intern("hello");
        assert_ne!(a, b); // case-sensitive dedup
    }

    #[test]
    fn test_default_impl() {
        let mut interner: StringInterner = Default::default();
        let id = interner.intern("default");
        assert_eq!(interner.lookup(id), "default");
    }

    #[test]
    fn test_is_empty_method() {
        let mut interner = StringInterner::new();
        assert!(interner.is_empty());
        interner.intern("x");
        assert!(!interner.is_empty());
    }
}
