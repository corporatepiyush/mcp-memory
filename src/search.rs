use crate::intern::StrId;

/// Inverted word index for fast entity search.
///
/// For each entity, we tokenize its name, type, and observations,
/// store each token → set of matching entity indices.
///
/// Uses a flat `Vec<(StrId, u32)>` sorted by (token, entity_idx)
/// for cache-friendly lookups via binary search.
pub struct SearchIndex {
    // Sorted by (token, entity_idx), no duplicates.
    entries: Vec<(StrId, u32)>,
    lower_buf: Vec<u8>,
}

impl SearchIndex {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            lower_buf: Vec::with_capacity(256),
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Index a single entity by its name, type, and observations.
    /// All strings must already be interned.
    /// `entity_idx` is the position in the entity storage vec.
    pub fn index_entity(
        &mut self,
        interner: &mut crate::intern::StringInterner,
        entity_idx: u32,
        name: StrId,
        entity_type: StrId,
        observations: &[StrId],
    ) {
        self.tokenize_and_insert(interner, entity_idx, name);
        self.tokenize_and_insert(interner, entity_idx, entity_type);
        for &obs in observations {
            self.tokenize_and_insert(interner, entity_idx, obs);
        }
    }

    /// Remove all entries for a given entity (before re-indexing).
    pub fn remove_entity(&mut self, entity_idx: u32) {
        self.entries.retain(|&(_, idx)| idx != entity_idx);
    }

    /// Search for entities matching the query.
    /// Uses binary search for exact token matches (O(log n + matches))
    /// and falls back to prefix matching for partial queries.
    pub fn search(&self, query: &str, interner: &crate::intern::StringInterner) -> Vec<u32> {
        if query.is_empty() || self.entries.is_empty() {
            return Vec::new();
        }

        let lower_query: String = query.to_ascii_lowercase();

        // Fast path: exact token match via binary search
        let mut matched = if let Some(token_id) = interner.get_optional(&lower_query) {
            let range_begin = self.entries.binary_search_by(|(t, _)| t.cmp(&token_id));
            let range_end = self.entries.binary_search_by(|(t, _)| {
                if *t <= token_id { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater }
            });
            if let (Ok(begin), Err(end)) = (range_begin, range_end) {
                self.entries[begin..end].iter().map(|&(_, idx)| idx).collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Prefix match scan for tokens starting with the query
        for &(token_id, entity_idx) in &self.entries {
            if matched.last().is_none_or(|&last| last != entity_idx) {
                let token = interner.lookup(token_id);
                if token.len() >= lower_query.len()
                    && token.as_bytes().starts_with(lower_query.as_bytes())
                {
                    matched.push(entity_idx);
                }
            }
        }

        matched.sort_unstable();
        matched.dedup();
        matched
    }

    fn tokenize_and_insert(
        &mut self,
        interner: &mut crate::intern::StringInterner,
        entity_idx: u32,
        text: StrId,
    ) {
        let s = interner.lookup(text);
        if s.is_empty() {
            return;
        }

        self.lower_buf.clear();
        self.lower_buf.extend(s.bytes().map(|b| b.to_ascii_lowercase()));
        let lowered = unsafe { std::str::from_utf8_unchecked(&self.lower_buf) };

        let tokens: Vec<&str> = lowered.split_whitespace().filter(|t| !t.is_empty()).collect();
        let interned: Vec<StrId> = tokens.iter().map(|t| interner.intern(t)).collect();

        for token_id in &interned {
            self.insert_entry(*token_id, entity_idx);
        }
    }

    fn insert_entry(&mut self, token: StrId, entity_idx: u32) {
        let entry = (token, entity_idx);
        let pos = match self.entries.binary_search(&entry) {
            Ok(_) => return,
            Err(pos) => pos,
        };
        self.entries.insert(pos, entry);
    }
}

impl Default for SearchIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::StringInterner;

    #[test]
    fn test_index_and_search() {
        let mut interner = StringInterner::new();
        let mut index = SearchIndex::new();

        let alice_name = interner.intern("Alice");
        let alice_type = interner.intern("person");
        let alice_obs = interner.intern("likes coffee");

        index.index_entity(&mut interner, 0, alice_name, alice_type, &[alice_obs]);

        let bob_name = interner.intern("Bob");
        let bob_type = interner.intern("person");
        let bob_obs = interner.intern("drinks tea");

        index.index_entity(&mut interner, 1, bob_name, bob_type, &[bob_obs]);

        let results = index.search("coffee", &interner);
        assert_eq!(results, vec![0]);

        let results = index.search("person", &interner);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_remove_entity() {
        let mut interner = StringInterner::new();
        let mut index = SearchIndex::new();

        let name = interner.intern("Test");
        let typ = interner.intern("type");

        index.index_entity(&mut interner, 0, name, typ, &[]);
        assert!(!index.is_empty());

        index.remove_entity(0);
        assert!(index.entries.iter().all(|&(_, idx)| idx != 0));
    }

    #[test]
    fn test_search_empty_query() {
        let interner = StringInterner::new();
        let index = SearchIndex::new();
        assert!(index.search("", &interner).is_empty());
    }

    #[test]
    fn test_search_no_match() {
        let mut interner = StringInterner::new();
        let mut index = SearchIndex::new();
        let name = interner.intern("Alice");
        let typ = interner.intern("person");
        index.index_entity(&mut interner, 0, name, typ, &[]);
        assert!(index.search("zzzzzz", &interner).is_empty());
    }

    #[test]
    fn test_search_case_insensitive() {
        let mut interner = StringInterner::new();
        let mut index = SearchIndex::new();
        let name = interner.intern("Alice");
        let typ = interner.intern("person");
        index.index_entity(&mut interner, 0, name, typ, &[]);
        let results = index.search("ALICE", &interner);
        assert_eq!(results, vec![0]);
    }

    #[test]
    fn test_search_partial_substring() {
        let mut interner = StringInterner::new();
        let mut index = SearchIndex::new();
        let name = interner.intern("Alice");
        let typ = interner.intern("person");
        index.index_entity(&mut interner, 0, name, typ, &[]);
        let results = index.search("Ali", &interner);
        assert_eq!(results, vec![0]);
    }

    #[test]
    fn test_multi_token_search() {
        let mut interner = StringInterner::new();
        let mut index = SearchIndex::new();
        let obs = interner.intern("likes drinking coffee");
        let alice = interner.intern("Alice");
        let person = interner.intern("person");
        index.index_entity(
            &mut interner,
            0,
            alice,
            person,
            &[obs],
        );
        assert_eq!(index.search("likes", &interner), vec![0]);
        assert_eq!(index.search("drinking", &interner), vec![0]);
        assert_eq!(index.search("coffee", &interner), vec![0]);
    }

    #[test]
    fn test_remove_then_reindex() {
        let mut interner = StringInterner::new();
        let mut index = SearchIndex::new();
        let name = interner.intern("Alice");
        let typ = interner.intern("person");
        index.index_entity(&mut interner, 0, name, typ, &[]);

        assert_eq!(index.search("Alice", &interner).len(), 1);
        index.remove_entity(0);
        assert!(index.search("Alice", &interner).is_empty());

        index.index_entity(&mut interner, 0, name, typ, &[]);
        assert_eq!(index.search("Alice", &interner).len(), 1);
    }

    #[test]
    fn test_query_longer_than_token() {
        let mut interner = StringInterner::new();
        let mut index = SearchIndex::new();
        let name = interner.intern("Alice");
        let person = interner.intern("person");
        index.index_entity(&mut interner, 0, name, person, &[]);
        assert!(index.search("AliceInWonderland", &interner).is_empty());
    }

    #[test]
    fn test_empty_index() {
        let interner = StringInterner::new();
        let index = SearchIndex::new();
        assert!(index.search("anything", &interner).is_empty());
    }

    #[test]
    fn test_clear_index() {
        let mut interner = StringInterner::new();
        let mut index = SearchIndex::new();
        let name = interner.intern("Alice");
        let person = interner.intern("person");
        index.index_entity(&mut interner, 0, name, person, &[]);
        assert!(!index.is_empty());
        index.clear();
        assert!(index.is_empty());
        assert!(index.search("Alice", &interner).is_empty());
    }
}
