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
        let mut texts = Vec::with_capacity(2 + observations.len());
        texts.push(name);
        texts.push(entity_type);
        texts.extend_from_slice(observations);
        self.insert_tokens(interner, entity_idx, &texts);
    }

    /// Incrementally index additional strings (e.g. newly added observations)
    /// for an entity that is *already* indexed, without removing and rebuilding
    /// its existing entries (P3). Token entries that already exist are deduped
    /// during the merge, so calling this with text that overlaps existing
    /// tokens is safe.
    pub fn index_additional(
        &mut self,
        interner: &mut crate::intern::StringInterner,
        entity_idx: u32,
        texts: &[StrId],
    ) {
        self.insert_tokens(interner, entity_idx, texts);
    }

    /// Remove all entries for a given entity (before re-indexing).
    pub fn remove_entity(&mut self, entity_idx: u32) {
        self.entries.retain(|&(_, idx)| idx != entity_idx);
    }

    /// Search for entities whose name/type/observation tokens match `query`
    /// case-insensitively by **prefix** (`"cof"` matches `"coffee"`).
    ///
    /// Note: this is an O(n) scan over every index entry. The binary-search
    /// step below only narrows exact-token hits, but the subsequent prefix scan
    /// already covers those (an exact match is also a prefix match), so the scan
    /// dominates — do not read the binary search as making this sublinear.
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

    /// Like [`search`], but returns `(entity_idx, score)` pairs sorted by
    /// descending score (then ascending idx for stability). `score` is the
    /// number of indexed-token hits the entity accumulated for the query —
    /// a cheap relevance proxy so callers can surface the best matches first.
    ///
    /// The scan is a single linear pass over the flat `entries` vec (no
    /// per-entity allocation until the final compaction), keeping it
    /// cache-friendly. A small `Vec<(idx, score)>` is gathered then sorted.
    pub fn search_ranked(&self, query: &str, interner: &crate::intern::StringInterner) -> Vec<(u32, u32)> {
        if query.is_empty() || self.entries.is_empty() {
            return Vec::new();
        }

        let lower_query: String = query.to_ascii_lowercase();
        let qbytes = lower_query.as_bytes();
        let qlen = qbytes.len();

        // Exact-token id (if the query is itself an interned token) lets us
        // score exact hits without a string compare.
        let exact_id = interner.get_optional(&lower_query);

        // (idx, score) gathered in one pass, idx-major so equal idxs are adjacent.
        let mut hits: Vec<(u32, u32)> = Vec::new();
        for &(token_id, entity_idx) in &self.entries {
            let matches = if Some(token_id) == exact_id {
                true
            } else {
                let token = interner.lookup(token_id);
                token.len() >= qlen && token.as_bytes().starts_with(qbytes)
            };
            if matches {
                match hits.last_mut() {
                    Some(last) if last.0 == entity_idx => last.1 += 1,
                    _ => hits.push((entity_idx, 1)),
                }
            }
        }

        // entries are sorted by (token, idx), so a single entity_idx may appear
        // in non-adjacent groups (once per matching token). Merge by idx, then
        // rank by score desc.
        hits.sort_unstable_by_key(|&(idx, _)| idx);
        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(hits.len());
        for (idx, score) in hits {
            match merged.last_mut() {
                Some(last) if last.0 == idx => last.1 += score,
                _ => merged.push((idx, score)),
            }
        }
        merged.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        merged
    }

    /// Tokenize every string in `texts`, collect the resulting
    /// `(token, entity_idx)` entries, and merge them into the sorted `entries`
    /// vec in a single O(N + K) pass (P2). This replaces the previous
    /// per-token `Vec::insert`, which shifted the tail on every token and made
    /// indexing O(K × N).
    fn insert_tokens(
        &mut self,
        interner: &mut crate::intern::StringInterner,
        entity_idx: u32,
        texts: &[StrId],
    ) {
        let mut additions: Vec<(StrId, u32)> = Vec::new();
        for &text in texts {
            let s = interner.lookup(text);
            if s.is_empty() {
                continue;
            }
            self.lower_buf.clear();
            self.lower_buf.extend(s.bytes().map(|b| b.to_ascii_lowercase()));
            let lowered = unsafe { std::str::from_utf8_unchecked(&self.lower_buf) };
            let tokens: Vec<&str> =
                lowered.split_whitespace().filter(|t| !t.is_empty()).collect();
            for token in tokens {
                additions.push((interner.intern(token), entity_idx));
            }
        }
        if additions.is_empty() {
            return;
        }
        additions.sort_unstable();
        additions.dedup();
        self.merge_entries(&additions);
    }

    /// Merge a pre-sorted, deduped slice of new entries into `entries`
    /// (also sorted and deduped) in one linear pass. Entries already present
    /// are skipped, preserving the no-duplicate invariant.
    fn merge_entries(&mut self, additions: &[(StrId, u32)]) {
        let old = std::mem::take(&mut self.entries);
        let mut merged = Vec::with_capacity(old.len() + additions.len());
        let (mut i, mut j) = (0, 0);
        while i < old.len() && j < additions.len() {
            match old[i].cmp(&additions[j]) {
                std::cmp::Ordering::Less => {
                    merged.push(old[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    merged.push(additions[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    merged.push(old[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        merged.extend_from_slice(&old[i..]);
        merged.extend_from_slice(&additions[j..]);
        self.entries = merged;
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
    fn test_search_ranked_orders_by_score() {
        let mut interner = StringInterner::new();
        let mut index = SearchIndex::new();
        // Entity 0: "coffee" appears in both name and observation → score 2.
        let n0 = interner.intern("coffee");
        let t0 = interner.intern("drink");
        let o0 = interner.intern("coffee beans");
        index.index_entity(&mut interner, 0, n0, t0, &[o0]);
        // Entity 1: "coffee" appears once (observation only) → score 1.
        let n1 = interner.intern("Bob");
        let t1 = interner.intern("person");
        let o1 = interner.intern("likes coffee");
        index.index_entity(&mut interner, 1, n1, t1, &[o1]);

        let ranked = index.search_ranked("coffee", &interner);
        assert_eq!(ranked.len(), 2);
        // Higher score first.
        assert_eq!(ranked[0].0, 0);
        assert!(ranked[0].1 >= ranked[1].1);
        assert_eq!(ranked[1].0, 1);
    }

    #[test]
    fn test_search_ranked_empty_query() {
        let interner = StringInterner::new();
        let index = SearchIndex::new();
        assert!(index.search_ranked("", &interner).is_empty());
    }

    #[test]
    fn test_search_is_prefix_not_substring() {
        let mut interner = StringInterner::new();
        let mut index = SearchIndex::new();
        let name = interner.intern("coffee");
        let typ = interner.intern("drink");
        index.index_entity(&mut interner, 0, name, typ, &[]);
        // Prefix of a token matches.
        assert_eq!(index.search("cof", &interner), vec![0]);
        assert_eq!(index.search("coffee", &interner), vec![0]);
        // Interior substrings do NOT match — this documents real behavior, not
        // the "substring search" the docs once claimed.
        assert!(index.search("ffee", &interner).is_empty());
        assert!(index.search("offe", &interner).is_empty());
    }

    #[test]
    fn test_search_ranked_is_prefix_not_substring() {
        let mut interner = StringInterner::new();
        let mut index = SearchIndex::new();
        let name = interner.intern("coffee");
        let typ = interner.intern("drink");
        index.index_entity(&mut interner, 0, name, typ, &[]);
        assert!(!index.search_ranked("cof", &interner).is_empty());
        assert!(index.search_ranked("ffee", &interner).is_empty());
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
