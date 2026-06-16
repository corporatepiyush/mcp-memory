use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use mcp_memory::kg::{Direction, KnowledgeGraph};
use mcp_memory::types::{Entity, Relation};
use rand::prelude::*;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_path() -> String {
    let pid = std::process::id();
    let seq = COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("/tmp/mcp_fuzzy_{pid}_{seq}.bin")
}

fn cleanup(path: &str) {
    let _ = std::fs::remove_file(path);
}

// =========================================================================
// Helpers: deterministic PRNG with controlled RNG
// =========================================================================

fn random_name(rng: &mut impl Rng, len: usize) -> String {
    let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyz0123456789_".chars().collect();
    (0..len).map(|_| chars[rng.gen_range(0..chars.len())]).collect()
}

fn random_entity(rng: &mut impl Rng) -> Entity {
    let name_len = rng.gen_range(3..12);
    let name = random_name(rng, name_len);
    let etype_len = rng.gen_range(3..8);
    let etype = random_name(rng, etype_len);
    let num_obs = rng.gen_range(0..5);
    let observations: Vec<String> = (0..num_obs)
        .map(|_| {
            let obs_len = rng.gen_range(2..15);
            random_name(rng, obs_len)
        })
        .collect();
    Entity { name, entity_type: etype, observations }
}

fn known_entity(rng: &mut impl Rng, names: &[String]) -> String {
    names[rng.gen_range(0..names.len())].clone()
}

// =========================================================================
// Invariant: verify graph consistency
// =========================================================================

fn check_invariants(kg: &KnowledgeGraph, live_names: &HashSet<String>) {
    let stats = kg.graph_stats();
    let expected_count = live_names.len();
    assert_eq!(
        stats["entities"].as_u64().unwrap() as usize,
        expected_count,
        "entity count mismatch"
    );

    // Every live entity must be fetchable
    for name in live_names {
        let entity = kg.get_entity(name);
        assert!(entity.is_some(), "live entity '{name}' not fetchable");
        let e = entity.unwrap();
        assert_eq!(&e.name, name);
    }

    // get_entity for a ghost must return None
    let ghost = format!("__ghost_{}", random_name(&mut thread_rng(), 6));
    if !live_names.contains(&ghost) {
        assert!(kg.get_entity(&ghost).is_none());
    }

    // All relation from/to must reference live entities
    let rels = kg.search_relations(None, None, None);
    for rel in &rels {
        assert!(
            live_names.contains(&rel.from),
            "relation from '{0}' not live",
            rel.from
        );
        assert!(
            live_names.contains(&rel.to),
            "relation to '{0}' not live",
            rel.to
        );
    }

    // read_graph must return the same entity count
    let graph = kg.read_graph();
    assert_eq!(graph.entities.len(), expected_count);
}

// =========================================================================
// 1. Random CRUD sequences
// =========================================================================

#[test]
fn test_random_crud_sequence_small() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([42u8; 32]);
    let mut live_names: HashSet<String> = HashSet::new();
    let mut all_relations: Vec<Relation> = Vec::new();

    for _ in 0..200 {
        let op: u32 = rng.gen_range(0..100);
        match op {
            0..=35 => {
                let entity = random_entity(&mut rng);
                if !live_names.contains(&entity.name) {
                    let created = kg.create_entities(&[entity.clone()]).unwrap();
                    if !created.is_empty() {
                        live_names.insert(entity.name.clone());
                    }
                }
            }
            36..=50 => {
                if !live_names.is_empty() {
                    let delete_count = rng.gen_range(1..=3.min(live_names.len()));
                    let names: Vec<String> = live_names.iter()
                        .choose_multiple(&mut rng, delete_count)
                        .into_iter()
                        .cloned()
                        .collect();
                    kg.delete_entities(&names).unwrap();
                    for n in &names {
                        live_names.remove(n);
                    }
                    all_relations.retain(|r| live_names.contains(&r.from) && live_names.contains(&r.to));
                }
            }
            51..=65 => {
                if !live_names.is_empty() {
                    let from = known_entity(&mut rng, &live_names.iter().cloned().collect::<Vec<_>>());
                    let to = known_entity(&mut rng, &live_names.iter().cloned().collect::<Vec<_>>());
                    if from != to {
                        let rtype = random_name(&mut rng, 4);
                        let rel = Relation { from, to, relation_type: rtype };
                        let created = kg.create_relations(&[rel.clone()]).unwrap();
                        if !created.is_empty() {
                            all_relations.push(rel);
                        }
                    }
                }
            }
            66..=75 => {
                if !all_relations.is_empty() {
                    let del_count = rng.gen_range(1..=2.min(all_relations.len()));
                    let to_del: Vec<Relation> = all_relations.iter()
                        .choose_multiple(&mut rng, del_count)
                        .into_iter()
                        .cloned()
                        .collect();
                    kg.delete_relations(&to_del).unwrap();
                    all_relations.retain(|r| !to_del.iter().any(|d| d == r));
                }
            }
            76..=85 => {
                if !live_names.is_empty() {
                    let name = known_entity(&mut rng, &live_names.iter().cloned().collect::<Vec<_>>());
                    let num_obs = rng.gen_range(1..=3);
                    let new_obs: Vec<String> = (0..num_obs)
                        .map(|_| random_name(&mut rng, 5))
                        .collect();
                    let added = kg.add_observations(&name, &new_obs).unwrap();
                    // Verify the observations were added
                    let entity = kg.get_entity(&name).unwrap();
                    for obs in &added {
                        assert!(entity.observations.contains(obs));
                    }
                }
            }
            86..=94 => {
                if !live_names.is_empty() {
                    let name = known_entity(&mut rng, &live_names.iter().cloned().collect::<Vec<_>>());
                    let entity = kg.get_entity(&name).unwrap();
                    if !entity.observations.is_empty() {
                        let del_count = rng.gen_range(1..=entity.observations.len().min(2));
                        let to_del: Vec<String> = entity.observations.iter()
                            .choose_multiple(&mut rng, del_count)
                            .into_iter()
                            .cloned()
                            .collect();
                        kg.delete_observations(&name, &to_del).unwrap();
                    }
                }
            }
            _ => {
                // Read-only ops: search, stats, path, open_nodes
                if !live_names.is_empty() && rng.gen_bool(0.5) {
                    let name = known_entity(&mut rng, &live_names.iter().cloned().collect::<Vec<_>>());
                    let _ = kg.search_nodes(&name[..2.min(name.len())]);
                    let _ = kg.open_nodes(&[name]);
                }
                let _ = kg.graph_stats();
            }
        }

        check_invariants(&kg, &live_names);
    }

    cleanup(&path);
}

// =========================================================================
// 2. Random operations with verified persistence replay
// =========================================================================

#[test]
fn test_random_persistence_roundtrip() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([123u8; 32]);
    let mut live_names: HashSet<String> = HashSet::new();

    // Phase 1: random mutations
    for _ in 0..100 {
        let op: u32 = rng.gen_range(0..100);
        match op {
            0..=50 => {
                let entity = random_entity(&mut rng);
                if !live_names.contains(&entity.name) {
                    if !kg.create_entities(&[entity.clone()]).unwrap().is_empty() {
                        live_names.insert(entity.name.clone());
                    }
                }
            }
            51..=70 => {
                if !live_names.is_empty() {
                    let pick = rng.gen_range(1..=2.min(live_names.len()));
                    let names: Vec<String> = live_names.iter()
                        .choose_multiple(&mut rng, pick)
                        .into_iter()
                        .cloned()
                        .collect();
                    kg.delete_entities(&names).unwrap();
                    for n in &names {
                        live_names.remove(n);
                    }
                }
            }
            _ => {
                if !live_names.is_empty() {
                    let name = known_entity(&mut rng, &live_names.iter().cloned().collect::<Vec<_>>());
                    let num_obs = rng.gen_range(1..=3);
                    let obs: Vec<String> = (0..num_obs).map(|_| random_name(&mut rng, 5)).collect();
                    kg.add_observations(&name, &obs).unwrap();
                }
            }
        }
    }

    // Check invariants before close
    check_invariants(&kg, &live_names);
    drop(kg);

    // Phase 2: replay from disk
    let mut kg2 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    check_invariants(&kg2, &live_names);

    // Phase 3: more mutations after replay
    for _ in 0..50 {
        let entity = random_entity(&mut rng);
        if !live_names.contains(&entity.name) {
            if !kg2.create_entities(&[entity.clone()]).unwrap().is_empty() {
                live_names.insert(entity.name.clone());
            }
        }
    }
    check_invariants(&kg2, &live_names);
    drop(kg2);

    // Phase 4: replay again
    let kg3 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    check_invariants(&kg3, &live_names);

    cleanup(&path);
}

// =========================================================================
// 3. Stress: large batch creates
// =========================================================================

#[test]
fn test_stress_bulk_create() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([99u8; 32]);

    let entities: Vec<Entity> = (0..500).map(|_| random_entity(&mut rng)).collect();
    let live_names: HashSet<String> = entities.iter().map(|e| e.name.clone()).collect();

    // Create all at once (in batches to avoid huge single calls)
    for chunk in entities.chunks(50) {
        let created = kg.create_entities(chunk).unwrap();
        assert_eq!(created.len(), chunk.len());
    }

    check_invariants(&kg, &live_names);
    assert_eq!(kg.graph_stats()["entities"].as_u64().unwrap(), 500);

    // Create relations between them
    let name_vec: Vec<String> = live_names.iter().cloned().collect();
    let mut rel_count = 0;
    for _ in 0..200 {
        let from = name_vec[rng.gen_range(0..name_vec.len())].clone();
        let to = name_vec[rng.gen_range(0..name_vec.len())].clone();
        if from != to {
            let rtype = random_name(&mut rng, 4);
            let rel = Relation { from, to, relation_type: rtype };
            if !kg.create_relations(&[rel]).unwrap().is_empty() {
                rel_count += 1;
            }
        }
    }

    check_invariants(&kg, &live_names);
    drop(kg);

    // Replay
    let kg2 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    check_invariants(&kg2, &live_names);
    let stats = kg2.graph_stats();
    assert_eq!(stats["entities"].as_u64().unwrap(), 500);
    assert_eq!(stats["relations"].as_u64().unwrap() as usize, rel_count);

    cleanup(&path);
}

// =========================================================================
// 4. Stress: many observations add/delete
// =========================================================================

#[test]
fn test_stress_observations_churn() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([177u8; 32]);

    // Create 20 base entities
    let entities: Vec<Entity> = (0..20).map(|_| random_entity(&mut rng)).collect();
    kg.create_entities(&entities).unwrap();
    let live_names: HashSet<String> = entities.iter().map(|e| e.name.clone()).collect();

    // Churn observations
    for _ in 0..300 {
        let name = known_entity(&mut rng, &live_names.iter().cloned().collect::<Vec<_>>());
        let num_new = rng.gen_range(1..=5);
        let new_obs: Vec<String> = (0..num_new)
            .map(|_| {
                let len = rng.gen_range(3..10);
                random_name(&mut rng, len)
            })
            .collect();
        kg.add_observations(&name, &new_obs).unwrap();

        // Delete some observations occasionally
        if rng.gen_bool(0.3) {
            let entity = kg.get_entity(&name).unwrap();
            if !entity.observations.is_empty() {
                let del_n = rng.gen_range(1..=entity.observations.len().min(3));
                let to_del: Vec<String> = entity.observations
                    .choose_multiple(&mut rng, del_n)
                    .cloned()
                    .collect();
                kg.delete_observations(&name, &to_del).unwrap();
            }
        }
    }

    check_invariants(&kg, &live_names);
    drop(kg);

    // Verify persistence
    let kg2 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    check_invariants(&kg2, &live_names);

    cleanup(&path);
}

// =========================================================================
// 5. Invariant: compact + replay should preserve state exactly
// =========================================================================

#[test]
fn test_fuzzy_compact_preserves_invariants() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([201u8; 32]);
    let mut live_names: HashSet<String> = HashSet::new();

    // Build up some state
    for _ in 0..80 {
        let entity = random_entity(&mut rng);
        if !live_names.contains(&entity.name) {
            if !kg.create_entities(&[entity.clone()]).unwrap().is_empty() {
                live_names.insert(entity.name.clone());
            }
        }
    }

    // Create some relations
    let name_vec: Vec<String> = live_names.iter().cloned().collect();
    for _ in 0..30 {
        let from = name_vec[rng.gen_range(0..name_vec.len())].clone();
        let to = name_vec[rng.gen_range(0..name_vec.len())].clone();
        if from != to {
            let rel = Relation { from, to, relation_type: "edge".into() };
            let _ = kg.create_relations(&[rel]);
        }
    }

    // Delete some entities
    let to_delete: Vec<String> = live_names.iter()
        .choose_multiple(&mut rng, 20)
        .into_iter()
        .cloned()
        .collect();
    kg.delete_entities(&to_delete).unwrap();
    for n in &to_delete {
        live_names.remove(n);
    }

    check_invariants(&kg, &live_names);

    // Compact
    kg.compact().unwrap();
    check_invariants(&kg, &live_names);

    // Replay from compacted log
    drop(kg);
    let kg2 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    check_invariants(&kg2, &live_names);

    // Compact empty (after delete all)
    drop(kg2);
    let mut kg3 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let all_names: Vec<String> = live_names.iter().cloned().collect();
    kg3.delete_entities(&all_names).unwrap();
    for n in &all_names {
        live_names.remove(n);
    }
    kg3.compact().unwrap();
    check_invariants(&kg3, &live_names);
    drop(kg3);

    let kg4 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    check_invariants(&kg4, &live_names);

    cleanup(&path);
}

// =========================================================================
// 6. Concurrent fuzzy stress
// =========================================================================

#[test]
fn test_concurrent_fuzzy_stress() {
    let path = tmp_path();
    let kg_mutex = Arc::new(RwLock::new(KnowledgeGraph::new(Path::new(&path)).unwrap()));
    let mut rng = SmallRng::from_seed([210u8; 32]);

    // Pre-seed with entities
    {
        let mut guard = kg_mutex.write().unwrap();
        let entities: Vec<Entity> = (0..30).map(|_| random_entity(&mut rng)).collect();
        guard.create_entities(&entities).unwrap();
    }

    let mut handles = Vec::new();
    let thread_count = 8;
    let ops_per_thread = 100;

    for t in 0..thread_count {
        let kg = Arc::clone(&kg_mutex);
        let seed: u64 = 1000 + t as u64;
        handles.push(std::thread::spawn(move || {
            let mut rng = SmallRng::from_seed(seed.to_le_bytes().repeat(4).try_into().unwrap());
            for _ in 0..ops_per_thread {
                let op: u32 = rng.gen_range(0..100);
                let mut guard = kg.write().unwrap();
                match op {
                    0..=40 => {
                        let entity = random_entity(&mut rng);
                        let _ = guard.create_entities(&[entity]);
                    }
                    41..=55 => {
                        let stats = guard.graph_stats();
                        let count = stats["entities"].as_u64().unwrap_or(0);
                        if count > 0 {
                            // Try deleting a random entity — pick from existing
                            let graph = guard.read_graph();
                            if !graph.entities.is_empty() {
                                let idx = rng.gen_range(0..graph.entities.len());
                                let name = graph.entities[idx].name.clone();
                                let _ = guard.delete_entities(&[name]);
                            }
                        }
                    }
                    56..=75 => {
                        let graph = guard.read_graph();
                        if graph.entities.len() >= 2 {
                            let a = &graph.entities[rng.gen_range(0..graph.entities.len())].name;
                            let b = &graph.entities[rng.gen_range(0..graph.entities.len())].name;
                            if a != b {
                                let rel = Relation {
                                    from: a.clone(),
                                    to: b.clone(),
                                    relation_type: "concurrent".into(),
                                };
                                let _ = guard.create_relations(&[rel]);
                            }
                        }
                    }
                    _ => {
                        // Read
                        let _ = guard.graph_stats();
                        if rng.gen_bool(0.5) {
                            let _ = guard.search_nodes(&random_name(&mut rng, 4));
                        }
                    }
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Verify the graph is consistent
    {
        let guard = kg_mutex.read().unwrap();
        let stats = guard.graph_stats();
        let entity_count = stats["entities"].as_u64().unwrap() as usize;
        let rel_count = stats["relations"].as_u64().unwrap() as usize;
        let graph = guard.read_graph();

        // Every relation must reference valid entities
        let entity_names: HashSet<&str> = graph.entities.iter().map(|e| e.name.as_str()).collect();
        for rel in &graph.relations {
            assert!(entity_names.contains(rel.from.as_str()), "stale from in relation");
            assert!(entity_names.contains(rel.to.as_str()), "stale to in relation");
        }

        // Relations should be unique
        let mut rel_set: HashSet<(&str, &str, &str)> = HashSet::new();
        for rel in &graph.relations {
            assert!(
                rel_set.insert((&rel.from, &rel.to, &rel.relation_type)),
                "duplicate relation found"
            );
        }

        // Entities should be unique
        let mut name_set: HashSet<&str> = HashSet::new();
        for e in &graph.entities {
            assert!(name_set.insert(e.name.as_str()), "duplicate entity name");
        }

        assert_eq!(name_set.len(), entity_count);
        assert_eq!(rel_set.len(), rel_count);
    }

    cleanup(&path);
}

// =========================================================================
// 7. Invariant: search results are always valid entity subsets
// =========================================================================

#[test]
fn test_fuzzy_search_invariants() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([188u8; 32]);
    let mut live_names: HashSet<String> = HashSet::new();

    // Create entities with varied content
    for _ in 0..100 {
        let entity = random_entity(&mut rng);
        if !live_names.contains(&entity.name) {
            if !kg.create_entities(&[entity.clone()]).unwrap().is_empty() {
                live_names.insert(entity.name.clone());
            }
        }
    }

    // Search with various queries and verify invariants
    let queries = [
        "", "a", "e", "z", "abc", "xyz", "test", "coffee", "hello",
        "A", "Z", "ALICE", "ALI", "__nonexistent__",
    ];
    for query in &queries {
        let result = kg.search_nodes(query);
        // Every returned entity must be live
        for entity in &result.entities {
            assert!(live_names.contains(&entity.name), "search returned non-live entity");
        }
        // No duplicate entities
        let mut names: HashSet<&str> = HashSet::new();
        for e in &result.entities {
            assert!(names.insert(e.name.as_str()), "duplicate in search results");
        }
        // Relations must reference entities in the result set
        let result_names: HashSet<&str> = result.entities.iter().map(|e| e.name.as_str()).collect();
        for rel in &result.relations {
            assert!(result_names.contains(rel.from.as_str()) || result_names.contains(rel.to.as_str()));
        }
    }

    // open_nodes invariants
    let test_names: Vec<String> = live_names.iter().cloned().take(5).collect();
    let result = kg.open_nodes(&test_names);
    assert_eq!(result.entities.len(), test_names.len());
    let mut names: HashSet<&str> = HashSet::new();
    for e in &result.entities {
        assert!(names.insert(e.name.as_str()));
        assert!(test_names.contains(&e.name));
    }
    let result_names: HashSet<&str> = result.entities.iter().map(|e| e.name.as_str()).collect();
    for rel in &result.relations {
        assert!(result_names.contains(rel.from.as_str()) || result_names.contains(rel.to.as_str()));
    }

    cleanup(&path);
}

// =========================================================================
// 8. Edge case: very large names and observations
// =========================================================================

#[test]
fn test_fuzzy_large_strings() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();

    // Create entity with near-max-size fields
    let big_name = "x".repeat(1000);
    let big_type = "y".repeat(100);
    let big_obs: Vec<String> = (0..50).map(|i| format!("obs_{}_", i) + &"z".repeat(500)).collect();

    let entity = Entity {
        name: big_name.clone(),
        entity_type: big_type.clone(),
        observations: big_obs.clone(),
    };
    let created = kg.create_entities(&[entity]).unwrap();
    assert_eq!(created.len(), 1);

    let fetched = kg.get_entity(&big_name).unwrap();
    assert_eq!(fetched.name, big_name);
    assert_eq!(fetched.entity_type, big_type);
    assert_eq!(fetched.observations.len(), 50);

    // Search should work
    let result = kg.search_nodes("x");
    assert_eq!(result.entities.len(), 1);

    // Add more large observations
    let more_obs: Vec<String> = (0..50).map(|i| format!("more_{}_", i) + &"w".repeat(500)).collect();
    let added = kg.add_observations(&big_name, &more_obs).unwrap();
    assert_eq!(added.len(), 50);

    let fetched = kg.get_entity(&big_name).unwrap();
    assert_eq!(fetched.observations.len(), 100);

    // Delete large observations
    kg.delete_observations(&big_name, &big_obs).unwrap();
    let fetched = kg.get_entity(&big_name).unwrap();
    assert_eq!(fetched.observations.len(), 50);

    // Persist and replay
    drop(kg);
    let kg2 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let fetched = kg2.get_entity(&big_name).unwrap();
    assert_eq!(fetched.observations.len(), 50);

    cleanup(&path);
}

// =========================================================================
// 9. Edge case: special characters and unicode
// =========================================================================

#[test]
fn test_fuzzy_unicode_stress() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([199u8; 32]);

    let unicode_chars: Vec<char> = vec![
        'é', 'ü', 'ñ', 'ç', 'à', 'è', 'ö', 'ä', 'ß', 'ÿ',
        'あ', 'い', 'う', 'え', 'お',
        '汉', '字', '中', '日', '語',
        'α', 'β', 'γ', 'δ', 'ε',
        '😀', '🚀', '🌟', '🔥', '❤',
    ];

    let mut live_names: HashSet<String> = HashSet::new();

    for _ in 0..50 {
        let name_len = rng.gen_range(2..8);
        let name: String = (0..name_len).map(|_| unicode_chars[rng.gen_range(0..unicode_chars.len())]).collect();
        let etype: String = (0..3).map(|_| unicode_chars[rng.gen_range(0..unicode_chars.len())]).collect();
        let obs: Vec<String> = (0..rng.gen_range(0..4))
            .map(|_| {
                let olen = rng.gen_range(2..6);
                (0..olen).map(|_| unicode_chars[rng.gen_range(0..unicode_chars.len())]).collect()
            })
            .collect();

        if !live_names.contains(&name) {
            let entity = Entity { name: name.clone(), entity_type: etype, observations: obs };
            if !kg.create_entities(&[entity]).unwrap().is_empty() {
                live_names.insert(name);
            }
        }
    }

    check_invariants(&kg, &live_names);

    // Search for unicode substrings
    for name in &live_names {
        if name.len() >= 2 {
            let prefix: String = name.chars().take(1).collect();
            let result = kg.search_nodes(&prefix);
            // Note: search is ASCII-case-insensitive, so non-ASCII prefixes may
            // not match. Every returned entity must still be a valid live name —
            // the search must never crash or invent entities.
            for e in &result.entities {
                assert!(live_names.contains(&e.name), "search returned unknown entity");
            }
        }
    }

    // Relations with unicode
    let name_vec: Vec<String> = live_names.iter().cloned().collect();
    for _ in 0..10 {
        if name_vec.len() >= 2 {
            let from = name_vec[rng.gen_range(0..name_vec.len())].clone();
            let to = name_vec[rng.gen_range(0..name_vec.len())].clone();
            if from != to {
                let rtype: String = (0..3).map(|_| unicode_chars[rng.gen_range(0..unicode_chars.len())]).collect();
                let rel = Relation { from, to, relation_type: rtype };
                let _ = kg.create_relations(&[rel]);
            }
        }
    }

    check_invariants(&kg, &live_names);
    drop(kg);

    let kg2 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    check_invariants(&kg2, &live_names);

    cleanup(&path);
}

// =========================================================================
// 10. find_path invariants
// =========================================================================

#[test]
fn test_fuzzy_find_path_invariants() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([155u8; 32]);
    let mut live_names: HashSet<String> = HashSet::new();

    // Create entities
    let entities: Vec<Entity> = (0..30).map(|_| random_entity(&mut rng)).collect();
    for e in &entities {
        live_names.insert(e.name.clone());
    }
    kg.create_entities(&entities).unwrap();

    // Create a random graph of relations
    let name_vec: Vec<String> = live_names.iter().cloned().collect();
    for _ in 0..60 {
        let from = name_vec[rng.gen_range(0..name_vec.len())].clone();
        let to = name_vec[rng.gen_range(0..name_vec.len())].clone();
        if from != to {
            let rel = Relation { from, to, relation_type: "knows".into() };
            let _ = kg.create_relations(&[rel]);
        }
    }

    // Test find_path invariants
    for _ in 0..50 {
        let from = name_vec[rng.gen_range(0..name_vec.len())].clone();
        let to = name_vec[rng.gen_range(0..name_vec.len())].clone();

        match kg.find_path(&from, &to) {
            Ok(path) => {
                // Path must start with 'from' and end with 'to'
                assert_eq!(path[0], from, "path must start at from");
                assert_eq!(path[path.len() - 1], to, "path must end at to");
                // All intermediate nodes must be live
                for node in &path {
                    assert!(live_names.contains(node), "path node '{node}' not live");
                }
                // Path must not have duplicates (BFS guarantees shortest)
                let mut seen: HashSet<&str> = HashSet::new();
                for node in &path {
                    assert!(seen.insert(node.as_str()), "duplicate in path");
                }
                // Each adjacent pair must have a relation. find_path treats
                // relations as undirected, so the edge may exist in either
                // direction — query both.
                for pair in path.windows(2) {
                    let forward = kg.search_relations(Some(&pair[0]), Some(&pair[1]), None);
                    let backward = kg.search_relations(Some(&pair[1]), Some(&pair[0]), None);
                    let found = !forward.is_empty() || !backward.is_empty();
                    assert!(found, "no relation between {} and {}", pair[0], pair[1]);
                }
            }
            Err(_) => {
                // No path is valid — verify there's indeed no connection
                // by doing a manual BFS (not necessary for invariant, but good)
            }
        }
    }

    cleanup(&path);
}

// =========================================================================
// 11. Invariant: relation dedup
// =========================================================================

#[test]
fn test_fuzzy_relation_dedup() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([111u8; 32]);

    let entities: Vec<Entity> = (0..10).map(|_| random_entity(&mut rng)).collect();
    kg.create_entities(&entities).unwrap();

    let name_vec: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
    let rel = Relation {
        from: name_vec[0].clone(),
        to: name_vec[1].clone(),
        relation_type: "dup_test".into(),
    };

    // Create the same relation 100 times
    for _ in 0..100 {
        let created = kg.create_relations(&[rel.clone()]).unwrap();
        if created.is_empty() {
            break;
        }
    }

    let rels = kg.search_relations(None, None, None);
    assert_eq!(rels.len(), 1, "duplicate relations were not deduped");

    cleanup(&path);
}

// =========================================================================
// 12. Invariant: entity dedup
// =========================================================================

#[test]
fn test_fuzzy_entity_dedup() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();

    let entity = Entity {
        name: "UniqueEntity".into(),
        entity_type: "test".into(),
        observations: vec!["obs1".into()],
    };

    for i in 0..100 {
        let created = kg.create_entities(&[entity.clone()]).unwrap();
        if i == 0 {
            assert_eq!(created.len(), 1);
        } else {
            assert!(created.is_empty(), "duplicate entity created on attempt {i}");
        }
    }

    let stats = kg.graph_stats();
    assert_eq!(stats["entities"], 1);

    // Delete and re-create — should work
    kg.delete_entities(&["UniqueEntity".into()]).unwrap();
    let created = kg.create_entities(&[entity.clone()]).unwrap();
    assert_eq!(created.len(), 1);

    cleanup(&path);
}

// =========================================================================
// 13. Mixed operations with graph_stats invariant
// =========================================================================

#[test]
fn test_fuzzy_stats_invariants() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([144u8; 32]);
    let mut live_names: HashSet<String> = HashSet::new();

    for _ in 0..200 {
        let op: u32 = rng.gen_range(0..100);
        match op {
            0..=45 => {
                let entity = random_entity(&mut rng);
                if !live_names.contains(&entity.name) {
                    if !kg.create_entities(&[entity.clone()]).unwrap().is_empty() {
                        live_names.insert(entity.name.clone());
                    }
                }
            }
            46..=65 => {
                if !live_names.is_empty() {
                    let pick = rng.gen_range(1..=3.min(live_names.len()));
                    let names: Vec<String> = live_names.iter()
                        .choose_multiple(&mut rng, pick)
                        .into_iter()
                        .cloned()
                        .collect();
                    kg.delete_entities(&names).unwrap();
                    for n in &names {
                        live_names.remove(n);
                    }
                }
            }
            66..=80 => {
                if !live_names.is_empty() {
                    let name = known_entity(&mut rng, &live_names.iter().cloned().collect::<Vec<_>>());
                    let num_obs = rng.gen_range(1..=3);
                    let obs: Vec<String> = (0..num_obs).map(|_| random_name(&mut rng, 5)).collect();
                    kg.add_observations(&name, &obs).unwrap();
                }
            }
            _ => {
                // delete_observations
                if !live_names.is_empty() {
                    let name = known_entity(&mut rng, &live_names.iter().cloned().collect::<Vec<_>>());
                    if let Some(entity) = kg.get_entity(&name) {
                        if !entity.observations.is_empty() {
                            let del_n = rng.gen_range(1..=entity.observations.len().min(2));
                            let to_del: Vec<String> = entity.observations
                                .choose_multiple(&mut rng, del_n)
                                .cloned()
                                .collect();
                            kg.delete_observations(&name, &to_del).unwrap();
                        }
                    }
                }
            }
        }

        // Verify stats at every step
        let stats = kg.graph_stats();
        let actual_entity_count = kg.read_graph().entities.len();
        assert_eq!(stats["entities"].as_u64().unwrap() as usize, live_names.len(),
            "entity count mismatch in stats");
        assert_eq!(stats["entities"].as_u64().unwrap() as usize, actual_entity_count,
            "stats vs read_graph mismatch");

        // Count total observations manually
        let total_obs: usize = kg.read_graph().entities.iter()
            .map(|e| e.observations.len())
            .sum();
        assert_eq!(stats["totalObservations"].as_u64().unwrap() as usize, total_obs,
            "totalObservations mismatch");
    }

    cleanup(&path);
}

// =========================================================================
// 14. All operations on empty graph
// =========================================================================

#[test]
fn test_fuzzy_empty_graph_ops() {
    let path = tmp_path();
    let kg = KnowledgeGraph::new(Path::new(&path)).unwrap();

    // Read operations on empty graph
    assert!(kg.read_graph().entities.is_empty());
    assert!(kg.read_graph().relations.is_empty());
    assert!(kg.get_entity("anything").is_none());
    assert!(kg.search_nodes("anything").entities.is_empty());
    assert!(kg.open_nodes(&["anything".into()]).entities.is_empty());
    assert!(kg.search_relations(None, None, None).is_empty());
    assert!(kg.search_relations(Some("x"), None, None).is_empty());
    assert_eq!(kg.graph_stats()["entities"], 0);
    assert!(kg.find_path("a", "b").is_err());

    cleanup(&path);
}

// =========================================================================
// 15. Operations with empty strings in various fields
// =========================================================================

#[test]
fn test_fuzzy_empty_and_whitespace_names() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();

    // Empty name should be rejected by validation
    let entity = Entity {
        name: "".into(),
        entity_type: "t".into(),
        observations: vec![],
    };
    let result = kg.create_entities(&[entity]);
    assert!(result.is_err(), "empty name should be rejected");

    // Entity with just spaces in name — allowed by validation (non-empty)
    let entity = Entity {
        name: "  ".into(),
        entity_type: "t".into(),
        observations: vec![],
    };
    let result = kg.create_entities(&[entity]);
    assert!(result.is_ok(), "whitespace name should be accepted");

    // Whitespace-only search should be safe
    let result = kg.search_nodes("   ");
    assert!(result.entities.is_empty() || result.entities.len() == 1);

    cleanup(&path);
}

// =========================================================================
// 16. open_nodes returns exactly the requested live entities
// =========================================================================

#[test]
fn test_fuzzy_open_nodes_invariants() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([211u8; 32]);

    let entities: Vec<Entity> = (0..120).map(|_| random_entity(&mut rng)).collect();
    let live_names: HashSet<String> = entities.iter().map(|e| e.name.clone()).collect();
    kg.create_entities(&entities).unwrap();
    let name_vec: Vec<String> = live_names.iter().cloned().collect();

    for _ in 0..200 {
        // Build a request mixing real names and never-created ghosts.
        let pick = rng.gen_range(0..=5.min(name_vec.len()));
        let mut request: Vec<String> = name_vec
            .choose_multiple(&mut rng, pick)
            .cloned()
            .collect();
        // Number of ghosts to append.
        let ghosts = rng.gen_range(0..3);
        for _ in 0..ghosts {
            let g = format!("__ghost_{}", random_name(&mut rng, 8));
            if !live_names.contains(&g) {
                request.push(g);
            }
        }
        let requested_set: HashSet<&str> = request.iter().map(String::as_str).collect();

        let out = kg.open_nodes(&request);

        // Every returned entity must have been requested and be live.
        let mut seen: HashSet<&str> = HashSet::new();
        for e in &out.entities {
            assert!(live_names.contains(&e.name), "open_nodes returned non-live entity");
            assert!(requested_set.contains(e.name.as_str()), "open_nodes returned unrequested entity");
            assert!(seen.insert(e.name.as_str()), "open_nodes returned duplicate entity");
        }
        // Every requested live name must be present in the output.
        let returned: HashSet<&str> = out.entities.iter().map(|e| e.name.as_str()).collect();
        for name in &request {
            if live_names.contains(name) {
                assert!(returned.contains(name.as_str()), "requested live name '{name}' missing");
            }
        }
    }

    cleanup(&path);
}

// =========================================================================
// 17. delete_relations removes only exact (from, to, type) matches
// =========================================================================

#[test]
fn test_fuzzy_delete_relations_precision() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([212u8; 32]);

    let entities: Vec<Entity> = (0..25).map(|_| random_entity(&mut rng)).collect();
    let name_vec: Vec<String> = {
        let set: HashSet<String> = entities.iter().map(|e| e.name.clone()).collect();
        set.into_iter().collect()
    };
    kg.create_entities(&entities).unwrap();

    // Model the relation set ourselves to compare against the store.
    let mut model: HashSet<(String, String, String)> = HashSet::new();
    let rtypes = ["knows", "likes", "owns", "manages"];

    for _ in 0..400 {
        let op = rng.gen_range(0..100);
        if op < 60 {
            // Create a random relation.
            let from = name_vec[rng.gen_range(0..name_vec.len())].clone();
            let to = name_vec[rng.gen_range(0..name_vec.len())].clone();
            let rtype = rtypes[rng.gen_range(0..rtypes.len())].to_string();
            let rel = Relation { from: from.clone(), to: to.clone(), relation_type: rtype.clone() };
            let created = kg.create_relations(&[rel]).unwrap();
            if !created.is_empty() {
                assert!(model.insert((from, to, rtype)), "store created a duplicate relation");
            } else {
                // Creation returned empty only for an existing duplicate.
                assert!(model.contains(&(from, to, rtype)), "empty create for a novel relation");
            }
        } else if !model.is_empty() {
            // Delete an existing relation; sometimes also attempt a bogus one.
            let existing: Vec<(String, String, String)> = model.iter().cloned().collect();
            let target = existing[rng.gen_range(0..existing.len())].clone();
            let mut to_del = vec![Relation {
                from: target.0.clone(),
                to: target.1.clone(),
                relation_type: target.2.clone(),
            }];
            // A non-existent relation must be a silent no-op.
            if rng.gen_bool(0.3) {
                to_del.push(Relation {
                    from: target.0.clone(),
                    to: target.1.clone(),
                    relation_type: "__never__".into(),
                });
            }
            kg.delete_relations(&to_del).unwrap();
            model.remove(&target);
        }

        // The store must match the model exactly.
        let live: HashSet<(String, String, String)> = kg
            .search_relations(None, None, None)
            .into_iter()
            .map(|r| (r.from, r.to, r.relation_type))
            .collect();
        assert_eq!(live, model, "relation set diverged from model");
    }

    cleanup(&path);
}

// =========================================================================
// 18. Durability: reopening after every mutation preserves the graph
// =========================================================================

#[test]
fn test_fuzzy_reopen_after_each_mutation() {
    let path = tmp_path();
    let mut rng = SmallRng::from_seed([213u8; 32]);
    let mut live_names: HashSet<String> = HashSet::new();
    let mut relations: HashSet<(String, String, String)> = HashSet::new();

    for _ in 0..120 {
        // Fresh handle each iteration — exercises log replay on open.
        let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();

        // State recovered from disk must match our model before mutating.
        let graph = kg.read_graph();
        assert_eq!(graph.entities.len(), live_names.len(), "entity count not durable");
        let disk_rels: HashSet<(String, String, String)> = graph
            .relations
            .iter()
            .map(|r| (r.from.clone(), r.to.clone(), r.relation_type.clone()))
            .collect();
        assert_eq!(disk_rels, relations, "relations not durable across reopen");

        let op = rng.gen_range(0..100);
        match op {
            0..=45 => {
                let entity = random_entity(&mut rng);
                if !live_names.contains(&entity.name)
                    && !kg.create_entities(&[entity.clone()]).unwrap().is_empty()
                {
                    live_names.insert(entity.name);
                }
            }
            46..=60 => {
                if !live_names.is_empty() {
                    let names: Vec<String> = live_names.iter().cloned().collect();
                    let victim = names[rng.gen_range(0..names.len())].clone();
                    kg.delete_entities(&[victim.clone()]).unwrap();
                    live_names.remove(&victim);
                    // Cascade: relations touching the victim are gone.
                    relations.retain(|(f, t, _)| f != &victim && t != &victim);
                }
            }
            61..=85 => {
                if live_names.len() >= 2 {
                    let names: Vec<String> = live_names.iter().cloned().collect();
                    let from = names[rng.gen_range(0..names.len())].clone();
                    let to = names[rng.gen_range(0..names.len())].clone();
                    if from != to {
                        let rel = Relation { from: from.clone(), to: to.clone(), relation_type: "rel".into() };
                        if !kg.create_relations(&[rel]).unwrap().is_empty() {
                            relations.insert((from, to, "rel".into()));
                        }
                    }
                }
            }
            _ => {
                if !relations.is_empty() {
                    let all: Vec<(String, String, String)> = relations.iter().cloned().collect();
                    let target = all[rng.gen_range(0..all.len())].clone();
                    kg.delete_relations(&[Relation {
                        from: target.0.clone(),
                        to: target.1.clone(),
                        relation_type: target.2.clone(),
                    }]).unwrap();
                    relations.remove(&target);
                }
            }
        }

        kg.flush_and_sync().unwrap();
        // kg dropped here, forcing the next iteration to replay from disk.
    }

    cleanup(&path);
}

// =========================================================================
// 19. get_neighbors invariants over a random graph
// =========================================================================

#[test]
fn test_fuzzy_neighbors_invariants() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([214u8; 32]);

    let entities: Vec<Entity> = (0..40).map(|_| random_entity(&mut rng)).collect();
    let live_names: HashSet<String> = entities.iter().map(|e| e.name.clone()).collect();
    kg.create_entities(&entities).unwrap();
    let name_vec: Vec<String> = live_names.iter().cloned().collect();

    for _ in 0..80 {
        let from = name_vec[rng.gen_range(0..name_vec.len())].clone();
        let to = name_vec[rng.gen_range(0..name_vec.len())].clone();
        if from != to {
            let _ = kg.create_relations(&[Relation {
                from,
                to,
                relation_type: "knows".into(),
            }]);
        }
    }

    let dirs = [Direction::Out, Direction::In, Direction::Both];
    for _ in 0..200 {
        let origin = name_vec[rng.gen_range(0..name_vec.len())].clone();
        let dir = dirs[rng.gen_range(0..dirs.len())];
        let depth = rng.gen_range(0..4);
        let out = kg.neighbors(&origin, dir, None, depth).unwrap();

        // Origin is always present; every entity is live and unique.
        let returned: HashSet<&str> = out.entities.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(returned.len(), out.entities.len(), "duplicate entity in neighbors");
        assert!(returned.contains(origin.as_str()), "origin missing from neighbors");
        for e in &out.entities {
            assert!(live_names.contains(&e.name), "neighbors returned non-live entity");
        }
        // Every returned relation has both endpoints inside the returned set.
        for r in &out.relations {
            assert!(returned.contains(r.from.as_str()), "relation from outside neighbor set");
            assert!(returned.contains(r.to.as_str()), "relation to outside neighbor set");
        }
        // depth 0 yields exactly the origin.
        if depth == 0 {
            assert_eq!(out.entities.len(), 1);
            assert!(out.relations.is_empty());
        }
    }

    // Unknown entity errors.
    assert!(kg.neighbors("__never__", Direction::Both, None, 1).is_err());
    cleanup(&path);
}

// =========================================================================
// 20. describe_entity matches search_relations
// =========================================================================

#[test]
fn test_fuzzy_describe_entity_consistency() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([215u8; 32]);

    let entities: Vec<Entity> = (0..30).map(|_| random_entity(&mut rng)).collect();
    let name_vec: Vec<String> = {
        let s: HashSet<String> = entities.iter().map(|e| e.name.clone()).collect();
        s.into_iter().collect()
    };
    kg.create_entities(&entities).unwrap();

    for _ in 0..100 {
        let from = name_vec[rng.gen_range(0..name_vec.len())].clone();
        let to = name_vec[rng.gen_range(0..name_vec.len())].clone();
        if from != to {
            let _ = kg.create_relations(&[Relation {
                from,
                to,
                relation_type: "rel".into(),
            }]);
        }
    }

    for name in &name_vec {
        let v = kg.describe_entity(name).unwrap();
        assert_eq!(v["entity"]["name"], name.as_str());
        let degree = v["degree"].as_u64().unwrap() as usize;

        // Incident degree equals outgoing + incoming relations.
        let outgoing = kg.search_relations(Some(name), None, None).len();
        let incoming = kg.search_relations(None, Some(name), None).len();
        assert_eq!(degree, outgoing + incoming, "degree mismatch for {name}");
        assert_eq!(v["relations"].as_array().unwrap().len(), degree);

        // Neighbor count never exceeds degree and is unique.
        let neighbors = v["neighbors"].as_array().unwrap();
        let uniq: HashSet<&str> = neighbors.iter().filter_map(|n| n.as_str()).collect();
        assert_eq!(uniq.len(), neighbors.len(), "duplicate neighbor");
        assert!(neighbors.len() <= degree);
    }

    cleanup(&path);
}

// =========================================================================
// 21. upsert is idempotent and never loses observations
// =========================================================================

#[test]
fn test_fuzzy_upsert_idempotent() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([216u8; 32]);

    // Model: name -> set of observations (type fixed at first creation).
    let mut model: std::collections::HashMap<String, HashSet<String>> = std::collections::HashMap::new();

    for _ in 0..400 {
        let entity = random_entity(&mut rng);
        let name = entity.name.clone();
        let obs_set: HashSet<String> = entity.observations.iter().cloned().collect();

        let out = kg.upsert_entities(&[entity]).unwrap();
        let created = out[0]["created"].as_bool().unwrap();
        let existed = model.contains_key(&name);
        assert_eq!(created, !existed, "created flag disagrees with model");

        model.entry(name.clone()).or_default().extend(obs_set);

        // Store observations must match the model union exactly.
        let stored = kg.get_entity(&name).unwrap();
        let stored_set: HashSet<String> = stored.observations.iter().cloned().collect();
        assert_eq!(&stored_set, model.get(&name).unwrap(), "observation set diverged");
        // No duplicate observations are ever stored.
        assert_eq!(stored.observations.len(), stored_set.len(), "duplicate observation stored");
    }

    // Entity count equals the number of distinct names upserted.
    let stats = kg.graph_stats();
    assert_eq!(stats["entities"].as_u64().unwrap() as usize, model.len());
    cleanup(&path);
}

// =========================================================================
// 22. filtered search is a faithful subset of unfiltered search
// =========================================================================

#[test]
fn test_fuzzy_filtered_search_subset() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([217u8; 32]);

    let entities: Vec<Entity> = (0..200).map(|_| random_entity(&mut rng)).collect();
    kg.create_entities(&entities).unwrap();

    // A pool of query tokens drawn from real entity names.
    let tokens: Vec<String> = entities
        .iter()
        .map(|e| e.name.chars().take(2).collect::<String>())
        .filter(|s| !s.is_empty())
        .collect();

    for _ in 0..150 {
        let q = tokens[rng.gen_range(0..tokens.len())].clone();
        let full = kg.search_nodes(&q);
        let full_names: HashSet<&str> = full.entities.iter().map(|e| e.name.as_str()).collect();

        // Pagination subset: limit caps the count and never invents entities.
        let limit = rng.gen_range(0..=full.entities.len().max(1));
        let page = kg.search_nodes_filtered(&q, None, 0, limit);
        assert!(page.entities.len() <= limit);
        for e in &page.entities {
            assert!(full_names.contains(e.name.as_str()), "filtered search invented an entity");
        }

        // Skipping `offset` matches yields exactly the remaining count.
        let offset = rng.gen_range(0..=full.entities.len() + 1);
        let rest = kg.search_nodes_filtered(&q, None, offset, usize::MAX);
        assert_eq!(rest.entities.len(), full.entities.len().saturating_sub(offset));
        for e in &rest.entities {
            assert!(full_names.contains(e.name.as_str()));
        }

        // Type filter: every returned entity actually has that type.
        if !full.entities.is_empty() {
            let etype = full.entities[0].entity_type.clone();
            let typed = kg.search_nodes_filtered(&q, Some(&etype), 0, usize::MAX);
            for e in &typed.entities {
                assert_eq!(e.entity_type, etype, "type filter leaked a wrong type");
                assert!(full_names.contains(e.name.as_str()));
            }
        }
    }

    cleanup(&path);
}

// =========================================================================
// 23. export round-trips through json and never panics
// =========================================================================

#[test]
fn test_fuzzy_export_consistency() {
    let path = tmp_path();
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([218u8; 32]);

    let entities: Vec<Entity> = (0..50).map(|_| random_entity(&mut rng)).collect();
    let name_vec: Vec<String> = {
        let s: HashSet<String> = entities.iter().map(|e| e.name.clone()).collect();
        s.into_iter().collect()
    };
    kg.create_entities(&entities).unwrap();
    for _ in 0..60 {
        let from = name_vec[rng.gen_range(0..name_vec.len())].clone();
        let to = name_vec[rng.gen_range(0..name_vec.len())].clone();
        if from != to {
            let _ = kg.create_relations(&[Relation { from, to, relation_type: "knows".into() }]);
        }
    }

    let graph = kg.read_graph();

    // JSON export deserializes back to the same entity/relation counts.
    let json = kg.export("json").unwrap();
    let parsed: mcp_memory::types::KnowledgeGraphOut = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.entities.len(), graph.entities.len());
    assert_eq!(parsed.relations.len(), graph.relations.len());

    // Mermaid/DOT contain a line per entity plus the header.
    let mermaid = kg.export("mermaid").unwrap();
    assert!(mermaid.starts_with("graph LR"));
    assert_eq!(
        mermaid.matches("[\"").count(),
        graph.entities.len(),
        "mermaid node count mismatch"
    );
    let dot = kg.export("dot").unwrap();
    assert!(dot.starts_with("digraph G {"));

    assert!(kg.export("xml").is_err());
    cleanup(&path);
}
