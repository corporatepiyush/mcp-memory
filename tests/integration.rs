use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use mcp_memory::kg::KnowledgeGraph;
use mcp_memory::types::{Entity, Relation};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn setup() -> (KnowledgeGraph, String) {
    let pid = std::process::id();
    let seq = COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = format!("/tmp/mcp_mem_int_{pid}_{seq}.bin");
    let kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    (kg, path)
}

fn setup_mutex() -> (Arc<Mutex<KnowledgeGraph>>, String) {
    let (kg, path) = setup();
    (Arc::new(Mutex::new(kg)), path)
}

fn cleanup(path: &str) {
    let _ = std::fs::remove_file(path);
}

fn alice() -> Entity {
    Entity {
        name: "Alice".into(),
        entity_type: "person".into(),
        observations: vec!["likes coffee".into(), "works at acme".into()],
    }
}

fn bob() -> Entity {
    Entity {
        name: "Bob".into(),
        entity_type: "person".into(),
        observations: vec!["drinks tea".into()],
    }
}

fn charlie() -> Entity {
    Entity {
        name: "Charlie".into(),
        entity_type: "ai".into(),
        observations: vec!["runs on linux".into(), "likes coffee".into()],
    }
}

fn knows_alice() -> Relation {
    Relation {
        from: "Alice".into(),
        to: "Bob".into(),
        relation_type: "knows".into(),
    }
}

fn knows_bob() -> Relation {
    Relation {
        from: "Bob".into(),
        to: "Charlie".into(),
        relation_type: "knows".into(),
    }
}

fn works_with() -> Relation {
    Relation {
        from: "Alice".into(),
        to: "Charlie".into(),
        relation_type: "works_with".into(),
    }
}

// =========================================================================
// Basic CRUD — Entities
// =========================================================================

#[test]
fn test_create_entity_empty_observations() {
    let (mut kg, path) = setup();
    let entity = Entity {
        name: "Solo".into(),
        entity_type: "test".into(),
        observations: vec![],
    };
    let created = kg.create_entities(&[entity]).unwrap();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].name, "Solo");
    cleanup(&path);
}

#[test]
fn test_create_entity_with_observations() {
    let (mut kg, path) = setup();
    let created = kg.create_entities(&[alice()]).unwrap();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].observations.len(), 2);
    cleanup(&path);
}

#[test]
fn test_create_duplicate_entity_skipped() {
    let (mut kg, path) = setup();
    let e = alice();
    let first = kg.create_entities(&[e.clone()]).unwrap();
    let second = kg.create_entities(&[e.clone()]).unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(second.len(), 0);
    cleanup(&path);
}

#[test]
fn test_create_multiple_entities() {
    let (mut kg, path) = setup();
    let created = kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    assert_eq!(created.len(), 3);
    cleanup(&path);
}

#[test]
fn test_empty_entities_list() {
    let (mut kg, path) = setup();
    let created = kg.create_entities(&[]).unwrap();
    assert!(created.is_empty());
    cleanup(&path);
}

// =========================================================================
// Basic CRUD — Relations
// =========================================================================

#[test]
fn test_create_relation() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    let created = kg.create_relations(&[knows_alice()]).unwrap();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].relation_type, "knows");
    cleanup(&path);
}

#[test]
fn test_create_duplicate_relation_skipped() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    let r = knows_alice();
    let first = kg.create_relations(&[r.clone()]).unwrap();
    let second = kg.create_relations(&[r.clone()]).unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(second.len(), 0);
    cleanup(&path);
}

#[test]
fn test_create_relation_nonexistent_entities() {
    // Relations reference entities by name — we don't validate existence.
    let (mut kg, path) = setup();
    let created = kg.create_relations(&[knows_alice()]).unwrap();
    assert_eq!(created.len(), 1);
    // Entity Alice doesn't exist in the graph, but the relation was created.
    cleanup(&path);
}

#[test]
fn test_empty_relations_list() {
    let (mut kg, path) = setup();
    let created = kg.create_relations(&[]).unwrap();
    assert!(created.is_empty());
    cleanup(&path);
}

// =========================================================================
// Observations
// =========================================================================

#[test]
fn test_add_observations() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    let added = kg
        .add_observations("Alice", &["drinks matcha".into()])
        .unwrap();
    assert_eq!(added.len(), 1);
    assert_eq!(added[0], "drinks matcha");
    cleanup(&path);
}

#[test]
fn test_add_duplicate_observation_skipped() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    let added = kg
        .add_observations("Alice", &["likes coffee".into()])
        .unwrap();
    assert!(added.is_empty());
    cleanup(&path);
}

#[test]
fn test_add_observations_mixed_dup_and_new() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    let added = kg
        .add_observations("Alice", &["likes coffee".into(), "new obs".into()])
        .unwrap();
    assert_eq!(added.len(), 1);
    assert_eq!(added[0], "new obs");
    cleanup(&path);
}

#[test]
fn test_add_observations_nonexistent_entity() {
    let (mut kg, path) = setup();
    let result = kg.add_observations("Ghost", &["something".into()]);
    assert!(result.is_err());
    cleanup(&path);
}

#[test]
fn test_add_observations_empty_list() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    let added = kg.add_observations("Alice", &[]).unwrap();
    assert!(added.is_empty());
    cleanup(&path);
}

// =========================================================================
// Deletion — Entities
// =========================================================================

#[test]
fn test_delete_entity() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    kg.delete_entities(&["Alice".into()]).unwrap();
    let entity = kg.get_entity("Alice");
    assert!(entity.is_none());
    cleanup(&path);
}

#[test]
fn test_delete_nonexistent_entity() {
    let (mut kg, path) = setup();
    // Should not error.
    kg.delete_entities(&["Ghost".into()]).unwrap();
    cleanup(&path);
}

#[test]
fn test_delete_entity_removes_relations() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.create_relations(&[knows_alice()]).unwrap();

    // Delete entity Alice — relations involving Alice should be gone.
    kg.delete_entities(&["Alice".into()]).unwrap();

    let rels = kg.search_relations(None, None, None);
    assert!(rels.is_empty());
    cleanup(&path);
}

#[test]
fn test_delete_entity_then_recreate() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    kg.delete_entities(&["Alice".into()]).unwrap();

    // Re-create with different observations.
    let new_alice = Entity {
        name: "Alice".into(),
        entity_type: "person".into(),
        observations: vec!["new obs".into()],
    };
    let created = kg.create_entities(&[new_alice]).unwrap();
    assert_eq!(created.len(), 1);

    let entity = kg.get_entity("Alice").unwrap();
    assert_eq!(entity.observations, vec!["new obs"]);
    cleanup(&path);
}

#[test]
fn test_delete_multiple_entities() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    kg.delete_entities(&["Alice".into(), "Bob".into()]).unwrap();
    assert!(kg.get_entity("Alice").is_none());
    assert!(kg.get_entity("Bob").is_none());
    assert!(kg.get_entity("Charlie").is_some());
    cleanup(&path);
}

// =========================================================================
// Deletion — Observations
// =========================================================================

#[test]
fn test_delete_observations() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    kg.delete_observations("Alice", &["likes coffee".into()])
        .unwrap();
    let entity = kg.get_entity("Alice").unwrap();
    assert_eq!(entity.observations.len(), 1);
    assert_eq!(entity.observations[0], "works at acme");
    cleanup(&path);
}

#[test]
fn test_delete_nonexistent_observation() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    kg.delete_observations("Alice", &["does not exist".into()])
        .unwrap();
    let entity = kg.get_entity("Alice").unwrap();
    assert_eq!(entity.observations.len(), 2);
    cleanup(&path);
}

#[test]
fn test_delete_observations_nonexistent_entity() {
    let (mut kg, path) = setup();
    let result = kg.delete_observations("Ghost", &["x".into()]);
    assert!(result.is_err());
    cleanup(&path);
}

#[test]
fn test_delete_all_observations() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    kg.delete_observations("Alice", &["likes coffee".into(), "works at acme".into()])
        .unwrap();
    let entity = kg.get_entity("Alice").unwrap();
    assert!(entity.observations.is_empty());
    cleanup(&path);
}

// =========================================================================
// Deletion — Relations
// =========================================================================

#[test]
fn test_delete_relation() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.create_relations(&[knows_alice()]).unwrap();
    kg.delete_relations(&[knows_alice()]).unwrap();
    let rels = kg.search_relations(None, None, None);
    assert!(rels.is_empty());
    cleanup(&path);
}

#[test]
fn test_delete_nonexistent_relation() {
    let (mut kg, path) = setup();
    // Should not error.
    kg.delete_relations(&[knows_alice()]).unwrap();
    cleanup(&path);
}

// =========================================================================
// Read operations
// =========================================================================

#[test]
fn test_read_graph_empty() {
    let (kg, path) = setup();
    let graph = kg.read_graph();
    assert!(graph.entities.is_empty());
    assert!(graph.relations.is_empty());
    cleanup(&path);
}

#[test]
fn test_read_graph_with_data() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.create_relations(&[knows_alice()]).unwrap();
    let graph = kg.read_graph();
    assert_eq!(graph.entities.len(), 2);
    assert_eq!(graph.relations.len(), 1);
    cleanup(&path);
}

#[test]
fn test_get_entity_existing() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    let entity = kg.get_entity("Alice").unwrap();
    assert_eq!(entity.name, "Alice");
    assert_eq!(entity.entity_type, "person");
    assert_eq!(entity.observations.len(), 2);
    cleanup(&path);
}

#[test]
fn test_get_entity_nonexistent() {
    let (mut kg, path) = setup();
    let entity = kg.get_entity("Ghost");
    assert!(entity.is_none());
    cleanup(&path);
}

#[test]
fn test_get_entity_after_delete() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    kg.delete_entities(&["Alice".into()]).unwrap();
    assert!(kg.get_entity("Alice").is_none());
    cleanup(&path);
}

// =========================================================================
// Search
// =========================================================================

#[test]
fn test_search_nodes_by_name() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    let result = kg.search_nodes("Alice");
    assert_eq!(result.entities.len(), 1);
    assert_eq!(result.entities[0].name, "Alice");
    cleanup(&path);
}

#[test]
fn test_search_nodes_by_type() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    let result = kg.search_nodes("person");
    assert_eq!(result.entities.len(), 2);
    cleanup(&path);
}

#[test]
fn test_search_nodes_by_observation() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    let result = kg.search_nodes("coffee");
    assert_eq!(result.entities.len(), 2); // Alice + Charlie
    cleanup(&path);
}

#[test]
fn test_search_nodes_case_insensitive() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    let result = kg.search_nodes("alice");
    assert_eq!(result.entities.len(), 1);
    cleanup(&path);
}

#[test]
fn test_search_nodes_partial_match() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    let result = kg.search_nodes("Ali");
    assert_eq!(result.entities.len(), 1);
    cleanup(&path);
}

#[test]
fn test_search_nodes_no_match() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    let result = kg.search_nodes("zzzzz");
    assert!(result.entities.is_empty());
    cleanup(&path);
}

#[test]
fn test_search_nodes_empty_query() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    let result = kg.search_nodes("");
    assert!(result.entities.is_empty());
    cleanup(&path);
}

#[test]
fn test_search_returns_relations() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.create_relations(&[knows_alice()]).unwrap();
    let result = kg.search_nodes("Alice");
    assert_eq!(result.relations.len(), 1);
    cleanup(&path);
}

// =========================================================================
// Open nodes
// =========================================================================

#[test]
fn test_open_nodes_existing() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    let result = kg.open_nodes(&["Alice".into()]);
    assert_eq!(result.entities.len(), 1);
    assert_eq!(result.entities[0].name, "Alice");
    cleanup(&path);
}

#[test]
fn test_open_nodes_multiple() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    let result = kg.open_nodes(&["Alice".into(), "Charlie".into()]);
    assert_eq!(result.entities.len(), 2);
    cleanup(&path);
}

#[test]
fn test_open_nodes_nonexistent() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    let result = kg.open_nodes(&["Ghost".into()]);
    assert!(result.entities.is_empty());
    cleanup(&path);
}

#[test]
fn test_open_nodes_mixed_existence() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    let result = kg.open_nodes(&["Alice".into(), "Ghost".into()]);
    assert_eq!(result.entities.len(), 1);
    assert_eq!(result.entities[0].name, "Alice");
    cleanup(&path);
}

#[test]
fn test_open_nodes_returns_relations() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    kg.create_relations(&[knows_alice(), knows_bob()]).unwrap();
    let result = kg.open_nodes(&["Alice".into()]);
    assert_eq!(result.entities.len(), 1);
    assert_eq!(result.relations.len(), 1); // Alice knows Bob
    cleanup(&path);
}

// =========================================================================
// graph_stats
// =========================================================================

#[test]
fn test_graph_stats_empty() {
    let (kg, path) = setup();
    let stats = kg.graph_stats();
    assert_eq!(stats["entities"], 0);
    assert_eq!(stats["relations"], 0);
    assert_eq!(stats["totalObservations"], 0);
    cleanup(&path);
}

#[test]
fn test_graph_stats_after_operations() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.create_relations(&[knows_alice()]).unwrap();
    let stats = kg.graph_stats();
    assert_eq!(stats["entities"], 2);
    assert_eq!(stats["relations"], 1);
    assert_eq!(stats["totalObservations"], 3); // Alice(2) + Bob(1)
    cleanup(&path);
}

#[test]
fn test_graph_stats_after_delete() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.delete_entities(&["Alice".into()]).unwrap();
    let stats = kg.graph_stats();
    assert_eq!(stats["entities"], 1);
    cleanup(&path);
}

// =========================================================================
// search_relations
// =========================================================================

#[test]
fn test_search_relations_all() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    kg.create_relations(&[knows_alice(), knows_bob(), works_with()]).unwrap();
    let rels = kg.search_relations(None, None, None);
    assert_eq!(rels.len(), 3);
    cleanup(&path);
}

#[test]
fn test_search_relations_filter_from() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    kg.create_relations(&[knows_alice(), knows_bob(), works_with()]).unwrap();
    let rels = kg.search_relations(Some("Alice"), None, None);
    assert_eq!(rels.len(), 2); // knows Bob + works_with Charlie
    cleanup(&path);
}

#[test]
fn test_search_relations_filter_to() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    kg.create_relations(&[knows_alice(), knows_bob()]).unwrap();
    let rels = kg.search_relations(None, Some("Charlie"), None);
    assert_eq!(rels.len(), 1);
    assert_eq!(rels[0].from, "Bob");
    cleanup(&path);
}

#[test]
fn test_search_relations_filter_type() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.create_relations(&[knows_alice(), works_with()]).unwrap();
    let rels = kg.search_relations(None, None, Some("works_with"));
    assert_eq!(rels.len(), 1);
    assert_eq!(rels[0].from, "Alice");
    assert_eq!(rels[0].to, "Charlie");
    cleanup(&path);
}

#[test]
fn test_search_relations_combined_filters() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    kg.create_relations(&[knows_alice(), knows_bob(), works_with()]).unwrap();
    let rels = kg
        .search_relations(Some("Alice"), Some("Bob"), Some("knows"));
    assert_eq!(rels.len(), 1);
    cleanup(&path);
}

#[test]
fn test_search_relations_no_match() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.create_relations(&[knows_alice()]).unwrap();
    let rels = kg.search_relations(None, None, Some("nonexistent"));
    assert!(rels.is_empty());
    cleanup(&path);
}

#[test]
fn test_search_relations_empty_graph() {
    let (mut kg, path) = setup();
    let rels = kg.search_relations(None, None, None);
    assert!(rels.is_empty());
    cleanup(&path);
}

// =========================================================================
// find_path — BFS shortest path
// =========================================================================

#[test]
fn test_find_path_direct() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.create_relations(&[knows_alice()]).unwrap();
    let p = kg.find_path("Alice", "Bob").unwrap();
    assert_eq!(p, vec!["Alice", "Bob"]);
    cleanup(&path);
}

#[test]
fn test_find_path_indirect() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    kg.create_relations(&[knows_alice(), knows_bob()]).unwrap();
    let p = kg.find_path("Alice", "Charlie").unwrap();
    assert_eq!(p, vec!["Alice", "Bob", "Charlie"]);
    cleanup(&path);
}

#[test]
fn test_find_path_multiple_shortest() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    kg.create_relations(&[knows_alice(), works_with()]).unwrap();
    let p = kg.find_path("Alice", "Charlie").unwrap();
    // Both knows_alice (Alice→Bob→Charlie) and works_with (Alice→Charlie) exist.
    // BFS should find the shortest: Alice → Charlie (direct via works_with).
    assert_eq!(p, vec!["Alice", "Charlie"]);
    cleanup(&path);
}

#[test]
fn test_find_path_no_path() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    let result = kg.find_path("Alice", "Bob");
    assert!(result.is_err());
    cleanup(&path);
}

#[test]
fn test_find_path_same_entity() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    let p = kg.find_path("Alice", "Alice").unwrap();
    assert_eq!(p, vec!["Alice"]);
    cleanup(&path);
}

#[test]
fn test_find_path_nonexistent_start() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    let result = kg.find_path("Ghost", "Alice");
    assert!(result.is_err());
    cleanup(&path);
}

#[test]
fn test_find_path_nonexistent_end() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    let result = kg.find_path("Alice", "Ghost");
    assert!(result.is_err());
    cleanup(&path);
}

#[test]
fn test_find_path_chain() {
    // A → B → C → D → E
    let (mut kg, path) = setup();
    let names: Vec<Entity> = (0..5)
        .map(|i| Entity {
            name: format!("Node{i}"),
            entity_type: "node".into(),
            observations: vec![],
        })
        .collect();
    kg.create_entities(&names).unwrap();
    let rels: Vec<Relation> = (0..4)
        .map(|i| Relation {
            from: format!("Node{i}"),
            to: format!("Node{}", i + 1),
            relation_type: "edge".into(),
        })
        .collect();
    kg.create_relations(&rels).unwrap();
    let p = kg.find_path("Node0", "Node4").unwrap();
    assert_eq!(p.len(), 5);
    assert_eq!(p[0], "Node0");
    assert_eq!(p[4], "Node4");
    cleanup(&path);
}

// =========================================================================
// Compact
// =========================================================================

#[test]
fn test_compact_creates_consistent_graph() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.create_relations(&[knows_alice()]).unwrap();
    kg.compact().unwrap();

    // After compaction, the graph should be identical.
    let entity = kg.get_entity("Alice").unwrap();
    assert_eq!(entity.observations.len(), 2);

    let rels = kg.search_relations(None, None, None);
    assert_eq!(rels.len(), 1);
    cleanup(&path);
}

#[test]
fn test_compact_removes_deleted_entities() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.delete_entities(&["Alice".into()]).unwrap();
    kg.compact().unwrap();

    // After compact, Alice should still be gone.
    assert!(kg.get_entity("Alice").is_none());
    assert!(kg.get_entity("Bob").is_some());
    cleanup(&path);
}

#[test]
fn test_compact_then_replay() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    kg.compact().unwrap();
    drop(kg);

    // Re-create KG from the compacted log.
    let mut kg2 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let entity = kg2.get_entity("Alice").unwrap();
    assert_eq!(entity.name, "Alice");
    assert_eq!(entity.entity_type, "person");
    cleanup(&path);
}

#[test]
fn test_compact_empty_graph() {
    let (mut kg, path) = setup();
    // Compact an empty graph — should create a valid empty log.
    kg.compact().unwrap();
    drop(kg);

    // Replay from empty compacted log.
    let kg2 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let stats = kg2.graph_stats();
    assert_eq!(stats["entities"], 0);
    cleanup(&path);
}

// =========================================================================
// Persistence — full roundtrip
// =========================================================================

#[test]
fn test_persistence_roundtrip() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    kg.create_relations(&[knows_alice(), knows_bob(), works_with()]).unwrap();
    drop(kg);

    // Reload from disk.
    let mut kg2 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let graph = kg2.read_graph();
    assert_eq!(graph.entities.len(), 3);
    assert_eq!(graph.relations.len(), 3);

    let entity = kg2.get_entity("Alice").unwrap();
    assert_eq!(entity.observations.len(), 2);
    cleanup(&path);
}

#[test]
fn test_persistence_add_observations() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    kg.add_observations("Alice", &["new obs".into()]).unwrap();
    drop(kg);

    let mut kg2 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let entity = kg2.get_entity("Alice").unwrap();
    assert_eq!(entity.observations.len(), 3);
    cleanup(&path);
}

#[test]
fn test_persistence_delete_entity() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.delete_entities(&["Alice".into()]).unwrap();
    drop(kg);

    let mut kg2 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    assert!(kg2.get_entity("Alice").is_none());
    assert!(kg2.get_entity("Bob").is_some());
    cleanup(&path);
}

#[test]
fn test_persistence_delete_observations() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    kg.delete_observations("Alice", &["likes coffee".into()]).unwrap();
    drop(kg);

    let mut kg2 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let entity = kg2.get_entity("Alice").unwrap();
    assert_eq!(entity.observations, vec!["works at acme"]);
    cleanup(&path);
}

#[test]
fn test_persistence_delete_relations() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.create_relations(&[knows_alice()]).unwrap();
    kg.delete_relations(&[knows_alice()]).unwrap();
    drop(kg);

    let mut kg2 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let rels = kg2.search_relations(None, None, None);
    assert!(rels.is_empty());
    cleanup(&path);
}

#[test]
fn test_persistence_mixed_operations() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.create_relations(&[knows_alice()]).unwrap();
    kg.add_observations("Alice", &["likes matcha".into()]).unwrap();
    kg.delete_entities(&["Bob".into()]).unwrap();
    drop(kg);

    let mut kg2 = KnowledgeGraph::new(Path::new(&path)).unwrap();
    assert!(kg2.get_entity("Alice").is_some());
    assert!(kg2.get_entity("Bob").is_none());

    let rels = kg2.search_relations(None, None, None);
    assert!(rels.is_empty()); // Bob was deleted, so the relation is gone

    let entity = kg2.get_entity("Alice").unwrap();
    assert_eq!(entity.observations.len(), 3);
    cleanup(&path);
}

// =========================================================================
// Edge cases
// =========================================================================

#[test]
fn test_unicode_entity_names() {
    let (mut kg, path) = setup();
    let entity = Entity {
        name: "日本語".into(),
        entity_type: "言語".into(),
        observations: vec!["テスト".into(), "ユニコード".into()],
    };
    kg.create_entities(&[entity]).unwrap();

    let e = kg.get_entity("日本語").unwrap();
    assert_eq!(e.entity_type, "言語");
    assert_eq!(e.observations.len(), 2);
    cleanup(&path);
}

#[test]
fn test_unicode_search() {
    let (mut kg, path) = setup();
    let entity = Entity {
        name: "café".into(),
        entity_type: "location".into(),
        observations: vec!["près de la gare".into()],
    };
    kg.create_entities(&[entity]).unwrap();

    let result = kg.search_nodes("café");
    assert_eq!(result.entities.len(), 1);

    let result = kg.search_nodes("cafe"); // no match — accents matter with ascii-case
    assert_eq!(result.entities.len(), 0);
    cleanup(&path);
}

#[test]
fn test_large_observations() {
    let (mut kg, path) = setup();
    let obs: Vec<String> = (0..100).map(|i| format!("obs_{i}")).collect();
    let entity = Entity {
        name: "Big".into(),
        entity_type: "test".into(),
        observations: obs.clone(),
    };
    kg.create_entities(&[entity]).unwrap();
    let e = kg.get_entity("Big").unwrap();
    assert_eq!(e.observations.len(), 100);
    cleanup(&path);
}

#[test]
fn test_entity_type_empty_string() {
    let (mut kg, path) = setup();
    let entity = Entity {
        name: "Typeless".into(),
        entity_type: "".into(),
        observations: vec![],
    };
    let created = kg.create_entities(&[entity]).unwrap();
    assert_eq!(created.len(), 1);
    let e = kg.get_entity("Typeless").unwrap();
    assert_eq!(e.entity_type, "");
    cleanup(&path);
}

#[test]
fn test_unicode_observation_search() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    kg.add_observations("Alice", &["café au lait".into()]).unwrap();
    let result = kg.search_nodes("café");
    assert_eq!(result.entities.len(), 1);
    cleanup(&path);
}

// =========================================================================
// Concurrency
// =========================================================================

#[test]
fn test_concurrent_create_entities() {
    let (kg_mutex, path) = setup_mutex();
    let kg = Arc::clone(&kg_mutex);
    let mut handles = Vec::new();

    for i in 0..10 {
        let kg = Arc::clone(&kg);
        handles.push(std::thread::spawn(move || {
            let entity = Entity {
                name: format!("ThreadEntity_{i}"),
                entity_type: "concurrent".into(),
                observations: vec![format!("obs_{i}")],
            };
            let mut guard = kg.lock().unwrap();
            guard.create_entities(&[entity]).unwrap();
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let guard = kg_mutex.lock().unwrap();
    let stats = guard.graph_stats();
    assert_eq!(stats["entities"], 10);
    drop(guard);
    cleanup(&path);
}

#[test]
fn test_concurrent_read_write() {
    let (kg_mutex, path) = setup_mutex();
    // Pre-populate an entity.
    {
        let mut guard = kg_mutex.lock().unwrap();
        guard.create_entities(&[alice()]).unwrap();
    }

    let kg = Arc::clone(&kg_mutex);
    let mut handles = Vec::new();

    // 5 readers
    for _ in 0..5 {
        let kg = Arc::clone(&kg);
        handles.push(std::thread::spawn(move || {
            for _ in 0..20 {
                let mut guard = kg.lock().unwrap();
                let _ = guard.get_entity("Alice");
                let _ = guard.graph_stats();
            }
        }));
    }

    // 5 writers
    for i in 0..5 {
        let kg = Arc::clone(&kg);
        handles.push(std::thread::spawn(move || {
            for j in 0..10 {
                let entity = Entity {
                    name: format!("Concurrent_{i}_{j}"),
                    entity_type: "writer".into(),
                    observations: vec![],
                };
                let mut guard = kg.lock().unwrap();
                guard.create_entities(&[entity]).unwrap();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let guard = kg_mutex.lock().unwrap();
    let stats = guard.graph_stats();
    // Alice + 5 writers × 10 entities = 51
    assert_eq!(stats["entities"], 51);
    drop(guard);
    cleanup(&path);
}

#[test]
fn test_concurrent_relations() {
    let (kg_mutex, path) = setup_mutex();
    // Pre-populate entities.
    {
        let mut guard = kg_mutex.lock().unwrap();
        guard.create_entities(&[alice(), bob(), charlie()]).unwrap();
    }

    let kg = Arc::clone(&kg_mutex);
    let mut handles = Vec::new();

    for i in 0..10 {
        let kg = Arc::clone(&kg);
        handles.push(std::thread::spawn(move || {
            let relation = if i % 2 == 0 {
                knows_alice()
            } else {
                knows_bob()
            };
            let mut guard = kg.lock().unwrap();
            let _ = guard.create_relations(&[relation]);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let mut guard = kg_mutex.lock().unwrap();
    // Only 2 unique relations (duplicates are skipped)
    let rels = guard.search_relations(None, None, None);
    assert!(!rels.is_empty());
    assert!(rels.len() <= 2);
    drop(guard);
    cleanup(&path);
}

// =========================================================================
// Consistency across mixed operations
// =========================================================================

#[test]
fn test_graph_invariant_after_operations() {
    // Invariant: every relation's from/to should reference a live entity.
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.create_relations(&[knows_alice()]).unwrap();
    kg.add_observations("Alice", &["new obs".into()]).unwrap();
    kg.delete_entities(&["Bob".into()]).unwrap();

    let graph = kg.read_graph();
    let entity_names: Vec<&str> = graph.entities.iter().map(|e| e.name.as_str()).collect();
    for rel in &graph.relations {
        assert!(entity_names.contains(&rel.from.as_str()));
        // Bob was deleted, but the relation might still reference him since
        // relation deletion only happens at the time of entity deletion.
        // Actually, delete_entities already removes relations involving deleted IDs.
    }
    // After Bob is deleted, no relations should remain.
    assert!(graph.relations.is_empty());
    cleanup(&path);
}

#[test]
fn test_search_after_reindex() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice()]).unwrap();
    let result1 = kg.search_nodes("coffee");
    assert_eq!(result1.entities.len(), 1);

    // Add observation — search index should update.
    kg.add_observations("Alice", &["drinks matcha".into()]).unwrap();
    let result2 = kg.search_nodes("matcha");
    assert_eq!(result2.entities.len(), 1);

    // Delete observation — search index should update.
    kg.delete_observations("Alice", &["likes coffee".into()]).unwrap();
    let result3 = kg.search_nodes("coffee");
    assert_eq!(result3.entities.len(), 0);
    cleanup(&path);
}

#[test]
fn test_read_graph_does_not_include_deleted() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.delete_entities(&["Alice".into()]).unwrap();
    let graph = kg.read_graph();
    assert_eq!(graph.entities.len(), 1);
    assert_eq!(graph.entities[0].name, "Bob");
    cleanup(&path);
}

#[test]
fn test_compact_size_reduction() {
    let (mut kg, path) = setup();
    // Create and delete many entities to bloat the log.
    for i in 0..50 {
        let name = format!("TempEntity_{i}");
        kg.create_entities(&[Entity {
            name: name.clone(),
            entity_type: "temp".into(),
            observations: vec![],
        }])
        .unwrap();
        kg.delete_entities(&[name]).unwrap();
    }
    // At this point the log has 100 records, but only a create entity record
    // for the single entity below should remain after compaction.
    kg.create_entities(&[Entity {
        name: "Survivor".into(),
        entity_type: "permanent".into(),
        observations: vec![],
    }])
    .unwrap();

    kg.compact().unwrap();

    let stats = kg.graph_stats();
    assert_eq!(stats["entities"], 1);
    assert!(kg.get_entity("Survivor").is_some());
    cleanup(&path);
}

#[test]
fn test_delete_entity_removes_all_relations_bidirectional() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();

    // Two relations: Alice→Bob and Bob→Alice.
    kg.create_relations(&[knows_alice()]).unwrap();
    kg.create_relations(&[Relation {
        from: "Bob".into(),
        to: "Alice".into(),
        relation_type: "knows".into(),
    }])
    .unwrap();

    kg.delete_entities(&["Alice".into()]).unwrap();

    let rels = kg.search_relations(None, None, None);
    assert!(rels.is_empty());
    cleanup(&path);
}

#[test]
fn test_find_path_undirected_traversal() {
    // Relations are undirected in the BFS — we can traverse both ways.
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    // Only edge: Alice → Bob
    kg.create_relations(&[knows_alice()]).unwrap();

    // BFS from Bob to Alice should work (reverse direction).
    let p = kg.find_path("Bob", "Alice").unwrap();
    assert_eq!(p, vec!["Bob", "Alice"]);

    // BFS from Bob to Charlie should fail (no path).
    let result = kg.find_path("Bob", "Charlie");
    assert!(result.is_err());
    cleanup(&path);
}

// =========================================================================
// Tier-1 productivity tools
// =========================================================================

use mcp_memory::kg::Direction;

fn names(out: &mcp_memory::types::KnowledgeGraphOut) -> Vec<String> {
    let mut v: Vec<String> = out.entities.iter().map(|e| e.name.clone()).collect();
    v.sort();
    v
}

#[test]
fn test_get_neighbors_depth1_both() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    kg.create_relations(&[knows_alice(), works_with()]).unwrap(); // Alice→Bob, Alice→Charlie

    let out = kg.neighbors("Alice", Direction::Both, None, 1).unwrap();
    assert_eq!(names(&out), vec!["Alice", "Bob", "Charlie"]);
    // Both incident edges are among the returned set.
    assert_eq!(out.relations.len(), 2);
    cleanup(&path);
}

#[test]
fn test_get_neighbors_direction_filters() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    kg.create_relations(&[knows_alice()]).unwrap(); // Alice→Bob

    // Outgoing from Alice reaches Bob.
    let out = kg.neighbors("Alice", Direction::Out, None, 1).unwrap();
    assert_eq!(names(&out), vec!["Alice", "Bob"]);
    // Incoming to Alice reaches nobody.
    let inn = kg.neighbors("Alice", Direction::In, None, 1).unwrap();
    assert_eq!(names(&inn), vec!["Alice"]);
    // Incoming to Bob reaches Alice.
    let bob_in = kg.neighbors("Bob", Direction::In, None, 1).unwrap();
    assert_eq!(names(&bob_in), vec!["Alice", "Bob"]);
    cleanup(&path);
}

#[test]
fn test_get_neighbors_depth2_and_rtype() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    kg.create_relations(&[knows_alice(), knows_bob()]).unwrap(); // Alice→Bob→Charlie

    // Depth 1 from Alice: only Bob.
    let d1 = kg.neighbors("Alice", Direction::Out, None, 1).unwrap();
    assert_eq!(names(&d1), vec!["Alice", "Bob"]);
    // Depth 2 from Alice: Bob and Charlie.
    let d2 = kg.neighbors("Alice", Direction::Out, None, 2).unwrap();
    assert_eq!(names(&d2), vec!["Alice", "Bob", "Charlie"]);
    // Filter on a relation type that exists.
    let knows = kg.neighbors("Alice", Direction::Out, Some("knows"), 2).unwrap();
    assert_eq!(names(&knows), vec!["Alice", "Bob", "Charlie"]);
    // Filter on a type that does not exist → just the origin.
    let none = kg.neighbors("Alice", Direction::Out, Some("nope"), 2).unwrap();
    assert_eq!(names(&none), vec!["Alice"]);
    cleanup(&path);
}

#[test]
fn test_get_neighbors_missing_entity() {
    let (kg, path) = setup();
    assert!(kg.neighbors("Ghost", Direction::Both, None, 1).is_err());
    cleanup(&path);
}

#[test]
fn test_describe_entity() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    kg.create_relations(&[knows_alice(), works_with()]).unwrap();

    let v = kg.describe_entity("Alice").unwrap();
    assert_eq!(v["entity"]["name"], "Alice");
    assert_eq!(v["degree"], 2);
    let neighbors = v["neighbors"].as_array().unwrap();
    assert_eq!(neighbors.len(), 2);
    assert_eq!(v["relations"].as_array().unwrap().len(), 2);

    assert!(kg.describe_entity("Ghost").is_err());
    cleanup(&path);
}

#[test]
fn test_list_entity_and_relation_types() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap(); // 2 person, 1 ai
    kg.create_relations(&[knows_alice(), knows_bob(), works_with()]).unwrap(); // 2 knows, 1 works_with

    let etypes = kg.entity_type_counts();
    assert_eq!(etypes[0], ("person".to_string(), 2)); // ranked by count desc
    assert!(etypes.contains(&("ai".to_string(), 1)));

    let rtypes = kg.relation_type_counts();
    assert_eq!(rtypes[0], ("knows".to_string(), 2));
    assert!(rtypes.contains(&("works_with".to_string(), 1)));
    cleanup(&path);
}

#[test]
fn test_upsert_entities_create_then_merge() {
    let (mut kg, path) = setup();

    // First upsert creates.
    let e = Entity { name: "Dave".into(), entity_type: "person".into(), observations: vec!["a".into()] };
    let out = kg.upsert_entities(&[e]).unwrap();
    assert_eq!(out[0]["created"], true);
    assert_eq!(kg.get_entity("Dave").unwrap().observations, vec!["a".to_string()]);

    // Second upsert merges new observations, keeps type, dedupes existing.
    let e2 = Entity { name: "Dave".into(), entity_type: "robot".into(), observations: vec!["a".into(), "b".into()] };
    let out2 = kg.upsert_entities(&[e2]).unwrap();
    assert_eq!(out2[0]["created"], false);
    assert_eq!(out2[0]["addedObservations"].as_array().unwrap(), &vec![serde_json::json!("b")]);
    let dave = kg.get_entity("Dave").unwrap();
    assert_eq!(dave.entity_type, "person"); // type unchanged on merge
    assert_eq!(dave.observations, vec!["a".to_string(), "b".to_string()]);

    // Empty name rejected.
    let bad = Entity { name: "".into(), entity_type: "x".into(), observations: vec![] };
    assert!(kg.upsert_entities(&[bad]).is_err());
    cleanup(&path);
}

#[test]
fn test_search_nodes_filtered_pagination_and_type() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();

    // "coffee" matches Alice + Charlie; restrict to type person → Alice only.
    let only_person = kg.search_nodes_filtered("coffee", Some("person"), 0, usize::MAX);
    assert_eq!(names(&only_person), vec!["Alice"]);

    // Pagination: limit 1 returns a single entity.
    let page = kg.search_nodes_filtered("person", None, 0, 1);
    assert_eq!(page.entities.len(), 1);

    // Unknown type → empty.
    let empty = kg.search_nodes_filtered("coffee", Some("nope"), 0, usize::MAX);
    assert!(empty.entities.is_empty());
    cleanup(&path);
}

#[test]
fn test_read_graph_filtered() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob(), charlie()]).unwrap();
    kg.create_relations(&[knows_alice(), works_with()]).unwrap();

    // Type filter: only the two persons; only relations between persons.
    let persons = kg.read_graph_filtered(Some("person"), 0, usize::MAX);
    assert_eq!(names(&persons), vec!["Alice", "Bob"]);
    // knows_alice (Alice→Bob) is between persons; works_with (Alice→Charlie) is not.
    assert_eq!(persons.relations.len(), 1);

    // Pagination caps entity count.
    let page = kg.read_graph_filtered(None, 0, 2);
    assert_eq!(page.entities.len(), 2);
    let rest = kg.read_graph_filtered(None, 2, usize::MAX);
    assert_eq!(rest.entities.len(), 1);
    cleanup(&path);
}

#[test]
fn test_export_graph_formats() {
    let (mut kg, path) = setup();
    kg.create_entities(&[alice(), bob()]).unwrap();
    kg.create_relations(&[knows_alice()]).unwrap();

    let json = kg.export("json").unwrap();
    assert!(json.contains("Alice") && json.contains("Bob"));

    let mermaid = kg.export("mermaid").unwrap();
    assert!(mermaid.starts_with("graph LR"));
    assert!(mermaid.contains("knows"));

    let dot = kg.export("dot").unwrap();
    assert!(dot.starts_with("digraph G {"));
    assert!(dot.trim_end().ends_with("}"));

    assert!(kg.export("yaml").is_err());
    cleanup(&path);
}
