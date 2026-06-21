use std::num::NonZeroUsize;
use std::path::Path;
use std::time::{Duration, Instant};

use mcp_memory::config::{Durability, SqliteTuning};
use mcp_memory::kg::{Direction, GraphHandle};
use mcp_memory::types::{Entity, Relation};

fn main() {
    let path = Path::new("/tmp/mcp_memory_bench.db");
    for ext in ["", "-wal", "-shm"] {
        let p = format!("/tmp/mcp_memory_bench.db{ext}");
        let _ = std::fs::remove_file(&p);
    }

    let kg = GraphHandle::new(path, Durability::Async, SqliteTuning::default(), NonZeroUsize::new(10000).unwrap(), 4).expect("create KG");

    // ── Seed ──────────────────────────────────────────────────────────
    const N: usize = 1000;
    const OBS_PER_ENTITY: usize = 5;
    let entities: Vec<Entity> = (0..N)
        .map(|i| Entity {
            name: format!("entity_{i}"),
            entity_type: if i % 2 == 0 {
                "person".into()
            } else {
                "place".into()
            },
            observations: (0..OBS_PER_ENTITY)
                .map(|j| format!("observation_{i}_{j}"))
                .collect(),
        })
        .collect();

    let relations: Vec<Relation> = (0..N - 1)
        .map(|i| {
            let j = i + 1;
            Relation {
                from: format!("entity_{i}"),
                to: format!("entity_{j}"),
                relation_type: "edge".into(),
            }
        })
        .collect();

    // ── Measure ───────────────────────────────────────────────────────
    macro_rules! measure {
        ($name:expr, $n:expr, $body:expr) => {{
            let mut total = Duration::ZERO;
            for _ in 0..$n {
                let start = Instant::now();
                let _ = $body;
                total += start.elapsed();
            }
            let avg = total / $n as u32;
            println!("  {:30} {:>8} runs  avg {:>10?}  total {:>10?}",
                $name, $n, avg, total);
        }};
    }

    println!(
        "Benchmark: {} entities, {} obs/entity, {} relations",
        N,
        OBS_PER_ENTITY,
        N - 1
    );
    println!();

    // Warmup
    let _ = kg.get_entity_count();
    let _ = kg.get_relation_count();

    measure!("create_entities", 1, { kg.create_entities(&entities) });

    measure!("get_entity (cache hit)", N, {
        kg.get_entity("entity_0")
    });

    // Flush entity seq
    let _ = kg.get_entity_count();

    measure!("create_relations", 1, { kg.create_relations(&relations) });

    measure!("get_entity_count", 100, { kg.get_entity_count() });
    measure!("get_relation_count", 100, { kg.get_relation_count() });

    measure!("degree (cache hit)", 100, {
        kg.degree("entity_0", Direction::Outgoing)
    });
    measure!("degree (both)", 100, {
        kg.degree("entity_50", Direction::Both)
    });

    measure!("get_entity (cache miss — force)", N, {
        kg.get_entity("entity_99")
    });

    measure!("search_nodes (name match)", 20, {
        kg.search_nodes_filtered("entity_5", None, 0, 10)
    });

    measure!("search_nodes (obs match)", 20, {
        kg.search_nodes_filtered("observation_3_2", None, 0, 10)
    });

    measure!("search_nodes (filtered)", 20, {
        kg.search_nodes_filtered("entity", Some("person"), 0, 10)
    });

    measure!("read_graph (all)", 3, {
        kg.read_graph_filtered(None, 0, usize::MAX)
    });

    measure!("read_graph (filtered)", 3, {
        kg.read_graph_filtered(Some("person"), 0, usize::MAX)
    });

    measure!("open_nodes (single)", 20, {
        kg.open_nodes(&["entity_0".into()])
    });

    measure!("open_nodes (5 names)", 20, {
        kg.open_nodes(&[
            "entity_0".into(),
            "entity_10".into(),
            "entity_20".into(),
            "entity_30".into(),
            "entity_40".into(),
        ])
    });

    measure!("find_path", 50, { kg.find_path("entity_0", "entity_99") });

    measure!("entities_exist (10 names)", 50, {
        kg.entities_exist(&[
            "entity_0".into(),
            "missing".into(),
            "entity_50".into(),
            "entity_99".into(),
            "also_missing".into(),
            "entity_25".into(),
            "entity_75".into(),
            "nope".into(),
            "entity_1".into(),
            "entity_100".into(),
        ])
    });

    measure!("describe_entity", 50, { kg.describe_entity("entity_42") });

    measure!("entity_type_counts", 20, { kg.entity_type_counts() });
    measure!("relation_type_counts", 20, { kg.relation_type_counts() });

    measure!("batch_get_entities (10)", 20, {
        kg.batch_get_entities(&[
            "entity_0".into(),
            "entity_1".into(),
            "entity_2".into(),
            "missing".into(),
            "entity_3".into(),
            "entity_4".into(),
            "entity_5".into(),
            "nonexistent".into(),
            "entity_6".into(),
            "entity_7".into(),
        ])
    });

    measure!("neighbors (depth 1)", 20, {
        kg.neighbors("entity_50", Direction::Both, None, 1)
    });

    measure!("neighbors (depth 2)", 10, {
        kg.neighbors("entity_50", Direction::Both, None, 2)
    });

    measure!("export (json)", 5, { kg.export("json", i64::MAX) });

    measure!("find_all_paths (A→C, depth 5)", 20, {
        kg.find_all_paths("entity_0", "entity_2", 5, 10)
    });

    // ── Mutating ops (single runs to avoid state contamination) ───────

    measure!("add_observations (2 obs)", 20, {
        kg.add_observations("entity_0", &["new_obs_a".into(), "new_obs_b".into()])
    });

    // Reset entity_0 for delete test
    let _ = kg.add_observations("entity_1", &["to_delete".into()]);
    measure!("delete_observations (1 obs)", 20, {
        kg.delete_observations("entity_1", &["to_delete".into()])
    });

    measure!("upsert_entities (type change + obs)", 20, {
        kg.upsert_entities(&[Entity {
            name: "entity_0".into(),
            entity_type: "person".into(),
            observations: vec!["existing".into(), "upserted_obs".into()],
        }])
    });

    measure!("search_relations (from)", 20, {
        kg.search_relations(Some("entity_0"), None, None)
    });

    measure!("search_relations (from+type)", 20, {
        kg.search_relations(Some("entity_0"), None, Some("edge"))
    });

    // Cleanup
    let _ = std::fs::remove_file(path);
    for ext in ["-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("/tmp/mcp_memory_bench.db{ext}"));
    }
}
