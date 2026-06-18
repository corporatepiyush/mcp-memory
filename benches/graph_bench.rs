//! Comprehensive microbenchmarks for every KnowledgeGraph operation.
//!
//! Run:
//!   cargo bench --bench graph_bench
//!
//! All benchmarks share one pre-built graph (40k entities, 120k relations) to
//! avoid paying the build cost repeatedly.  Write-tool benchmarks operate on a
//! fresh graph of the same size so mutations don't pollute the shared read graph.

use std::hint::black_box;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;

use mcp_memory::kg::{Direction, GraphHandle, KnowledgeGraph};
use mcp_memory::types::{Entity, Relation};

const N_ENTITIES: usize = 40_000;
const N_RELATIONS: usize = 120_000;

static SEQ: AtomicU64 = AtomicU64::new(0);

fn bench_path() -> (String, String) {
    let pid = std::process::id();
    let seq = SEQ.fetch_add(1, Ordering::SeqCst);
    let base = format!("{}/mcp_bench_{pid}_{seq}", std::env::temp_dir().display());
    (format!("{}.bin", base), format!("{}_write.bin", base))
}

/// Build a deterministic large graph.
fn build_graph(path: &str) -> (KnowledgeGraph, Vec<String>) {
    let _ = std::fs::remove_file(path);
    let mut kg = KnowledgeGraph::new(Path::new(path)).unwrap();
    let mut rng = SmallRng::seed_from_u64(7);

    let names: Vec<String> = (0..N_ENTITIES).map(|i| format!("entity_{i}")).collect();
    let entities: Vec<Entity> = names
        .iter()
        .map(|name| {
            let etype = format!("type_{}", rng.random_range(0..32));
            let obs = vec![format!("obs_{}", rng.random_range(0..1_000_000))];
            Entity { name: name.clone(), entity_type: etype, observations: obs }
        })
        .collect();
    for chunk in entities.chunks(10_000) {
        kg.create_entities(chunk).unwrap();
    }

    let relations: Vec<Relation> = (0..N_RELATIONS)
        .map(|_| {
            let from = names[rng.random_range(0..names.len())].clone();
            let to = names[rng.random_range(0..names.len())].clone();
            Relation { from, to, relation_type: "knows".into() }
        })
        .collect();
    for chunk in relations.chunks(10_000) {
        kg.create_relations(chunk).unwrap();
    }
    kg.flush_and_sync().unwrap();
    (kg, names)
}

/// Build a small writeable graph for write benchmarks (100 entities, 300 relations).
fn build_small_write_graph(w_path: &str) -> (KnowledgeGraph, Vec<String>) {
    let _ = std::fs::remove_file(w_path);
    let mut kg = KnowledgeGraph::new(Path::new(w_path)).unwrap();
    let mut rng = SmallRng::seed_from_u64(7);

    let names: Vec<String> = (0..100).map(|i| format!("entity_{i}")).collect();
    let entities: Vec<Entity> = names
        .iter()
        .map(|name| {
            let etype = format!("type_{}", rng.random_range(0..8));
            let obs = vec![format!("obs_{}", rng.random_range(0..100))];
            Entity { name: name.clone(), entity_type: etype, observations: obs }
        })
        .collect();
    kg.create_entities(&entities).unwrap();

    let relations: Vec<Relation> = (0..300)
        .map(|_| {
            let from = names[rng.random_range(0..names.len())].clone();
            let to = names[rng.random_range(0..names.len())].clone();
            Relation { from, to, relation_type: "knows".into() }
        })
        .collect();
    kg.create_relations(&relations).unwrap();
    kg.flush_and_sync().unwrap();
    (kg, names)
}

fn benches(c: &mut Criterion) {
    let (r_path, w_path) = bench_path();
    let (kg, names) = build_graph(&r_path);
    let _ = build_small_write_graph(&w_path);
    let mut rng = SmallRng::seed_from_u64(13);

    // ── Read tools ────────────────────────────────────────────────────────

    c.bench_function("get_entity", |b| {
        b.iter(|| {
            let name = &names[rng.random_range(0..names.len())];
            black_box(kg.get_entity(black_box(name)));
        })
    });

    c.bench_function("batch_get_entities_10", |b| {
        b.iter(|| {
            let pick: Vec<String> = (0..10)
                .map(|_| names[rng.random_range(0..names.len())].clone())
                .collect();
            black_box(kg.batch_get_entities(black_box(&pick)));
        })
    });

    c.bench_function("search_relations_from", |b| {
        b.iter(|| {
            let name = &names[rng.random_range(0..names.len())];
            black_box(kg.search_relations(Some(name), None, None));
        })
    });

    c.bench_function("search_relations_to", |b| {
        b.iter(|| {
            let name = &names[rng.random_range(0..names.len())];
            black_box(kg.search_relations(None, Some(name), None));
        })
    });

    c.bench_function("open_nodes_10", |b| {
        b.iter(|| {
            let pick: Vec<String> = (0..10)
                .map(|_| names[rng.random_range(0..names.len())].clone())
                .collect();
            black_box(kg.open_nodes(black_box(&pick)));
        })
    });

    c.bench_function("describe_entity", |b| {
        b.iter(|| {
            let name = &names[rng.random_range(0..names.len())];
            let _ = black_box(kg.describe_entity(black_box(name)));
        })
    });

    c.bench_function("neighbors_depth1", |b| {
        b.iter(|| {
            let name = &names[rng.random_range(0..names.len())];
            black_box(kg.neighbors(black_box(name), Direction::Both, None, 1).unwrap());
        })
    });

    c.bench_function("neighbors_depth2", |b| {
        b.iter(|| {
            let name = &names[rng.random_range(0..names.len())];
            black_box(kg.neighbors(black_box(name), Direction::Both, None, 2).unwrap());
        })
    });

    c.bench_function("find_path", |b| {
        b.iter(|| {
            let a = &names[rng.random_range(0..names.len())];
            let z = &names[rng.random_range(0..names.len())];
            let _ = black_box(kg.find_path(black_box(a), black_box(z)));
        })
    });

    c.bench_function("find_all_paths", |b| {
        b.iter(|| {
            let a = &names[rng.random_range(0..names.len())];
            let z = &names[rng.random_range(0..names.len())];
            let _ = black_box(kg.find_all_paths(black_box(a), black_box(z), 4, 10));
        })
    });

    c.bench_function("extract_subgraph_depth1", |b| {
        b.iter(|| {
            let pick: Vec<String> = (0..3)
                .map(|_| names[rng.random_range(0..names.len())].clone())
                .collect();
            black_box(kg.extract_subgraph(black_box(&pick), 1).unwrap());
        })
    });

    c.bench_function("extract_subgraph_depth2", |b| {
        b.iter(|| {
            let pick: Vec<String> = (0..3)
                .map(|_| names[rng.random_range(0..names.len())].clone())
                .collect();
            black_box(kg.extract_subgraph(black_box(&pick), 2).unwrap());
        })
    });

    c.bench_function("read_graph_full", |b| {
        b.iter(|| black_box(kg.read_graph()));
    });

    c.bench_function("read_graph_filtered_type", |b| {
        b.iter(|| black_box(kg.read_graph_filtered(Some("type_0"), 0, 1000)));
    });

    c.bench_function("search_nodes", |b| {
        b.iter(|| black_box(kg.search_nodes(black_box("entity_"))));
    });

    c.bench_function("search_nodes_filtered", |b| {
        b.iter(|| black_box(kg.search_nodes_filtered(black_box("entity_"), Some("type_0"), 0, 20)));
    });

    c.bench_function("graph_stats", |b| {
        b.iter(|| black_box(kg.graph_stats()));
    });

    c.bench_function("entity_type_counts", |b| {
        b.iter(|| black_box(kg.entity_type_counts()));
    });

    c.bench_function("relation_type_counts", |b| {
        b.iter(|| black_box(kg.relation_type_counts()));
    });

    c.bench_function("export_json", |b| {
        b.iter(|| black_box(kg.export("json")));
    });

    c.bench_function("export_mermaid", |b| {
        b.iter(|| black_box(kg.export("mermaid")));
    });

    c.bench_function("export_dot", |b| {
        b.iter(|| black_box(kg.export("dot")));
    });

    // ── Write tools (on fresh graph) ──────────────────────────────────────
    //
    // Each write benchmark uses a dedicated fresh graph so mutations never
    // leak between benchmarks.

    c.bench_function("create_entities_100", |b| {
        b.iter_batched_ref(
            || {
                let (kg, _) = build_small_write_graph(&format!("{w_path}_create"));
                kg
            },
            |kg| {
                let entities: Vec<Entity> = (0..100)
                    .map(|i| Entity {
                        name: format!("new_create_{i}"),
                        entity_type: "bench_type".into(),
                        observations: vec!["obs".into()],
                    })
                    .collect();
                black_box(kg.create_entities(black_box(&entities)).unwrap());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    c.bench_function("create_relations_100", |b| {
        b.iter_batched_ref(
            || {
                let (kg, names) = build_small_write_graph(&format!("{w_path}_rel"));
                (kg, names)
            },
            |(kg, names)| {
                let relations: Vec<Relation> = (0..100)
                    .map(|_| {
                        let from = names[rng.random_range(0..names.len())].clone();
                        let to = names[rng.random_range(0..names.len())].clone();
                        Relation { from, to, relation_type: "knows".into() }
                    })
                    .collect();
                black_box(kg.create_relations(black_box(&relations)).unwrap());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    c.bench_function("add_observations", |b| {
        b.iter_batched_ref(
            || {
                let (kg, names) = build_small_write_graph(&format!("{w_path}_addobs"));
                let idx = rng.random_range(0..names.len());
                let name = names[idx].clone();
                let contents: Vec<String> = (0..10)
                    .map(|i| format!("new_obs_{name}_{i}"))
                    .collect();
                (kg, name, contents)
            },
            |(kg, name, contents)| {
                black_box(kg.add_observations(black_box(name), black_box(contents)).unwrap());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    c.bench_function("delete_entities", |b| {
        b.iter_batched_ref(
            || {
                let (kg, names) = build_small_write_graph(&format!("{w_path}_delent"));
                let del: Vec<String> = names[0..100].to_vec();
                (kg, del)
            },
            |(kg, del)| {
                black_box(kg.delete_entities(black_box(del)).unwrap());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    c.bench_function("delete_observations", |b| {
        b.iter_batched_ref(
            || {
                let (kg, names) = build_small_write_graph(&format!("{w_path}_delobs"));
                let name = names[0].clone();
                let obs = vec!["nonexistent_obs_to_delete".to_string()];
                (kg, name, obs)
            },
            |(kg, name, obs)| {
                let _ = black_box(kg.delete_observations(black_box(name), black_box(obs)));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    c.bench_function("delete_relations", |b| {
        b.iter_batched_ref(
            || {
                let (kg, names) = build_small_write_graph(&format!("{w_path}_delrel"));
                let rels: Vec<Relation> = (0..100)
                    .map(|_| Relation {
                        from: names[rng.random_range(0..names.len())].clone(),
                        to: names[rng.random_range(0..names.len())].clone(),
                        relation_type: "knows".into(),
                    })
                    .collect();
                (kg, rels)
            },
            |(kg, rels)| {
                black_box(kg.delete_relations(black_box(rels)).unwrap());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    c.bench_function("upsert_new", |b| {
        b.iter_batched_ref(
            || {
                let (kg, _) = build_small_write_graph(&format!("{w_path}_upsnew"));
                kg
            },
            |kg| {
                let entities: Vec<Entity> = (0..100)
                    .map(|i| Entity {
                        name: format!("new_upsert_{i}"),
                        entity_type: "bench_type".into(),
                        observations: vec!["obs".into()],
                    })
                    .collect();
                black_box(kg.upsert_entities(black_box(&entities)).unwrap());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    c.bench_function("upsert_existing", |b| {
        b.iter_batched_ref(
            || {
                let (kg, names) = build_small_write_graph(&format!("{w_path}_upsex"));
                let entities: Vec<Entity> = names[0..100]
                    .iter()
                    .map(|n| Entity {
                        name: n.clone(),
                        entity_type: "bench_type".into(),
                        observations: vec!["new_upsert_obs".into()],
                    })
                    .collect();
                (kg, entities)
            },
            |(kg, entities)| {
                black_box(kg.upsert_entities(black_box(entities)).unwrap());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    c.bench_function("merge_entities", |b| {
        b.iter_batched_ref(
            || {
                let (kg, names) = build_small_write_graph(&format!("{w_path}_merge"));
                let src = names[0].clone();
                let tgt = names[1].clone();
                (kg, src, tgt)
            },
            |(kg, src, tgt)| {
                black_box(kg.merge_entities(black_box(src), black_box(tgt)).unwrap());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    c.bench_function("compact", |b| {
        b.iter_batched_ref(
            || {
                let (mut kg, names) = build_small_write_graph(&format!("{w_path}_compact"));
                // Delete some entities so compact has work to do
                let del: Vec<String> = names[0..50].to_vec();
                kg.delete_entities(&del).unwrap();
                kg
            },
            |kg| {
                black_box(kg.compact().unwrap());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // ── Dispatch (server) benchmarks ──────────────────────────────────────
    //
    // These measure the full `dispatch_line` path, including JSON-RPC parsing,
    // handler logic, and response serialization. They use a pre-built
    // GraphHandle so the read_graph cache is warm on the 2nd+ iteration.

    let gh_path = format!("{}/mcp_gh_{}_{}", std::env::temp_dir().display(), std::process::id(), SEQ.fetch_add(1, Ordering::SeqCst));
    {
        // Populate a graph via GraphHandle
        let gh = GraphHandle::new(Path::new(&gh_path)).unwrap();
        let entities: Vec<Entity> = (0..10_000)
            .map(|i| Entity {
                name: format!("entity_{i}"),
                entity_type: "bench_type".into(),
                observations: vec!["obs".into()],
            })
            .collect();
        for chunk in entities.chunks(10_000) {
            let mut wg = gh.write();
            wg.create_entities(chunk).unwrap();
            wg.flush_and_sync().unwrap();
        }
        drop(gh); // sync snapshot
    }
    let gh = GraphHandle::new(Path::new(&gh_path)).unwrap();

    c.bench_function("dispatch_read_graph", |b| {
        let line = r#"{"jsonrpc":"2.0","method":"tools/call","id":1,"params":{"name":"read_graph","arguments":{}}}"#;
        b.iter(|| {
            let resp = mcp_memory::server::dispatch_line(black_box(line), &gh);
            black_box(resp)
        });
    });

    c.bench_function("dispatch_search_nodes", |b| {
        let line = r#"{"jsonrpc":"2.0","method":"tools/call","id":1,"params":{"name":"search_nodes","arguments":{"query":"entity"}}}"#;
        b.iter(|| {
            let resp = mcp_memory::server::dispatch_line(black_box(line), &gh);
            black_box(resp)
        });
    });

    // Compare dispatch_read_graph for a small graph (100 entities) to show
    // the effect of cache + fast serialization vs. the old path.
    let small_path = format!("{}/mcp_gh_small_{}_{}", std::env::temp_dir().display(), std::process::id(), SEQ.fetch_add(1, Ordering::SeqCst));
    {
        let gh = GraphHandle::new(Path::new(&small_path)).unwrap();
        let entities: Vec<Entity> = (0..100)
            .map(|i| Entity {
                name: format!("entity_{i}"),
                entity_type: "bench_type".into(),
                observations: vec!["obs".into()],
            })
            .collect();
        let mut wg = gh.write();
        wg.create_entities(&entities).unwrap();
        wg.flush_and_sync().unwrap();
    }
    let gh_small = GraphHandle::new(Path::new(&small_path)).unwrap();

    c.bench_function("dispatch_read_graph_small_100", |b| {
        let line = r#"{"jsonrpc":"2.0","method":"tools/call","id":1,"params":{"name":"read_graph","arguments":{}}}"#;
        b.iter(|| {
            let resp = mcp_memory::server::dispatch_line(black_box(line), &gh_small);
            black_box(resp)
        });
    });

    // Compare: reading a Value-based tool through the standard dispatch path
    c.bench_function("dispatch_initialize", |b| {
        let line = r#"{"jsonrpc":"2.0","method":"initialize","id":1}"#;
        b.iter(|| {
            let resp = mcp_memory::server::dispatch_line(black_box(line), &gh);
            black_box(resp)
        });
    });

    // ── Raw serialization comparison ───────────────────────────────────────
    let snap = gh.read();
    c.bench_function("read_graph_owned", |b| {
        // Old path: build owned structs, then serde_json::to_string
        b.iter(|| {
            let out = black_box(&snap).read_graph();
            black_box(serde_json::to_string(&out).unwrap())
        });
    });

    c.bench_function("read_graph_json_direct", |b| {
        // New path: write JSON directly without owned structs
        b.iter(|| {
            black_box(black_box(&snap).read_graph_json())
        });
    });

    // Cleanup
    let _ = std::fs::remove_file(&r_path);
    let _ = std::fs::remove_file(&w_path);
    let _ = std::fs::remove_file(&gh_path);
    let _ = std::fs::remove_file(&small_path);
}

criterion_group!(graph, benches);
criterion_main!(graph);
