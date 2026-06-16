//! Hot-path microbenchmarks for the in-memory graph.
//!
//! The point of these is to make the cache-layout question (the `cache_align`
//! feature) an empirical one. Run both layouts on the target box and compare:
//!
//! ```text
//! cargo bench --bench graph_bench
//! cargo bench --bench graph_bench --features cache_align
//! ```
//!
//! `get_entity` is the straddle-sensitive **point read** (random slot index);
//! `read_graph` / `search_relations` / `find_path` are **scans** where the
//! aligned layout trades extra stride bytes for no line splits.

use std::hint::black_box;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;

use mcp_memory::kg::{Direction, KnowledgeGraph};
use mcp_memory::types::{Entity, Relation};

// Sized so the entity-slot vec comfortably exceeds a typical L2 on x86, where
// cache-line straddle (the thing `cache_align` targets) actually shows up.
// Lower these if the one-time index build dominates your iteration loop.
const N_ENTITIES: usize = 40_000;
const N_RELATIONS: usize = 120_000;

static SEQ: AtomicU64 = AtomicU64::new(0);

fn bench_path() -> String {
    let pid = std::process::id();
    let seq = SEQ.fetch_add(1, Ordering::SeqCst);
    format!("{}/mcp_bench_{pid}_{seq}.bin", std::env::temp_dir().display())
}

/// Build a deterministic large graph once and return it plus the entity names.
fn build_graph() -> (KnowledgeGraph, String, Vec<String>) {
    let path = bench_path();
    let _ = std::fs::remove_file(&path);
    let mut kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
    let mut rng = SmallRng::from_seed([7u8; 32]);

    let names: Vec<String> = (0..N_ENTITIES).map(|i| format!("entity_{i}")).collect();
    let entities: Vec<Entity> = names
        .iter()
        .map(|name| {
            let etype = format!("type_{}", rng.gen_range(0..32));
            let obs = vec![format!("obs_{}", rng.gen_range(0..1_000_000))];
            Entity { name: name.clone(), entity_type: etype, observations: obs }
        })
        .collect();
    // Insert in chunks to stay under the per-request cap-free internal API.
    for chunk in entities.chunks(10_000) {
        kg.create_entities(chunk).unwrap();
    }

    let relations: Vec<Relation> = (0..N_RELATIONS)
        .map(|_| {
            let from = names[rng.gen_range(0..names.len())].clone();
            let to = names[rng.gen_range(0..names.len())].clone();
            Relation { from, to, relation_type: "knows".into() }
        })
        .collect();
    for chunk in relations.chunks(10_000) {
        kg.create_relations(chunk).unwrap();
    }
    kg.flush_and_sync().unwrap();
    (kg, path, names)
}

fn benches(c: &mut Criterion) {
    let (kg, path, names) = build_graph();
    let mut rng = SmallRng::from_seed([13u8; 32]);

    // Point read: random slot lookups — the cache-line-straddle case.
    c.bench_function("get_entity_random", |b| {
        b.iter(|| {
            let name = &names[rng.gen_range(0..names.len())];
            black_box(kg.get_entity(black_box(name)));
        })
    });

    // Scan: filter the flat relation vec by source.
    c.bench_function("search_relations_from", |b| {
        b.iter(|| {
            let name = &names[rng.gen_range(0..names.len())];
            black_box(kg.search_relations(Some(name), None, None));
        })
    });

    // Scan + BFS over the relation adjacency.
    c.bench_function("find_path_random", |b| {
        b.iter(|| {
            let a = &names[rng.gen_range(0..names.len())];
            let z = &names[rng.gen_range(0..names.len())];
            let _ = black_box(kg.find_path(black_box(a), black_box(z)));
        })
    });

    // 1-hop neighborhood: single relation-vec pass.
    c.bench_function("neighbors_depth1", |b| {
        b.iter(|| {
            let name = &names[rng.gen_range(0..names.len())];
            black_box(kg.neighbors(black_box(name), Direction::Both, None, 1).unwrap());
        })
    });

    // Full entity scan.
    c.bench_function("read_graph_full", |b| {
        b.iter(|| black_box(kg.read_graph()));
    });

    // Index-backed substring search.
    c.bench_function("search_nodes", |b| {
        b.iter(|| black_box(kg.search_nodes(black_box("entity_42"))));
    });

    // Note: point *mutations* (add_observations/delete_observations) resolve a
    // slot via name_table then read/write that single StoredEntity in place —
    // the same slot-access pattern as `get_entity_random` above, so that bench
    // is the representative cache-line-straddle case for writes too.

    let _ = std::fs::remove_file(&path);
}

criterion_group!(graph, benches);
criterion_main!(graph);
