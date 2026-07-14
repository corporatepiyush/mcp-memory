use serde_json::{Value, json};

use crate::errors::{MCSError, Result};
use crate::kg::{GraphHandle, push_json_str};
use crate::vector_store::{EntityId, VectorStore, with_scratch};
use rustc_hash::FxHashMap;

type HybridResult = Vec<(String, String, f64, f64, f64)>;

use rusqlite::params;

const MAX_EMBEDDING_DIMS: usize = 4096;
const MAX_TOP_K: usize = 100;
const DEFAULT_TOP_K: usize = 10;
const MAX_NAME_BYTES: usize = 1024;
/// Cap on items in a single `vector_batch_upsert` call.
const MAX_BATCH_ITEMS: usize = 1024;

fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(MCSError::InvalidParams("Name must not be empty".into()));
    }
    if name.len() > MAX_NAME_BYTES {
        return Err(MCSError::InvalidParams(format!(
            "Name too long (max {MAX_NAME_BYTES} bytes)"
        )));
    }
    Ok(())
}

fn parse_embedding(val: &Value) -> Result<Vec<f64>> {
    let arr = val
        .as_array()
        .ok_or_else(|| MCSError::InvalidParams("'embedding' must be an array of numbers".into()))?;
    if arr.is_empty() {
        return Err(MCSError::InvalidParams("Embedding must not be empty".into()));
    }
    if arr.len() > MAX_EMBEDDING_DIMS {
        return Err(MCSError::InvalidParams(format!(
            "Embedding too large (max {MAX_EMBEDDING_DIMS} dimensions)"
        )));
    }
    let emb: Vec<f64> = arr
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| MCSError::InvalidParams("Embedding values must be numbers".into()))
        })
        .collect::<Result<_>>()?;
    Ok(emb)
}

fn opt_usize(params: &Value, key: &str, default: usize) -> Result<usize> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v.as_u64().map(|n| n as usize).ok_or_else(|| {
            MCSError::InvalidParams(format!("'{key}' must be a non-negative integer"))
        }),
    }
}

fn opt_f64(params: &Value, key: &str, default: f64) -> Result<f64> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v.as_f64().ok_or_else(|| {
            MCSError::InvalidParams(format!("'{key}' must be a number"))
        }),
    }
}

fn text_content(text: &str) -> Value {
    json!({
        "content": [{
            "type": "text",
            "text": text
        }]
    })
}

fn build_content_response(inner_json: &str) -> String {
    let mut out = String::with_capacity(64 + inner_json.len() + (inner_json.len() / 8));
    out.push_str(r#"{"content":[{"type":"text","text":"#);
    push_json_str(&mut out, inner_json);
    out.push_str(r#"}]}"#);
    out
}

pub fn handle_vector_upsert_embedding(
    vs: &VectorStore,
    _kg: &GraphHandle,
    args: Option<&Value>,
) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;

    let entity_name = params
        .get("entityName")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'entityName' parameter".into()))?;
    validate_name(entity_name)?;

    let embedding = parse_embedding(
        params
            .get("embedding")
            .ok_or_else(|| MCSError::InvalidParams("Missing 'embedding' parameter".into()))?,
    )?;

    let model = params
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    with_scratch(|buf| {
        buf.reserve(embedding.len());
        buf.extend(embedding.iter().map(|&v| v as f32));
        vs.upsert_embedding(entity_name, buf, model)
    })?;

    let text = serde_json::to_string(&json!({
        "entityName": entity_name,
        "dims": vs.dims(),
        "model": model,
    }))
    .map_err(MCSError::JsonError)?;

    Ok(text_content(&text))
}

pub fn handle_vector_search_entities(
    vs: &VectorStore,
    _kg: &GraphHandle,
    args: Option<&Value>,
) -> Result<String> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;

    let embedding = parse_embedding(
        params
            .get("embedding")
            .ok_or_else(|| MCSError::InvalidParams("Missing 'embedding' parameter".into()))?,
    )?;

    let top_k = opt_usize(params, "topK", DEFAULT_TOP_K)?
        .clamp(1, MAX_TOP_K);

    let entity_type = params
        .get("entityType")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());

    let json = with_scratch(|buf| {
        buf.reserve(embedding.len());
        buf.extend(embedding.iter().map(|&v| v as f32));
        vs.search_entities_json(buf, top_k, entity_type)
    })?;

    Ok(build_content_response(&json))
}

pub fn handle_vector_delete_embedding(
    vs: &VectorStore,
    _kg: &GraphHandle,
    args: Option<&Value>,
) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;

    let entity_name = params
        .get("entityName")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'entityName' parameter".into()))?;
    validate_name(entity_name)?;

    let deleted = vs.delete_embedding(entity_name)?;

    let text = serde_json::to_string(&json!({
        "deleted": deleted,
        "entityName": entity_name,
    }))
    .map_err(MCSError::JsonError)?;

    Ok(text_content(&text))
}

pub fn handle_hybrid_search(
    vs: &VectorStore,
    kg: &GraphHandle,
    args: Option<&Value>,
) -> Result<String> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;

    let query_text = params
        .get("queryText")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'queryText' parameter".into()))?;

    let query_embedding = parse_embedding(
        params
            .get("queryEmbedding")
            .ok_or_else(|| MCSError::InvalidParams("Missing 'queryEmbedding' parameter".into()))?,
    )?;

    let text_weight = opt_f64(params, "textWeight", 0.5)?;
    let vec_weight = opt_f64(params, "vecWeight", 0.5)?;
    let top_k = opt_usize(params, "topK", DEFAULT_TOP_K)?
        .clamp(1, MAX_TOP_K);

    let results = with_scratch(|buf| {
        buf.reserve(query_embedding.len());
        buf.extend(query_embedding.iter().map(|&v| v as f32));
        perform_hybrid_search(vs, kg, query_text, buf, text_weight, vec_weight, top_k)
    })?;

    let mut out = String::with_capacity(128 + results.len() * 80);
    out.push_str(r#"{"results":["#);
    for (i, (name, etype, score, txt_score, vec_score)) in results.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(r#"{"name":"#);
        push_json_str(&mut out, name);
        out.push_str(r#","entityType":"#);
        push_json_str(&mut out, etype);
        use std::fmt::Write;
        write!(
            out,
            r#","score":{:.6},"textScore":{:.6},"vecScore":{:.6}}}"#,
            score, txt_score, vec_score
        )
        .unwrap();
    }
    out.push_str(r#"],"count":"#);
    out.push_str(&results.len().to_string());
    out.push('}');

    Ok(build_content_response(&out))
}

fn perform_hybrid_search(
    vs: &VectorStore,
    kg: &GraphHandle,
    query_text: &str,
    query_emb: &[f32],
    text_weight: f64,
    vec_weight: f64,
    top_k: usize,
) -> Result<HybridResult> {
    let fetch_k = top_k * 3;
    let rrf_constant = 60.0;

    let vec_matches = vs.search_embeddings(query_emb, fetch_k)?;

    let kg_results = kg.search_nodes_filtered(query_text, None, 0, fetch_k);
    let mut text_matches: Vec<EntityIdAndName> = Vec::with_capacity(kg_results.len());
    for entity in &kg_results {
        if let Ok(Some(_)) = vs.get_entity_type(
            vs.name_to_id.get(&entity.name).map(|r| *r.value()).unwrap_or(-1),
        ) {
            let id = vs.name_to_id.get(&entity.name).map(|r| *r.value());
            text_matches.push(EntityIdAndName {
                id: id.unwrap_or(-1),
            });
        } else {
            let conn = vs.db.lock();
            let h = crate::kg::name_hash(&entity.name);
            let id: Option<i64> = conn
                .query_row(
                    "SELECT id FROM entity WHERE name_hash = ?1 AND name = ?2 AND flags = 0",
                    params![h, entity.name],
                    |row| row.get(0),
                )
                .ok();
            text_matches.push(EntityIdAndName {
                id: id.unwrap_or(-1),
            });
        }
    }

    let mut score_map: FxHashMap<EntityId, AggScore> = FxHashMap::with_capacity_and_hasher(
        vec_matches.len() + text_matches.len(),
        rustc_hash::FxBuildHasher,
    );

    for (rank, (id, _dist)) in vec_matches.iter().enumerate() {
        let entry = score_map.entry(*id).or_insert_with(|| AggScore {
            id: *id,
            total: 0.0,
            vec_score: 0.0,
            text_score: 0.0,
        });
        let rrf = vec_weight * (1.0 / (rrf_constant + rank as f64));
        entry.total += rrf;
        entry.vec_score += rrf;
    }

    for (rank, tm) in text_matches.iter().enumerate() {
        let entry = score_map.entry(tm.id).or_insert_with(|| AggScore {
            id: tm.id,
            total: 0.0,
            vec_score: 0.0,
            text_score: 0.0,
        });
        let rrf = text_weight * (1.0 / (rrf_constant + rank as f64));
        entry.total += rrf;
        entry.text_score += rrf;
    }

    let mut scored: Vec<AggScore> = score_map.into_values().collect();
    scored.sort_unstable_by(|a, b| b.total.partial_cmp(&a.total).unwrap_or(std::cmp::Ordering::Equal));

    if vs.graph_node_count() > 0 {
        let g = vs.graph.read();
        for entry in &mut scored {
            if let Some(nx) = vs.node_map.get(&entry.id) {
                let deg = g.neighbors(*nx).count() as f64;
                if deg > 0.0 {
                    let boost = 0.1 * (deg / (deg + 5.0));
                    entry.total += boost;
                }
            }
        }
        scored.sort_unstable_by(|a, b| b.total.partial_cmp(&a.total).unwrap_or(std::cmp::Ordering::Equal));
    }

    let conn = vs.db.lock();
    let mut results = Vec::with_capacity(top_k.min(scored.len()));
    for entry in scored.iter().take(top_k) {
        let name = vs
            .id_to_name
            .get(&entry.id)
            .map(|r| r.value().clone())
            .or_else(|| {
                conn.query_row(
                    "SELECT name FROM entity WHERE id = ?1 AND flags = 0",
                    params![entry.id],
                    |row| row.get::<_, String>(0),
                )
                .ok()
            })
            .unwrap_or_default();

        let etype: String = conn
            .query_row(
                "SELECT t.name FROM entity e JOIN type_dict t ON t.id = e.type_id WHERE e.id = ?1 AND e.flags = 0",
                params![entry.id],
                |row| row.get(0),
            )
            .unwrap_or_default();

        results.push((name, etype, entry.total, entry.text_score, entry.vec_score));
    }

    Ok(results)
}

struct EntityIdAndName {
    id: EntityId,
}

struct AggScore {
    id: EntityId,
    total: f64,
    vec_score: f64,
    text_score: f64,
}

pub fn handle_refresh_graph_cache(
    vs: &VectorStore,
    _kg: &GraphHandle,
    _args: Option<&Value>,
) -> Result<Value> {
    vs.rebuild_graph_cache()?;
    let text = serde_json::to_string(&json!({
        "nodes": vs.graph_node_count(),
        "edges": vs.graph_edge_count(),
    }))
    .map_err(MCSError::JsonError)?;
    Ok(text_content(&text))
}

pub fn handle_vector_store_stats(
    vs: &VectorStore,
    _kg: &GraphHandle,
    _args: Option<&Value>,
) -> Result<Value> {
    let (graph_bytes, vectors_bytes) = vs.index_memory_breakdown();
    let index_kind = match vs.index_kind() {
        crate::vector_store::IndexKind::Hnsw => "hnsw",
        crate::vector_store::IndexKind::Ivf => "ivf",
        crate::vector_store::IndexKind::TurboQuant => "turboquant",
    };
    let text = serde_json::to_string(&json!({
        "embeddingCount": vs.count(),
        "dims": vs.dims(),
        "indexKind": index_kind,
        "petgraphNodes": vs.graph_node_count(),
        "petgraphEdges": vs.graph_edge_count(),
        "indexCapacity": vs.index_capacity(),
        "indexMemoryBytes": vs.index_memory_bytes(),
        "indexGraphBytes": graph_bytes,
        "indexVectorsBytes": vectors_bytes,
    }))
    .map_err(MCSError::JsonError)?;
    Ok(text_content(&text))
}

/// Convert parsed `f64` numbers into the `f32` scratch buffer.
fn to_f32(emb: &[f64]) -> Vec<f32> {
    emb.iter().map(|&v| v as f32).collect()
}

#[inline]
fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Render resolved `(name, entityType, score)` rows as the standard results JSON.
fn build_named_results(rows: &[(String, String, f64)]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(64 + rows.len() * 64);
    out.push_str(r#"{"results":["#);
    for (i, (name, etype, score)) in rows.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(r#"{"name":"#);
        push_json_str(&mut out, name);
        out.push_str(r#","entityType":"#);
        push_json_str(&mut out, etype);
        write!(out, r#","score":{score:.6}}}"#).unwrap();
    }
    out.push_str(r#"],"count":"#);
    out.push_str(&rows.len().to_string());
    out.push('}');
    out
}

/// Bulk-ingest embeddings: `{ items: [{entityName, embedding, model?}, ...] }`.
/// Each item is upserted independently; per-item failures are reported rather
/// than aborting the batch — the shape RAG ingestion pipelines expect.
pub fn handle_vector_batch_upsert(
    vs: &VectorStore,
    _kg: &GraphHandle,
    args: Option<&Value>,
) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let items = params
        .get("items")
        .and_then(|v| v.as_array())
        .ok_or_else(|| MCSError::InvalidParams("'items' must be an array".into()))?;
    if items.len() > MAX_BATCH_ITEMS {
        return Err(MCSError::InvalidParams(format!(
            "Too many items (max {MAX_BATCH_ITEMS})"
        )));
    }

    // Parse every item first, then store the whole batch under one SQLite
    // transaction (one WAL commit instead of one per item).
    let mut errors: Vec<Value> = Vec::new();
    let mut parsed: Vec<(&str, Vec<f32>, &str)> = Vec::with_capacity(items.len());
    for item in items {
        let name = match item.get("entityName").and_then(|v| v.as_str()) {
            Some(n) if !n.is_empty() && n.len() <= MAX_NAME_BYTES => n,
            _ => {
                errors.push(json!({"entityName": item.get("entityName"), "error": "invalid entityName"}));
                continue;
            }
        };
        let emb = match item.get("embedding").map(parse_embedding) {
            Some(Ok(e)) => e,
            Some(Err(e)) => {
                errors.push(json!({"entityName": name, "error": e.to_string()}));
                continue;
            }
            None => {
                errors.push(json!({"entityName": name, "error": "missing embedding"}));
                continue;
            }
        };
        let model = item.get("model").and_then(|v| v.as_str()).unwrap_or("");
        parsed.push((name, to_f32(&emb), model));
    }

    let mut upserted = 0usize;
    for ((name, _, _), result) in parsed.iter().zip(vs.upsert_embeddings_batch(&parsed)) {
        match result {
            Ok(()) => upserted += 1,
            Err(e) => errors.push(json!({"entityName": name, "error": e.to_string()})),
        }
    }

    let text = serde_json::to_string(&json!({
        "upserted": upserted,
        "failed": errors.len(),
        "errors": errors,
    }))
    .map_err(MCSError::JsonError)?;
    Ok(text_content(&text))
}

/// Fetch the stored embedding for an entity: `{ entityName }`.
pub fn handle_vector_get_embedding(
    vs: &VectorStore,
    _kg: &GraphHandle,
    args: Option<&Value>,
) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let name = params
        .get("entityName")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'entityName' parameter".into()))?;
    validate_name(name)?;

    match vs.get_embedding_by_name(name)? {
        Some((_id, emb, model)) => {
            let text = serde_json::to_string(&json!({
                "entityName": name,
                "dims": emb.len(),
                "model": model,
                "embedding": emb,
            }))
            .map_err(MCSError::JsonError)?;
            Ok(text_content(&text))
        }
        None => {
            let text = serde_json::to_string(&json!({
                "entityName": name,
                "embedding": Value::Null,
                "found": false,
            }))
            .map_err(MCSError::JsonError)?;
            Ok(text_content(&text))
        }
    }
}

/// "More like this": find entities nearest to a given entity's own embedding.
/// `{ entityName, topK?, entityType?, excludeSelf? }`.
pub fn handle_vector_search_by_entity(
    vs: &VectorStore,
    _kg: &GraphHandle,
    args: Option<&Value>,
) -> Result<String> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let name = params
        .get("entityName")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'entityName' parameter".into()))?;
    validate_name(name)?;
    let top_k = opt_usize(params, "topK", DEFAULT_TOP_K)?.clamp(1, MAX_TOP_K);
    let entity_type = params
        .get("entityType")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    let exclude_self = params
        .get("excludeSelf")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let (id, emb, _model) = vs.get_embedding_by_name(name)?.ok_or_else(|| {
        MCSError::InvalidParams(format!("Entity '{name}' has no embedding"))
    })?;

    let mut exclude = std::collections::HashSet::new();
    if exclude_self {
        exclude.insert(id);
    }
    let rows = vs.search_resolved(&emb, top_k, entity_type, &exclude)?;
    let named: Vec<(String, String, f64)> = rows
        .into_iter()
        .map(|(_, n, t, d)| (n, t, f64::from(d)))
        .collect();
    Ok(build_content_response(&build_named_results(&named)))
}

/// Example-based recommendation: build a query from positive (and optional
/// negative) example entities and search. `{ positive: [names], negative?:
/// [names], topK?, entityType? }`. The example entities are excluded from results.
pub fn handle_vector_recommend(
    vs: &VectorStore,
    _kg: &GraphHandle,
    args: Option<&Value>,
) -> Result<String> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let top_k = opt_usize(params, "topK", DEFAULT_TOP_K)?.clamp(1, MAX_TOP_K);
    let entity_type = params
        .get("entityType")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());

    let positive = collect_names(params, "positive")?;
    if positive.is_empty() {
        return Err(MCSError::InvalidParams(
            "'positive' must contain at least one entity name".into(),
        ));
    }
    let negative = collect_names(params, "negative").unwrap_or_default();

    let dims = vs.dims() as usize;
    let mut query = vec![0.0f64; dims];
    let mut exclude = std::collections::HashSet::new();

    let mut pos_count = 0usize;
    for n in &positive {
        if let Some((id, emb, _)) = vs.get_embedding_by_name(n)? {
            if emb.len() != dims {
                continue;
            }
            for (q, &e) in query.iter_mut().zip(&emb) {
                *q += f64::from(e);
            }
            exclude.insert(id);
            pos_count += 1;
        }
    }
    if pos_count == 0 {
        return Err(MCSError::InvalidParams(
            "None of the 'positive' entities have embeddings".into(),
        ));
    }
    for q in query.iter_mut() {
        *q /= pos_count as f64;
    }

    let mut neg_count = 0usize;
    let mut neg = vec![0.0f64; dims];
    for n in &negative {
        if let Some((id, emb, _)) = vs.get_embedding_by_name(n)? {
            if emb.len() != dims {
                continue;
            }
            for (q, &e) in neg.iter_mut().zip(&emb) {
                *q += f64::from(e);
            }
            exclude.insert(id);
            neg_count += 1;
        }
    }
    if neg_count > 0 {
        for (q, n) in query.iter_mut().zip(&neg) {
            *q -= n / neg_count as f64;
        }
    }

    let qf = to_f32(&query);
    let rows = vs.search_resolved(&qf, top_k, entity_type, &exclude)?;
    let named: Vec<(String, String, f64)> = rows
        .into_iter()
        .map(|(_, n, t, d)| (n, t, f64::from(d)))
        .collect();
    Ok(build_content_response(&build_named_results(&named)))
}

/// Maximal Marginal Relevance search: diversified semantic retrieval.
/// `{ embedding, topK?, fetchK?, lambda?, entityType? }`. `lambda` in `[0,1]`
/// trades relevance (1.0) against diversity (0.0). Reduces near-duplicate hits —
/// a common RAG context-selection step. The reported `score` is the MMR score.
pub fn handle_vector_mmr_search(
    vs: &VectorStore,
    _kg: &GraphHandle,
    args: Option<&Value>,
) -> Result<String> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let embedding = parse_embedding(
        params
            .get("embedding")
            .ok_or_else(|| MCSError::InvalidParams("Missing 'embedding' parameter".into()))?,
    )?;
    let top_k = opt_usize(params, "topK", DEFAULT_TOP_K)?.clamp(1, MAX_TOP_K);
    let fetch_k = opt_usize(params, "fetchK", (top_k * 4).max(20))?.clamp(top_k, MAX_TOP_K);
    let lambda = opt_f64(params, "lambda", 0.5)?.clamp(0.0, 1.0);
    let entity_type = params
        .get("entityType")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());

    let query = to_f32(&embedding);

    // Fetch a candidate pool, then greedily select for MMR.
    let pool = vs.search_embeddings(&query, fetch_k)?;
    let mut cands: Vec<MmrCand> = Vec::with_capacity(pool.len());
    for (id, _dist) in pool {
        let (name, etype) = vs.resolve_name_type(id);
        if name.is_empty() {
            continue;
        }
        if let Some(ft) = entity_type
            && etype != ft
        {
            continue;
        }
        if let Some(emb) = vs.get_embedding_by_id(id)? {
            let rel = cosine_sim(&query, &emb);
            cands.push(MmrCand { name, etype, emb, rel });
        }
    }

    let mut selected: Vec<MmrCand> = Vec::with_capacity(top_k.min(cands.len()));
    let mut scores: Vec<f64> = Vec::with_capacity(top_k.min(cands.len()));
    while selected.len() < top_k && !cands.is_empty() {
        let mut best_idx = 0usize;
        let mut best_mmr = f64::NEG_INFINITY;
        for (i, c) in cands.iter().enumerate() {
            let max_sim = selected
                .iter()
                .map(|s| cosine_sim(&c.emb, &s.emb))
                .fold(0.0f64, f64::max);
            let mmr = lambda * c.rel - (1.0 - lambda) * max_sim;
            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = i;
            }
        }
        let chosen = cands.swap_remove(best_idx);
        selected.push(chosen);
        scores.push(best_mmr);
    }

    let named: Vec<(String, String, f64)> = selected
        .into_iter()
        .zip(scores)
        .map(|(c, s)| (c.name, c.etype, s))
        .collect();
    Ok(build_content_response(&build_named_results(&named)))
}

struct MmrCand {
    name: String,
    etype: String,
    emb: Vec<f32>,
    rel: f64,
}

/// Rebuild/retrain the ANN index (IVF k-means; HNSW is a no-op). `{}`.
pub fn handle_vector_reindex(
    vs: &VectorStore,
    _kg: &GraphHandle,
    _args: Option<&Value>,
) -> Result<Value> {
    vs.reindex()?;
    let kind = match vs.index_kind() {
        crate::vector_store::IndexKind::Hnsw => "hnsw",
        crate::vector_store::IndexKind::Ivf => "ivf",
        crate::vector_store::IndexKind::TurboQuant => "turboquant",
    };
    let text = serde_json::to_string(&json!({
        "reindexed": true,
        "indexKind": kind,
        "embeddingCount": vs.count(),
    }))
    .map_err(MCSError::JsonError)?;
    Ok(text_content(&text))
}

fn collect_names(params: &Value, key: &str) -> Result<Vec<String>> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(arr)) => {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                let s = v.as_str().ok_or_else(|| {
                    MCSError::InvalidParams(format!("'{key}' must be an array of strings"))
                })?;
                out.push(s.to_string());
            }
            Ok(out)
        }
        Some(_) => Err(MCSError::InvalidParams(format!(
            "'{key}' must be an array of strings"
        ))),
    }
}
