use serde_json::{Value, json};

use crate::errors::{MCSError, Result};
use crate::kg::{GraphHandle, push_json_str};
use crate::vector_store::{EntityId, VectorStore, with_scratch};

type HybridResult = Vec<(String, String, f64, f64, f64)>;

use rusqlite::params;

const MAX_EMBEDDING_DIMS: usize = 4096;
const MAX_TOP_K: usize = 100;
const DEFAULT_TOP_K: usize = 10;
const MAX_NAME_BYTES: usize = 1024;

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

    let mut score_map: std::collections::HashMap<EntityId, AggScore> =
        std::collections::HashMap::with_capacity(vec_matches.len() + text_matches.len());

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
    let text = serde_json::to_string(&json!({
        "embeddingCount": vs.count(),
        "dims": vs.dims(),
        "petgraphNodes": vs.graph_node_count(),
        "petgraphEdges": vs.graph_edge_count(),
    }))
    .map_err(MCSError::JsonError)?;
    Ok(text_content(&text))
}
