use serde_json::{Value, json};

use crate::errors::{MCSError, Result};
use crate::kg::{push_json_str, GraphHandle, ReadSnapshot};

const MAX_NAME_BYTES: usize = 1024;
const MAX_OBSERVATION_BYTES: usize = 65536;
const MAX_ENTITIES_PER_REQUEST: usize = 1000;
const MAX_RELATIONS_PER_REQUEST: usize = 1000;
const MAX_OBSERVATIONS_PER_ENTITY: usize = 1000;
const MAX_NEIGHBOR_DEPTH: usize = 16;

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

fn validate_observation(content: &str) -> Result<()> {
    if content.len() > MAX_OBSERVATION_BYTES {
        return Err(MCSError::InvalidParams(format!(
            "Observation too long (max {MAX_OBSERVATION_BYTES} bytes)"
        )));
    }
    Ok(())
}

macro_rules! text_content {
    ($text:expr) => {
        json!({
            "content": [{
                "type": "text",
                "text": $text
            }]
        })
    };
}

fn lock_graph_read(kg: &GraphHandle) -> ReadSnapshot {
    kg.read()
}

fn lock_graph_write(kg: &GraphHandle) -> crate::kg::WriteGuard<'_> {
    kg.write()
}

fn build_content_response(inner_json: &str) -> String {
    let mut out = String::with_capacity(64 + inner_json.len() + (inner_json.len() / 8));
    out.push_str(r#"{"content":[{"type":"text","text":"#);
    push_json_str(&mut out, inner_json);
    out.push_str(r#"}]}"#);
    out
}

pub fn handle_read_graph(kg: &GraphHandle, args: Option<&Value>) -> Result<String> {
    let params = args.unwrap_or(&Value::Null);
    let entity_type = params.get("entityType").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
    let offset = opt_usize(params, "offset", 0)?;
    let limit = opt_usize(params, "limit", usize::MAX)?;

    if entity_type.is_none() && offset == 0 && limit == usize::MAX {
        // Fast path: use cached or raw-JSON serialization, bypassing Value tree.
        Ok(build_content_response(&kg.read_graph_cached()))
    } else {
        // Filtered/paginated read: use the borrowing view and serialize once.
        let graph = lock_graph_read(kg);
        let entities: Vec<crate::types::Entity> = graph
            .entity_slots
            .iter()
            .filter_map(|s| s.as_ref().filter(|e| e.is_live()))
            .filter(|e| entity_type.is_none_or(|t| graph.interner.lookup(e.entity_type) == t))
            .skip(offset)
            .take(limit)
            .map(|e| graph.entity_to_output(e))
            .collect();
        // No entity type filter on relations — just filter by matched names.
        let mut matched: std::collections::HashSet<_> = entities.iter()
            .filter_map(|e| graph.interner.get_optional(&e.name))
            .collect();
        if matched.is_empty() && entity_type.is_none() {
            // Full read without type filter but with pagination — need all names.
            matched = graph.entity_slots.iter()
                .filter_map(|s| s.as_ref().filter(|e| e.is_live()).map(|e| e.name))
                .collect();
        }
        let relations: Vec<crate::types::Relation> = graph
            .relations
            .iter()
            .filter(|r| matched.contains(&r.from) && matched.contains(&r.to))
            .map(|r| graph.relation_to_output(r))
            .collect();
        let text = serde_json::to_string(&crate::types::KnowledgeGraphOut { entities, relations })
            .map_err(MCSError::JsonError)?;
        Ok(build_content_response(&text))
    }
}

pub fn handle_create_entities(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let entities_val = params
        .get("entities")
        .ok_or_else(|| MCSError::InvalidParams("Missing 'entities' parameter".into()))?;

    let input_entities: Vec<crate::types::Entity> = serde_json::from_value(entities_val.clone())
        .map_err(|e| MCSError::InvalidParams(format!("Invalid entity: {e}")))?;

    if input_entities.len() > MAX_ENTITIES_PER_REQUEST {
        return Err(MCSError::InvalidParams(format!(
            "Too many entities (max {MAX_ENTITIES_PER_REQUEST})"
        )));
    }
    for entity in &input_entities {
        validate_name(&entity.name)?;
        validate_name(&entity.entity_type)?;
        if entity.observations.len() > MAX_OBSERVATIONS_PER_ENTITY {
            return Err(MCSError::InvalidParams(format!(
                "Too many observations per entity (max {MAX_OBSERVATIONS_PER_ENTITY})"
            )));
        }
        for obs in &entity.observations {
            validate_observation(obs)?;
        }
    }

    let result;
    {
        let mut graph = lock_graph_write(kg);
        result = graph.create_entities(&input_entities)?;
    }
    let text = serde_json::to_string(&result).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_create_relations(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let relations_val = params
        .get("relations")
        .ok_or_else(|| MCSError::InvalidParams("Missing 'relations' parameter".into()))?;

    let input_relations: Vec<crate::types::Relation> = serde_json::from_value(relations_val.clone())
        .map_err(|e| MCSError::InvalidParams(format!("Invalid relation: {e}")))?;

    if input_relations.len() > MAX_RELATIONS_PER_REQUEST {
        return Err(MCSError::InvalidParams(format!(
            "Too many relations (max {MAX_RELATIONS_PER_REQUEST})"
        )));
    }
    for rel in &input_relations {
        validate_name(&rel.from)?;
        validate_name(&rel.to)?;
        validate_name(&rel.relation_type)?;
    }

    let mut graph = lock_graph_write(kg);
    let result = graph.create_relations(&input_relations)?;
    let text = serde_json::to_string(&result).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_add_observations(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let observations_val = params
        .get("observations")
        .ok_or_else(|| MCSError::InvalidParams("Missing 'observations' parameter".into()))?;

    let observations: Vec<Value> = serde_json::from_value(observations_val.clone())
        .map_err(|e| MCSError::InvalidParams(format!("Invalid observations: {e}")))?;

    let mut results = Vec::new();

    let mut graph = lock_graph_write(kg);

    for obs in &observations {
        let entity_name = obs
            .get("entityName")
            .and_then(|v| v.as_str())
            .ok_or_else(|| MCSError::InvalidParams("Missing 'entityName' in observation".into()))?;

        let contents: Vec<String> = obs
            .get("contents")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        validate_name(entity_name)?;
        if contents.len() > MAX_OBSERVATIONS_PER_ENTITY {
            return Err(MCSError::InvalidParams(format!(
                "Too many observations per entity (max {MAX_OBSERVATIONS_PER_ENTITY})"
            )));
        }
        for content in &contents {
            validate_observation(content)?;
        }

        let added = graph.add_observations(entity_name, &contents)?;

        results.push(json!({
            "entityName": entity_name,
            "addedObservations": added
        }));
    }
    let text = serde_json::to_string(&json!({"results": results})).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_delete_entities(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let entity_names: Vec<String> = params
        .get("entityNames")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .ok_or_else(|| MCSError::InvalidParams("Missing or invalid 'entityNames' parameter".into()))?;

    let mut graph = lock_graph_write(kg);
    graph.delete_entities(&entity_names)?;

    Ok(text_content!("Entities deleted successfully"))
}

pub fn handle_delete_observations(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let deletions = params
        .get("deletions")
        .and_then(|v| v.as_array())
        .ok_or_else(|| MCSError::InvalidParams("Missing or invalid 'deletions' parameter".into()))?;

    let mut graph = lock_graph_write(kg);

    for deletion in deletions {
        let entity_name = deletion
            .get("entityName")
            .and_then(|v| v.as_str())
            .ok_or_else(|| MCSError::InvalidParams("Missing 'entityName' in deletion".into()))?;
        let observations: Vec<String> = deletion
            .get("observations")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        graph.delete_observations(entity_name, &observations)?;
    }

    Ok(text_content!("Observations deleted successfully"))
}

pub fn handle_delete_relations(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let relations_val = params
        .get("relations")
        .ok_or_else(|| MCSError::InvalidParams("Missing 'relations' parameter".into()))?;

    let input_relations: Vec<crate::types::Relation> = serde_json::from_value(relations_val.clone())
        .map_err(|e| MCSError::InvalidParams(format!("Invalid relation: {e}")))?;

    let mut graph = lock_graph_write(kg);
    graph.delete_relations(&input_relations)?;

    Ok(text_content!("Relations deleted successfully"))
}

pub fn handle_search_nodes(kg: &GraphHandle, args: Option<&Value>) -> Result<String> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let query = params
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'query' parameter".into()))?;

    let entity_type = params.get("entityType").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
    let offset = opt_usize(params, "offset", 0)?;
    let limit = opt_usize(params, "limit", usize::MAX)?;

    let graph = lock_graph_read(kg);
    let matching = graph.search_entities(query)?;
    let filtered: Vec<crate::types::Entity> = matching
        .into_iter()
        .filter(|e| entity_type.is_none_or(|t| e.entity_type == t))
        .skip(offset)
        .take(limit)
        .collect();
    let text = serde_json::to_string(&filtered).map_err(MCSError::JsonError)?;
    Ok(build_content_response(&text))
}

pub fn handle_open_nodes(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let names: Vec<String> = params
        .get("names")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .ok_or_else(|| MCSError::InvalidParams("Missing or invalid 'names' parameter".into()))?;

    let graph = lock_graph_read(kg);
    let out = graph.open_nodes(&names);
    let text = serde_json::to_string(&out).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_get_entity(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'name' parameter".into()))?;

    let graph = lock_graph_read(kg);
    match graph.get_entity(name) {
        Some(entity) => {
            let text = serde_json::to_string(&entity).map_err(MCSError::JsonError)?;
            Ok(text_content!(text))
        }
        None => Err(MCSError::InvalidParams(format!("Entity '{name}' not found"))),
    }
}

pub fn handle_graph_stats(kg: &GraphHandle) -> Result<Value> {
    let graph = lock_graph_read(kg);
    let stats = graph.graph_stats();
    let text = serde_json::to_string(&stats).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_search_relations(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.unwrap_or(&serde_json::Value::Null);

    let from = params.get("from").and_then(|v| v.as_str());
    let to = params.get("to").and_then(|v| v.as_str());
    let rtype = params.get("relationType").and_then(|v| v.as_str());

    let graph = lock_graph_read(kg);
    let results = graph.search_relations(from, to, rtype);
    let text = serde_json::to_string(&results).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_find_path(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let from = params
        .get("from")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'from' parameter".into()))?;
    let to = params
        .get("to")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'to' parameter".into()))?;

    let graph = lock_graph_read(kg);
    let path = graph.find_path(from, to)?;
    let text = serde_json::to_string(&path).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_compact(kg: &GraphHandle) -> Result<Value> {
    let mut graph = lock_graph_write(kg);
    graph.compact()?;
    Ok(text_content!("Log compacted successfully"))
}

fn opt_usize(params: &Value, key: &str, default: usize) -> Result<usize> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_u64()
            .map(|n| n as usize)
            .ok_or_else(|| MCSError::InvalidParams(format!("'{key}' must be a non-negative integer"))),
    }
}

pub fn handle_get_neighbors(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'name' parameter".into()))?;
    validate_name(name)?;

    let direction = crate::kg::Direction::parse(params.get("direction").and_then(|v| v.as_str()));
    let rtype = params.get("relationType").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
    let depth = opt_usize(params, "depth", 1)?;
    if depth > MAX_NEIGHBOR_DEPTH {
        return Err(MCSError::InvalidParams(format!(
            "depth too large (max {MAX_NEIGHBOR_DEPTH})"
        )));
    }

    let graph = lock_graph_read(kg);
    let result = graph.neighbors(name, direction, rtype, depth as u32)?;
    let text = serde_json::to_string(&result).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_describe_entity(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'name' parameter".into()))?;
    validate_name(name)?;

    let graph = lock_graph_read(kg);
    let result = graph.describe_entity(name)?;
    let text = serde_json::to_string(&result).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_list_entity_types(kg: &GraphHandle) -> Result<Value> {
    let graph = lock_graph_read(kg);
    let counts = graph.entity_type_counts();
    let arr: Vec<Value> = counts
        .into_iter()
        .map(|(t, c)| json!({ "type": t, "count": c }))
        .collect();
    let text = serde_json::to_string(&arr).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_list_relation_types(kg: &GraphHandle) -> Result<Value> {
    let graph = lock_graph_read(kg);
    let counts = graph.relation_type_counts();
    let arr: Vec<Value> = counts
        .into_iter()
        .map(|(t, c)| json!({ "type": t, "count": c }))
        .collect();
    let text = serde_json::to_string(&arr).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_upsert_entities(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let entities_val = params
        .get("entities")
        .ok_or_else(|| MCSError::InvalidParams("Missing 'entities' parameter".into()))?;

    let input_entities: Vec<crate::types::Entity> = serde_json::from_value(entities_val.clone())
        .map_err(|e| MCSError::InvalidParams(format!("Invalid entity: {e}")))?;

    if input_entities.len() > MAX_ENTITIES_PER_REQUEST {
        return Err(MCSError::InvalidParams(format!(
            "Too many entities (max {MAX_ENTITIES_PER_REQUEST})"
        )));
    }
    for entity in &input_entities {
        validate_name(&entity.name)?;
        validate_name(&entity.entity_type)?;
        if entity.observations.len() > MAX_OBSERVATIONS_PER_ENTITY {
            return Err(MCSError::InvalidParams(format!(
                "Too many observations per entity (max {MAX_OBSERVATIONS_PER_ENTITY})"
            )));
        }
        for obs in &entity.observations {
            validate_observation(obs)?;
        }
    }

    let mut graph = lock_graph_write(kg);
    let results = graph.upsert_entities(&input_entities)?;
    let text = serde_json::to_string(&json!({ "results": results })).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_merge_entities(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let source = params
        .get("source")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'source' parameter".into()))?;
    let target = params
        .get("target")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'target' parameter".into()))?;
    validate_name(source)?;
    validate_name(target)?;

    let mut graph = lock_graph_write(kg);
    let result = graph.merge_entities(source, target)?;
    let text = serde_json::to_string(&result).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_extract_subgraph(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let names: Vec<String> = params
        .get("names")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .ok_or_else(|| MCSError::InvalidParams("Missing or invalid 'names' parameter".into()))?;
    let depth = opt_usize(params, "depth", 1)? as u32;
    if depth > MAX_NEIGHBOR_DEPTH as u32 {
        return Err(MCSError::InvalidParams(format!(
            "depth too large (max {MAX_NEIGHBOR_DEPTH})"
        )));
    }

    let graph = lock_graph_read(kg);
    let result = graph.extract_subgraph(&names, depth)?;
    let text = serde_json::to_string(&result).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_batch_get_entities(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let names: Vec<String> = params
        .get("names")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .ok_or_else(|| MCSError::InvalidParams("Missing or invalid 'names' parameter".into()))?;

    let graph = lock_graph_read(kg);
    let results = graph.batch_get_entities(&names);
    let arr: Vec<Value> = results
        .into_iter()
        .map(|opt| match opt {
            Some(entity) => serde_json::to_value(entity).unwrap_or(Value::Null),
            None => Value::Null,
        })
        .collect();
    let text = serde_json::to_string(&arr).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_find_all_paths(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let from = params
        .get("from")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'from' parameter".into()))?;
    let to = params
        .get("to")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'to' parameter".into()))?;
    let max_depth = opt_usize(params, "maxDepth", 6)?;
    let max_paths = opt_usize(params, "maxPaths", 50)?;

    let graph = lock_graph_read(kg);
    let paths = graph.find_all_paths(from, to, max_depth, max_paths)?;
    let text = serde_json::to_string(&paths).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

pub fn handle_export_graph(kg: &GraphHandle, args: Option<&Value>) -> Result<Value> {
    let format = args
        .and_then(|p| p.get("format"))
        .and_then(|v| v.as_str())
        .unwrap_or("json");

    let graph = lock_graph_read(kg);
    let text = graph.export(format)?;
    Ok(text_content!(text))
}
