//! MCP tool handlers for the tree-sitter code knowledge graph.
//!
//! These map parsed code symbols (see [`crate::code`]) onto a knowledge graph
//! so the regular search/traversal primitives work on code, and expose
//! code-focused tools: `code_index`, `code_watch`, `code_outline`,
//! `code_search`, `code_get_symbol`.
//!
//! Symbols are stored as entities named `{relpath}::{symbol}` with type
//! `code:<kind>`; metadata (file, line range, signature, doc) lives in
//! observations. Edges: `defines` (file→symbol), `calls`/`references`
//! (caller→callee, resolved only when the callee name is unambiguous).
//!
//! Multiple independent projects are supported by **physical partitioning**:
//! each `code_index`/`code_watch` call takes an optional `project` identifier
//! (default `"default"`) that selects a dedicated SQLite database, opened via
//! [`crate::code_registry`]. Projects therefore never collide and are fully
//! isolated from the main memory graph and from the knowledge-graph tools.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::code::{self, Def, MAX_SYMBOLS_PER_FILE};
use crate::errors::{MCSError, Result};
use crate::kg::GraphHandle;
use crate::types::{Entity, Relation};

/// Cap on files processed in a single `code_index` call.
const MAX_INDEX_FILES: usize = 100_000;
/// Total symbols across all files (prevents OOM on huge repos).
const MAX_TOTAL_SYMBOLS: usize = 5_000_000;
/// Batch size for graph writes (keeps each write transaction bounded).
const WRITE_BATCH: usize = 1_000;
/// Default / max result rows for `code_search`.
const DEFAULT_SEARCH_LIMIT: usize = 20;
const MAX_SEARCH_LIMIT: usize = 500;
/// Cap on callers/callees returned by `code_get_symbol`.
const MAX_EDGES_RETURNED: usize = 500;

macro_rules! text_content {
    ($text:expr) => {
        json!({ "content": [{ "type": "text", "text": $text }] })
    };
}

fn to_json(v: &impl serde::Serialize) -> Result<Value> {
    let text = serde_json::to_string(v).map_err(MCSError::JsonError)?;
    Ok(text_content!(text))
}

/// Read + validate the optional `project` argument, defaulting to
/// [`crate::code_registry::DEFAULT_PROJECT`]. Each project maps to its own DB.
fn project_of(params: &Value) -> Result<String> {
    let p = params
        .get("project")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(crate::code_registry::DEFAULT_PROJECT);
    crate::code_registry::validate_project(p)?;
    Ok(p.to_string())
}

/// Repo-relative, forward-slash path used as the `code:file` entity name.
fn rel_path(p: &Path, base: &Path) -> String {
    let r = if p.is_absolute() {
        p.strip_prefix(base).unwrap_or(p)
    } else {
        p
    };
    r.to_string_lossy().replace('\\', "/")
}

/// Read a single-valued `key: value` observation off an entity.
fn obs_val<'a>(entity: &'a Entity, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}: ");
    entity
        .observations
        .iter()
        .find_map(|o| o.strip_prefix(&prefix))
}

/// Strip the `code:` prefix from an entity type for display.
fn kind_of(entity: &Entity) -> &str {
    entity.entity_type.strip_prefix("code:").unwrap_or(&entity.entity_type)
}

fn is_code_entity(entity: &Entity) -> bool {
    entity.entity_type.starts_with("code:")
}

/// Compact, location-focused view of a code symbol entity.
fn symbol_row(entity: &Entity) -> Value {
    json!({
        "name": entity.name,
        "kind": kind_of(entity),
        "file": obs_val(entity, "file"),
        "lines": obs_val(entity, "lines"),
        "lang": obs_val(entity, "lang"),
        "signature": obs_val(entity, "signature"),
        "doc": obs_val(entity, "doc"),
        "snippet": obs_val(entity, "snippet"),
    })
}

// ---------------------------------------------------------------------------
// code_index
// ---------------------------------------------------------------------------

/// Parsed symbols for one file, with qualified names already assigned.
struct FileWork {
    rel: String,
    lang: &'static str,
    hash: String,
    /// Whether a `code:file` entity already existed (drives purge-skip).
    existed: bool,
    named: Vec<(Def, String)>,
    refs: Vec<code::Ref>,
}

/// Outcome of processing a single path during the parallel parse phase.
enum Outcome {
    Indexed(Box<FileWork>),
    Skipped,
    Failed,
    Unsupported,
}

/// Read + hash + (incrementally) parse one file. CPU-bound and independent per
/// file, so this runs on the parse thread pool. Reads use the graph's
/// concurrent read pool; no writes happen here.
fn parse_one(kg: &GraphHandle, path: &Path, base: &Path,
             force: bool, want_snippet: bool, total_symbols: &AtomicUsize) -> Outcome {
    let Some(lang) = code::detect(path) else {
        return Outcome::Unsupported;
    };
    let rel = rel_path(path, base);
    let Ok(bytes) = std::fs::read(path) else {
        return Outcome::Failed;
    };
    let hash = code::hash_bytes(&bytes);

    // Project isolation is physical (one DB per project), so the file entity is
    // just the repo-relative path — no project prefix needed.
    let existing = kg.get_entity(&rel).ok().flatten();
    let existed = existing.is_some();
    // Incremental: skip unchanged files (matching stored hash).
    if !force
        && let Some(e) = &existing
        && obs_val(e, "hash") == Some(hash.as_str())
    {
        return Outcome::Skipped;
    }

    let parsed = code::parse_source_opts(lang, &bytes, want_snippet);
    let mut seen: HashSet<String> = HashSet::new();
    let mut named: Vec<(Def, String)> = Vec::with_capacity(parsed.defs.len());
    for d in parsed.defs.into_iter().take(MAX_SYMBOLS_PER_FILE) {
        let mut q = format!("{rel}::{}", d.name);
        if !seen.insert(q.clone()) {
            q = format!("{q}::L{}", d.line_start);
            seen.insert(q.clone());
        }
        named.push((d, q));
    }

    // Accumulate towards the total symbol cap.
    let prev = total_symbols.fetch_add(named.len(), Ordering::Relaxed);
    if prev + named.len() > MAX_TOTAL_SYMBOLS {
        // Undo — we overshot. Non-atomic for correctness: the caller's cap check
        // stops new files from being accepted; any surplus is simply ignored in
        // the merge phase below.
        return Outcome::Skipped;
    }

    Outcome::Indexed(Box::new(FileWork {
        rel,
        lang: lang.name(),
        hash,
        existed,
        named,
        refs: parsed.refs,
    }))
}

pub fn handle_code_index(args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let path = params
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'path' parameter".into()))?;
    let project = project_of(params)?;
    let kg = crate::code_registry::resolve(&project)?;
    let kg = kg.as_ref();
    let force = params.get("force").and_then(|v| v.as_bool()).unwrap_or(false);
    let snippets = params.get("snippets").and_then(|v| v.as_bool()).unwrap_or(false);

    let root = Path::new(path);
    if !root.exists() {
        return Err(MCSError::InvalidParams(format!("Path not found: {path}")));
    }
    // Canonicalize so entity names are stable regardless of how the path is
    // spelled (symlinks, `.`, `..`) — critical for matching the symlink-resolved
    // paths the watcher receives from the OS. Falls back to the raw path.
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let base = canonical_base();
    let files = code::walk(&root, code::MAX_FILE_BYTES);
    index_paths(kg, files, &base, force, snippets)
}

/// The canonicalized current working directory, used as the base for
/// repo-relative entity names. Shared by the indexer and the watcher so both
/// derive identical names.
pub(crate) fn canonical_base() -> std::path::PathBuf {
    std::env::current_dir()
        .and_then(|d| d.canonicalize())
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
}

/// Repo-relative entity name for a path under `base` (matches [`parse_one`]).
/// Exposed for the watcher to purge symbols of deleted files.
pub(crate) fn file_entity_name(path: &Path, base: &Path) -> String {
    rel_path(path, base)
}

/// Map a caller-supplied file path to its stored entity name. Relative paths
/// are assumed already repo-relative; absolute paths are canonicalized and
/// based the same way [`handle_code_index`] stores them.
fn lookup_file_name(file: &str) -> String {
    let p = Path::new(file);
    if p.is_absolute() {
        let c = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
        rel_path(&c, &canonical_base())
    } else {
        file.to_string()
    }
}

/// Parse + write a known set of `files` into `kg`. Shared by the `code_index`
/// tool (after walking a path) and the watcher (a debounced batch of changed
/// files). `base` anchors repo-relative entity names; the same `base` must be
/// used across calls for a project so re-indexing updates rather than
/// duplicates. Batching a whole change set through one call keeps the parse
/// pool and write transactions amortized instead of per-file.
pub(crate) fn index_paths(
    kg: &GraphHandle,
    mut files: Vec<std::path::PathBuf>,
    base: &Path,
    force: bool,
    snippets: bool,
) -> Result<Value> {
    files.truncate(MAX_INDEX_FILES);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Parse phase (parallel): read + hash + parse each file across the CPU
    // cores. Files are independent and parsing is the dominant cost; reads use
    // the concurrent read pool. The single-writer graph mutations stay serial
    // in the merge phase below.
    let n = files.len();
    let n_threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(4)
        .min(n.max(1));
    let next = AtomicUsize::new(0);
    let total_symbols = AtomicUsize::new(0);
    let buckets: Vec<Vec<Outcome>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        // Pre-check total symbol cap to avoid unnecessary work.
                        if total_symbols.load(Ordering::Relaxed) >= MAX_TOTAL_SYMBOLS {
                            continue;
                        }
                        local.push(parse_one(kg, &files[i], base, force, snippets, &total_symbols));
                    }
                    local
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Merge phase (serial): tally outcomes and build the global symbol index
    // (bare name -> qualified names) used to resolve unambiguous call edges.
    let mut work: Vec<FileWork> = Vec::new();
    let mut def_index: HashMap<String, Vec<String>> = HashMap::new();
    let mut files_indexed = 0usize;
    let mut files_skipped = 0usize;
    let mut files_failed = 0usize;
    for outcome in buckets.into_iter().flatten() {
        match outcome {
            Outcome::Indexed(fw) => {
                for (d, q) in &fw.named {
                    def_index.entry(d.name.clone()).or_default().push(q.clone());
                }
                work.push(*fw);
                files_indexed += 1;
            }
            Outcome::Skipped => files_skipped += 1,
            Outcome::Failed => files_failed += 1,
            Outcome::Unsupported => {}
        }
    }

    // Write phase (serial, single writer). Streamed in `WRITE_BATCH` chunks so
    // the transient entity/relation buffers stay bounded regardless of repo
    // size; the parsed `work` is the only large allocation. Entities are written
    // in full *before* any relation, since relations resolve their endpoints by
    // name and would silently drop against a not-yet-written entity.

    // Pass 1: purge changed files and write all entities.
    let mut ebuf: Vec<Entity> = Vec::with_capacity(WRITE_BATCH);
    let mut symbols = 0usize;
    for fw in &work {
        if fw.existed {
            kg.code_purge_file(&fw.rel)?;
        }
        ebuf.push(Entity {
            name: fw.rel.clone(),
            entity_type: "code:file".into(),
            observations: vec![
                format!("lang: {}", fw.lang),
                format!("hash: {}", fw.hash),
                format!("symbols: {}", fw.named.len()),
                format!("indexed_at: {now}"),
            ],
        });
        for (d, q) in &fw.named {
            let mut obs = vec![
                format!("kind: {}", d.kind),
                format!("lang: {}", fw.lang),
                format!("file: {}", fw.rel),
                format!("lines: {}-{}", d.line_start, d.line_end),
                format!("signature: {}", d.signature),
            ];
            if let Some(doc) = &d.doc {
                obs.push(format!("doc: {doc}"));
            }
            if !d.snippet.is_empty() {
                obs.push(format!("snippet: {}", d.snippet));
            }
            ebuf.push(Entity {
                name: q.clone(),
                entity_type: format!("code:{}", d.kind),
                observations: obs,
            });
            symbols += 1;
        }
        if ebuf.len() >= WRITE_BATCH {
            kg.upsert_entities(&ebuf)?;
            ebuf.clear();
        }
    }
    if !ebuf.is_empty() {
        kg.upsert_entities(&ebuf)?;
    }

    // Pass 2: write `defines` edges and unambiguously-resolved call edges.
    let mut rbuf: Vec<Relation> = Vec::with_capacity(WRITE_BATCH);
    let mut rel_seen: HashSet<(String, String, &'static str)> = HashSet::new();
    let mut relation_count = 0usize;
    for fw in &work {
        let file_entity = &fw.rel;
        for (_, q) in &fw.named {
            rbuf.push(Relation {
                from: file_entity.clone(),
                to: q.clone(),
                relation_type: "defines".into(),
            });
            relation_count += 1;
        }
        for r in &fw.refs {
            let Some(targets) = def_index.get(&r.name) else { continue };
            if targets.len() != 1 {
                continue; // ambiguous or unresolved — drop (no false edges)
            }
            let callee = &targets[0];
            let caller = enclosing(&fw.named, r.line)
                .map(|q| q.to_string())
                .unwrap_or_else(|| file_entity.clone());
            if &caller == callee {
                continue;
            }
            let rtype: &'static str = if r.kind == "call" { "calls" } else { "references" };
            if !rel_seen.insert((caller.clone(), callee.clone(), rtype)) {
                continue;
            }
            rbuf.push(Relation {
                from: caller,
                to: callee.clone(),
                relation_type: rtype.into(),
            });
            relation_count += 1;
        }
        if rbuf.len() >= WRITE_BATCH {
            kg.create_relations(&rbuf)?;
            rbuf.clear();
        }
    }
    if !rbuf.is_empty() {
        kg.create_relations(&rbuf)?;
    }

    to_json(&json!({
        "files_indexed": files_indexed,
        "files_skipped": files_skipped,
        "files_failed": files_failed,
        "symbols": symbols,
        "relations": relation_count,
    }))
}

/// Smallest-span definition whose line range encloses `line`, if any.
fn enclosing(named: &[(Def, String)], line: usize) -> Option<&str> {
    named
        .iter()
        .filter(|(d, _)| d.line_start <= line && line <= d.line_end)
        .min_by_key(|(d, _)| d.line_end - d.line_start)
        .map(|(_, q)| q.as_str())
}

// ---------------------------------------------------------------------------
// code_outline
// ---------------------------------------------------------------------------

pub fn handle_code_outline(args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let file = params
        .get("file")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'file' parameter".into()))?;
    let file = file.replace('\\', "/");
    let project = project_of(params)?;
    let kg = crate::code_registry::resolve(&project)?;
    let kg = kg.as_ref();

    // Map the caller's path to the stored entity name. A relative path is
    // already repo-relative (matches the stored name); an absolute path is
    // canonicalized + based exactly as the indexer does.
    let lookup = lookup_file_name(&file);
    let defines = kg.search_relations(Some(&lookup), None, Some("defines"), Some(MAX_SYMBOLS_PER_FILE));
    let names: Vec<String> = defines.into_iter().map(|r| r.to).collect();
    if names.is_empty() {
        return to_json(&json!({
            "file": file,
            "symbols": [],
            "note": "no symbols indexed for this file; run code_index first",
        }));
    }
    let mut rows: Vec<Value> = kg
        .batch_get_entities(&names)
        .into_iter()
        .flatten()
        .map(|e| symbol_row(&e))
        .collect();
    // Order by starting line for a readable outline.
    rows.sort_by_key(|r| {
        r.get("lines")
            .and_then(|v| v.as_str())
            .and_then(|s| s.split('-').next())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
    });

    to_json(&json!({ "file": file, "symbols": rows }))
}

// ---------------------------------------------------------------------------
// code_search
// ---------------------------------------------------------------------------

pub fn handle_code_search(args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let query = params
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'query' parameter".into()))?;
    let kind = params.get("kind").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
    let lang = params.get("lang").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
    let project = project_of(params)?;
    let kg = crate::code_registry::resolve(&project)?;
    let kg = kg.as_ref();
    let limit = params
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(DEFAULT_SEARCH_LIMIT)
        .clamp(1, MAX_SEARCH_LIMIT);

    // Over-fetch then drop file entities / apply kind+lang filters (search has a
    // single-type filter). Project scoping is implicit in the per-project DB.
    let raw = kg.search_nodes_filtered(query, None, 0, limit.saturating_mul(5).min(1000));
    let rows: Vec<Value> = raw
        .into_iter()
        .filter(|e| e.entity_type != "code:file")
        .filter(|e| kind.is_none_or(|k| kind_of(e) == k))
        .filter(|e| lang.is_none_or(|l| obs_val(e, "lang") == Some(l)))
        .take(limit)
        .map(|e| symbol_row(&e))
        .collect();

    to_json(&json!({ "results": rows }))
}

// ---------------------------------------------------------------------------
// code_get_symbol
// ---------------------------------------------------------------------------

pub fn handle_code_get_symbol(args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'name' parameter".into()))?;
    let project = project_of(params)?;
    let kg = crate::code_registry::resolve(&project)?;
    let kg = kg.as_ref();

    // Resolve within the project DB: exact (fully-qualified) name first, else
    // fuzzy by bare name suffix.
    let mut matches: Vec<Entity> = Vec::new();
    if let Ok(Some(e)) = kg.get_entity(name)
        && is_code_entity(&e)
    {
        matches.push(e);
    }
    if matches.is_empty() {
        let suffix = format!("::{name}");
        matches = kg
            .search_nodes_filtered(name, None, 0, 200)
            .into_iter()
            .filter(is_code_entity)
            .filter(|e| e.name.ends_with(&suffix))
            .take(10)
            .collect();
    }
    if matches.is_empty() {
        return Err(MCSError::InvalidParams(format!(
            "No code symbol matching '{name}' (run code_index first?)"
        )));
    }

    let edge_types = ["calls", "references"];
    let results: Vec<Value> = matches
        .iter()
        .map(|e| {
            let mut callers: Vec<String> = Vec::new();
            let mut callees: Vec<String> = Vec::new();
            for t in edge_types {
                for r in kg.search_relations(None, Some(&e.name), Some(t), Some(MAX_EDGES_RETURNED)) {
                    callers.push(r.from);
                }
                for r in kg.search_relations(Some(&e.name), None, Some(t), Some(MAX_EDGES_RETURNED)) {
                    callees.push(r.to);
                }
            }
            callers.truncate(MAX_EDGES_RETURNED);
            callees.truncate(MAX_EDGES_RETURNED);
            let mut row = symbol_row(e);
            row["callers"] = json!(callers);
            row["callees"] = json!(callees);
            row
        })
        .collect();

    if results.len() == 1 {
        to_json(&results.into_iter().next().unwrap())
    } else {
        to_json(&json!({ "matches": results }))
    }
}

// ---------------------------------------------------------------------------
// code_embed / code_semantic_search (HNSW ANN over code symbols)
// ---------------------------------------------------------------------------

/// Cap on items in a single `code_embed` call.
const MAX_EMBED_ITEMS: usize = 1_000;

/// Parse a JSON array of numbers into an `f32` embedding vector.
fn parse_embedding_f32(val: &Value) -> Result<Vec<f32>> {
    let arr = val
        .as_array()
        .ok_or_else(|| MCSError::InvalidParams("'embedding' must be an array of numbers".into()))?;
    if arr.is_empty() {
        return Err(MCSError::InvalidParams("embedding must not be empty".into()));
    }
    arr.iter()
        .map(|v| {
            v.as_f64()
                .map(|n| n as f32)
                .ok_or_else(|| MCSError::InvalidParams("embedding values must be numbers".into()))
        })
        .collect()
}

/// Attach client-supplied embeddings to indexed code symbols so they become
/// searchable by [`handle_code_semantic_search`]. Embeddings live in the same
/// per-project database as the symbols, in an HNSW index keyed by symbol entity.
/// `{ project?, items: [{ name, embedding }] }` — `name` is a symbol name
/// (fully-qualified `file::sym`, as returned by the other `code_*` tools).
pub fn handle_code_embed(args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let project = project_of(params)?;
    let items = params
        .get("items")
        .and_then(|v| v.as_array())
        .ok_or_else(|| MCSError::InvalidParams("'items' must be an array".into()))?;
    if items.len() > MAX_EMBED_ITEMS {
        return Err(MCSError::InvalidParams(format!(
            "Too many items (max {MAX_EMBED_ITEMS})"
        )));
    }

    let vs = crate::code_vec_registry::resolve(&project)?;

    let mut upserted = 0usize;
    let mut errors: Vec<Value> = Vec::new();
    for item in items {
        let name = match item.get("name").and_then(|v| v.as_str()) {
            Some(n) if !n.is_empty() => n,
            _ => {
                errors.push(json!({"name": item.get("name"), "error": "invalid name"}));
                continue;
            }
        };
        let emb = match item.get("embedding").map(parse_embedding_f32) {
            Some(Ok(e)) => e,
            Some(Err(e)) => {
                errors.push(json!({"name": name, "error": e.to_string()}));
                continue;
            }
            None => {
                errors.push(json!({"name": name, "error": "missing embedding"}));
                continue;
            }
        };
        match vs.upsert_embedding(name, &emb, "code") {
            Ok(()) => upserted += 1,
            Err(e) => errors.push(json!({"name": name, "error": e.to_string()})),
        }
    }

    to_json(&json!({
        "project": project,
        "dims": vs.dims(),
        "upserted": upserted,
        "failed": errors.len(),
        "errors": errors,
    }))
}

/// Semantic (vector) search over code symbols using the per-project HNSW index.
/// `{ project?, embedding: [..dims], limit?, kind?, lang? }` — `embedding` is a
/// query vector of the configured dimension (default 768). Returns the nearest
/// symbols as location rows plus a `score` (ANN distance; smaller = closer).
pub fn handle_code_semantic_search(args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let project = project_of(params)?;
    let embedding = parse_embedding_f32(
        params
            .get("embedding")
            .ok_or_else(|| MCSError::InvalidParams("Missing 'embedding' parameter".into()))?,
    )?;
    let kind = params.get("kind").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
    let lang = params.get("lang").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
    let limit = params
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(DEFAULT_SEARCH_LIMIT)
        .clamp(1, MAX_SEARCH_LIMIT);

    let kg = crate::code_registry::resolve(&project)?;
    let vs = crate::code_vec_registry::resolve(&project)?;

    // Over-fetch then resolve names, fetch full entities, and apply kind/lang
    // filters — preserving the ANN distance order (ascending = closer).
    let fetch = limit.saturating_mul(5).min(100);
    let hits = vs.search_embeddings(&embedding, fetch)?;
    let mut names: Vec<String> = Vec::with_capacity(hits.len());
    let mut dist_by_name: std::collections::HashMap<String, f32> =
        std::collections::HashMap::with_capacity(hits.len());
    for (id, dist) in hits {
        if let Some(name) = vs.id_to_name().get(&id).map(|r| r.value().clone()) {
            dist_by_name.entry(name.clone()).or_insert(dist);
            names.push(name);
        }
    }

    let entities = kg.batch_get_entities(&names);
    let rows: Vec<Value> = names
        .iter()
        .zip(entities)
        .filter_map(|(name, e)| e.map(|e| (name, e)))
        .filter(|(_, e)| is_code_entity(e) && e.entity_type != "code:file")
        .filter(|(_, e)| kind.is_none_or(|k| kind_of(e) == k))
        .filter(|(_, e)| lang.is_none_or(|l| obs_val(e, "lang") == Some(l)))
        .take(limit)
        .map(|(name, e)| {
            let mut row = symbol_row(&e);
            row["score"] = json!(dist_by_name.get(name.as_str()).copied().unwrap_or(0.0));
            row
        })
        .collect();

    to_json(&json!({ "results": rows }))
}

/// Start watching a project directory for file changes and re-index on
/// modification. Spawns a background thread that monitors the directory
/// tree with a debounced file-watcher. The initial index runs synchronously
/// before returning.
///
/// The background thread holds the project's `Arc<GraphHandle>` (resolved from
/// [`crate::code_registry`]) for its lifetime, pinning the canonical instance
/// so re-index calls share one entity cache.
pub fn handle_code_watch(args: Option<&Value>) -> Result<Value> {
    let params = args.ok_or_else(|| MCSError::InvalidParams("Missing parameters".into()))?;
    let path = params
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MCSError::InvalidParams("Missing 'path' parameter".into()))?;
    let project = project_of(params)?;
    let force = params.get("force").and_then(|v| v.as_bool()).unwrap_or(false);
    let snippets = params.get("snippets").and_then(|v| v.as_bool()).unwrap_or(false);

    let root = std::path::PathBuf::from(path);
    if !root.exists() {
        return Err(MCSError::InvalidParams(format!("Path not found: {path}")));
    }
    // Watch the canonicalized root so OS events carry the same (symlink-resolved)
    // paths the indexer stored, keeping incremental updates and deletes aligned.
    let root = root.canonicalize().unwrap_or(root);
    let watch_path = root.to_string_lossy().to_string();

    // Initial index immediately (also opens/warms the project DB).
    let index_args = json!({
        "path": &watch_path,
        "project": project,
        "force": force,
        "snippets": snippets,
    });
    let _ = handle_code_index(Some(&index_args))?;

    // Pin the canonical handle and spawn the background watcher.
    let kg_arc = crate::code_registry::resolve(&project)?;
    crate::watcher::spawn_watcher(kg_arc, watch_path.clone(), &project, snippets);

    to_json(&json!({
        "status": "watching",
        "project": project,
        "path": watch_path,
    }))
}
