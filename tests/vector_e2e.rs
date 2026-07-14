use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Once;

static DB_COUNTER: AtomicU32 = AtomicU32::new(0);
static CLEANUP: Once = Once::new();

/// Remove any orphaned test DB files left over from prior runs.
fn cleanup_orphaned_dbs() {
    CLEANUP.call_once(|| {
        if let Ok(entries) = std::fs::read_dir("/tmp") {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str())
                    && name.starts_with("vec_e2e_") && (name.ends_with(".db") || name.ends_with(".db-wal") || name.ends_with(".db-shm")) {
                        let _ = std::fs::remove_file(&path);
                    }
            }
        }
    });
}

struct VecClient {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: std::process::ChildStdout,
    db_path: String,
}

impl Drop for VecClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        for ext in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{}", self.db_path, ext));
        }
    }
}

fn spawn_vec_server() -> VecClient {
    spawn_vec_server_with(&[])
}

/// Spawn a stdio server, appending `extra` CLI args after the defaults.
fn spawn_vec_server_with(extra: &[&str]) -> VecClient {
    cleanup_orphaned_dbs();
    let n = DB_COUNTER.fetch_add(1, Ordering::SeqCst);
    let db_path = format!("/tmp/vec_e2e_{n}.db");
    for ext in ["", "-wal", "-shm"] {
        let p = format!("{db_path}{ext}");
        let _ = std::fs::remove_file(&p);
    }

    let bin = std::env::var("CARGO_BIN_EXE_MCP_MEMORY")
        .unwrap_or_else(|_| "target/debug/mcp-memory".into());

    let mut cmd = Command::new(&bin);
    cmd.arg("-f")
        .arg(&db_path)
        .arg("--transport")
        .arg("stdio")
        .arg("--enable-all")
        .arg("--log-level")
        .arg("error");
    // Default to tiny embeddings unless the test picks its own dimension
    // (clap rejects a flag given twice, so this cannot be an override).
    if !extra.contains(&"--embedding-dims") {
        cmd.arg("--embedding-dims").arg("4");
    }
    cmd.args(extra);
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn mcp-memory");

    VecClient {
        stdin: child.stdin.take().unwrap(),
        stdout: child.stdout.take().unwrap(),
        child,
        db_path,
    }
}

impl VecClient {
    fn send(&mut self, msg: &str) {
        use std::io::Write;
        writeln!(self.stdin, "{msg}").expect("write to stdin");
        self.stdin.flush().expect("flush stdin");
    }

    fn recv(&mut self) -> String {
        use std::io::{BufRead, BufReader};
        let mut buf = String::new();
        BufReader::new(&mut self.stdout)
            .read_line(&mut buf)
            .expect("read from stdout");
        buf.trim().to_string()
    }

    fn call_tool(&mut self, name: &str, args: &serde_json::Value) -> serde_json::Value {
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": args
            },
            "id": 2
        });
        let line = serde_json::to_string(&req).expect("serialize request");
        self.send(&line);
        let resp = self.recv();
        serde_json::from_str(&resp).expect("parse response")
    }

    fn tool_text(&mut self, name: &str, args: &serde_json::Value) -> String {
        let resp = self.call_tool(name, args);
        resp["result"]["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| {
                if let Some(is_err) = resp["result"]["isError"].as_bool()
                    && is_err {
                        panic!(
                            "Tool '{name}' returned isError: {}",
                            resp["result"]["content"][0]["text"]
                                .as_str()
                                .unwrap_or("unknown error")
                        );
                    }
                panic!("expected result.content[0].text, got: {resp}")
            })
            .to_string()
    }

    fn send_raw(&mut self, raw: &str) -> String {
        self.send(raw);
        self.recv()
    }

    fn initialize(&mut self) {
        let resp = self.send_raw(
            r#"{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-11-25"},"id":1}"#,
        );
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(
            v.get("result").is_some(),
            "initialize failed: {resp}"
        );
    }

    fn assert_tools_list(&mut self) {
        let resp = self.send_raw(r#"{"jsonrpc":"2.0","method":"tools/list","id":1}"#);
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let tools = v["result"]["tools"]
            .as_array()
            .expect("tools/list should return array");
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert!(
            names.contains(&"create_entities"),
            "missing KG tool: {names:?}"
        );
        assert!(
            names.contains(&"vector_upsert_embedding"),
            "missing vector tool: {names:?}"
        );
        assert!(
            names.contains(&"vector_search_entities"),
            "missing vector_search tool: {names:?}"
        );
        assert!(
            names.contains(&"hybrid_search"),
            "missing hybrid_search tool: {names:?}"
        );
    }
}

fn make_embedding(dims: usize, value: f64) -> Vec<f64> {
    vec![value; dims]
}

fn make_varied_embedding(dims: usize, base: f64) -> Vec<f64> {
    (0..dims).map(|i| base + (i as f64 * 0.1)).collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[test]
fn test_vector_e2e_initialize() {
    let mut c = spawn_vec_server();
    c.initialize();
}

#[test]
fn test_vector_e2e_tools_list() {
    let mut c = spawn_vec_server();
    c.initialize();
    c.assert_tools_list();
}

#[test]
fn test_vector_e2e_upsert_and_search() {
    let mut c = spawn_vec_server();

    // Create entities in KG
    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "alice", "entityType": "person", "observations": ["likes math"]},
            {"name": "bob", "entityType": "person", "observations": ["likes sports"]}
        ]}),
    );

    // Upsert embeddings
    let emb_a = make_varied_embedding(4, 1.0);
    let emb_b = make_varied_embedding(4, 0.1);

    let resp = c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({
            "entityName": "alice",
            "embedding": emb_a,
            "model": "test-model"
        }),
    );
    assert!(resp.contains("alice"), "upsert alice: {resp}");

    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({
            "entityName": "bob",
            "embedding": emb_b,
        }),
    );

    // Search by vector — alice should be first for query similar to alice
    let query = make_varied_embedding(4, 1.0);
    let text = c.tool_text(
        "vector_search_entities",
        &serde_json::json!({
            "embedding": query,
            "topK": 5
        }),
    );
    assert!(text.contains("alice"), "search should find alice: {text}");
    assert!(text.contains("bob"), "search should find bob: {text}");
    assert!(text.contains("score"), "search should include scores: {text}");
}

#[test]
fn test_vector_e2e_delete_embedding() {
    let mut c = spawn_vec_server();

    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "alice", "entityType": "person", "observations": []}
        ]}),
    );

    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({
            "entityName": "alice",
            "embedding": make_embedding(4, 1.0)
        }),
    );

    // Delete embedding
    let text = c.tool_text(
        "vector_delete_embedding",
        &serde_json::json!({"entityName": "alice"}),
    );
    assert!(text.contains(r#""deleted":true"#), "should be deleted: {text}");

    // Search should be empty
    let text = c.tool_text(
        "vector_search_entities",
        &serde_json::json!({
            "embedding": make_embedding(4, 1.0),
            "topK": 5
        }),
    );
    assert!(text.contains(r#""count":0"#), "no results: {text}");
}

#[test]
fn test_vector_e2e_nonexistent_entity() {
    let mut c = spawn_vec_server();

    let resp = c.call_tool(
        "vector_upsert_embedding",
        &serde_json::json!({
            "entityName": "nonexistent",
            "embedding": make_embedding(4, 1.0)
        }),
    );
    let is_err = resp["result"]["isError"].as_bool().unwrap_or(false);
    assert!(is_err, "should error for nonexistent entity: {resp}");
}

#[test]
fn test_vector_e2e_dimension_mismatch() {
    let mut c = spawn_vec_server();

    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "alice", "entityType": "person", "observations": []}
        ]}),
    );

    // Upsert with wrong dimension (8 instead of 4)
    let resp = c.call_tool(
        "vector_upsert_embedding",
        &serde_json::json!({
            "entityName": "alice",
            "embedding": make_embedding(8, 1.0)
        }),
    );
    let is_err = resp["result"]["isError"].as_bool().unwrap_or(false);
    assert!(is_err, "should error for dim mismatch: {resp}");
}

#[test]
fn test_vector_e2e_search_type_filter() {
    let mut c = spawn_vec_server();

    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "alice", "entityType": "person", "observations": []},
            {"name": "acme", "entityType": "organization", "observations": []}
        ]}),
    );

    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "alice", "embedding": make_embedding(4, 1.0)}),
    );
    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "acme", "embedding": make_embedding(4, 0.95)}),
    );

    // Filter by person — should only get alice
    let text = c.tool_text(
        "vector_search_entities",
        &serde_json::json!({
            "embedding": make_embedding(4, 1.0),
            "entityType": "person"
        }),
    );
    assert!(text.contains("alice"), "should contain alice: {text}");
    assert!(!text.contains("acme"), "should not contain acme: {text}");
}

#[test]
fn test_vector_e2e_upsert_replace() {
    let mut c = spawn_vec_server();

    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "alice", "entityType": "person", "observations": []}
        ]}),
    );

    // First embedding
    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "alice", "embedding": make_embedding(4, 1.0)}),
    );

    // Replace with different embedding
    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "alice", "embedding": make_embedding(4, 0.1)}),
    );

    // Search with query close to 0.1 — alice should still appear
    let text = c.tool_text(
        "vector_search_entities",
        &serde_json::json!({"embedding": make_embedding(4, 0.1)}),
    );
    assert!(text.contains("alice"), "alice should be found: {text}");
}

#[test]
fn test_vector_e2e_hybrid_search() {
    let mut c = spawn_vec_server();

    // Create entities with text content
    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "Einstein", "entityType": "scientist", "observations": ["physics", "relativity", "Nobel prize"]},
            {"name": "Newton", "entityType": "scientist", "observations": ["physics", "gravity", "calculus"]},
            {"name": "Mozart", "entityType": "musician", "observations": ["music", "composer", "symphony"]}
        ]}),
    );

    // Add embeddings (Einstein and Newton have similar vectors, Mozart is different)
    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "Einstein", "embedding": make_varied_embedding(4, 1.0)}),
    );
    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "Newton", "embedding": make_varied_embedding(4, 0.9)}),
    );
    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "Mozart", "embedding": make_varied_embedding(4, 0.0)}),
    );

    // Hybrid search: text query for "physics" + vector query close to Einstein
    let text = c.tool_text(
        "hybrid_search",
        &serde_json::json!({
            "queryText": "physics",
            "queryEmbedding": make_varied_embedding(4, 1.0),
            "textWeight": 0.5,
            "vecWeight": 0.5,
            "topK": 5
        }),
    );
    assert!(text.contains("Einstein"), "hybrid should find Einstein: {text}");
    assert!(text.contains("Newton"), "hybrid should find Newton: {text}");
    assert!(text.contains("score"), "hybrid should include scores: {text}");
}

#[test]
fn test_vector_e2e_refresh_graph_cache() {
    let mut c = spawn_vec_server();

    // Create entities
    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "alice", "entityType": "person", "observations": []},
            {"name": "bob", "entityType": "person", "observations": []}
        ]}),
    );

    // Add embeddings
    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "alice", "embedding": make_embedding(4, 1.0)}),
    );
    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "bob", "embedding": make_embedding(4, 0.5)}),
    );

    // Add relation
    c.tool_text(
        "create_relations",
        &serde_json::json!({"relations": [
            {"from": "alice", "to": "bob", "relationType": "knows"}
        ]}),
    );

    // Refresh graph cache
    let text = c.tool_text("vector_refresh_graph_cache", &serde_json::json!({}));
    assert!(text.contains("\"nodes\""), "refresh should return node count: {text}");
}

#[test]
fn test_vector_e2e_store_stats() {
    let mut c = spawn_vec_server();

    let text = c.tool_text("vector_store_stats", &serde_json::json!({}));
    assert!(text.contains("embeddingCount"), "stats should show count: {text}");
    assert!(text.contains("dims"), "stats should show dims: {text}");
}

#[test]
fn test_vector_e2e_search_top_k() {
    let mut c = spawn_vec_server();

    // Create 6 entities
    let entities: Vec<serde_json::Value> = (0..6u32)
        .map(|i| {
            serde_json::json!({
                "name": format!("e{i}"),
                "entityType": "test",
                "observations": []
            })
        })
        .collect();

    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": entities}),
    );

    for i in 0..6u32 {
        c.tool_text(
            "vector_upsert_embedding",
            &serde_json::json!({
                "entityName": format!("e{i}"),
                "embedding": make_embedding(4, f64::from(i) * 0.2)
            }),
        );
    }

    // Search with topK=3
    let text = c.tool_text(
        "vector_search_entities",
        &serde_json::json!({
            "embedding": make_embedding(4, 0.0),
            "topK": 3
        }),
    );
    assert!(text.contains(r#""count":3"#), "should return exactly 3: {text}");
}

#[test]
fn test_vector_e2e_kg_tools_still_work() {
    let mut c = spawn_vec_server();

    // Standard KG operations should still work
    let text = c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "test", "entityType": "test", "observations": ["obs"]}
        ]}),
    );
    assert!(!text.contains("error"), "KG create should work: {text}");

    let text = c.tool_text("search_nodes", &serde_json::json!({"query": "test"}));
    assert!(text.contains("test"), "KG search should work: {text}");

    let text = c.tool_text("graph_stats", &serde_json::json!({}));
    assert!(text.contains("entities"), "KG stats should work: {text}");
}

#[test]
fn test_vector_e2e_search_empty_store() {
    let mut c = spawn_vec_server();

    // Search with no embeddings
    let text = c.tool_text(
        "vector_search_entities",
        &serde_json::json!({
            "embedding": make_embedding(4, 1.0),
            "topK": 5
        }),
    );
    assert!(text.contains(r#""count":0"#), "empty store: {text}");
}

/// Assert that a tool call comes back as a protocol-level tool error
/// (`result.isError == true`) rather than succeeding.
fn assert_tool_error(c: &mut VecClient, name: &str, args: &serde_json::Value) {
    let resp = c.call_tool(name, args);
    let is_err = resp["result"]["isError"].as_bool().unwrap_or(false);
    assert!(is_err, "expected isError for {name}, got: {resp}");
}

#[test]
fn test_vector_e2e_upsert_missing_params() {
    let mut c = spawn_vec_server();
    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "alice", "entityType": "person", "observations": []}
        ]}),
    );

    // Missing embedding
    assert_tool_error(
        &mut c,
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "alice"}),
    );
    // Missing entityName
    assert_tool_error(
        &mut c,
        "vector_upsert_embedding",
        &serde_json::json!({"embedding": make_embedding(4, 1.0)}),
    );
    // Empty entityName
    assert_tool_error(
        &mut c,
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "", "embedding": make_embedding(4, 1.0)}),
    );
    // Empty embedding array
    assert_tool_error(
        &mut c,
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "alice", "embedding": []}),
    );
    // Non-numeric embedding value
    assert_tool_error(
        &mut c,
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "alice", "embedding": ["a", "b", "c", "d"]}),
    );
}

#[test]
fn test_vector_e2e_search_missing_embedding() {
    let mut c = spawn_vec_server();
    assert_tool_error(
        &mut c,
        "vector_search_entities",
        &serde_json::json!({"topK": 5}),
    );
}

#[test]
fn test_vector_e2e_hybrid_missing_params() {
    let mut c = spawn_vec_server();
    // Missing queryEmbedding
    assert_tool_error(
        &mut c,
        "hybrid_search",
        &serde_json::json!({"queryText": "physics"}),
    );
    // Missing queryText
    assert_tool_error(
        &mut c,
        "hybrid_search",
        &serde_json::json!({"queryEmbedding": make_embedding(4, 1.0)}),
    );
}

#[test]
fn test_vector_e2e_unknown_tool() {
    let mut c = spawn_vec_server();
    let resp = c.call_tool("vector_does_not_exist", &serde_json::json!({}));
    // Unknown methods come back as JSON-RPC errors, not tool results.
    assert!(
        resp.get("error").is_some(),
        "unknown tool should be a protocol error: {resp}"
    );
}

#[test]
fn test_vector_e2e_topk_clamped() {
    let mut c = spawn_vec_server();
    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "alice", "entityType": "person", "observations": []}
        ]}),
    );
    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "alice", "embedding": make_embedding(4, 1.0)}),
    );
    // topK far above the 100 cap must not error — it should clamp and return.
    let text = c.tool_text(
        "vector_search_entities",
        &serde_json::json!({"embedding": make_embedding(4, 1.0), "topK": 100000}),
    );
    assert!(text.contains("alice"), "clamped search should find alice: {text}");
}

#[test]
fn test_vector_e2e_custom_index_config() {
    // Non-default HNSW knobs (L2 metric, f16 quantization, custom expansion)
    // must be accepted and produce a working index end-to-end.
    let mut c = spawn_vec_server_with(&[
        "--vec-metric",
        "l2sq",
        "--vec-quantization",
        "f16",
        "--vec-connectivity",
        "32",
        "--vec-expansion-add",
        "128",
        "--vec-expansion-search",
        "64",
    ]);

    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "alice", "entityType": "person", "observations": []},
            {"name": "bob", "entityType": "person", "observations": []}
        ]}),
    );
    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "alice", "embedding": make_varied_embedding(4, 1.0)}),
    );
    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "bob", "embedding": make_varied_embedding(4, 0.1)}),
    );

    let text = c.tool_text(
        "vector_search_entities",
        &serde_json::json!({"embedding": make_varied_embedding(4, 1.0), "topK": 5}),
    );
    assert!(text.contains("alice"), "custom-config search should work: {text}");
}

#[test]
fn test_vector_e2e_stats_after_data() {
    let mut c = spawn_vec_server();
    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "alice", "entityType": "person", "observations": []},
            {"name": "bob", "entityType": "person", "observations": []}
        ]}),
    );
    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "alice", "embedding": make_embedding(4, 1.0)}),
    );
    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "bob", "embedding": make_embedding(4, 0.5)}),
    );

    let text = c.tool_text("vector_store_stats", &serde_json::json!({}));
    assert!(
        text.contains(r#""embeddingCount":2"#),
        "stats should report 2 embeddings: {text}"
    );
    assert!(text.contains(r#""dims":4"#), "stats should report dims=4: {text}");
}

// ─── Modern AI workload tools ─────────────────────────────────────────────

/// Seed entities `n0..nN` and (optionally) embeddings via batch upsert.
fn seed(c: &mut VecClient, names: &[(&str, &str)]) {
    let entities: Vec<serde_json::Value> = names
        .iter()
        .map(|(n, t)| serde_json::json!({"name": n, "entityType": t, "observations": []}))
        .collect();
    c.tool_text("create_entities", &serde_json::json!({"entities": entities}));
}

#[test]
fn test_vector_e2e_batch_upsert() {
    let mut c = spawn_vec_server();
    seed(&mut c, &[("a", "doc"), ("b", "doc"), ("missing_entity_ok", "doc")]);

    // One item targets a non-existent entity → reported in errors, not fatal.
    let text = c.tool_text(
        "vector_batch_upsert",
        &serde_json::json!({"items": [
            {"entityName": "a", "embedding": make_embedding(4, 1.0)},
            {"entityName": "b", "embedding": make_embedding(4, 0.2), "model": "m"},
            {"entityName": "ghost", "embedding": make_embedding(4, 0.3)}
        ]}),
    );
    assert!(text.contains(r#""upserted":2"#), "expected 2 upserted: {text}");
    assert!(text.contains(r#""failed":1"#), "expected 1 failed: {text}");

    // Both successful embeddings are now searchable.
    let s = c.tool_text(
        "vector_search_entities",
        &serde_json::json!({"embedding": make_embedding(4, 1.0), "topK": 5}),
    );
    assert!(s.contains("\"a\"") && s.contains("\"b\""), "search after batch: {s}");
}

#[test]
fn test_vector_e2e_get_embedding() {
    let mut c = spawn_vec_server();
    seed(&mut c, &[("a", "doc")]);
    c.tool_text(
        "vector_upsert_embedding",
        &serde_json::json!({"entityName": "a", "embedding": [0.1, 0.2, 0.3, 0.4], "model": "m1"}),
    );

    let text = c.tool_text("vector_get_embedding", &serde_json::json!({"entityName": "a"}));
    assert!(text.contains(r#""model":"m1""#), "get_embedding model: {text}");
    assert!(text.contains("0.1") && text.contains("0.4"), "get_embedding values: {text}");

    let missing = c.tool_text("vector_get_embedding", &serde_json::json!({"entityName": "nobody"}));
    assert!(missing.contains(r#""found":false"#), "missing embedding: {missing}");
}

#[test]
fn test_vector_e2e_search_by_entity() {
    let mut c = spawn_vec_server();
    seed(&mut c, &[("a", "doc"), ("b", "doc"), ("c", "doc")]);
    c.tool_text("vector_upsert_embedding", &serde_json::json!({"entityName": "a", "embedding": make_embedding(4, 1.0)}));
    c.tool_text("vector_upsert_embedding", &serde_json::json!({"entityName": "b", "embedding": make_embedding(4, 0.98)}));
    c.tool_text("vector_upsert_embedding", &serde_json::json!({"entityName": "c", "embedding": make_embedding(4, 0.1)}));

    // "More like a": should return b/c but not a itself (excludeSelf default).
    let text = c.tool_text("vector_search_by_entity", &serde_json::json!({"entityName": "a", "topK": 5}));
    assert!(!text.contains("\"a\""), "should exclude self: {text}");
    assert!(text.contains("\"b\""), "should include similar b: {text}");

    // With excludeSelf=false, a appears (and is the closest to itself).
    let text2 = c.tool_text("vector_search_by_entity", &serde_json::json!({"entityName": "a", "topK": 5, "excludeSelf": false}));
    assert!(text2.contains("\"a\""), "should include self when excludeSelf=false: {text2}");
}

#[test]
fn test_vector_e2e_recommend() {
    let mut c = spawn_vec_server();
    seed(&mut c, &[("a", "doc"), ("b", "doc"), ("c", "doc"), ("d", "doc")]);
    c.tool_text("vector_upsert_embedding", &serde_json::json!({"entityName": "a", "embedding": [1.0, 0.0, 0.0, 0.0]}));
    c.tool_text("vector_upsert_embedding", &serde_json::json!({"entityName": "b", "embedding": [0.9, 0.1, 0.0, 0.0]}));
    c.tool_text("vector_upsert_embedding", &serde_json::json!({"entityName": "c", "embedding": [0.0, 1.0, 0.0, 0.0]}));
    c.tool_text("vector_upsert_embedding", &serde_json::json!({"entityName": "d", "embedding": [0.85, 0.15, 0.0, 0.0]}));

    // Liking a & b (the "first axis" cluster) should surface d, not c, and
    // exclude the example entities themselves.
    let text = c.tool_text(
        "vector_recommend",
        &serde_json::json!({"positive": ["a", "b"], "topK": 5}),
    );
    assert!(!text.contains("\"a\"") && !text.contains("\"b\""), "examples excluded: {text}");
    assert!(text.contains("\"d\""), "should recommend d: {text}");

    // Empty positive → error.
    let resp = c.call_tool("vector_recommend", &serde_json::json!({"positive": []}));
    let is_err = resp.get("error").is_some()
        || resp["result"]["isError"].as_bool().unwrap_or(false);
    assert!(is_err, "empty positive should error: {resp}");
}

#[test]
fn test_vector_e2e_mmr_search() {
    let mut c = spawn_vec_server();
    seed(&mut c, &[("a", "doc"), ("a2", "doc"), ("b", "doc")]);
    // a and a2 are near-duplicates; b is different.
    c.tool_text("vector_upsert_embedding", &serde_json::json!({"entityName": "a", "embedding": [1.0, 0.0, 0.0, 0.0]}));
    c.tool_text("vector_upsert_embedding", &serde_json::json!({"entityName": "a2", "embedding": [0.99, 0.01, 0.0, 0.0]}));
    c.tool_text("vector_upsert_embedding", &serde_json::json!({"entityName": "b", "embedding": [0.0, 1.0, 0.0, 0.0]}));

    // Pure relevance (lambda=1) for a query near a: top-2 are the duplicates.
    let rel = c.tool_text(
        "vector_mmr_search",
        &serde_json::json!({"embedding": [1.0, 0.0, 0.0, 0.0], "topK": 2, "lambda": 1.0}),
    );
    assert!(rel.contains("\"a\"") && rel.contains("\"a2\""), "relevance picks duplicates: {rel}");

    // Diversity-leaning (lambda=0.2): after picking a, the near-duplicate a2 is
    // penalised so the dissimilar b is preferred for the second slot.
    let div = c.tool_text(
        "vector_mmr_search",
        &serde_json::json!({"embedding": [1.0, 0.0, 0.0, 0.0], "topK": 2, "lambda": 0.2}),
    );
    assert!(div.contains("\"b\""), "diversity should surface b: {div}");
}

#[test]
fn test_vector_e2e_ivf_backend_end_to_end() {
    // Run the whole flow against the IVF-Flat backend.
    let mut c = spawn_vec_server_with(&["--vec-index", "ivf", "--ivf-nlist", "2", "--ivf-nprobe", "2"]);
    c.initialize();

    let stats0 = c.tool_text("vector_store_stats", &serde_json::json!({}));
    assert!(stats0.contains(r#""indexKind":"ivf""#), "should report ivf: {stats0}");

    seed(&mut c, &[("a", "doc"), ("b", "doc"), ("c", "doc"), ("d", "doc")]);
    c.tool_text(
        "vector_batch_upsert",
        &serde_json::json!({"items": [
            {"entityName": "a", "embedding": [1.0, 1.0, 0.0, 0.0]},
            {"entityName": "b", "embedding": [1.0, 0.9, 0.0, 0.0]},
            {"entityName": "c", "embedding": [0.0, 0.0, 1.0, 1.0]},
            {"entityName": "d", "embedding": [0.0, 0.0, 1.0, 0.9]}
        ]}),
    );

    // Reindex (trains IVF k-means).
    let re = c.tool_text("vector_reindex", &serde_json::json!({}));
    assert!(re.contains(r#""reindexed":true"#), "reindex: {re}");
    assert!(re.contains(r#""indexKind":"ivf""#), "reindex kind: {re}");

    // Search near the {a,b} cluster returns those, not the far {c,d} cluster.
    let s = c.tool_text(
        "vector_search_entities",
        &serde_json::json!({"embedding": [1.0, 1.0, 0.0, 0.0], "topK": 2}),
    );
    assert!(s.contains("\"a\""), "ivf search should find a: {s}");
    assert!(!s.contains("\"c\"") && !s.contains("\"d\""), "ivf search should not return far cluster: {s}");
}

#[test]
fn test_vector_e2e_turboquant_backend_end_to_end() {
    // Run the whole flow against the TurboQuant quantized backend. The
    // backend requires dims in 384..=1536, so override the helper's tiny
    // default (clap takes the last occurrence of a flag).
    let mut c = spawn_vec_server_with(&[
        "--vec-index",
        "turbo",
        "--tq-bits",
        "4",
        "--embedding-dims",
        "384",
    ]);
    c.initialize();

    let stats0 = c.tool_text("vector_store_stats", &serde_json::json!({}));
    assert!(stats0.contains(r#""indexKind":"turboquant""#), "should report turboquant: {stats0}");

    // Two well-separated direction clusters in 384 dims.
    let emb = |hot: [usize; 2], warm: f32| {
        let mut v = vec![0.0f32; 384];
        v[hot[0]] = 1.0;
        v[hot[1]] = warm;
        v
    };
    seed(&mut c, &[("a", "doc"), ("b", "doc"), ("c", "doc"), ("d", "doc")]);
    c.tool_text(
        "vector_batch_upsert",
        &serde_json::json!({"items": [
            {"entityName": "a", "embedding": emb([0, 1], 1.0)},
            {"entityName": "b", "embedding": emb([0, 1], 0.9)},
            {"entityName": "c", "embedding": emb([200, 201], 1.0)},
            {"entityName": "d", "embedding": emb([200, 201], 0.9)}
        ]}),
    );

    // Reindex is a no-op for the data-oblivious quantizer but must succeed.
    let re = c.tool_text("vector_reindex", &serde_json::json!({}));
    assert!(re.contains(r#""reindexed":true"#), "reindex: {re}");
    assert!(re.contains(r#""indexKind":"turboquant""#), "reindex kind: {re}");

    // Search near the {a,b} cluster returns those, not the far {c,d} cluster.
    let s = c.tool_text(
        "vector_search_entities",
        &serde_json::json!({"embedding": emb([0, 1], 1.0), "topK": 2}),
    );
    assert!(s.contains("\"a\""), "turboquant search should find a: {s}");
    assert!(!s.contains("\"c\"") && !s.contains("\"d\""), "turboquant search should not return far cluster: {s}");

    // Deletion works against the quantized store.
    let del = c.tool_text("vector_delete_embedding", &serde_json::json!({"entityName": "a"}));
    assert!(del.contains("true") || del.contains("deleted"), "delete: {del}");
    let s2 = c.tool_text(
        "vector_search_entities",
        &serde_json::json!({"embedding": emb([0, 1], 1.0), "topK": 4}),
    );
    assert!(!s2.contains("\"a\""), "deleted entity must not be returned: {s2}");
}
