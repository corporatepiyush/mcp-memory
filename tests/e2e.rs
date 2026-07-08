use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

static DB_COUNTER: AtomicU32 = AtomicU32::new(0);

struct McpClient {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: std::process::ChildStdout,
    db_path: String,
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        for ext in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{}", self.db_path, ext));
        }
    }
}

fn spawn_server() -> McpClient {
    let n = DB_COUNTER.fetch_add(1, Ordering::SeqCst);
    let db_path = format!("/tmp/test_e2e_{n}.db");
    for ext in ["", "-wal", "-shm"] {
        let p = format!("{db_path}{ext}");
        let _ = std::fs::remove_file(&p);
    }

    let bin = std::env::var("CARGO_BIN_EXE_MCP_MEMORY")
        .unwrap_or_else(|_| "target/debug/mcp-memory".into());
    let mut child = Command::new(&bin)
        .arg("-f")
        .arg(&db_path)
        .arg("--transport")
        .arg("stdio")
        .arg("--log-level")
        .arg("error")
        .arg("--enable-graph-read")
        .arg("--enable-graph-write")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn mcp-memory");
    McpClient {
        stdin: child.stdin.take().unwrap(),
        stdout: child.stdout.take().unwrap(),
        child,
        db_path,
    }
}

impl McpClient {
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
                panic!("expected result.content[0].text, got: {resp}")
            })
            .to_string()
    }
}

#[test]
fn e2e_create_and_read_graph() {
    let mut c = spawn_server();

    let text = c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "Ada", "entityType": "person", "observations": ["mathematician"]}
        ]}),
    );
    assert!(!text.contains("error"), "create_entities failed: {text}");

    let text = c.tool_text("read_graph", &serde_json::json!({}));
    assert!(text.contains("Ada"), "read_graph missing Ada: {text}");

    let text = c.tool_text("search_nodes", &serde_json::json!({"query": "mathematician"}));
    assert!(text.contains("Ada"), "search_nodes missing Ada: {text}");

    let text = c.tool_text("open_nodes", &serde_json::json!({"names": ["Ada"]}));
    assert!(text.contains("Ada"), "open_nodes missing Ada: {text}");
}

#[test]
fn e2e_add_delete_observations() {
    let mut c = spawn_server();

    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "E", "entityType": "t", "observations": ["a"]}
        ]}),
    );

    let text = c.tool_text(
        "add_observations",
        &serde_json::json!({"observations": [
            {"entityName": "E", "contents": ["b", "c"]}
        ]}),
    );
    assert!(!text.contains("error"), "add_observations failed: {text}");

    let text = c.tool_text("read_graph", &serde_json::json!({}));
    assert!(text.contains("\"a\""), "read_graph missing a: {text}");
    assert!(text.contains("\"b\""), "read_graph missing b: {text}");
    assert!(text.contains("\"c\""), "read_graph missing c: {text}");

    c.tool_text(
        "delete_observations",
        &serde_json::json!({"deletions": [
            {"entityName": "E", "observations": ["b"]}
        ]}),
    );

    let text = c.tool_text("read_graph", &serde_json::json!({}));
    assert!(text.contains("\"a\""), "should still have a: {text}");
    assert!(!text.contains("\"b\""), "should not have b: {text}");
    assert!(text.contains("\"c\""), "should still have c: {text}");
}

#[test]
fn e2e_relations_and_paths() {
    let mut c = spawn_server();

    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "A", "entityType": "node", "observations": []},
            {"name": "B", "entityType": "node", "observations": []},
            {"name": "C", "entityType": "node", "observations": []}
        ]}),
    );

    c.tool_text(
        "create_relations",
        &serde_json::json!({"relations": [
            {"from": "A", "to": "B", "relationType": "edge"},
            {"from": "B", "to": "C", "relationType": "edge"}
        ]}),
    );

    let text = c.tool_text("find_path", &serde_json::json!({"from": "A", "to": "C"}));
    assert!(!text.contains("error"), "find_path failed: {text}");
    assert!(text.contains("A") && text.contains("C"), "find_path: {text}");

    let text = c.tool_text("graph_stats", &serde_json::json!({}));
    assert!(text.contains("\"entities\":3"), "graph_stats: {text}");
    assert!(text.contains("\"relations\":2"), "graph_stats: {text}");
}

#[test]
fn e2e_search_filtered() {
    let mut c = spawn_server();

    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "E1", "entityType": "person", "observations": ["math"]},
            {"name": "E2", "entityType": "place", "observations": ["math"]}
        ]}),
    );

    let text = c.tool_text("search_nodes", &serde_json::json!({"query": "math"}));
    assert!(text.contains("E1") && text.contains("E2"), "search both: {text}");

    let text = c.tool_text("open_nodes", &serde_json::json!({"names": ["E1", "E2"]}));
    assert!(text.contains("E1") && text.contains("E2"), "open both: {text}");
}

#[test]
fn e2e_delete_and_stats() {
    let mut c = spawn_server();

    // Create 3 entities + relations.
    c.tool_text("create_entities", &serde_json::json!({"entities": [
        {"name": "X", "entityType": "alpha", "observations": ["x-obs"]},
        {"name": "Y", "entityType": "beta",  "observations": ["y-obs"]},
        {"name": "Z", "entityType": "alpha", "observations": []}
    ]}));
    c.tool_text("create_relations", &serde_json::json!({"relations": [
        {"from": "X", "to": "Y", "relationType": "linked"},
        {"from": "Y", "to": "Z", "relationType": "linked"}
    ]}));

    // Verify initial stats.
    let st = c.tool_text("graph_stats", &serde_json::json!({}));
    assert!(st.contains("\"entities\":3"), "3 entities: {st}");
    assert!(st.contains("\"relations\":2"), "2 relations: {st}");

    let types = c.tool_text("list_entity_types", &serde_json::json!({}));
    assert!(types.contains("\"count\":2"), "alpha has 2: {types}");

    // entity_exists.
    let exist = c.tool_text("entity_exists", &serde_json::json!({"names": ["X","Y","Missing"]}));
    assert_eq!(exist, "[true,true,false]", "entity_exists: {exist}");

    // Delete one relation, verify.
    c.tool_text("delete_relations", &serde_json::json!({"relations": [
        {"from": "X", "to": "Y", "relationType": "linked"}
    ]}));
    let st = c.tool_text("graph_stats", &serde_json::json!({}));
    assert!(st.contains("\"relations\":1"), "1 relation remains: {st}");

    // Delete entity Y (should cascade relations) + entity Z.
    c.tool_text("delete_entities", &serde_json::json!({"entityNames": ["Y", "Z"]}));
    let st = c.tool_text("graph_stats", &serde_json::json!({}));
    assert!(st.contains("\"entities\":1"), "1 entity left: {st}");
    assert!(st.contains("\"relations\":0"), "0 relations: {st}");

    // observations cascaded.
    let open = c.tool_text("open_nodes", &serde_json::json!({"names": ["X"]}));
    assert!(open.contains("x-obs"), "X obs remain: {open}");

    // Relation type count still reflects the cascade (delete_entities does not
    // decrement relation type counts — the type entry persists in type_dict).
    let rtypes = c.tool_text("list_relation_types", &serde_json::json!({}));
    assert!(rtypes.contains("\"type\":\"linked\""), "linked type exists: {rtypes}");
}

#[test]
fn e2e_upsert_merge_and_wipe() {
    let mut c = spawn_server();

    // Create entities.
    c.tool_text("create_entities", &serde_json::json!({"entities": [
        {"name": "Src", "entityType": "old", "observations": ["a", "b"]},
        {"name": "Tgt", "entityType": "old", "observations": ["c"]}
    ]}));

    // Upsert — change type and add obs.
    c.tool_text("upsert_entities", &serde_json::json!({"entities": [
        {"name": "Tgt", "entityType": "new", "observations": ["c", "d"]}
    ]}));
    let open = c.tool_text("open_nodes", &serde_json::json!({"names": ["Tgt"]}));
    assert!(open.contains("new"), "type changed: {open}");
    assert!(open.contains("\"c\""), "c preserved: {open}");
    assert!(open.contains("\"d\""), "d added: {open}");

    // Merge Src → Tgt.
    c.tool_text("merge_entities", &serde_json::json!({"source": "Src", "target": "Tgt"}));
    let st = c.tool_text("graph_stats", &serde_json::json!({}));
    assert!(st.contains("\"entities\":1"), "only Tgt remains: {st}");

    // Tgt should have all observations (a, b, c, d).
    let open = c.tool_text("open_nodes", &serde_json::json!({"names": ["Tgt"]}));
    assert!(open.contains("\"a\""), "merged a: {open}");
    assert!(open.contains("\"d\""), "merged d: {open}");

    // Degree of Tgt (none, no relations).
    let deg = c.tool_text("degree", &serde_json::json!({"name": "Tgt"}));
    assert!(deg.contains("\"degree\":0"), "degree 0: {deg}");

    // Export graph as JSON.
    let exp = c.tool_text("export_graph", &serde_json::json!({"format": "json"}));
    assert!(exp.contains("Tgt"), "export contains Tgt: {exp}");
    assert!(exp.contains("new"), "export contains new type: {exp}");
}

#[test]
fn e2e_relations_and_describe() {
    let mut c = spawn_server();

    c.tool_text("create_entities", &serde_json::json!({"entities": [
        {"name": "A", "entityType": "t", "observations": []},
        {"name": "B", "entityType": "t", "observations": []},
        {"name": "C", "entityType": "t", "observations": []}
    ]}));
    c.tool_text("create_relations", &serde_json::json!({"relations": [
        {"from": "A", "to": "B", "relationType": "knows"},
        {"from": "B", "to": "C", "relationType": "knows"},
        {"from": "A", "to": "C", "relationType": "likes"}
    ]}));

    // search_relations by from.
    let r = c.tool_text("search_relations", &serde_json::json!({"from": "A"}));
    assert!(r.contains("knows"), "A→B knows: {r}");
    assert!(r.contains("likes"), "A→C likes: {r}");

    // search_relations by from + type.
    let r = c.tool_text("search_relations", &serde_json::json!({"from": "A", "relationType": "likes"}));
    assert!(!r.contains("knows"), "filtered knows out: {r}");
    assert!(r.contains("likes"), "filtered likes in: {r}");

    // search_relations by to.
    let r = c.tool_text("search_relations", &serde_json::json!({"to": "C"}));
    assert!(r.contains("knows"), "B→C knows: {r}");
    assert!(r.contains("likes"), "A→C likes: {r}");

    // get_neighbors (depth 1, outgoing).
    let n = c.tool_text("get_neighbors", &serde_json::json!({"name": "A", "direction": "OUTGOING"}));
    assert!(n.contains("B"), "A neighbor B: {n}");
    assert!(n.contains("C"), "A neighbor C: {n}");

    // get_neighbors (depth 1, incoming for C).
    let n = c.tool_text("get_neighbors", &serde_json::json!({"name": "C", "direction": "INCOMING"}));
    assert!(n.contains("A"), "C incoming A: {n}");
    assert!(n.contains("B"), "C incoming B: {n}");

    // describe_entity returns the entity with its observations.
    let d = c.tool_text("describe_entity", &serde_json::json!({"name": "A"}));
    assert!(d.contains("\"name\":\"A\""), "desc has A: {d}");

    // degree (outgoing for A = 2).
    let deg = c.tool_text("degree", &serde_json::json!({"name": "A"}));
    assert!(deg.contains("\"degree\":2"), "A degree 2: {deg}");

    // degree (both for B = 2, 1 in + 1 out).
    let deg = c.tool_text("degree", &serde_json::json!({"name": "B", "direction": "BOTH"}));
    assert!(deg.contains("\"degree\":2"), "B degree 2: {deg}");

    // degree (incoming for A = 0).
    let deg = c.tool_text("degree", &serde_json::json!({"name": "A", "direction": "INCOMING"}));
    assert!(deg.contains("\"degree\":0"), "A incoming 0: {deg}");

    // find_all_paths A→C.
    let p = c.tool_text("find_all_paths", &serde_json::json!({"from": "A", "to": "C"}));
    assert!(p.contains("A"), "paths include A: {p}");
    assert!(p.contains("C"), "paths include C: {p}");

    // batch_get_entities for A, B, C.
    let bg = c.tool_text("batch_get_entities", &serde_json::json!({"names": ["A", "B", "C"]}));
    assert!(bg.contains("A") && bg.contains("B") && bg.contains("C"), "batch get all: {bg}");
}

#[test]
fn e2e_read_graph_relations_scoped_to_page() {
    let mut c = spawn_server();

    c.tool_text("create_entities", &serde_json::json!({"entities": [
        {"name": "A", "entityType": "n", "observations": []},
        {"name": "B", "entityType": "n", "observations": []},
        {"name": "C", "entityType": "n", "observations": []}
    ]}));
    c.tool_text("create_relations", &serde_json::json!({"relations": [
        {"from": "A", "to": "B", "relationType": "edge"},
        {"from": "B", "to": "C", "relationType": "edge"}
    ]}));

    // Full read_graph: both edges present.
    let full = c.tool_text("read_graph", &serde_json::json!({}));
    let v: serde_json::Value = serde_json::from_str(&full).unwrap();
    assert_eq!(v["entities"].as_array().unwrap().len(), 3, "all entities: {full}");
    assert_eq!(v["relations"].as_array().unwrap().len(), 2, "both edges: {full}");

    // First-entity page (A): its edge A->B straddles the page boundary, so no
    // relations are returned.
    let page = c.tool_text("read_graph", &serde_json::json!({"limit": 1}));
    let v: serde_json::Value = serde_json::from_str(&page).unwrap();
    assert_eq!(v["entities"].as_array().unwrap().len(), 1, "one entity: {page}");
    assert_eq!(v["relations"].as_array().unwrap().len(), 0, "no scoped edges: {page}");
}

#[test]
fn e2e_search_edge_cases() {
    let mut c = spawn_server();

    // Empty database search.
    let s = c.tool_text("search_nodes", &serde_json::json!({"query": "anything"}));
    assert_eq!(s, "[]", "empty search: {s}");

    c.tool_text("create_entities", &serde_json::json!({"entities": [
        {"name": "Alice", "entityType": "person", "observations": ["likes math"]},
        {"name": "Bob",   "entityType": "person", "observations": ["likes math", "likes science"]}
    ]}));

    // Search with filter type.
    let s = c.tool_text("search_nodes", &serde_json::json!({"query": "likes", "entityType": "person"}));
    assert!(s.contains("Alice"), "filtered search Alice: {s}");

    // Search with offset/limit.
    let s = c.tool_text("search_nodes", &serde_json::json!({"query": "likes", "offset": 1, "limit": 1}));
    assert!(!s.contains("Alice"), "offset skips Alice: {s}");
    assert!(s.contains("Bob"), "limit includes Bob: {s}");

    // read_graph with type filter.
    let g = c.tool_text("read_graph", &serde_json::json!({"type": "person"}));
    assert!(g.contains("Alice"), "filtered graph: {g}");

    // read_graph with offset/limit.
    let g = c.tool_text("read_graph", &serde_json::json!({"offset": 1, "limit": 1}));
    assert!(!g.contains("Alice"), "offset graph: {g}");

    // describe_entity with relations (A has none).
    let d = c.tool_text("describe_entity", &serde_json::json!({"name": "Alice"}));
    assert!(d.contains("Alice"), "describe: {d}");
}

#[test]
fn e2e_vectors_disabled_by_default() {
    let mut c = spawn_server();

    // tools/list must not advertise the vector tools on a plain KG server.
    c.send(r#"{"jsonrpc":"2.0","method":"tools/list","id":1}"#);
    let list = c.recv();
    assert!(
        list.contains("create_entities"),
        "base tools present: {list}"
    );
    assert!(
        !list.contains("vector_upsert_embedding") && !list.contains("hybrid_search"),
        "vector tools must be hidden when --vectors is off: {list}"
    );

    // Calling a vector tool must fail (vectors disabled) rather than succeed.
    let resp = c.call_tool(
        "vector_search_entities",
        &serde_json::json!({"embedding": [1.0, 1.0, 1.0, 1.0], "topK": 3}),
    );
    let is_protocol_error = resp.get("error").is_some();
    let is_tool_error = resp["result"]["isError"].as_bool().unwrap_or(false);
    assert!(
        is_protocol_error || is_tool_error,
        "vector tool must error when disabled: {resp}"
    );
}

#[test]
fn e2e_pipelined_requests_all_answered_and_correlated() {
    // The stdio transport dispatches up to --stdio-concurrency requests in
    // flight and may write responses in completion order. Fire a burst of
    // pipelined requests in one write and verify every id gets exactly one
    // well-formed response, regardless of arrival order.
    use std::collections::HashSet;
    use std::io::{BufRead, BufReader, Write};

    let mut c = spawn_server();
    c.tool_text(
        "create_entities",
        &serde_json::json!({"entities": [
            {"name": "P", "entityType": "t", "observations": ["o"]}
        ]}),
    );

    let mut batch = String::new();
    for id in 100u64..160 {
        // Alternate cheap and heavier calls so completions interleave.
        let req = if id % 2 == 0 {
            serde_json::json!({"jsonrpc": "2.0", "id": id, "method": "tools/call",
                "params": {"name": "get_entity", "arguments": {"name": "P"}}})
        } else {
            serde_json::json!({"jsonrpc": "2.0", "id": id, "method": "tools/call",
                "params": {"name": "search_nodes", "arguments": {"query": "o"}}})
        };
        batch.push_str(&serde_json::to_string(&req).unwrap());
        batch.push('\n');
    }
    c.stdin.write_all(batch.as_bytes()).unwrap();
    c.stdin.flush().unwrap();

    // One persistent reader: pipelined responses share the buffer.
    let mut reader = BufReader::new(&mut c.stdout);
    let mut seen: HashSet<u64> = HashSet::new();
    for _ in 0..60 {
        let mut line = String::new();
        reader.read_line(&mut line).expect("read response line");
        let v: serde_json::Value = serde_json::from_str(line.trim()).expect("parse response");
        let id = v["id"].as_u64().expect("response id");
        assert!((100..160).contains(&id), "unexpected id {id}");
        assert!(v.get("result").is_some(), "id {id} errored: {line}");
        assert!(seen.insert(id), "duplicate response for id {id}");
    }
    assert_eq!(seen.len(), 60, "every pipelined request must be answered once");
}

#[test]
fn e2e_strict_ordering_with_concurrency_one() {
    // --stdio-concurrency 1 must preserve request order, so pipelined
    // dependent writes (create entity, then relate it) work.
    use std::io::{BufRead, BufReader, Write};

    let n = DB_COUNTER.fetch_add(1, Ordering::SeqCst);
    let db_path = format!("/tmp/test_e2e_{n}.db");
    let bin = std::env::var("CARGO_BIN_EXE_MCP_MEMORY")
        .unwrap_or_else(|_| "target/debug/mcp-memory".into());
    let mut child = Command::new(&bin)
        .args(["-f", &db_path, "--transport", "stdio", "--log-level", "error",
               "--enable-graph-read", "--enable-graph-write",
               "--stdio-concurrency", "1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();

    let reqs = [
        serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "create_entities", "arguments": {"entities": [
                {"name": "A", "entityType": "t", "observations": []},
                {"name": "B", "entityType": "t", "observations": []}]}}}),
        serde_json::json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "create_relations", "arguments": {"relations": [
                {"from": "A", "to": "B", "relationType": "linked"}]}}}),
        serde_json::json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "find_path", "arguments": {"from": "A", "to": "B"}}}),
    ];
    let batch: String = reqs
        .iter()
        .map(|r| serde_json::to_string(r).unwrap() + "\n")
        .collect();
    stdin.write_all(batch.as_bytes()).unwrap();
    stdin.flush().unwrap();

    let mut reader = BufReader::new(stdout);
    for expect_id in 1u64..=3 {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let v: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(v["id"].as_u64(), Some(expect_id), "order violated: {line}");
        let is_err = v["result"]["isError"].as_bool().unwrap_or(false);
        assert!(v.get("result").is_some() && !is_err, "id {expect_id} failed: {line}");
    }

    let _ = child.kill();
    let _ = child.wait();
    for ext in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{db_path}{ext}"));
    }
}
