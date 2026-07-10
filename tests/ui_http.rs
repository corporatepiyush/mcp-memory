//! HTTP-transport tests for the browser knowledge-graph viewer (`GET /ui` and
//! `GET /ui/graph`). These exercise the routes added alongside the MCP handlers:
//! the self-contained viewer shell, the JSON data endpoint it renders, and the
//! two gates on that data — the `graph-read` permission and the bearer token.
//!
//! A raw `TcpStream` is the client (no HTTP-client dependency), matching
//! `tests/vector_http.rs`.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct HttpServer {
    child: Child,
    port: u16,
    db_path: String,
}

impl Drop for HttpServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        for ext in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{}", self.db_path, ext));
        }
    }
}

/// Grab a currently-free localhost port by binding to :0 and releasing it.
fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    l.local_addr().unwrap().port()
}

/// Spawn a server with the given `--enable-*` category flags and optional token.
fn spawn_http_server(enable_args: &[&str], auth_token: Option<&str>) -> HttpServer {
    let port = free_port();
    let pid = std::process::id();
    let db_path = format!("/tmp/ui_http_{pid}_{port}.db");
    for ext in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{db_path}{ext}"));
    }

    let bin = std::env::var("CARGO_BIN_EXE_MCP_MEMORY")
        .unwrap_or_else(|_| "target/debug/mcp-memory".into());

    let mut cmd = Command::new(&bin);
    cmd.arg("-f")
        .arg(&db_path)
        .arg("--transport")
        .arg("http")
        .arg("--bind")
        .arg(format!("127.0.0.1:{port}"))
        .arg("--log-level")
        .arg("error")
        .args(enable_args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(tok) = auth_token {
        cmd.arg("--auth-token").arg(tok);
    }

    let child = cmd.spawn().expect("failed to spawn mcp-memory");

    // Wait until the HTTP stack is actually serving, not merely until the port
    // is bound: `GET /ui` needs no auth or permission, so a 200 from it means
    // `axum::serve` is accepting and dispatching (avoids a first-request race).
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if try_request(port, "GET", "/ui", None, None).is_some_and(|r| r.0 == 200) {
            break;
        }
        assert!(Instant::now() < deadline, "server did not start serving");
        std::thread::sleep(Duration::from_millis(50));
    }

    HttpServer {
        child,
        port,
        db_path,
    }
}

/// Send one HTTP request over a fresh connection; return (status, headers, body).
/// Panics if the connection is refused (use [`try_request`] to tolerate that).
fn request(
    port: u16,
    method: &str,
    path: &str,
    json_body: Option<&str>,
    bearer: Option<&str>,
) -> (u16, String, String) {
    try_request(port, method, path, json_body, bearer).expect("connect")
}

/// Like [`request`], but returns `None` if the connection is refused — used by
/// the startup health check, where refusals are expected until the server binds.
fn try_request(
    port: u16,
    method: &str,
    path: &str,
    json_body: Option<&str>,
    bearer: Option<&str>,
) -> Option<(u16, String, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();

    let mut req = format!("{method} {path} HTTP/1.1\r\n");
    req.push_str("Host: 127.0.0.1\r\n");
    req.push_str("Accept: application/json, text/html\r\n");
    if let Some(tok) = bearer {
        req.push_str(&format!("Authorization: Bearer {tok}\r\n"));
    }
    if let Some(body) = json_body {
        req.push_str("Content-Type: application/json\r\n");
        req.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    req.push_str("Connection: close\r\n\r\n");
    if let Some(body) = json_body {
        req.push_str(body);
    }

    stream.write_all(req.as_bytes()).expect("write request");
    stream.flush().unwrap();

    let mut raw = Vec::new();
    let _ = stream.read_to_end(&mut raw);
    let text = String::from_utf8_lossy(&raw).to_string();

    let status = text
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok())
        .unwrap_or(0);

    let (headers, body) = text
        .split_once("\r\n\r\n")
        .map(|(h, b)| (h.to_string(), b.to_string()))
        .unwrap_or_default();

    Some((status, headers, body))
}

fn get(port: u16, path: &str, bearer: Option<&str>) -> (u16, String, String) {
    request(port, "GET", path, None, bearer)
}

/// Populate a tiny graph over authed/unauthed HTTP so `/ui/graph` has content.
fn seed_graph(port: u16, bearer: Option<&str>) {
    let create = r#"{"jsonrpc":"2.0","method":"tools/call","params":{"name":"create_entities","arguments":{"entities":[{"name":"Alice","entityType":"person","observations":["likes hiking"]},{"name":"Acme","entityType":"company","observations":[]}]}},"id":2}"#;
    let (status, _, _) = request(port, "POST", "/mcp", Some(create), bearer);
    assert_eq!(status, 200, "seed create_entities should succeed");

    let rel = r#"{"jsonrpc":"2.0","method":"tools/call","params":{"name":"create_relations","arguments":{"relations":[{"from":"Alice","to":"Acme","relationType":"works_at"}]}},"id":3}"#;
    let (status, _, _) = request(port, "POST", "/mcp", Some(rel), bearer);
    assert_eq!(status, 200, "seed create_relations should succeed");
}

#[test]
fn test_ui_shell_served_as_html() {
    let srv = spawn_http_server(&["--enable-all"], None);
    let (status, headers, body) = get(srv.port, "/ui", None);
    assert_eq!(status, 200, "GET /ui should return the viewer page");
    assert!(
        headers.to_lowercase().contains("content-type: text/html"),
        "viewer must be served as HTML, headers: {headers}"
    );
    assert!(body.contains("<title>"), "expected an HTML document: {body:.120}");
    // The shell references its split CSS/JS assets by absolute path.
    assert!(body.contains("/ui/graph.css"), "shell should link the stylesheet");
    assert!(body.contains("/ui/graph.js"), "shell should load the script");
}

#[test]
fn test_ui_assets_served_with_content_types() {
    let srv = spawn_http_server(&["--enable-all"], None);

    let (status, headers, body) = get(srv.port, "/ui/graph.css", None);
    assert_eq!(status, 200, "GET /ui/graph.css should succeed");
    assert!(
        headers.to_lowercase().contains("content-type: text/css"),
        "CSS must be served as text/css, headers: {headers}"
    );
    assert!(!body.is_empty(), "stylesheet should not be empty");

    let (status, headers, body) = get(srv.port, "/ui/graph.js", None);
    assert_eq!(status, 200, "GET /ui/graph.js should succeed");
    assert!(
        headers.to_lowercase().contains("javascript"),
        "JS must be served with a javascript content-type, headers: {headers}"
    );
    // The script drives traversal via the /ui/expand endpoint.
    assert!(body.contains("/ui/expand"), "viewer should call /ui/expand to traverse");
}

#[test]
fn test_ui_expand_returns_neighborhood() {
    let srv = spawn_http_server(&["--enable-all"], None);
    seed_graph(srv.port, None);

    // Expanding Alice must return her plus her neighbour Acme and the edge.
    let (status, headers, body) = get(srv.port, "/ui/expand?name=Alice", None);
    assert_eq!(status, 200, "GET /ui/expand should succeed: {body}");
    assert!(
        headers.to_lowercase().contains("content-type: application/json"),
        "expand data must be JSON, headers: {headers}"
    );
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let names: Vec<&str> = v["entities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"Alice") && names.contains(&"Acme"), "got {names:?}");
    assert!(v["relations"].as_array().unwrap().iter().any(|r| r["relationType"] == "works_at"));
}

#[test]
fn test_ui_expand_unknown_entity_is_404() {
    let srv = spawn_http_server(&["--enable-all"], None);
    let (status, _, _) = get(srv.port, "/ui/expand?name=DoesNotExist", None);
    assert_eq!(status, 404, "expanding a missing entity should be 404");
}

#[test]
fn test_ui_expand_requires_name_and_permission() {
    // Missing name → 400 (client error).
    let srv = spawn_http_server(&["--enable-all"], None);
    let (status, _, _) = get(srv.port, "/ui/expand", None);
    assert_eq!(status, 400, "expand without a name should be 400");
    drop(srv);

    // Read disabled → 403, same gate as /ui/graph.
    let srv = spawn_http_server(&["--enable-graph-write"], None);
    let (status, _, _) = get(srv.port, "/ui/expand?name=Alice", None);
    assert_eq!(status, 403, "graph-read disabled must forbid /ui/expand");
}

#[test]
fn test_ui_graph_returns_entities_and_relations() {
    let srv = spawn_http_server(&["--enable-all"], None);
    seed_graph(srv.port, None);

    let (status, headers, body) = get(srv.port, "/ui/graph", None);
    assert_eq!(status, 200, "GET /ui/graph should succeed: {body}");
    assert!(
        headers.to_lowercase().contains("content-type: application/json"),
        "graph data must be JSON, headers: {headers}"
    );
    let v: serde_json::Value = serde_json::from_str(&body).expect("valid JSON payload");
    assert!(v["entities"].as_array().unwrap().iter().any(|e| e["name"] == "Alice"));
    assert!(v["relations"].as_array().unwrap().iter().any(|r| r["relationType"] == "works_at"));
    // Legend + stats are injected by the handler on top of read_graph's shape.
    assert!(v["entityTypes"].as_array().unwrap().iter().any(|t| t["type"] == "person"));
    assert_eq!(v["stats"]["entities"], 2);
    assert_eq!(v["stats"]["relations"], 1);
    // Pagination cursor drives the viewer's Prev/Next controls.
    assert_eq!(v["page"]["offset"], 0);
    assert_eq!(v["page"]["returned"], 2);
    assert_eq!(v["page"]["hasMore"], false);
}

#[test]
fn test_ui_graph_entity_type_filter() {
    let srv = spawn_http_server(&["--enable-all"], None);
    seed_graph(srv.port, None);

    let (status, _, body) = get(srv.port, "/ui/graph?entityType=company", None);
    assert_eq!(status, 200, "filtered graph should succeed: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let names: Vec<&str> = v["entities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["Acme"], "filter should return only company entities");
}

#[test]
fn test_ui_graph_requires_graph_read() {
    // Write enabled but read disabled: the viewer's data endpoint is forbidden.
    let srv = spawn_http_server(&["--enable-graph-write"], None);
    let (status, _, body) = get(srv.port, "/ui/graph", None);
    assert_eq!(status, 403, "graph-read disabled must forbid /ui/graph");
    assert!(
        body.contains("graph-read"),
        "403 body should explain the missing permission: {body}"
    );
}

#[test]
fn test_ui_graph_auth_gate() {
    let srv = spawn_http_server(&["--enable-all"], Some("s3cret"));
    seed_graph(srv.port, Some("s3cret"));

    // No credentials → 401.
    let (status, _, _) = get(srv.port, "/ui/graph", None);
    assert_eq!(status, 401, "missing token must be rejected");

    // Wrong token via query → 401.
    let (status, _, _) = get(srv.port, "/ui/graph?token=nope", None);
    assert_eq!(status, 401, "wrong token must be rejected");

    // Correct token via the ?token= query fallback → 200.
    let (status, _, body) = get(srv.port, "/ui/graph?token=s3cret", None);
    assert_eq!(status, 200, "query-param token should be accepted: {body}");
    assert!(body.contains("Alice"), "authed graph should have data: {body}");

    // Correct token via the Authorization header → 200.
    let (status, _, _) = get(srv.port, "/ui/graph", Some("s3cret"));
    assert_eq!(status, 200, "bearer header token should be accepted");

    // The shell itself carries no data, so it is reachable without a token.
    let (status, _, _) = get(srv.port, "/ui", None);
    assert_eq!(status, 200, "the /ui shell should not require auth");
}

/// Create `n` entities named `person_0000`..`person_(n-1)` in one batch.
fn seed_many(port: u16, n: usize) {
    let ents: Vec<String> = (0..n)
        .map(|i| format!(r#"{{"name":"person_{i:04}","entityType":"person","observations":["note {i}"]}}"#))
        .collect();
    let body = format!(
        r#"{{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"create_entities","arguments":{{"entities":[{}]}}}},"id":9}}"#,
        ents.join(",")
    );
    let (status, _, _) = request(port, "POST", "/mcp", Some(&body), None);
    assert_eq!(status, 200, "seed_many should succeed");
}

#[test]
fn test_ui_graph_pagination_cursor() {
    let srv = spawn_http_server(&["--enable-all"], None);
    seed_many(srv.port, 25);

    // First page: 10 of 25, more to come.
    let (_, _, body) = get(srv.port, "/ui/graph?limit=10&offset=0", None);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["entities"].as_array().unwrap().len(), 10);
    assert_eq!(v["page"]["offset"], 0);
    assert_eq!(v["page"]["returned"], 10);
    assert_eq!(v["page"]["hasMore"], true);
    assert_eq!(v["stats"]["entities"], 25);

    // Last page: offset 20 leaves 5, no more.
    let (_, _, body) = get(srv.port, "/ui/graph?limit=10&offset=20", None);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["entities"].as_array().unwrap().len(), 5);
    assert_eq!(v["page"]["hasMore"], false);
}

#[test]
fn test_ui_search_paginated_nodes_only() {
    let srv = spawn_http_server(&["--enable-all"], None);
    seed_many(srv.port, 25);

    let (status, headers, body) = get(srv.port, "/ui/search?q=person&limit=10&offset=0", None);
    assert_eq!(status, 200, "search should succeed: {body}");
    assert!(
        headers.to_lowercase().contains("content-type: application/json"),
        "search data must be JSON, headers: {headers}"
    );
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["entities"].as_array().unwrap().len(), 10, "first search page");
    assert_eq!(v["page"]["hasMore"], true, "25 matches → more pages");
    // Search returns matched nodes only; the user expands for relationships.
    assert_eq!(v["relations"].as_array().unwrap().len(), 0);

    // Second page paginates the same query.
    let (_, _, body) = get(srv.port, "/ui/search?q=person&limit=10&offset=20", None);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["entities"].as_array().unwrap().len(), 5);
    assert_eq!(v["page"]["hasMore"], false);
}

#[test]
fn test_ui_search_prefix_and_permission() {
    let srv = spawn_http_server(&["--enable-all"], None);
    seed_graph(srv.port, None); // Alice (person), Acme (company)

    // A prefix ("Ac") matches "Acme" — search-as-you-type behaviour.
    let (status, _, body) = get(srv.port, "/ui/search?q=Ac", None);
    assert_eq!(status, 200);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let names: Vec<&str> = v["entities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"Acme"), "prefix search should find Acme, got {names:?}");
    drop(srv);

    // Same graph-read gate as the rest of the viewer.
    let srv = spawn_http_server(&["--enable-graph-write"], None);
    let (status, _, _) = get(srv.port, "/ui/search?q=x", None);
    assert_eq!(status, 403, "search must require graph-read");
}

#[test]
fn test_ui_graph_omits_observation_bodies() {
    let srv = spawn_http_server(&["--enable-all"], None);
    seed_graph(srv.port, None); // Alice has one observation ("likes hiking")

    let (status, _, body) = get(srv.port, "/ui/graph", None);
    assert_eq!(status, 200, "graph should succeed: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let alice = v["entities"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["name"] == "Alice")
        .expect("Alice present");
    // The list payload carries a count, not the bodies — those lazy-load via /ui/node.
    assert_eq!(alice["obsCount"], 1, "Alice's observation count should be present");
    assert!(
        alice.get("observations").is_none(),
        "list payload must omit observation bodies: {alice}"
    );
}

#[test]
fn test_ui_node_lazy_loads_observations() {
    let srv = spawn_http_server(&["--enable-all"], None);
    seed_graph(srv.port, None);

    let (status, headers, body) = get(srv.port, "/ui/node?name=Alice", None);
    assert_eq!(status, 200, "node fetch should succeed: {body}");
    assert!(
        headers.to_lowercase().contains("content-type: application/json"),
        "node data must be JSON, headers: {headers}"
    );
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["name"], "Alice");
    assert_eq!(v["entityType"], "person");
    // The single-node endpoint carries the full observation bodies.
    let obs: Vec<&str> = v["observations"].as_array().unwrap().iter().map(|o| o.as_str().unwrap()).collect();
    assert_eq!(obs, vec!["likes hiking"], "node fetch should return observation bodies");

    // Unknown entity → 404.
    let (status, _, _) = get(srv.port, "/ui/node?name=DoesNotExist", None);
    assert_eq!(status, 404, "unknown entity should be 404");
}

#[test]
fn test_ui_node_requires_graph_read() {
    let srv = spawn_http_server(&["--enable-graph-write"], None);
    let (status, _, _) = get(srv.port, "/ui/node?name=Alice", None);
    assert_eq!(status, 403, "graph-read disabled must forbid /ui/node");
}
