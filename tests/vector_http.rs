//! HTTP-transport tests for `mcp-memory-vec`, focused on bearer-token auth.
//!
//! These are regression tests for the vector server's Streamable-HTTP handler:
//! when `--auth-token` is configured, every `/mcp` request must carry a matching
//! `Authorization` header or be rejected with `401 Unauthorized`. A raw
//! `TcpStream` is used as the client so no HTTP-client dependency is needed.

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

fn spawn_http_server(auth_token: Option<&str>) -> HttpServer {
    let port = free_port();
    let pid = std::process::id();
    let db_path = format!("/tmp/vec_http_{pid}_{port}.db");
    for ext in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{db_path}{ext}"));
    }

    let bin = std::env::var("CARGO_BIN_EXE_MCP_MEMORY_VEC")
        .unwrap_or_else(|_| "target/debug/mcp-memory-vec".into());

    let mut cmd = Command::new(&bin);
    cmd.arg("-f")
        .arg(&db_path)
        .arg("--transport")
        .arg("http")
        .arg("--bind")
        .arg(format!("127.0.0.1:{port}"))
        .arg("--log-level")
        .arg("error")
        .arg("--embedding-dims")
        .arg("4")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(tok) = auth_token {
        cmd.arg("--auth-token").arg(tok);
    }

    let child = cmd.spawn().expect("failed to spawn mcp-memory-vec");

    // Wait until the server is accepting connections.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            break;
        }
        assert!(Instant::now() < deadline, "server did not start listening");
        std::thread::sleep(Duration::from_millis(50));
    }

    HttpServer {
        child,
        port,
        db_path,
    }
}

/// Send a single POST /mcp request and return (status_code, body).
fn post_mcp(port: u16, body: &str, bearer: Option<&str>) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();

    let mut req = String::new();
    req.push_str("POST /mcp HTTP/1.1\r\n");
    req.push_str("Host: 127.0.0.1\r\n");
    req.push_str("Content-Type: application/json\r\n");
    req.push_str("Accept: application/json\r\n");
    if let Some(tok) = bearer {
        req.push_str(&format!("Authorization: Bearer {tok}\r\n"));
    }
    req.push_str(&format!("Content-Length: {}\r\n", body.len()));
    req.push_str("Connection: close\r\n");
    req.push_str("\r\n");
    req.push_str(body);

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

    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();

    (status, body)
}

const INIT_BODY: &str =
    r#"{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-11-25"},"id":1}"#;

#[test]
fn test_http_no_auth_allows_requests() {
    let srv = spawn_http_server(None);
    let (status, body) = post_mcp(srv.port, INIT_BODY, None);
    assert_eq!(status, 200, "no-auth server should accept requests: {body}");
    assert!(body.contains("serverInfo"), "expected init result: {body}");
}

#[test]
fn test_http_auth_rejects_missing_token() {
    let srv = spawn_http_server(Some("s3cret"));
    let (status, _body) = post_mcp(srv.port, INIT_BODY, None);
    assert_eq!(status, 401, "request without bearer token must be rejected");
}

#[test]
fn test_http_auth_rejects_wrong_token() {
    let srv = spawn_http_server(Some("s3cret"));
    let (status, _body) = post_mcp(srv.port, INIT_BODY, Some("wrong"));
    assert_eq!(status, 401, "request with wrong bearer token must be rejected");
}

#[test]
fn test_http_auth_accepts_correct_token() {
    let srv = spawn_http_server(Some("s3cret"));
    let (status, body) = post_mcp(srv.port, INIT_BODY, Some("s3cret"));
    assert_eq!(status, 200, "correct token should be accepted: {body}");
    assert!(body.contains("serverInfo"), "expected init result: {body}");
}

#[test]
fn test_http_auth_full_tool_flow() {
    let srv = spawn_http_server(Some("s3cret"));

    // Create an entity, attach an embedding, then search — all over authed HTTP.
    let create = r#"{"jsonrpc":"2.0","method":"tools/call","params":{"name":"create_entities","arguments":{"entities":[{"name":"alice","entityType":"person","observations":["math"]}]}},"id":2}"#;
    let (status, _) = post_mcp(srv.port, create, Some("s3cret"));
    assert_eq!(status, 200, "create_entities over HTTP should succeed");

    let upsert = r#"{"jsonrpc":"2.0","method":"tools/call","params":{"name":"vector_upsert_embedding","arguments":{"entityName":"alice","embedding":[1.0,1.0,1.0,1.0]}},"id":3}"#;
    let (status, body) = post_mcp(srv.port, upsert, Some("s3cret"));
    assert_eq!(status, 200, "upsert over HTTP should succeed: {body}");

    let search = r#"{"jsonrpc":"2.0","method":"tools/call","params":{"name":"vector_search_entities","arguments":{"embedding":[1.0,1.0,1.0,1.0],"topK":5}},"id":4}"#;
    let (status, body) = post_mcp(srv.port, search, Some("s3cret"));
    assert_eq!(status, 200, "search over HTTP should succeed: {body}");
    assert!(body.contains("alice"), "search should find alice: {body}");
}
