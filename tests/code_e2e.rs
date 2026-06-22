//! End-to-end tests for the tree-sitter code-indexing tools (`--code`).
//! Each test spins up the real server over stdio against a temp fixture tree.

#![cfg(feature = "code")]

use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

const RUST_SRC: &str = r#"
/// Adds two numbers.
pub fn alpha(x: i32) -> i32 {
    beta(x) + 1
}

fn beta(x: i32) -> i32 {
    x * 2
}

pub struct Thing {
    pub value: i32,
}
"#;

const PY_SRC: &str = r#"
def greet(name):
    return "hi " + name

class Greeter:
    def hello(self):
        return greet("world")
"#;

struct Client {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: std::process::ChildStdout,
    db_path: String,
    src_dir: std::path::PathBuf,
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        for ext in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{}", self.db_path, ext));
        }
        let _ = std::fs::remove_dir_all(&self.src_dir);
    }
}

impl Client {
    fn send(&mut self, msg: &str) {
        use std::io::Write;
        writeln!(self.stdin, "{msg}").expect("write stdin");
        self.stdin.flush().expect("flush stdin");
    }

    fn recv(&mut self) -> String {
        use std::io::{BufRead, BufReader};
        let mut buf = String::new();
        BufReader::new(&mut self.stdout)
            .read_line(&mut buf)
            .expect("read stdout");
        buf.trim().to_string()
    }

    fn call(&mut self, name: &str, args: serde_json::Value) -> serde_json::Value {
        let req = serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": { "name": name, "arguments": args }
        });
        self.send(&serde_json::to_string(&req).unwrap());
        serde_json::from_str(&self.recv()).expect("parse response")
    }

    /// Inner JSON object decoded from the tool's text content.
    fn call_json(&mut self, name: &str, args: serde_json::Value) -> serde_json::Value {
        let resp = self.call(name, args);
        let text = resp["result"]["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("missing text content: {resp}"));
        serde_json::from_str(text).expect("parse inner json")
    }
}

fn setup() -> Client {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let db_path = format!("/tmp/code_e2e_{n}.db");
    for ext in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{db_path}{ext}"));
    }
    let src_dir = std::env::temp_dir().join(format!("code_e2e_src_{n}"));
    let _ = std::fs::remove_dir_all(&src_dir);
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(src_dir.join("lib.rs"), RUST_SRC).unwrap();
    std::fs::write(src_dir.join("app.py"), PY_SRC).unwrap();

    let bin = std::env::var("CARGO_BIN_EXE_MCP_MEMORY")
        .unwrap_or_else(|_| "target/debug/mcp-memory".into());
    let mut child = Command::new(&bin)
        .args(["-f", &db_path, "--code", "--transport", "stdio", "--log-level", "error"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn mcp-memory");
    Client {
        stdin: child.stdin.take().unwrap(),
        stdout: child.stdout.take().unwrap(),
        child,
        db_path,
        src_dir,
    }
}

/// The stored entity name for a fixture file (absolute path, forward slashes).
fn file_name(c: &Client, leaf: &str) -> String {
    c.src_dir.join(leaf).to_string_lossy().replace('\\', "/")
}

#[test]
fn code_index_then_search_get_outline() {
    let mut c = setup();
    let dir = c.src_dir.to_string_lossy().to_string();

    // Index the fixture tree.
    let idx = c.call_json("code_index", serde_json::json!({ "path": dir }));
    assert_eq!(idx["files_indexed"], 2, "indexed both files: {idx}");
    assert!(idx["symbols"].as_u64().unwrap() >= 5, "expected >=5 symbols: {idx}");

    // Search finds a symbol with location + signature.
    let res = c.call_json("code_search", serde_json::json!({ "query": "alpha" }));
    let rows = res["results"].as_array().unwrap();
    let alpha = rows
        .iter()
        .find(|r| r["name"].as_str().unwrap().ends_with("::alpha"))
        .unwrap_or_else(|| panic!("alpha not in results: {res}"));
    assert_eq!(alpha["kind"], "function");
    assert!(alpha["signature"].as_str().unwrap().contains("fn alpha"));

    // get_symbol: beta is called by alpha (caller edge resolved).
    let beta = c.call_json("code_get_symbol", serde_json::json!({ "name": "beta" }));
    let callers = beta["callers"].as_array().unwrap();
    assert!(
        callers.iter().any(|c| c.as_str().unwrap().ends_with("::alpha")),
        "alpha should call beta: {beta}"
    );

    // Outline of the rust fixture lists its defs with real line ranges.
    let outline = c.call_json(
        "code_outline",
        serde_json::json!({ "file": file_name(&c, "lib.rs") }),
    );
    let names: Vec<&str> = outline["symbols"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap().rsplit("::").next().unwrap())
        .collect();
    assert!(names.contains(&"alpha"), "outline missing alpha: {outline}");
    assert!(names.contains(&"beta"), "outline missing beta: {outline}");
    assert!(names.contains(&"Thing"), "outline missing Thing: {outline}");
    // Thing's range should span more than one line (struct body).
    let thing = outline["symbols"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["name"].as_str().unwrap().ends_with("::Thing"))
        .unwrap();
    let lines = thing["lines"].as_str().unwrap();
    let (a, b) = lines.split_once('-').unwrap();
    assert!(b.parse::<u32>().unwrap() > a.parse::<u32>().unwrap(), "multi-line span: {lines}");
}

#[test]
fn code_index_is_incremental() {
    let mut c = setup();
    let dir = c.src_dir.to_string_lossy().to_string();

    let first = c.call_json("code_index", serde_json::json!({ "path": dir.clone() }));
    assert_eq!(first["files_indexed"], 2);

    // Nothing changed → everything skipped on the second run.
    let second = c.call_json("code_index", serde_json::json!({ "path": dir.clone() }));
    assert_eq!(second["files_indexed"], 0, "no re-index expected: {second}");
    assert_eq!(second["files_skipped"], 2, "both skipped: {second}");

    // force re-parses regardless of hash.
    let forced = c.call_json("code_index", serde_json::json!({ "path": dir, "force": true }));
    assert_eq!(forced["files_indexed"], 2, "force reindexes: {forced}");
}

#[test]
fn code_tools_present_by_default() {
    // Code tools are advertised by default (when code feature is on).
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let db_path = format!("/tmp/code_e2e_default_{n}.db");
    for ext in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{db_path}{ext}"));
    }
    let bin = std::env::var("CARGO_BIN_EXE_MCP_MEMORY")
        .unwrap_or_else(|_| "target/debug/mcp-memory".into());
    let mut child = Command::new(&bin)
        .args(["-f", &db_path, "--transport", "stdio", "--log-level", "error"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    {
        use std::io::{BufRead, BufReader, Write};
        let mut stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        writeln!(stdin, "{}", r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#).unwrap();
        stdin.flush().unwrap();
        let mut buf = String::new();
        BufReader::new(stdout).read_line(&mut buf).unwrap();
        let resp: serde_json::Value = serde_json::from_str(buf.trim()).unwrap();
        let names: Vec<String> = resp["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect();
        assert!(
            names.iter().any(|n| n == "code_index"),
            "code_index should be advertised by default: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "code_search"),
            "code_search should be advertised by default: {names:?}"
        );
    }
    let _ = child.kill();
    let _ = child.wait();
    for ext in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{db_path}{ext}"));
    }
}
