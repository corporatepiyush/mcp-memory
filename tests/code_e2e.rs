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

pub enum Color { Red, Blue }

trait Walker {
    fn walk(&self) -> i32;
}

impl Walker for Thing {
    fn walk(&self) -> i32 {
        self.value
    }
}

pub async fn process() -> Result<(), String> {
    Ok(())
}

macro_rules! def_binop {
    ($name:ident, $op:tt) => {
        fn $name(a: i32, b: i32) -> i32 { a $op b }
    };
}
def_binop!(sub, -);

mod inner {
    pub fn nested() {}
}
"#;

const PY_SRC: &str = r#"
from typing import Optional

def greet(name: str) -> str:
    return "hi " + name

class Greeter:
    greeting: str

    def __init__(self, greeting: str = "hello") -> None:
        self.greeting = greeting

    def hello(self) -> str:
        return greet(self.greeting)

    @classmethod
    def default(cls) -> "Greeter":
        return cls()

    @staticmethod
    def version() -> str:
        return "1.0"

class AdminGreeter(Greeter):
    def hello(self) -> str:
        return "admin: " + super().hello()

def handler(prefix: str = "") -> None:
    inner = lambda x: x.strip()
    g = Greeter.default()
    print(inner(prefix + g.hello()))
"#;

const C_SRC: &str = r#"
#include <stdio.h>
#include <stdlib.h>

int add(int a, int b) {
    return a + b;
}

struct Point {
    int x;
    int y;
};

typedef struct Buffer {
    char *data;
    size_t len;
} Buffer;

union Data {
    int i;
    float f;
    char c;
};

static inline int max(int a, int b) {
    return a > b ? a : b;
}

void greet(const char *name) {
    add(1, 2);
    printf("Hello %s", name);
}

static void internal_helper(void) {
    int x = max(10, 20);
}
"#;

const CPP_SRC: &str = r#"
#include <vector>
#include <memory>

class Calculator {
public:
    Calculator() = default;
    virtual ~Calculator() = default;

    int add(int a, int b) { return a + b; }
    virtual int multiply(int a, int b);
};

class AdvancedCalc : public Calculator {
public:
    int multiply(int a, int b) override {
        return a * b;
    }

    int power(int base, int exp) {
        int r = 1;
        for (int i = 0; i < exp; i++) r *= base;
        return r;
    }
};

template<typename T>
class Vector {
public:
    void push(const T& val);
    T pop();
};

namespace util {
    int helper(int x) { return x * 2; }

    template<typename T>
    T max(T a, T b) { return a > b ? a : b; }
}
"#;

const RB_SRC: &str = r#"
module MathOps
  PI = 3.14159

  def self.square(x)
    x * x
  end
end

class Greeter
  attr_reader :name

  def initialize(name)
    @name = name
  end

  def hello
    "Hello, #{@name}"
  end

  def self.default
    new("World")
  end
end

class AdminGreeter < Greeter
  def hello
    "[ADMIN] #{super}"
  end
end

def add(a, b)
  a + b
end

def handler(items)
  items.map { |x| x * 2 }
       .select(&:even?)
       .reduce(0, :+)
end
"#;

const PHP_SRC: &str = r#"<?php

namespace App\Service;

use Psr\Log\LoggerInterface;

class UserService {
    private LoggerInterface $logger;

    public function __construct(LoggerInterface $logger) {
        $this->logger = $logger;
    }

    public function find(int $id): ?string {
        $this->logger->info("Finding user {id}", ["id" => $id]);
        return "user";
    }

    public static function createDefault(): self {
        return new self(new NullLogger());
    }
}

interface CacheInterface {
    public function get(string $key): mixed;
    public function set(string $key, mixed $value, int $ttl = 0): void;
}

trait Timestampable {
    public function getCreatedAt(): \DateTime {
        return $this->createdAt;
    }
}

function helper_sort(array &$arr): void {
    sort($arr);
}
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
    std::fs::write(src_dir.join("math.c"), C_SRC).unwrap();
    std::fs::write(src_dir.join("calc.cpp"), CPP_SRC).unwrap();
    std::fs::write(src_dir.join("greeter.rb"), RB_SRC).unwrap();
    std::fs::write(src_dir.join("service.php"), PHP_SRC).unwrap();

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
    assert_eq!(idx["files_indexed"], 6, "indexed all 6 files: {idx}");
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

    // Outline the rust fixture lists its defs with real line ranges.
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
    assert!(names.contains(&"Walker"), "outline missing Walker trait: {outline}");
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
    assert_eq!(first["files_indexed"], 6);

    // Nothing changed → everything skipped on the second run.
    let second = c.call_json("code_index", serde_json::json!({ "path": dir.clone() }));
    assert_eq!(second["files_indexed"], 0, "no re-index expected: {second}");
    assert_eq!(second["files_skipped"], 6, "all 6 skipped: {second}");

    // force re-parses regardless of hash.
    let forced = c.call_json("code_index", serde_json::json!({ "path": dir, "force": true }));
    assert_eq!(forced["files_indexed"], 6, "force reindexes: {forced}");
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

// ── New Language Integration Tests ──────────────────────────────────────

#[test]
fn code_index_c_language_e2e() {
    let mut c = setup();
    let dir = c.src_dir.to_string_lossy().to_string();

    // Index and search for a C symbol.
    let idx = c.call_json("code_index", serde_json::json!({ "path": dir }));
    assert!(idx["files_indexed"].as_u64().unwrap() >= 1);

    let res = c.call_json("code_search", serde_json::json!({ "query": "add", "lang": "c" }));
    let rows = res["results"].as_array().unwrap();
    let add = rows.iter().find(|r| r["name"].as_str().unwrap().ends_with("::add"));
    assert!(add.is_some(), "C add function should be found: {res}");
    assert_eq!(add.unwrap()["kind"], "function");

    // Outline the C file.
    let outline = c.call_json(
        "code_outline",
        serde_json::json!({ "file": file_name(&c, "math.c") }),
    );
    let names: Vec<&str> = outline["symbols"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap().rsplit("::").next().unwrap())
        .collect();
    assert!(names.contains(&"add"), "C outline missing add: {outline}");
    assert!(names.contains(&"Point"), "C outline missing Point: {outline}");
    assert!(names.contains(&"greet"), "C outline missing greet: {outline}");
    assert!(names.contains(&"Buffer"), "C outline missing Buffer typedef: {outline}");
    assert!(names.contains(&"max"), "C outline missing max inline: {outline}");
}

#[test]
fn code_index_cpp_language_e2e() {
    let mut c = setup();
    let dir = c.src_dir.to_string_lossy().to_string();

    let idx = c.call_json("code_index", serde_json::json!({ "path": dir }));
    assert!(idx["files_indexed"].as_u64().unwrap() >= 1);

    // Search for C++ class and methods.
    let res = c.call_json("code_search", serde_json::json!({ "query": "Calculator", "lang": "cpp" }));
    let rows = res["results"].as_array().unwrap();
    let calc = rows.iter().find(|r| r["name"].as_str().unwrap().ends_with("::Calculator"));
    assert!(calc.is_some(), "Calculator class should be found: {res}");
    assert_eq!(calc.unwrap()["kind"], "class");

    let outline = c.call_json(
        "code_outline",
        serde_json::json!({ "file": file_name(&c, "calc.cpp") }),
    );
    let names: Vec<&str> = outline["symbols"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap().rsplit("::").next().unwrap())
        .collect();
    assert!(names.contains(&"Calculator"), "C++ outline missing Calculator: {outline}");
    assert!(names.contains(&"add"), "C++ outline missing add method: {outline}");
    assert!(names.contains(&"multiply"), "C++ outline missing multiply: {outline}");
    assert!(names.contains(&"AdvancedCalc"), "C++ outline missing AdvancedCalc: {outline}");
    assert!(names.contains(&"power"), "C++ outline missing power: {outline}");
    assert!(names.contains(&"Vector"), "C++ outline missing Vector template: {outline}");
}

#[test]
fn code_index_ruby_language_e2e() {
    let mut c = setup();
    let dir = c.src_dir.to_string_lossy().to_string();

    let idx = c.call_json("code_index", serde_json::json!({ "path": dir }));
    assert!(idx["files_indexed"].as_u64().unwrap() >= 1);

    let res = c.call_json("code_search", serde_json::json!({ "query": "add", "lang": "ruby" }));
    let rows = res["results"].as_array().unwrap();
    let add = rows.iter().find(|r| r["name"].as_str().unwrap().ends_with("::add"));
    assert!(add.is_some(), "Ruby add method should be found: {res}");

    let outline = c.call_json(
        "code_outline",
        serde_json::json!({ "file": file_name(&c, "greeter.rb") }),
    );
    let names: Vec<&str> = outline["symbols"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap().rsplit("::").next().unwrap())
        .collect();
    assert!(names.contains(&"add"), "Ruby outline missing add: {outline}");
    assert!(names.contains(&"Greeter"), "Ruby outline missing Greeter: {outline}");
    assert!(names.contains(&"hello"), "Ruby outline missing hello: {outline}");
    assert!(names.contains(&"AdminGreeter"), "Ruby outline missing AdminGreeter: {outline}");
    assert!(names.contains(&"MathOps"), "Ruby outline missing MathOps: {outline}");
    assert!(names.contains(&"handler"), "Ruby outline missing handler: {outline}");
}

#[test]
fn code_index_php_language_e2e() {
    let mut c = setup();
    let dir = c.src_dir.to_string_lossy().to_string();

    let idx = c.call_json("code_index", serde_json::json!({ "path": dir }));
    assert!(idx["files_indexed"].as_u64().unwrap() >= 1);

    let res = c.call_json("code_search", serde_json::json!({ "query": "UserService", "lang": "php" }));
    let rows = res["results"].as_array().unwrap();
    let svc = rows.iter().find(|r| r["name"].as_str().unwrap().ends_with("::UserService"));
    assert!(svc.is_some(), "UserService class should be found: {res}");

    let outline = c.call_json(
        "code_outline",
        serde_json::json!({ "file": file_name(&c, "service.php") }),
    );
    let names: Vec<&str> = outline["symbols"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap().rsplit("::").next().unwrap())
        .collect();
    assert!(names.contains(&"UserService"), "PHP outline missing UserService: {outline}");
    assert!(names.contains(&"find"), "PHP outline missing find method: {outline}");
    assert!(names.contains(&"helper_sort"), "PHP outline missing helper_sort: {outline}");
    assert!(names.contains(&"CacheInterface"), "PHP outline missing CacheInterface: {outline}");
    assert!(names.contains(&"Timestampable"), "PHP outline missing Timestampable: {outline}");
    assert!(names.contains(&"createDefault"), "PHP outline missing createDefault: {outline}");
}

#[test]
fn code_index_filter_by_kind_and_lang() {
    let mut c = setup();
    let dir = c.src_dir.to_string_lossy().to_string();

    let _idx = c.call_json("code_index", serde_json::json!({ "path": dir }));

    // Search only for classes.
    let res = c.call_json("code_search", serde_json::json!({ "query": "a", "kind": "class" }));
    let rows = res["results"].as_array().unwrap();
    for row in rows {
        assert_eq!(row["kind"], "class", "filtered results should only be classes: {res}");
    }

    // Search only for functions in C.
    let res = c.call_json("code_search", serde_json::json!({ "query": "a", "kind": "function", "lang": "c" }));
    for row in res["results"].as_array().unwrap() {
        assert_eq!(row["kind"], "function", "should be function: {res}");
        assert_eq!(row["lang"].as_str(), Some("c"), "should be C: {res}");
    }
}

#[test]
fn code_index_outline_all_languages() {
    let mut c = setup();
    let dir = c.src_dir.to_string_lossy().to_string();

    let _idx = c.call_json("code_index", serde_json::json!({ "path": dir }));

    // Outline each file — every supported language must produce a non-empty outline.
    let files = [
        ("lib.rs", "rust"),
        ("app.py", "python"),
        ("math.c", "c"),
        ("calc.cpp", "cpp"),
        ("greeter.rb", "ruby"),
        ("service.php", "php"),
    ];
    for (leaf, _lang) in files {
        let outline = c.call_json(
            "code_outline",
            serde_json::json!({ "file": file_name(&c, leaf) }),
        );
        let n = outline["symbols"].as_array().map(|a| a.len()).unwrap_or(0);
        assert!(n > 0, "{leaf} outline should have >=1 symbol, got {outline}");
    }
}

#[test]
fn code_index_get_symbol_across_languages() {
    let mut c = setup();
    let dir = c.src_dir.to_string_lossy().to_string();

    let _idx = c.call_json("code_index", serde_json::json!({ "path": dir }));

    // get_symbol should work for each new language using unique bare names.
    for name in &["Thing", "Point", "Calculator", "UserService", "helper_sort"] {
        let result = c.call_json("code_get_symbol", serde_json::json!({ "name": name }));
        // The symbol should have kind, file, signature fields (single match).
        assert!(result.get("kind").is_some(), "get_symbol({name}) should have kind: {result}");
        assert!(result.get("file").is_some(), "get_symbol({name}) should have file: {result}");
        assert!(result.get("signature").is_some(), "get_symbol({name}) should have signature: {result}");
    }
}
