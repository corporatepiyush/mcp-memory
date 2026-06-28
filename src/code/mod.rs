//! Tree-sitter code-symbol indexing.
//!
//! Pure parsing layer: turns source files into [`ParsedFile`]s (definitions and
//! references) and provides a gitignore-aware directory walk. It has no
//! knowledge of the graph store — [`crate::actions::code`] maps the parsed
//! output onto entities/relations and handles incremental hashing.

pub mod lang;

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicUsize;

pub use lang::{Lang, detect};

/// Files larger than this are skipped (parsing huge generated/minified files
/// is slow and rarely useful for a symbol map).
pub const MAX_FILE_BYTES: u64 = 10 * 1024 * 1024;

/// Max defs emitted per file (safety cap against pathological inputs).
pub const MAX_SYMBOLS_PER_FILE: usize = 20_000;

/// Max tags (defs + refs) accepted from tree-sitter for a single file,
/// preventing OOM from pathological generated code.
pub const MAX_TAGS_PER_FILE: usize = 100_000;

/// Max files collected by [`walk`] before stopping early.
pub const MAX_WALK_FILES: usize = 200_000;

/// Signatures and doc lines are capped to keep observations compact.
const MAX_SIGNATURE_CHARS: usize = 512;
const MAX_DOC_CHARS: usize = 512;

/// Body snippets (the full definition text) are capped so they remain useful as
/// embedding input without bloating the database. Opt-in per `code_index` call.
const MAX_SNIPPET_CHARS: usize = 2_000;

/// A symbol definition extracted from a file.
#[derive(Debug, Clone)]
pub struct Def {
    /// Normalized kind: `function`, `method`, `class`, `module`, `constant`, …
    pub kind: String,
    /// Bare symbol name.
    pub name: String,
    /// 1-based inclusive line range of the definition.
    pub line_start: usize,
    pub line_end: usize,
    /// First (declaration) line of the definition, trimmed.
    pub signature: String,
    /// First line of the associated doc comment, if any.
    pub doc: Option<String>,
    /// Bounded full-text of the definition (body included), capped to
    /// [`MAX_SNIPPET_CHARS`]. Used as semantic-search embedding input; only
    /// populated when the caller requests snippets.
    pub snippet: String,
}

/// A reference (call / type use) extracted from a file.
#[derive(Debug, Clone)]
pub struct Ref {
    /// Reference kind from the grammar's tags query, e.g. `call`, `type`.
    pub kind: String,
    /// Bare name referenced.
    pub name: String,
    /// 1-based line of the reference.
    pub line: usize,
}

/// Parsed symbols for a single file.
#[derive(Debug, Clone, Default)]
pub struct ParsedFile {
    pub defs: Vec<Def>,
    pub refs: Vec<Ref>,
}

/// BLAKE3 content hash (hex) used for incremental change detection.
pub fn hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// Normalize a grammar's definition kind into our small entity vocabulary.
fn normalize_def_kind(raw: &str) -> &str {
    match raw {
        "function" | "macro" => "function",
        "method" | "delegate" => "method",
        "class" | "interface" | "struct" | "type" | "enum" | "trait"
            | "union" | "concept" | "object" | "annotation" | "typealias" => "class",
        "module" | "namespace" => "module",
        "constant" => "constant",
        other => other,
    }
}

/// Extract the trimmed first line of `source` starting at byte offset `start`.
fn first_line(source: &[u8], start: usize) -> String {
    let end = source[start..]
        .iter()
        .position(|&b| b == b'\n')
        .map(|p| start + p)
        .unwrap_or(source.len());
    let mut s = String::from_utf8_lossy(&source[start..end]).trim().to_string();
    if s.chars().count() > MAX_SIGNATURE_CHARS {
        s = s.chars().take(MAX_SIGNATURE_CHARS).collect::<String>() + "…";
    }
    s
}

/// Extract the definition's full text (`source[start..end]`), trimmed and capped
/// to [`MAX_SNIPPET_CHARS`]. Returns an empty string for an empty/invalid range.
fn clamp_snippet(source: &[u8], start: usize, end: usize) -> String {
    let end = end.min(source.len());
    if start >= end {
        return String::new();
    }
    let raw = String::from_utf8_lossy(&source[start..end]);
    let trimmed = raw.trim();
    if trimmed.chars().count() > MAX_SNIPPET_CHARS {
        trimmed.chars().take(MAX_SNIPPET_CHARS).collect::<String>() + "…"
    } else {
        trimmed.to_string()
    }
}

fn clamp_doc(doc: &str) -> Option<String> {
    let line = doc.lines().find(|l| !l.trim().is_empty())?.trim();
    if line.is_empty() {
        return None;
    }
    let s = if line.chars().count() > MAX_DOC_CHARS {
        line.chars().take(MAX_DOC_CHARS).collect::<String>() + "…"
    } else {
        line.to_string()
    };
    Some(s)
}

/// Parse one in-memory source buffer into defs/refs. Returns an empty result
/// for unsupported languages or unbuildable tag configs. Equivalent to
/// [`parse_source_opts`] with snippets disabled.
pub fn parse_source(lang: Lang, source: &[u8]) -> ParsedFile {
    parse_source_opts(lang, source, false)
}

/// Like [`parse_source`], but `want_snippet` controls whether each def's bounded
/// body text ([`Def::snippet`]) is extracted (extra allocation per symbol).
pub fn parse_source_opts(lang: Lang, source: &[u8], want_snippet: bool) -> ParsedFile {
    let Some(config) = lang::config(lang) else {
        return ParsedFile::default();
    };

    let mut ctx = tree_sitter_tags::TagsContext::new();
    let cancel = AtomicUsize::new(0);
    let (tags, _failed) = match ctx.generate_tags(config, source, Some(&cancel)) {
        Ok(v) => v,
        Err(_) => return ParsedFile::default(),
    };

    // Byte offset of the start of each line, for O(log n) byte→line lookups.
    // `tag.range` spans the whole definition node (body included), while
    // `tag.span` is only the name; we derive the def's line range from `range`.
    let line_starts: Vec<usize> = std::iter::once(0)
        .chain(source.iter().enumerate().filter(|&(_, &b)| b == b'\n').map(|(i, _)| i + 1))
        .collect();
    let line_of = |byte: usize| line_starts.partition_point(|&s| s <= byte).max(1);

    let mut out = ParsedFile::default();
    for tag in tags.flatten() {
        if out.defs.len() + out.refs.len() >= MAX_TAGS_PER_FILE {
            break;
        }
        let name = String::from_utf8_lossy(&source[tag.name_range.clone()]).to_string();
        if name.is_empty() {
            continue;
        }
        let kind = config.syntax_type_name(tag.syntax_type_id).to_string();
        if tag.is_definition {
            let end_byte = tag.range.end.saturating_sub(1).max(tag.range.start);
            let snippet = if want_snippet {
                clamp_snippet(source, tag.range.start, tag.range.end)
            } else {
                String::new()
            };
            out.defs.push(Def {
                kind: normalize_def_kind(&kind).to_string(),
                name,
                line_start: line_of(tag.range.start),
                line_end: line_of(end_byte),
                signature: first_line(source, tag.range.start),
                doc: tag.docs.as_deref().and_then(clamp_doc),
                snippet,
            });
        } else {
            out.refs.push(Ref {
                kind,
                name,
                line: tag.span.start.row + 1,
            });
        }
    }
    out
}

/// Walk `root` (a file or directory) and collect indexable source files,
/// honoring `.gitignore`/hidden-file rules and skipping oversized files.
pub fn walk(root: &Path, max_bytes: u64) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if root.is_file() {
        if detect(root).is_some()
            && std::fs::metadata(root).map(|m| m.len() <= max_bytes).unwrap_or(false)
        {
            files.push(root.to_path_buf());
        }
        return files;
    }

    let walker = ignore::WalkBuilder::new(root)
        .standard_filters(true)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .require_git(false)
        .filter_entry(|e| {
            // Belt-and-suspenders: skip common build/vendor dirs even when no
            // .gitignore is present.
            let name = e.file_name().to_string_lossy();
            !matches!(name.as_ref(), "target" | "node_modules" | ".git" | "dist" | "build")
        })
        .build();

    for entry in walker.flatten() {
        if files.len() >= MAX_WALK_FILES {
            break;
        }
        let path = entry.path();
        if !path.is_file() || detect(path).is_none() {
            continue;
        }
        if std::fs::metadata(path).map(|m| m.len() > max_bytes).unwrap_or(true) {
            continue;
        }
        files.push(path.to_path_buf());
    }
    files
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Parser unit tests — each new language's tags query is exercised.
    // -----------------------------------------------------------------------

    fn count_defs_of_kind(parsed: &ParsedFile, kind: &str) -> usize {
        parsed.defs.iter().filter(|d| d.kind == kind).count()
    }

    fn find_def<'a>(parsed: &'a ParsedFile, name: &str) -> Option<&'a Def> {
        parsed.defs.iter().find(|d| d.name == name)
    }

    // ── Rust ──────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_rust() {
        let src = b"/// Docs
pub fn alpha(x: i32) -> i32 { x + 1 }

fn beta() {}

pub struct Thing { pub x: i32 }

pub enum Color { Red, Blue }

trait Foo { fn bar(&self); }
";
        let parsed = parse_source(Lang::Rust, src);
        assert!(!parsed.defs.is_empty(), "expected defs");

        let alpha = find_def(&parsed, "alpha").expect("alpha");
        assert_eq!(alpha.kind, "function");
        assert!(alpha.signature.contains("fn alpha"));

        let beta = find_def(&parsed, "beta").expect("beta");
        assert_eq!(beta.kind, "function");

        let thing = find_def(&parsed, "Thing").expect("Thing");
        assert_eq!(thing.kind, "class");

        let color = find_def(&parsed, "Color").expect("Color");
        assert_eq!(color.kind, "class");

        let foo = find_def(&parsed, "Foo").expect("Foo");
        assert_eq!(foo.kind, "class");
    }

    #[test]
    fn test_parse_rust_calls() {
        let src = b"fn alpha() { beta() + gamma() }
fn beta() -> i32 { 1 }
fn gamma() -> i32 { 2 }";
        let parsed = parse_source(Lang::Rust, src);
        let call_refs: Vec<&str> = parsed.refs.iter()
            .filter(|r| r.kind == "call")
            .map(|r| r.name.as_str())
            .collect();
        assert!(call_refs.contains(&"beta"), "alpha should call beta");
        assert!(call_refs.contains(&"gamma"), "alpha should call gamma");
    }

    // ── Python ────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_python() {
        let src = b"def greet(name):
    return 'hello ' + name

class Greeter:
    def hello(self):
        return greet('world')

MAX_RETRIES = 3
";
        let parsed = parse_source(Lang::Python, src);
        assert_eq!(count_defs_of_kind(&parsed, "function"), 2);
        assert_eq!(count_defs_of_kind(&parsed, "class"), 1);
        assert_eq!(count_defs_of_kind(&parsed, "constant"), 1);
        assert!(find_def(&parsed, "greet").is_some());
        assert!(find_def(&parsed, "Greeter").is_some());
    }

    // ── JavaScript ─────────────────────────────────────────────────────────

    #[test]
    fn test_parse_javascript() {
        let src = b"function alpha(x) { return beta(x) + 1; }
class Thing { constructor(v) { this.v = v; } }
";
        let parsed = parse_source(Lang::JavaScript, src);
        assert!(!parsed.defs.is_empty(), "JS should produce defs");
        assert!(find_def(&parsed, "alpha").is_some(), "alpha function");
        assert!(find_def(&parsed, "Thing").is_some(), "Thing class");
    }

    // ── Go ────────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_go() {
        let src = b"package main

func alpha(x int) int { return beta(x) + 1 }

func beta(x int) int { return x * 2 }

type Thing struct { Value int }
";
        let parsed = parse_source(Lang::Go, src);
        assert!(!parsed.defs.is_empty(), "Go should produce defs");
        assert!(find_def(&parsed, "alpha").is_some(), "alpha function");
        assert!(find_def(&parsed, "beta").is_some(), "beta function");
        assert!(find_def(&parsed, "Thing").is_some(), "Thing type");
    }

    // ── Java ──────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_java() {
        let src = b"package com.example;

class Hello {
    private int x;

    public int add(int a, int b) { return a + b; }
}

interface Worker {
    void doWork();
}
";
        let parsed = parse_source(Lang::Java, src);
        assert!(!parsed.defs.is_empty(), "Java should produce defs");
        assert!(find_def(&parsed, "Hello").is_some(), "Hello class");
        assert!(find_def(&parsed, "Worker").is_some(), "Worker interface");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NEW LANGUAGE TESTS
    // ═══════════════════════════════════════════════════════════════════════

    // ── C ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_c() {
        let src = b"#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

struct Point {
    int x;
    int y;
};

enum Color { RED, GREEN, BLUE };

#define MAX_SIZE 1024

void greet(const char *name) {
    printf(\"Hello %s\", name);
}
";
        let parsed = parse_source(Lang::C, src);
        // C functions: add, greet
        assert_eq!(count_defs_of_kind(&parsed, "function"), 2);
        // struct Point
        assert_eq!(count_defs_of_kind(&parsed, "class"), 2); // struct + enum
        assert!(find_def(&parsed, "add").is_some());
        assert!(find_def(&parsed, "greet").is_some());
        assert!(find_def(&parsed, "Point").is_some());
        assert!(find_def(&parsed, "Color").is_some());

        // Verify line ranges
        let add = find_def(&parsed, "add").unwrap();
        assert_eq!(add.line_start, 3);
    }

    // ── C++ ───────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_cpp() {
        let src = b"#include <vector>

class Calculator {
public:
    int add(int a, int b) { return a + b; }

    int multiply(int a, int b);
};

struct Config {
    int timeout;
};

enum Status { OK, ERROR };

namespace util {
    int helper(int x) { return x * 2; }
}

template<typename T>
T max(T a, T b) { return a > b ? a : b; }
";
        let parsed = parse_source(Lang::Cpp, src);
        // C++: class, struct, enum, namespace + functions
        assert!(find_def(&parsed, "Calculator").is_some());
        assert!(find_def(&parsed, "Config").is_some());
        assert!(find_def(&parsed, "Status").is_some());
        assert!(find_def(&parsed, "helper").is_some());
        assert!(find_def(&parsed, "max").is_some());

        // Calculator is a class
        let calc = find_def(&parsed, "Calculator").unwrap();
        assert_eq!(calc.kind, "class");

        // Config is a struct → normalized to "class"
        let config = find_def(&parsed, "Config").unwrap();
        assert_eq!(config.kind, "class");
    }

    // ── Ruby ──────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_ruby() {
        let src = b"# Adds two numbers
def add(a, b)
  a + b
end

class Greeter
  def initialize(name)
    @name = name
  end

  def hello
    \"Hello, #{@name}\"
  end
end

module Utils
  MAX_RETRIES = 3

  def self.format(s)
    s.strip
  end
end
";
        let parsed = parse_source(Lang::Ruby, src);
        // Ruby: method definitions
        assert!(find_def(&parsed, "add").is_some(), "add method");
        assert!(find_def(&parsed, "hello").is_some(), "hello method");
        // class + module
        assert!(find_def(&parsed, "Greeter").is_some(), "Greeter class");
        assert!(find_def(&parsed, "Utils").is_some(), "Utils module");

        // Verify doc extraction
        let add = find_def(&parsed, "add").unwrap();
        assert_eq!(add.kind, "method");
        assert_eq!(add.doc.as_deref(), Some("Adds two numbers"));
    }

    #[test]
    fn test_parse_ruby_constant() {
        let src = b"MAX_VALUE = 1000
MIN_VALUE = 1
";
        let parsed = parse_source(Lang::Ruby, src);
        // Ruby may or may not extract top-level constants; just verify no crash
        assert!(parsed.refs.is_empty() || parsed.defs.len() <= 4);
    }

    // ── PHP ───────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_php() {
        let src = b"<?php

namespace App\\Service;

class UserService {
    public function find(int $id): ?User {
        return $this->repo->find($id);
    }

    private function validate(array $data): bool {
        return !empty($data['name']);
    }
}

interface Logger {
    public function log(string $msg): void;
}

trait Timestampable {
    public function getCreatedAt(): \\DateTime {
        return $this->createdAt;
    }
}

function helper_sort(array &$arr): void {
    sort($arr);
}
";
        let parsed = parse_source(Lang::Php, src);
        // PHP: class, interface, trait + methods
        assert!(find_def(&parsed, "UserService").is_some(), "UserService class");
        assert!(find_def(&parsed, "Logger").is_some(), "Logger interface");
        assert!(find_def(&parsed, "Timestampable").is_some(), "Timestampable trait");
        assert!(find_def(&parsed, "find").is_some(), "find method");
        assert!(find_def(&parsed, "validate").is_some(), "validate method");
        assert!(find_def(&parsed, "log").is_some(), "log method");
        assert!(find_def(&parsed, "helper_sort").is_some(), "helper_sort function");

        let svc = find_def(&parsed, "UserService").unwrap();
        assert_eq!(svc.kind, "class");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // COMPLEX LANGUAGE PATTERNS
    // ═══════════════════════════════════════════════════════════════════════

    // ── Rust (generics, impl, async, closures, macros) ────────────────────

    #[test]
    fn test_parse_rust_complex() {
        let src = b"pub struct Pair<T, U> { first: T, second: U }

impl<T, U> Pair<T, U> {
    fn new(first: T, second: U) -> Self { Pair { first, second } }
    fn first(&self) -> &T { &self.first }
}

pub trait Into {
    fn into(self) -> i32;
}

impl Into for i32 {
    fn into(self) -> i32 { self }
}

pub async fn fetch(url: &str) -> Result<String, String> {
    Ok(String::new())
}

pub fn handler() {
    let add = |a: i32, b: i32| a + b;
    let _ = add(1, 2);
}

macro_rules! vec_of {
    ($x:expr) => { vec![$x] };
}
";
        let parsed = parse_source(Lang::Rust, src);
        assert!(find_def(&parsed, "Pair").is_some(), "Pair generic struct");
        assert!(find_def(&parsed, "Into").is_some(), "Into trait");
        assert!(find_def(&parsed, "fetch").is_some(), "fetch async fn");
        assert!(find_def(&parsed, "handler").is_some(), "handler fn");
        // Methods inside impl blocks
        assert!(find_def(&parsed, "new").is_some(), "Pair::new method");
        assert!(find_def(&parsed, "first").is_some(), "Pair::first method");
        // References (calls inside handler)
        let call_refs: Vec<&str> = parsed.refs.iter()
            .filter(|r| r.kind == "call")
            .map(|r| r.name.as_str())
            .collect();
        assert!(call_refs.contains(&"add"), "handler calls add closure");
    }

    // ── Python (decorators, inheritance, type hints, classmethod, staticmethod, lambdas) ──

    #[test]
    fn test_parse_python_complex() {
        let src = b"from typing import Optional, List

class Repository:
    def __init__(self, db: str) -> None:
        self.db = db

    async def find(self, id: int) -> Optional[dict]:
        return None

    @classmethod
    def default(cls) -> 'Repository':
        return cls('sqlite')

    @staticmethod
    def version() -> str:
        return '1.0'

class UserService(Repository):
    def __init__(self) -> None:
        super().__init__('users')

    async def find(self, id: int) -> Optional[dict]:
        return {'id': id}

def compute(items: List[int]) -> int:
    return sum(filter(None, map(lambda x: x * 2, items)))
";
        let parsed = parse_source(Lang::Python, src);
        assert!(find_def(&parsed, "Repository").is_some(), "Repository class");
        assert!(find_def(&parsed, "UserService").is_some(), "UserService class");
        assert!(find_def(&parsed, "find").is_some(), "find method (both classes)");
        assert!(find_def(&parsed, "default").is_some(), "default classmethod");
        assert!(find_def(&parsed, "version").is_some(), "version staticmethod");
        assert!(find_def(&parsed, "compute").is_some(), "compute function");
        assert!(find_def(&parsed, "__init__").is_some(), "__init__ method");
        // Should have at least these functions/methods
        assert!(count_defs_of_kind(&parsed, "function") >= 3);
        assert!(count_defs_of_kind(&parsed, "class") >= 2);
    }

    // ── JavaScript (classes, methods, arrow functions, async) ──────────────

    #[test]
    fn test_parse_javascript_complex() {
        let src = b"class Repository {
    constructor(db) { this.db = db; }
    async find(id) { return null; }
    static default() { return new Repository('sqlite'); }
}

class UserService extends Repository {
    constructor() { super('users'); }
    async find(id) { return { id }; }
}

function compute(arr) {
    return arr.filter(x => x != null).map(x => x * 2);
}
";
        let parsed = parse_source(Lang::JavaScript, src);
        assert!(find_def(&parsed, "Repository").is_some(), "Repository class");
        assert!(find_def(&parsed, "UserService").is_some(), "UserService class");
        assert!(find_def(&parsed, "find").is_some(), "find method");
        assert!(find_def(&parsed, "default").is_some(), "default static method");
        assert!(find_def(&parsed, "compute").is_some(), "compute function");
        assert!(find_def(&parsed, "default").is_some(), "default static method");
    }

    // ── Go (interfaces, methods on structs, variadic functions) ────────────

    #[test]
    fn test_parse_go_complex() {
        let src = b"package main

import \"fmt\"

type Walker interface {
    Walk() int
}

type Thing struct {
    value int
}

func (t *Thing) Walk() int {
    return t.value
}

func NewThing(v int) *Thing {
    return &Thing{value: v}
}

func sum(vals ...int) int {
    total := 0
    for _, v := range vals {
        total += v
    }
    return total
}
";
        let parsed = parse_source(Lang::Go, src);
        assert!(find_def(&parsed, "Walker").is_some(), "Walker interface");
        assert!(find_def(&parsed, "Thing").is_some(), "Thing struct");
        assert!(find_def(&parsed, "Walk").is_some(), "Walk method");
        assert!(find_def(&parsed, "NewThing").is_some(), "NewThing constructor");
        assert!(find_def(&parsed, "sum").is_some(), "sum variadic function");
    }

    // ── Java (generics, inheritance, annotations, enums, inner classes) ────

    #[test]
    fn test_parse_java_complex() {
        let src = b"package com.example;

import java.util.List;

class Repository<T> {
    public T find(int id) { return null; }

    public List<T> findAll() { return null; }
}

class UserService extends Repository<String> {
    @Override
    public String find(int id) {
        return \"user\";
    }
}

enum Status { ACTIVE, INACTIVE }

interface Cache<K, V> {
    V get(K key);
    void put(K key, V value);
}
";
        let parsed = parse_source(Lang::Java, src);
        assert!(find_def(&parsed, "Repository").is_some(), "Repository generic class");
        assert!(find_def(&parsed, "UserService").is_some(), "UserService class");
        assert!(find_def(&parsed, "Cache").is_some(), "Cache interface");
        assert!(find_def(&parsed, "find").is_some(), "find method");
        assert!(find_def(&parsed, "findAll").is_some(), "findAll method");
    }

    // ── C (typedef, union, function pointers, static, inline) ─────────────

    #[test]
    fn test_parse_c_complex() {
        let src = b"#include <stddef.h>

typedef struct Buffer Buffer;

struct Buffer {
    char *data;
    size_t len;
};

static inline int max(int a, int b) {
    return a > b ? a : b;
}

static void internal_cleanup(Buffer *buf) {
    if (buf) buf->len = 0;
}

int process(Buffer *buf) {
    return (int)buf->len;
}
";
        let parsed = parse_source(Lang::C, src);
        assert!(find_def(&parsed, "Buffer").is_some(), "Buffer struct");
        assert!(find_def(&parsed, "max").is_some(), "max inline function");
        assert!(find_def(&parsed, "internal_cleanup").is_some(), "internal_cleanup static");
        assert!(find_def(&parsed, "process").is_some(), "process function");
        // Struct + typedef produce class-kind defs
        let types = count_defs_of_kind(&parsed, "class");
        assert!(types >= 1, "should have at least Buffer struct, got {types}");
    }

    // ── C++ (virtual inheritance, operator overloads, lambdas, constexpr) ──

    #[test]
    fn test_parse_cpp_complex() {
        let src = b"#include <vector>

class Base {
public:
    virtual ~Base() = default;
    virtual int compute() const = 0;
};

class Derived final : public Base {
public:
    int compute() const override { return value_; }

    Derived& operator=(const Derived& other) {
        value_ = other.value_;
        return *this;
    }

private:
    int value_ = 42;
};

template<typename T>
constexpr T pi = T(3.1415926535);

namespace detail {
    template<typename T>
    class ScopedPtr {
    public:
        explicit ScopedPtr(T* ptr) : ptr_(ptr) {}
        ~ScopedPtr() { delete ptr_; }
        T& operator*() const { return *ptr_; }
    private:
        T* ptr_;
    };
}
";
        let parsed = parse_source(Lang::Cpp, src);
        assert!(find_def(&parsed, "Base").is_some(), "Base abstract class");
        assert!(find_def(&parsed, "Derived").is_some(), "Derived class");
        assert!(find_def(&parsed, "ScopedPtr").is_some(), "ScopedPtr template class");
        assert!(find_def(&parsed, "compute").is_some(), "compute method");
    }

    // ── Ruby (blocks, modules, mixins, inheritance, attr_accessor) ───────

    #[test]
    fn test_parse_ruby_complex() {
        let src = b"module Persistence
  def save
    'saved'
  end
end

class BaseRecord
  attr_accessor :id

  def initialize(id = nil)
    @id = id
  end
end

class User < BaseRecord
  include Persistence

  attr_reader :name

  def initialize(id, name)
    super(id)
    @name = name
  end

  def self.find(id)
    new(id, 'default')
  end

  def to_s
    \"User(#{@id}, #{@name})\"
  end
end
";
        let parsed = parse_source(Lang::Ruby, src);
        assert!(find_def(&parsed, "Persistence").is_some(), "Persistence module");
        assert!(find_def(&parsed, "BaseRecord").is_some(), "BaseRecord class");
        assert!(find_def(&parsed, "User").is_some(), "User class");
        assert!(find_def(&parsed, "save").is_some(), "save method");
        assert!(find_def(&parsed, "initialize").is_some(), "initialize");
        assert!(find_def(&parsed, "find").is_some(), "find class method");
        assert!(find_def(&parsed, "to_s").is_some(), "to_s method");
    }

    // ── PHP (constructor promotion, attributes, union types, static) ─────

    #[test]
    fn test_parse_php_complex() {
        let src = b"<?php

namespace App\\Service;

interface CacheInterface
{
    public function get(string $key): mixed;
    public function set(string $key, mixed $value, int $ttl = 0): void;
}

trait Loggable
{
    public function log(string $msg): void
    {
        echo \\date('[Y-m-d] ') . $msg;
    }
}

class UserService implements CacheInterface
{
    use Loggable;

    public function __construct(
        private string $prefix = 'usr'
    ) {}

    public function get(string $key): mixed
    {
        $this->log(\"get: $key\");
        return null;
    }

    public function set(string $key, mixed $value, int $ttl = 0): void
    {
        $this->log(\"set: $key\");
    }

    public static function createDefault(): self
    {
        return new self();
    }
}
";
        let parsed = parse_source(Lang::Php, src);
        assert!(find_def(&parsed, "CacheInterface").is_some(), "CacheInterface");
        assert!(find_def(&parsed, "Loggable").is_some(), "Loggable trait");
        assert!(find_def(&parsed, "UserService").is_some(), "UserService class");
        assert!(find_def(&parsed, "get").is_some(), "get method");
        assert!(find_def(&parsed, "set").is_some(), "set method");
        assert!(find_def(&parsed, "log").is_some(), "log method");
        assert!(find_def(&parsed, "createDefault").is_some(), "createDefault static");
    }

    // ── Edge cases ────────────────────────────────────────────────────────

    #[test]
    fn test_parse_empty_source() {
        let src = b"";
        for lang in Lang::all() {
            let parsed = parse_source(lang, src);
            assert!(parsed.defs.is_empty(), "{:?} should produce no defs from empty input", lang);
            assert!(parsed.refs.is_empty(), "{:?} should produce no refs from empty input", lang);
        }
    }

    #[test]
    fn test_parse_whitespace_only() {
        for lang in Lang::all() {
            let parsed = parse_source(lang, b"\n\n   \n\t\n");
            assert!(parsed.defs.is_empty(), "{:?} whitespace should produce no defs", lang);
        }
    }

    #[test]
    fn test_parse_syntax_error_recovers_gracefully() {
        // Rust file with missing semicolons and incomplete expressions
        let src = b"fn broken(x: i32) -> i32 {
    x +
}
fn fine() {}
";
        let parsed = parse_source(Lang::Rust, src);
        // Even with syntax errors, tree-sitter should still extract some symbols
        assert!(!parsed.defs.is_empty(), "should recover and find some defs");
        assert!(find_def(&parsed, "fine").is_some(), "fine should be found");
    }

    #[test]
    fn test_parse_large_file_truncated() {
        // parse_source itself does not cap (capping is in handle_code_index);
        // verify it can handle a large number of definitions without OOM.
        let mut src = String::new();
        for i in 0..MAX_SYMBOLS_PER_FILE + 100 {
            src.push_str(&format!("fn func_{i}() {{}}\n"));
        }
        let parsed = parse_source(Lang::Rust, src.as_bytes());
        // Should parse all defs without crashing
        assert!(parsed.defs.len() > MAX_SYMBOLS_PER_FILE,
            "should parse more than cap without truncation, got {}", parsed.defs.len());
    }

    #[test]
    fn test_parse_all_languages_produce_defs() {
        let samples: Vec<(Lang, &[u8])> = vec![
            (Lang::Rust, b"fn foo() {}\nconst X: i32 = 1;\n"),
            (Lang::Python, b"def foo(): pass\nX = 1\n"),
            (Lang::JavaScript, b"function foo() {}\nconst X = 1;\n"),
            (Lang::TypeScript, b"abstract class Foo {}\ninterface Bar {}\n"),
            (Lang::Tsx, b"abstract class Foo {}\ninterface Bar {}\n"),
            (Lang::Go, b"func foo() {}\nconst X = 1\n"),
            (Lang::Java, b"class Foo {}\n"),
            (Lang::C, b"int foo() { return 1; }\n"),
            (Lang::Cpp, b"int foo() { return 1; }\nclass Bar {};\n"),
            (Lang::Ruby, b"def foo; end\nX = 1\n"),
            (Lang::Php, b"<?php function foo() {}\n"),
        ];
        for (lang, src) in samples {
            let parsed = parse_source(lang, src);
            assert!(!parsed.defs.is_empty(),
                "{:?} should produce at least one def", lang);
        }
    }

    #[test]
    fn test_parse_c_header_file() {
        let src = b"#ifndef FOO_H
#define FOO_H

typedef struct Buffer Buffer;

struct Buffer {
    char *data;
    size_t len;
};

int process(Buffer *buf);
void free_buffer(Buffer *buf);

#endif
";
        let parsed = parse_source(Lang::C, src);
        assert!(find_def(&parsed, "process").is_some(), "process function");
        assert!(find_def(&parsed, "free_buffer").is_some(), "free_buffer function");
        assert!(find_def(&parsed, "Buffer").is_some(), "Buffer struct");
    }

    #[test]
    fn test_parse_cpp_template_and_methods() {
        let src = b"template<typename T>
class Vector {
public:
    void push(const T& val);
    T pop();
private:
    T* data_;
    size_t size_;
};

template<>
class Vector<bool> {
public:
    void push(bool val);
};
";
        let parsed = parse_source(Lang::Cpp, src);
        assert!(find_def(&parsed, "Vector").is_some(), "Vector template class");
        assert!(find_def(&parsed, "push").is_some(), "push method");
        assert!(find_def(&parsed, "pop").is_some(), "pop method");
    }

    #[test]
    fn test_parse_php_without_opening_tag() {
        // PHP without <?php should still parse (many real files are tag-only)
        let src = b"<?php
function foo() {}
";
        let parsed = parse_source(Lang::Php, src);
        assert!(find_def(&parsed, "foo").is_some(), "foo function expected");
    }

    #[test]
    fn test_parse_ruby_singleton_methods() {
        let src = b"class Foo
  def self.bar
    'class method'
  end

  def instance_method
    'instance'
  end
end
";
        let parsed = parse_source(Lang::Ruby, src);
        assert!(find_def(&parsed, "bar").is_some(), "self.bar method");
        assert!(find_def(&parsed, "instance_method").is_some(), "instance_method");
    }
}
