pub mod actions;
pub mod config;
pub mod errors;
pub mod http;
pub mod kg;
pub mod protocol;
pub mod server;
pub mod tools;
pub mod types;

use clap::{Parser, ValueEnum};

/// Wire transport the server listens on. The JSON-RPC/MCP semantics are
/// identical across all three — only the framing differs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "lowercase")]
pub enum Transport {
    /// Newline-delimited JSON-RPC over stdin/stdout (default; for Claude
    /// Desktop / Claude Code and other process-spawning clients).
    Stdio,
    /// Newline-delimited JSON-RPC over a TCP socket (one message per line),
    /// accepting many concurrent connections.
    Tcp,
    /// MCP Streamable HTTP: POST JSON-RPC to `/mcp` (responses as JSON or, when
    /// the client `Accept`s it, an SSE stream), plus `GET /mcp` for a standalone
    /// server→client SSE stream.
    Http,
}

#[derive(Parser, Debug)]
#[command(name = "MCP Memory Server")]
#[command(about = "Knowledge graph memory server for MCP — entities, relations, and observations persisted in SQLite with FTS5 search", long_about = None)]
pub struct Args {
    /// Path to the memory file
    #[arg(short = 'f', long = "memory-file")]
    pub memory_file: Option<String>,

    /// Transport to listen on: stdio, tcp, or http
    #[arg(short = 't', long = "transport", value_enum, default_value_t = Transport::Stdio)]
    pub transport: Transport,

    /// Address to bind for the `tcp` and `http` transports
    #[arg(short = 'b', long = "bind", default_value = "127.0.0.1:8080")]
    pub bind: String,

    /// Log level
    #[arg(short, long, default_value = "info")]
    pub log_level: String,

    /// Bearer token required on the `tcp` (first line) and `http`
    /// (`Authorization` header) transports. Overrides `--auth-token-file` and
    /// the `MCP_MEMORY_AUTH_TOKEN` env var. stdio is never authenticated.
    #[arg(long = "auth-token")]
    pub auth_token: Option<String>,

    /// Path to a file whose trimmed contents are the bearer token. An empty
    /// file is rejected (fail closed). Ignored if `--auth-token` is set.
    #[arg(long = "auth-token-file")]
    pub auth_token_file: Option<String>,

    /// SQLite mmap size in bytes (default: 256 MiB).
    #[arg(long = "mmap-size", default_value_t = 268435456)]
    pub mmap_size: i64,

    /// Entity-metadata LRU cache capacity (0 = unbounded).
    #[arg(long = "lru-cache-size", default_value_t = 10000)]
    pub lru_cache_size: usize,

    /// Number of read-only SQLite connections backing concurrent reads. WAL
    /// mode allows readers to run in parallel with each other and the single
    /// writer; a larger pool raises read concurrency at the cost of a little
    /// memory. Clamped to at least 1.
    #[arg(long = "read-pool-size", default_value_t = 4)]
    pub read_pool_size: usize,
}
