pub mod actions;
#[cfg(feature = "code")]
pub mod code;
#[cfg(feature = "code")]
pub mod code_registry;
pub mod config;
pub mod errors;
pub mod http;
pub mod ivf;
pub mod kg;
pub mod protocol;
pub mod server;
pub mod tls;
pub mod tools;
pub mod types;
pub mod watcher;
pub mod vector_actions;
pub mod vector_store;

use clap::{Parser, ValueEnum};
use usearch::{MetricKind, ScalarKind};
use vector_store::{IndexKind, VectorConfig};

/// ANN index backend selectable from the CLI.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum VecIndex {
    /// usearch HNSW graph index (default): best recall/latency.
    Hnsw,
    /// IVF-Flat: k-means partitioned, lighter memory, fast to build/rebuild.
    Ivf,
}

impl From<VecIndex> for IndexKind {
    fn from(v: VecIndex) -> Self {
        match v {
            VecIndex::Hnsw => IndexKind::Hnsw,
            VecIndex::Ivf => IndexKind::Ivf,
        }
    }
}

/// Distance metric for the vector index.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum VecMetric {
    /// Cosine similarity (default; good for normalized embeddings).
    Cos,
    /// Inner / dot product.
    Ip,
    /// Squared Euclidean (L2) distance.
    L2sq,
}

impl From<VecMetric> for MetricKind {
    fn from(m: VecMetric) -> Self {
        match m {
            VecMetric::Cos => MetricKind::Cos,
            VecMetric::Ip => MetricKind::IP,
            VecMetric::L2sq => MetricKind::L2sq,
        }
    }
}

/// Scalar representation stored in the index (lower precision = less memory).
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum VecQuant {
    /// 32-bit float (default; full precision).
    F32,
    /// 16-bit half-precision IEEE float (half the memory, slight recall loss).
    F16,
    /// 16-bit brain float (half the memory, wider range than f16).
    Bf16,
    /// 8-bit integer quantization (quarter the memory).
    I8,
}

impl From<VecQuant> for ScalarKind {
    fn from(q: VecQuant) -> Self {
        match q {
            VecQuant::F32 => ScalarKind::F32,
            VecQuant::F16 => ScalarKind::F16,
            VecQuant::Bf16 => ScalarKind::BF16,
            VecQuant::I8 => ScalarKind::I8,
        }
    }
}

/// Wire transport the server listens on. The JSON-RPC/MCP semantics are
/// identical across all three — only the framing differs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "lowercase")]
pub enum Transport {
    /// Newline-delimited JSON-RPC over stdin/stdout (default; for Claude
    /// Desktop / Claude Code and other process-spawning clients).
    Stdio,
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

    /// Transport to listen on: stdio or http
    #[arg(short = 't', long = "transport", value_enum, default_value_t = Transport::Stdio)]
    pub transport: Transport,

    /// Address to bind for the `http` transport
    #[arg(short = 'b', long = "bind", default_value = "127.0.0.1:8080")]
    pub bind: String,

    /// Log level
    #[arg(short, long, default_value = "info")]
    pub log_level: String,

    /// Bearer token required on the `http` (`Authorization` header) transport.
    /// Overrides `--auth-token-file` and the `MCP_MEMORY_AUTH_TOKEN` env var.
    /// stdio is never authenticated.
    #[arg(long = "auth-token")]
    pub auth_token: Option<String>,

    /// Path to a file whose trimmed contents are the bearer token. An empty
    /// file is rejected (fail closed). Ignored if `--auth-token` is set.
    #[arg(long = "auth-token-file")]
    pub auth_token_file: Option<String>,

    /// SQLite mmap size in bytes (default: 64 MiB).
    #[arg(long = "mmap-size", default_value_t = 67108864)]
    pub mmap_size: i64,

    /// SQLite page size in bytes; power of two (default: 4096, matches the Linux
    /// page / filesystem block size). Only applies to a freshly-created database.
    #[arg(long = "page-size", default_value_t = 4096)]
    pub page_size: i64,

    /// SQLite page cache size in MiB (default: 32).
    #[arg(long = "cache-size-mb", default_value_t = 32)]
    pub cache_size_mb: i64,

    /// SQLite busy timeout in milliseconds (default: 5000).
    #[arg(long = "busy-timeout-ms", default_value_t = 5000)]
    pub busy_timeout_ms: u64,

    /// Interval in milliseconds for a background `wal_checkpoint(PASSIVE)` that
    /// bounds the durability window in async mode (default: 500). 0 disables it.
    #[arg(long = "wal-flush-ms", default_value_t = 500)]
    pub wal_flush_ms: u64,

    /// Entity-metadata LRU cache capacity (0 falls back to 10000).
    #[arg(long = "lru-cache-size", default_value_t = 10000)]
    pub lru_cache_size: usize,

    /// Number of read-only SQLite connections backing concurrent reads. WAL
    /// mode allows readers to run in parallel with each other and the single
    /// writer; a larger pool raises read concurrency at the cost of a little
    /// memory (each connection carries its own page cache). `0` (default)
    /// auto-scales to the CPU count, clamped to [1, 32].
    #[arg(long = "read-pool-size", default_value_t = 4)]
    pub read_pool_size: usize,

    /// Path to a PEM certificate chain to serve the `http` transport over TLS
    /// (HTTPS). Requires --tls-key. Falls back to the MCP_TLS_CERT env var.
    /// When unset, the `http` transport stays plaintext.
    #[arg(long = "tls-cert")]
    pub tls_cert: Option<String>,

    /// Path to the PEM private key matching --tls-cert. Falls back to the
    /// MCP_TLS_KEY env var.
    #[arg(long = "tls-key")]
    pub tls_key: Option<String>,

    // ── Tool exposure ────────────────────────────────────────────────────
    // No tools are exposed unless explicitly enabled. Each flag turns on one
    // category (hidden from tools/list and rejected from tools/call when its
    // category is disabled). Use --enable-all for every category at once.
    /// Expose ALL tool categories (overrides the individual --enable-* flags).
    #[arg(long = "enable-all", default_value_t = false)]
    pub enable_all: bool,

    /// Enable read-only knowledge-graph tools (queries, traversal, export).
    #[arg(long = "enable-graph-read", default_value_t = false)]
    pub enable_graph_read: bool,

    /// Enable knowledge-graph mutation tools (create/delete/merge/compact/upsert).
    #[arg(long = "enable-graph-write", default_value_t = false)]
    pub enable_graph_write: bool,

    /// Enable vector / semantic search: the `vector_*` and `hybrid_search` tools
    /// backed by a usearch index. The `--embedding-dims` / `--vec-*` flags only
    /// take effect when this is set.
    #[arg(long = "enable-vectors", default_value_t = false)]
    pub enable_vectors: bool,

    /// Enable tree-sitter code-symbol indexing: the `code_*` tools that parse
    /// source files and store symbols (and call/define edges) in the graph.
    /// Only effective when built with the `code` feature (on by default).
    #[arg(long = "enable-code", default_value_t = false)]
    pub enable_code: bool,

    /// Embedding dimension for vector search (default: 384). Requires --enable-vectors.
    #[arg(long = "embedding-dims", default_value_t = 384)]
    pub embedding_dims: u32,

    /// Distance metric for the vector index. Requires --enable-vectors.
    #[arg(long = "vec-metric", value_enum, default_value_t = VecMetric::Cos)]
    pub vec_metric: VecMetric,

    /// Scalar quantization for the vector index (lower = less memory). Requires --enable-vectors.
    #[arg(long = "vec-quantization", value_enum, default_value_t = VecQuant::F32)]
    pub vec_quantization: VecQuant,

    /// HNSW graph degree `M` (higher = better recall, more memory). Requires --enable-vectors.
    #[arg(long = "vec-connectivity", default_value_t = 16)]
    pub vec_connectivity: usize,

    /// HNSW `efConstruction` (higher = better index quality, slower inserts). Requires --enable-vectors.
    #[arg(long = "vec-expansion-add", default_value_t = 200)]
    pub vec_expansion_add: usize,

    /// HNSW `efSearch` (higher = better recall, slower queries). Requires --enable-vectors.
    #[arg(long = "vec-expansion-search", default_value_t = 50)]
    pub vec_expansion_search: usize,

    /// ANN index backend: `hnsw` (default) or `ivf` (IVF-Flat). Requires --enable-vectors.
    #[arg(long = "vec-index", value_enum, default_value_t = VecIndex::Hnsw)]
    pub vec_index: VecIndex,

    /// IVF: number of Voronoi cells / centroids (default: 256). Requires --vec-index ivf.
    #[arg(long = "ivf-nlist", default_value_t = 256)]
    pub ivf_nlist: usize,

    /// IVF: cells probed per query — higher = better recall, slower (default: 8).
    /// Requires --vec-index ivf.
    #[arg(long = "ivf-nprobe", default_value_t = 8)]
    pub ivf_nprobe: usize,
}

impl Args {
    /// Resolve the set of enabled tool categories from the `--enable-*` flags.
    /// `--enable-all` turns on every category; otherwise only the categories
    /// whose individual flag is set. With no flags, the result is empty and no
    /// tools are exposed.
    pub fn enabled_categories(&self) -> Vec<tools::ToolCategory> {
        use tools::ToolCategory as C;
        if self.enable_all {
            return C::ALL.to_vec();
        }
        let mut cats = Vec::new();
        let mut push = |on: bool, cat: C| {
            if on {
                cats.push(cat);
            }
        };
        push(self.enable_graph_read, C::GraphRead);
        push(self.enable_graph_write, C::GraphWrite);
        push(self.enable_vectors, C::Vectors);
        push(self.enable_code, C::Code);
        cats
    }

    /// Build the vector index configuration from the `--embedding-dims` /
    /// `--vec-*` / `--ivf-*` flags. Only meaningful when `--enable-vectors` is set.
    pub fn vector_config(&self) -> VectorConfig {
        VectorConfig {
            dims: self.embedding_dims,
            index_kind: self.vec_index.into(),
            metric: self.vec_metric.into(),
            quantization: self.vec_quantization.into(),
            connectivity: self.vec_connectivity,
            expansion_add: self.vec_expansion_add,
            expansion_search: self.vec_expansion_search,
            ivf_nlist: self.ivf_nlist,
            ivf_nprobe: self.ivf_nprobe,
        }
    }
}
