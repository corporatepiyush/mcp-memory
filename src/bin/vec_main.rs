use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, ValueEnum};
use tracing_subscriber::EnvFilter;

use mcp_memory::config::{Config, Durability};
use mcp_memory::vector_server::VectorServer;
use mcp_memory::vector_store::VectorConfig;
use mcp_memory::Transport;
use usearch::{MetricKind, ScalarKind};

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
    /// 16-bit float (half the memory, slight recall loss).
    F16,
    /// 8-bit integer quantization (quarter the memory).
    I8,
}

impl From<VecQuant> for ScalarKind {
    fn from(q: VecQuant) -> Self {
        match q {
            VecQuant::F32 => ScalarKind::F32,
            VecQuant::F16 => ScalarKind::F16,
            VecQuant::I8 => ScalarKind::I8,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "MCP Memory Vector Server")]
#[command(about = "Knowledge graph + vector search memory server for MCP — entities, relations, observations, and vector embeddings with semantic search")]
pub struct VecArgs {
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

    /// Bearer token for tcp/http transports
    #[arg(long = "auth-token")]
    pub auth_token: Option<String>,

    /// Path to a file whose trimmed contents are the bearer token
    #[arg(long = "auth-token-file")]
    pub auth_token_file: Option<String>,

    /// SQLite mmap size in bytes (default: 256 MiB)
    #[arg(long = "mmap-size", default_value_t = 268435456)]
    pub mmap_size: i64,

    /// Entity-metadata LRU cache capacity (0 = unbounded)
    #[arg(long = "lru-cache-size", default_value_t = 10000)]
    pub lru_cache_size: usize,

    /// Number of read-only SQLite connections for concurrent reads
    #[arg(long = "read-pool-size", default_value_t = 4)]
    pub read_pool_size: usize,

    /// Path to a PEM certificate chain for HTTPS
    #[arg(long = "tls-cert")]
    pub tls_cert: Option<String>,

    /// Path to the PEM private key matching --tls-cert
    #[arg(long = "tls-key")]
    pub tls_key: Option<String>,

    /// Embedding dimension for vector search (default: 384)
    #[arg(long = "embedding-dims", default_value_t = 384)]
    pub embedding_dims: u32,

    /// Distance metric for the vector index
    #[arg(long = "vec-metric", value_enum, default_value_t = VecMetric::Cos)]
    pub vec_metric: VecMetric,

    /// Scalar quantization for the vector index (lower = less memory)
    #[arg(long = "vec-quantization", value_enum, default_value_t = VecQuant::F32)]
    pub vec_quantization: VecQuant,

    /// HNSW graph degree `M` (higher = better recall, more memory)
    #[arg(long = "vec-connectivity", default_value_t = 16)]
    pub vec_connectivity: usize,

    /// HNSW `efConstruction` (higher = better index quality, slower inserts)
    #[arg(long = "vec-expansion-add", default_value_t = 200)]
    pub vec_expansion_add: usize,

    /// HNSW `efSearch` (higher = better recall, slower queries)
    #[arg(long = "vec-expansion-search", default_value_t = 50)]
    pub vec_expansion_search: usize,
}

fn main() -> ExitCode {
    let args = VecArgs::parse();

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(&args.log_level)),
        )
        .init();

    let auth_token: Option<Arc<str>> = args
        .auth_token
        .map(|t| Arc::from(t.as_str()))
        .or_else(|| {
            args.auth_token_file
                .as_ref()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .map(|s| Arc::from(s.as_str()))
        });

    let tls_cert = args.tls_cert.map(std::path::PathBuf::from);
    let tls_key = args.tls_key.map(std::path::PathBuf::from);

    let config = Config {
        memory_file_path: args
            .memory_file
            .unwrap_or_else(|| "memory.mcpmem".to_string()),
        transport: args.transport,
        bind_addr: args.bind.clone(),
        durability: Durability::Async,
        auth_token,
        mmap_size: args.mmap_size,
        lru_cache_size: args.lru_cache_size,
        read_pool_size: args.read_pool_size.max(1),
        tls_cert,
        tls_key,
    };

    let vec_config = VectorConfig {
        dims: args.embedding_dims,
        metric: args.vec_metric.into(),
        quantization: args.vec_quantization.into(),
        connectivity: args.vec_connectivity,
        expansion_add: args.vec_expansion_add,
        expansion_search: args.vec_expansion_search,
    };

    let rt = tokio::runtime::Runtime::new().unwrap();

    let result = rt.block_on(async {
        let server = VectorServer::new(config, vec_config)?;
        match args.transport {
            Transport::Stdio => server.run_stdio().await,
            Transport::Tcp => server.run_tcp(&args.bind).await,
            Transport::Http => server.run_http(&args.bind).await,
        }
    });

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!("Server error: {e:?}");
            ExitCode::FAILURE
        }
    }
}
