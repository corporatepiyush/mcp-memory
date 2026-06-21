//! `mcp-memory-vec` — a thin backward-compatible alias for the unified
//! `mcp-memory` server with vector search forced on.
//!
//! The two servers were merged: the knowledge-graph server now hosts the
//! `vector_*` / `hybrid_search` tools behind the `--vectors` flag (see
//! `src/server.rs`). This binary simply parses the same arguments and enables
//! vectors unconditionally, so existing `mcp-memory-vec` configs keep working.
//! New deployments can equivalently run `mcp-memory --vectors`.

use std::process::ExitCode;

use clap::Parser;
use tracing_subscriber::EnvFilter;

use mcp_memory::config::Config;
use mcp_memory::server::MCPServer;
use mcp_memory::{Args, Transport};

fn main() -> ExitCode {
    let mut args = Args::parse();
    // This binary is the vector-enabled entry point regardless of the flag.
    args.vectors = true;

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&args.log_level)),
        )
        .init();

    mcp_memory::tls::ensure_crypto_provider();

    let config = match Config::from_args(&args) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Configuration error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let vec_config = args.vector_config();
    let bind = config.bind_addr.clone();
    let transport = config.transport;

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            tracing::error!("Failed to start tokio runtime: {e}");
            return ExitCode::FAILURE;
        }
    };

    let result = rt.block_on(async {
        let server = MCPServer::new(config, vec_config)?;
        match transport {
            Transport::Stdio => server.run_stdio().await,
            Transport::Tcp => server.run_tcp(&bind).await,
            Transport::Http => server.run_http(&bind).await,
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
