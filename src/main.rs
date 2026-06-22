use anyhow::Result;
use clap::Parser;
use mcp_memory::{config, server};
use std::sync::Arc;
use tracing::info;

fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(inner_main())
}

async fn inner_main() -> Result<()> {
    let args = mcp_memory::Args::parse();

    // Install the rustls `ring` crypto provider as the process default up front
    // (idempotent) so the HTTPS transport can build its TLS config. See src/tls.rs.
    mcp_memory::tls::ensure_crypto_provider();

    init_tracing(&args.log_level)?;

    info!("Starting MCP Memory Server");
    info!("Version: {}", env!("CARGO_PKG_VERSION"));

    let config = Arc::new(config::Config::from_args(&args)?);
    info!("Memory file: {}", config.memory_file_path);
    info!(
        "Vector search: {}",
        if config.vectors_enabled { "enabled" } else { "disabled" }
    );
    info!(
        "Code index: {}",
        if config.code_enabled { "enabled" } else { "disabled" }
    );

    let mcp_server = server::MCPServer::new((*config).clone(), args.vector_config())?;
    info!("Server initialized successfully");

    match args.transport {
        mcp_memory::Transport::Stdio => {
            info!("Running in stdio mode");
            mcp_server.run_stdio().await?;
        }
        mcp_memory::Transport::Tcp => {
            info!("Running in TCP mode on {}", config.bind_addr);
            mcp_server.run_tcp(&config.bind_addr).await?;
        }
        mcp_memory::Transport::Http => {
            info!("Running in HTTP mode on {}", config.bind_addr);
            mcp_server.run_http(&config.bind_addr).await?;
        }
    }

    info!("Server shutdown complete");
    Ok(())
}

fn init_tracing(log_level: &str) -> Result<()> {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(log_level))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt::layer().with_writer(std::io::stderr))
        .init();

    Ok(())
}
