use crate::errors::{MCSError, Result};
use crate::Transport;
use std::sync::Arc;

/// How aggressively to push WAL writes to durable storage before acknowledging
/// the client.
///
/// The default [`Async`](Durability::Async) flushes to the kernel page cache
/// and returns immediately; the background sync thread calls `fsync` within
/// ~1 second. Journal-mode filesystems (ext4, APFS, NTFS) typically absorb a
/// power loss within that window.
///
/// [`Sync`](Durability::Sync) calls `fsync` before returning, confirming the
/// data is on stable media. Use this when every write must survive an immediate
/// power failure, at the cost of higher write latency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    Async,
    Sync,
}

impl Durability {
    pub const fn is_sync(self) -> bool {
        matches!(self, Durability::Sync)
    }
}

impl std::str::FromStr for Durability {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "async" | "Async" => Ok(Durability::Async),
            "sync" | "Sync" => Ok(Durability::Sync),
            _ => Err(format!("unknown durability '{s}'; expected 'async' or 'sync'")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub memory_file_path: String,
    pub transport: Transport,
    pub bind_addr: String,
    pub durability: Durability,
    /// Optional bearer token required on the `tcp` and `http` transports. When
    /// `None`, those transports accept unauthenticated connections (stdio is
    /// always local and never authenticated).
    pub auth_token: Option<Arc<str>>,
    pub mmap_size: i64,
    pub lru_cache_size: usize,
    /// Size of the read-only connection pool (concurrent reads). Always >= 1.
    pub read_pool_size: usize,
    /// PEM certificate chain for serving the `http` transport over TLS (HTTPS).
    /// `None` (the default) keeps the transport plaintext. Engaged only when
    /// both `tls_cert` and `tls_key` are set.
    pub tls_cert: Option<std::path::PathBuf>,
    /// PEM private key matching `tls_cert`.
    pub tls_key: Option<std::path::PathBuf>,
}

impl Config {
    pub fn from_args(args: &super::Args) -> Result<Self> {
        let memory_file_path = args
            .memory_file
            .clone()
            .or_else(|| std::env::var("MEMORY_FILE_PATH").ok())
            .unwrap_or_else(|| "memory.mcpmem".to_string());

        // Resolve the auth token from --auth-token, then --auth-token-file, then
        // the MCP_MEMORY_AUTH_TOKEN env var. A configured-but-empty token file
        // is a hard error: fail closed rather than silently disabling auth.
        let auth_token: Option<Arc<str>> = if let Some(t) = args.auth_token.clone() {
            Some(Arc::from(t.as_str()))
        } else if let Some(path) = args.auth_token_file.clone() {
            let contents = std::fs::read_to_string(&path).map_err(|e| {
                MCSError::InvalidParams(format!("failed to read --auth-token-file '{path}': {e}"))
            })?;
            let token = contents.trim();
            if token.is_empty() {
                return Err(MCSError::InvalidParams(format!(
                    "--auth-token-file '{path}' is empty; refusing to start with auth disabled"
                )));
            }
            Some(Arc::from(token))
        } else {
            std::env::var("MCP_MEMORY_AUTH_TOKEN")
                .ok()
                .filter(|t| !t.is_empty())
                .map(|t| Arc::from(t.as_str()))
        };

        let durability = if let Ok(env) = std::env::var("MCP_MEMORY_DURABILITY") {
            env.parse().unwrap_or_else(|e| {
                tracing::warn!("MCP_MEMORY_DURABILITY parse failed: {e}; falling back to Async");
                Durability::Async
            })
        } else {
            Durability::Async
        };

        // TLS cert/key for the `http` transport, from CLI flags or env vars.
        // Both must be supplied together; one without the other is a hard error.
        let tls_cert = args
            .tls_cert
            .clone()
            .or_else(|| std::env::var("MCP_TLS_CERT").ok())
            .filter(|s| !s.is_empty())
            .map(std::path::PathBuf::from);
        let tls_key = args
            .tls_key
            .clone()
            .or_else(|| std::env::var("MCP_TLS_KEY").ok())
            .filter(|s| !s.is_empty())
            .map(std::path::PathBuf::from);
        if tls_cert.is_some() != tls_key.is_some() {
            return Err(MCSError::InvalidParams(
                "--tls-cert and --tls-key must be provided together (or both omitted for plaintext HTTP)"
                    .to_string(),
            ));
        }

        Ok(Config {
            memory_file_path,
            transport: args.transport,
            bind_addr: args.bind.clone(),
            durability,
            auth_token,
            mmap_size: args.mmap_size,
            lru_cache_size: args.lru_cache_size,
            read_pool_size: args.read_pool_size.max(1),
            tls_cert,
            tls_key,
        })
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            memory_file_path: "memory.mcpmem".to_string(),
            transport: Transport::Stdio,
            bind_addr: "127.0.0.1:8080".to_string(),
            durability: Durability::Async,
            auth_token: None,
            mmap_size: 268435456,
            lru_cache_size: 10000,
            read_pool_size: 4,
            tls_cert: None,
            tls_key: None,
        }
    }
}


