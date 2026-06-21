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

/// Tunable SQLite pragmas applied when opening the database. `page_size` and
/// `auto_vacuum` only take effect on a freshly-created database (they are fixed
/// once the file has content); the rest apply on every open. Defaults target a
/// Linux host (4 KiB pages match the OS page / filesystem block size).
#[derive(Debug, Clone, Copy)]
pub struct SqliteTuning {
    /// `PRAGMA mmap_size` in bytes.
    pub mmap_size: i64,
    /// `PRAGMA page_size` in bytes (fresh DB only). Must be a power of two.
    pub page_size: i64,
    /// `PRAGMA cache_size` magnitude in KiB (applied as the negative form).
    pub cache_size_kb: i64,
    /// `PRAGMA busy_timeout` in milliseconds.
    pub busy_timeout_ms: u64,
    /// `PRAGMA journal_size_limit` in bytes.
    pub journal_size_limit: i64,
}

impl Default for SqliteTuning {
    fn default() -> Self {
        Self {
            mmap_size: 268_435_456,          // 256 MiB
            page_size: 4096,                 // 4 KiB — matches Linux page/fs block
            cache_size_kb: 50_000,           // ~50 MiB
            busy_timeout_ms: 5000,
            journal_size_limit: 134_217_728, // 128 MiB
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
    /// `PRAGMA page_size` in bytes (fresh DB only).
    pub page_size: i64,
    /// `PRAGMA cache_size` magnitude in KiB.
    pub cache_size_kb: i64,
    /// `PRAGMA busy_timeout` in milliseconds.
    pub busy_timeout_ms: u64,
    /// Interval in milliseconds for the background `wal_checkpoint(PASSIVE)`
    /// flush. `0` disables it (rely on SQLite auto-checkpoint + maintenance).
    pub wal_flush_ms: u64,
    pub lru_cache_size: usize,
    /// Size of the read-only connection pool (concurrent reads). Always >= 1.
    pub read_pool_size: usize,
    /// PEM certificate chain for serving the `http` transport over TLS (HTTPS).
    /// `None` (the default) keeps the transport plaintext. Engaged only when
    /// both `tls_cert` and `tls_key` are set.
    pub tls_cert: Option<std::path::PathBuf>,
    /// PEM private key matching `tls_cert`.
    pub tls_key: Option<std::path::PathBuf>,
    /// Enable the vector / semantic-search subsystem (`vector_*` + `hybrid_search`
    /// tools backed by a usearch HNSW index). Off by default.
    pub vectors_enabled: bool,
}

/// Resolve the read-only connection-pool size. `0` means "auto": scale to the
/// number of available CPUs (clamped to `[1, 32]` so a many-core host doesn't
/// open an unreasonable number of connections, each carrying its own page
/// cache). Any explicit value is honoured but floored at 1.
pub fn resolve_read_pool_size(requested: usize) -> usize {
    if requested == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 32)
    } else {
        requested.max(1)
    }
}

impl Config {
    /// Build the SQLite pragma tuning from this config, keeping the fixed
    /// `journal_size_limit` default.
    pub fn sqlite_tuning(&self) -> SqliteTuning {
        SqliteTuning {
            mmap_size: self.mmap_size,
            page_size: self.page_size,
            cache_size_kb: self.cache_size_kb,
            busy_timeout_ms: self.busy_timeout_ms,
            ..SqliteTuning::default()
        }
    }

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
            page_size: args.page_size,
            cache_size_kb: args.cache_size_mb.saturating_mul(1024),
            busy_timeout_ms: args.busy_timeout_ms,
            wal_flush_ms: args.wal_flush_ms,
            lru_cache_size: args.lru_cache_size,
            read_pool_size: resolve_read_pool_size(args.read_pool_size),
            tls_cert,
            tls_key,
            vectors_enabled: args.vectors,
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
            page_size: SqliteTuning::default().page_size,
            cache_size_kb: SqliteTuning::default().cache_size_kb,
            busy_timeout_ms: SqliteTuning::default().busy_timeout_ms,
            wal_flush_ms: 250,
            lru_cache_size: 10000,
            read_pool_size: 4,
            tls_cert: None,
            tls_key: None,
            vectors_enabled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_read_pool_size_auto_scales_within_bounds() {
        let auto = resolve_read_pool_size(0);
        assert!((1..=32).contains(&auto), "auto pool {auto} out of [1,32]");
        assert_eq!(
            auto,
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 32)
        );
    }

    #[test]
    fn test_resolve_read_pool_size_honours_explicit_values() {
        assert_eq!(resolve_read_pool_size(1), 1);
        assert_eq!(resolve_read_pool_size(8), 8);
        // A huge explicit value is honoured (only the auto path is clamped).
        assert_eq!(resolve_read_pool_size(100), 100);
    }
}

