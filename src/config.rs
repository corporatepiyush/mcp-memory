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

        Ok(Config {
            memory_file_path,
            transport: args.transport,
            bind_addr: args.bind.clone(),
            durability,
            auth_token,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use crate::Args;

    #[test]
    fn test_config_defaults() {
        let args = Args::parse_from(["mcp-memory"]);
        let cfg = Config::from_args(&args).unwrap();
        assert_eq!(cfg.memory_file_path, "memory.mcpmem");
    }

    #[test]
    fn test_config_custom_path() {
        let args = Args::parse_from(["mcp-memory", "--memory-file", "/tmp/test.jsonl"]);
        let cfg = Config::from_args(&args).unwrap();
        assert_eq!(cfg.memory_file_path, "/tmp/test.jsonl");
    }
}
