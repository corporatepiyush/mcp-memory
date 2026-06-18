use crate::errors::Result;
use crate::Transport;

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
}

impl Config {
    pub fn from_args(args: &super::Args) -> Result<Self> {
        let memory_file_path = args
            .memory_file
            .clone()
            .or_else(|| std::env::var("MEMORY_FILE_PATH").ok())
            .unwrap_or_else(|| "memory.mcpmem".to_string());

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
