use crate::errors::Result;
use crate::Transport;

#[derive(Debug, Clone)]
pub struct Config {
    pub memory_file_path: String,
    pub transport: Transport,
    pub bind_addr: String,
}

impl Config {
    pub fn from_args(args: &super::Args) -> Result<Self> {
        let memory_file_path = args
            .memory_file
            .clone()
            .or_else(|| std::env::var("MEMORY_FILE_PATH").ok())
            .unwrap_or_else(|| "memory.mcpmem".to_string());

        Ok(Config {
            memory_file_path,
            transport: args.transport,
            bind_addr: args.bind.clone(),
        })
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            memory_file_path: "memory.mcpmem".to_string(),
            transport: Transport::Stdio,
            bind_addr: "127.0.0.1:8080".to_string(),
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
