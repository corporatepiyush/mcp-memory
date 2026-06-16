pub mod actions;
pub mod config;
pub mod errors;
pub mod intern;
pub mod kg;
pub mod protocol;
pub mod search;
pub mod server;
pub mod store;
pub mod tools;
pub mod types;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "MCP Memory Server")]
#[command(about = "Knowledge graph memory server for MCP — entities, relations, and observations persisted via binary log", long_about = None)]
pub struct Args {
    /// Path to the memory file
    #[arg(short = 'f', long = "memory-file")]
    pub memory_file: Option<String>,

    /// Log level
    #[arg(short, long, default_value = "info")]
    pub log_level: String,
}
