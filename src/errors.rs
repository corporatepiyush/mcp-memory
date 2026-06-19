use thiserror::Error;

#[derive(Error, Debug)]
pub enum MCSError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Method not found: {0}")]
    MethodNotFound(String),

    #[error("Invalid params: {0}")]
    InvalidParams(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl MCSError {
    pub const fn error_code(&self) -> i64 {
        match self {
            MCSError::ParseError(_) => -32700,
            MCSError::MethodNotFound(_) => -32601,
            MCSError::InvalidParams(_) => -32602,
            MCSError::MemoryError(_) => -32000,
            MCSError::IoError(_) => -32003,
            MCSError::JsonError(_) => -32700,
            MCSError::SerializationError(_) => -32004,
        }
    }
}

pub type Result<T> = std::result::Result<T, MCSError>;


