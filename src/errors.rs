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
        }
    }
}

pub type Result<T> = std::result::Result<T, MCSError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(MCSError::ParseError("".into()).error_code(), -32700);
        assert_eq!(MCSError::MethodNotFound("".into()).error_code(), -32601);
        assert_eq!(MCSError::InvalidParams("".into()).error_code(), -32602);
        assert_eq!(MCSError::MemoryError("".into()).error_code(), -32000);
        assert_eq!(MCSError::IoError(std::io::Error::other("")).error_code(), -32003);
        assert_eq!(MCSError::JsonError(serde_json::from_str::<()>("invalid").unwrap_err()).error_code(), -32700);
    }
}
