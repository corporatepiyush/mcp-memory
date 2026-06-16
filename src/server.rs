use serde_json::{Value, json};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::error;

use crate::actions::memory;
use crate::config::Config;
use crate::errors::{MCSError, Result};
use crate::kg::KnowledgeGraph;
use crate::protocol::{JsonRpcRequest, JsonRpcResponse};
use crate::tools;

const BUFFER_CAPACITY: usize = 65536;
const NEWLINE: &[u8] = b"\n";

enum LineRead {
    Line,
    Eof,
    TooLong,
}

async fn read_line_capped<R>(reader: &mut R, out: &mut String, max: usize) -> std::io::Result<LineRead>
where
    R: AsyncBufReadExt + Unpin,
{
    out.clear();
    let mut buf: Vec<u8> = Vec::new();
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            if buf.is_empty() {
                return Ok(LineRead::Eof);
            }
            *out = String::from_utf8(buf.clone()).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Non-UTF-8 input")
            })?;
            return Ok(LineRead::Line);
        }
        match available.iter().position(|&b| b == b'\n') {
            Some(i) => {
                if buf.len() + i + 1 > max {
                    reader.consume(i + 1);
                    return Ok(LineRead::TooLong);
                }
                buf.extend_from_slice(&available[..=i]);
                reader.consume(i + 1);
                *out = String::from_utf8(buf.clone()).map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "Non-UTF-8 input")
                })?;
                return Ok(LineRead::Line);
            }
            None => {
                let take = available.len();
                if buf.len() + take > max {
                    reader.consume(take);
                    return Ok(LineRead::TooLong);
                }
                buf.extend_from_slice(available);
                reader.consume(take);
            }
        }
    }
}

fn parse_error(msg: String) -> JsonRpcResponse {
    let mcp_error = MCSError::ParseError(msg);
    JsonRpcResponse::error(None, mcp_error.error_code(), mcp_error.to_string())
}

fn parse_request(line: &str) -> std::result::Result<JsonRpcRequest, String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err("Empty request".to_string());
    }
    serde_json::from_str::<JsonRpcRequest>(trimmed).map_err(|e| e.to_string())
}

pub struct MCPServer {
    _config: Arc<Config>,
    kg: Arc<Mutex<KnowledgeGraph>>,
}

impl MCPServer {
    pub fn new(config: Config) -> Result<Self> {
        let path = Path::new(&config.memory_file_path);
        let kg = KnowledgeGraph::new(path)
            .map_err(MCSError::IoError)?;

        Ok(Self {
            _config: Arc::new(config),
            kg: Arc::new(Mutex::new(kg)),
        })
    }

    pub async fn run_stdio(&self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::with_capacity(BUFFER_CAPACITY, stdin);
        let mut stdout = tokio::io::stdout();
        let mut line = String::with_capacity(1024);
        let mut response_buf = Vec::with_capacity(65536);
        let max = 16 * 1024 * 1024;

        loop {
            match read_line_capped(&mut reader, &mut line, max).await {
                Ok(LineRead::Eof) => break,
                Ok(LineRead::Line) => {
                    process_one_line(&line, &self.kg, &mut response_buf, &mut stdout).await?;
                }
                Ok(LineRead::TooLong) => {
                    let err = MCSError::InvalidParams("Request exceeds maximum size of 16MB".into());
                    let response = JsonRpcResponse::error(None, err.error_code(), err.to_string());
                    response_buf.clear();
                    serde_json::to_writer(&mut response_buf, &response).map_err(MCSError::JsonError)?;
                    response_buf.extend_from_slice(NEWLINE);
                    stdout.write_all(&response_buf).await.map_err(MCSError::IoError)?;
                    stdout.flush().await.map_err(MCSError::IoError)?;
                    break;
                }
                Err(e) => {
                    error!("IO error: {}", e);
                    break;
                }
            }
        }
        Ok(())
    }
}

async fn process_one_line<W: AsyncWriteExt + Unpin>(
    line: &str,
    kg: &Mutex<KnowledgeGraph>,
    response_buf: &mut Vec<u8>,
    writer: &mut W,
) -> Result<()> {
    let (response, is_notification) = match parse_request(line) {
        Ok(req) => {
            let is_notif = req.id.is_none();
            match process_request(&req, kg) {
                Ok(result) => (JsonRpcResponse::success(req.id, result), is_notif),
                Err(e) => (JsonRpcResponse::error(req.id, e.error_code(), e.to_string()), is_notif),
            }
        }
        Err(e) => (parse_error(e), false),
    };

    if is_notification {
        return Ok(());
    }

    response_buf.clear();
    serde_json::to_writer(&mut *response_buf, &response).map_err(MCSError::JsonError)?;
    response_buf.extend_from_slice(NEWLINE);

    writer.write_all(response_buf).await.map_err(MCSError::IoError)?;
    writer.flush().await.map_err(MCSError::IoError)?;
    Ok(())
}

fn process_request(req: &JsonRpcRequest, kg: &Mutex<KnowledgeGraph>) -> Result<Value> {
    match req.method.as_str() {
        "initialize" => handle_initialize(),
        "tools/list" => handle_tools_list(),
        "tools/call" => handle_tools_call(req, kg),
        "ping" => handle_ping(),
        method if method.starts_with("notifications/") => handle_notification(method),
        _ => Err(MCSError::MethodNotFound(req.method.clone())),
    }
}

const fn handle_ping() -> Result<Value> {
    Ok(Value::Null)
}

fn handle_notification(method: &str) -> Result<Value> {
    tracing::trace!("Received notification: {method}");
    Ok(Value::Null)
}

fn handle_initialize() -> Result<Value> {
    Ok(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": { "listChanged": false }
        },
        "serverInfo": {
            "name": "mcp-memory",
            "version": env!("CARGO_PKG_VERSION")
        }
    }))
}

fn handle_tools_list() -> Result<Value> {
    static CACHED: OnceLock<Value> = OnceLock::new();
    if let Some(cached) = CACHED.get() {
        return Ok(cached.clone());
    }
    let tools_json = include_str!("../tools.json");
    let tools: Vec<Value> =
        serde_json::from_str(tools_json).map_err(MCSError::JsonError)?;
    let result = json!({ "tools": tools });
    let _ = CACHED.set(result.clone());
    Ok(result)
}

fn handle_tools_call(req: &JsonRpcRequest, kg: &Mutex<KnowledgeGraph>) -> Result<Value> {
    let tool_name = req
        .params
        .as_ref()
        .and_then(|p| p.get("name").and_then(|v| v.as_str()))
        .ok_or_else(|| MCSError::InvalidParams("Missing 'name' parameter".into()))?;

    let tool_args = req.params.as_ref().and_then(|p| p.get("arguments"));

    if !tools::tool_exists(tool_name) {
        return Err(MCSError::MethodNotFound(tool_name.to_string()));
    }

    match tool_name {
        "create_entities" => memory::handle_create_entities(kg, tool_args),
        "create_relations" => memory::handle_create_relations(kg, tool_args),
        "add_observations" => memory::handle_add_observations(kg, tool_args),
        "delete_entities" => memory::handle_delete_entities(kg, tool_args),
        "delete_observations" => memory::handle_delete_observations(kg, tool_args),
        "delete_relations" => memory::handle_delete_relations(kg, tool_args),
        "read_graph" => memory::handle_read_graph(kg),
        "search_nodes" => memory::handle_search_nodes(kg, tool_args),
        "open_nodes" => memory::handle_open_nodes(kg, tool_args),
        "get_entity" => memory::handle_get_entity(kg, tool_args),
        "graph_stats" => memory::handle_graph_stats(kg),
        "search_relations" => memory::handle_search_relations(kg, tool_args),
        "find_path" => memory::handle_find_path(kg, tool_args),
        "compact" => memory::handle_compact(kg),
        tool => Err(MCSError::MethodNotFound(tool.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn setup_kg() -> (Arc<Mutex<KnowledgeGraph>>, String) {
        let pid = std::process::id();
        let seq = COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = format!("/tmp/mcp_mem_test_{pid}_{seq}.bin");
        let kg = KnowledgeGraph::new(Path::new(&path)).unwrap();
        (Arc::new(Mutex::new(kg)), path)
    }

    fn cleanup(path: &str) {
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_parse_valid_request() {
        let line = r#"{"jsonrpc":"2.0","method":"initialize","id":1}"#;
        let req = parse_request(line).unwrap();
        assert_eq!(req.method, "initialize");
    }

    #[test]
    fn test_parse_invalid_json() {
        let err = parse_request("{invalid}").unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn test_handle_initialize_response() {
        let (kg, path) = setup_kg();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "initialize".to_string(),
            params: None,
            id: Some(Value::Number(1.into())),
        };
        let result = process_request(&req, &kg).unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert_eq!(result["serverInfo"]["name"], "mcp-memory");
        cleanup(&path);
    }
}
