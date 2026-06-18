use serde_json::{Value, json};
use std::path::Path;
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tracing::{error, info};

use crate::actions::memory;
use crate::config::Config;
use crate::errors::{MCSError, Result};
use crate::kg::GraphHandle;
use crate::protocol::{JsonRpcRequest, JsonRpcResponse};
use crate::tools;

const BUFFER_CAPACITY: usize = 65536;
const NEWLINE: &[u8] = b"\n";
/// Maximum size of a single inbound JSON-RPC message (shared by all transports).
pub const MAX_REQUEST_BYTES: usize = 16 * 1024 * 1024;

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

/// Process one parsed JSON-RPC message. `None` means "no reply" — the message
/// was a notification (no `id`), per JSON-RPC.
pub fn process_value(value: Value, kg: &GraphHandle) -> Option<Value> {
    let req: JsonRpcRequest = match serde_json::from_value(value) {
        Ok(r) => r,
        Err(e) => return Some(to_value(parse_error(e.to_string()))),
    };
    req.id.as_ref()?;
    let response = match process_request(&req, kg) {
        Ok(result) => JsonRpcResponse::success(req.id, result),
        Err(e) => JsonRpcResponse::error(req.id, e.error_code(), e.to_string()),
    };
    Some(to_value(response))
}

/// Dispatch one framed line (stdio / tcp). Returns the serialized response, or
/// `None` for a notification.
pub fn dispatch_line(line: &str, kg: &GraphHandle) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Some(serde_json::to_string(&parse_error("Empty request".into())).unwrap());
    }
    let value: Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(e) => return Some(serde_json::to_string(&parse_error(e.to_string())).unwrap()),
    };
    process_value(value, kg).map(|v| serde_json::to_string(&v).unwrap())
}

/// Dispatch a Streamable-HTTP POST body, which may be a single JSON-RPC message
/// or a batch array. `Ok(None)` means the body held only notifications (HTTP
/// 202, empty body); `Err` means the body was not valid JSON.
pub fn dispatch_http_body(
    body: &str,
    kg: &GraphHandle,
) -> std::result::Result<Option<Value>, String> {
    let value: Value = serde_json::from_str(body.trim()).map_err(|e| e.to_string())?;
    match value {
        Value::Array(items) => {
            let responses: Vec<Value> =
                items.into_iter().filter_map(|v| process_value(v, kg)).collect();
            Ok((!responses.is_empty()).then_some(Value::Array(responses)))
        }
        other => Ok(process_value(other, kg)),
    }
}

#[inline]
fn to_value(resp: JsonRpcResponse) -> Value {
    serde_json::to_value(resp).expect("JsonRpcResponse always serializes")
}

pub struct MCPServer {
    _config: Arc<Config>,
    kg: Arc<GraphHandle>,
}

impl MCPServer {
    pub fn new(config: Config) -> Result<Self> {
        let path = Path::new(&config.memory_file_path);
        let kg = GraphHandle::new(path).map_err(MCSError::IoError)?;

        Ok(Self {
            _config: Arc::new(config),
            kg: Arc::new(kg),
        })
    }

    /// Expose the shared graph handle (used to drive the HTTP transport).
    pub fn graph(&self) -> Arc<GraphHandle> {
        Arc::clone(&self.kg)
    }

    /// stdio transport: newline-delimited JSON-RPC over stdin/stdout.
    pub async fn run_stdio(&self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::with_capacity(BUFFER_CAPACITY, stdin);
        let mut stdout = tokio::io::stdout();
        serve_line_conn(&mut reader, &mut stdout, &self.kg).await
    }

    /// TCP transport: each accepted connection speaks newline-delimited
    /// JSON-RPC, exactly like stdio. Connections are served concurrently and
    /// share the one graph behind its mutex.
    pub async fn run_tcp(&self, addr: &str) -> Result<()> {
        let listener = TcpListener::bind(addr).await.map_err(MCSError::IoError)?;
        info!("Listening for TCP MCP connections on {addr}");
        loop {
            let (socket, peer) = listener.accept().await.map_err(MCSError::IoError)?;
            let kg = Arc::clone(&self.kg);
            tokio::spawn(async move {
                let (read_half, mut write_half) = socket.into_split();
                let mut reader = BufReader::with_capacity(BUFFER_CAPACITY, read_half);
                if let Err(e) = serve_line_conn(&mut reader, &mut write_half, &kg).await {
                    error!("TCP connection {peer} error: {e}");
                }
            });
        }
    }

    /// MCP Streamable HTTP transport (POST/GET `/mcp`, JSON or SSE responses).
    pub async fn run_http(&self, addr: &str) -> Result<()> {
        crate::http::run(addr, self.graph()).await
    }
}

/// Drive one line-framed connection (stdio or a single TCP socket): read
/// newline-delimited JSON-RPC requests, write newline-delimited responses.
/// Notifications produce no output. Returns when the peer closes the stream.
async fn serve_line_conn<R, W>(reader: &mut R, writer: &mut W, kg: &GraphHandle) -> Result<()>
where
    R: AsyncBufReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    let mut line = String::with_capacity(1024);
    let mut out = Vec::with_capacity(BUFFER_CAPACITY);

    loop {
        match read_line_capped(reader, &mut line, MAX_REQUEST_BYTES).await {
            Ok(LineRead::Eof) => break,
            Ok(LineRead::Line) => {
                if let Some(resp) = dispatch_line(&line, kg) {
                    out.clear();
                    out.extend_from_slice(resp.as_bytes());
                    out.extend_from_slice(NEWLINE);
                    writer.write_all(&out).await.map_err(MCSError::IoError)?;
                    writer.flush().await.map_err(MCSError::IoError)?;
                }
            }
            Ok(LineRead::TooLong) => {
                let err = MCSError::InvalidParams("Request exceeds maximum size of 16MB".into());
                let response = JsonRpcResponse::error(None, err.error_code(), err.to_string());
                out.clear();
                serde_json::to_writer(&mut out, &response).map_err(MCSError::JsonError)?;
                out.extend_from_slice(NEWLINE);
                writer.write_all(&out).await.map_err(MCSError::IoError)?;
                writer.flush().await.map_err(MCSError::IoError)?;
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

fn process_request(req: &JsonRpcRequest, kg: &GraphHandle) -> Result<Value> {
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
    static CACHED: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
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

fn handle_tools_call(req: &JsonRpcRequest, kg: &GraphHandle) -> Result<Value> {
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
        "read_graph" => memory::handle_read_graph(kg, tool_args),
        "search_nodes" => memory::handle_search_nodes(kg, tool_args),
        "open_nodes" => memory::handle_open_nodes(kg, tool_args),
        "get_entity" => memory::handle_get_entity(kg, tool_args),
        "graph_stats" => memory::handle_graph_stats(kg),
        "search_relations" => memory::handle_search_relations(kg, tool_args),
        "find_path" => memory::handle_find_path(kg, tool_args),
        "compact" => memory::handle_compact(kg),
        "get_neighbors" => memory::handle_get_neighbors(kg, tool_args),
        "describe_entity" => memory::handle_describe_entity(kg, tool_args),
        "list_entity_types" => memory::handle_list_entity_types(kg),
        "list_relation_types" => memory::handle_list_relation_types(kg),
        "upsert_entities" => memory::handle_upsert_entities(kg, tool_args),
        "export_graph" => memory::handle_export_graph(kg, tool_args),
        "merge_entities" => memory::handle_merge_entities(kg, tool_args),
        "extract_subgraph" => memory::handle_extract_subgraph(kg, tool_args),
        "batch_get_entities" => memory::handle_batch_get_entities(kg, tool_args),
        "find_all_paths" => memory::handle_find_all_paths(kg, tool_args),
        tool => Err(MCSError::MethodNotFound(tool.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn setup_kg() -> (Arc<GraphHandle>, String) {
        let pid = std::process::id();
        let seq = COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = format!("/tmp/mcp_mem_test_{pid}_{seq}.bin");
        let kg = GraphHandle::new(Path::new(&path)).unwrap();
        (Arc::new(kg), path)
    }

    fn cleanup(path: &str) {
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_dispatch_line_valid_request() {
        let (kg, path) = setup_kg();
        let line = r#"{"jsonrpc":"2.0","method":"initialize","id":1}"#;
        let resp = dispatch_line(line, &kg).unwrap();
        let v: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(v["id"], 1);
        assert_eq!(v["result"]["serverInfo"]["name"], "mcp-memory");
        cleanup(&path);
    }

    #[test]
    fn test_dispatch_line_invalid_json() {
        let (kg, path) = setup_kg();
        let resp = dispatch_line("{invalid}", &kg).unwrap();
        let v: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(v["error"]["code"], -32700);
        assert!(v["id"].is_null());
        cleanup(&path);
    }

    #[test]
    fn test_dispatch_line_empty() {
        let (kg, path) = setup_kg();
        let resp = dispatch_line("   \n", &kg).unwrap();
        let v: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(v["error"]["code"], -32700);
        cleanup(&path);
    }

    #[test]
    fn test_notification_has_no_response() {
        let (kg, path) = setup_kg();
        let line = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        assert!(dispatch_line(line, &kg).is_none());
        cleanup(&path);
    }

    #[test]
    fn test_unknown_method_error() {
        let (kg, path) = setup_kg();
        let line = r#"{"jsonrpc":"2.0","method":"does/not/exist","id":7}"#;
        let v: Value = serde_json::from_str(&dispatch_line(line, &kg).unwrap()).unwrap();
        assert_eq!(v["id"], 7);
        assert_eq!(v["error"]["code"], -32601);
        cleanup(&path);
    }

    #[test]
    fn test_tools_call_roundtrip_via_dispatch() {
        let (kg, path) = setup_kg();
        let create = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"create_entities","arguments":{"entities":[{"name":"Ada","entityType":"person","observations":["math"]}]}}}"#;
        assert!(dispatch_line(create, &kg).is_some());

        let read = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"read_graph","arguments":{}}}"#;
        let v: Value = serde_json::from_str(&dispatch_line(read, &kg).unwrap()).unwrap();
        let text = v["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Ada"));
        cleanup(&path);
    }

    #[test]
    fn test_http_body_batch_and_notifications() {
        let (kg, path) = setup_kg();
        let batch = r#"[
            {"jsonrpc":"2.0","method":"initialize","id":1},
            {"jsonrpc":"2.0","method":"notifications/initialized"}
        ]"#;
        let out = dispatch_http_body(batch, &kg).unwrap().unwrap();
        let arr = out.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["id"], 1);

        let notif = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        assert!(dispatch_http_body(notif, &kg).unwrap().is_none());

        assert!(dispatch_http_body("{bad", &kg).is_err());
        cleanup(&path);
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
