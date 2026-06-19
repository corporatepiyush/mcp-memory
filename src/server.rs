use serde_json::{Value, json};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tracing::{error, info};

use crate::actions::memory;
use crate::config::Config;
use crate::errors::{MCSError, Result};
use crate::kg::GraphHandle;
use crate::protocol::{JsonRpcRequest, JsonRpcResponse};
use crate::tools;

/// Outcome of processing a request: either a pre-escaped JSON Value (small
/// payloads) or a pre-serialized JSON *string* of the `result` field (avoids
/// a second serialization pass for large payloads such as `read_graph`).
enum HandlerResult {
    Value(Value),
    RawResult(String),
}

const BUFFER_CAPACITY: usize = 65536;
const NEWLINE: &[u8] = b"\n";
/// Maximum size of a single inbound JSON-RPC message (shared by all transports).
pub const MAX_REQUEST_BYTES: usize = 16 * 1024 * 1024;
/// Maximum number of concurrent TCP connections (C4).
const MAX_TCP_CONNECTIONS: usize = 128;

enum LineRead {
    Line,
    Eof,
    TooLong,
}

async fn read_line_capped<R>(
    reader: &mut R,
    out: &mut String,
    max: usize,
) -> std::io::Result<LineRead>
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
            // Move `buf` into the String — no copy. `buf` is not used afterward.
            *out = String::from_utf8(buf).map_err(|_| {
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
                *out = String::from_utf8(buf).map_err(|_| {
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
    
    match process_request(&req, kg) {
        Ok(HandlerResult::Value(result)) => {
            Some(to_value(JsonRpcResponse::success(req.id, result)))
        }
        Ok(HandlerResult::RawResult(_)) => {
            // RawResult cannot pass through Value — dispatch_line and
            // dispatch_http_body handle it via separate code paths.
            unreachable!("RawResult must be handled at the dispatch level, not via process_value");
        }
        Err(e) => Some(to_value(JsonRpcResponse::error(
            req.id,
            e.error_code(),
            e.to_string(),
        ))),
    }
}

/// Dispatch one framed line (stdio / tcp). Returns the serialized response, or
/// `None` for a notification.
pub fn dispatch_line(line: &str, kg: &GraphHandle) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Some(serde_json::to_string(&parse_error("Empty request".into())).unwrap());
    }
    let raw: Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(e) => return Some(serde_json::to_string(&parse_error(e.to_string())).unwrap()),
    };
    let req: JsonRpcRequest = match serde_json::from_value(raw) {
        Ok(r) => r,
        Err(e) => return Some(serde_json::to_string(&parse_error(e.to_string())).unwrap()),
    };
    req.id.as_ref()?;
    match process_request(&req, kg) {
        Ok(HandlerResult::Value(result)) => {
            let resp = JsonRpcResponse::success(req.id, result);
            Some(serde_json::to_string(&resp).unwrap())
        }
        Ok(HandlerResult::RawResult(result_json)) => {
            let id_json = serde_json::to_string(&req.id).unwrap();
            let mut out = String::with_capacity(64 + id_json.len() + result_json.len());
            out.push_str(r#"{"jsonrpc":"2.0","id":"#);
            out.push_str(&id_json);
            out.push_str(",\"result\":");
            out.push_str(&result_json);
            out.push('}');
            Some(out)
        }
        Err(e) => {
            let resp = JsonRpcResponse::error(req.id, e.error_code(), e.to_string());
            Some(serde_json::to_string(&resp).unwrap())
        }
    }
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
            // Batches are rare and never huge — keep Value path for simplicity.
            let responses: Vec<Value> = items
                .into_iter()
                .filter_map(|v| process_value_http(v, kg))
                .collect();
            Ok((!responses.is_empty()).then_some(Value::Array(responses)))
        }
        other => Ok(process_value_http(other, kg)),
    }
}

/// HTTP variant of process_value that handles RawResult by converting to Value
/// (acceptable since HTTP payloads are typically much smaller in this context).
fn process_value_http(value: Value, kg: &GraphHandle) -> Option<Value> {
    let req: JsonRpcRequest = match serde_json::from_value(value) {
        Ok(r) => r,
        Err(e) => return Some(to_value(parse_error(e.to_string()))),
    };
    req.id.as_ref()?;
    match process_request(&req, kg) {
        Ok(HandlerResult::Value(result)) => {
            Some(to_value(JsonRpcResponse::success(req.id, result)))
        }
        Ok(HandlerResult::RawResult(result_json)) => {
            // Parse the pre-serialized result back into a Value for HTTP delivery.
            // This is a small extra cost for the HTTP transport; the stdio/TCP
            // path (dispatch_line) avoids it entirely.
            let result_val: Value = serde_json::from_str(&result_json).unwrap_or(Value::Null);
            Some(to_value(JsonRpcResponse::success(req.id, result_val)))
        }
        Err(e) => Some(to_value(JsonRpcResponse::error(
            req.id,
            e.error_code(),
            e.to_string(),
        ))),
    }
}

#[inline]
fn to_value(resp: JsonRpcResponse) -> Value {
    serde_json::to_value(resp).expect("JsonRpcResponse always serializes")
}

pub struct MCPServer {
    config: Arc<Config>,
    kg: Arc<GraphHandle>,
}

impl MCPServer {
    pub fn new(config: Config) -> Result<Self> {
        let path = Path::new(&config.memory_file_path);
        let lru_cache = NonZeroUsize::new(config.lru_cache_size).unwrap_or_else(|| {
            NonZeroUsize::new(10000).expect("10000 > 0")
        });
        let kg = GraphHandle::new(
            path,
            config.durability,
            config.mmap_size,
            lru_cache,
            config.read_pool_size,
        )?;

        Ok(Self {
            config: Arc::new(config),
            kg: Arc::new(kg),
        })
    }

    /// Expose the shared graph handle (used to drive the HTTP transport).
    pub fn graph(&self) -> Arc<GraphHandle> {
        Arc::clone(&self.kg)
    }

    /// stdio transport: newline-delimited JSON-RPC over stdin/stdout.
    pub async fn run_stdio(&self) -> Result<()> {
        spawn_maintenance(self.kg.clone());
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::with_capacity(BUFFER_CAPACITY, stdin);
        let mut stdout = tokio::io::stdout();
        serve_line_conn(&mut reader, &mut stdout, Arc::clone(&self.kg)).await
    }

    /// TCP transport: each accepted connection speaks newline-delimited
    /// JSON-RPC, exactly like stdio. Connections are served concurrently (up to
    /// [`MAX_TCP_CONNECTIONS`]) and share the one graph behind its mutex.
    pub async fn run_tcp(&self, addr: &str) -> Result<()> {
        spawn_maintenance(self.kg.clone());
        let listener = TcpListener::bind(addr).await.map_err(MCSError::IoError)?;
        let semaphore = Arc::new(Semaphore::new(MAX_TCP_CONNECTIONS));
        let auth_token = self.config.auth_token.clone();
        info!(
            "Listening for TCP MCP connections on {addr} (max {MAX_TCP_CONNECTIONS}, auth {})",
            if auth_token.is_some() { "on" } else { "off" }
        );
        loop {
            let permit = Arc::clone(&semaphore).acquire_owned().await;
            let (socket, peer) = listener.accept().await.map_err(MCSError::IoError)?;
            let kg = Arc::clone(&self.kg);
            let auth_token = auth_token.clone();
            tokio::spawn(async move {
                let _permit = permit; // held for the connection lifetime
                let (read_half, mut write_half) = socket.into_split();
                let mut reader = BufReader::with_capacity(BUFFER_CAPACITY, read_half);
                // When a token is configured, the client must send it as the
                // first line before any JSON-RPC traffic.
                if let Some(ref expected) = auth_token {
                    match authenticate_line_conn(&mut reader, expected).await {
                        Ok(true) => {}
                        Ok(false) => {
                            let _ = write_half.write_all(AUTH_REQUIRED_LINE.as_bytes()).await;
                            let _ = write_half.flush().await;
                            return;
                        }
                        Err(e) => {
                            error!("TCP auth error for {peer}: {e}");
                            return;
                        }
                    }
                }
                if let Err(e) = serve_line_conn(&mut reader, &mut write_half, kg).await {
                    error!("TCP connection {peer} error: {e}");
                }
            });
        }
    }

    /// MCP Streamable HTTP transport (POST/GET `/mcp`, JSON or SSE responses).
    pub async fn run_http(&self, addr: &str) -> Result<()> {
        spawn_maintenance(self.kg.clone());
        crate::http::run(addr, self.graph(), self.config.auth_token.clone()).await
    }
}

/// Spawn a background task that runs periodic database maintenance every
/// 5 minutes until the runtime shuts down.
fn spawn_maintenance(kg: Arc<GraphHandle>) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(300));
        interval.tick().await; // skip immediate first tick
        loop {
            interval.tick().await;
            let kg = kg.clone();
            tokio::task::spawn_blocking(move || {
                if let Err(e) = kg.run_maintenance() {
                    tracing::warn!("Maintenance error: {e}");
                }
            })
            .await
            .ok();
        }
    });
}

/// JSON-RPC error line returned to a TCP client that fails authentication.
const AUTH_REQUIRED_LINE: &str = "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32001,\
\"message\":\"Authentication required: send the bearer token as the first line\"},\"id\":null}\n";

/// Read the first line of a connection and compare it (constant-time) to the
/// expected bearer token. Returns `Ok(false)` on EOF / oversized first line.
async fn authenticate_line_conn<R>(reader: &mut R, expected: &str) -> Result<bool>
where
    R: AsyncBufReadExt + Unpin,
{
    let mut line = String::new();
    match read_line_capped(reader, &mut line, MAX_REQUEST_BYTES)
        .await
        .map_err(MCSError::IoError)?
    {
        LineRead::Line => Ok(token_matches(&line, expected)),
        _ => Ok(false),
    }
}

/// Drive one line-framed connection (stdio or a single TCP socket): read
/// newline-delimited JSON-RPC requests, write newline-delimited responses.
/// Notifications produce no output. Returns when the peer closes the stream.
/// The dispatch path (graph lock + optional fsync) is offloaded to
/// [`tokio::task::spawn_blocking`] to keep the async reactor responsive (C3).
async fn serve_line_conn<R, W>(reader: &mut R, writer: &mut W, kg: Arc<GraphHandle>) -> Result<()>
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
                let line_copy = line.clone();
                let kg_clone = Arc::clone(&kg);
                let resp =
                    tokio::task::spawn_blocking(move || dispatch_line(&line_copy, &kg_clone))
                        .await
                        .map_err(|join_err| {
                            error!("dispatch task panicked: {join_err}");
                            MCSError::IoError(std::io::Error::other("dispatch task panicked"))
                        })?;
                if let Some(resp) = resp {
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

fn process_request(req: &JsonRpcRequest, kg: &GraphHandle) -> Result<HandlerResult> {
    match req.method.as_str() {
        "initialize" => Ok(HandlerResult::Value(handle_initialize(req))),
        "tools/list" => Ok(HandlerResult::Value(handle_tools_list())),
        "tools/call" => handle_tools_call(req, kg),
        "ping" => Ok(HandlerResult::Value(Value::Null)),
        method if method.starts_with("notifications/") => {
            tracing::trace!("Received notification: {method}");
            Ok(HandlerResult::Value(Value::Null))
        }
        _ => Err(MCSError::MethodNotFound(req.method.clone())),
    }
}

/// MCP protocol revisions this server can speak, newest first (for `initialize`
/// version negotiation).
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] =
    &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];
/// Newest revision we implement; offered when the client requests an unknown one.
const LATEST_PROTOCOL_VERSION: &str = "2025-11-25";

/// `instructions` surfaced to the client and appended to the model's system prompt.
const SERVER_INSTRUCTIONS: &str = "Knowledge-graph memory MCP server. Entity names are unique and \
case-sensitive. Use `create_entities`/`create_relations` to build the graph, `add_observations` to \
attach facts, and `search_nodes`/`open_nodes`/`read_graph` to retrieve. Prefer `upsert_entities` for \
idempotent writes and `merge_entities` to collapse duplicates. Tool failures are returned with \
`isError: true` rather than as protocol errors — read the message and retry.";

fn handle_initialize(req: &JsonRpcRequest) -> Value {
    // Version negotiation: echo a supported requested revision, else offer latest.
    let protocol_version = req
        .params
        .as_ref()
        .and_then(|p| p.get("protocolVersion"))
        .and_then(Value::as_str)
        .filter(|v| SUPPORTED_PROTOCOL_VERSIONS.contains(v))
        .unwrap_or(LATEST_PROTOCOL_VERSION);

    json!({
        "protocolVersion": protocol_version,
        "capabilities": {
            "tools": { "listChanged": false }
        },
        "serverInfo": {
            "name": "mcp-memory",
            "version": env!("CARGO_PKG_VERSION")
        },
        "instructions": SERVER_INSTRUCTIONS
    })
}

/// Wrap a tool execution failure as an MCP `CallToolResult` with `isError: true`
/// so the model sees the message and can self-correct, instead of receiving an
/// opaque JSON-RPC protocol error. (Successful results are already content-
/// wrapped by the action handlers.)
#[inline]
fn tool_error(message: &str) -> Value {
    json!({
        "content": [{ "type": "text", "text": message }],
        "isError": true
    })
}

/// Constant-time bearer-token check. Accepts the raw token or a `Bearer <token>`
/// form; surrounding whitespace is trimmed.
pub fn token_matches(presented: &str, expected: &str) -> bool {
    use subtle::ConstantTimeEq;
    let presented = presented.trim();
    let presented = presented
        .strip_prefix("Bearer ")
        .unwrap_or(presented)
        .trim();
    presented.as_bytes().ct_eq(expected.as_bytes()).into()
}

fn handle_tools_list() -> Value {
    static CACHED: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    if let Some(cached) = CACHED.get() {
        return cached.clone();
    }
    let tools_json = include_str!("../tools.json");
    let tools: Vec<Value> = serde_json::from_str(tools_json)
        .expect("tools.json is valid JSON compiled at build time");
    let result = json!({ "tools": tools });
    let _ = CACHED.set(result.clone());
    result
}

fn handle_tools_call(req: &JsonRpcRequest, kg: &GraphHandle) -> Result<HandlerResult> {
    let tool_name = req
        .params
        .as_ref()
        .and_then(|p| p.get("name").and_then(|v| v.as_str()))
        .ok_or_else(|| MCSError::InvalidParams("Missing 'name' parameter".into()))?;

    let tool_args = req.params.as_ref().and_then(|p| p.get("arguments"));

    if !tools::tool_exists(tool_name) {
        return Err(MCSError::MethodNotFound(tool_name.to_string()));
    }

    let result = match tool_name {
        // Raw-result handlers (large payloads, avoid second serialization pass).
        "read_graph" => memory::handle_read_graph(kg, tool_args).map(HandlerResult::RawResult),
        "search_nodes" => memory::handle_search_nodes(kg, tool_args).map(HandlerResult::RawResult),
        // Standard Value handlers.
        "create_entities" => {
            memory::handle_create_entities(kg, tool_args).map(HandlerResult::Value)
        }
        "create_relations" => {
            memory::handle_create_relations(kg, tool_args).map(HandlerResult::Value)
        }
        "add_observations" => {
            memory::handle_add_observations(kg, tool_args).map(HandlerResult::Value)
        }
        "delete_entities" => {
            memory::handle_delete_entities(kg, tool_args).map(HandlerResult::Value)
        }
        "delete_observations" => {
            memory::handle_delete_observations(kg, tool_args).map(HandlerResult::Value)
        }
        "delete_relations" => {
            memory::handle_delete_relations(kg, tool_args).map(HandlerResult::Value)
        }
        "open_nodes" => memory::handle_open_nodes(kg, tool_args).map(HandlerResult::Value),
        "get_entity" => memory::handle_get_entity(kg, tool_args).map(HandlerResult::Value),
        "graph_stats" => memory::handle_graph_stats(kg).map(HandlerResult::Value),
        "search_relations" => {
            memory::handle_search_relations(kg, tool_args).map(HandlerResult::Value)
        }
        "find_path" => memory::handle_find_path(kg, tool_args).map(HandlerResult::Value),
        "compact" => memory::handle_compact(kg).map(HandlerResult::Value),
        "get_neighbors" => memory::handle_get_neighbors(kg, tool_args).map(HandlerResult::Value),
        "describe_entity" => {
            memory::handle_describe_entity(kg, tool_args).map(HandlerResult::Value)
        }
        "list_entity_types" => memory::handle_list_entity_types(kg).map(HandlerResult::Value),
        "list_relation_types" => memory::handle_list_relation_types(kg).map(HandlerResult::Value),
        "upsert_entities" => {
            memory::handle_upsert_entities(kg, tool_args).map(HandlerResult::Value)
        }
        "export_graph" => memory::handle_export_graph(kg, tool_args).map(HandlerResult::Value),
        "merge_entities" => memory::handle_merge_entities(kg, tool_args).map(HandlerResult::Value),
        "extract_subgraph" => {
            memory::handle_extract_subgraph(kg, tool_args).map(HandlerResult::Value)
        }
        "batch_get_entities" => {
            memory::handle_batch_get_entities(kg, tool_args).map(HandlerResult::Value)
        }
        "find_all_paths" => memory::handle_find_all_paths(kg, tool_args).map(HandlerResult::Value),
        "entity_exists" => memory::handle_entity_exists(kg, tool_args).map(HandlerResult::Value),
        "degree" => memory::handle_degree(kg, tool_args).map(HandlerResult::Value),
        tool => Err(MCSError::MethodNotFound(tool.to_string())),
    };

    // Tool execution failures become isError CallToolResults so the model can
    // read the message and self-correct, instead of an opaque protocol error.
    Ok(result.unwrap_or_else(|e| {
        error!("Tool '{tool_name}' error: {e}");
        HandlerResult::Value(tool_error(&e.to_string()))
    }))
}


