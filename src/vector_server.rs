use std::convert::Infallible;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tracing::{error, info};

use crate::config::Config;
use crate::errors::{MCSError, Result};
use crate::kg::GraphHandle;
use crate::protocol::{JsonRpcRequest, JsonRpcResponse};
use crate::tools;
use crate::vector_actions;
use crate::vector_store::{VectorConfig, VectorStore};

enum HandlerResult {
    Value(Value),
    RawResult(String),
}

const BUFFER_CAPACITY: usize = 65536;
const NEWLINE: &[u8] = b"\n";
const MAX_REQUEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_TCP_CONNECTIONS: usize = 128;

#[derive(Clone, Copy, PartialEq, Eq)]
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

const SUPPORTED_PROTOCOL_VERSIONS: &[&str] =
    &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];
const LATEST_PROTOCOL_VERSION: &str = "2025-11-25";

const VECTOR_SERVER_INSTRUCTIONS: &str = "Knowledge-graph memory MCP server with vector search. \
Entity names are unique and case-sensitive. Use `create_entities`/`create_relations` to build the \
graph, and `vector_upsert_embedding` to add vector embeddings. Search semantically with \
`vector_search_entities` or combine text + vector with `hybrid_search`. Tool failures are \
returned with `isError: true` rather than as protocol errors.";

fn handle_initialize(req: &JsonRpcRequest) -> Value {
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
            "name": "mcp-memory-vec",
            "version": env!("CARGO_PKG_VERSION")
        },
        "instructions": VECTOR_SERVER_INSTRUCTIONS
    })
}

static VECTOR_TOOLS_LIST: std::sync::OnceLock<Value> = std::sync::OnceLock::new();

fn handle_tools_list() -> Value {
    if let Some(cached) = VECTOR_TOOLS_LIST.get() {
        return cached.clone();
    }
    let base_tools: Vec<Value> = serde_json::from_str(include_str!("../tools.json"))
        .expect("tools.json is valid JSON");
    let vec_tools: Vec<Value> = serde_json::from_str(include_str!("../vector_tools.json"))
        .expect("vector_tools.json is valid JSON");
    let mut all = base_tools;
    all.extend(vec_tools);
    let result = json!({ "tools": all });
    let _ = VECTOR_TOOLS_LIST.set(result.clone());
    result
}

#[inline]
fn tool_error(message: &str) -> Value {
    json!({
        "content": [{ "type": "text", "text": message }],
        "isError": true
    })
}

fn handle_tools_call(req: &JsonRpcRequest, kg: &GraphHandle, vs: &VectorStore) -> Result<HandlerResult> {
    let tool_name = req
        .params
        .as_ref()
        .and_then(|p| p.get("name").and_then(|v| v.as_str()))
        .ok_or_else(|| MCSError::InvalidParams("Missing 'name' parameter".into()))?;

    let tool_args = req.params.as_ref().and_then(|p| p.get("arguments"));

    if !tools::tool_exists(tool_name) && !is_vector_tool_name(tool_name) {
        return Err(MCSError::MethodNotFound(tool_name.to_string()));
    }

    let result = match tool_name {
        // Vector tools
        "vector_upsert_embedding" => {
            vector_actions::handle_vector_upsert_embedding(vs, kg, tool_args)
                .map(HandlerResult::Value)
        }
        "vector_search_entities" => {
            vector_actions::handle_vector_search_entities(vs, kg, tool_args)
                .map(HandlerResult::RawResult)
        }
        "vector_delete_embedding" => {
            vector_actions::handle_vector_delete_embedding(vs, kg, tool_args)
                .map(HandlerResult::Value)
        }
        "hybrid_search" => {
            vector_actions::handle_hybrid_search(vs, kg, tool_args)
                .map(HandlerResult::RawResult)
        }
        "vector_refresh_graph_cache" => {
            vector_actions::handle_refresh_graph_cache(vs, kg, tool_args)
                .map(HandlerResult::Value)
        }
        "vector_store_stats" => {
            vector_actions::handle_vector_store_stats(vs, kg, tool_args)
                .map(HandlerResult::Value)
        }
        // KG tools — delegate to existing handlers
        "read_graph" | "search_nodes" => {
            let kg_only = crate::server::dispatch_line(
                &serialize_request(req),
                kg,
            );
            match kg_only {
                Some(resp) => {
                    let v: Value = serde_json::from_str(&resp)
                        .map_err(MCSError::JsonError)?;
                    if let Some(result_val) = v.get("result") {
                        Ok(HandlerResult::Value(result_val.clone()))
                    } else {
                        Err(MCSError::MemoryError("KG dispatch failed".into()))
                    }
                }
                None => Ok(HandlerResult::Value(Value::Null)),
            }
        }
        _ => {
            // Delegate to existing KG handlers by calling dispatch_line
            let kg_only = crate::server::dispatch_line(
                &serialize_request(req),
                kg,
            );
            match kg_only {
                Some(resp) => {
                    let v: Value = serde_json::from_str(&resp)
                        .map_err(MCSError::JsonError)?;
                    if let Some(result_val) = v.get("result") {
                        Ok(HandlerResult::Value(result_val.clone()))
                    } else {
                        Err(MCSError::MemoryError("KG dispatch failed".into()))
                    }
                }
                None => Ok(HandlerResult::Value(Value::Null)),
            }
        }
    };

    Ok(result.unwrap_or_else(|e| {
        error!("Tool '{tool_name}' error: {e}");
        HandlerResult::Value(tool_error(&e.to_string()))
    }))
}

fn is_vector_tool_name(name: &str) -> bool {
    matches!(
        name,
        "vector_upsert_embedding"
            | "vector_search_entities"
            | "vector_delete_embedding"
            | "hybrid_search"
            | "vector_refresh_graph_cache"
            | "vector_store_stats"
    )
}

fn serialize_request(req: &JsonRpcRequest) -> String {
    let params = req.params.as_ref().map(|p| {
        let name = p.get("name").cloned().unwrap_or(Value::Null);
        let args = p.get("arguments").cloned();
        json!({
            "name": name,
            "arguments": args
        })
    });
    let wrapped = JsonRpcRequest {
        jsonrpc: req.jsonrpc.clone(),
        id: req.id.clone(),
        method: req.method.clone(),
        params,
    };
    serde_json::to_string(&wrapped).unwrap_or_default()
}

fn process_request_value(value: Value, kg: &GraphHandle, vs: &VectorStore) -> Option<Value> {
    let req: JsonRpcRequest = match serde_json::from_value(value) {
        Ok(r) => r,
        Err(e) => return Some(to_value(parse_error(e.to_string()))),
    };
    req.id.as_ref()?;

    match process_request(&req, kg, vs) {
        Ok(HandlerResult::Value(result)) => {
            Some(to_value(JsonRpcResponse::success(req.id, result)))
        }
        Ok(HandlerResult::RawResult(result_json)) => {
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

fn process_request(req: &JsonRpcRequest, kg: &GraphHandle, vs: &VectorStore) -> Result<HandlerResult> {
    match req.method.as_str() {
        "initialize" => Ok(HandlerResult::Value(handle_initialize(req))),
        "tools/list" => Ok(HandlerResult::Value(handle_tools_list())),
        "tools/call" => handle_tools_call(req, kg, vs),
        "ping" => Ok(HandlerResult::Value(Value::Null)),
        method if method.starts_with("notifications/") => {
            tracing::trace!("Received notification: {method}");
            Ok(HandlerResult::Value(Value::Null))
        }
        _ => Err(MCSError::MethodNotFound(req.method.clone())),
    }
}

pub fn dispatch_line(line: &str, kg: &GraphHandle, vs: &VectorStore) -> Option<String> {
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

    match process_request(&req, kg, vs) {
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

const AUTH_REQUIRED_LINE: &str = "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32001,\
\"message\":\"Authentication required: send the bearer token as the first line\"},\"id\":null}\n";

async fn authenticate_line_conn<R>(reader: &mut R, expected: &str) -> Result<bool>
where
    R: AsyncBufReadExt + Unpin,
{
    let mut line = String::new();
    match read_line_capped(reader, &mut line, MAX_REQUEST_BYTES)
        .await
        .map_err(MCSError::IoError)?
    {
        LineRead::Line => Ok(crate::server::token_matches(&line, expected)),
        _ => Ok(false),
    }
}

async fn serve_line_conn<R, W>(
    reader: &mut R,
    writer: &mut W,
    kg: Arc<GraphHandle>,
    vs: Arc<VectorStore>,
) -> Result<()>
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
                let vs_clone = Arc::clone(&vs);
                let resp = tokio::task::spawn_blocking(move || {
                    dispatch_line(&line_copy, &kg_clone, &vs_clone)
                })
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

fn spawn_maintenance(kg: Arc<GraphHandle>) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(300));
        interval.tick().await;
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

pub fn dispatch_http_body(
    body: &str,
    kg: &GraphHandle,
    vs: &VectorStore,
) -> std::result::Result<Option<Value>, String> {
    let value: Value = serde_json::from_str(body.trim()).map_err(|e| e.to_string())?;
    match value {
        Value::Array(items) => {
            let responses: Vec<Value> = items
                .into_iter()
                .filter_map(|v| process_request_value(v, kg, vs))
                .collect();
            Ok((!responses.is_empty()).then_some(Value::Array(responses)))
        }
        other => Ok(process_request_value(other, kg, vs)),
    }
}

pub struct VectorServer {
    config: Arc<Config>,
    kg: Arc<GraphHandle>,
    vs: Arc<VectorStore>,
}

impl VectorServer {
    pub fn new(config: Config, vec_config: VectorConfig) -> Result<Self> {
        let path = Path::new(&config.memory_file_path);
        let lru_cache = std::num::NonZeroUsize::new(config.lru_cache_size).unwrap_or_else(|| {
            std::num::NonZeroUsize::new(10000).expect("10000 > 0")
        });
        let kg = GraphHandle::new(
            path,
            config.durability,
            config.mmap_size,
            lru_cache,
            config.read_pool_size,
        )?;
        let vs = VectorStore::with_config(path, &vec_config)?;

        Ok(Self {
            config: Arc::new(config),
            kg: Arc::new(kg),
            vs: Arc::new(vs),
        })
    }

    pub fn graph(&self) -> Arc<GraphHandle> {
        Arc::clone(&self.kg)
    }

    pub fn vector_store(&self) -> Arc<VectorStore> {
        Arc::clone(&self.vs)
    }

    pub async fn run_stdio(&self) -> Result<()> {
        spawn_maintenance(self.kg.clone());
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::with_capacity(BUFFER_CAPACITY, stdin);
        let mut stdout = tokio::io::stdout();
        serve_line_conn(&mut reader, &mut stdout, Arc::clone(&self.kg), Arc::clone(&self.vs)).await
    }

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
            let vs = Arc::clone(&self.vs);
            let auth_token = auth_token.clone();
            tokio::spawn(async move {
                let _permit = permit;
                let (read_half, mut write_half) = socket.into_split();
                let mut reader = BufReader::with_capacity(BUFFER_CAPACITY, read_half);
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
                if let Err(e) = serve_line_conn(&mut reader, &mut write_half, kg, vs).await {
                    error!("TCP connection {peer} error: {e}");
                }
            });
        }
    }

    pub async fn run_http(&self, addr: &str) -> Result<()> {
        spawn_maintenance(self.kg.clone());
        self.run_http_inner(addr).await
    }

    async fn run_http_inner(&self, addr: &str) -> Result<()> {
        use axum::routing::{get, post};
        use axum::Router;

        let kg = Arc::clone(&self.kg);
        let vs = Arc::clone(&self.vs);
        let auth_token = self.config.auth_token.clone();

        let app = Router::new()
            .route("/mcp", post(handle_http_post))
            .route("/mcp", get(handle_http_get))
            .with_state(HttpState { kg, vs, auth_token });

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(MCSError::IoError)?;
        info!("MCP Streamable HTTP listening on {addr}");

        if let (Some(cert), Some(key)) = (
            self.config.tls_cert.clone(),
            self.config.tls_key.clone(),
        ) {
            let tls_config = crate::tls::server_config(&cert, &key)
                .await
                .map_err(MCSError::IoError)?;
            axum_server::bind_rustls(listener.local_addr().unwrap(), tls_config)
                .serve(app.into_make_service())
                .await
                .map_err(|e| MCSError::IoError(std::io::Error::other(e)))?;
        } else {
            axum::serve(listener, app)
                .await
                .map_err(|e| MCSError::IoError(std::io::Error::other(e)))?;
        }
        Ok(())
    }
}

#[derive(Clone)]
struct HttpState {
    kg: Arc<GraphHandle>,
    vs: Arc<VectorStore>,
    auth_token: Option<Arc<str>>,
}

/// `true` when the request is allowed: either no token is configured, or the
/// `Authorization` header carries the expected bearer token.
fn http_authorized(state: &HttpState, headers: &axum::http::HeaderMap) -> bool {
    match state.auth_token {
        None => true,
        Some(ref expected) => headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|presented| crate::server::token_matches(presented, expected)),
    }
}

async fn handle_http_post(
    axum::extract::State(state): axum::extract::State<HttpState>,
    headers: axum::http::HeaderMap,
    body: String,
) -> axum::response::Response {
    use axum::response::sse::Event;
    use axum::response::{IntoResponse, Json};
    use axum::http::StatusCode;

    if !http_authorized(&state, &headers) {
        return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
    }

    let result = tokio::task::spawn_blocking(move || {
        dispatch_http_body(&body, &state.kg, &state.vs)
    })
    .await;

    let outcome = match result {
        Ok(inner) => inner,
        Err(join_err) => {
            error!("dispatch task panicked: {join_err}");
            return (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response();
        }
    };

    match outcome {
        Ok(None) => StatusCode::ACCEPTED.into_response(),
        Ok(Some(value)) => {
            let wants_sse = headers
                .get(axum::http::header::ACCEPT)
                .and_then(|v| v.to_str().ok())
                .is_some_and(|a| a.contains("text/event-stream"));
            if wants_sse {
                let json = serde_json::to_string(&value).unwrap();
                let stream = futures::stream::once(async move {
                    Ok::<Event, Infallible>(Event::default().data(json))
                });
                axum::response::sse::Sse::new(stream).into_response()
            } else {
                Json(value).into_response()
            }
        }
        Err(e) => {
            let resp = json!({
                "jsonrpc": "2.0",
                "error": { "code": -32700, "message": format!("Parse error: {e}") },
                "id": null
            });
            (StatusCode::BAD_REQUEST, Json(resp)).into_response()
        }
    }
}

async fn handle_http_get(
    axum::extract::State(state): axum::extract::State<HttpState>,
    headers: axum::http::HeaderMap,
) -> axum::response::Response {
    use axum::response::sse::{Event, KeepAlive, Sse};
    use axum::response::IntoResponse;
    use axum::http::StatusCode;

    if !http_authorized(&state, &headers) {
        return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
    }

    let stream = futures::stream::pending::<std::result::Result<Event, Infallible>>();
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
