use serde_json::{Value, json};
use std::num::NonZeroUsize;
use std::path::Path;
#[cfg(feature = "code")]
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::error;

#[cfg(feature = "code")]
use crate::actions::code as code_actions;
use crate::actions::memory;
use crate::config::Config;
use crate::errors::{MCSError, Result};
use crate::kg::GraphHandle;
use crate::protocol::{JsonRpcRequest, JsonRpcResponse};
use crate::tools;
use crate::vector_actions;
use crate::vector_store::{VectorConfig, VectorStore};

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

/// Process-wide exposure flags for the knowledge-graph tool categories, set once
/// at startup from `config.enabled_categories`. KG tools carry no per-request
/// state, so a global flag avoids threading the enabled set through every
/// dispatch signature (mirrors `CODE_ENABLED`). Vectors are gated by the
/// presence of the `VectorStore`, code by `CODE_ENABLED`.
static GRAPH_READ_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static GRAPH_WRITE_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[inline]
fn graph_read_enabled() -> bool {
    GRAPH_READ_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}
#[inline]
fn graph_write_enabled() -> bool {
    GRAPH_WRITE_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

enum LineRead {
    Line,
    Eof,
    TooLong,
}

/// Read one newline-terminated line, capping at `max` bytes. Both `buf` (the
/// byte accumulator) and `out` (the decoded line) are caller-owned and reused
/// across calls, so a long-lived connection performs no per-request line
/// allocation — only the decode copy into `out`'s retained capacity.
async fn read_line_capped<R>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    out: &mut String,
    max: usize,
) -> std::io::Result<LineRead>
where
    R: AsyncBufReadExt + Unpin,
{
    buf.clear();
    out.clear();
    let finish = |buf: &[u8], out: &mut String| -> std::io::Result<LineRead> {
        let s = std::str::from_utf8(buf).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Non-UTF-8 input")
        })?;
        out.push_str(s);
        Ok(LineRead::Line)
    };
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            if buf.is_empty() {
                return Ok(LineRead::Eof);
            }
            return finish(buf, out);
        }
        match available.iter().position(|&b| b == b'\n') {
            Some(i) => {
                if buf.len() + i + 1 > max {
                    reader.consume(i + 1);
                    return Ok(LineRead::TooLong);
                }
                buf.extend_from_slice(&available[..=i]);
                reader.consume(i + 1);
                return finish(buf, out);
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

/// Dispatch one framed line (stdio / tcp). Returns the serialized response, or
/// `None` for a notification. `vs` is `Some` only when vector support is enabled.
pub fn dispatch_line(line: &str, kg: &GraphHandle, vs: Option<&VectorStore>) -> Option<String> {
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

/// Dispatch a Streamable-HTTP POST body, which may be a single JSON-RPC message
/// or a batch array. `Ok(None)` means the body held only notifications (HTTP
/// 202, empty body); `Err` means the body was not valid JSON.
pub fn dispatch_http_body(
    body: &str,
    kg: &GraphHandle,
    vs: Option<&VectorStore>,
) -> std::result::Result<Option<Value>, String> {
    let value: Value = serde_json::from_str(body.trim()).map_err(|e| e.to_string())?;
    match value {
        Value::Array(items) => {
            // Batches are rare and never huge — keep Value path for simplicity.
            let responses: Vec<Value> = items
                .into_iter()
                .filter_map(|v| process_value_http(v, kg, vs))
                .collect();
            Ok((!responses.is_empty()).then_some(Value::Array(responses)))
        }
        other => Ok(process_value_http(other, kg, vs)),
    }
}

/// Process one JSON-RPC message for the HTTP transport, converting any
/// `RawResult` back into a `Value` (acceptable since HTTP payloads are typically
/// much smaller in this context). `None` means the message was a notification.
fn process_value_http(value: Value, kg: &GraphHandle, vs: Option<&VectorStore>) -> Option<Value> {
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
    /// `Some` when vector support is enabled (`--vectors`); drives the extra
    /// `vector_*` / `hybrid_search` tools. `None` for a pure knowledge-graph server.
    vs: Option<Arc<VectorStore>>,
}

impl MCPServer {
    /// Build a server. The vector subsystem (usearch index + petgraph mirror) is
    /// only constructed when `config.vectors_enabled` is set; `vec_config` is
    /// ignored otherwise.
    pub fn new(config: Config, vec_config: VectorConfig) -> Result<Self> {
        let path = Path::new(&config.memory_file_path);
        let lru_cache = NonZeroUsize::new(config.lru_cache_size).unwrap_or_else(|| {
            NonZeroUsize::new(10000).expect("10000 > 0")
        });
        let kg = Arc::new(GraphHandle::new(
            path,
            config.durability,
            config.sqlite_tuning(),
            lru_cache,
            config.read_pool_size,
        )?);

        let vs = if config.vectors_enabled {
            Some(Arc::new(VectorStore::with_config(path, &vec_config)?))
        } else {
            None
        };

        // Publish the knowledge-graph exposure flags for the dispatch path.
        use crate::tools::ToolCategory;
        GRAPH_READ_ENABLED.store(
            config.enabled_categories.contains(&ToolCategory::GraphRead),
            std::sync::atomic::Ordering::Relaxed,
        );
        GRAPH_WRITE_ENABLED.store(
            config.enabled_categories.contains(&ToolCategory::GraphWrite),
            std::sync::atomic::Ordering::Relaxed,
        );

        #[cfg(feature = "code")]
        {
            CODE_ENABLED.store(config.code_enabled, std::sync::atomic::Ordering::Relaxed);
            if config.code_enabled {
                // Per-project code databases live in a sibling directory keyed to
                // the main memory file, so distinct memory DBs never collide.
                let base = PathBuf::from(format!("{}.code", config.memory_file_path));
                crate::code_registry::init(
                    base.clone(),
                    config.durability,
                    config.sqlite_tuning(),
                    lru_cache,
                    config.read_pool_size,
                );
                // Code semantic search shares the per-project code databases: an
                // HNSW index is opened on the same files, keyed by symbol entity id.
                crate::code_vec_registry::init(base, config.code_embedding_dims);
            }
        }

        Ok(Self {
            config: Arc::new(config),
            kg,
            vs,
        })
    }

    /// Convenience constructor for a pure knowledge-graph server (no vectors).
    pub fn new_kg(config: Config) -> Result<Self> {
        let mut config = config;
        config.vectors_enabled = false;
        Self::new(config, VectorConfig::new(0))
    }

    /// Expose the shared graph handle (used to drive the HTTP transport).
    pub fn graph(&self) -> Arc<GraphHandle> {
        Arc::clone(&self.kg)
    }

    /// The shared vector store, if vector support is enabled.
    pub fn vector_store(&self) -> Option<Arc<VectorStore>> {
        self.vs.clone()
    }

    /// stdio transport: newline-delimited JSON-RPC over stdin/stdout.
    pub async fn run_stdio(&self) -> Result<()> {
        spawn_maintenance(self.kg.clone());
        spawn_wal_flush(self.kg.clone(), self.config.wal_flush_ms);
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::with_capacity(BUFFER_CAPACITY, stdin);
        let mut stdout = tokio::io::stdout();
        serve_line_conn(&mut reader, &mut stdout, Arc::clone(&self.kg), self.vs.clone()).await
    }

    /// MCP Streamable HTTP transport (POST/GET `/mcp`, JSON or SSE responses).
    pub async fn run_http(&self, addr: &str) -> Result<()> {
        spawn_maintenance(self.kg.clone());
        spawn_wal_flush(self.kg.clone(), self.config.wal_flush_ms);
        crate::http::run(
            addr,
            self.graph(),
            self.vs.clone(),
            self.config.auth_token.clone(),
            self.config.tls_cert.clone(),
            self.config.tls_key.clone(),
        )
        .await
    }
}

/// Spawn a background task that fsyncs committed WAL frames every
/// `interval_ms` milliseconds via a non-blocking passive checkpoint, bounding
/// the durability window in async mode. A zero interval disables the task.
fn spawn_wal_flush(kg: Arc<GraphHandle>, interval_ms: u64) {
    if interval_ms == 0 {
        return;
    }
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
        interval.tick().await; // skip immediate first tick
        loop {
            interval.tick().await;
            let kg = kg.clone();
            tokio::task::spawn_blocking(move || {
                if let Err(e) = kg.checkpoint_passive() {
                    tracing::warn!("WAL flush error: {e}");
                }
            })
            .await
            .ok();
        }
    });
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

/// Drive one line-framed connection (stdio or a single TCP socket): read
/// newline-delimited JSON-RPC requests, write newline-delimited responses.
/// Notifications produce no output. Returns when the peer closes the stream.
/// The dispatch path (graph lock + optional fsync) is offloaded to
/// [`tokio::task::spawn_blocking`] to keep the async reactor responsive (C3).
async fn serve_line_conn<R, W>(
    reader: &mut R,
    writer: &mut W,
    kg: Arc<GraphHandle>,
    vs: Option<Arc<VectorStore>>,
) -> Result<()>
where
    R: AsyncBufReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    let mut line = String::with_capacity(1024);
    let mut read_buf = Vec::with_capacity(1024);
    let mut out = Vec::with_capacity(BUFFER_CAPACITY);

    loop {
        match read_line_capped(reader, &mut read_buf, &mut line, MAX_REQUEST_BYTES).await {
            Ok(LineRead::Eof) => break,
            Ok(LineRead::Line) => {
                let line_copy = line.clone();
                let kg_clone = Arc::clone(&kg);
                let vs_clone = vs.clone();
                let resp = tokio::task::spawn_blocking(move || {
                    dispatch_line(&line_copy, &kg_clone, vs_clone.as_deref())
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

fn process_request(
    req: &JsonRpcRequest,
    kg: &GraphHandle,
    vs: Option<&VectorStore>,
) -> Result<HandlerResult> {
    match req.method.as_str() {
        "initialize" => Ok(HandlerResult::Value(handle_initialize(req, vs.is_some()))),
        "tools/list" => Ok(HandlerResult::Value(handle_tools_list(vs.is_some()))),
        "tools/call" => handle_tools_call(req, kg, vs),
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

/// Extra guidance appended to [`SERVER_INSTRUCTIONS`] when vector support is on.
const VECTOR_INSTRUCTIONS: &str = " Vector search is enabled: use `vector_upsert_embedding` to \
attach embeddings to entities, `vector_search_entities` for semantic search, and `hybrid_search` to \
combine text + vector relevance.";

fn handle_initialize(req: &JsonRpcRequest, vectors_enabled: bool) -> Value {
    // Version negotiation: echo a supported requested revision, else offer latest.
    let protocol_version = req
        .params
        .as_ref()
        .and_then(|p| p.get("protocolVersion"))
        .and_then(Value::as_str)
        .filter(|v| SUPPORTED_PROTOCOL_VERSIONS.contains(v))
        .unwrap_or(LATEST_PROTOCOL_VERSION);

    let instructions = if vectors_enabled {
        format!("{SERVER_INSTRUCTIONS}{VECTOR_INSTRUCTIONS}")
    } else {
        SERVER_INSTRUCTIONS.to_string()
    };

    json!({
        "protocolVersion": protocol_version,
        "capabilities": {
            "tools": { "listChanged": false }
        },
        "serverInfo": {
            "name": "mcp-memory",
            "version": env!("CARGO_PKG_VERSION")
        },
        "instructions": instructions
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

/// The base knowledge-graph tools, parsed from `tools.json` at build time.
fn base_tools() -> &'static Vec<Value> {
    static BASE: std::sync::OnceLock<Vec<Value>> = std::sync::OnceLock::new();
    BASE.get_or_init(|| {
        serde_json::from_str(include_str!("../tools.json"))
            .expect("tools.json is valid JSON compiled at build time")
    })
}

/// The vector tools, parsed from `vector_tools.json` at build time.
fn vector_tools() -> &'static Vec<Value> {
    static VEC: std::sync::OnceLock<Vec<Value>> = std::sync::OnceLock::new();
    VEC.get_or_init(|| {
        serde_json::from_str(include_str!("../vector_tools.json"))
            .expect("vector_tools.json is valid JSON compiled at build time")
    })
}

/// The code-indexing tools, parsed from `code_tools.json` at build time.
#[cfg(feature = "code")]
fn code_tools() -> &'static Vec<Value> {
    static CODE: std::sync::OnceLock<Vec<Value>> = std::sync::OnceLock::new();
    CODE.get_or_init(|| {
        serde_json::from_str(include_str!("../code_tools.json"))
            .expect("code_tools.json is valid JSON compiled at build time")
    })
}

/// `tools/list` response. Each tool is advertised only when its category is
/// enabled, so the server never lists a tool it would reject. Knowledge-graph
/// tools are gated by the graph-read / graph-write flags; vector and code tools
/// by their subsystems being enabled.
fn handle_tools_list(vectors_enabled: bool) -> Value {
    let (read, write) = (graph_read_enabled(), graph_write_enabled());
    let mut all: Vec<Value> = base_tools()
        .iter()
        .filter(|t| {
            t.get("name").and_then(Value::as_str).is_some_and(|n| {
                if tools::is_write_tool(n) { write } else { read }
            })
        })
        .cloned()
        .collect();
    if vectors_enabled {
        all.extend(vector_tools().iter().cloned());
    }
    #[cfg(feature = "code")]
    if code_enabled() {
        all.extend(code_tools().iter().cloned());
    }
    json!({ "tools": all })
}

/// Process-wide flag for the code-indexing subsystem, set once at server
/// startup from `config.code_enabled`. Code tools carry no per-request state
/// (unlike the vector store), so a global flag avoids threading a bool through
/// every dispatch signature.
#[cfg(feature = "code")]
static CODE_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "code")]
fn code_enabled() -> bool {
    CODE_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}
#[cfg(not(feature = "code"))]
const fn code_enabled() -> bool {
    false
}

fn handle_tools_call(
    req: &JsonRpcRequest,
    kg: &GraphHandle,
    vs: Option<&VectorStore>,
) -> Result<HandlerResult> {
    let tool_name = req
        .params
        .as_ref()
        .and_then(|p| p.get("name").and_then(|v| v.as_str()))
        .ok_or_else(|| MCSError::InvalidParams("Missing 'name' parameter".into()))?;

    let tool_args = req.params.as_ref().and_then(|p| p.get("arguments"));

    if tools::is_vector_tool_name(tool_name) {
        let Some(vs) = vs else {
            return Err(MCSError::MethodNotFound(format!(
                "{tool_name} (vector support disabled; start the server with --enable-vectors)"
            )));
        };
        let result = match tool_name {
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
                vector_actions::handle_hybrid_search(vs, kg, tool_args).map(HandlerResult::RawResult)
            }
            "vector_refresh_graph_cache" => {
                vector_actions::handle_refresh_graph_cache(vs, kg, tool_args)
                    .map(HandlerResult::Value)
            }
            "vector_store_stats" => {
                vector_actions::handle_vector_store_stats(vs, kg, tool_args)
                    .map(HandlerResult::Value)
            }
            "vector_batch_upsert" => {
                vector_actions::handle_vector_batch_upsert(vs, kg, tool_args)
                    .map(HandlerResult::Value)
            }
            "vector_get_embedding" => {
                vector_actions::handle_vector_get_embedding(vs, kg, tool_args)
                    .map(HandlerResult::Value)
            }
            "vector_search_by_entity" => {
                vector_actions::handle_vector_search_by_entity(vs, kg, tool_args)
                    .map(HandlerResult::RawResult)
            }
            "vector_recommend" => {
                vector_actions::handle_vector_recommend(vs, kg, tool_args)
                    .map(HandlerResult::RawResult)
            }
            "vector_mmr_search" => {
                vector_actions::handle_vector_mmr_search(vs, kg, tool_args)
                    .map(HandlerResult::RawResult)
            }
            "vector_reindex" => {
                vector_actions::handle_vector_reindex(vs, kg, tool_args).map(HandlerResult::Value)
            }
            other => Err(MCSError::MethodNotFound(other.to_string())),
        };
        return Ok(result.unwrap_or_else(|e| {
            error!("Tool '{tool_name}' error: {e}");
            HandlerResult::Value(tool_error(&e.to_string()))
        }));
    }

    if tools::is_code_tool_name(tool_name) {
        if !code_enabled() {
            return Err(MCSError::MethodNotFound(format!(
                "{tool_name} (code indexing disabled; start the server with --enable-code)"
            )));
        }
        #[cfg(feature = "code")]
        {
            let result = match tool_name {
                "code_index" => {
                    code_actions::handle_code_index(tool_args).map(HandlerResult::Value)
                }
                "code_outline" => {
                    code_actions::handle_code_outline(tool_args).map(HandlerResult::Value)
                }
                "code_search" => {
                    code_actions::handle_code_search(tool_args).map(HandlerResult::Value)
                }
                "code_get_symbol" => {
                    code_actions::handle_code_get_symbol(tool_args).map(HandlerResult::Value)
                }
                "code_watch" => {
                    code_actions::handle_code_watch(tool_args).map(HandlerResult::Value)
                }
                "code_embed" => {
                    code_actions::handle_code_embed(tool_args).map(HandlerResult::Value)
                }
                "code_semantic_search" => {
                    code_actions::handle_code_semantic_search(tool_args).map(HandlerResult::Value)
                }
                other => Err(MCSError::MethodNotFound(other.to_string())),
            };
            return Ok(result.unwrap_or_else(|e| {
                error!("Tool '{tool_name}' error: {e}");
                HandlerResult::Value(tool_error(&e.to_string()))
            }));
        }
        #[cfg(not(feature = "code"))]
        return Err(MCSError::MethodNotFound(format!(
            "{tool_name} (built without the 'code' feature)"
        )));
    }

    // Knowledge-graph category gate: a KG tool is reachable only if it exists
    // AND its category (graph-read for queries, graph-write for mutations) was
    // enabled at startup. Disabled tools are hidden from tools/list, so a call
    // to one is treated as an unknown method.
    let Some(meta) = tools::ALL_TOOLS.iter().find(|t| t.name == tool_name) else {
        return Err(MCSError::MethodNotFound(tool_name.to_string()));
    };
    let category_enabled = if meta.write {
        graph_write_enabled()
    } else {
        graph_read_enabled()
    };
    if !category_enabled {
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
            let r = memory::handle_delete_entities(kg, tool_args);
            if r.is_ok()
                && let Some(vs) = vs
                && let Some(args) = tool_args.and_then(|a| a.get("entityNames")).and_then(|v| v.as_array())
            {
                let names: Vec<String> = args.iter().filter_map(|v| v.as_str().map(String::from)).collect();
                vs.invalidate_entity_cache(&names);
            }
            r.map(HandlerResult::Value)
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


