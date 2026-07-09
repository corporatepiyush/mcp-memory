//! MCP **Streamable HTTP** transport (the 2025-03-26 transport that
//! superseded the older HTTP+SSE pair).
//!
//! * `POST /mcp` — the client sends one JSON-RPC message (or a batch array).
//!   The reply is delivered as `application/json` by default, or as a one-shot
//!   `text/event-stream` (SSE) event when the client `Accept`s it. A body of
//!   only notifications gets `202 Accepted` with no content.
//! * `GET /mcp` — opens a standalone server→client SSE stream. This server has
//!   no server-initiated messages, so the stream simply stays open with
//!   keep-alives; it exists for spec compliance.
//!
//! `/` is also wired to the same handlers for convenience. The JSON-RPC
//! semantics are identical to the stdio and TCP transports — only framing
//! differs (see [`crate::server::dispatch_http_body`]).
//!
//! Two extra routes serve a **browser knowledge-graph viewer** (HTTP transport
//! only):
//! * `GET /ui` — a self-contained, dependency-free HTML/canvas graph explorer.
//! * `GET /ui/graph` — the JSON the viewer renders (entities, relations, type
//!   legend, stats). Gated behind the same `graph-read` permission as
//!   `read_graph`; auth via the `Authorization` header or a `?token=` fallback.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tracing::{error, info};

use crate::errors::{MCSError, Result};
use crate::kg::GraphHandle;
use crate::server;
use crate::vector_store::VectorStore;

/// The graph viewer's static assets, embedded at build time (served from `/ui`).
const UI_INDEX_HTML: &str = include_str!("ui/index.html");
const UI_CSS: &str = include_str!("ui/graph.css");
const UI_JS: &str = include_str!("ui/graph.js");

/// Upper bound on entities returned to the viewer in one `GET /ui/graph` load,
/// mirroring the `read_graph` search cap. Keeps the payload — and the browser's
/// force layout — bounded for very large graphs.
const MAX_UI_NODES: usize = 1000;

/// Upper bound on hops for a single `GET /ui/expand` traversal (double-click to
/// expand). One hop matches the Neo4j "expand relationships" gesture; the cap
/// bounds a single interaction's payload.
const MAX_UI_EXPAND_DEPTH: u32 = 3;

/// Shared state for the HTTP handlers: the graph, the optional vector store, and
/// an optional bearer token required on every request when present.
#[derive(Clone)]
pub struct HttpState {
    kg: Arc<GraphHandle>,
    vs: Option<Arc<VectorStore>>,
    auth_token: Option<Arc<str>>,
}

/// Build the axum router for the HTTP transport. Exposed so tests can drive it
/// with `tower::ServiceExt::oneshot` without binding a socket.
pub fn router(state: HttpState) -> Router {
    Router::new()
        .route("/mcp", post(post_handler).get(get_handler))
        .route("/", post(post_handler).get(get_handler))
        .route("/ui", get(ui_handler))
        .route("/ui/graph.css", get(ui_css_handler))
        .route("/ui/graph.js", get(ui_js_handler))
        .route("/ui/graph", get(ui_graph_handler))
        .route("/ui/search", get(ui_search_handler))
        .route("/ui/expand", get(ui_expand_handler))
        .layer(DefaultBodyLimit::max(server::MAX_REQUEST_BYTES))
        .with_state(state)
}

/// Bind `addr` and serve the HTTP transport until the process is killed.
///
/// When `tls_cert` and `tls_key` are both set, the transport is served over TLS
/// (HTTPS); otherwise it stays plaintext. The caller (`config.rs`) guarantees
/// the two are set together.
pub async fn run(
    addr: &str,
    kg: Arc<GraphHandle>,
    vs: Option<Arc<VectorStore>>,
    auth_token: Option<Arc<str>>,
    tls_cert: Option<std::path::PathBuf>,
    tls_key: Option<std::path::PathBuf>,
) -> Result<()> {
    let auth = if auth_token.is_some() { "on" } else { "off" };
    let state = HttpState { kg, vs, auth_token };

    if let (Some(cert), Some(key)) = (tls_cert, tls_key) {
        let tls = crate::tls::server_config(&cert, &key)
            .await
            .map_err(MCSError::IoError)?;
        let socket_addr = resolve_addr(addr)?;
        info!("Listening for HTTPS (Streamable) MCP on https://{socket_addr}/mcp (TLS, auth {auth})");
        axum_server::bind_rustls(socket_addr, tls)
            .serve(router(state).into_make_service())
            .await
            .map_err(MCSError::IoError)?;
    } else {
        let listener = TcpListener::bind(addr).await.map_err(MCSError::IoError)?;
        info!("Listening for HTTP (Streamable) MCP on http://{addr}/mcp (auth {auth})");
        axum::serve(listener, router(state))
            .await
            .map_err(MCSError::IoError)?;
    }
    Ok(())
}

/// Resolve a `host:port` string to a single `SocketAddr` for `axum_server`,
/// which binds an address rather than an already-bound listener.
fn resolve_addr(addr: &str) -> Result<std::net::SocketAddr> {
    use std::net::ToSocketAddrs;
    addr.to_socket_addrs()
        .map_err(MCSError::IoError)?
        .next()
        .ok_or_else(|| {
            MCSError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("could not resolve bind address '{addr}'"),
            ))
        })
}

fn wants_sse(headers: &HeaderMap) -> bool {
    headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|a| a.contains("text/event-stream"))
}

/// `true` when the request is allowed: either no token is configured, or the
/// `Authorization` header carries the expected bearer token.
fn authorized(state: &HttpState, headers: &HeaderMap) -> bool {
    match state.auth_token {
        None => true,
        Some(ref expected) => headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|presented| server::token_matches(presented, expected)),
    }
}

async fn post_handler(State(state): State<HttpState>, headers: HeaderMap, body: String) -> Response {
    if !authorized(&state, &headers) {
        return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
    }
    let kg = state.kg;
    let vs = state.vs;
    // The dispatch path locks the graph and may perform a blocking fsync, so
    // run it off the async worker pool (keeps the HTTP reactor responsive).
    let result = tokio::task::spawn_blocking(move || {
        server::dispatch_http_body(&body, &kg, vs.as_deref())
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
        // Body held only notifications → nothing to return.
        Ok(None) => StatusCode::ACCEPTED.into_response(),
        Ok(Some(value)) => {
            if wants_sse(&headers) {
                // One JSON-RPC reply delivered as a single SSE event, then close.
                let json = serde_json::to_string(&value).unwrap();
                let stream = futures::stream::once(async move {
                    Ok::<Event, Infallible>(Event::default().data(json))
                });
                Sse::new(stream).into_response()
            } else {
                Json(value).into_response()
            }
        }
        Err(e) => {
            // Malformed JSON body → JSON-RPC parse error.
            let resp = json!({
                "jsonrpc": "2.0",
                "error": { "code": -32700, "message": format!("Parse error: {e}") },
                "id": null
            });
            (StatusCode::BAD_REQUEST, Json(resp)).into_response()
        }
    }
}

async fn get_handler(State(state): State<HttpState>, headers: HeaderMap) -> Response {
    if !authorized(&state, &headers) {
        return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
    }
    // No server-initiated messages: an open, keep-alive'd stream for compliance.
    let stream = futures::stream::pending::<std::result::Result<Event, Infallible>>();
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Like [`authorized`], but also accepts the bearer token from a `token` query
/// parameter. A browser navigating to `/ui/graph` can't set request headers, so
/// the viewer passes the token this way (or via the `Authorization` header when
/// scripted). When no token is configured, access is open (as with `authorized`).
fn authorized_ui(state: &HttpState, headers: &HeaderMap, query_token: Option<&str>) -> bool {
    authorized(state, headers)
        || matches!(
            state.auth_token,
            Some(ref expected) if query_token.is_some_and(|t| server::token_matches(t, expected))
        )
}

/// `GET /ui` — serve the browser graph viewer's HTML shell. The shell and its
/// `/ui/graph.css` + `/ui/graph.js` assets hold no graph data, so they are served
/// without auth; the data they fetch (`/ui/graph`, `/ui/expand`) is what carries
/// the auth + permission gate.
async fn ui_handler() -> Response {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        UI_INDEX_HTML,
    )
        .into_response()
}

/// `GET /ui/graph.css` — the viewer stylesheet (static asset, no auth).
async fn ui_css_handler() -> Response {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        UI_CSS,
    )
        .into_response()
}

/// `GET /ui/graph.js` — the viewer application script (static asset, no auth).
async fn ui_js_handler() -> Response {
    (
        [(header::CONTENT_TYPE, "application/javascript; charset=utf-8")],
        UI_JS,
    )
        .into_response()
}

/// Shared auth + `graph-read` gate for the viewer's data endpoints
/// (`/ui/graph`, `/ui/search`, `/ui/expand`). Returns the error `Response` to
/// send back, or `None` when the request may proceed.
fn ui_data_gate(
    state: &HttpState,
    headers: &HeaderMap,
    params: &HashMap<String, String>,
) -> Option<Response> {
    if !authorized_ui(state, headers, params.get("token").map(String::as_str)) {
        return Some((StatusCode::UNAUTHORIZED, "Unauthorized").into_response());
    }
    if !server::graph_read_enabled() {
        return Some(
            (
                StatusCode::FORBIDDEN,
                "graph-read tools are disabled; start the server with --enable-graph-read (or --enable-all) to view the graph",
            )
                .into_response(),
        );
    }
    None
}

fn parse_usize(params: &HashMap<String, String>, key: &str, default: usize) -> usize {
    params.get(key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Run a blocking JSON-payload builder off the async reactor (the graph lock may
/// block) and map its `Result<String>` to an HTTP response. Shared by the
/// viewer's `/ui/graph` and `/ui/search` data endpoints.
async fn ui_json<F>(kg: Arc<GraphHandle>, what: &'static str, build: F) -> Response
where
    F: FnOnce(&GraphHandle) -> Result<String> + Send + 'static,
{
    match tokio::task::spawn_blocking(move || build(&kg)).await {
        Ok(Ok(json)) => ([(header::CONTENT_TYPE, "application/json")], json).into_response(),
        Ok(Err(e)) => {
            error!("{what} error: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
        Err(join_err) => {
            error!("{what} task panicked: {join_err}");
            (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response()
        }
    }
}

/// `GET /ui/graph` — a page of the whole graph for the viewer: entities, the
/// relations among them, the entity-type legend, overall stats, and a pagination
/// cursor. Requires the `graph-read` category (like `read_graph`) and the same
/// bearer-token gate as the MCP endpoints. Query params: `entityType` (filter),
/// `offset`, `limit` (capped at [`MAX_UI_NODES`]), and `token` (auth fallback).
async fn ui_graph_handler(
    State(state): State<HttpState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if let Some(resp) = ui_data_gate(&state, &headers, &params) {
        return resp;
    }
    let entity_type = params.get("entityType").filter(|s| !s.is_empty()).cloned();
    let offset = parse_usize(&params, "offset", 0);
    let limit = parse_usize(&params, "limit", 300).clamp(1, MAX_UI_NODES);
    ui_json(state.kg, "/ui/graph", move |kg| {
        build_graph_payload(kg, entity_type.as_deref(), offset, limit)
    })
    .await
}

/// `GET /ui/search` — a page of FTS5 matches for the viewer's search box, in the
/// same `{entities, relations, entityTypes, stats, page}` shape as `/ui/graph`
/// (the matched nodes; the user double-clicks to expand their relationships).
/// Same auth + `graph-read` gate. Query params: `q` (the query; prefix-matched),
/// `entityType` (filter), `offset`, `limit` (capped at [`MAX_UI_NODES`]),
/// and `token`.
async fn ui_search_handler(
    State(state): State<HttpState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if let Some(resp) = ui_data_gate(&state, &headers, &params) {
        return resp;
    }
    let query = params.get("q").map(|s| s.trim().to_string()).unwrap_or_default();
    let entity_type = params.get("entityType").filter(|s| !s.is_empty()).cloned();
    let offset = parse_usize(&params, "offset", 0);
    let limit = parse_usize(&params, "limit", 100).clamp(1, MAX_UI_NODES);
    ui_json(state.kg, "/ui/search", move |kg| {
        build_search_payload(kg, &query, entity_type.as_deref(), offset, limit)
    })
    .await
}

/// Attach the viewer's shared metadata to a `{entities, relations}` payload: the
/// entity-type legend, the graph-wide totals, and the pagination cursor
/// (`offset` / `limit` / `returned` / `hasMore`) that drives the Prev/Next
/// controls without a second round-trip.
fn augment_payload(
    graph: &mut Value,
    type_counts: Vec<(String, usize)>,
    entities_total: usize,
    relations_total: usize,
    offset: usize,
    limit: usize,
    returned: usize,
    has_more: bool,
) {
    if let Value::Object(m) = graph {
        let types: Vec<Value> = type_counts
            .into_iter()
            .map(|(t, c)| json!({ "type": t, "count": c }))
            .collect();
        m.insert("entityTypes".into(), Value::Array(types));
        m.insert(
            "stats".into(),
            json!({ "entities": entities_total, "relations": relations_total }),
        );
        m.insert(
            "page".into(),
            json!({ "offset": offset, "limit": limit, "returned": returned, "hasMore": has_more }),
        );
    }
}

/// Assemble the `/ui/graph` JSON: a page of the graph from
/// [`GraphHandle::read_graph_filtered`] plus the shared viewer metadata.
/// `hasMore` compares this page against the scope total (the filtered type's
/// count, or the whole-graph entity count) so the viewer can enable Next.
fn build_graph_payload(
    kg: &GraphHandle,
    entity_type: Option<&str>,
    offset: usize,
    limit: usize,
) -> Result<String> {
    let graph_str = kg.read_graph_filtered(entity_type, offset, limit)?;
    let mut graph: Value = serde_json::from_str(&graph_str).map_err(MCSError::JsonError)?;

    let type_counts = kg.entity_type_counts();
    let entities_total = kg.get_entity_count().unwrap_or(0);
    let relations_total = kg.get_relation_count().unwrap_or(0);
    let scope_total = match entity_type {
        Some(t) if !t.is_empty() => {
            type_counts.iter().find(|(n, _)| n == t).map_or(0, |(_, c)| *c)
        }
        _ => entities_total,
    };
    let returned = graph.get("entities").and_then(Value::as_array).map_or(0, |a| a.len());
    let has_more = offset.saturating_add(returned) < scope_total;

    augment_payload(
        &mut graph, type_counts, entities_total, relations_total, offset, limit, returned, has_more,
    );
    serde_json::to_string(&graph).map_err(MCSError::JsonError)
}

/// Turn a free-text search box query into a safe FTS5 MATCH expression: keep
/// alphanumeric/underscore tokens (dropping punctuation that would otherwise be
/// FTS operators and silently fail the query), AND them together, and make the
/// final token a prefix (`term*`) for a natural search-as-you-type feel.
fn fts_query(raw: &str) -> String {
    let tokens: Vec<String> = raw
        .split_whitespace()
        .map(|t| t.chars().filter(|c| c.is_alphanumeric() || *c == '_').collect::<String>())
        .filter(|t| !t.is_empty())
        .collect();
    let n = tokens.len();
    tokens
        .into_iter()
        .enumerate()
        .map(|(i, t)| if i + 1 == n { format!("{t}*") } else { t })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Assemble the `/ui/search` JSON: a page of FTS5 matches from
/// [`GraphHandle::search_nodes_filtered`] as `{entities, relations: []}` (the
/// matched nodes; the user double-clicks to expand their relationships) plus the
/// shared viewer metadata. Fetches `limit + 1` to detect `hasMore` cheaply.
fn build_search_payload(
    kg: &GraphHandle,
    query: &str,
    entity_type: Option<&str>,
    offset: usize,
    limit: usize,
) -> Result<String> {
    let fts = fts_query(query);
    let mut hits = kg.search_nodes_filtered(&fts, entity_type, offset, limit.saturating_add(1));
    let has_more = hits.len() > limit;
    hits.truncate(limit);
    let returned = hits.len();

    let mut graph = json!({ "entities": hits, "relations": [] });
    let type_counts = kg.entity_type_counts();
    let entities_total = kg.get_entity_count().unwrap_or(0);
    let relations_total = kg.get_relation_count().unwrap_or(0);
    augment_payload(
        &mut graph, type_counts, entities_total, relations_total, offset, limit, returned, has_more,
    );
    serde_json::to_string(&graph).map_err(MCSError::JsonError)
}

/// `GET /ui/expand` — the neighbourhood of one entity, for the viewer's
/// double-click-to-expand traversal. Returns `{entities, relations}` (the same
/// shape as `/ui/graph`) from [`GraphHandle::neighbors`], which the viewer merges
/// into the current graph. Same auth + `graph-read` gate as `/ui/graph`.
///
/// Query params: `name` (required, the entity to expand), `depth` (1..=
/// [`MAX_UI_EXPAND_DEPTH`], default 1), `direction` (`outgoing` / `incoming` /
/// `both`, default both), and `token` (auth fallback).
async fn ui_expand_handler(
    State(state): State<HttpState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if let Some(resp) = ui_data_gate(&state, &headers, &params) {
        return resp;
    }
    let Some(name) = params.get("name").filter(|s| !s.is_empty()).cloned() else {
        return (StatusCode::BAD_REQUEST, "missing 'name' parameter").into_response();
    };
    let depth = params
        .get("depth")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(1)
        .clamp(1, MAX_UI_EXPAND_DEPTH);
    // `Direction::parse` expects the uppercase MCP spelling; default is `Both`.
    let direction = crate::kg::Direction::parse(
        params
            .get("direction")
            .map(|s| s.to_uppercase())
            .as_deref(),
    );

    let kg = state.kg;
    let result =
        tokio::task::spawn_blocking(move || kg.neighbors(&name, direction, None, depth)).await;

    match result {
        Ok(Ok(json)) => (
            [(header::CONTENT_TYPE, "application/json")],
            json,
        )
            .into_response(),
        // An unknown entity is a client error (bad `name`), not a server fault.
        Ok(Err(MCSError::InvalidParams(msg))) => (StatusCode::NOT_FOUND, msg).into_response(),
        Ok(Err(e)) => {
            error!("/ui/expand error: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
        Err(join_err) => {
            error!("/ui/expand task panicked: {join_err}");
            (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response()
        }
    }
}


