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

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::json;
use tokio::net::TcpListener;
use tracing::{error, info};

use crate::errors::{MCSError, Result};
use crate::kg::GraphHandle;
use crate::server;

/// Shared state for the HTTP handlers: the graph plus an optional bearer token
/// required on every request when present.
#[derive(Clone)]
pub struct HttpState {
    kg: Arc<GraphHandle>,
    auth_token: Option<Arc<str>>,
}

/// Build the axum router for the HTTP transport. Exposed so tests can drive it
/// with `tower::ServiceExt::oneshot` without binding a socket.
pub fn router(state: HttpState) -> Router {
    Router::new()
        .route("/mcp", post(post_handler).get(get_handler))
        .route("/", post(post_handler).get(get_handler))
        .layer(DefaultBodyLimit::max(server::MAX_REQUEST_BYTES))
        .with_state(state)
}

/// Bind `addr` and serve the HTTP transport until the process is killed.
pub async fn run(addr: &str, kg: Arc<GraphHandle>, auth_token: Option<Arc<str>>) -> Result<()> {
    let listener = TcpListener::bind(addr).await.map_err(MCSError::IoError)?;
    info!(
        "Listening for HTTP (Streamable) MCP on http://{addr}/mcp (auth {})",
        if auth_token.is_some() { "on" } else { "off" }
    );
    let state = HttpState { kg, auth_token };
    axum::serve(listener, router(state)).await.map_err(MCSError::IoError)?;
    Ok(())
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
    // The dispatch path locks the graph and may perform a blocking fsync, so
    // run it off the async worker pool (keeps the HTTP reactor responsive).
    let result = tokio::task::spawn_blocking(move || server::dispatch_http_body(&body, &kg)).await;

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


