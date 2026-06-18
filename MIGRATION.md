# Migration Guide — mcp-memory 1.x → 2.0.0

**Release theme:** MCP specification compliance (protocol revision **`2025-11-25`**).

Version 2.0.0 is a **major** release because **tool failures changed from
JSON-RPC protocol errors to `isError` results**. Successful tool results are
unchanged — they were already returned as spec-compliant `content` arrays — so
most clients need no changes. The release also adds **optional authentication**
for the TCP and HTTP transports.

---

## Breaking changes

### 1. Tool failures are returned as results, not protocol errors

**Before (1.x)** — a failing `tools/call` returned a JSON-RPC error:

```jsonc
{ "jsonrpc": "2.0", "id": 1, "error": { "code": -32602, "message": "Missing 'name'..." } }
```

**After (2.0.0)** — the same failure returns a `CallToolResult` with
`isError: true`, so the model can read it and self-correct:

```jsonc
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": { "content": [{ "type": "text", "text": "Missing 'name'..." }], "isError": true }
}
```

Protocol-level errors (malformed JSON-RPC, missing `name`, unknown
tool/method) are still returned as JSON-RPC `error` objects.

**Migration:** check `result.isError` on every `tools/call` response instead of
relying solely on the JSON-RPC `error` field.

### 2. Negotiated protocol version is now `2025-11-25`

`initialize` returns `"protocolVersion": "2025-11-25"` by default and negotiates:
a supported requested revision (`2025-11-25`, `2025-06-18`, `2025-03-26`,
`2024-11-05`) is echoed back; otherwise the latest is offered. Clients pinned to
`2024-11-05` keep working.

---

## New in 2.0.0

### Authentication (opt-in)

The TCP and HTTP transports can now require a bearer token. **stdio is never
authenticated** (it is inherently local).

```bash
# Inline token
mcp-memory --transport tcp  --auth-token "s3cr3t"
mcp-memory --transport http --auth-token "s3cr3t"

# From a file (trimmed; an empty file is rejected — fail closed)
mcp-memory --transport http --auth-token-file /run/secrets/mcp_token

# From the environment
MCP_MEMORY_AUTH_TOKEN=s3cr3t mcp-memory --transport tcp
```

- **TCP:** the client sends the token as the **first line**, before any
  JSON-RPC traffic. A wrong/missing token gets a JSON-RPC `-32001` error and the
  connection is closed.
- **HTTP:** the token goes in the `Authorization` header
  (`Authorization: Bearer s3cr3t`). A wrong/missing token gets `401 Unauthorized`.
- Comparison is constant-time.

If no token is configured, behaviour is unchanged (unauthenticated) — so this is
**not** a breaking change, but binding a non-loopback address without a token
exposes the whole graph to the network.

### Other additions

- **`instructions`** field in `InitializeResult`.
- **Protocol version negotiation** in `initialize`.

---

## Not yet implemented (roadmap)

Intentionally **not** advertised as capabilities until implemented:

| Feature | Notes |
|---|---|
| `resources/*` (`mem://…` URIs) | entities/graph/search as readable resources |
| `prompts/*` | `explore-entity`, `find-connections` |
| `completion/complete` | autocomplete entity / relation-type names |
| `logging`, progress, cancellation | long traversals (`find_all_paths`), `compact`, `export` |
| TLS on HTTP transport | terminate TLS at a proxy for now |

---

## Upgrade checklist

- [ ] Reinstall: `cargo install mcp-memory`.
- [ ] Check `result.isError` on `tools/call` responses (tool failures are no
      longer JSON-RPC errors).
- [ ] If exposing TCP/HTTP beyond loopback, set `--auth-token` /
      `--auth-token-file` / `MCP_MEMORY_AUTH_TOKEN` and update clients to send it.
- [ ] Verify your client tolerates `protocolVersion: "2025-11-25"` (or pin an
      older supported revision in `initialize`).
