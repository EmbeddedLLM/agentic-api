# MCP Gateway Integration

Target: `crates/agentic-server-core/`

References:

- [OpenAI Responses MCP guide](https://developers.openai.com/api/docs/guides/tools-connectors-mcp)
- [MCP tools specification](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
- [MCP resources specification](https://modelcontextprotocol.io/specification/2025-06-18/server/resources)
- [Codex MCP client](https://github.com/openai/codex/tree/main/codex-rs/codex-mcp)

## Goal

Agentic API accepts an OpenAI Responses `type: "mcp"` declaration for tools exposed by a remote MCP server. The
gateway connects to each declared server, discovers its tools with `tools/list`, presents the discovered tools to the
upstream model as function tools, and executes model-selected tools with `tools/call`.

The request declaration contains server identity and connection information, not a tool name:

```json
{
  "type": "mcp",
  "server_label": "counter",
  "server_url": "http://localhost:8000/mcp",
  "allowed_tools": ["increment", "get_value"],
  "require_approval": "never"
}
```

The MCP server owns the tool names returned by `tools/list`. A client cannot add `name` to the MCP declaration to
select an operation.

This phase covers the request, discovery, normalization, registry, and execution path. Aligning public output items
and streaming events with the OpenAI `mcp_call` lifecycle is a separate follow-up.

## Supported MCP surface

The gateway supports MCP tools through this flow:

```text
Responses type:mcp declaration
  -> connect and initialize MCP client
  -> tools/list
  -> filter allowed_tools
  -> normalize discovered tools for the upstream model
  -> model emits an internal function call
  -> tools/call
  -> append a function call output for the next inference round
```

The internal function-tool representation lets a model served by vLLM select a discovered MCP tool. It is not part
of the client request contract.

## MCP resources are not a gateway feature

Agentic API does not implement `resources/list`, `resources/templates/list`, `resources/read`, resource subscriptions,
or a synthetic `read_mcp_resource` tool.

This is intentional. The MCP specification classifies resources as application-controlled context. A host normally
lets a user or application browse or select a resource, reads it, and attaches its content to the model context.
Interactive clients such as Codex and Claude Code own that lifecycle themselves.

The previous function bridge accepted a client-declared function named `read_mcp_resource`, decoded custom metadata
containing MCP server URLs, and translated the function call into `resources/read`. That path was not part of the
OpenAI Responses `type: "mcp"` request contract and has been removed. A function named `read_mcp_resource` is now an
ordinary client-executed function tool.

If resource support is needed in the future, it should be designed as a host-facing context API. It should not be
added by overloading the current MCP tool handler. The future design must define resource discovery, URI selection,
content and MIME handling, attachment limits, authorization, caching, and update subscriptions. Alternatively, an MCP
server can expose a real `read_resource` tool through `tools/list`; the gateway will treat it as an ordinary MCP tool
and invoke it through `tools/call`.

## Components

### `McpClient`

`McpClient` is a thin asynchronous wrapper around an `rmcp` client service. Its gateway execution surface is limited
to MCP tools:

```rust
impl McpClient {
    pub async fn connect(
        server_url: &str,
        headers: Option<HashMap<String, String>>,
    ) -> Result<Self, McpError>;

    pub async fn list_tools(&self) -> Result<Vec<rmcp::model::Tool>, McpError>;

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> Result<rmcp::model::CallToolResult, McpError>;
}
```

The client may also connect to operator-configured stdio MCP servers. Request-provided MCP declarations only accept
remote HTTP URLs.

### `McpClientPool`

`McpClientPool` owns clients keyed by `server_label`. Request-scoped HTTP clients are constructed from
`McpToolParam`. Gateway configuration may construct HTTP or stdio clients through `McpServerEntry`.

Request-provided URLs allow loopback hosts by default. Additional trusted hostnames may be configured through
`AGENTIC_MCP_ALLOWED_HOSTS`. URL validation, pinned DNS addresses, disabled automatic proxy discovery, and disabled
redirects prevent later routing changes from bypassing the configured trust boundary.

### `GatewayExecutors`

`GatewayExecutors` caches discovered MCP handlers by `server_label`. A server label may appear only once in a request.
For an uncached declaration it:

1. Builds an MCP connection from the declaration.
2. Calls `tools/list`.
3. Applies `allowed_tools`.
4. Creates one `McpDiscoveredHandler` per remaining tool.
5. Caches the handlers under the declaration's `server_label`.

An empty final allowed set is a configuration error.

### `McpHandler`

`McpHandler` normalizes and executes discovered MCP tools. An executable handler contains the `McpClient` bound to
the server that advertised the tool. A spec-only handler has no client and exists only for
`ResponsesTool::to_function_tools()` normalization. The handler no longer switches between `tools/call` and
`resources/read`.

```rust
pub struct McpHandler {
    client: Option<Arc<McpClient>>,
}
```

During execution, the registry entry supplies `McpDiscoveredToolParam`, which contains:

- the public `server_label`
- the public MCP `tool_name`
- the internal model-visible name
- the tool schema returned by `tools/list`

`McpHandler::execute()` reads that identity, validates the model arguments as a JSON object, calls `tools/call`, and
serializes the MCP result for the next inference round.

### `ToolRegistry`

`ToolRegistry::build_with_handlers()` owns the complete routing table, including discovered MCP tools. Discovery
happens before `RequestPayload::to_upstream_request()` normalizes the request:

```text
ResponsesTool::Mcp
  -> GatewayExecutors::mcp_handler()
  -> declaration._agentic_discovered_tools is populated
  -> each discovered handler is inserted into ToolRegistry
  -> ResponsesTool::to_function_tools()
  -> McpHandler::spec_from_param(...).normalize(...)
```

The `_agentic_discovered_tools` field is internal state and is rejected on the public request wire.

Each upstream function name includes both server and tool identity, for example `mcp__counter__increment`. Names are
sanitized and bounded to the upstream function-name limit. The registry retains the original server and tool identity
for gateway dispatch and public output mapping.

## Turn execution

The gateway's existing tool loop handles MCP tools together with other gateway-executed built-in tools:

```text
build request-scoped registry
  -> discover MCP tools
  -> normalize request for upstream inference
  -> receive internal function call
  -> registry dispatches to McpHandler
  -> McpClient tools/call
  -> append function call output for the next upstream round
```

Tool execution failures become failed tool call output and are returned to the model for the next round; they do not
automatically fail the whole Responses request.
