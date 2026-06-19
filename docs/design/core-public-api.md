# Design: `agentic-core` Public API

> Status: Active — implementation in progress
> References: [ADR-03](../adr/ADR-03_gateway_integration.md), [Issue #42](https://github.com/vllm-project/agentic-api/issues/42),
> [Issue #54](https://github.com/vllm-project/agentic-api/issues/54), [Praxis #354](https://github.com/praxis-proxy/praxis/issues/354)
> Owner: @ashwing (tool dispatch, loop control, streaming tee) + @maralbahari (base loop, store integration)

---

## Foundation: PR #46

[PR #46](https://github.com/vllm-project/agentic-api/pull/46) by @maralbahari implements the base executor loop for text-only stateful conversations:

| Function | File | What it does |
|----------|------|--------------|
| `execute()` | `executor/engine.rs` | Entry point — rehydrate → infer → persist |
| `rehydrate_conversation()` | `executor/engine.rs` | Load history from store, build enriched request |
| `call_inference()` | `executor/engine.rs` | Returns `impl Stream` of SSE lines (sync fn, not async — stream is lazy) |
| `persist_response()` | `executor/engine.rs` | Save response + items to store (takes handlers as explicit params) |
| `ResponseAccumulator` | `executor/accumulator.rs` | SSE state machine — collects stream into ResponsePayload |
| `ExecutionContext` | `executor/request.rs` | Runtime deps: handlers, HTTP client, LLM URL |
| `RequestContext` | `executor/request.rs` | Per-turn state: original + enriched request, IDs |

This design builds on top of PR #46 — it does not duplicate or replace that work.

---

## What This Design Adds

The base loop handles text messages. This design extends it with:

1. **Tool dispatch** — detect function_call items in output, execute via traits, loop back
2. **Loop control** — `LoopDecision` enum driving re-entry with iteration limits
3. **Streaming tee** — forward SSE to client in real-time while accumulating for tool detection
4. **Extended SSE events** — function_call, reasoning, file_search, web_search event types
5. **Tool executor traits** — MCP, web_search, vector_store as pluggable implementations
6. **Codex CLI compatibility** — recognize Codex client-side tool types and route them without server execution

---

## Implementation Phases

Each phase = one PR with tests. Phases are ordered by dependency.

### Phase 1: SSE Event Normalizer Module (lands on main — no PR #46 dependency)

**PR scope:** New `events/` module in `agentic-core` — separate from executor, no dependency on PR #46.

Per @maralbahari's feedback ([PR #46 discussion](https://github.com/vllm-project/agentic-api/pull/46#discussion_r3352104210)): the SSE event handling should be a **separate core module** to avoid bloating the accumulator. Design draws from PydanticAI's `StreamedResponse._process_event()`.

```
crates/agentic-core/src/
  events/
    mod.rs          // pub mod normalize; pub mod types;
    types.rs        // SSEEventType (28+ variants) + typed EventPayload enum
    normalize.rs    // normalize_sse_line(&str) -> EventFrame { event_type, payload }
```

- `EventFrame { event_type: SSEEventType, payload: EventPayload }` — typed output from raw SSE
- `normalize_sse_line()` — zero-copy where possible, maps `data: {...}` to typed frame
- Expanded `SSEEventType` covering all Responses API events
- Unit tests verifying correct parsing of function_call, reasoning, and tool-call events

```rust
pub enum SSEEventType {
    ResponseCreated,
    ResponseInProgress,
    ResponseOutputItemAdded,
    ResponseOutputItemDone,         // detect completed tool calls
    ResponseOutputTextDelta,
    ResponseOutputTextDone,
    FunctionCallArgumentsDelta,     // streaming function args
    FunctionCallArgumentsDone,      // complete function call
    ContentPartAdded,
    ContentPartDone,
    ReasoningSummaryTextDelta,
    ReasoningSummaryTextDone,
    ResponseCompleted,
    ResponseFailed,
    ResponseIncomplete,
    // Built-in tool events
    FileSearchCallSearching,
    FileSearchCallCompleted,
    WebSearchCallSearching,
    WebSearchCallCompleted,
    // Catch-all
    Other,
}
```

Once PR #46 merges, a follow-up PR refactors the accumulator to consume `EventFrame` instead of doing inline JSON parsing.

**Size:** ~300 lines | **Blocked by:** nothing (lands on main) | **Target:** 3rd merged PR

---

### Phase 2: Loop Control + Tool Dispatch (depends on PR #46)

**PR scope:** `executor/dispatch.rs`, `executor/tool_context.rs`, extend `engine.rs`.

Core contribution — the agentic loop re-entry mechanism:

```rust
pub enum LoopDecision {
    Continue(Vec<InputItem>),   // tool results to append, re-enter inference
    Done,                       // no tool calls, response is final
    Incomplete(String),         // max iterations or unrecoverable failure
}

pub async fn dispatch_tools(
    output: &[OutputItem],
    tool_ctx: &ToolContext,
    iteration: usize,
) -> ExecutorResult<LoopDecision>

/// Initially non-streaming only (returns Left). Streaming support added in Phase 3.
pub async fn execute_loop(
    request: RequestPayload,
    exec_ctx: Arc<ExecutionContext>,
    tool_ctx: &ToolContext,
) -> ExecutorResult<ResponsePayload>
```

`execute_loop` wraps PR #46's functions in a tool-dispatch loop:
1. Rehydrate (delegates to PR #46's `rehydrate_conversation`)
2. Call inference (delegates to PR #46's `call_inference` — returns stream lazily)
3. Accumulate response (via `ResponseAccumulator::from_stream`)
4. Check output for `OutputItem::FunctionCall` → `dispatch_tools` → loop, client action, or done
5. Persist final response (delegates to PR #46's `persist_response` with explicit handlers)

**Phase 2 is non-streaming only.** The tool loop inspects the full accumulated response before deciding. Streaming + tool dispatch (forwarding events to client while detecting tool calls) requires Phase 3's tee pattern.

`ToolContext` holds optional executor references:

```rust
pub struct ToolContext {
    pub mcp_executor: Option<Arc<dyn McpToolExecutor>>,
    pub web_search: Option<Arc<dyn WebSearchProvider>>,
    pub vector_store: Option<Arc<dyn VectorStoreClient>>,
    pub max_iterations: usize,
}
```

**Size:** ~400 lines | **Blocked by:** PR #46 merge | **Target:** first feature PR (Phase 2 of committer track)

---

### Phase 3: Streaming Tee (depends on PR #46)

**PR scope:** `executor/stream_tee.rs`, refactor `run_stream` path.

PR #46's streaming path accumulates everything before emitting to client. This replaces it with a tee:

```rust
pub struct StreamTee {
    client_tx: mpsc::Sender<String>,     // forward to client
    accumulator: ResponseAccumulator,     // detect tool calls
}

impl StreamTee {
    pub fn split(
        raw_stream: impl Stream<Item = Result<String, ExecutorError>>,
        conversation_id: Option<&str>,
    ) -> (BoxStream, impl Future<Output = ResponsePayload>)
}
```

Returns two handles:
- `BoxStream` — yields SSE events to client in real-time
- `Future<ResponsePayload>` — resolves when stream completes, contains accumulated output for tool detection

This enables the real-time streaming requirement from ADR-01 §3 — events should reach the client as they arrive, interleaved with the tool loop, rather than buffered until completion.

**Size:** ~300 lines | **Blocked by:** PR #46 merge | **Target:** feature PR

---

### Phase 4: Tool Executor Traits + Mock Implementations (depends on Phase 2)

**PR scope:** `tools/` module.

```rust
// Native async traits (Rust 1.75+, no #[async_trait] boxing needed)
pub trait McpToolExecutor: Send + Sync {
    fn execute(
        &self,
        tool_name: &str,
        arguments: &Value,
        server_config: &Value,
    ) -> impl Future<Output = Result<Value, ExecutorError>> + Send;
}

pub trait WebSearchProvider: Send + Sync {
    /// context_size: "low" | "medium" | "high" — controls result verbosity
    fn search(
        &self,
        query: &str,
        context_size: &str,
    ) -> impl Future<Output = Result<Value, ExecutorError>> + Send;
}

pub trait VectorStoreClient: Send + Sync {
    fn search(
        &self,
        store_id: &str,
        query: &str,
        max_results: u32,
    ) -> impl Future<Output = Result<Vec<Value>, ExecutorError>> + Send;
}
```

This PR includes mock implementations for integration testing (in-memory tool executors that return canned responses). Real implementations (MCP client, Brave search, Qdrant) come in later PRs.

**Note:** The dispatch layer routes by tool type: function calls → `McpToolExecutor`, file_search → `VectorStoreClient` (@franciscojavierarceo's OGX integration, PR #34), web_search → `WebSearchProvider`.

**Size:** ~500 lines | **Blocked by:** Phase 2 | **Target:** feature PR

---

## Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | `ToolContext` separate from `ExecutionContext` | Keeps PR #46's struct focused on inference; tool deps are additive |
| D2 | `LoopDecision` carries tool results directly | Avoids mutating shared state between dispatch and re-entry |
| D3 | Streaming tee as separate module, not refactor of accumulator | Preserves PR #46's non-streaming path unchanged |
| D4 | Traits for tool executors, not concrete types | Enables OGX (PR #34), mock testing, and future providers |
| D5 | Phase 1 lands on main independently of PR #46 | Separate `events/` module has no executor dependency — unblocks Phase 2 while #46 is still in review |
| D6 | Tool traits compatible with OGX (PR #34) | OGX is one backend behind the trait interface — doesn't constrain the dispatch API |

---

## Praxis Filter Mapping

How the complete pipeline maps to @leseb's proposed filter chain:

| # | Praxis Filter | Core Function | Phase | Owner |
|---|---------------|---------------|-------|-------|
| 0 | `request_validate` | `validate_request()` | Future | — |
| 1 | `response_store` (init) | `init_store()` | Future | — |
| 2 | `rehydrate` | `rehydrate_conversation()` | PR #46 | @maralbahari |
| 3 | `file_resolve` | `resolve_files()` | Future | @franciscojavierarceo |
| 4 | `tool_parse` | `parse_tools()` | Future | @franciscojavierarceo |
| 5 | `responses_proxy` | `call_inference()` | PR #46 | @maralbahari |
| 5.5 | `event_normalize` | `normalize_sse_line()` | Phase 1 | @ashwing |
| 6 | `stream_events` | `transform_stream()` / tee | Phase 3 | @ashwing |
| 7 | `tool_dispatch` | `dispatch_tools()` | Phase 2 | @ashwing |
| 8 | `mcp_tool` | `McpToolExecutor::execute()` | Phase 4 | @ashwing |
| 9 | `web_search` | `WebSearchProvider::search()` | Phase 4 | @franciscojavierarceo |
| 10 | `file_search` | `VectorStoreClient::search()` | Phase 4 | @franciscojavierarceo |
| 11 | `compact` | `compact_context()` | Future | — |
| 12 | `reasoning` | `summarize_reasoning()` | Future | — |
| 13 | `response_store` (resp) | `persist_response()` | PR #46 | @maralbahari |

---

## Codex Integration

Allow `agentic-api` to serve as the upstream layer for Codex CLI in the coming PR by @haoshan98, related to [Issue #54](https://github.com/vllm-project/agentic-api/issues/54).
`agentic-api` should accept Codex CLI traffic, route inference to vLLM-supported models, preserve
`previous_response_id` and conversation persistence, and pass client-owned tool calls back to Codex for local
execution.

The immediate compatibility gap is request parsing and pass-through behavior. `agentic-api` already supports
`type: "function"`, but it does not yet recognize the Responses API tool shapes Codex uses for local/client tools:
`namespace`, `tool_search`, and `custom`. Today those request tools can fail before they reach upstream inference.
This section scopes the Codex integration to accepting those tool types losslessly and returning client-owned tool
calls to Codex CLI for local execution.

Server-hosted tool types such as `file_search`, `web_search_preview`, and `code_interpreter` remain future
server-side work. They should not be conflated with this Codex compatibility pass.

### Codex Tool Type Taxonomy

Codex CLI sends tool declarations that are executed locally by the CLI. For this phase, `agentic-core` only needs
to recognize and preserve these shapes, normalize them for vLLM when necessary, and avoid treating them as
gateway-executed tools.

| Tool type | Executor | Core behavior |
|-----------|----------|---------------|
| `function` | Codex CLI by default | Already supported on the wire. Codex requests should return calls to Codex unless configuration marks the tool as gateway-owned. |
| `namespace` | Codex CLI | Accept the model-facing container shape and preserve child function metadata. Calls still arrive as `function_call` with an optional namespace. |
| `tool_search` | Codex CLI | Accept deferred-discovery shape, preserve `execution`, and return calls/output handling to Codex when `execution = "client"`. |
| `custom` | Codex CLI | Accept free-form/grammar tool shape and preserve `format` metadata for Codex. |

The key requirement is compatibility, not server execution. These tools should pass through `agentic-api`
without request validation failures, and any client-owned model-emitted calls should be surfaced to Codex CLI
rather than executed inside the gateway.

The key distinction is execution owner, not just wire type. `function` is a shared wire type: Codex can own local
functions, while future gateway integrations may also expose server-executed functions. The request normalizer must
classify every model-visible tool before inference and carry that registry through response handling.

### Public Type Additions

The current `ResponsesTool = FunctionTool` alias is too narrow for Codex. Replace it with a tagged tool enum that
preserves unknown shapes while giving Codex-used variants first-class names.

Do not implement this as `#[serde(tag = "type")]` wrapping the existing `FunctionTool` struct directly, because
`FunctionTool` already stores the wire `type` field. Either use variant-specific payload structs that omit the
already-consumed tag, as sketched below, or write manual deserialization that preserves the raw object.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesTool {
    Known(KnownResponsesTool),
    Unknown(Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum KnownResponsesTool {
    #[serde(rename = "function")]
    Function(ResponsesFunctionTool),
    #[serde(rename = "namespace")]
    Namespace(CodexNamespaceTool),
    #[serde(rename = "tool_search")]
    ToolSearch(CodexToolSearchTool),
    #[serde(rename = "custom")]
    Custom(CodexCustomTool),
}

pub struct ResponsesFunctionTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
    pub strict: Option<bool>,
    #[serde(default)]
    pub defer_loading: bool,
}

pub struct CodexNamespaceTool {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub tools: Vec<CodexNamespaceMember>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CodexNamespaceMember {
    #[serde(rename = "function")]
    Function(ResponsesFunctionTool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSearchExecution {
    Server,
    Client,
}

pub struct CodexToolSearchTool {
    pub description: Option<String>,
    pub execution: Option<ToolSearchExecution>,
    pub parameters: Option<Value>,
}

pub struct CodexCustomTool {
    pub name: String,
    pub description: Option<String>,
    pub format: Option<Value>,
    #[serde(default)]
    pub defer_loading: bool,
}
```

The storage and upstream request paths should preserve raw unknown tools as `Value`. Unknown tool types must not
be executed by default.

### Codex Call Shapes

Codex's local router treats namespace as part of a function tool name, not as a separate payload type:

| Response item | Local Codex payload | Gateway behavior |
|---------------|---------------------|------------------|
| `function_call` | `ToolPayload::Function { arguments }` | Preserve optional `namespace` and return the call to Codex when the tool is client-owned. |
| `tool_search_call` with `execution = "client"` | `ToolPayload::ToolSearch` | Return to Codex for local deferred discovery. |
| Hosted `tool_search_call` | Provider-owned | Do not execute locally; provider/upstream owns it. |
| `custom_tool_call` | `ToolPayload::Custom { input }` | Preserve free-form `input`, not JSON-schema function arguments. |

`custom_tool_call` is for free-form/custom Responses tools, including grammar-based patch or code tools. It should
not be normalized as JSON-schema function arguments unless the adapter can reconstruct the original custom call
exactly.

### Normalization And Registry

Codex compatibility needs two related operations:

1. Build an upstream-safe tool list for the selected inference backend.
2. Keep a lossless registry that maps model-emitted calls back to the original client-visible tool declaration.

If vLLM only accepts flat function declarations, `tool_search` and `custom` become request normalization concerns.
`namespace` is mostly model-facing spec organization: `ToolSpec::Namespace` wraps function tools, and the model
still emits `function_call` with an optional `namespace`. Any backend-specific flattening is an adapter detail,
not the public semantics of the tool.

```rust
pub struct NormalizedTools {
    pub upstream_tools: Vec<Value>,
    pub registry: ToolRegistry,
}

pub struct ToolName {
    pub namespace: Option<String>,
    pub name: String,
}

pub enum ToolExecutionOwner {
    Client,
    Gateway,
}

pub struct ToolRegistryEntry {
    pub owner: ToolExecutionOwner,
    pub original_type: String,
    pub original_name: ToolName,
    pub model_name: ToolName,
    pub original_tool: Value,
}
```

For a namespace tool:

```json
{
  "type": "namespace",
  "name": "mcp__github",
  "tools": [
    { "name": "create_issue", "description": "Create issue", "parameters": {} }
  ]
}
```

the normalizer records a registry entry keyed by the split `ToolName`:

```text
ToolName { namespace: Some("mcp__github"), name: "create_issue" }
  -> owner = Client, original_type = "namespace"
```

Because the registry keys by split `ToolName`, two tools named `run` in different namespaces can coexist. When the
upstream response includes a `namespace` field on a `function_call`, preserve it. When an upstream backend only
returns an encoded flat name, recover the namespace and child tool name from the registry before returning the
response to Codex. This likely requires extending `FunctionToolCall` with:

```rust
pub namespace: Option<String>,
```

`tool_search` may need to be adapted for backends that only understand functions, but the registry must preserve
`execution` and map client-executed `tool_search_call` / `tool_search_output` items back to the
Responses-compatible shape. Hosted/non-client tool search remains provider-owned and should not be handled by the
local Codex route. `custom` carries free-form `input` instead of JSON arguments, so the registry must retain
`format` and enough raw metadata to reconstruct the Codex-visible call.

### Pass-Through Behavior

Routing rules:

1. Client-owned calls (`function`, namespaced `function`, client-executed `tool_search`, and `custom`) are returned
   to Codex without gateway execution.
2. Gateway-owned calls execute inside `agentic-api` only when explicitly supported by a registered executor.
3. `namespace`, `tool_search`, and `custom` request declarations must not fail deserialization or validation.
4. The registry preserves the original request tool shape so the returned call can be interpreted by Codex CLI.
5. Unknown tool types are not executed by default. Preserve them when possible and reject them only when the
   upstream cannot receive a safe normalized declaration.

For Codex-owned calls, the gateway should not synthesize tool outputs. It persists the assistant call item, returns
the call to Codex, and expects Codex to continue the conversation with the corresponding tool output item after
local execution.

The loop needs an explicit client-action decision so this path is not confused with either `Done` or
`Continue`:

```rust
pub enum LoopDecision {
    Continue(Vec<InputItem>),
    RequiresClientAction(Vec<OutputItem>),
    Done,
    Incomplete(String),
}
```

### Auto-Approval Model Alias

[Issue #54](https://github.com/vllm-project/agentic-api/issues/54) also notes Codex's auto-approval request path. MVP support should add a simple model alias map in
configuration:

```toml
[model_aliases]
codex-auto-review = "real-upstream-model"
```

`ExecutionContext` resolves aliases before `call_inference()`. A Codex-specific `/v1/models` response with model metadata can come later; the alias map is sufficient to unblock CLI compatibility without expanding the public API.

---

## Open Questions

1. **`execute_loop` vs refactoring `execute`:** Should the loop wrapper be a new function or replace PR #46's `execute()`? Pending maralbahari's response on PR #46 review.
2. **Streaming tee ownership model:** `Arc<Mutex<>>` vs channel-based accumulation. Will prototype both in Phase 3 PR.
3. **ResponseStore trait unification:** PR #33 has separate `ConversationStore` + `ResponseStore`. Keep separate or unify? Defer until Phase 4 when we need to abstract over them.
