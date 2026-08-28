# `POST /v1/responses`: repository entrypoints and full round trip

This document is a Python-friendly map of the Rust repository and the complete logical path of a `POST /v1/responses`
request. It follows the local executor path, including rehydration, tool normalization, upstream inference, the built-in
tool loop, persistence, and the response returned to the client. It also shows the shorter pass-through path.

The short answer is that the HTTP gateway starts in
[`crates/agentic-server/src/main.rs`](crates/agentic-server/src/main.rs#L307), while the main Responses orchestration
starts in [`ExecuteRequest::run`](crates/agentic-server-core/src/executor/engine.rs#L555).

## 1. Rust mental model for a Python developer

Rust/Cargo separates a repository into a few concepts that Python often leaves implicit:

| Rust concept | Rough Python analogy | Meaning here |
| --- | --- | --- |
| Cargo workspace | monorepo | The root [`Cargo.toml`](Cargo.toml#L1) groups packages under `crates/`. |
| Package | installable Python distribution | A directory with a `Cargo.toml`, such as `crates/agentic-server`. |
| Crate | one compiled library or executable | A package can produce a library crate and multiple binary crates. |
| Target | one build output | A library, binary, test, example, or benchmark produced by Cargo. |
| `src/main.rs` | `if __name__ == "__main__":` | Default executable entrypoint for a package. |
| `src/bin/name.rs` | another console-script entrypoint | An additional executable named `name`. |
| `src/lib.rs` | public package/module interface | Exports code shared by binaries and other crates. |
| `mod server;` | load a local module | Makes `server.rs` part of the crate's module tree. |
| `use foo::Bar` | `from foo import Bar` | Brings a name into scope. |
| `pub` | exported/public | Makes an item visible outside its module or crate. |
| `#[tokio::main]` | create an async event loop, then run `main()` | Starts the Tokio runtime so `main` can use async I/O. |
| `Result<T, E>` and `?` | return a value or raise/propagate an exception | `?` returns early when the result is an error. |
| `.await` | `await` | Suspends the current task while I/O proceeds; it does not block the Tokio worker thread. |
| `Arc<T>` | shared, reference-counted application state | `Arc::clone` cheaply clones a pointer, not the underlying database/client state. |

The key difference is that one Cargo package can have more than one executable. Finding one `main.rs` does not imply
that it is the repository's only entrypoint.

## 2. Workspace, crates, and executable entrypoints

The root workspace is declared in [`Cargo.toml`](Cargo.toml#L1) and contains three packages:

| Package | Role |
| --- | --- |
| `agentic-server` | Axum HTTP/WebSocket transport, configuration, server lifecycle, and two executables. |
| `agentic-server-core` (`agentic_core` in Rust imports) | Framework-independent types, rehydration, inference, tools, tool loop, and persistence. |
| `agentic-praxis` | Placeholder for a future Praxis integration. |

The dependency direction is intentional: the transport crate calls the core crate; core does not depend on Axum.

`agentic-server` produces two executables:

| Executable | Entrypoint | Purpose |
| --- | --- | --- |
| `agentic-server` | [`crates/agentic-server/src/main.rs`](crates/agentic-server/src/main.rs#L307) | The HTTP/WebSocket gateway. Cargo infers this default binary from `src/main.rs`. |
| `agentic` | [`crates/agentic-server/src/bin/agentic.rs`](crates/agentic-server/src/bin/agentic.rs#L11) | User-facing launcher for the gateway and coding harnesses. It is declared explicitly in [`crates/agentic-server/Cargo.toml`](crates/agentic-server/Cargo.toml#L52). |

For API request handling, follow `agentic-server`; the `agentic` launcher is not in the request path.

## 3. Gateway startup chain

The gateway process starts at `#[tokio::main] async fn main()`:

```text
crates/agentic-server/src/main.rs::main
  ├─ initialize tracing
  ├─ parse CLI flags/environment/config.toml
  ├─ construct agentic_core::config::Config
  └─ server::run(...) or server::run_with_llm(...)
       ├─ optionally discover OIDC configuration
       ├─ wait for the upstream LLM to become ready
       ├─ build_state(...)
       │    ├─ ProxyState::new(...)
       │    └─ ExecutionContext::from_config(...)
       │         ├─ response/conversation storage handlers
       │         ├─ shared reqwest HTTP client
       │         ├─ built-in tool executors
       │         └─ upstream LLM base URL
       ├─ build_router_with_auth(...)
       ├─ TcpListener::bind(...)
       └─ axum::serve(...)
```

The important source points are:

- Process entry and configuration: [`main.rs`](crates/agentic-server/src/main.rs#L307)
- State construction: [`server.rs::build_state`](crates/agentic-server/src/server.rs#L35)
- Router, listener, and server: [`server.rs::serve_gateway`](crates/agentic-server/src/server.rs#L52)
- Shared executor dependencies: [`ExecutionContext`](crates/agentic-server-core/src/executor/request.rs#L52)

`ExecutionContext::responses_url()` appends `/v1/responses` to the configured LLM base URL. For example,
`http://127.0.0.1:8000` becomes `http://127.0.0.1:8000/v1/responses`.

## 4. API and route surface

Routes are wired in
[`build_router_with_auth`](crates/agentic-server/src/app.rs#L242):

| Method and path | Purpose |
| --- | --- |
| `POST /v1/responses` | OpenAI-compatible Responses API over JSON or server-sent events (SSE). |
| `GET /v1/responses` | Responses API WebSocket upgrade. It reaches the same executor from a long-lived session loop. |
| `POST /v1/responses/compact` | Explicit context compaction. |
| `POST /v1/conversations` | Create a durable conversation. |
| `POST /v1/messages` | Anthropic-compatible Messages API. |
| `POST /v1/messages/count_tokens` | Anthropic-compatible token counting. |
| `GET /v1/models` | Upstream model listing with gateway handling as configured. |
| `GET /health` | Process liveness. |
| `GET /ready` | Storage and upstream readiness. |

`/health` and `/ready` are public. The remaining routes are protected when OIDC authentication is configured.

## 5. The first fork: pass through or execute locally

The HTTP handler is
[`responses`](crates/agentic-server/src/handler/http/responses.rs#L49). It first retains two representations of the
same body:

- `bytes`: the original raw request body, used when passing the request through unchanged.
- `payload`: a deserialized, typed [`RequestPayload`](crates/agentic-server-core/src/types/request_response.rs#L14),
  used when the gateway must inspect or modify the request.

The fork is exactly:

```rust
if should_execute {
    execute_responses(&state, parts, payload).await
} else {
    proxy_responses(&state, parts, bytes).await
}
```

`should_execute` is true when any of these apply:

- `store` is true;
- `previous_response_id` is present;
- `conversation_id` is present;
- the input contains a compaction item or compaction trigger;
- non-empty `context_management` is present; or
- the request contains any tool declaration other than a plain `ResponsesTool::Function`.

That last test is structural: `Custom`, `Namespace`, and `Unknown` also select the executor path, even though they are
not all built-in tools and are not all locally executable. Selecting the executor does not by itself mean the gateway
will execute every declared tool.

`RequestPayload.store` defaults to `true`, so a simple request normally uses the local executor unless it explicitly
sends `"store": false`. A request with `store: false` can still use the executor when another condition above requires
local behavior.

The two branches are:

```text
responses()
├─ should_execute == false
│  └─ proxy_responses()
│     └─ proxy_request()
│        └─ pass request through to upstream; no local executor or persistence
│
└─ should_execute == true
   └─ execute_responses()
      └─ ExecuteRequest::new(payload, Arc::clone(&state.exec_ctx))
         └─ .with_auth(auth)
            └─ .run().await
```

The exact link to `ExecuteRequest::run()` is in
[`execute_responses`](crates/agentic-server/src/handler/http/responses.rs#L29). Its result has two successful shapes:

- `Either::Left(ResponsePayload)`: a complete non-streaming JSON response.
- `Either::Right(BoxStream)`: a stream of complete SSE frames.

### Inbound identity versus upstream LLM credentials

There are two different authentication concerns:

1. **Inbound OIDC identity authentication** decides whether the caller may use the gateway.
2. **Upstream LLM authentication** supplies the credential used when the gateway calls vLLM or another inference
   service.

When OIDC is enabled for the OpenAI-compatible routes, the
[`require_oidc`](crates/agentic-server/src/auth.rs#L351) middleware verifies the caller's bearer token, stores the
authenticated principal in request extensions, and removes the inbound `Authorization` and OpenAI `x-api-key` headers.
They therefore cannot be mistaken for an upstream LLM credential. The handler then falls back to the configured
`OPENAI_API_KEY`; if that key is configured and non-empty, it is used for the upstream call. Otherwise the executor
sends no upstream bearer credential.

Without OIDC, the executor path's `extract_bearer` uses a caller bearer token when supplied and otherwise falls back to
the configured `OPENAI_API_KEY`. On the pass-through path, eligible caller credential headers are preserved; the proxy
injects the configured key only when the client supplied neither `Authorization` nor `x-api-key`.

## 6. `ExecuteRequest::run()` and rehydration

[`ExecuteRequest::run`](crates/agentic-server-core/src/executor/engine.rs#L555) begins with:

```rust
let ctx = rehydrate_conversation(self.payload, &self.exec_ctx).await?;
```

Yes: this is the item-history rehydration step. The project-specific meaning of **rehydration** is loading stored items,
restoring their order and effective request settings, and building the input for a continuation.

[`rehydrate_conversation`](crates/agentic-server-core/src/executor/rehydrate.rs#L26) builds a `RequestContext` with:

- `original_request`: an unchanged copy used for storage semantics and response metadata;
- `enriched_request`: the request that can be augmented before upstream inference;
- `new_input_items`: only the newly submitted input, retained for persistence;
- a generated response ID;
- optional conversation ID and version information.

It then selects one path:

```text
conversation_id present
  → load a conversation snapshot and prepend its item history

previous_response_id present
  → rehydrate the stored-response chain
  → restore effective tools/tool_choice when appropriate
  → prepend its item history

neither ID present
  → use only the new input items
```

Supplying both IDs is rejected. After rehydration, `run()` chooses the client transport requested by
`original_request.stream`:

```text
stream == false → run_blocking(...) → Either::Left(ResponsePayload)
stream == true  → run_stream(...)   → Either::Right(BoxStream)
```

Here “blocking” means **non-streaming API behavior**—wait for one complete upstream JSON body. It does not mean blocking
the Tokio worker thread; the network operations are asynchronous.

Both modes then use
[`run_until_gateway_tools_complete`](crates/agentic-server-core/src/executor/engine.rs#L98), which has another important
branch:

```text
input has compaction_trigger
  → run_compaction_trigger(...)
  → run one direct blocking summarization inference
  → return a compaction response without entering the ordinary tool loop

ordinary input
  → run_gateway_tool_loop(...)
```

Inside the ordinary loop, `maybe_compact_context(...)` can also perform an automatic blocking compaction inference
before the main response inference for a round. Consequently, the ten-round limit described later caps ordinary
tool-loop/main-response rounds, not every upstream model invocation: automatic compaction can add a model call.

## 7. Tool registry versus tool normalization

These are separate operations with separate purposes.

### 7.1 Request-scoped tool registry: how calls will be routed

Before the inference rounds, [`run_gateway_tool_loop`](crates/agentic-server-core/src/executor/engine.rs#L119) calls:

```rust
ToolRegistry::build_with_handlers(tools, &mut executors).await?
```

The [`ToolRegistry`](crates/agentic-server-core/src/tool/registry.rs#L159) maps each model-visible tool name to routing
metadata. It can:

- flatten Codex namespace member names consistently;
- discover tools from Model Context Protocol (MCP) servers;
- associate names with their original tool type and configuration;
- attach an executor for a gateway-executed built-in tool; and
- retain mappings needed to restore the public output shape after inference.

The registry answers: **“When the model emits this tool name, what is it and who executes it?”** It is request-scoped
runtime routing state, not part of the Responses wire format.

### 7.2 Tool normalization: what shape the upstream accepts

Normalization happens later, immediately before each upstream request, in
[`RequestPayload::to_upstream_request`](crates/agentic-server-core/src/types/request_response.rs#L123). Both
`fetch_blocking_payload` and `fetch_stream_payload` call it.

It:

1. resolves namespace members to flat, model-visible names;
2. validates the resolved declarations;
3. calls `ResponsesTool::to_function_tools` for every declaration, producing zero or more upstream function tools;
4. wraps each produced function tool as `UpstreamTool::Function`;
5. normalizes `tool_choice`; and
6. always sends `parallel_tool_calls: false`, asking the upstream not to generate parallel function calls.

The per-tool conversion is in
[`ResponsesTool::to_function_tools`](crates/agentic-server-core/src/tool/normalize.rs#L90). A declaration can expand
to more than one function tool (notably MCP discovery), one function tool, or none. `FileSearch`, `CodeInterpreter`,
and `Unknown` are currently skipped during normalization; they produce no upstream tool declaration. For example, a
public web search declaration becomes one upstream function tool. The public meaning is preserved even though the
upstream wire shape changes.

`parallel_tool_calls: false` does not mean gateway tool calls are all executed one at a time. If a round nevertheless
contains multiple gateway-executed built-in tool calls, the gateway executes them concurrently with an order-preserving
sliding window bounded at five calls. It then assembles results in model-output order before continuing the loop.

The distinction is:

```text
rehydration       = Which prior items and effective settings belong in this request?
tool registry     = How will a returned tool call be classified, restored, and executed?
tool normalization= What function-tool JSON shape must be sent to the upstream inference server?
```

## 8. What “payload” means

**Payload** is ordinary networking terminology, not a special Rust feature. It means the meaningful data carried by a
request or response, separate from transport details such as HTTP headers and status codes.

In this path:

| Name | Meaning |
| --- | --- |
| `RequestPayload` | Typed Rust representation of the client's Responses JSON request body. |
| `UpstreamRequest` | Normalized request body sent to the upstream LLM endpoint. |
| `ResponsePayload` | Typed, complete Responses object for one upstream inference round (later combined across rounds). |
| `StreamPayload` | Internal wrapper containing the accumulated `ResponsePayload` plus deferred streaming events. It is not a separate public API response. |

`ResponsePayload` includes the response ID, model, status, output items, token usage, continuation/conversation IDs,
and error or incomplete details.

## 9. Fetching a blocking or streaming payload

Both functions call the same upstream `/v1/responses` endpoint and ultimately produce a complete `ResponsePayload`.
They differ in how the upstream response is transported.

### 9.1 `fetch_blocking_payload()`

[`fetch_blocking_payload`](crates/agentic-server-core/src/executor/upstream.rs#L35) performs:

```text
enriched RequestPayload
  → to_upstream_request(false)        # normalize; set stream=false
  → serialize_to_string(...)
  → fetch_response_json(...)
  → POST upstream /v1/responses
  → await the complete JSON body
  → ResponseAccumulator::from_json(...)
  → finalize one ResponsePayload
```

### 9.2 `fetch_stream_payload()`

[`fetch_stream_payload`](crates/agentic-server-core/src/executor/upstream.rs#L58) performs:

```text
enriched RequestPayload
  → to_upstream_request(true)         # normalize; set stream=true
  → serialize_to_string(...)
  → call_inference(...)
  → POST upstream /v1/responses
  → read raw SSE data lines incrementally
  → ResponseAccumulator/FunctionSseTranslator normalize them into typed event frames
  ├─ emit eligible typed streaming events toward the client
  └─ accumulate all normalized events with ResponseAccumulator
  → finalize one complete ResponsePayload
  → return StreamPayload { payload, deferred_events }
```

The stream still needs a complete in-memory payload because the gateway must inspect all output items after each round:
did the model finish, emit a client-executed function call, or request a gateway-executed built-in tool? The accumulator
also provides the complete state needed for persistence.

```text
upstream SSE events ───────────────→ eligible events reach the client incrementally
          │
          └─ ResponseAccumulator ──→ complete ResponsePayload
                                      ├─ tool-loop decision
                                      └─ persistence
```

## 10. Where the HTTP request is really sent

For a non-streaming round, the call chain is:

```text
fetch_blocking_payload()
  → fetch_response_json()
    → send_request()
      → client.post(url).headers(...).body(upstream_json)
      → optional bearer_auth(...)
      → req.send().await
```

The actual network operation is this line in
[`send_request`](crates/agentic-server-core/src/executor/inference.rs#L77):

```rust
let resp = req.send().await ...?;
```

Building `client.post(...).body(...)` only creates a request builder. `.send().await` opens/uses the connection and
sends the HTTP request. The upstream is a separate HTTP service, normally vLLM; the Rust gateway does not call the
model as an in-process function.

Conceptually:

```text
RequestPayload
  │ to_upstream_request()
  ▼
normalized UpstreamRequest
  │ serialize_to_string()
  ▼
JSON String
  │ reqwest::Client::post().body()
  ▼
HTTP request builder
  │ req.send().await
  ▼
upstream LLM: {llm_base_url}/v1/responses
```

The streaming path reaches the same `send_request()` through
[`call_inference`](crates/agentic-server-core/src/executor/inference.rs#L146), then consumes `resp.bytes_stream()` as
complete SSE lines.

## 11. Repeated inference and the built-in tool loop

“Repeat inference if necessary” is implemented by the `for` loop inside
[`run_gateway_tool_loop`](crates/agentic-server-core/src/executor/engine.rs#L119):

```rust
for round in 0..MAX_GATEWAY_TOOL_ROUNDS {
    // fetch_blocking_payload(...) or fetch_stream_payload(...)
    // inspect output, execute applicable built-in tool calls
    // classify the round
}
```

`MAX_GATEWAY_TOOL_ROUNDS` is 10. Because the main upstream fetch is inside the loop, every new iteration is a new main
response inference round. The cap applies to those tool-loop rounds, not every model invocation: the
`maybe_compact_context(...)` call at the start of a round may first add a blocking automatic-compaction inference. After
the main response, the gateway:

1. restores model output to its public tool representation where needed;
2. inspects the current output items;
3. identifies client-executed function calls;
4. executes applicable gateway-executed built-in tool calls;
5. adds public output items to the combined response; and
6. calls [`classify_round`](crates/agentic-server-core/src/executor/gateway.rs#L65).

`LoopDecision` has four outcomes:

| Decision | Condition | Effect |
| --- | --- | --- |
| `RequiresClientAction` | At least one client-executed function call is present. | Return the turn so the client can execute it and later submit a function call output. This takes precedence if built-in tool calls are also present. |
| `Done` | No gateway-executed built-in tool produced a call output. | Finalize and return the response. |
| `Incomplete(reason)` | Built-in tools ran on the last permitted round. | Return accumulated work with `status: "incomplete"`, keeping a consistent continuation history. |
| `Continue` | Built-in tools ran and rounds remain. | Append replayable model output items plus gateway function-call outputs to the enriched input, then begin the next main response round. |

More precisely, `append_output_items_to_input` converts every replayable current output item—not only tool calls—back
into input form. `append_tool_outputs` then adds the gateway-produced `function_call_output` items. The `Continue` arm
also sets `tool_choice` to `auto`. There is no explicit Rust `continue;` statement: after the match arm ends, control
reaches the bottom of the `for` body naturally and the next iteration begins.

Example:

```text
Round 0
  user input
    → inference #1
    → model emits web-search tool call
    → gateway executes web search
    → append replayable model output + gateway function call output to enriched input
    → LoopDecision::Continue

Round 1
  enriched input now includes the earlier call and output
    → inference #2
    → model emits final assistant message
    → no gateway-executed built-in tool output
    → LoopDecision::Done
```

A **turn** is the user-visible unit of work; it can contain several internal **inference rounds**.

## 12. Persistence and return to the client

For non-streaming execution, [`run_blocking`](crates/agentic-server-core/src/executor/engine.rs#L402) waits for the tool
loop to finish, calls `persist_if_needed`, and returns the final `ResponsePayload`.

For streaming execution, `run_stream` relays intermediate events while the loop runs. When the final payload is ready,
it persists first and only then exposes the terminal `response.completed`/`response.incomplete` event. This ordering
prevents a client that disconnects immediately after the terminal event from cancelling persistence.

[`persist_if_needed`](crates/agentic-server-core/src/executor/persist.rs#L21) persists when the original request has any
of:

- `store: true`;
- a previous response ID; or
- a conversation ID.

Completed and incomplete turns are stored. Explicit conversation requests use the conversation handler; other flows,
including previous-response continuations, use the response handler.

Finally, the Axum handler converts the successful executor result. The error representation depends on whether the SSE
response has already started:

```text
Either::Left(ResponsePayload) → axum::Json(...) → HTTP JSON response
Either::Right(BoxStream)      → sse_response(...) → HTTP SSE response
pre-stream/non-stream error   → HTTP status + typed JSON error response
error after SSE is established→ typed SSE error event + data: [DONE]
```

Parsing, routing, rehydration, and other failures that happen before `ExecuteRequest::run()` returns a stream can still
be represented as normal HTTP errors. Once the handler has returned a successful SSE response and the stream is being
consumed, its HTTP status is already committed; inference, fatal tool-pipeline/orchestration, persistence, or task
failures are therefore emitted inside the stream as an error event followed by `[DONE]`. Ordinary built-in tool
execution failures and timeouts are normally represented as failed function call outputs and fed back to the model,
rather than becoming stream-level errors.

## 13. Full end-to-end round trip

```text
Client
  │ POST /v1/responses
  ▼
Axum router: build_router_with_auth
  ▼
responses() handler
  ├─ parse raw bytes + typed RequestPayload
  ├─ should_execute == false
  │    └─ pass request through to upstream and return its response
  │
  └─ should_execute == true
       ▼
     execute_responses()
       ▼
     ExecuteRequest::new(...).with_auth(...).run()
       ▼
     rehydrate_conversation()
       ├─ preserve original request
       ├─ build enriched request
       └─ prepend stored item history/effective settings when continuing
       ▼
     run_blocking() or run_stream()
       ▼
     run_until_gateway_tools_complete()
       ├─ compaction_trigger
       │    └─ run_compaction_trigger() → direct blocking summarization → return
       └─ ordinary request
            ▼
          run_gateway_tool_loop()
          ├─ build request-scoped ToolRegistry
          └─ for each main response/tool-loop round, at most 10:
            ├─ maybe_compact_context() may add a blocking compaction inference
            ├─ to_upstream_request(stream)
            │    ├─ flatten namespace members
            │    ├─ validate tools
            │    └─ normalize each declaration to zero or more upstream function tools
            ├─ serialize normalized request to JSON
            ├─ POST {llm_base_url}/v1/responses via req.send().await
            ├─ parse full JSON, or normalize raw SSE lines and accumulate ResponsePayload
            ├─ inspect model output
            ├─ execute gateway-executed built-in tool calls concurrently (ordered, maximum five in flight)
            └─ LoopDecision
                 ├─ Continue
                 │    ├─ append replayable model output items
                 │    ├─ append gateway function-call outputs
                 │    └─ next main response round
                 ├─ RequiresClientAction → return function call to client
                 ├─ Done                 → finalize response
                 └─ Incomplete           → finalize partial response
       ▼
     persist completed/incomplete state when required
       ▼
     JSON or SSE
       ▼
Client
```

If an error occurs before streaming begins, the client receives an HTTP JSON error. If it occurs after SSE has begun,
the client receives an SSE error event followed by `[DONE]`.

This is the complete logical round trip for the executor path. Individual database queries, specific tool executors,
event translation details, and WebSocket session mechanics are deeper subflows, but they do not change this top-level
lifecycle.

## 14. Recommended reading order

1. [`ARCHITECTURE.md`](ARCHITECTURE.md) — crate boundaries and the maintained architecture map.
2. [`TERMINOLOGY.md`](TERMINOLOGY.md) — normative project vocabulary.
3. [`crates/agentic-server/src/main.rs`](crates/agentic-server/src/main.rs#L307) — process startup.
4. [`crates/agentic-server/src/server.rs`](crates/agentic-server/src/server.rs#L35) — state construction and serving.
5. [`crates/agentic-server/src/app.rs`](crates/agentic-server/src/app.rs#L242) — all routes.
6. [`handler/http/responses.rs`](crates/agentic-server/src/handler/http/responses.rs#L49) — pass-through/executor fork.
7. [`executor/engine.rs`](crates/agentic-server-core/src/executor/engine.rs#L119) — orchestration and repeated inference.
8. [`executor/rehydrate.rs`](crates/agentic-server-core/src/executor/rehydrate.rs#L26) — item-history loading.
9. [`types/request_response.rs`](crates/agentic-server-core/src/types/request_response.rs#L108) — request, upstream, and response payloads.
10. [`executor/upstream.rs`](crates/agentic-server-core/src/executor/upstream.rs#L35) — blocking and streaming fetches.
11. [`executor/inference.rs`](crates/agentic-server-core/src/executor/inference.rs#L55) — actual upstream HTTP transport.
12. [`tool/registry.rs`](crates/agentic-server-core/src/tool/registry.rs#L159) and
    [`tool/normalize.rs`](crates/agentic-server-core/src/tool/normalize.rs#L75) — routing versus normalization.
13. [`executor/persist.rs`](crates/agentic-server-core/src/executor/persist.rs#L21) — storage decision and write path.

## 15. Useful run commands

Run the gateway explicitly:

```bash
cargo run -p agentic-server --bin agentic-server -- \
  --llm-api-base http://127.0.0.1:8000
```

Inspect the launcher:

```bash
cargo run -p agentic-server --bin agentic -- --help
```

## 16. Read-only inspection commands used during the walkthrough

The walkthrough used repository search and source inspection; it did not change production code or send a request:

```bash
git status --short

rg --files -g 'Cargo.toml' -g '*.rs' -g 'ARCHITECTURE.md' \
  -g 'TERMINOLOGY.md'

rg -n 'Router|\.route\(|fn main|ExecuteRequest|run_gateway_tool_loop' crates/

rg -n 'build_with_handlers|to_upstream_request|to_function_tools|normalize_sse_line' \
  crates/agentic-server-core/src -g '*.rs'

rg -n 'enum LoopDecision|fn classify_round|execute_output_calls|append_tool_outputs' \
  crates/agentic-server-core/src/executor/gateway.rs

rg -n 'MAX_GATEWAY_TOOL_ROUNDS' crates/agentic-server-core/src

sed -n '…' <relevant-file>
nl -ba <relevant-file>

cargo metadata --no-deps --format-version 1
```

The only writes made for this documentation request are `round-trip.md` and `round-trip.html`.
