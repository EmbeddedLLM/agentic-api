# Design: Codex CLI Integration

> **References:** [Issue #54](https://github.com/vllm-project/agentic-api/issues/54),
> [PR #67](https://github.com/vllm-project/agentic-api/pull/67),
> [PR #84](https://github.com/vllm-project/agentic-api/pull/84),
> [PR #91](https://github.com/vllm-project/agentic-api/pull/91)
> **Owner:** @haoshan98 for Codex compatibility. Latest `main` owns the shared tool framework lineage from PR #67.

---

## Summary

`agentic-api` can sit between Codex CLI and a vLLM-backed Responses-compatible model endpoint.

Codex can declare grouped tools with `type: "namespace"`, while vLLM-compatible upstreams accept ordinary
`type: "function"` tool declarations. The gateway keeps the Codex-facing namespace shape at the request and storage
boundary, flattens namespace members only at an upstream boundary, and restores model-visible flat function calls back
to Codex's public `{ namespace, name }` shape in final and streaming responses.

The important split:

- **Codex compatibility:** preserve Codex request/response shapes, namespace identity, and continuation state.
- **Shared framework:** provide generic tool normalization, registry ownership, gateway execution, and tool-loop
  orchestration.

---

## Implemented Scope

The integration supports both the typed stateful executor path and the stateless HTTP proxy path:

- `ResponsesTool::Namespace` preserves the public Codex namespace declaration.
- `CodexNamespaceHandler` owns Codex-specific namespace flattening and restoration.
- `RequestPayload::to_upstream_request()` flattens namespace function members to vLLM-compatible function tools.
- Namespaced `tool_choice` values are rewritten to the same flat names sent upstream.
- `ToolRegistry` builds a request-scoped namespace map once and uses it for final payload and streaming event
  restoration.
- Stateful continuation stores effective tools, tool choice, instructions, and response/conversation linkage for later
  `previous_response_id` or `conversation_id` turns.
- WebSocket Responses execution uses the same typed executor path and restores namespace tool-call events before sending
  them to clients.
- Stateless HTTP proxying flattens raw namespace tools and a namespaced `tool_choice` before forwarding to vLLM. It
  retains the original `tools` declaration and restores it, plus any flat function calls, in JSON and SSE responses.

## HTTP Execution Paths

The HTTP Responses handler chooses one path per request:

| Request shape | Path | Namespace behavior |
|---------------|------|--------------------|
| `store=false`, no `previous_response_id` or `conversation_id`, and only client-owned `function` / `namespace` tools | Raw proxy | Flatten raw namespace members and rewrite namespaced `tool_choice` before forwarding. Restore original tools and calls in the upstream response. |
| `store=true`, a continuation ID, a conversation ID, or any gateway-owned tool | Typed executor | Rehydrate and persist state as needed, build `RequestPayload::to_upstream_request()`, and restore calls through `ToolRegistry`. |

The proxy is intentionally narrower than the executor: it does not create gateway response IDs, rehydrate history,
persist responses, or execute tools. It preserves the upstream response except for the namespace conversion needed to
make the request vLLM-compatible. When restoration changes a response body, it removes the upstream `Content-Length`.

WebSocket Responses requests always use the typed executor path because that protocol forces a stored streaming
response.

## Cassette Coverage

`record_codex_cli_tool_call_cassettes.sh gateway-proxy-http` records one-turn, `store=false` HTTP/SSE fixtures for
both ordinary function tools and namespace tools. The function fixture is the transparent proxy control case; the
namespace fixture verifies restoration of the original `tools` declaration and the `{ namespace, name }` function
call. Both are intentionally excluded from the two-turn continuation cassette matrix.

---

## Namespace Flattening

The model-visible namespace member format is:

```text
agentic_ns__{namespace}__{member}
```

For example, Codex can send:

```json
{
  "type": "namespace",
  "name": "mcp__agentic_fixture",
  "tools": [
    { "type": "function", "name": "add_numbers" }
  ]
}
```

The upstream request receives:

```json
{
  "type": "function",
  "name": "agentic_ns__mcp__agentic_fixture__add_numbers"
}
```

When the model calls that flat function, the gateway restores:

```json
{
  "type": "function_call",
  "namespace": "mcp__agentic_fixture",
  "name": "add_numbers"
}
```

---

## Collision Handling

The `agentic_ns__` prefix marks gateway-generated namespace member names. If a declared top-level function already uses
the generated name for a namespace member, or if two namespace members generate the same flat name, both execution paths
reject the request as invalid before it reaches upstream. Forwarding either shape would make a later model call ambiguous
and impossible to restore reliably to `{ namespace, name }`.

---

## Compatibility Rules

The gateway should not detect requests by user agent, route, or "is this Codex?" heuristics. Compatibility is driven by
Responses tool shapes and execution semantics, so it can be always on.

| Shape | Behavior |
|-------|----------|
| `function` | Client-owned by default. Preserve declaration and return matching calls to the client unless configured as gateway-owned. |
| `namespace` | Client-owned Codex grouping for function tools. Flatten members only at the selected upstream boundary, then restore returned calls. |
| `web_search_preview` | Gateway-owned when configured; normalized to the gateway web-search function tool. |
| `mcp`, `file_search`, `code_interpreter` | Preserve typed request shape; only forward once a gateway handler supports that tool kind. |
| Unknown tool | Preserve as raw-compatible unknown data where supported. Never execute by default. |

For response items:

| Response item | Behavior |
|---------------|----------|
| `function_call` | Preserve optional `namespace`; restore flat namespace calls before returning to Codex. |
| `web_search_call` | Gateway-owned result from the web-search executor. |
| Unknown output item | Preserve raw-compatible data where supported. Never execute by default. |

---

## Continuation

Codex-owned tool calls must survive response-store continuation.

Expected rehydration shape:

```text
prior context + assistant tool call + Codex tool output + new input
```

On a turn that returns client-owned tool calls, storage keeps the assistant call item. On the next turn, Codex submits
the matching tool output item, and `previous_response_id` rebuilds the full sequence while preserving effective tool
metadata from the previous response unless the client explicitly overrides it.

---

## Out Of Scope

- Gateway-side model aliasing.
- A gateway-side Codex runtime.
- Executing Codex namespace tools in the gateway. Codex still owns client-side tool execution.
