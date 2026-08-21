# Testing Claude Code and Codex through the `agentic` CLI

This guide is the shareable checklist for exercising a coding harness (Claude Code or Codex) against Agentic API,
both with a local gateway started by the CLI and against a gateway already running on Kubernetes. It was written
while verifying [issue #190](https://github.com/vllm-project/agentic-api/issues/190) / [PR
#197](https://github.com/vllm-project/agentic-api/pull/197) and records the exact commands and expected output.

## Prerequisites

- A running OpenAI-compatible upstream. The examples use vLLM serving Qwen on `http://127.0.0.1:8000`:

  ```console
  curl -s http://127.0.0.1:8000/v1/models | python3 -c 'import sys,json; print([m["id"] for m in json.load(sys.stdin)["data"]])'
  ```

- The harness binary on `PATH`: `claude --version` or `codex --version`. Override discovery with `AGENTIC_CLAUDE_BIN`
  or `AGENTIC_CODEX_BIN`.
- Both Agentic API binaries built from this repository:

  ```console
  cargo build -p agentic-server --bins
  ```

## CLI behavior worth knowing

| Behavior | Detail |
|---|---|
| `--upstream` must be a full URL | `http://` or `https://` with a host. A typo such as `http//127.0.0.1:8000` is rejected at parse time with `invalid upstream URL`. |
| `--model` is optional with `--upstream` | When omitted, the CLI calls `GET {upstream}/v1/models` and uses the first model listed. Pass `--model` to pick another when the upstream serves several. |
| Claude effort is pinned to `medium` | Claude Code defaults to `high`, which Qwen's vLLM chat template rejects (`ValueError`). The CLI always passes `--effort medium` and sets `CLAUDE_CODE_EFFORT_LEVEL=medium` (the env var wins inside Claude Code). Override both with `AGENTIC_CLAUDE_EFFORT=low|medium|xhigh`. |
| `--yolo` | Adds `--dangerously-skip-permissions` (Claude) or `--dangerously-bypass-approvals-and-sandbox` (Codex). Use only in an externally isolated environment. |
| `--skip-llm-ready-check` | Skips the upstream `/health` probe. Avoid it while testing: the probe is what surfaces an unreachable upstream before the harness starts. |
| Arguments after `--` | Forwarded verbatim to the harness (`-p`, `--resume`, `exec`, ...). |

## 1. Validate the configuration

```console
./target/debug/agentic validate --upstream http://127.0.0.1:8000 --harness claude
```

Expected: `Agentic API configuration looks valid.` This checks the gateway port is free, the database URL is usable,
the harness binary resolves, and the upstream URL is well formed.

## 2. Non-interactive smoke test

Claude Code:

```console
./target/debug/agentic run claude --upstream http://127.0.0.1:8000 -- -p "Reply with exactly one word: pong"
```

Codex:

```console
./target/debug/agentic run codex --upstream http://127.0.0.1:8000 -- exec "Reply with exactly one word: pong"
```

Expected lifecycle output, then the harness answer and a clean exit:

```text
Starting Claude via http://127.0.0.1:3000
... agentic_server::server: LLM ready: http://127.0.0.1:8000
... agentic_server::server: gateway listening on 127.0.0.1:3000
Claude Code gateway: http://127.0.0.1:3000 (model: Qwen/Qwen3.8-27B-FP8)
pong
```

Claude Code prints a warning that the model name is not one it recognizes and assumes a 200k context window. That
is expected for any non-Anthropic model name and does not affect the request.

## 3. Tool-call round trip

Tool calls are where the parallel-tool-call handling from #190 is exercised, so run at least one. Start a normal
session and approve the permission prompt when the harness asks to run the command:

```console
./target/debug/agentic run claude --upstream http://127.0.0.1:8000
> Run 'ls crates' with the Bash tool and list the directory names.
```

Expected: the harness runs the command and answers with `agentic-praxis`, `agentic-server`, `agentic-server-core`.
The gateway log must not contain `invalid tool config`.

Permission prompts are the default. Only for an unattended run in an externally isolated environment (CI, a
throwaway container) add `--yolo`, which forwards the harness's native bypass flag:

```console
./target/debug/agentic run claude --upstream http://127.0.0.1:8000 --yolo \
  -- -p "Run 'ls crates' with the Bash tool and list the directory names."
```

## 4. Interactive session

```console
./target/debug/agentic run claude --upstream http://127.0.0.1:8000
```

Inside the session, useful checks are a plain question (no tools), a file read, a multi-step edit, and `/model`
(should show the discovered or passed model). `Ctrl-C` stops the harness and the gateway together; confirm nothing
is left behind with `pgrep -fa agentic-server`.

## 5. Reproduce and verify #190 directly

The bug was a gateway-side rejection of `parallel_tool_calls: true` whenever a built-in tool was declared. Codex
always sends `true`, so every Codex session with a built-in tool failed:

```console
curl -s -w '\nHTTP %{http_code}\n' -H 'content-type: application/json' http://127.0.0.1:3000/v1/responses -d '{
  "model": "Qwen/Qwen3.8-27B-FP8",
  "input": "Reply with the single word: pong",
  "max_output_tokens": 64,
  "parallel_tool_calls": true,
  "tools": [{"type": "web_search_preview"}]
}'
```

| Gateway build | Result |
|---|---|
| Before PR #197 | `HTTP 400` — `invalid tool config: parallel_tool_calls must be false when using built-in tools` |
| PR #197 or later | `HTTP 200`, `"status": "completed"`; the gateway forwards `parallel_tool_calls: false` upstream and serializes tool calls |

Also run the mixed shape (a `function` tool plus a built-in such as `code_interpreter`) with `parallel_tool_calls:
true`; it must return `HTTP 200` as well. Unit coverage lives in
`crates/agentic-server-core/src/types/request_response.rs` (`to_upstream_request_*parallel_tool_calls*`).

## 6. Against a Kubernetes deployment

The CLI always starts its own local gateway, so to test a cluster-hosted gateway point the harness at it directly
with the same environment the CLI would set. The kind-based development cluster below is the one from
[Deploy agentic-api on Kubernetes](../deploying/kubernetes.md).

### Roll out a new image

```console
docker build -t agentic-api:kind .
kind load docker-image agentic-api:kind --name agentic-api
kubectl rollout restart deploy/agentic-api
kubectl rollout status deploy/agentic-api --timeout=180s
```

### Port-forward and run the same checks

```console
kubectl port-forward svc/agentic-api 9000:9000 &

# #190 repro (expect HTTP 200)
curl -s -w '\nHTTP %{http_code}\n' -H 'content-type: application/json' http://127.0.0.1:9000/v1/responses -d '{
  "model": "Qwen/Qwen3.8-27B-FP8", "input": "Reply with the single word: pong", "max_output_tokens": 64,
  "parallel_tool_calls": true, "tools": [{"type": "web_search_preview"}]
}'

# Claude Code through the cluster gateway
ANTHROPIC_BASE_URL=http://127.0.0.1:9000 \
ANTHROPIC_API_KEY=agentic-api-local \
ANTHROPIC_MODEL=Qwen/Qwen3.8-27B-FP8 \
ANTHROPIC_SMALL_FAST_MODEL=Qwen/Qwen3.8-27B-FP8 \
CLAUDE_CODE_EFFORT_LEVEL=medium \
claude --effort medium -p "Reply with exactly one word: pong"
```

Codex needs a `CODEX_HOME` with the same provider configuration the CLI generates. Keep it outside `/tmp` (Codex
refuses to create its helper binaries there) and close stdin for `exec`, otherwise Codex waits for additional prompt
input when it is not attached to a terminal:

```console
export CODEX_HOME=$HOME/.cache/agentic-codex-k8s
mkdir -p "$CODEX_HOME"
cat > "$CODEX_HOME/config.toml" <<EOF
model = "Qwen/Qwen3.8-27B-FP8"
model_provider = "agentic-api"
model_catalog_json = "$CODEX_HOME/model_catalog.json"

[model_providers.agentic-api]
name = "Agentic API"
base_url = "http://127.0.0.1:9000/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = true
EOF
# Copy model_catalog.json from a CLI session (printed path) or from prepare_codex_home in
# crates/agentic-server/src/agentic_harness.rs, replacing the slug with your model id.
codex exec --skip-git-repo-check "Reply with exactly one word: pong" </dev/null
```

Codex uses the gateway's WebSocket transport (`supports_websockets = true`), so this also verifies `/v1/responses`
over WebSocket through the port-forward. On Linux hosts where Codex's sandbox (`codex-linux-sandbox`/bwrap) cannot
start, shell tool calls fail inside Codex itself; that is unrelated to the gateway and `--dangerously-bypass-approvals-and-sandbox`
(or the CLI's `--yolo`) confirms the gateway side.

Replace `agentic-api-local` with a real key when the deployment enforces inbound authentication. Finish by
confirming the gateway logged no errors during the run:

```console
kubectl logs deploy/agentic-api --since=10m | grep -ci error   # expect 0
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `invalid upstream URL` at startup | Malformed `--upstream` (missing `://`, no scheme, no host) | Pass a full `http://host:port` URL. |
| `HTTP 502` on every request | Gateway cannot reach the upstream, typically a wrong URL combined with `--skip-llm-ready-check` | Drop `--skip-llm-ready-check` so the readiness probe fails fast, then fix the URL. |
| `There's an issue with the selected model` | Model name does not match what the upstream serves | Omit `--model` to auto-discover, or copy the id from `/v1/models` exactly. |
| Template `ValueError` mentioning effort | Claude Code sent `high` | Do not override `AGENTIC_CLAUDE_EFFORT` with `high`; valid values for Qwen are `low`, `medium`, `xhigh`. |
| `HTTP 503` from a cluster gateway | `/ready` failing because the gateway cannot reach its upstream or database | `kubectl logs deploy/agentic-api` and look for `gateway dependencies not ready`. |
| `readiness.ready=false` warnings every minute or two while the upstream is healthy | Gateway build predates the readiness-client pooling fix ([#199](https://github.com/vllm-project/agentic-api/pull/199)): a pooled keep-alive connection that the upstream closed fails with `hyper::Error(IncompleteMessage)` | Rebuild from a tree that includes #199 and redeploy; add `agentic_server::handler::http::models=debug` to `RUST_LOG` to see the probe error. |
| Pods in `CrashLoopBackOff` with `failed to create temporary configuration file: Read-only file system` | Gateway build predates the read-only-home fix ([#199](https://github.com/vllm-project/agentic-api/pull/199)), and the base mounts a read-only root filesystem | Rebuild from a tree that includes #199 and redeploy; that base also mounts an `emptyDir` at `/var/lib/agentic-api`. Until then, mount a writable volume at `/var/lib/agentic-api` in your overlay. |
| `kubectl apply -k` fails with `cycle detected` | Overlay directory placed inside `deploy/kubernetes` | Move the overlay to a sibling directory such as `deploy/overlays/<env>` and reference `../../kubernetes` (a working kind example ships with #199). |
| Codex `exec` prints `Reading additional input from stdin...` and hangs | stdin is not a terminal | Append `</dev/null`. |
| Codex warns it could not create PATH aliases | `CODEX_HOME` under `/tmp` | Use a home under `$HOME`. |
| A long stream stops without a terminal event during a rollout | Bounded drain: 5 s `preStop` plus up to 8 s of in-flight draining, then the pod exits | Expected for responses longer than the drain window; clients should reconnect and continue with `previous_response_id`. |
| `parallel_tool_calls must be false when using built-in tools` | Gateway predates PR #197 | Rebuild and redeploy the image. |
