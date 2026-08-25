#!/usr/bin/env bash
set -euo pipefail

CLAUDE_BIN="${CLAUDE_BIN:-claude}"
AGENTIC_SERVER_BIN="${AGENTIC_SERVER_BIN:-target/debug/agentic-server}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CASSETTE="${CASSETTE:-crates/agentic-server-core/tests/cassettes/messages/messages-web-search-Qwen-Qwen3-30B-A3B-FP8-streaming.yaml}"

choose_port() {
  "$PYTHON_BIN" -c 'import socket; sock = socket.socket(); sock.bind(("127.0.0.1", 0)); print(sock.getsockname()[1]); sock.close()'
}

REPLAY_PORT="${REPLAY_PORT:-$(choose_port)}"
GATEWAY_PORT="${GATEWAY_PORT:-$(choose_port)}"

if ! command -v "$CLAUDE_BIN" >/dev/null 2>&1; then
  echo "error: Claude Code is not installed: ${CLAUDE_BIN}" >&2
  exit 2
fi
if [[ ! -x "$AGENTIC_SERVER_BIN" ]]; then
  echo "error: agentic-server is not executable: ${AGENTIC_SERVER_BIN}; run cargo build -p agentic-server" >&2
  exit 2
fi
if [[ ! -f "$CASSETTE" ]]; then
  echo "error: Messages cassette not found: ${CASSETTE}" >&2
  exit 2
fi

temp_dir="$(mktemp -d)"
capture_path="${temp_dir}/capture.jsonl"
replay_log="${temp_dir}/replay.log"
gateway_log="${temp_dir}/gateway.log"
claude_output="${temp_dir}/claude.json"
claude_debug="${temp_dir}/claude-debug.log"
claude_config="${temp_dir}/claude-config"
replay_pid=""
gateway_pid=""
mkdir -p "$claude_config"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$gateway_pid" ]]; then
    kill "$gateway_pid" >/dev/null 2>&1 || true
    wait "$gateway_pid" >/dev/null 2>&1 || true
  fi
  if [[ -n "$replay_pid" ]]; then
    kill "$replay_pid" >/dev/null 2>&1 || true
    wait "$replay_pid" >/dev/null 2>&1 || true
  fi
  if [[ "$status" -ne 0 ]]; then
    echo "--- replay server log ---" >&2
    sed -n '1,240p' "$replay_log" >&2 || true
    echo "--- agentic-server log ---" >&2
    sed -n '1,240p' "$gateway_log" >&2 || true
    echo "--- Claude Code output ---" >&2
    sed -n '1,240p' "$claude_output" >&2 || true
    echo "--- Claude Code debug log ---" >&2
    sed -n '1,240p' "$claude_debug" >&2 || true
    echo "--- replay capture ---" >&2
    sed -n '1,240p' "$capture_path" >&2 || true
  fi
  rm -r "$temp_dir"
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_until_ready() {
  local label="$1"
  local url="$2"
  for attempt in $(seq 1 60); do
    if curl --connect-timeout 1 --max-time 2 --fail --silent "$url" >/dev/null; then
      return 0
    fi
    echo "${label} not ready (attempt ${attempt}/60)"
    sleep 1
  done
  echo "error: ${label} did not become ready" >&2
  return 1
}

"$PYTHON_BIN" scripts/claude_code_replay_server.py serve \
  --cassette "$CASSETTE" \
  --capture "$capture_path" \
  --port "$REPLAY_PORT" \
  >"$replay_log" 2>&1 &
replay_pid=$!
wait_until_ready "replay server" "http://127.0.0.1:${REPLAY_PORT}/health"

env \
  LLM_API_BASE="http://127.0.0.1:${REPLAY_PORT}" \
  GATEWAY_HOST=127.0.0.1 \
  GATEWAY_PORT="$GATEWAY_PORT" \
  SKIP_LLM_READY_CHECK=true \
  DATABASE_URL="sqlite://${temp_dir}/agentic.db" \
  MESSAGES_GATEWAY_TOOL_ALIASES=WebSearch=web_search \
  YOU_API_KEY=ci-placeholder \
  YOU_API_BASE_URL="http://127.0.0.1:${REPLAY_PORT}" \
  "$AGENTIC_SERVER_BIN" \
  >"$gateway_log" 2>&1 &
gateway_pid=$!
wait_until_ready "agentic-server" "http://127.0.0.1:${GATEWAY_PORT}/ready"

prompt="Use WebSearch to find the latest stable Rust release, then answer with its version only."
env \
  -u CLAUDE_CODE_USE_VERTEX \
  -u ANTHROPIC_VERTEX_PROJECT_ID \
  -u CLAUDE_CODE_USE_BEDROCK \
  -u CLAUDE_CODE_USE_FOUNDRY \
  ANTHROPIC_BASE_URL="http://127.0.0.1:${GATEWAY_PORT}" \
  ANTHROPIC_API_KEY=ci-placeholder \
  ANTHROPIC_MODEL=qwen3 \
  ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3 \
  ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3 \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3 \
  CLAUDE_CONFIG_DIR="$claude_config" \
  DISABLE_AUTOUPDATER=1 \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
  "$CLAUDE_BIN" \
  --safe-mode \
  --print \
  "$prompt" \
  --output-format json \
  --debug-file "$claude_debug" \
  --no-session-persistence \
  --permission-mode dontAsk \
  --model qwen3 \
  --tools WebSearch \
  --allowedTools WebSearch \
  >"$claude_output"

"$PYTHON_BIN" - "$claude_output" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
assert result.get("is_error") is False, result
assert "1.89.0" in result.get("result", ""), result
print(f"Claude Code result: {result['result']}")
PY

"$PYTHON_BIN" scripts/claude_code_replay_server.py assert-capture --capture "$capture_path"
