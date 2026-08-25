#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export V_API_BASE="${V_API_BASE:-http://127.0.0.1:8000/v1}"
export V_API_KEY="${V_API_KEY:-demo}"
export AGENTIC_MODEL="${AGENTIC_MODEL:-${V_MODEL:-agentic-api}}"
export V_MODEL="$AGENTIC_MODEL"
export MESSAGES_MODEL_OVERRIDE="${MESSAGES_MODEL_OVERRIDE:-$AGENTIC_MODEL}"
export GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
export GATEWAY_PORT="${GATEWAY_PORT:-3020}"
export DATABASE_URL="${DATABASE_URL:-sqlite:///tmp/agentic_api_3020.db}"
export SKIP_LLM_READY_CHECK="${SKIP_LLM_READY_CHECK:-true}"
export AGENTIC_DEBUG="${AGENTIC_DEBUG:-true}"

if [[ "$AGENTIC_DEBUG" == "true" || "$AGENTIC_DEBUG" == "1" ]]; then
  export RUST_LOG="${RUST_LOG:-agentic_server=debug,agentic_core=debug,agentic_core::proxy=trace}"
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-$V_API_KEY}"

server_args=(
  --gateway-host "$GATEWAY_HOST"
  --gateway-port "$GATEWAY_PORT"
  --llm-api-base "$V_API_BASE"
)
if [[ "$SKIP_LLM_READY_CHECK" == "true" || "$SKIP_LLM_READY_CHECK" == "1" ]]; then
  server_args+=(--skip-llm-ready-check)
fi

echo "Starting agentic-api gateway on http://${GATEWAY_HOST}:${GATEWAY_PORT}"
echo "Upstream base: ${V_API_BASE}"
echo "Model: ${AGENTIC_MODEL}"

exec cargo run -p agentic-server --bin agentic-server -- "${server_args[@]}"
