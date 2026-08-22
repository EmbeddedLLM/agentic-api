#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

gateway_url="${AGENTIC_GATEWAY_URL:-http://127.0.0.1:3020}"
model="${AGENTIC_MODEL:-${V_MODEL:-agentic-api}}"
claude_config_dir="${AGENTIC_CLAUDE_CONFIG_DIR:-/tmp/claude-agentic-3020}"
claude_bin="${CLAUDE_BIN:-${AGENTIC_CLAUDE_BIN:-}}"

if [[ -z "$claude_bin" ]]; then
  claude_bin="$(command -v claude || true)"
fi
if [[ -z "$claude_bin" || ! -x "$claude_bin" ]]; then
  echo 'error: Claude Code not found; set CLAUDE_BIN=/path/to/claude' >&2
  exit 127
fi
if [[ "$model" == */* ]]; then
  echo "error: Claude Code requires a slash-free served model alias; start vLLM with --served-model-name and set AGENTIC_MODEL to that alias" >&2
  exit 2
fi

claude_args=(--bare --permission-mode manual --effort "${AGENTIC_CLAUDE_EFFORT:-medium}" --model "$model")
if [[ "${AGENTIC_YOLO:-0}" == "1" || "${AGENTIC_YOLO:-}" == "true" ]]; then
  claude_args+=(--dangerously-skip-permissions)
fi

exec env \
  -u CLAUDE_CODE_USE_VERTEX \
  -u ANTHROPIC_VERTEX_PROJECT_ID \
  -u ANTHROPIC_MODEL \
  CLAUDE_CONFIG_DIR="$claude_config_dir" \
  CLAUDE_CODE_EFFORT_LEVEL="${AGENTIC_CLAUDE_EFFORT:-medium}" \
  ANTHROPIC_BASE_URL="$gateway_url" \
  ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-demo}" \
  ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-${ANTHROPIC_API_KEY:-demo}}" \
  ANTHROPIC_MODEL="$model" \
  ANTHROPIC_SMALL_FAST_MODEL="$model" \
  ANTHROPIC_DEFAULT_OPUS_MODEL="$model" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="$model" \
  ANTHROPIC_DEFAULT_HAIKU_MODEL="$model" \
  "$claude_bin" \
  "${claude_args[@]}" \
  "$@"
