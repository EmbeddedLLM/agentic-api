#!/usr/bin/env bash
# record_mcp_cassettes.sh
#
# Records MCP gateway tool cassettes through tests/cassettes/record_cassette.py.
#
# This records the gateway-facing MCP request:
#   - the request includes {"type":"mcp","name":"read_mcp_resource"}
#   - the gateway normalizes that to the model-facing function tool
#   - the gateway executor runs the MCP tool loop and records the final response
#
# Prerequisites:
#   - agentic-api gateway running at GATEWAY_URL
#   - gateway upstream model has tool-call support
#   - gateway has an MCP server named MCP_SERVER_LABEL rooted at the agentic-api
#     repository, so it can serve the repo-relative MCP_RESOURCE_URI below.
#
# Usage:
#   bash crates/agentic-server-core/tests/cassettes/record_mcp_cassettes.sh
#   MCP_SERVER_URL=http://localhost:8000/mcp GATEWAY_URL=http://localhost:9000 MODEL=Qwen/Qwen3-30B-A3B-FP8 bash crates/agentic-server-core/tests/cassettes/record_mcp_cassettes.sh

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/../../../.." && pwd)"
BASE_DIR="$SCRIPTS_DIR/mcp"
TOOLS_FILE="$BASE_DIR/tools.json"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:9000}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
MODEL_SLUG="$(echo "$MODEL" | tr '/: ' '---')"
MCP_SERVER_LABEL="${MCP_SERVER_LABEL:-repo}"
MCP_SERVER_URL="${MCP_SERVER_URL:-http://localhost:8000/mcp}"
MCP_RESOURCE_URI="${MCP_RESOURCE_URI:-repo://crates/agentic-server-core/tests/cassettes/web_search/gpt_oss_web_search_nonstreaming.yaml}"
NONSTREAMING_OUTPUT="$BASE_DIR/mcp-read-resource-${MODEL_SLUG}-nonstreaming.yaml"
STREAMING_OUTPUT="$BASE_DIR/mcp-read-resource-${MODEL_SLUG}-streaming.yaml"
REPO_PLACEHOLDER="<AGENTIC_API_REPO>"

green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n'  "$*"; }

sanitize_cassette() {
  local file="$1"
  perl -0pi -e "s|\\Q$REPO_ROOT\\E|$REPO_PLACEHOLDER|g" "$file"
}

mkdir -p "$BASE_DIR"

if [[ -z "$MCP_SERVER_URL" ]]; then
  echo "ERROR: MCP_SERVER_URL must point to an MCP server that serves $MCP_RESOURCE_URI" >&2
  exit 1
fi

cat > "$TOOLS_FILE" <<JSON
[
  {
    "type": "mcp",
    "name": "read_mcp_resource",
    "server_label": "$MCP_SERVER_LABEL",
    "server_url": "$MCP_SERVER_URL"
  }
]
JSON

PROMPT="You have one MCP resource tool available: read_mcp_resource. The MCP server label is ${MCP_SERVER_LABEL}. Call read_mcp_resource exactly once with server ${MCP_SERVER_LABEL} and uri ${MCP_RESOURCE_URI}. Then summarize the gpt_oss_web_search_nonstreaming.yaml cassette in 2-3 sentences and mention that read_mcp_resource was used."

bold "Gateway: $GATEWAY_URL"
bold "Model:   $MODEL"
bold "Tools:   $TOOLS_FILE"
bold "Server:  $MCP_SERVER_LABEL"
bold "URL:     $MCP_SERVER_URL"
bold "URI:     $MCP_RESOURCE_URI"
echo

bold "═══════════════════════════════════════════════════════════════"
bold "MCP cassette — read_mcp_resource, non-streaming"
bold "Expected model behavior:"
bold "  1. Call read_mcp_resource exactly once with server=$MCP_SERVER_LABEL"
bold "  2. Use uri=$MCP_RESOURCE_URI"
bold "  3. Summarize the gateway-executed MCP resource output"
bold "═══════════════════════════════════════════════════════════════"
echo

printf '%s\n' "$PROMPT" \
| python "$SCRIPTS_DIR/record_cassette.py" \
    --mode responses \
    --turns 1 \
    --no-stream \
    --model "$MODEL" \
    --gateway "$GATEWAY_URL" \
    --tools "$TOOLS_FILE" \
    --tool-choice "required" \
    --output "$NONSTREAMING_OUTPUT"

sanitize_cassette "$NONSTREAMING_OUTPUT"
green "✓ MCP cassette recorded -> $NONSTREAMING_OUTPUT"

echo
bold "═══════════════════════════════════════════════════════════════"
bold "MCP cassette — read_mcp_resource, streaming"
bold "═══════════════════════════════════════════════════════════════"
echo

printf '%s\n' "$PROMPT" \
| python "$SCRIPTS_DIR/record_cassette.py" \
    --mode responses \
    --turns 1 \
    --stream \
    --model "$MODEL" \
    --gateway "$GATEWAY_URL" \
    --tools "$TOOLS_FILE" \
    --tool-choice "required" \
    --output "$STREAMING_OUTPUT"

sanitize_cassette "$STREAMING_OUTPUT"
green "✓ MCP cassette recorded -> $STREAMING_OUTPUT"
