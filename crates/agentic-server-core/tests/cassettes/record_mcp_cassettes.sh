#!/usr/bin/env bash
# record_mcp_cassettes.sh
#
# Records gateway and OpenAI MCP cassettes through
# tests/cassettes/record_cassette.py.
#
# This records the gateway-facing MCP request:
#   - the request includes Codex's `read_mcp_resource` function bridge with
#     request-scoped MCP connection metadata
#   - the gateway normalizes that to the model-facing function tool
#   - the gateway executor runs the MCP tool loop and records the final response
#
# Gateway resource cases retain Codex's `read_mcp_resource` compatibility
# function because OpenAI's native MCP integration only exposes `tools/list`
# results; it does not expose MCP resources automatically.
#
# Records these gateway cassettes:
#   - happy path, non-streaming and streaming
#   - unhappy path: server_url fails the SSRF host allowlist ("unknown MCP
#     server", no connection attempted)
#   - unhappy path: server_url is loopback but nothing is listening
#     ("failed to connect")
#   - unhappy path: the MCP server is reachable but the resource URI does
#     not exist (resources/read fails)
#   - native MCP counter conversation, streaming only
#
# Also records the two-turn OpenAI counter reference conversation:
#   - list the discovered counter tools
#   - increment twice, then read the final value
#
# Prerequisites:
#   - agentic-api gateway running at GATEWAY_URL
#   - gateway upstream model has tool-call support
#   - gateway has an MCP server named MCP_SERVER_LABEL rooted at the agentic-api
#     repository, so it can serve the repo-relative MCP_RESOURCE_URI below.
#
# Usage:
#   bash crates/agentic-server-core/tests/cassettes/record_mcp_cassettes.sh
#   MCP_RECORD_SET=openai bash crates/agentic-server-core/tests/cassettes/record_mcp_cassettes.sh
#   MCP_RECORD_SET=all bash crates/agentic-server-core/tests/cassettes/record_mcp_cassettes.sh
#   MCP_SERVER_URL=http://localhost:8000/mcp GATEWAY_URL=http://localhost:9000 MODEL=Qwen/Qwen3-30B-A3B-FP8 bash crates/agentic-server-core/tests/cassettes/record_mcp_cassettes.sh

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/../../../.." && pwd)"
BASE_DIR="$SCRIPTS_DIR/mcp"
TOOLS_FILE="$BASE_DIR/tools.json"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:9000}"
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"
MODEL_SLUG="$(echo "$MODEL" | tr '/: ' '---')"
MCP_RECORD_SET="${MCP_RECORD_SET:-gateway}"
MCP_SERVER_LABEL="${MCP_SERVER_LABEL:-repo}"
MCP_SERVER_URL="${MCP_SERVER_URL:-http://localhost:8000/mcp}"
COUNTER_SERVER_LABEL="${COUNTER_SERVER_LABEL:-counter}"
GATEWAY_COUNTER_MCP_SERVER_URL="${GATEWAY_COUNTER_MCP_SERVER_URL:-http://localhost:8000/mcp}"
OPENAI_MCP_SERVER_URL="${OPENAI_MCP_SERVER_URL:-}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"
# Deliberately unroutable (RFC 5737 TEST-NET-1) — rejected by the gateway's
# SSRF host allowlist before any connection is attempted, so the tool call
# fails with "unknown MCP server", not a connection error.
MCP_UNREACHABLE_SERVER_URL="${MCP_UNREACHABLE_SERVER_URL:-http://192.0.2.1:8000/mcp}"
# Loopback (passes the allowlist) but nothing listens on this port, so the
# gateway actually attempts to connect and fails with a real connection
# error — "MCP server '{label}' failed to connect: {error}".
MCP_CONNECTION_REFUSED_SERVER_URL="${MCP_CONNECTION_REFUSED_SERVER_URL:-http://127.0.0.1:1/mcp}"
MCP_RESOURCE_URI="${MCP_RESOURCE_URI:-repo://crates/agentic-server-core/tests/cassettes/web_search/gpt_oss_web_search_nonstreaming.yaml}"
MCP_MISSING_RESOURCE_URI="${MCP_MISSING_RESOURCE_URI:-repo://crates/agentic-server-core/tests/cassettes/mcp/does-not-exist.yaml}"
NONSTREAMING_OUTPUT="$BASE_DIR/mcp-read-resource-${MODEL_SLUG}-nonstreaming.yaml"
STREAMING_OUTPUT="$BASE_DIR/mcp-read-resource-${MODEL_SLUG}-streaming.yaml"
UNREACHABLE_SERVER_OUTPUT="$BASE_DIR/mcp-read-resource-unreachable-server-${MODEL_SLUG}-streaming.yaml"
CONNECTION_REFUSED_OUTPUT="$BASE_DIR/mcp-read-resource-connection-refused-${MODEL_SLUG}-streaming.yaml"
MISSING_RESOURCE_OUTPUT="$BASE_DIR/mcp-read-resource-missing-resource-${MODEL_SLUG}-streaming.yaml"
COUNTER_STREAMING_OUTPUT="$BASE_DIR/mcp-counter-list-then-increment-${MODEL_SLUG}-streaming.yaml"
OPENAI_COUNTER_OUTPUT="$BASE_DIR/mcp-openai-reference-list-then-increment-gpt-4o-streaming.yaml"
REPO_PLACEHOLDER="<AGENTIC_API_REPO>"

green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n'  "$*"; }

sanitize_cassette() {
  local file="$1"
  perl -0pi -e "s|\\Q$REPO_ROOT\\E|$REPO_PLACEHOLDER|g" "$file"
}

# Writes the single reusable tools.json, pointed at whichever server_url the
# scenario needs (real server for happy-path/missing-resource, a bad URL for
# the unreachable/connection-refused unhappy paths).
write_tools_file() {
  local server_url="$1"
  cat > "$TOOLS_FILE" <<JSON
[
  {
    "type": "function",
    "name": "read_mcp_resource",
    "description": "Read a resource exposed by an MCP server.",
    "parameters": {
      "type": "object",
      "properties": {
        "server": {"type": "string"},
        "uri": {"type": "string"}
      },
      "required": ["server", "uri"],
      "additionalProperties": false
    },
    "strict": true,
    "metadata": {
      "server_label": "$MCP_SERVER_LABEL",
      "server_url": "$server_url"
    }
  }
]
JSON
}

write_native_mcp_tools_file() {
  local server_label="$1"
  local server_url="$2"
  cat > "$TOOLS_FILE" <<JSON
[
  {
    "type": "mcp",
    "server_label": "$server_label",
    "server_url": "$server_url",
    "require_approval": "never"
  }
]
JSON
}

mkdir -p "$BASE_DIR"

case "$MCP_RECORD_SET" in
  gateway|openai|all) ;;
  *)
    echo "ERROR: MCP_RECORD_SET must be gateway, openai, or all" >&2
    exit 1
    ;;
esac

if [[ "$MCP_RECORD_SET" == "gateway" || "$MCP_RECORD_SET" == "all" ]]; then

if [[ -z "$MCP_SERVER_URL" ]]; then
  echo "ERROR: MCP_SERVER_URL must point to an MCP server that serves $MCP_RESOURCE_URI" >&2
  exit 1
fi

write_tools_file "$MCP_SERVER_URL"

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

echo
bold "═══════════════════════════════════════════════════════════════"
bold "MCP cassette — read_mcp_resource, unreachable server (unhappy path)"
bold "Expected model behavior:"
bold "  1. Call read_mcp_resource with server=$MCP_SERVER_LABEL"
bold "  2. server_url fails the gateway's SSRF host allowlist, so no"
bold "     connection is even attempted"
bold "  3. The model receives an \"unknown MCP server\" error and reports it"
bold "═══════════════════════════════════════════════════════════════"
echo

write_tools_file "$MCP_UNREACHABLE_SERVER_URL"

UNREACHABLE_PROMPT="You have one MCP resource tool available: read_mcp_resource. The MCP server label is ${MCP_SERVER_LABEL}. Call read_mcp_resource exactly once with server ${MCP_SERVER_LABEL} and uri ${MCP_RESOURCE_URI}. If the call fails, report the error you received in one sentence."

printf '%s\n' "$UNREACHABLE_PROMPT" \
| python "$SCRIPTS_DIR/record_cassette.py" \
    --mode responses \
    --turns 1 \
    --stream \
    --model "$MODEL" \
    --gateway "$GATEWAY_URL" \
    --tools "$TOOLS_FILE" \
    --tool-choice "required" \
    --output "$UNREACHABLE_SERVER_OUTPUT"

sanitize_cassette "$UNREACHABLE_SERVER_OUTPUT"
green "✓ MCP cassette recorded -> $UNREACHABLE_SERVER_OUTPUT"

echo
bold "═══════════════════════════════════════════════════════════════"
bold "MCP cassette — read_mcp_resource, connection refused (unhappy path)"
bold "Expected model behavior:"
bold "  1. Call read_mcp_resource with server=$MCP_SERVER_LABEL"
bold "  2. server_url is loopback (passes the allowlist) but nothing is"
bold "     listening, so the gateway's connection attempt itself fails"
bold "  3. The model receives a \"failed to connect\" error and reports it"
bold "═══════════════════════════════════════════════════════════════"
echo

write_tools_file "$MCP_CONNECTION_REFUSED_SERVER_URL"

CONNECTION_REFUSED_PROMPT="You have one MCP resource tool available: read_mcp_resource. The MCP server label is ${MCP_SERVER_LABEL}. Call read_mcp_resource exactly once with server ${MCP_SERVER_LABEL} and uri ${MCP_RESOURCE_URI}. If the call fails, report the error you received in one sentence."

printf '%s\n' "$CONNECTION_REFUSED_PROMPT" \
| python "$SCRIPTS_DIR/record_cassette.py" \
    --mode responses \
    --turns 1 \
    --stream \
    --model "$MODEL" \
    --gateway "$GATEWAY_URL" \
    --tools "$TOOLS_FILE" \
    --tool-choice "required" \
    --output "$CONNECTION_REFUSED_OUTPUT"

sanitize_cassette "$CONNECTION_REFUSED_OUTPUT"
green "✓ MCP cassette recorded -> $CONNECTION_REFUSED_OUTPUT"

echo
bold "═══════════════════════════════════════════════════════════════"
bold "MCP cassette — read_mcp_resource, missing resource (unhappy path)"
bold "Expected model behavior:"
bold "  1. Call read_mcp_resource with server=$MCP_SERVER_LABEL"
bold "  2. The MCP server connects fine but resources/read fails —"
bold "     $MCP_MISSING_RESOURCE_URI does not exist"
bold "  3. The model receives a resource-not-found error and reports it"
bold "═══════════════════════════════════════════════════════════════"
echo

write_tools_file "$MCP_SERVER_URL"

MISSING_RESOURCE_PROMPT="You have one MCP resource tool available: read_mcp_resource. The MCP server label is ${MCP_SERVER_LABEL}. Call read_mcp_resource exactly once with server ${MCP_SERVER_LABEL} and uri ${MCP_MISSING_RESOURCE_URI}."

printf '%s\n' "$MISSING_RESOURCE_PROMPT" \
| python "$SCRIPTS_DIR/record_cassette.py" \
    --mode responses \
    --turns 1 \
    --stream \
    --model "$MODEL" \
    --gateway "$GATEWAY_URL" \
    --tools "$TOOLS_FILE" \
    --tool-choice "required" \
    --max-output-tokens 4096 \
    --output "$MISSING_RESOURCE_OUTPUT"

sanitize_cassette "$MISSING_RESOURCE_OUTPUT"
green "✓ MCP cassette recorded -> $MISSING_RESOURCE_OUTPUT"

echo
bold "═══════════════════════════════════════════════════════════════"
bold "MCP cassette — native counter tools, streaming"
bold "Expected model behavior:"
bold "  1. List the tools discovered from server=$COUNTER_SERVER_LABEL"
bold "  2. Increment twice, one call at a time"
bold "  3. Call get_value and return COUNTER_FINAL=<value>"
bold "═══════════════════════════════════════════════════════════════"
echo

write_native_mcp_tools_file "$COUNTER_SERVER_LABEL" "$GATEWAY_COUNTER_MCP_SERVER_URL"

printf '%s\n%s\n' \
  "List the MCP tools you have access to from the '${COUNTER_SERVER_LABEL}' server. For each one, give its name and a one-sentence description. Do not call any tool yet." \
  "Use the discovered MCP counter tools to prove that counter state is preserved across MCP connections. Call increment exactly twice, one call at a time. After the second increment, call get_value exactly once. Do not call any other tool. When get_value returns, reply exactly COUNTER_FINAL=<value>." \
| python "$SCRIPTS_DIR/record_cassette.py" \
    --mode responses \
    --turns 2 \
    --stream \
    --model "$MODEL" \
    --gateway "$GATEWAY_URL" \
    --tools "$TOOLS_FILE" \
    --tool-choice auto \
    --max-output-tokens 4096 \
    --output "$COUNTER_STREAMING_OUTPUT"

sanitize_cassette "$COUNTER_STREAMING_OUTPUT"
green "✓ MCP cassette recorded -> $COUNTER_STREAMING_OUTPUT"

# Leave the reusable tools file in its gateway resource configuration.
write_tools_file "$MCP_SERVER_URL"
fi

if [[ "$MCP_RECORD_SET" == "openai" || "$MCP_RECORD_SET" == "all" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY must be set for MCP_RECORD_SET=$MCP_RECORD_SET" >&2
    exit 1
  fi
  if [[ -z "$OPENAI_MCP_SERVER_URL" ]]; then
    echo "ERROR: OPENAI_MCP_SERVER_URL must be the public HTTPS URL of the counter MCP server" >&2
    exit 1
  fi

  write_native_mcp_tools_file "$COUNTER_SERVER_LABEL" "$OPENAI_MCP_SERVER_URL"

  echo
  bold "═══════════════════════════════════════════════════════════════"
  bold "OpenAI reference — native counter tools, streaming"
  bold "═══════════════════════════════════════════════════════════════"
  echo

  printf '%s\n%s\n' \
    "List the MCP tools you have access to from the '${COUNTER_SERVER_LABEL}' server. For each one, give its name and a one-sentence description. Do not call any tool yet." \
    "Use the discovered MCP counter tools to prove that counter state is preserved across MCP connections. Call counter__increment exactly twice, one call at a time. After the second increment, call counter__get_value exactly once. Do not call any other tool. When get_value returns, reply exactly COUNTER_FINAL=<value>." \
  | python "$SCRIPTS_DIR/record_cassette.py" \
      --mode responses \
      --turns 2 \
      --stream \
      --model "$OPENAI_MODEL" \
      --openai "https://api.openai.com" \
      --tools "$TOOLS_FILE" \
      --tool-choice auto \
      --max-output-tokens 4096 \
      --output "$OPENAI_COUNTER_OUTPUT"

  green "✓ OpenAI MCP reference recorded -> $OPENAI_COUNTER_OUTPUT"
fi
