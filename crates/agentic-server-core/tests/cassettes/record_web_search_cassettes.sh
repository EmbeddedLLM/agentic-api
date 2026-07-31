#!/usr/bin/env bash
# Records the same web-search scenario against the gateway and OpenAI.
#
# Each provider records:
#   1. one streaming response
#   2. one non-streaming response
#
# The default records both providers so the gateway behavior can be compared
# with the OpenAI Responses API ground truth. OPENAI_API_KEY must be set for
# the default run.
#
# Usage from the repository root:
#   bash crates/agentic-server-core/tests/cassettes/record_web_search_cassettes.sh
#   WEB_SEARCH_RECORD_SET=gateway \
#     bash crates/agentic-server-core/tests/cassettes/record_web_search_cassettes.sh
#   WEB_SEARCH_RECORD_SET=openai OPENAI_API_KEY=sk-... \
#     bash crates/agentic-server-core/tests/cassettes/record_web_search_cassettes.sh

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPTS_DIR/web_search"
TOOLS_FILE="$BASE_DIR/tools.json"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:9000}"
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"
MODEL_SLUG="$(echo "$MODEL" | tr '/: ' '---')"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.6}"
OPENAI_MODEL_SLUG="$(echo "$OPENAI_MODEL" | tr '/: ' '---')"
WEB_SEARCH_RECORD_SET="${WEB_SEARCH_RECORD_SET:-all}"
PROMPT='Use web search to search for the exact query "potato", then summarize the result in one sentence.'

green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n'  "$*"; }

record_single_turn() {
  local endpoint_flag="$1"
  local endpoint="$2"
  local model="$3"
  local output="$4"
  local stream_flag="$5"
  local temporary_output

  temporary_output="$(mktemp "$BASE_DIR/.web-search-cassette.XXXXXX")"

  if ! printf '%s\n' "$PROMPT" \
    | python "$SCRIPTS_DIR/record_cassette.py" \
        --mode responses \
        --turns 1 \
        "$stream_flag" \
        --model "$model" \
        "$endpoint_flag" "$endpoint" \
        --tools "$TOOLS_FILE" \
        --tool-choice auto \
        --max-output-tokens 1024 \
        --output "$temporary_output"
  then
    rm -f -- "$temporary_output"
    return 1
  fi

  mv -- "$temporary_output" "$output"
  green "✓ web-search cassette recorded -> $output"
}

record_provider_suite() {
  local provider="$1"
  local endpoint_flag="$2"
  local endpoint="$3"
  local model="$4"
  local streaming_output="$5"
  local nonstreaming_output="$6"

  bold "$provider web-search cassettes"
  bold "Endpoint: $endpoint"
  bold "Model:    $model"

  bold "$provider streaming web-search response"
  record_single_turn \
    "$endpoint_flag" "$endpoint" "$model" "$streaming_output" --stream

  bold "$provider non-streaming web-search response"
  record_single_turn \
    "$endpoint_flag" "$endpoint" "$model" "$nonstreaming_output" --no-stream
}

case "$WEB_SEARCH_RECORD_SET" in
  gateway|openai|all) ;;
  *)
    echo "ERROR: WEB_SEARCH_RECORD_SET must be gateway, openai, or all" >&2
    exit 1
    ;;
esac

if [[ ! -f "$TOOLS_FILE" ]]; then
  echo "ERROR: web-search tools file does not exist: $TOOLS_FILE" >&2
  exit 1
fi

# Validate OpenAI requirements before recording the gateway suite so a default
# `all` run cannot leave only half of the comparison fixtures.
if [[ "$WEB_SEARCH_RECORD_SET" == "openai" || "$WEB_SEARCH_RECORD_SET" == "all" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY must be set for WEB_SEARCH_RECORD_SET=$WEB_SEARCH_RECORD_SET" >&2
    exit 1
  fi
fi

mkdir -p "$BASE_DIR"

if [[ "$WEB_SEARCH_RECORD_SET" == "openai" || "$WEB_SEARCH_RECORD_SET" == "all" ]]; then
  record_provider_suite \
    OpenAI \
    --openai https://api.openai.com \
    "$OPENAI_MODEL" \
    "$BASE_DIR/web-search-openai-reference-${OPENAI_MODEL_SLUG}-streaming.yaml" \
    "$BASE_DIR/web-search-openai-reference-${OPENAI_MODEL_SLUG}-nonstreaming.yaml"
fi

if [[ "$WEB_SEARCH_RECORD_SET" == "gateway" || "$WEB_SEARCH_RECORD_SET" == "all" ]]; then
  record_provider_suite \
    Gateway \
    --gateway "$GATEWAY_URL" \
    "$MODEL" \
    "$BASE_DIR/web-search-gateway-${MODEL_SLUG}-streaming.yaml" \
    "$BASE_DIR/web-search-gateway-${MODEL_SLUG}-nonstreaming.yaml"
fi
