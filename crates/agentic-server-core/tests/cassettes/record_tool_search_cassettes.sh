#!/usr/bin/env bash
# Record the client-executed tool-search lifecycle against the OpenAI API.
#
# Usage from the repository root:
#   OPENAI_API_KEY=sk-... \
#     bash crates/agentic-server-core/tests/cassettes/record_tool_search_cassettes.sh

set -euo pipefail

CASSETTE_DIR="crates/agentic-server-core/tests/cassettes/tool_search"
PROMPT='You must call tool_search now to find the shipping ETA tool for order_42. Do not call get_shipping_eta yet and do not answer without calling tool_search.'

record_tool_search() {
  local stream_flag="$1"
  local suffix="$2"

  printf '%s\n' "$PROMPT" \
    | python crates/agentic-server-core/tests/cassettes/record_cassette.py \
        --mode responses \
        --turns 1 \
        "$stream_flag" \
        --openai https://api.openai.com \
        --model "${OPENAI_TOOL_SEARCH_MODEL:-gpt-5.6}" \
        --tools "$CASSETTE_DIR/tools.json" \
        --tool-choice required \
        --max-output-tokens 1024 \
        --output "$CASSETTE_DIR/tool-search-openai-reference-${OPENAI_TOOL_SEARCH_MODEL:-gpt-5.6}-${suffix}.yaml"
}

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY must be set" >&2
  exit 1
fi

record_tool_search --stream streaming
record_tool_search --no-stream nonstreaming
