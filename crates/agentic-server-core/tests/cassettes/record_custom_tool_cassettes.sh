#!/usr/bin/env bash
# Records the same client-executed custom-tool flow against vLLM, the gateway,
# and OpenAI.
#
# Each provider records a two-turn streaming and non-streaming freeform cassette:
#   1. the model emits raw text in a custom_tool_call
#   2. the recorder submits custom_tool_call_output and captures the final reply
#
# Usage from the repository root:
#   OPENAI_API_KEY=sk-... \
#     bash crates/agentic-server-core/tests/cassettes/record_custom_tool_cassettes.sh
#   CUSTOM_TOOL_RECORD_SET=gateway \
#     bash crates/agentic-server-core/tests/cassettes/record_custom_tool_cassettes.sh
#   CUSTOM_TOOL_RECORD_SET=vllm VLLM_URL=http://localhost:5050 \
#     bash crates/agentic-server-core/tests/cassettes/record_custom_tool_cassettes.sh
#   CUSTOM_TOOL_RECORD_SET=openai OPENAI_API_KEY=sk-... \
#     bash crates/agentic-server-core/tests/cassettes/record_custom_tool_cassettes.sh

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPTS_DIR/custom_tool"
FREEFORM_TOOLS_FILE="$BASE_DIR/custom_tool.json"
TOOL_OUTPUTS_FILE="$BASE_DIR/tool_outputs.json"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:9000}"
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"
MODEL_SLUG="$(echo "$MODEL" | tr '/: ' '---')"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.6}"
OPENAI_MODEL_SLUG="$(echo "$OPENAI_MODEL" | tr '/: ' '---')"
CUSTOM_TOOL_RECORD_SET="${CUSTOM_TOOL_RECORD_SET:-all}"
FIRST_PROMPT='You must call the agentic_raw_echo custom tool exactly once with exactly CUSTOM_CASSETTE_OK as its raw text input.'
SECOND_PROMPT='Use the custom tool output provided above. Do not call any tool again. Reply with exactly CUSTOM_CASSETTE_OUTPUT_OK.'

green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n'  "$*"; }

record_scenario() {
  local endpoint_flag="$1"
  local endpoint="$2"
  local model="$3"
  local output="$4"
  local stream_flag="$5"
  local temporary_output

  temporary_output="$(mktemp "$BASE_DIR/.custom-tool-cassette.XXXXXX")"

  if ! printf '%s\n%s\n' "$FIRST_PROMPT" "$SECOND_PROMPT" \
    | python "$SCRIPTS_DIR/record_cassette.py" \
        --mode responses \
        --turns 2 \
        "$stream_flag" \
        --model "$model" \
        "$endpoint_flag" "$endpoint" \
        --tools "$FREEFORM_TOOLS_FILE" \
        --tool-outputs "$TOOL_OUTPUTS_FILE" \
        --max-output-tokens 2048 \
        --output "$temporary_output"
  then
    rm -f -- "$temporary_output"
    return 1
  fi

  mv -- "$temporary_output" "$output"
  green "✓ custom-tool cassette recorded -> $output"
}

record_provider_suite() {
  local provider="$1"
  local endpoint_flag="$2"
  local endpoint="$3"
  local model="$4"
  local output_prefix="$5"

  bold "$provider custom-tool cassettes"
  bold "Endpoint: $endpoint"
  bold "Model:    $model"

  bold "$provider streaming custom-tool flow"
  record_scenario \
    "$endpoint_flag" "$endpoint" "$model" \
    "$BASE_DIR/${output_prefix}-streaming.yaml" \
    --stream

  bold "$provider non-streaming custom-tool flow"
  record_scenario \
    "$endpoint_flag" "$endpoint" "$model" \
    "$BASE_DIR/${output_prefix}-nonstreaming.yaml" \
    --no-stream

}

case "$CUSTOM_TOOL_RECORD_SET" in
  gateway|vllm|openai|all) ;;
  *)
    echo "ERROR: CUSTOM_TOOL_RECORD_SET must be gateway, vllm, openai, or all" >&2
    exit 1
    ;;
esac

for required_file in "$FREEFORM_TOOLS_FILE" "$TOOL_OUTPUTS_FILE"; do
  if [[ ! -f "$required_file" ]]; then
    echo "ERROR: required custom-tool fixture does not exist: $required_file" >&2
    exit 1
  fi
done

if [[ "$CUSTOM_TOOL_RECORD_SET" == "openai" || "$CUSTOM_TOOL_RECORD_SET" == "all" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY must be set for CUSTOM_TOOL_RECORD_SET=$CUSTOM_TOOL_RECORD_SET" >&2
    exit 1
  fi
fi

if [[ "$CUSTOM_TOOL_RECORD_SET" == "openai" || "$CUSTOM_TOOL_RECORD_SET" == "all" ]]; then
  record_provider_suite \
    OpenAI \
    --openai https://api.openai.com \
    "$OPENAI_MODEL" \
    "custom-tool-openai-reference-${OPENAI_MODEL_SLUG}"
fi

if [[ "$CUSTOM_TOOL_RECORD_SET" == "gateway" || "$CUSTOM_TOOL_RECORD_SET" == "all" ]]; then
  record_provider_suite \
    Gateway \
    --gateway "$GATEWAY_URL" \
    "$MODEL" \
    "custom-tool-gateway-${MODEL_SLUG}"
fi
