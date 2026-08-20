#!/usr/bin/env bash
# Records client tool-search characterization against OpenAI, direct vLLM, or
# the gateway blocking, HTTP/SSE, or WebSocket profiles.
#
# Usage from the repository root:
#   TOOL_SEARCH_RECORD_SET=openai-reference OPENAI_API_KEY=sk-... \
#     bash crates/agentic-server-core/tests/cassettes/record_tool_search_cassettes.sh
#   TOOL_SEARCH_RECORD_SET=direct-vllm VLLM_URL=http://localhost:8000 \
#     bash crates/agentic-server-core/tests/cassettes/record_tool_search_cassettes.sh
#   TOOL_SEARCH_RECORD_SET=gateway-nonstreaming GATEWAY_URL=http://localhost:9000 \
#     bash crates/agentic-server-core/tests/cassettes/record_tool_search_cassettes.sh
#   TOOL_SEARCH_RECORD_SET=gateway-streaming GATEWAY_URL=http://localhost:9000 \
#     bash crates/agentic-server-core/tests/cassettes/record_tool_search_cassettes.sh
#   TOOL_SEARCH_RECORD_SET=gateway-websocket GATEWAY_URL=http://localhost:9000 \
#     bash crates/agentic-server-core/tests/cassettes/record_tool_search_cassettes.sh

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPTS_DIR/tool_search"
RETURNED_TOOLS="$BASE_DIR/returned_tools.json"
FUNCTION_OUTPUTS="$BASE_DIR/function_outputs.json"
PROMPTS="$BASE_DIR/prompts.txt"
OPENAI_TOOLS="$BASE_DIR/openai_tools.json"
VLLM_INITIAL_TOOLS="$BASE_DIR/vllm_initial_tools.json"
VLLM_NEXT_TOOLS="$BASE_DIR/vllm_tools_after_search.json"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.6}"
VLLM_MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
VLLM_URL="${VLLM_URL:-}"
GATEWAY_MODEL="${GATEWAY_MODEL:-${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}}"
GATEWAY_URL="${GATEWAY_URL:-}"
TOOL_SEARCH_RECORD_SET="${TOOL_SEARCH_RECORD_SET:-all}"

model_slug() {
  printf '%s' "$1" | tr '/: ' '---'
}

validate_recording() {
  local cassette="$1"
  local projection="$2"
  local initial_tools="$3"
  local next_tools="$4"

  python - "$cassette" "$projection" "$RETURNED_TOOLS" "$initial_tools" "$next_tools" <<'PY'
import json
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
projection = sys.argv[2]
expected_returned_tools = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
expected_initial_tools = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
expected_next_tools = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8")) if sys.argv[5] else None
document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
turns = document.get("turns") or []
if len(turns) != 3:
    raise SystemExit(f"ERROR: expected three recorded turns in {path}, found {len(turns)}")


def terminal_response(turn):
    response = turn.get("response") or {}
    websocket = (turn.get("request") or {}).get("transport") == "websocket"
    expected_status = 101 if websocket else 200
    if response.get("status_code") != expected_status:
        raise SystemExit(f"ERROR: recording returned HTTP {response.get('status_code')}: {response.get('body')}")
    if isinstance(response.get("body"), dict):
        return response["body"]
    for raw in response.get("sse") or []:
        for line in raw.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line.removeprefix("data: "))
            if event.get("type") == "response.completed":
                return event.get("response") or {}
            if event.get("type") in {"error", "response.failed"}:
                raise SystemExit(f"ERROR: streaming recording failed: {event}")
    raise SystemExit("ERROR: streaming recording has no response.completed event")


responses = [terminal_response(turn) for turn in turns]
outputs = [response.get("output") or [] for response in responses]
search_type = "function_call" if projection == "normalized" else "tool_search_call"
first_calls = [item for item in outputs[0] if item.get("type") in {"tool_search_call", "function_call", "custom_tool_call"}]
search_calls = [
    item
    for item in outputs[0]
    if item.get("type") == search_type
    and (projection != "normalized" or item.get("name") == "tool_search")
]
if len(first_calls) != 1 or len(search_calls) != 1:
    raise SystemExit(f"ERROR: expected one {projection} search call, found {search_calls}")
search_arguments = search_calls[0].get("arguments")
if projection != "normalized":
    if search_calls[0].get("execution") != "client" or search_calls[0].get("status") != "completed":
        raise SystemExit(f"ERROR: public search call must be explicitly client/completed: {search_calls[0]}")
    if not isinstance(search_arguments, dict) or not search_arguments.get("query"):
        raise SystemExit(f"ERROR: public search arguments must be a non-empty query object: {search_arguments}")
else:
    if search_calls[0].get("status") != "completed":
        raise SystemExit(f"ERROR: normalized search call must be explicitly completed: {search_calls[0]}")
    try:
        normalized_arguments = json.loads(search_arguments)
    except (TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"ERROR: normalized search arguments are invalid: {search_arguments}") from error
    if not isinstance(normalized_arguments, dict) or not normalized_arguments.get("query"):
        raise SystemExit(f"ERROR: normalized search arguments must contain a query: {normalized_arguments}")

first_stream = (turns[0].get("response") or {}).get("sse") or []
if projection != "normalized" and first_stream:
    events = []
    for raw in first_stream:
        for line in raw.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line.removeprefix("data: ")))
    sequence_numbers = [event.get("sequence_number") for event in events]
    if sequence_numbers != list(range(len(events))):
        raise SystemExit(f"ERROR: public stream sequence numbers are not contiguous: {sequence_numbers}")
    if any(event.get("type") in {"response.function_call_arguments.delta", "response.function_call_arguments.done"} for event in events):
        raise SystemExit("ERROR: public search stream leaked normalized function argument events")
    if any(
        event.get("type") in {"response.output_item.added", "response.output_item.done"}
        and (event.get("item") or {}).get("type") == "function_call"
        and (event.get("item") or {}).get("name") == "tool_search"
        for event in events
    ):
        raise SystemExit("ERROR: public search stream leaked a normalized synthetic function item")
    if any(
        tool.get("type") == "function" and tool.get("name") == "tool_search"
        for event in events
        for tool in ((event.get("response") or {}).get("tools") or [])
    ):
        raise SystemExit("ERROR: public search stream leaked the private synthetic declaration")
    lifecycle = [
        event
        for event in events
        if event.get("type") in {"response.output_item.added", "response.output_item.done"}
        and (event.get("item") or {}).get("type") == "tool_search_call"
    ]
    if [event.get("type") for event in lifecycle] != ["response.output_item.added", "response.output_item.done"]:
        raise SystemExit(f"ERROR: public search lifecycle is incomplete or reordered: {lifecycle}")
    added = lifecycle[0].get("item") or {}
    done = lifecycle[1].get("item") or {}
    if added.get("status") != "in_progress" or added.get("arguments") != {}:
        raise SystemExit(f"ERROR: invalid public search added item: {added}")
    if done.get("status") != "completed" or done.get("arguments") != search_arguments:
        raise SystemExit(f"ERROR: invalid public search done item: {done}")
    if (
        added.get("id") != done.get("id")
        or added.get("call_id") != done.get("call_id")
        or lifecycle[0].get("output_index") != lifecycle[1].get("output_index")
    ):
        raise SystemExit("ERROR: public search lifecycle changed item/call identity")
    if done not in outputs[0]:
        raise SystemExit("ERROR: terminal response output differs from public search done item")

second_calls = [item for item in outputs[1] if item.get("type") in {"tool_search_call", "function_call", "custom_tool_call"}]
loaded_calls = [
    item
    for item in outputs[1]
    if item.get("type") == "function_call" and item.get("name") == "get_weather"
]
if len(second_calls) != 1 or len(loaded_calls) != 1:
    raise SystemExit(f"ERROR: expected one loaded get_weather call, found {loaded_calls}")
if loaded_calls[0].get("status") != "completed":
    raise SystemExit(f"ERROR: loaded get_weather call must be explicitly completed: {loaded_calls[0]}")
try:
    loaded_arguments = json.loads(loaded_calls[0].get("arguments") or "null")
except json.JSONDecodeError as error:
    raise SystemExit("ERROR: loaded function arguments are not valid JSON") from error
if loaded_arguments != {"city": "Paris"}:
    raise SystemExit(f"ERROR: loaded function arguments did not equal city=Paris: {loaded_arguments}")

turn_two_input = (turns[1].get("request") or {}).get("body", {}).get("input") or []
turn_three_input = (turns[2].get("request") or {}).get("body", {}).get("input") or []
search_output_type = "function_call_output" if projection == "normalized" else "tool_search_output"
search_outputs = [
    item
    for item in turn_two_input
    if item.get("type") == search_output_type and item.get("call_id") == search_calls[0].get("call_id")
]
function_outputs = [
    item
    for item in turn_three_input
    if item.get("type") == "function_call_output" and item.get("call_id") == loaded_calls[0].get("call_id")
]
if len(search_outputs) != 1 or len(function_outputs) != 1:
    raise SystemExit("ERROR: recorded continuation call IDs do not link to their preceding calls")
if projection != "normalized":
    if (
        search_outputs[0].get("type") != "tool_search_output"
        or search_outputs[0].get("execution") != "client"
        or search_outputs[0].get("status") != "completed"
        or search_outputs[0].get("tools") != expected_returned_tools
    ):
        raise SystemExit(f"ERROR: invalid public search output: {search_outputs[0]}")
else:
    if search_outputs[0].get("type") != "function_call_output":
        raise SystemExit(f"ERROR: invalid normalized search output: {search_outputs[0]}")
    raw_normalized_output = search_outputs[0].get("output") or ""
    normalized_output = json.loads(raw_normalized_output or "null")
    expected_canonical_output = json.dumps(
        {"tools": expected_returned_tools},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if raw_normalized_output != expected_canonical_output:
        raise SystemExit("ERROR: normalized search output is not canonical")
    if normalized_output != {"tools": expected_returned_tools}:
        raise SystemExit(f"ERROR: normalized search output has no tools: {normalized_output}")
if function_outputs[0].get("type") != "function_call_output":
    raise SystemExit(f"ERROR: invalid loaded function output: {function_outputs[0]}")

messages = [item for item in outputs[2] if item.get("type") == "message"]
final_calls = [item for item in outputs[2] if item.get("type") in {"tool_search_call", "function_call", "custom_tool_call"}]
if final_calls:
    raise SystemExit(f"ERROR: final response contained tool calls: {final_calls}")
text = "".join(
    part.get("text", "")
    for message in messages
    for part in message.get("content") or []
    if part.get("type") == "output_text"
)
if text.strip() != "PARIS_WEATHER_OK":
    raise SystemExit(f"ERROR: final response text did not match: {text!r}")

request_bodies = [(turn.get("request") or {}).get("body") or {} for turn in turns]
if request_bodies[0].get("tools") != expected_initial_tools:
    raise SystemExit("ERROR: first turn tools differ from the initial fixture")
if projection == "public-stored":
    if not all(body.get("store") is True for body in request_bodies):
        raise SystemExit("ERROR: public characterization must use stored continuation")
    if request_bodies[1].get("previous_response_id") != responses[0].get("id"):
        raise SystemExit("ERROR: public turn two does not continue turn one")
    if request_bodies[2].get("previous_response_id") != responses[1].get("id"):
        raise SystemExit("ERROR: public turn three does not continue turn two")
    if "tools" in request_bodies[1] or "tools" in request_bodies[2]:
        raise SystemExit("ERROR: public continuation must omit top-level tools after search")
else:
    if not all(body.get("store") is False for body in request_bodies):
        raise SystemExit("ERROR: manual tool-search replay must use store=false")
    if any("previous_response_id" in body for body in request_bodies):
        raise SystemExit("ERROR: manual tool-search replay must omit previous_response_id")
    if projection == "normalized":
        if request_bodies[1].get("tools") != expected_next_tools or request_bodies[2].get("tools") != expected_next_tools:
            raise SystemExit("ERROR: direct-vLLM continuation did not retain post-search tools")
    elif "tools" in request_bodies[1] or "tools" in request_bodies[2]:
        raise SystemExit("ERROR: gateway manual replay must omit top-level tools after search")
    first_input = request_bodies[0].get("input") or []
    second_input = request_bodies[1].get("input") or []
    third_input = request_bodies[2].get("input") or []
    expected_second_prefix = first_input + outputs[0]
    expected_third_prefix = second_input + outputs[1]
    if second_input[: len(expected_second_prefix)] != expected_second_prefix:
        raise SystemExit("ERROR: manual turn two does not replay full turn-one item history")
    if third_input[: len(expected_third_prefix)] != expected_third_prefix:
        raise SystemExit("ERROR: manual turn three does not replay full prior item history")
PY
}

record_scenario() {
  local endpoint_flag="$1"
  local endpoint="$2"
  local model="$3"
  local tools="$4"
  local next_tools="$5"
  local projection="$6"
  local output="$7"
  local recorder_args=("${@:8}")
  local temporary_output
  local next_tools_args=()
  local continuation_args=()

  temporary_output="$(mktemp "$BASE_DIR/.tool-search-cassette.XXXXXX")"
  if [[ -n "$next_tools" ]]; then
    next_tools_args=(--tools-after-search "$next_tools")
  fi
  if [[ "$projection" == "normalized" || "$projection" == "gateway-public" ]]; then
    continuation_args=(--no-store --manual-item-replay)
  fi

  if ! python "$SCRIPTS_DIR/record_cassette.py" \
    --mode responses \
    --turns 3 \
    "${recorder_args[@]}" \
    --model "$model" \
    "$endpoint_flag" "$endpoint" \
    --tools "$tools" \
    --tool-choice auto \
    --tool-outputs "$FUNCTION_OUTPUTS" \
    --tool-search-output-tools "$RETURNED_TOOLS" \
    "${next_tools_args[@]}" \
    "${continuation_args[@]}" \
    --max-output-tokens 4096 \
    --output "$temporary_output" < "$PROMPTS"
  then
    rm -f -- "$temporary_output"
    return 1
  fi

  if ! validate_recording "$temporary_output" "$projection" "$tools" "$next_tools"; then
    rm -f -- "$temporary_output"
    return 1
  fi
  chmod 664 "$temporary_output"
  mv -- "$temporary_output" "$output"
  printf 'recorded %s\n' "$output"
}

record_provider() {
  local provider="$1"
  local endpoint_flag="$2"
  local endpoint="$3"
  local model="$4"
  local tools="$5"
  local next_tools="$6"
  local projection="$7"
  local prefix="$8"
  local slug

  slug="$(model_slug "$model")"
  printf 'Recording %s blocking tool-search characterization\n' "$provider"
  record_scenario \
    "$endpoint_flag" "$endpoint" "$model" "$tools" "$next_tools" "$projection" \
    "$BASE_DIR/${prefix}-${slug}-nonstreaming.yaml" --no-stream
  printf 'Recording %s streaming tool-search characterization\n' "$provider"
  record_scenario \
    "$endpoint_flag" "$endpoint" "$model" "$tools" "$next_tools" "$projection" \
    "$BASE_DIR/${prefix}-${slug}-streaming.yaml" --stream
}

case "$TOOL_SEARCH_RECORD_SET" in
  openai-reference|openai|direct-vllm|vllm|gateway-nonstreaming|gateway-streaming|gateway-websocket|gateway|all) ;;
  *)
    printf 'ERROR: TOOL_SEARCH_RECORD_SET must be openai-reference, direct-vllm, gateway-nonstreaming, gateway-streaming, gateway-websocket, gateway, or all\n' >&2
    exit 1
    ;;
esac

for required_file in \
  "$RETURNED_TOOLS" \
  "$FUNCTION_OUTPUTS" \
  "$PROMPTS" \
  "$OPENAI_TOOLS" \
  "$VLLM_INITIAL_TOOLS" \
  "$VLLM_NEXT_TOOLS"
do
  if [[ ! -f "$required_file" ]]; then
    printf 'ERROR: required fixture does not exist: %s\n' "$required_file" >&2
    exit 1
  fi
done

if [[ "$TOOL_SEARCH_RECORD_SET" =~ ^(openai-reference|openai|all)$ ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  printf 'ERROR: OPENAI_API_KEY is required for %s\n' "$TOOL_SEARCH_RECORD_SET" >&2
  exit 1
fi

if [[ "$TOOL_SEARCH_RECORD_SET" =~ ^(direct-vllm|vllm|all)$ ]] && [[ -z "$VLLM_URL" ]]; then
  printf 'ERROR: VLLM_URL is required for %s\n' "$TOOL_SEARCH_RECORD_SET" >&2
  exit 1
fi
if [[ "$TOOL_SEARCH_RECORD_SET" =~ ^(gateway-nonstreaming|gateway-streaming|gateway-websocket|gateway|all)$ ]] && [[ -z "$GATEWAY_URL" ]]; then
  printf 'ERROR: GATEWAY_URL is required for %s\n' "$TOOL_SEARCH_RECORD_SET" >&2
  exit 1
fi

if [[ "$TOOL_SEARCH_RECORD_SET" =~ ^(openai-reference|openai|all)$ ]]; then
  record_provider \
    OpenAI --openai https://api.openai.com "$OPENAI_MODEL" \
    "$OPENAI_TOOLS" "" public-stored tool-search-openai-reference
fi

if [[ "$TOOL_SEARCH_RECORD_SET" =~ ^(gateway-nonstreaming|gateway|all)$ ]]; then
  slug="$(model_slug "$GATEWAY_MODEL")"
  printf 'Recording gateway blocking tool-search flow\n'
  record_scenario \
    --gateway "$GATEWAY_URL" "$GATEWAY_MODEL" "$OPENAI_TOOLS" "" gateway-public \
    "$BASE_DIR/tool-search-gateway-${slug}-nonstreaming.yaml" --no-stream
fi

if [[ "$TOOL_SEARCH_RECORD_SET" =~ ^(gateway-streaming|gateway|all)$ ]]; then
  slug="$(model_slug "$GATEWAY_MODEL")"
  printf 'Recording gateway HTTP/SSE tool-search flow\n'
  record_scenario \
    --gateway "$GATEWAY_URL" "$GATEWAY_MODEL" "$OPENAI_TOOLS" "" public-stored \
    "$BASE_DIR/tool-search-gateway-${slug}-streaming.yaml" --stream
fi

if [[ "$TOOL_SEARCH_RECORD_SET" =~ ^(gateway-websocket|gateway|all)$ ]]; then
  slug="$(model_slug "$GATEWAY_MODEL")"
  printf 'Recording gateway WebSocket tool-search flow\n'
  record_scenario \
    --gateway "$GATEWAY_URL" "$GATEWAY_MODEL" "$OPENAI_TOOLS" "" public-stored \
    "$BASE_DIR/tool-search-gateway-${slug}-websocket.yaml" --stream --transport websocket
fi

if [[ "$TOOL_SEARCH_RECORD_SET" =~ ^(direct-vllm|vllm|all)$ ]]; then
  record_provider \
    direct-vLLM --vllm "$VLLM_URL" "$VLLM_MODEL" \
    "$VLLM_INITIAL_TOOLS" "$VLLM_NEXT_TOOLS" normalized tool-search-direct-vllm
fi
