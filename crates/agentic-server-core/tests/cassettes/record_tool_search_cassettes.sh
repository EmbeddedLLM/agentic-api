#!/usr/bin/env bash
# Records the same client-executed tool-search scenario against OpenAI and the gateway
# over HTTP and Responses WebSocket mode.
#
# Usage from the repository root:
#   OPENAI_API_KEY=sk-... \
#     bash crates/agentic-server-core/tests/cassettes/record_tool_search_cassettes.sh
#   TOOL_SEARCH_RECORD_SET=gateway GATEWAY_URL=http://127.0.0.1:3018 \
#     bash crates/agentic-server-core/tests/cassettes/record_tool_search_cassettes.sh
#   TOOL_SEARCH_TRANSPORT_SET=websocket TOOL_SEARCH_RECORD_SET=all OPENAI_API_KEY=sk-... \
#     GATEWAY_URL=http://127.0.0.1:3018 \
#     bash crates/agentic-server-core/tests/cassettes/record_tool_search_cassettes.sh

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPTS_DIR/tool_search"
TOOLS_FILE="$BASE_DIR/tools.json"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:9000}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
MODEL_SLUG="$(echo "$MODEL" | tr '/: ' '---')"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.6}"
OPENAI_MODEL_SLUG="$(echo "$OPENAI_MODEL" | tr '/: ' '---')"
TOOL_SEARCH_RECORD_SET="${TOOL_SEARCH_RECORD_SET:-all}"
TOOL_SEARCH_TRANSPORT_SET="${TOOL_SEARCH_TRANSPORT_SET:-all}"
PROMPT='Call tool_search now to find the shipping ETA tool for order_42. Do not call get_shipping_eta yet and do not answer without calling tool_search.'

validate_recording() {
  local file="$1"
  local stream_flag="$2"
  local transport="$3"

  python - "$file" "$stream_flag" "$transport" <<'PY'
import json
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
streaming = sys.argv[2] == "--stream"
transport = sys.argv[3]
document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
turns = document.get("turns") or []
if len(turns) != 1:
    raise SystemExit(f"ERROR: expected one recorded turn in {path}, found {len(turns)}")

turn = turns[0]
request = (turn.get("request") or {}).get("body") or {}
tools = request.get("tools") or []
search_tools = [tool for tool in tools if tool.get("type") == "tool_search"]
if len(search_tools) != 1 or search_tools[0].get("execution") != "client":
    raise SystemExit("ERROR: cassette request must contain one client-executed tool_search declaration")
deferred = [tool for tool in tools if tool.get("name") == "get_shipping_eta"]
if (
    len(deferred) != 1
    or deferred[0].get("defer_loading") is not True
    or deferred[0].get("strict") is not False
):
    raise SystemExit("ERROR: cassette request must contain deferred get_shipping_eta with strict false")

response = turn.get("response") or {}
if transport == "websocket":
    if (turn.get("request") or {}).get("transport") != "websocket":
        raise SystemExit("ERROR: WebSocket cassette request must identify the websocket transport")
    if (turn.get("request") or {}).get("method") != "WEBSOCKET":
        raise SystemExit("ERROR: WebSocket cassette request must use the WEBSOCKET method")
    if request.get("type") != "response.create" or "stream" in request:
        raise SystemExit("ERROR: WebSocket request must be response.create without the HTTP stream field")
    if response.get("status_code") != 101:
        raise SystemExit(f"ERROR: WebSocket recording did not upgrade: {response.get('status_code')}")
elif response.get("status_code") != 200:
    raise SystemExit(f"ERROR: recording returned HTTP {response.get('status_code')}: {response.get('body')}")

if streaming:
    if transport == "websocket":
        raw_events = response.get("websocket") or []
        if not raw_events:
            raise SystemExit("ERROR: WebSocket recording must contain response.websocket messages")
    else:
        raw_events = []
        for raw in response.get("sse") or []:
            raw_events.extend(
                line.removeprefix("data: ")
                for line in raw.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            )

    events = []
    for raw in raw_events:
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    errors = [event.get("error") for event in events if event.get("type") == "error"]
    if errors:
        raise SystemExit(f"ERROR: streaming recording returned an error event: {errors[0]}")
    for event_type in ("response.created", "response.in_progress", "response.completed"):
        lifecycle = [event.get("response") for event in events if event.get("type") == event_type]
        if len(lifecycle) != 1:
            raise SystemExit(f"ERROR: expected one {event_type} event, found {len(lifecycle)}")
        lifecycle_tools = (lifecycle[0] or {}).get("tools") or []
        lifecycle_search = [tool for tool in lifecycle_tools if tool.get("type") == "tool_search"]
        lifecycle_deferred = [tool for tool in lifecycle_tools if tool.get("name") == "get_shipping_eta"]
        if len(lifecycle_search) != 1 or lifecycle_search[0].get("execution") != "client":
            raise SystemExit(f"ERROR: {event_type} must expose native client tool_search")
        if len(lifecycle_deferred) != 1 or lifecycle_deferred[0].get("defer_loading") is not True:
            raise SystemExit(f"ERROR: {event_type} must preserve deferred get_shipping_eta")
        if lifecycle_deferred[0].get("strict") is not False:
            raise SystemExit(f"ERROR: {event_type} must preserve strict false on get_shipping_eta")
    added = [
        (position, event.get("item") or {})
        for position, event in enumerate(events)
        if event.get("type") == "response.output_item.added"
        and (event.get("item") or {}).get("type") == "tool_search_call"
    ]
    done = [
        (position, event.get("item") or {})
        for position, event in enumerate(events)
        if event.get("type") == "response.output_item.done"
        and (event.get("item") or {}).get("type") == "tool_search_call"
    ]
    completed_positions = [
        position for position, event in enumerate(events) if event.get("type") == "response.completed"
    ]
    if len(added) != 1 or len(done) != 1 or len(completed_positions) != 1:
        raise SystemExit("ERROR: expected one added, done, and completed tool-search lifecycle")
    if not added[0][0] < done[0][0] < completed_positions[0]:
        raise SystemExit("ERROR: tool-search lifecycle must be added, done, then response.completed")
    added_call_id = added[0][1].get("call_id")
    if not isinstance(added_call_id, str) or not added_call_id or done[0][1].get("call_id") != added_call_id:
        raise SystemExit("ERROR: tool_search_call must preserve a nonempty call_id from added through done")
    if added[0][1].get("status") != "in_progress" or done[0][1].get("status") != "completed":
        raise SystemExit("ERROR: tool_search_call must transition from in_progress to completed")
    completed = [event.get("response") for event in events if event.get("type") == "response.completed"]
    body = completed[-1] if completed else None
else:
    body = response.get("body")

if not isinstance(body, dict) or body.get("status") != "completed":
    raise SystemExit(f"ERROR: recording did not complete: {body}")
output = body.get("output") or []
if any(item.get("type") == "function_call" and item.get("name") == "tool_search" for item in output):
    raise SystemExit("ERROR: provider function fallback leaked instead of canonical tool_search_call")
if streaming and any(
    event.get("item", {}).get("type") == "function_call"
    and event.get("item", {}).get("name") == "tool_search"
    for event in events
):
    raise SystemExit("ERROR: provider function fallback leaked in a streaming event")
calls = [item for item in output if item.get("type") == "tool_search_call"]
if len(calls) != 1:
    raise SystemExit(f"ERROR: expected one tool_search_call, found {len(calls)}")
call = calls[0]
if call.get("execution") != "client" or call.get("status") != "completed":
    raise SystemExit(f"ERROR: tool_search_call is not a completed client call: {call}")
if not isinstance(call.get("call_id"), str) or not call["call_id"]:
    raise SystemExit("ERROR: client tool_search_call must have a nonempty call_id")
if streaming and call["call_id"] != added_call_id:
    raise SystemExit(
        "ERROR: tool_search_call must preserve the same call_id in added, done, and response.completed output"
    )
PY
}

record_single_turn() {
  local endpoint_flag="$1"
  local endpoint="$2"
  local model="$3"
  local output="$4"
  local stream_flag="$5"
  local transport="$6"
  local temporary_output

  temporary_output="$(mktemp "$BASE_DIR/.tool-search-cassette.XXXXXX")"
  if ! printf '%s\n' "$PROMPT" \
    | python "$SCRIPTS_DIR/record_cassette.py" \
        --mode responses \
        --turns 1 \
        --transport "$transport" \
        "$stream_flag" \
        "$endpoint_flag" "$endpoint" \
        --model "$model" \
        --tools "$TOOLS_FILE" \
        --tool-choice required \
        --max-output-tokens 1024 \
        --output "$temporary_output"
  then
    rm -f -- "$temporary_output"
    return 1
  fi

  if ! validate_recording "$temporary_output" "$stream_flag" "$transport"; then
    rm -f -- "$temporary_output"
    return 1
  fi
  mv -- "$temporary_output" "$output"
  printf 'Recorded %s\n' "$output"
}

record_provider_suite() {
  local endpoint_flag="$1"
  local endpoint="$2"
  local model="$3"
  local output_prefix="$4"

  if [[ "$TOOL_SEARCH_TRANSPORT_SET" == "http" || "$TOOL_SEARCH_TRANSPORT_SET" == "all" ]]; then
    record_single_turn \
      "$endpoint_flag" "$endpoint" "$model" "$BASE_DIR/${output_prefix}-streaming.yaml" --stream http
    record_single_turn \
      "$endpoint_flag" "$endpoint" "$model" "$BASE_DIR/${output_prefix}-nonstreaming.yaml" --no-stream http
  fi
  if [[ "$TOOL_SEARCH_TRANSPORT_SET" == "websocket" || "$TOOL_SEARCH_TRANSPORT_SET" == "all" ]]; then
    record_single_turn \
      "$endpoint_flag" "$endpoint" "$model" "$BASE_DIR/${output_prefix}-websocket-streaming.yaml" --stream websocket
  fi
}

case "$TOOL_SEARCH_RECORD_SET" in
  gateway|openai|all) ;;
  *)
    echo "ERROR: TOOL_SEARCH_RECORD_SET must be gateway, openai, or all" >&2
    exit 1
    ;;
esac

case "$TOOL_SEARCH_TRANSPORT_SET" in
  http|websocket|all) ;;
  *)
    echo "ERROR: TOOL_SEARCH_TRANSPORT_SET must be http, websocket, or all" >&2
    exit 1
    ;;
esac

if [[ ! -f "$TOOLS_FILE" ]]; then
  echo "ERROR: tool-search tools file does not exist: $TOOLS_FILE" >&2
  exit 1
fi

if [[ "$TOOL_SEARCH_RECORD_SET" == "openai" || "$TOOL_SEARCH_RECORD_SET" == "all" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY must be set for TOOL_SEARCH_RECORD_SET=$TOOL_SEARCH_RECORD_SET" >&2
    exit 1
  fi
fi

mkdir -p "$BASE_DIR"

if [[ "$TOOL_SEARCH_RECORD_SET" == "openai" || "$TOOL_SEARCH_RECORD_SET" == "all" ]]; then
  record_provider_suite \
    --openai https://api.openai.com "$OPENAI_MODEL" \
    "tool-search-openai-reference-${OPENAI_MODEL_SLUG}"
fi

if [[ "$TOOL_SEARCH_RECORD_SET" == "gateway" || "$TOOL_SEARCH_RECORD_SET" == "all" ]]; then
  record_provider_suite \
    --gateway "$GATEWAY_URL" "$MODEL" \
    "tool-search-gateway-${MODEL_SLUG}"
fi
