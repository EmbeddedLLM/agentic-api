import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import claude_code_replay_server as replay


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
CASSETTE = (
    REPOSITORY_ROOT
    / "crates/agentic-server-core/tests/cassettes/messages/"
    "messages-web-search-Qwen-Qwen3-30B-A3B-FP8-streaming.yaml"
)
RESPONSES_CASSETTE = (
    REPOSITORY_ROOT
    / "crates/agentic-server-core/tests/cassettes/reasoning/responses/"
    "reasoning-single-Qwen-Qwen3-30B-A3B-FP8-streaming.yaml"
)
QWEN_MODEL = "Qwen/Qwen3-30B-A3B-FP8"


def cache_control(ttl: str = "5m") -> dict[str, str]:
    return {"type": "ephemeral", "ttl": ttl}


def messages_request(*, with_tool_result: bool = False) -> dict:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Find Rust.", "cache_control": cache_control()}],
        }
    ]
    if with_tool_result:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "tool-1", "name": "WebSearch", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "Rust 1.89.0"}],
                },
            ]
        )
    return {
        "model": "qwen3",
        "stream": True,
        "system": [
            {"type": "text", "text": "attribution"},
            {"type": "text", "text": "instructions", "cache_control": cache_control("1h")},
            {"type": "text", "text": "runtime", "cache_control": cache_control()},
        ],
        "messages": messages,
        "tools": [
            {
                "name": "WebSearch",
                "description": "Search the web",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    }


def messages_transport(**header_overrides: str) -> dict:
    headers = {
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "interleaved-thinking-2025-05-14",
        "x-api-key": "ci-placeholder",
        "x-claude-code-session-id": "session-1",
    }
    headers.update(header_overrides)
    return {"path": "/v1/messages?beta=true", "headers": headers}


def capture_records(first_request: dict, second_request: dict) -> list[dict]:
    return [
        {"kind": "messages_transport", "body": messages_transport()},
        {"kind": "messages", "body": first_request},
        {"kind": "search", "body": {"query": "latest stable Rust release"}},
        {"kind": "messages_transport", "body": messages_transport()},
        {"kind": "messages", "body": second_request},
    ]


class ReplayServerTests(unittest.TestCase):
    def test_load_turns_reads_recorded_streams(self) -> None:
        turns = replay.load_turns(CASSETTE)

        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].status_code, 200)
        self.assertEqual(turns[0].content_type, "text/event-stream; charset=utf-8")
        self.assertIn(b"event: message_start", turns[0].body)

    def test_adapt_stream_uses_declared_claude_tool_name(self) -> None:
        recorded = 'data: {"content_block":{"name":"web_search"}}\n\n'

        adapted = replay.adapt_stream(recorded, "WebSearch")

        self.assertIn('"name":"WebSearch"', adapted)
        self.assertNotIn('"name":"web_search"', adapted)

    def test_load_turns_reads_recorded_responses_stream(self) -> None:
        turns = replay.load_turns(RESPONSES_CASSETTE)

        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].status_code, 200)
        self.assertIn(b"event: response.created", turns[0].body)
        self.assertIn(b"HELLO", turns[0].body)

    def test_validate_responses_capture_accepts_codex_wire_shape(self) -> None:
        records = [
            {
                "kind": "responses_transport",
                "body": {"path": "/v1/responses", "headers": {}},
            },
            {
                "kind": "responses",
                "body": {
                    "model": QWEN_MODEL,
                    "stream": True,
                    "input": "Reply with exactly one word: HELLO",
                },
            },
        ]

        replay.validate_responses_capture(records, QWEN_MODEL)

    def test_validate_responses_capture_rejects_wrong_model(self) -> None:
        records = [
            {
                "kind": "responses_transport",
                "body": {"path": "/v1/responses", "headers": {}},
            },
            {
                "kind": "responses",
                "body": {"model": "claude-sonnet-4-5", "stream": True, "input": "HELLO"},
            },
        ]

        with self.assertRaisesRegex(AssertionError, "requested model"):
            replay.validate_responses_capture(records, QWEN_MODEL)

    def test_responses_route_replays_recorded_stream_and_captures_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            capture_path = Path(temp_dir) / "capture.jsonl"
            capture_path.write_text("")
            state = replay.ReplayState(replay.load_turns(RESPONSES_CASSETTE), capture_path)
            server = ThreadingHTTPServer(("127.0.0.1", 0), replay.make_handler(state))
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{server.server_port}/v1/responses",
                    data=json.dumps(
                        {"model": QWEN_MODEL, "stream": True, "input": "HELLO"}
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    body = response.read()
                records = replay.load_capture(capture_path)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

        self.assertIn(b"event: response.created", body)
        replay.validate_responses_capture(records, QWEN_MODEL)

    def test_validate_capture_accepts_claude_code_wire_shape_and_tool_round(self) -> None:
        records = capture_records(messages_request(), messages_request(with_tool_result=True))

        replay.validate_capture(records)

    def test_validate_capture_rejects_more_than_four_cache_breakpoints(self) -> None:
        first = messages_request()
        second = messages_request(with_tool_result=True)
        for request in (first, second):
            request["tools"][0]["cache_control"] = cache_control("1h")
            request["system"].append(
                {"type": "text", "text": "extra runtime", "cache_control": cache_control()}
            )

        with self.assertRaisesRegex(AssertionError, "at most four cache breakpoints"):
            replay.validate_capture(capture_records(first, second))

    def test_validate_capture_rejects_short_ttl_before_long_ttl(self) -> None:
        first = messages_request()
        second = messages_request(with_tool_result=True)
        for request in (first, second):
            request["tools"][0]["cache_control"] = cache_control()

        with self.assertRaisesRegex(AssertionError, "1h cache breakpoints must precede 5m"):
            replay.validate_capture(capture_records(first, second))

    def test_validate_capture_rejects_missing_claude_transport_header(self) -> None:
        first_transport = messages_transport()
        del first_transport["headers"]["anthropic-beta"]
        records = capture_records(messages_request(), messages_request(with_tool_result=True))
        records[0]["body"] = first_transport

        with self.assertRaises(AssertionError):
            replay.validate_capture(records)

    def test_validate_capture_requires_two_messages_rounds(self) -> None:
        records = [
            {"kind": "messages", "body": messages_request()},
            {"kind": "search", "body": {"query": "latest stable Rust release"}},
        ]

        with self.assertRaisesRegex(AssertionError, "two Messages rounds"):
            replay.validate_capture(records)


if __name__ == "__main__":
    unittest.main()
