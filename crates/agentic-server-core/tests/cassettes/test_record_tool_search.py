"""Focused offline tests for client tool-search cassette recording."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

import record_cassette


RETURNED_TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
        "defer_loading": True,
    }
]


class RecordToolSearchTests(unittest.TestCase):
    def test_proxy_start_requires_an_owned_listener_before_request_execution(self) -> None:
        class FailedServer:
            started = False
            should_exit = False
            force_exit = False

            def run(self) -> None:
                return None

        failed_server = FailedServer()
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "capture.yaml"
            with (
                mock.patch.object(record_cassette.uvicorn, "Server", return_value=failed_server),
                mock.patch.object(record_cassette, "run_responses") as run_responses,
            ):
                result = CliRunner().invoke(
                    record_cassette.main,
                    [
                        "--mode", "responses",
                        "--turns", "1",
                        "--gateway", "http://gateway.test",
                        "--model", "test-model",
                        "--no-stream",
                        "--output", str(capture),
                    ],
                )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("failed to own", str(result.exception))
        self.assertTrue(failed_server.should_exit)
        run_responses.assert_not_called()

    def test_gateway_cli_profile_accepts_public_store_false_manual_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            tools = directory_path / "tools.json"
            outputs = directory_path / "outputs.json"
            returned = directory_path / "returned.json"
            capture = directory_path / "capture.yaml"
            tools.write_text(json.dumps([{"type": "tool_search", "execution": "client"}]), encoding="utf-8")
            outputs.write_text(json.dumps({"get_weather": "sunny"}), encoding="utf-8")
            returned.write_text(json.dumps(RETURNED_TOOLS), encoding="utf-8")

            with (
                mock.patch.object(record_cassette, "_start_proxy", return_value=object()),
                mock.patch.object(record_cassette, "_stop_proxy"),
                mock.patch.object(record_cassette, "run_responses") as run_responses,
            ):
                result = CliRunner().invoke(
                    record_cassette.main,
                    [
                        "--mode", "responses",
                        "--turns", "3",
                        "--gateway", "http://gateway.test",
                        "--model", "test-model",
                        "--no-stream",
                        "--no-store",
                        "--manual-item-replay",
                        "--tools", str(tools),
                        "--tool-outputs", str(outputs),
                        "--tool-search-output-tools", str(returned),
                        "--output", str(capture),
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIs(run_responses.call_args.kwargs["manual_item_replay"], True)
        self.assertIs(run_responses.call_args.args[4], False, "gateway profile must set store=false")

    def test_gateway_websocket_cli_profile_accepts_stored_tool_search_flow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            tools = directory_path / "tools.json"
            outputs = directory_path / "outputs.json"
            returned = directory_path / "returned.json"
            capture = directory_path / "capture.yaml"
            tools.write_text(json.dumps([{"type": "tool_search", "execution": "client"}]), encoding="utf-8")
            outputs.write_text(json.dumps({"get_weather": "sunny"}), encoding="utf-8")
            returned.write_text(json.dumps(RETURNED_TOOLS), encoding="utf-8")

            with (
                mock.patch.object(record_cassette, "_start_proxy") as start_proxy,
                mock.patch.object(record_cassette, "run_responses") as run_responses,
            ):
                result = CliRunner().invoke(
                    record_cassette.main,
                    [
                        "--mode", "responses",
                        "--turns", "3",
                        "--gateway", "http://gateway.test",
                        "--transport", "websocket",
                        "--model", "test-model",
                        "--stream",
                        "--tools", str(tools),
                        "--tool-outputs", str(outputs),
                        "--tool-search-output-tools", str(returned),
                        "--output", str(capture),
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        start_proxy.assert_not_called()
        self.assertIs(run_responses.call_args.args[4], True)
        self.assertEqual(run_responses.call_args.args[7], "websocket")

    def test_websocket_handshake_preserves_coalesced_first_frame(self) -> None:
        class Socket:
            def __init__(self) -> None:
                self.chunks = [b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n\r\n\x81\x02ok"]

            def recv(self, _size: int) -> bytes:
                return self.chunks.pop(0) if self.chunks else b""

        client = record_cassette.WebSocketClient("ws://gateway.test", {})
        client.sock = Socket()

        response = client._read_http_response()

        self.assertTrue(response.endswith("\r\n\r\n"))
        self.assertEqual(client.receive_text(), "ok")

    def test_websocket_rejects_oversized_frames_before_reading_payload(self) -> None:
        class Socket:
            def __init__(self) -> None:
                self.chunks = [b"\x81\x7f", (5).to_bytes(8, "big")]

            def recv(self, _size: int) -> bytes:
                return self.chunks.pop(0) if self.chunks else b""

        client = record_cassette.WebSocketClient("ws://gateway.test", {})
        client.sock = Socket()

        with (
            mock.patch.object(record_cassette, "MAX_WEBSOCKET_FRAME_BYTES", 4),
            self.assertRaisesRegex(ValueError, "frame exceeded"),
        ):
            client.receive_text()

    def test_websocket_rejects_oversized_fragmented_messages(self) -> None:
        class Socket:
            def __init__(self) -> None:
                self.chunks = [b"\x01\x03", b"abc", b"\x80\x02", b"de"]

            def recv(self, _size: int) -> bytes:
                return self.chunks.pop(0) if self.chunks else b""

        socket = Socket()
        client = record_cassette.WebSocketClient("ws://gateway.test", {})
        client.sock = socket

        with (
            mock.patch.object(record_cassette, "MAX_WEBSOCKET_MESSAGE_BYTES", 4),
            self.assertRaisesRegex(ValueError, "message exceeded"),
        ):
            client.receive_text()

        self.assertEqual(socket.chunks, [b"de"], "oversized payload must not be read")

    def test_websocket_recording_rejects_an_oversized_capture(self) -> None:
        class Socket:
            def __enter__(self) -> "Socket":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def send_text(self, _text: str) -> None:
                return None

            def receive_text(self) -> str:
                return "12345"

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.yaml"
            with (
                mock.patch.object(record_cassette, "WebSocketClient", return_value=Socket()),
                mock.patch.object(record_cassette, "MAX_WEBSOCKET_CAPTURE_BYTES", 4),
                mock.patch.object(record_cassette, "_append_turn") as append_turn,
                self.assertRaisesRegex(ValueError, "byte recording limit"),
            ):
                record_cassette._send_websocket(
                    {"model": "test", "input": "hello"},
                    "http://gateway.test",
                    {},
                    output,
                )

        append_turn.assert_not_called()

    def test_websocket_recording_stops_on_response_failed(self) -> None:
        failed = {
            "type": "response.failed",
            "response": {
                "id": "resp_failed",
                "status": "failed",
                "error": {"code": "provider_failure", "message": "stopped"},
            },
        }

        class Socket:
            def __init__(self) -> None:
                self.messages = [json.dumps(failed), None]

            def __enter__(self) -> "Socket":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def send_text(self, _text: str) -> None:
                return None

            def receive_text(self) -> str | None:
                return self.messages.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.yaml"
            with (
                mock.patch.object(record_cassette, "WebSocketClient", return_value=Socket()),
                mock.patch.object(record_cassette, "_append_turn") as append_turn,
            ):
                response = record_cassette._send_websocket(
                    {"model": "test", "input": "hello"},
                    "http://gateway.test",
                    {},
                    output,
                )

        self.assertEqual(response, failed["response"])
        turn = append_turn.call_args.args[1]
        self.assertEqual(turn["response"]["status_code"], 101)
        self.assertEqual(json.loads(turn["response"]["websocket"][0]), failed)
        self.assertTrue(turn["response"]["sse"][0].startswith("event: response.failed\n"))

    def test_public_returned_fixture_preserves_deferral_but_vllm_next_tools_clear_it(self) -> None:
        fixture_directory = Path(__file__).with_name("tool_search")
        returned = json.loads((fixture_directory / "returned_tools.json").read_text(encoding="utf-8"))
        vllm_next = json.loads(
            (fixture_directory / "vllm_tools_after_search.json").read_text(encoding="utf-8")
        )
        self.assertIs(returned[0]["defer_loading"], True)
        loaded = next(tool for tool in vllm_next if tool.get("name") == "get_weather")
        self.assertNotIn("defer_loading", loaded)

    def test_public_search_then_function_outputs_share_one_continuation_builder(self) -> None:
        search_calls = record_cassette._extract_tool_calls(
            {
                "output": [
                    {
                        "id": "tsc_public",
                        "type": "tool_search_call",
                        "call_id": "call_search_public",
                        "execution": "client",
                        "status": "completed",
                        "arguments": {"query": "weather tool"},
                    }
                ]
            }
        )

        search_continuation = record_cassette._build_tool_continuation(
            search_calls,
            {"get_weather": '{"temperature_c":21}'},
            RETURNED_TOOLS,
            None,
        )

        self.assertTrue(search_continuation.loaded_search_tools)
        self.assertEqual(
            search_continuation.input_items,
            [
                {
                    "type": "tool_search_output",
                    "call_id": "call_search_public",
                    "execution": "client",
                    "status": "completed",
                    "tools": RETURNED_TOOLS,
                }
            ],
        )

        function_calls = record_cassette._extract_tool_calls(
            {
                "output": [
                    {
                        "id": "fc_weather",
                        "type": "function_call",
                        "call_id": "call_weather",
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    }
                ]
            }
        )
        function_continuation = record_cassette._build_tool_continuation(
            function_calls,
            {"get_weather": '{"temperature_c":21}'},
            RETURNED_TOOLS,
            None,
        )

        self.assertFalse(function_continuation.loaded_search_tools)
        self.assertEqual(
            function_continuation.input_items,
            [
                {
                    "type": "function_call_output",
                    "call_id": "call_weather",
                    "output": '{"temperature_c":21}',
                }
            ],
        )

    def test_normalized_search_uses_canonical_function_output_projection(self) -> None:
        calls = record_cassette._extract_tool_calls(
            {
                "output": [
                    {
                        "id": "fc_search",
                        "type": "function_call",
                        "call_id": "call_search_normalized",
                        "name": "tool_search",
                        "arguments": '{"query":"weather tool"}',
                    }
                ]
            }
        )

        continuation = record_cassette._build_tool_continuation(
            calls,
            {"get_weather": '{"temperature_c":21}'},
            RETURNED_TOOLS,
            None,
        )

        self.assertTrue(continuation.loaded_search_tools)
        self.assertEqual(continuation.input_items[0]["type"], "function_call_output")
        self.assertEqual(continuation.input_items[0]["call_id"], "call_search_normalized")
        expected = json.dumps(
            {"tools": RETURNED_TOOLS},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.assertEqual(continuation.input_items[0]["output"], expected)
        self.assertEqual(json.loads(continuation.input_items[0]["output"]), {"tools": RETURNED_TOOLS})

    def test_tool_continuations_reject_empty_ids_and_missing_search_tools(self) -> None:
        for call in (
            {"type": "tool_search_call", "call_id": ""},
            {"type": "function_call", "name": "tool_search"},
            {"type": "function_call", "name": "get_weather", "call_id": "   "},
        ):
            with self.subTest(call=call), self.assertRaises(ValueError):
                record_cassette._extract_tool_calls({"output": [call]})

        with self.assertRaises(ValueError):
            record_cassette._build_tool_continuation(
                [
                    {
                        "type": "tool_search_call",
                        "call_id": "call_without_tools",
                    }
                ],
                {},
                None,
                None,
            )
        with self.assertRaisesRegex(ValueError, "explicit output fixture"):
            record_cassette._build_tool_continuation(
                [
                    {
                        "type": "function_call",
                        "name": "get_weather",
                        "call_id": "call_without_output",
                    }
                ],
                {},
                RETURNED_TOOLS,
                None,
            )

    def test_existing_outputs_and_central_secret_validation(self) -> None:
        continuation = record_cassette._build_tool_continuation(
            [
                {
                    "type": "function_call",
                    "call_id": "call_function",
                    "name": "lookup",
                },
                {
                    "type": "custom_tool_call",
                    "call_id": "call_custom",
                    "name": "raw_echo",
                },
            ],
            {"lookup": "function result", "raw_echo": "custom result"},
            None,
            "continue",
        )
        self.assertEqual(
            [item["type"] for item in continuation.input_items],
            ["function_call_output", "custom_tool_call_output", "message"],
        )

        safe_turn = {
            "request": {
                "headers": {"authorization": "Bearer live-key"},
                "query_params": {},
                "body": {
                    "input": "hello",
                    "parameters": {
                        "type": "object",
                        "properties": {"password": {"type": "string"}},
                    },
                },
            },
            "response": {"headers": {}, "body": {"output": []}},
        }
        prepared = record_cassette._prepare_turn_for_write(safe_turn, environment={})
        self.assertEqual(prepared["request"]["headers"]["authorization"], "Bearer ***")
        self.assertEqual(safe_turn["request"]["headers"]["authorization"], "Bearer live-key")

        unsafe_turns = (
            {
                "request": {
                    "headers": {},
                    "query_params": {},
                    "body": {"tools": [{"headers": {"x-api-key": "nested-secret"}}]},
                },
                "response": {},
            },
            {
                "request": {
                    "headers": {},
                    "query_params": {"api_key": "query-secret"},
                    "body": {},
                },
                "response": {},
            },
            {
                "request": {"headers": {}, "query_params": {}, "body": {}},
                "response": {"body": {"error": {"message": "failed with sk-live-secret"}}},
            },
            {
                "request": {
                    "headers": {},
                    "query_params": {},
                    "body": {
                        "tools": [
                            {
                                "type": "mcp",
                                "server_url": "https://mcp.example.test/run",
                                "headers": {"X-Tenant": "tenant-secret"},
                            }
                        ]
                    },
                },
                "response": {},
            },
            {
                "request": {
                    "headers": {},
                    "query_params": {},
                    "body": {
                        "tools": [
                            {
                                "type": "mcp",
                                "server_url": "https://user@mcp.example.test/run?tenant=private",
                            }
                        ]
                    },
                },
                "response": {},
            },
            {
                "request": {
                    "headers": {},
                    "query_params": {},
                    "body": {
                        "image_url": "https://files.example.test/object?X-Amz-Credential=credential&X-Amz-Signature=signed"
                    },
                },
                "response": {},
            },
            {
                "request": {"headers": {}, "query_params": {}, "body": {}},
                "response": {
                    "body": {
                        "output": '{"tools":[{"type":"mcp","headers":{"X-Tenant":"nested-secret"}}]}'
                    }
                },
            },
        )
        environments = (
            {},
            {},
            {"OPENAI_API_KEY": "sk-live-secret"},
            {},
            {},
            {},
            {},
        )
        for unsafe_turn, environment in zip(unsafe_turns, environments, strict=True):
            with self.subTest(turn=unsafe_turn), self.assertRaises(
                record_cassette.SecretRecordingError
            ):
                record_cassette._prepare_turn_for_write(unsafe_turn, environment=environment)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.yaml"
            with self.assertRaises(record_cassette.SecretRecordingError):
                record_cassette._append_turn(
                    output,
                    unsafe_turns[0],
                    environment={},
                )
            self.assertFalse(output.exists())

    def test_append_turn_preserves_existing_mode_and_cleans_temporary_file(self) -> None:
        turn = {
            "request": {"headers": {}, "query_params": {}, "body": {"input": "safe"}},
            "response": {"headers": {}, "body": {"output": []}},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.yaml"
            output.write_text("turns: []\n", encoding="utf-8")
            output.chmod(0o664)

            record_cassette._append_turn(output, turn, environment={})

            self.assertEqual(output.stat().st_mode & 0o777, 0o664)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*")), [])

            new_output = Path(directory) / "new-capture.yaml"
            record_cassette._append_turn(new_output, turn, environment={})
            self.assertEqual(new_output.stat().st_mode & 0o077, 0)
            self.assertEqual(list(new_output.parent.glob(f".{new_output.name}.*")), [])

    def test_linear_responses_flow_switches_to_next_tools_and_rejects_branches(self) -> None:
        initial_tools = [{"type": "function", "name": "tool_search"}]
        next_tools = initial_tools + RETURNED_TOOLS
        responses = [
            {
                "id": "resp_search",
                "output": [
                    {
                        "type": "function_call",
                        "name": "tool_search",
                        "call_id": "call_search",
                        "status": "completed",
                        "arguments": '{"query":"weather tool"}',
                    }
                ],
            },
            {
                "id": "resp_function",
                "output": [
                    {
                        "type": "function_call",
                        "name": "get_weather",
                        "call_id": "call_weather",
                        "status": "completed",
                        "arguments": '{"city":"Paris"}',
                    }
                ],
            },
            {"id": "resp_final", "output": [{"type": "message"}]},
        ]
        sent_bodies: list[dict] = []

        def fake_send(_client: object, body: dict, *_args: object, **_kwargs: object) -> dict:
            sent_bodies.append(body)
            return responses[len(sent_bodies) - 1]

        with (
            mock.patch.object(
                record_cassette,
                "_prompt",
                side_effect=["find a weather tool", "call it", "finish"],
            ),
            mock.patch.object(record_cassette, "_send", side_effect=fake_send),
        ):
            record_cassette.run_responses(
                client=object(),
                turns=3,
                model="test-model",
                stream=False,
                store=False,
                branches=[],
                proxy_url="http://unused",
                tools=initial_tools,
                tool_outputs={"get_weather": '{"temperature_c":21}'},
                tool_search_output_tools=RETURNED_TOOLS,
                tools_after_search=next_tools,
                manual_item_replay=True,
            )

        self.assertEqual([body["tools"] for body in sent_bodies], [initial_tools, next_tools, next_tools])
        self.assertTrue(all(body["store"] is False for body in sent_bodies))
        self.assertTrue(all("previous_response_id" not in body for body in sent_bodies))

        turn_one_input = sent_bodies[0]["input"]
        turn_two_input = sent_bodies[1]["input"]
        turn_three_input = sent_bodies[2]["input"]
        self.assertEqual(
            turn_one_input,
            [{"type": "message", "role": "user", "content": "find a weather tool"}],
        )
        self.assertEqual(turn_two_input[: len(turn_one_input)], turn_one_input)
        self.assertEqual(turn_two_input[1]["type"], "function_call")
        self.assertEqual(turn_two_input[1]["call_id"], "call_search")
        self.assertEqual(turn_two_input[2]["type"], "function_call_output")
        self.assertEqual(turn_two_input[2]["call_id"], "call_search")
        self.assertEqual(turn_three_input[: len(turn_two_input)], turn_two_input)
        loaded_call = next(
            item
            for item in turn_three_input[len(turn_two_input) :]
            if item.get("type") == "function_call"
        )
        loaded_output = next(
            item
            for item in turn_three_input[len(turn_two_input) :]
            if item.get("type") == "function_call_output"
        )
        self.assertEqual(loaded_call["call_id"], "call_weather")
        self.assertEqual(loaded_output["call_id"], "call_weather")
        self.assertTrue(all(body["parallel_tool_calls"] is False for body in sent_bodies))

        with self.assertRaisesRegex(record_cassette.click.UsageError, "branching"):
            record_cassette.run_responses(
                client=object(),
                turns=3,
                model="test-model",
                stream=False,
                store=True,
                branches=[(1, 2)],
                proxy_url="http://unused",
                tools=initial_tools,
                tool_outputs={"get_weather": "result"},
                tool_search_output_tools=RETURNED_TOOLS,
            )

    def test_public_linear_responses_flow_keeps_public_top_level_tools(self) -> None:
        public_tools = [
            {"type": "tool_search", "execution": "client"},
            {
                "type": "function",
                "name": "get_weather",
                "defer_loading": True,
            },
        ]
        responses = [
            {
                "id": "resp_search",
                "output": [
                    {
                        "type": "tool_search_call",
                        "call_id": "call_search",
                        "execution": "client",
                        "status": "completed",
                        "arguments": {"query": "weather tool"},
                    }
                ],
            },
            {
                "id": "resp_function",
                "output": [
                    {
                        "type": "function_call",
                        "name": "get_weather",
                        "call_id": "call_weather",
                        "status": "completed",
                        "arguments": '{"city":"Paris"}',
                    }
                ],
            },
            {"id": "resp_final", "output": [{"type": "message"}]},
        ]
        sent_bodies: list[dict] = []

        def fake_send(_client: object, body: dict, *_args: object, **_kwargs: object) -> dict:
            sent_bodies.append(body)
            return responses[len(sent_bodies) - 1]

        with (
            mock.patch.object(
                record_cassette,
                "_prompt",
                side_effect=["find a weather tool", "call it", "finish"],
            ),
            mock.patch.object(record_cassette, "_send", side_effect=fake_send),
        ):
            record_cassette.run_responses(
                client=object(),
                turns=3,
                model="test-model",
                stream=False,
                store=True,
                branches=[],
                proxy_url="http://unused",
                tools=public_tools,
                tool_outputs={"get_weather": '{"temperature_c":21}'},
                tool_search_output_tools=RETURNED_TOOLS,
            )

        self.assertEqual(sent_bodies[0]["tools"], public_tools)
        self.assertNotIn("tools", sent_bodies[1])
        self.assertNotIn("tools", sent_bodies[2])
        public_output = sent_bodies[1]["input"][0]
        self.assertEqual(public_output["type"], "tool_search_output")
        self.assertEqual(public_output["call_id"], "call_search")
        self.assertEqual(public_output["execution"], "client")
        self.assertEqual(public_output["status"], "completed")
        self.assertEqual(public_output["tools"], RETURNED_TOOLS)
        self.assertEqual(sent_bodies[2]["input"][0]["type"], "function_call_output")
        self.assertTrue(all(body["parallel_tool_calls"] is False for body in sent_bodies))

    def test_gateway_public_manual_replay_is_store_false_and_omits_tools_after_search(self) -> None:
        public_tools = [
            {"type": "tool_search", "execution": "client"},
            {"type": "function", "name": "get_weather", "defer_loading": True},
        ]
        responses = [
            {
                "id": "resp_search",
                "output": [{
                    "type": "tool_search_call",
                    "id": "tsc_search",
                    "call_id": "call_search",
                    "execution": "client",
                    "status": "completed",
                    "arguments": {"query": "weather"},
                }],
            },
            {
                "id": "resp_function",
                "output": [{
                    "type": "function_call",
                    "id": "fc_weather",
                    "name": "get_weather",
                    "call_id": "call_weather",
                    "status": "completed",
                    "arguments": '{"city":"Paris"}',
                }],
            },
            {"id": "resp_final", "output": [{"type": "message"}]},
        ]
        sent_bodies: list[dict] = []

        def fake_send(_client: object, body: dict, *_args: object, **_kwargs: object) -> dict:
            sent_bodies.append(body)
            return responses[len(sent_bodies) - 1]

        with (
            mock.patch.object(record_cassette, "_prompt", side_effect=["find", "call", "finish"]),
            mock.patch.object(record_cassette, "_send", side_effect=fake_send),
        ):
            record_cassette.run_responses(
                client=object(),
                turns=3,
                model="test-model",
                stream=False,
                store=False,
                branches=[],
                proxy_url="http://unused",
                tools=public_tools,
                tool_outputs={"get_weather": "sunny"},
                tool_search_output_tools=RETURNED_TOOLS,
                manual_item_replay=True,
            )

        self.assertTrue(all(body["store"] is False for body in sent_bodies))
        self.assertTrue(all("previous_response_id" not in body for body in sent_bodies))
        self.assertEqual(sent_bodies[0]["tools"], public_tools)
        self.assertNotIn("tools", sent_bodies[1])
        self.assertNotIn("tools", sent_bodies[2])
        self.assertEqual(sent_bodies[1]["input"][1]["type"], "tool_search_call")
        self.assertEqual(sent_bodies[1]["input"][2]["type"], "tool_search_output")
        self.assertEqual(sent_bodies[2]["input"][3]["type"], "message")
        self.assertEqual(sent_bodies[2]["input"][4]["type"], "function_call")
        self.assertEqual(sent_bodies[2]["input"][5]["type"], "function_call_output")

    def test_turn_validation_requires_search_and_object_loaded_arguments(self) -> None:
        valid_public = {
            "type": "tool_search_call",
            "call_id": "call_search",
            "execution": "client",
            "status": "completed",
            "arguments": {"query": "weather tool"},
        }
        valid_normalized = {
            "type": "function_call",
            "name": "tool_search",
            "call_id": "call_search",
            "status": "completed",
            "arguments": '{"query":"weather tool"}',
        }
        valid_loaded = {
            "type": "function_call",
            "name": "get_weather",
            "call_id": "call_weather",
            "status": "completed",
            "arguments": '{"city":"Paris"}',
        }
        record_cassette._validate_tool_search_turn_calls(2, [valid_public], RETURNED_TOOLS)
        record_cassette._validate_tool_search_turn_calls(2, [valid_normalized], RETURNED_TOOLS)
        record_cassette._validate_tool_search_turn_calls(3, [valid_loaded], RETURNED_TOOLS)
        record_cassette._validate_tool_search_turn_calls(
            3,
            [{**valid_loaded, "arguments": '{"city":"London"}'}],
            RETURNED_TOOLS,
        )

        invalid_calls = (
            (2, {key: value for key, value in valid_public.items() if key != "execution"}),
            (2, {**valid_public, "execution": "server"}),
            (2, {**valid_public, "status": "in_progress"}),
            (2, {**valid_public, "arguments": '{"query":"weather tool"}'}),
            (2, {key: value for key, value in valid_normalized.items() if key != "status"}),
            (2, {**valid_normalized, "status": "in_progress"}),
            (2, {**valid_normalized, "arguments": "{}"}),
            (2, {**valid_normalized, "arguments": "not-json"}),
            (3, {key: value for key, value in valid_loaded.items() if key != "status"}),
            (3, {**valid_loaded, "status": "in_progress"}),
            (3, {**valid_loaded, "arguments": "[]"}),
        )
        for turn, call in invalid_calls:
            with self.subTest(turn=turn, call=call), self.assertRaises(ValueError):
                record_cassette._validate_tool_search_turn_calls(turn, [call], RETURNED_TOOLS)

    def test_linear_flow_rejects_a_final_tool_call(self) -> None:
        responses = [
            {
                "id": "resp_search",
                "output": [
                    {
                        "type": "tool_search_call",
                        "call_id": "call_search",
                        "execution": "client",
                        "status": "completed",
                        "arguments": {"query": "weather tool"},
                    }
                ],
            },
            {
                "id": "resp_function",
                "output": [
                    {
                        "type": "function_call",
                        "name": "get_weather",
                        "call_id": "call_weather",
                        "status": "completed",
                        "arguments": '{"city":"Paris"}',
                    }
                ],
            },
            {
                "id": "resp_bad_final",
                "output": [
                    {
                        "type": "function_call",
                        "name": "get_weather",
                        "call_id": "call_again",
                        "arguments": '{"city":"Paris"}',
                    }
                ],
            },
        ]

        with (
            mock.patch.object(record_cassette, "_prompt", side_effect=["search", "call", "finish"]),
            mock.patch.object(record_cassette, "_send", side_effect=responses),
            self.assertRaisesRegex(ValueError, "final response"),
        ):
            record_cassette.run_responses(
                client=object(),
                turns=3,
                model="test-model",
                stream=False,
                store=True,
                branches=[],
                proxy_url="http://unused",
                tools=[{"type": "tool_search"}],
                tool_outputs={"get_weather": "weather"},
                tool_search_output_tools=RETURNED_TOOLS,
            )


if __name__ == "__main__":
    unittest.main()
