"""
Interactive multi-turn cassette recorder.

Starts an embedded recording proxy between this script and the upstream API,
then drives multi-turn conversations so every request/response is captured
into a YAML cassette.

Wiring:

  [this script] → [embedded proxy:<proxy-port>] → [OpenAI API | vLLM]
                   (cassette recorded here)

Modes:
  conv        (default) Creates a conversation via POST /v1/conversations, then
              passes conversation id on every turn.
  isolation   Two independent conversations (each with its own conversation id)
              recorded into the same cassette.
  mixed       Creates a conversation; turn 1 uses conversation id, turns 2+
              switch to previous_response_id only (drops conversation id).
  responses   No conversation created. Chains turns purely via
              previous_response_id. Supports --openai and --vllm backends.

Usage:
    python tests/cassettes/record_cassette.py --turns 2 --no-stream --output path/to/cassette.yaml
    python tests/cassettes/record_cassette.py --turns 3 --mode isolation --no-stream --output path/to/cassette.yaml
    python tests/cassettes/record_cassette.py --turns 3 --mode mixed --no-stream --output path/to/cassette.yaml
    python tests/cassettes/record_cassette.py --turns 3 --mode conv --branch-from 1 --branch-turn-number 2 --no-stream --output path/to/cassette.yaml
    python tests/cassettes/record_cassette.py --turns 5 --mode conv --branch-from 1 --branch-turn-number 3 --branch-from 2 --branch-turn-number 5 --no-stream --output path/to/cassette.yaml
    python tests/cassettes/record_cassette.py --turns 2 --mode responses --vllm http://localhost:8000 --model Qwen/Qwen3-30B-A3B-FP8 --no-stream --output path/to/cassette.yaml
"""

import json
import logging
import os
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import click
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import AsyncClient
from yaml import dump as yaml_dump, safe_load as yaml_load

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("cassette_proxy")

MODEL = "gpt-4o"
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7070
TIMEOUT = 60 * 5

EXCLUDED_RESPONSE_HEADERS = {
    "content-encoding",
    "content-length",
    "transfer-encoding",
    "connection",
}

RECORDED_HEADERS = {
    "content-type",
    "authorization",
    "user-agent",
    "accept",
    "x-run-id",
}


def _mask_authorization(value: str) -> str:
    if not value:
        return value
    lower = value.lower()
    if lower.startswith("bearer "):
        return "Bearer ***"
    return "***"


def _filter_request_headers(headers) -> dict:
    return {
        k: v if k.lower() != "authorization" else _mask_authorization(v)
        for k, v in headers.items()
        if k.lower() in RECORDED_HEADERS
    }


def _filter_response_headers(headers) -> dict:
    return {
        k: v for k, v in headers.items() if k.lower() not in EXCLUDED_RESPONSE_HEADERS
    }


def _turn_number(output_file: Path) -> int:
    if not output_file.exists():
        return 1
    content = output_file.read_text(encoding="utf-8")
    if not content.strip():
        return 1
    data = yaml_load(content)
    if not data or "turns" not in data:
        return 1
    return len(data["turns"]) + 1


def _append_turn(output_file: Path, turn: dict[str, Any]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and output_file.stat().st_size > 0:
        data = yaml_load(output_file.read_text(encoding="utf-8")) or {}
    else:
        data = {}
    turns: list = data.get("turns", [])
    turns.append(turn)
    data["turns"] = turns
    with open(output_file, "w", encoding="utf-8") as f:
        yaml_dump(data, f, allow_unicode=True, default_flow_style=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = AsyncClient(timeout=TIMEOUT)
    yield
    await app.state.http_client.aclose()


proxy_app = FastAPI(lifespan=lifespan)


@proxy_app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
)
async def proxy_request(request: Request, path: str) -> Response:
    http_client: AsyncClient = request.app.state.http_client
    target_host: str = request.app.state.target_host
    output_file: Path = request.app.state.output_file

    turn_num = _turn_number(output_file)
    filename = f"t{turn_num}"

    target_url = f"{target_host}/{path}"
    if str(request.query_params):
        target_url += f"?{request.query_params}"

    raw_body = await request.body()
    parsed_body = json.loads(raw_body.decode("utf-8")) if raw_body else {}

    turn: dict[str, Any] = {
        "filename": filename,
        "request": {
            "method": request.method,
            "path": f"/{path}",
            "query_params": dict(request.query_params),
            "body": parsed_body,
            "headers": _filter_request_headers(request.headers),
        },
        "response": {},
    }

    forward_headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    if parsed_body.get("stream", False):

        async def _stream() -> AsyncGenerator[str, None]:
            async with http_client.stream(
                method=request.method,
                url=target_url,
                headers=forward_headers,
                content=raw_body,
                timeout=TIMEOUT,
            ) as response:
                yield response  # type: ignore[misc]
                if response.status_code != 200:
                    chunk_str = (await response.aread()).decode()
                    try:
                        turn["response"]["body"] = json.loads(chunk_str)
                    except Exception:
                        turn["response"]["body"] = chunk_str
                    yield chunk_str
                else:
                    sse_events: list[str] = []
                    try:
                        async for line in response.aiter_lines():
                            chunk = f"{line}\n"
                            yield chunk
                            sse_events.append(chunk)
                    except Exception as e:
                        turn["response"]["stream_error"] = (
                            f"{e.__class__.__name__}: {e}"
                        )
                    finally:
                        turn["response"]["sse"] = sse_events
                turn["response"]["status_code"] = response.status_code
                turn["response"]["headers"] = {
                    "content-type": response.headers.get(
                        "content-type", "text/event-stream"
                    )
                }
                _append_turn(output_file, turn)
                print(f"  [recorded turn {turn_num} -> {output_file.name}]")

        agen = _stream()
        upstream = await anext(agen)
        return StreamingResponse(
            agen,
            status_code=upstream.status_code,
            headers=_filter_response_headers(upstream.headers),
            media_type=upstream.headers.get("content-type", "text/event-stream"),
        )

    else:
        response = await http_client.request(
            method=request.method,
            url=target_url,
            headers=forward_headers,
            content=raw_body,
            timeout=TIMEOUT,
        )
        media_type = response.headers.get("content-type", "application/json")
        body: Any = response.json() if response.status_code == 200 else response.text
        if response.status_code != 200 and "application/json" in media_type:
            try:
                body = json.loads(body)
            except Exception:
                pass
        turn["response"]["body"] = body
        turn["response"]["status_code"] = response.status_code
        turn["response"]["headers"] = {"content-type": media_type}
        _append_turn(output_file, turn)
        print(f"  [recorded turn {turn_num} -> {output_file.name}]")
        return JSONResponse(
            content=body,
            status_code=response.status_code,
            headers=_filter_response_headers(response.headers),
            media_type=media_type,
        )


# ── proxy lifecycle ───────────────────────────────────────────────────────────


def _start_proxy(output_file: Path, target_host: str, port: int) -> uvicorn.Server:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("", encoding="utf-8")
    proxy_app.state.output_file = output_file
    proxy_app.state.target_host = target_host

    config = uvicorn.Config(proxy_app, host=PROXY_HOST, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # TCP-only readiness check — no HTTP request forwarded to upstream
    for _ in range(40):
        try:
            with socket.create_connection((PROXY_HOST, port), timeout=0.3):
                break
        except OSError:
            time.sleep(0.3)

    return server


def _stop_proxy(server: uvicorn.Server) -> None:
    server.should_exit = True
    time.sleep(0.5)


def _create_conversation(client: httpx.Client, proxy_url: str) -> str:
    resp = client.post(f"{proxy_url}/v1/conversations", json={}, timeout=30)
    resp.raise_for_status()
    conv_id = resp.json().get("id")
    print(f"[conversation created: {conv_id}]")
    return conv_id


def _send_nonstreaming(client: httpx.Client, body: dict, proxy_url: str) -> str | None:
    resp = client.post(f"{proxy_url}/v1/responses", json=body, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    print(f"\n[Response]\n{json.dumps(data, indent=2)}\n")
    return data.get("id")


def _send_streaming(client: httpx.Client, body: dict, proxy_url: str) -> str | None:
    response_id = None
    print("\n[Streaming response]")
    with client.stream(
        "POST", f"{proxy_url}/v1/responses", json=body, timeout=300
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            print(line)
            if line.startswith("data:") and line != "data: [DONE]":
                try:
                    payload = json.loads(line[5:].strip())
                    if payload.get("type") == "response.completed":
                        response_id = payload.get("response", {}).get("id")
                except Exception:
                    pass
    print()
    return response_id


def _send(client: httpx.Client, body: dict, stream: bool, proxy_url: str) -> str | None:
    return (
        _send_streaming(client, body, proxy_url)
        if stream
        else _send_nonstreaming(client, body, proxy_url)
    )


def _prompt(label: str) -> str:
    try:
        return input(label).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)


def run_conv(
    client: httpx.Client,
    turns: int,
    model: str,
    stream: bool,
    store: bool,
    branches: list[tuple[int, int | None]],
    proxy_url: str,
) -> None:
    conv_id = _create_conversation(client, proxy_url)
    response_ids: dict[int, str] = {}
    # map: branch_turn_number -> branch_from (which turn's response to use as previous)
    branch_map: dict[int, int] = {}
    extra_branches: list[int] = []  # branch_from values with no branch_turn_number
    for branch_from, branch_turn_number in branches:
        if branch_turn_number is not None:
            branch_map[branch_turn_number] = branch_from
        else:
            extra_branches.append(branch_from)

    previous_response_id: str | None = None
    for turn in range(1, turns + 1):
        if turn in branch_map:
            branch_from = branch_map[turn]
            if branch_from not in response_ids:
                raise click.UsageError(
                    f"--branch-from {branch_from} at turn {turn} has no recorded response "
                    f"(available: {sorted(response_ids)})"
                )
            previous_response_id = response_ids[branch_from]
            click.echo(
                f"\n[Branch] turn {turn} chains from turn {branch_from} (response_id={previous_response_id})"
            )
        prompt = _prompt(f"Turn {turn}/{turns} — enter prompt: ")
        body: dict = {"model": model, "input": prompt, "stream": stream, "store": store}
        if previous_response_id:
            body["previous_response_id"] = previous_response_id
        else:
            body["conversation"] = conv_id
        response_id = _send(client, body, stream, proxy_url)
        if response_id:
            response_ids[turn] = response_id
            previous_response_id = response_id

    # branches without a branch_turn_number get one extra turn each
    for b_idx, branch_from in enumerate(extra_branches, start=1):
        if branch_from not in response_ids:
            raise click.UsageError(
                f"Extra branch {b_idx}: --branch-from {branch_from} has no recorded response "
                f"(available: {sorted(response_ids)})"
            )
        branch_resp_id = response_ids[branch_from]
        click.echo(
            f"\n[Extra branch {b_idx}] from turn {branch_from} (response_id={branch_resp_id}), turn {turns + 1}"
        )
        prompt = _prompt(
            f"Turn {turns + 1} (extra branch from turn {branch_from}) — enter prompt: "
        )
        body = {
            "model": model,
            "input": prompt,
            "stream": stream,
            "store": store,
            "previous_response_id": branch_resp_id,
            "conversation": conv_id,
        }
        _send(client, body, stream, proxy_url)


def run_isolation(
    client: httpx.Client,
    turns: int,
    model: str,
    stream: bool,
    store: bool,
    proxy_url: str,
) -> None:
    for conv_label in ("A", "B"):
        click.echo(f"\n--- Conversation {conv_label} ({turns} turns) ---")
        conv_id = _create_conversation(client, proxy_url)
        for turn in range(1, turns + 1):
            prompt = _prompt(
                f"Conv {conv_label} | Turn {turn}/{turns} — enter prompt: "
            )
            body: dict = {
                "model": model,
                "input": prompt,
                "stream": stream,
                "store": store,
                "conversation": conv_id,
            }
            _send(client, body, stream, proxy_url)


def run_mixed(
    client: httpx.Client,
    turns: int,
    model: str,
    stream: bool,
    store: bool,
    proxy_url: str,
) -> None:
    conv_id = _create_conversation(client, proxy_url)
    previous_response_id: str | None = None

    for turn in range(1, turns + 1):
        prompt = _prompt(f"Turn {turn}/{turns} — enter prompt: ")
        body: dict = {"model": model, "input": prompt, "stream": stream, "store": store}
        if previous_response_id:
            body["previous_response_id"] = previous_response_id
        else:
            body["conversation"] = conv_id
        previous_response_id = _send(client, body, stream, proxy_url)


def run_responses(
    client: httpx.Client,
    turns: int,
    model: str,
    stream: bool,
    store: bool,
    branches: list[tuple[int, int | None]],
    proxy_url: str,
) -> None:
    response_ids: dict[int, str] = {}
    branch_map: dict[int, int] = {}
    extra_branches: list[int] = []
    for branch_from, branch_turn_number in branches:
        if branch_turn_number is not None:
            branch_map[branch_turn_number] = branch_from
        else:
            extra_branches.append(branch_from)

    previous_response_id: str | None = None
    for turn in range(1, turns + 1):
        if turn in branch_map:
            branch_from = branch_map[turn]
            if branch_from not in response_ids:
                raise click.UsageError(
                    f"--branch-from {branch_from} at turn {turn} has no recorded response "
                    f"(available: {sorted(response_ids)})"
                )
            previous_response_id = response_ids[branch_from]
            click.echo(
                f"\n[Branch] turn {turn} chains from turn {branch_from} (response_id={previous_response_id})"
            )
        prompt = _prompt(f"Turn {turn}/{turns} — enter prompt: ")
        body: dict = {"model": model, "input": prompt, "stream": stream, "store": store}
        if previous_response_id and store:
            body["previous_response_id"] = previous_response_id
        response_id = _send(client, body, stream, proxy_url)
        previous_response_id = response_id if store else None
        if response_id:
            response_ids[turn] = response_id

    for b_idx, branch_from in enumerate(extra_branches, start=1):
        if branch_from not in response_ids:
            raise click.UsageError(
                f"Extra branch {b_idx}: --branch-from {branch_from} has no recorded response "
                f"(available: {sorted(response_ids)})"
            )
        branch_resp_id = response_ids[branch_from]
        click.echo(
            f"\n[Extra branch {b_idx}] from turn {branch_from} (response_id={branch_resp_id}), turn {turns + 1}"
        )
        prompt = _prompt(
            f"Turn {turns + 1} (extra branch from turn {branch_from}) — enter prompt: "
        )
        body = {
            "model": model,
            "input": prompt,
            "stream": stream,
            "store": store,
            "previous_response_id": branch_resp_id,
        }
        _send(client, body, stream, proxy_url)


# ── main ──────────────────────────────────────────────────────────────────────


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--turns", "-n", required=True, type=int, help="Number of turns to record."
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output cassette YAML path.",
)
@click.option(
    "--mode",
    type=click.Choice(["conv", "isolation", "mixed", "responses"]),
    default="conv",
    show_default=True,
    help="Recording mode.",
)
@click.option(
    "--branch-from",
    type=int,
    multiple=True,
    metavar="TURN",
    help="Rewind to this turn's response (repeatable, one per branch).",
)
@click.option(
    "--branch-turn-number",
    type=int,
    multiple=True,
    metavar="TURN",
    help="First turn number for the corresponding branch (repeatable, pairs with --branch-from).",
)
@click.option(
    "--stream/--no-stream",
    default=True,
    show_default=True,
    help="Use streaming responses.",
)
@click.option(
    "--model", default=MODEL, show_default=True, help="Model name to pass in requests."
)
@click.option(
    "--no-store", is_flag=True, default=False, help="Set store=false in requests."
)
@click.option(
    "--proxy-port",
    type=int,
    default=PROXY_PORT,
    show_default=True,
    help="Local port for the embedded recording proxy.",
)
@click.option(
    "--openai",
    "openai_url",
    metavar="URL",
    default=None,
    help="OpenAI upstream URL (default https://api.openai.com). Reads OPENAI_API_KEY.",
)
@click.option(
    "--vllm",
    "vllm_url",
    metavar="URL",
    default=None,
    help="vLLM upstream URL, e.g. http://localhost:8000 (responses mode only, no auth).",
)
def main(
    turns: int,
    output: str,
    mode: str,
    branch_from: tuple[int, ...],
    branch_turn_number: tuple[int, ...],
    stream: bool,
    model: str,
    no_store: bool,
    proxy_port: int,
    openai_url: str | None,
    vllm_url: str | None,
) -> None:
    """Interactive multi-turn cassette recorder (proxy embedded)."""
    if branch_turn_number and not branch_from:
        raise click.UsageError("--branch-turn-number requires --branch-from.")
    if len(branch_turn_number) > len(branch_from):
        raise click.UsageError(
            "More --branch-turn-number values than --branch-from values."
        )
    # Pair each branch-from with its branch-turn-number (None if not provided)
    branches: list[tuple[int, int | None]] = [
        (bf, branch_turn_number[i] if i < len(branch_turn_number) else None)
        for i, bf in enumerate(branch_from)
    ]
    if vllm_url and openai_url:
        raise click.UsageError("--openai and --vllm are mutually exclusive.")
    if vllm_url and mode != "responses":
        raise click.UsageError(
            f"--vllm is only supported with --mode responses (got --mode {mode})."
        )

    if vllm_url:
        target = vllm_url.rstrip("/")
        headers: dict = {}
        backend_label = f"vLLM:   {target}"
    else:
        target = (openai_url or "https://api.openai.com").rstrip("/")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise click.ClickException(
                "OPENAI_API_KEY environment variable is not set."
            )
        headers = {"Authorization": f"Bearer {api_key}"}
        backend_label = f"OpenAI: {target}"

    output_file = Path(output).resolve()
    proxy_url = f"http://{PROXY_HOST}:{proxy_port}"
    store = not no_store

    click.echo(f"Mode: {mode} | Turns: {turns} | Stream: {stream} | Model: {model}")
    click.echo(f"Output:  {output_file}")
    click.echo(backend_label)
    click.echo(f"Proxy:   {proxy_url}  (requests go through here for recording)")

    server = _start_proxy(output_file, target, proxy_port)
    click.echo(f"Proxy ready on {proxy_url}\n")

    try:
        with httpx.Client(headers=headers) as client:
            if mode == "conv":
                run_conv(client, turns, model, stream, store, branches, proxy_url)
            elif mode == "isolation":
                run_isolation(client, turns, model, stream, store, proxy_url)
            elif mode == "mixed":
                run_mixed(client, turns, model, stream, store, proxy_url)
            elif mode == "responses":
                run_responses(client, turns, model, stream, store, branches, proxy_url)
    finally:
        _stop_proxy(server)

    click.echo(f"\nAll turns recorded -> {output_file}")


if __name__ == "__main__":
    main()
