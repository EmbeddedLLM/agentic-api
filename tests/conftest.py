import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from agentic_api.config.runtime import RuntimeConfig
from agentic_api.core.proxy import ProxyClientManager
from agentic_api.database.session import reset_session_factory
from agentic_api.entrypoints.app import create_app
from tests.utils.replay import (
    CassetteReplayer,
    build_cassette_upstream,
    cassettes_dir,
    make_replayer,
    make_replayer_from_multi_turn,
)


@pytest.fixture(scope="session")
def anyio_backend() -> tuple[str, dict[str, Any]]:
    return "asyncio", {}


def build_test_runtime_config(
    *, llm_api_base: str = "http://upstream", openai_api_key: str | None = "test-key"
) -> RuntimeConfig:
    return RuntimeConfig(
        llm_api_base=llm_api_base,
        openai_api_key=openai_api_key,
        gateway_host="0.0.0.0",
        gateway_port=9000,
        gateway_workers=1,
        upstream_ready_timeout_s=5.0,
        upstream_ready_interval_s=0.1,
        response_store_enabled=False,
    )


def build_upstream_stub() -> FastAPI:
    """Minimal upstream vLLM stub that handles /v1/models and /v1/responses."""
    app = FastAPI(title="Upstream Stub")

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        payload = {
            "object": "list",
            "data": [{"id": "model-a", "object": "model"}],
            "query_a": request.query_params.getlist("a"),
            "authorization": request.headers.get("authorization"),
            "saw_proxy_authorization_header": (
                "proxy-authorization" in request.headers
            ),
        }
        body = json.dumps(payload, separators=(",", ":")).encode()
        return Response(
            status_code=200,
            content=body,
            headers={
                "content-type": "application/json",
                "content-length": str(len(body)),
                "x-upstream": "models",
                "connection": "keep-alive",
            },
        )

    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        body = await request.json()

        if body.get("echo_auth"):
            return Response(
                status_code=200,
                content=json.dumps(
                    {"authorization": request.headers.get("authorization")}
                ).encode(),
                headers={"content-type": "application/json", "x-upstream": "responses"},
            )

        if body.get("force_error") == 429:
            return JSONResponse(
                status_code=429,
                content={"error": {"message": "rate limited", "code": "rate_limit"}},
                headers={"x-upstream": "error"},
            )

        if body.get("stream") is True:

            async def _stream() -> AsyncIterator[bytes]:
                yield b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                _stream(),
                status_code=200,
                headers={
                    "content-type": "text/event-stream; charset=utf-8",
                    "x-upstream": "responses-stream",
                },
            )

        out = b'{"id":"resp_test","object":"response","status":"completed"}'
        return Response(
            status_code=200,
            content=out,
            headers={
                "content-type": "application/json",
                "content-length": str(len(out)),
                "x-upstream": "responses",
                "connection": "keep-alive",
            },
        )

    return app


class _FixedProxyClientManager(ProxyClientManager):
    """ProxyClientManager that always returns a pre-built client."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        super().__init__()
        self._fixed_client = client

    def get_client(self, *, allow_sse_passthrough: bool) -> httpx.AsyncClient:
        return self._fixed_client

    async def aclose(self) -> None:
        return


@pytest.fixture
async def db_engine():
    from agentic_api.database.db_engine import create_db_engine_async
    from agentic_api.database.schema import SchemaManager

    engine = create_db_engine_async(
        db_url="sqlite+aiosqlite:///:memory:",
        db_dialect="sqlite",
    )
    schema = SchemaManager(engine)
    await schema.ensure_ready(gateway_workers=1, db_dialect="sqlite")
    yield engine
    await engine.dispose()
    reset_session_factory()


@pytest.fixture
async def gateway_client() -> AsyncIterator[httpx.AsyncClient]:
    upstream_app = build_upstream_stub()
    upstream_transport = httpx.ASGITransport(app=upstream_app)
    upstream_client = httpx.AsyncClient(
        transport=upstream_transport, base_url="http://upstream"
    )

    runtime_config = build_test_runtime_config(
        llm_api_base="http://upstream",
        openai_api_key="env-upstream-key",
    )
    gateway_app = create_app(runtime_config)

    async with LifespanManager(gateway_app):
        # Lifespan has run — app.state.runtime_config and proxy_client_manager are set.
        # Override the proxy client manager to route requests to the in-process upstream stub.
        gateway_app.state.proxy_client_manager = _FixedProxyClientManager(
            upstream_client
        )

        transport = httpx.ASGITransport(app=gateway_app)
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://gateway"
            ) as client:
                yield client
        finally:
            await upstream_client.aclose()


# ── cassette fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def cassette_replayer_factory() -> Callable[..., CassetteReplayer]:
    """Return a factory: make_replayer(*filenames) → CassetteReplayer."""
    cdir = cassettes_dir()
    return lambda *filenames: make_replayer(*filenames, cassette_dir=cdir)


@pytest.fixture
def conversation_cassette_replayer_factory() -> Callable[[str], CassetteReplayer]:
    """Factory for multi-turn conversation cassettes: factory(filename) → CassetteReplayer."""
    cdir = cassettes_dir() / "text_only" / "conversation"
    return lambda filename: make_replayer_from_multi_turn(filename, cassette_dir=cdir)


@pytest.fixture
def responses_cassette_replayer_factory() -> Callable[[str], CassetteReplayer]:
    """Factory for multi-turn responses cassettes: factory(filename) → CassetteReplayer."""
    cdir = cassettes_dir() / "text_only" / "responses"
    return lambda filename: make_replayer_from_multi_turn(filename, cassette_dir=cdir)


def _build_cassette_gateway(
    replayer_factory: Callable[..., CassetteReplayer],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[httpx.AsyncClient, Callable[..., None]]]:
    """Shared implementation for cassette-backed gateway fixtures."""
    return _cassette_gateway_context(replayer_factory, monkeypatch)


async def _cassette_gateway_context(
    replayer_factory: Callable[..., CassetteReplayer],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[httpx.AsyncClient, Callable[..., None]]]:
    from pydantic_ai.providers.openai import OpenAIProvider
    import agentic_api.core.engine as engine_mod

    replayer_holder: list[CassetteReplayer | None] = [None]
    upstream_app = build_cassette_upstream(replayer_holder)

    upstream_transport = httpx.ASGITransport(app=upstream_app)
    upstream_client = httpx.AsyncClient(
        transport=upstream_transport, base_url="http://upstream"
    )

    def _patched_provider(runtime_config) -> OpenAIProvider:
        base = runtime_config.llm_api_base.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return OpenAIProvider(
            api_key="test-key", base_url=base, http_client=upstream_client
        )

    monkeypatch.setattr(engine_mod, "_build_openai_provider", _patched_provider)

    runtime_config = RuntimeConfig(
        llm_api_base="http://upstream",
        openai_api_key="test-key",
        gateway_host="0.0.0.0",
        gateway_port=9000,
        gateway_workers=1,
        upstream_ready_timeout_s=5.0,
        upstream_ready_interval_s=0.1,
        db_url="sqlite+aiosqlite:///:memory:",
        response_store_enabled=True,
    )
    gateway_app = create_app(runtime_config)

    async with LifespanManager(gateway_app):
        transport = httpx.ASGITransport(app=gateway_app)

        def set_replayer(filename: str) -> None:
            replayer_holder[0] = replayer_factory(filename)

        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://gateway"
            ) as client:
                yield client, set_replayer
        finally:
            await upstream_client.aclose()


@pytest.fixture
async def cassette_gateway_client(
    cassette_replayer_factory: Callable[..., CassetteReplayer],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[httpx.AsyncClient, Callable[..., None]]]:
    """Yield (gateway_client, set_replayer) for cassette-based e2e tests (legacy multi-file API).

    Usage::

        client, use_cassettes = cassette_gateway_client
        use_cassettes("file1.yaml", "file2.yaml")
        resp = await client.post("/v1/responses", ...)
    """

    # Wrap the multi-filename factory into the single-filename interface expected by
    # _cassette_gateway_context, preserving backwards compatibility.
    def _factory(filename: str) -> CassetteReplayer:
        return cassette_replayer_factory(filename)

    async for item in _cassette_gateway_context(_factory, monkeypatch):
        yield item


@pytest.fixture
async def conversation_gateway_client(
    conversation_cassette_replayer_factory: Callable[[str], CassetteReplayer],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[httpx.AsyncClient, Callable[[str], None]]]:
    """Yield (gateway_client, use_cassette) backed by text_only/conversation/ cassettes.

    Usage::

        client, use_cassette = conversation_gateway_client
        use_cassette("conv-two-turn-qwen3-30b-nonstreaming.yaml")
        resp = await client.post("/v1/responses", ...)
    """
    async for item in _cassette_gateway_context(
        conversation_cassette_replayer_factory, monkeypatch
    ):
        yield item


@pytest.fixture
async def responses_gateway_client(
    responses_cassette_replayer_factory: Callable[[str], CassetteReplayer],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[httpx.AsyncClient, Callable[[str], None]]]:
    """Yield (gateway_client, use_cassette) backed by text_only/responses/ cassettes.

    Usage::

        client, use_cassette = responses_gateway_client
        use_cassette("resp-two-turn-qwen3-30b-nonstreaming.yaml")
        resp = await client.post("/v1/responses", ...)
    """
    async for item in _cassette_gateway_context(
        responses_cassette_replayer_factory, monkeypatch
    ):
        yield item
