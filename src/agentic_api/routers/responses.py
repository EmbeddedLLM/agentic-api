import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

from agentic_api.core.engine import Engine
from agentic_api.core.proxy import ProxyClientManager, proxy_responses
from agentic_api.utils.exceptions import BadInputError, ResponsesAPIError
from agentic_api.store.conversation import ConversationStore
from agentic_api.store.response import ResponseStore
from agentic_api.types.responses import ResponsesRequest

router = APIRouter()


@router.post("/v1/responses")
async def create_response(request: Request) -> Response:
    runtime_config = request.app.state.runtime_config
    response_store: ResponseStore | None = request.app.state.response_store
    conversation_store: ConversationStore | None = request.app.state.conversation_store

    # TODO: let the engine run in stateless mode (no response store); port proxy
    # passthrough into a dedicated proxy router and remove this branch.
    if response_store is None:
        proxy_client_manager: ProxyClientManager = (
            request.app.state.proxy_client_manager
        )
        return await proxy_responses(
            request=request,
            runtime_config=runtime_config,
            proxy_client_manager=proxy_client_manager,
        )

    try:
        body = await request.json()
        responses_request = ResponsesRequest.model_validate(body)
    except (ValueError, ValidationError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    engine = Engine(
        responses_request,
        response_store=response_store,
        conversation_store=conversation_store
        if responses_request.conversation_store_enabled
        else None,
        runtime_config=runtime_config,
    )

    try:
        result = await engine.run()
    except ResponsesAPIError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": str(exc),
                    "type": exc.error_type,
                    "param": exc.param,
                    "code": exc.code,
                }
            },
        )
    except BadInputError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(exc), "type": "invalid_request_error"}},
        )

    # Streaming: result is an async generator of SSE frames.
    if responses_request.stream:
        return StreamingResponse(
            result,  # type: ignore[arg-type]
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    # Non-streaming: result is a ResponsesResponse.
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),  # type: ignore[union-attr]
    )
