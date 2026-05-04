from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import Protocol


class _ResponsesChunk(Protocol):
    type: str

    def as_responses_chunk(self) -> str: ...


DONE_MARKER = "data: [DONE]\n\n"
TERMINAL_EVENT_TYPES = {"response.completed", "response.failed", "response.incomplete"}


async def stream_responses_sse(
    events: AsyncIterable[_ResponsesChunk],
) -> AsyncIterator[str]:
    """Encode typed Responses stream events into SSE frames, including the spec terminal marker."""
    done_emitted = False
    async for event in events:
        yield event.as_responses_chunk()
        if not done_emitted and event.type in TERMINAL_EVENT_TYPES:
            yield DONE_MARKER
            done_emitted = True

    # Defensive: if upstream ended without an explicit terminal lifecycle event, still close
    # the SSE stream with the spec marker to avoid hanging clients.
    if not done_emitted:
        yield DONE_MARKER
