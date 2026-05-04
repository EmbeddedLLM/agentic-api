from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from agentic_api.database.item import Item, get_items
from agentic_api.database.response import Response, get_response
from agentic_api.database.session import configure_session_factory, session_transaction
from agentic_api.store.translator import ItemPayload, StoreInputTranslator
from agentic_api.utils.exceptions import BadInputError, ResponsesAPIError
from agentic_api.types.responses import (
    InputItem,
    OutputItem,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesTool,
    ToolChoice,
)
from agentic_api.utils.common import utcnow, uuid7_str

logger = logging.getLogger(__name__)

_PERSISTABLE_RESPONSE_STATUSES = frozenset({"completed", "incomplete"})


class ResponseMetadata(BaseModel):
    """Effective configuration stored on the Response row for next-turn rehydration."""

    model: str
    previous_response_id: str | None = None
    effective_tools: list[ResponsesTool] | None = None
    effective_tool_choice: ToolChoice
    effective_instructions: str | None = None


@dataclass(frozen=True, slots=True)
class StoredResponse:
    response_id: str
    conversation_id: str | None
    previous_response_id: str | None
    model: str
    created_at: datetime
    history_item_ids: list[str]
    metadata: ResponseMetadata


@session_transaction
async def _persist_response_checkpoint(
    *,
    item_tuples: list[tuple[str, dict[str, Any]]],
    response_id: str,
    previous_response_id: str | None,
    history_item_ids: list[str],
    metadata: dict[str, Any],
) -> list[Any]:
    now = utcnow()
    items = [
        Item(id=item_id, data=data, created_at=now) for item_id, data in item_tuples
    ]
    response = Response(
        id=response_id,
        previous_response_id=previous_response_id,
        history_item_ids=history_item_ids,
        metadata_=metadata,
        created_at=now,
        updated_at=now,
    )
    return [*items, response]


class ResponseStore:
    """Communicator between the Responses API layer and the three-table DB schema.

    get           — loads a Response row and returns a StoredResponse read model.
    put_completed — serialises history items + response checkpoint atomically via
                    @session_transaction (single commit, all-or-nothing).
    rehydrate     — bulk-fetches Item rows by history_item_ids, restores order,
                    and returns the hydrated history for the next agent turn.
    """

    def __init__(self, *, engine: AsyncEngine) -> None:
        self._translator = StoreInputTranslator()
        configure_session_factory(engine)

    async def get(self, *, response_id: str) -> StoredResponse | None:
        response_row: Response | None = await get_response(id=response_id)
        if response_row is None:
            return None

        metadata = ResponseMetadata.model_validate(response_row.metadata_ or {})
        return StoredResponse(
            response_id=response_row.id,
            conversation_id=response_row.conversation_id,
            previous_response_id=response_row.previous_response_id,
            model=metadata.model,
            created_at=response_row.created_at,
            history_item_ids=response_row.history_item_ids or [],
            metadata=metadata,
        )

    async def get_or_raise(self, *, response_id: str) -> StoredResponse:
        stored = await self.get(response_id=response_id)
        if stored is None:
            raise ResponsesAPIError(
                f"No response found with id '{response_id}'.",
                status_code=400,
                param="previous_response_id",
                code="previous_response_not_found",
            )
        return stored

    async def put_completed(
        self,
        *,
        request: ResponsesRequest,
        hydrated_request: ResponsesRequest,
        response: ResponsesResponse,
    ) -> None:
        if response.status not in _PERSISTABLE_RESPONSE_STATUSES:
            return
        if not response.id:
            return
        if not request.response_store_enabled:
            return

        hydrated_input = self._translator.normalize_input(hydrated_request.input)
        history_items: list[InputItem | OutputItem] = [
            *hydrated_input,
            *response.output,
        ]

        item_ids: list[str] = []
        item_tuples: list[tuple[str, dict[str, Any]]] = []
        for hist_item in history_items:
            item_id = uuid7_str("item_")
            item_ids.append(item_id)
            item_tuples.append(
                (item_id, ItemPayload(item=hist_item).model_dump(mode="json"))
            )

        metadata = ResponseMetadata(
            model=response.model,
            previous_response_id=response.previous_response_id,
            effective_tools=hydrated_request.tools,
            effective_tool_choice=hydrated_request.tool_choice,
            effective_instructions=hydrated_request.instructions,
        )

        try:
            await _persist_response_checkpoint(
                item_tuples=item_tuples,
                response_id=response.id,
                previous_response_id=response.previous_response_id,
                history_item_ids=item_ids,
                metadata=metadata.model_dump(mode="json"),
            )
        except IntegrityError as e:
            raise BadInputError(f"Response id already exists: {response.id}") from e

    async def rehydrate(
        self, *, stored: StoredResponse
    ) -> list[InputItem | OutputItem]:
        """Bulk-fetch Item rows by history_item_ids and restore the original order.

        Rehydration model (Responses API path):
        1. Bulk-fetch all Item rows referenced by history_item_ids.
        2. Restore item order in application code.
        """
        items_by_id: dict[str, Item] = {
            item.id: item for item in await get_items(ids=stored.history_item_ids)
        }
        missing = [
            item_id for item_id in stored.history_item_ids if item_id not in items_by_id
        ]
        if missing:
            logger.warning(
                "rehydrate: %d item(s) missing for response %s: %s",
                len(missing),
                stored.id,
                missing,
            )
        return [
            ItemPayload.model_validate(items_by_id[item_id].data).item
            for item_id in stored.history_item_ids
            if item_id in items_by_id
        ]
