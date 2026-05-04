from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from agentic_api.database.conversation import (
    create_conversation,
    create_conversation_if_not_exists,
    get_conversation,
)
from agentic_api.database.item import Item, get_items_by_conversation
from agentic_api.database.response import Response
from agentic_api.database.session import configure_session_factory, session_transaction
from agentic_api.store.response import ResponseMetadata
from agentic_api.store.translator import ItemPayload
from agentic_api.types.responses import InputItem, OutputItem
from agentic_api.utils.common import utcnow, uuid7_str
from agentic_api.utils.exceptions import BadInputError, ResponsesAPIError


@dataclass(frozen=True, slots=True)
class StoredConversation:
    conversation_id: str
    created_at: datetime
    metadata: ResponseMetadata | None = None


@session_transaction
async def _persist_conversation_turn(
    *,
    item_tuples: list[tuple[str, dict[str, Any]]],
    conversation_id: str,
    response_id: str,
    previous_response_id: str | None,
    seq_start: int,
    metadata: dict[str, Any],
) -> list[Item | Response]:
    """Atomically write new Item rows and a Response checkpoint.

    Items are written with conversation_id + seq so history is reconstructed via
    ORDER BY seq — no mutable ID list on the Conversation row.
    """
    now = utcnow()
    items = [
        Item(
            id=item_id,
            data=data,
            created_at=now,
            conversation_id=conversation_id,
            seq=seq_start + i,
        )
        for i, (item_id, data) in enumerate(item_tuples)
    ]
    item_ids = [item_id for item_id, _ in item_tuples]
    response = Response(
        id=response_id,
        conversation_id=conversation_id,
        previous_response_id=previous_response_id,
        history_item_ids=item_ids,
        metadata_=metadata,
        created_at=now,
        updated_at=now,
    )
    return [*items, response]


class ConversationStore:
    """Read/write interface between the Conversation API layer and the three-table DB schema.

    create        — inserts a new Conversation row with a server-generated ID.
    get_or_create — load an existing Conversation by ID, or create a new one if not found.
    get           — loads a Conversation row and returns a StoredConversation read model.
    put_turn      — atomically writes new Item rows (with conversation_id + seq) and a
                    Response checkpoint — no read-modify-write on the Conversation row.
    rehydrate     — queries Items by conversation_id ORDER BY seq and returns ordered history.
    """

    def __init__(self, *, engine: AsyncEngine, db_dialect: str = "sqlite") -> None:
        configure_session_factory(engine)
        self._insert_fn = pg_insert if db_dialect == "postgresql" else sqlite_insert

    async def create(self) -> StoredConversation:
        """Create a new Conversation with a server-generated ID."""
        row = await create_conversation(id=uuid7_str("conv_"))
        return StoredConversation(conversation_id=row.id, created_at=row.created_at)

    async def get_or_create(self, *, conversation_id: str) -> StoredConversation:
        """Return an existing Conversation by ID, or create a new one if not found.

        Uses an atomic INSERT ... ON CONFLICT DO NOTHING so concurrent requests with
        the same conversation_id are safe — the DB ensures exactly one row is created.
        """
        row = await create_conversation_if_not_exists(
            id=conversation_id, insert_fn=self._insert_fn
        )
        return StoredConversation(conversation_id=row.id, created_at=row.created_at)

    async def get(self, *, conversation_id: str) -> StoredConversation | None:
        row = await get_conversation(id=conversation_id)
        if row is None:
            return None
        return StoredConversation(
            conversation_id=row.id,
            created_at=row.created_at,
            metadata=ResponseMetadata.model_validate(row.metadata_)
            if row.metadata_
            else None,
        )

    async def put_turn(
        self,
        *,
        conversation_id: str,
        response_id: str,
        previous_response_id: str | None,
        new_items: list[InputItem | OutputItem],
        metadata_: dict[str, Any],
    ) -> StoredConversation:
        """Persist a new conversation turn atomically.

        Within a single Session commit:
        1. Bulk-insert new Item rows with conversation_id + seq.
        2. Insert Response checkpoint.

        seq is assigned as (current item count) + offset so items from concurrent
        turns don't collide — each turn's items are appended after existing ones.

        Raises BadInputError if conversation_id does not exist or response_id already exists.
        """
        stored = await self.get(conversation_id=conversation_id)
        if stored is None:
            raise BadInputError(f"Conversation not found: {conversation_id}")

        existing_items = await get_items_by_conversation(
            conversation_id=conversation_id
        )
        seq_start = len(existing_items)

        item_tuples: list[tuple[str, dict[str, Any]]] = [
            (uuid7_str("item_"), ItemPayload(item=item).model_dump(mode="json"))
            for item in new_items
        ]

        try:
            await _persist_conversation_turn(
                item_tuples=item_tuples,
                conversation_id=conversation_id,
                response_id=response_id,
                previous_response_id=previous_response_id,
                seq_start=seq_start,
                metadata=metadata_,
            )
        except IntegrityError as e:
            raise BadInputError(f"Response id already exists: {response_id}") from e

        return stored

    async def rehydrate(self, *, conversation_id: str) -> list[InputItem | OutputItem]:
        """Return the full ordered history for a conversation."""
        stored = await self.get(conversation_id=conversation_id)
        if stored is None:
            raise ResponsesAPIError(
                f"Conversation '{conversation_id}' not found.",
                status_code=400,
                param="conversation_id",
                code="conversation_not_found",
            )

        items = await get_items_by_conversation(conversation_id=conversation_id)
        return [ItemPayload.model_validate(item.data).item for item in items]
