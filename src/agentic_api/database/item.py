from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from agentic_api.database import Base
from agentic_api.utils.common import utcnow
from agentic_api.database.session import (
    session_add_all,
    session_add_one,
    session_delete,
    session_get_all,
    session_get_one,
)


class Item(Base):
    """An immutable payload unit — one user message, one assistant output item, one tool result.

    `data` stores a versioned JSON blob (`ItemPayload`) so the schema can evolve
    without altering the table. Items are never mutated after creation.

    `conversation_id` and `seq` are set for conversation-path items so history can be
    reconstructed with ORDER BY seq instead of a mutable ID list on the Conversation row.
    Standalone (non-conversation) items leave both NULL.
    """

    __tablename__ = "items"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    data: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    conversation_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        default=None,
    )
    seq: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        default=None,
    )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@session_add_one
async def create_item(*, id: str, data: dict[str, Any]) -> Item:
    """Insert a new Item row. Raises IntegrityError if the ID already exists."""
    return Item(id=id, data=data, created_at=utcnow())


@session_add_all
async def create_items(items: list[tuple[str, dict[str, Any]]]) -> list[Item]:
    """Bulk-insert Item rows. Raises IntegrityError if any ID already exists."""
    now = utcnow()
    return [Item(id=item_id, data=data, created_at=now) for item_id, data in items]


@session_get_one
async def get_item(*, id: str):
    """Fetch a single Item by primary key. Returns None if not found."""
    return select(Item).where(Item.id == id)


@session_get_all
async def get_items(*, ids: list[str]):
    """Bulk-fetch Items by ID. Returns a list in unspecified order."""
    return select(Item).where(Item.id.in_(ids))


@session_get_all
async def get_items_by_conversation(*, conversation_id: str):
    """Fetch all Items belonging to a conversation, ordered by seq."""
    return (
        select(Item).where(Item.conversation_id == conversation_id).order_by(Item.seq)
    )


@session_delete
async def delete_item(*, id: str):
    """Delete an Item by ID. Silently does nothing if not found."""
    return select(Item).where(Item.id == id)
