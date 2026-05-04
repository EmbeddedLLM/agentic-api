from __future__ import annotations

from datetime import datetime
from typing import Any

from collections.abc import Callable

from sqlalchemy import DateTime, String, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from agentic_api.database import Base
from agentic_api.utils.common import utcnow
from agentic_api.database.session import (
    run_in_session,
    session_add_one,
    session_delete,
    session_get_all,
    session_get_one,
)


class Conversation(Base):
    """An ordered collection of Items representing a full conversation thread.

    Used by the Conversation API path. History is stored on Item rows via
    `conversation_id` + `seq` — no mutable ID list on this row.

    `metadata_` is an open JSON object for caller-supplied context (e.g. title,
    external IDs, tags). Not interpreted by the store.
    """

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
        default=None,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Responses that belong to this conversation.
    responses: Mapped[list[Any]] = relationship(
        "Response",
        foreign_keys="Response.conversation_id",
        back_populates=None,
        lazy="raise",
        order_by="Response.created_at",
    )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@session_add_one
async def create_conversation(
    *,
    id: str,
    metadata: dict[str, Any] | None = None,
) -> Conversation:
    """Insert a new Conversation row. Raises IntegrityError if the ID already exists."""
    now = utcnow()
    return Conversation(
        id=id,
        metadata_=metadata,
        created_at=now,
        updated_at=now,
    )


@run_in_session
async def create_conversation_if_not_exists(
    session: AsyncSession,
    *,
    id: str,
    insert_fn: Callable,
    metadata: dict[str, Any] | None = None,
) -> Conversation:
    """Insert a Conversation row if it doesn't exist, or return the existing one.

    Uses INSERT ... ON CONFLICT DO NOTHING so the create-if-not-exists is atomic
    at the DB level — safe under concurrent requests with the same conversation_id.

    insert_fn must be a dialect-specific insert (pg_insert or sqlite_insert), chosen
    once at startup from RuntimeConfig.db_dialect and passed in by ConversationStore.
    """
    now = utcnow()
    stmt = (
        insert_fn(Conversation)
        .values(
            id=id,
            metadata_=metadata,
            created_at=now,
            updated_at=now,
        )
        .on_conflict_do_nothing(index_elements=["id"])
    )
    await session.execute(stmt)
    await session.flush()
    result = await session.execute(select(Conversation).where(Conversation.id == id))
    return result.scalar_one()


@session_get_one
async def get_conversation(*, id: str):
    """Fetch a single Conversation by primary key. Returns None if not found."""
    return select(Conversation).where(Conversation.id == id)


@session_get_all
async def get_conversations(*, ids: list[str]):
    """Bulk-fetch Conversations by ID. Returns a list in unspecified order."""
    return select(Conversation).where(Conversation.id.in_(ids))


@session_delete
async def delete_conversation(*, id: str):
    """Delete a Conversation by ID. Silently does nothing if not found.

    Note: Response rows with this conversation_id will have their conversation_id
    set to NULL (ondelete=SET NULL on the FK).
    """
    return select(Conversation).where(Conversation.id == id)
