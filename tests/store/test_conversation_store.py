from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from agentic_api.store.conversation import ConversationStore
from agentic_api.types.responses import InputMessage, OutputMessage
from agentic_api.utils.exceptions import BadInputError, ResponsesAPIError

from tests.utils import make_assistant_msg as _assistant_msg
from tests.utils import make_response_metadata as _metadata
from tests.utils import make_user_msg as _user_msg


@pytest.fixture
async def store(db_engine: AsyncEngine) -> ConversationStore:
    return ConversationStore(engine=db_engine)


@pytest.mark.anyio
async def test_get_returns_none_for_missing(store: ConversationStore) -> None:
    result = await store.get(conversation_id="conv_does_not_exist")
    assert result is None


@pytest.mark.anyio
async def test_get_or_create_creates_new_conversation(store: ConversationStore) -> None:
    stored = await store.get_or_create(conversation_id="conv_001")
    assert stored.conversation_id == "conv_001"
    assert await store.rehydrate(conversation_id="conv_001") == []


@pytest.mark.anyio
async def test_get_or_create_returns_existing(store: ConversationStore) -> None:
    await store.get_or_create(conversation_id="conv_002")
    # Second call with the same id should return the existing row, not create a new one.
    stored = await store.get_or_create(conversation_id="conv_002")
    assert stored.conversation_id == "conv_002"


@pytest.mark.anyio
async def test_put_turn_appends_items(store: ConversationStore) -> None:
    await store.get_or_create(conversation_id="conv_pt1")

    result = await store.put_turn(
        conversation_id="conv_pt1",
        response_id="resp_pt1",
        previous_response_id=None,
        new_items=[_user_msg("hello"), _assistant_msg("hi")],
        metadata_=_metadata(),
    )

    assert result.conversation_id == "conv_pt1"
    assert len(await store.rehydrate(conversation_id="conv_pt1")) == 2


@pytest.mark.anyio
async def test_put_turn_accumulates_across_turns(store: ConversationStore) -> None:
    await store.get_or_create(conversation_id="conv_acc")

    await store.put_turn(
        conversation_id="conv_acc",
        response_id="resp_acc1",
        previous_response_id=None,
        new_items=[_user_msg("turn 1"), _assistant_msg("answer 1")],
        metadata_=_metadata(),
    )
    await store.put_turn(
        conversation_id="conv_acc",
        response_id="resp_acc2",
        previous_response_id="resp_acc1",
        new_items=[_user_msg("turn 2"), _assistant_msg("answer 2")],
        metadata_=_metadata(),
    )

    assert len(await store.rehydrate(conversation_id="conv_acc")) == 4


@pytest.mark.anyio
async def test_put_turn_raises_for_missing_conversation(
    store: ConversationStore,
) -> None:
    with pytest.raises(BadInputError):
        await store.put_turn(
            conversation_id="conv_ghost",
            response_id="resp_x",
            previous_response_id=None,
            new_items=[_user_msg("hi")],
            metadata_=_metadata(),
        )


@pytest.mark.anyio
async def test_put_turn_raises_for_duplicate_response_id(
    store: ConversationStore,
) -> None:
    await store.get_or_create(conversation_id="conv_dup")
    await store.put_turn(
        conversation_id="conv_dup",
        response_id="resp_dup",
        previous_response_id=None,
        new_items=[_user_msg("hi")],
        metadata_=_metadata(),
    )

    with pytest.raises(BadInputError):
        await store.put_turn(
            conversation_id="conv_dup",
            response_id="resp_dup",
            previous_response_id=None,
            new_items=[_user_msg("hi again")],
            metadata_=_metadata(),
        )


@pytest.mark.anyio
async def test_rehydrate_raises_for_missing_conversation(
    store: ConversationStore,
) -> None:
    with pytest.raises(ResponsesAPIError) as exc_info:
        await store.rehydrate(conversation_id="conv_ghost")
    assert exc_info.value.code == "conversation_not_found"


@pytest.mark.anyio
async def test_rehydrate_empty_conversation(store: ConversationStore) -> None:
    await store.get_or_create(conversation_id="conv_empty")
    items = await store.rehydrate(conversation_id="conv_empty")
    assert items == []


@pytest.mark.anyio
async def test_rehydrate_restores_items_in_order(store: ConversationStore) -> None:
    await store.get_or_create(conversation_id="conv_order")
    await store.put_turn(
        conversation_id="conv_order",
        response_id="resp_order1",
        previous_response_id=None,
        new_items=[_user_msg("first"), _assistant_msg("reply")],
        metadata_=_metadata(),
    )

    items = await store.rehydrate(conversation_id="conv_order")
    assert len(items) == 2
    assert isinstance(items[0], InputMessage)
    assert items[0].role == "user"
    assert isinstance(items[1], OutputMessage)
    assert items[1].content[0].text == "reply"


@pytest.mark.anyio
async def test_rehydrate_multi_turn_order(store: ConversationStore) -> None:
    await store.get_or_create(conversation_id="conv_multi")

    await store.put_turn(
        conversation_id="conv_multi",
        response_id="resp_m1",
        previous_response_id=None,
        new_items=[_user_msg("turn 1"), _assistant_msg("answer 1")],
        metadata_=_metadata(),
    )
    await store.put_turn(
        conversation_id="conv_multi",
        response_id="resp_m2",
        previous_response_id="resp_m1",
        new_items=[_user_msg("turn 2"), _assistant_msg("answer 2")],
        metadata_=_metadata(),
    )

    items = await store.rehydrate(conversation_id="conv_multi")
    assert len(items) == 4
    roles = [getattr(i, "role", None) for i in items]
    assert roles == ["user", "assistant", "user", "assistant"]
