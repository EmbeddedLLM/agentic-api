from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, TypeVar

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from typing_extensions import ParamSpec

__all__ = [
    "Session",
    "InvalidSessionUsage",
    "configure_session_factory",
    "reset_session_factory",
    "run_in_session",
    "session_add_one",
    "session_add_all",
    "session_get_one",
    "session_get_all",
    "session_delete",
    "session_transaction",
]

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

_active_session: ContextVar[AsyncSession | None] = ContextVar(
    "active_session", default=None
)

_session_factory: async_sessionmaker[AsyncSession] | None = None
_session_factory_engine: AsyncEngine | None = None


def reset_session_factory() -> None:
    """Clear the module-level session factory. Intended for use in tests only."""
    global _session_factory, _session_factory_engine
    _session_factory = None
    _session_factory_engine = None


def configure_session_factory(engine: AsyncEngine) -> None:
    """Bind the module-level session factory to the given engine. Idempotent for the same engine.

    Raises RuntimeError if called again with a different engine — this indicates a
    programming error (two stores configured against different engines).
    """
    global _session_factory, _session_factory_engine
    if _session_factory is not None:
        if _session_factory_engine is not engine:
            raise RuntimeError(
                "configure_session_factory called with a different engine. "
                "Only one engine is supported per process."
            )
        return
    _session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    _session_factory_engine = engine


def _get_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError(
            "Session factory is not configured. "
            "Call configure_session_factory() at startup."
        )
    return _session_factory


class InvalidSessionUsage(RuntimeError):
    """Raised when a session is used improperly."""


class Session:
    """Async context manager that owns an SQLAlchemy AsyncSession.

    On clean exit:  flush + commit.
    On exception:   rollback, then re-raise.

    Nested entry is not allowed — use run_in_session() or the operation
    decorators, which join the active session transparently.

    Usage::

        async with Session() as session:
            ...  # commits on exit

        @session_add_one
        async def create_item(*, id: str, data: dict) -> Item:
            return Item(id=id, data=data)
    """

    def __init__(self) -> None:
        self._session: AsyncSession | None = None
        self._token = None

    async def __aenter__(self) -> AsyncSession:
        if _active_session.get() is not None:
            raise InvalidSessionUsage(
                "Attempted to open a new session while another is already active. "
                "Use the operation decorators or run_in_session() to join the active session."
            )
        session = _get_factory()()
        self._session = session
        self._token = _active_session.set(session)
        return session

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: Any,
    ) -> None:
        session = self._session
        if session is None:
            return
        try:
            if exc_value is None:
                await session.flush()
                await session.commit()
            else:
                await session.rollback()
        finally:
            await session.close()
            if self._token is not None:
                _active_session.reset(self._token)
            self._session = None
            self._token = None


def get_active_session() -> AsyncSession | None:
    """Return the currently active session for this context, or None."""
    return _active_session.get()


@asynccontextmanager
async def _join_or_open() -> AsyncGenerator[AsyncSession, None]:
    """Join the active session if one exists, otherwise open a new one."""
    existing = _active_session.get()
    if existing is not None:
        yield existing
    else:
        async with Session() as session:
            yield session


def run_in_session(fn: Callable[P, R]) -> Callable[P, R]:
    """Decorate an async function to run inside a session, injecting it as the first argument.

    Joins the active session if one exists, otherwise opens and commits a new one.

    The decorated function must accept `session: AsyncSession` as its first argument::

        @run_in_session
        async def my_func(session: AsyncSession, *, id: str) -> Item:
            result = await session.execute(select(Item).where(Item.id == id))
            return result.scalar_one_or_none()
    """

    @wraps(fn)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        async with _join_or_open() as session:
            return await fn(session, *args, **kwargs)  # type: ignore[arg-type]

    return wrapped  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Operation decorators
# Each decorator wraps an async function and handles exactly one session op.
# The wrapped function must NOT call any session methods itself.
# ---------------------------------------------------------------------------


def session_add_one(fn: Callable[P, T]) -> Callable[P, T]:
    """Add a single ORM instance returned by the function to the session.

    The wrapped function must return the instance to persist::

        @session_add_one
        async def create_item(*, id: str, data: dict) -> Item:
            return Item(id=id, data=data, created_at=_utcnow())
    """

    @wraps(fn)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        async with _join_or_open() as session:
            instance: T = await fn(*args, **kwargs)  # type: ignore[arg-type]
            session.add(instance)
            await session.flush()
            return instance

    return wrapped  # type: ignore[return-value]


def session_add_all(fn: Callable[P, list[T]]) -> Callable[P, list[T]]:
    """Add a list of ORM instances returned by the function to the session.

    The wrapped function must return the list of instances to persist::

        @session_add_all
        async def create_items(items: list[tuple[str, dict]]) -> list[Item]:
            return [Item(id=i, data=d, created_at=_utcnow()) for i, d in items]
    """

    @wraps(fn)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> list[T]:
        async with _join_or_open() as session:
            instances: list[T] = await fn(*args, **kwargs)  # type: ignore[arg-type]
            for instance in instances:
                session.add(instance)
            await session.flush()
            return instances

    return wrapped  # type: ignore[return-value]


def session_get_one(fn: Callable[P, Select[tuple[T]]]) -> Callable[P, T | None]:
    """Execute a SELECT returned by the function and return a single result or None.

    The wrapped function must return a SQLAlchemy Select statement::

        @session_get_one
        async def get_item(*, id: str) -> Select[tuple[Item]]:
            return select(Item).where(Item.id == id)
    """

    @wraps(fn)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T | None:
        async with _join_or_open() as session:
            stmt: Select[tuple[T]] = await fn(*args, **kwargs)  # type: ignore[arg-type]
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    return wrapped  # type: ignore[return-value]


def session_get_all(fn: Callable[P, Select[tuple[T]]]) -> Callable[P, list[T]]:
    """Execute a SELECT returned by the function and return all results as a list.

    The wrapped function must return a SQLAlchemy Select statement::

        @session_get_all
        async def get_items(*, ids: list[str]) -> Select[tuple[Item]]:
            return select(Item).where(Item.id.in_(ids))
    """

    @wraps(fn)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> list[T]:
        async with _join_or_open() as session:
            stmt: Select[tuple[T]] = await fn(*args, **kwargs)  # type: ignore[arg-type]
            result = await session.execute(stmt)
            return list(result.scalars().all())

    return wrapped  # type: ignore[return-value]


def session_delete(fn: Callable[P, Select[tuple[T]]]) -> Callable[P, None]:
    """Execute a SELECT returned by the function, then delete the found row if it exists.

    Silently does nothing if the row is not found.

    The wrapped function must return a SQLAlchemy Select statement::

        @session_delete
        async def delete_item(*, id: str) -> Select[tuple[Item]]:
            return select(Item).where(Item.id == id)
    """

    @wraps(fn)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> None:
        async with _join_or_open() as session:
            stmt: Select[tuple[T]] = await fn(*args, **kwargs)  # type: ignore[arg-type]
            result = await session.execute(stmt)
            instance = result.scalar_one_or_none()
            if instance is not None:
                await session.delete(instance)
                await session.flush()

    return wrapped  # type: ignore[return-value]


def session_transaction(fn: Callable[P, list[Any]]) -> Callable[P, list[Any]]:
    """Run a multi-write function atomically in a single session.

    The wrapped function must return a list of ORM instances to persist.
    All instances are added and flushed in one session — a single commit
    covers the entire list, so either all writes succeed or none do.

    The wrapped function must NOT call any session methods itself::

        @session_transaction
        async def persist_response_checkpoint(
            *,
            item_tuples: list[tuple[str, dict]],
            response_id: str,
            ...
        ) -> list[Base]:
            items = [Item(id=i, data=d, created_at=_utcnow()) for i, d in item_tuples]
            response = Response(id=response_id, ...)
            return [*items, response]
    """

    @wraps(fn)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> list[Any]:
        async with _join_or_open() as session:
            instances: list[Any] = await fn(*args, **kwargs)  # type: ignore[arg-type]
            for instance in instances:
                session.add(instance)
            await session.flush()
            return instances

    return wrapped  # type: ignore[return-value]
