"""
DB engine plumbing.

Owns engine creation (dialect-specific connect args, pooling), SQLite PRAGMA
tuning, and the Postgres advisory lock helper used during schema init.
"""

import asyncio
import hashlib
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import logging

from sqlalchemy import NullPool, TextClause, event, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)

SQLITE_BUSY_TIMEOUT_MS = 5_000


def _apply_sqlite_pragmas(dbapi_connection) -> None:  # type: ignore[no-untyped-def]
    """Apply SQLite performance PRAGMAs on every new connection."""
    raw = getattr(dbapi_connection, "driver_connection", None) or dbapi_connection
    raw = getattr(raw, "_conn", raw)
    try:
        raw.execute("PRAGMA journal_mode=WAL;")
        raw.execute("PRAGMA synchronous=NORMAL;")
        raw.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS};")
        raw.execute("PRAGMA foreign_keys=ON;")
    except Exception as e:
        logger.warning(f"Failed to apply SQLite PRAGMAs: {e!r}")


def create_db_engine_async(*, db_url: str, db_dialect: str) -> AsyncEngine:
    """Create and return an async SQLAlchemy engine for the given URL and dialect."""
    kwargs: dict = {}

    if db_dialect == "sqlite":
        logger.debug("Using SQLite DB.")
        connect_args: dict = {"check_same_thread": False}
    elif db_dialect == "postgresql":
        logger.debug("Using PostgreSQL DB.")
        connect_args = {}
        kwargs["poolclass"] = NullPool
    else:
        raise ValueError(f'DB dialect "{db_dialect}" is not supported.')

    engine = create_async_engine(db_url, connect_args=connect_args, **kwargs)

    if db_dialect == "sqlite":

        @event.listens_for(engine.sync_engine, "connect")
        def _on_connect(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
            _apply_sqlite_pragmas(dbapi_connection)

    return engine


# ---------------------------------------------------------------------------
# Postgres advisory lock
# ---------------------------------------------------------------------------

_cached_text: dict[str, TextClause] = {}


def _cached_text_clause(query: str) -> TextClause:
    if query not in _cached_text:
        _cached_text[query] = text(query)
    return _cached_text[query]


def postgres_advisory_lock_key(name: str) -> int:
    """Convert a stable string name into an int64 Postgres advisory lock key."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


@asynccontextmanager
async def postgres_advisory_lock(
    conn: AsyncConnection,
    *,
    name: str,
    timeout_s: float = 60.0,
    poll_interval_s: float = 0.1,
) -> AsyncGenerator[None, None]:
    """Acquire a Postgres advisory lock for the current session and release on exit.

    Intended for one-time init tasks (schema creation, migrations) where multiple
    gateway instances may race at startup.
    """
    key = postgres_advisory_lock_key(name)
    deadline = time.perf_counter() + timeout_s
    locked = False
    try:
        while True:
            result = await conn.execute(
                _cached_text_clause("SELECT pg_try_advisory_lock(:k)"), {"k": key}
            )
            locked = bool(result.scalar_one())
            if locked:
                break
            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for Postgres advisory lock. "
                    "Another instance may be stuck performing one-time initialization."
                )
            await asyncio.sleep(poll_interval_s)
        yield
    finally:
        if locked:
            try:
                await conn.execute(
                    _cached_text_clause("SELECT pg_advisory_unlock(:k)"), {"k": key}
                )
            except Exception:
                logger.warning(
                    "Failed to release PostgreSQL advisory lock (key=%d); lock may have leaked.",
                    key,
                    exc_info=True,
                )
