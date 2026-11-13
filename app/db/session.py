"""Async SQLAlchemy engine and session factory for PostgreSQL."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_db_engine(database_url: str) -> AsyncEngine:
    """Create the async engine. `pool_pre_ping` discards stale connections before use."""
    return create_async_engine(database_url, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create a session factory. Objects stay usable after commit, so routes can return them."""
    return async_sessionmaker(engine, expire_on_commit=False)