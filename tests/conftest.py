"""Shared fixtures. Each test gets fresh tables in the Postgres test database."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401  # registers the tables on Base.metadata
from app.core.config import settings
from app.db.base import Base


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Create tables, yield a session, then drop the tables."""
    if not settings.database_url.endswith("_test"):
        raise RuntimeError("Tests must run against a database whose name ends in _test")

    engine = create_async_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()