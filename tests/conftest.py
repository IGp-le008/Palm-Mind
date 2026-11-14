"""Shared fixtures: a Postgres test database with per-test rollback."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, create_async_engine

from app.core.config import settings
from app.db import models  # noqa: F401  # imports register the tables on Base.metadata
from app.db.base import Base


@pytest_asyncio.fixture(scope="session")
async def engine() -> AsyncIterator[AsyncEngine]:
    """Create all tables once per test run, and drop them at the end."""
    if not settings.database_url.endswith("_test"):
        pytest.exit("DATABASE_URL must point to a database whose name ends in _test")

    engine = create_async_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Give each test a session. Everything it writes is rolled back afterwards."""
    connection: AsyncConnection = await engine.connect()
    transaction = await connection.begin()
    session = AsyncSession(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    yield session
    await session.close()
    await transaction.rollback()
    await connection.close()