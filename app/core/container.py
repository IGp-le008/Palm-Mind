"""Dependency wiring.

Builds shared infrastructure clients once at startup and closes them on shutdown.
Services receive these objects (or the abstractions built on them) rather than creating
their own connections. The vector store and model providers are added here when their
modules are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings


@dataclass(frozen=True)
class Container:
    """Shared infrastructure clients for the lifetime of the application."""

    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    redis: Redis

    @classmethod
    def create(cls, config: Settings) -> Container:
        """Validate configuration and construct all clients. Does not open connections yet."""
        config.validate_for_storage()

        engine = create_async_engine(config.database_url, pool_pre_ping=True)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        redis = Redis.from_url(config.redis_url, decode_responses=True)

        return cls(
            settings=config,
            engine=engine,
            session_factory=session_factory,
            redis=redis,
        )

    async def close(self) -> None:
        """Release connections. Call once on application shutdown."""
        await self.redis.aclose()
        await self.engine.dispose()
