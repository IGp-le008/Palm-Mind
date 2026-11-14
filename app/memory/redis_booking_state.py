"""Partial booking state in Redis, stored beside the session's chat history."""

from __future__ import annotations

from redis.asyncio import Redis

from app.schemas.booking import BookingDraft


class RedisBookingState:
    """Stores one draft per session under `chat:{session_id}:booking`, with an expiry."""

    def __init__(self, redis: Redis, *, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        self._redis = redis
        self._ttl_seconds = ttl_seconds

    async def get(self, session_id: str) -> BookingDraft | None:
        raw = await self._redis.get(_key(session_id))
        if raw is None:
            return None
        return BookingDraft.model_validate_json(raw)

    async def save(self, session_id: str, draft: BookingDraft) -> None:
        await self._redis.set(_key(session_id), draft.model_dump_json(), ex=self._ttl_seconds)

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(_key(session_id))


def _key(session_id: str) -> str:
    return f"chat:{session_id}:booking"
