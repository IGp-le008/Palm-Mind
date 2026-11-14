"""Chat memory in Redis: one JSON list per session, with a sliding expiry."""

from __future__ import annotations

import json
from dataclasses import asdict

from redis.asyncio import Redis

from app.memory.base import Turn


class RedisChatMemory:
    """Keeps the last `max_turns` turns per session. Every write refreshes the TTL.

    Turns are stored under `chat:{session_id}:messages`. Two writes to the same session
    at the same moment can lose one turn (last write wins). A single user sends one
    message at a time, so this is accepted and recorded in DECISIONS.md.
    """

    def __init__(self, redis: Redis, *, max_turns: int, ttl_seconds: int) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive.")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        self._redis = redis
        self._max_turns = max_turns
        self._ttl_seconds = ttl_seconds

    async def get_turns(self, session_id: str) -> list[Turn]:
        raw = await self._redis.get(_key(session_id))
        if raw is None:
            return []
        return _decode(raw)

    async def append_turn(self, session_id: str, turn: Turn) -> None:
        turns = [*await self.get_turns(session_id), turn][-self._max_turns :]
        payload = json.dumps([asdict(item) for item in turns])
        await self._redis.set(_key(session_id), payload, ex=self._ttl_seconds)

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(_key(session_id))


def _key(session_id: str) -> str:
    return f"chat:{session_id}:messages"


def _decode(raw: bytes | str) -> list[Turn]:
    """Parse stored JSON. Malformed entries are skipped rather than failing the chat."""
    text = raw.decode() if isinstance(raw, bytes) else raw
    payload: object = json.loads(text)
    if not isinstance(payload, list):
        return []
    turns: list[Turn] = []
    for item in payload:
        if isinstance(item, dict):
            user, assistant = item.get("user"), item.get("assistant")
            if isinstance(user, str) and isinstance(assistant, str):
                turns.append(Turn(user=user, assistant=assistant))
    return turns
