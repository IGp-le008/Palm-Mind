"""Booking extractor: retries once after invalid JSON, then gives up without raising."""

from __future__ import annotations

import datetime as dt

from app.llm.base import ChatMessage
from app.services.booking_extractor import BookingExtractor

TODAY = dt.date(2099, 1, 1)
HOURS = (dt.time(9), dt.time(18))


class ScriptedChat:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    async def complete(self, messages: list[ChatMessage]) -> str:
        self.calls += 1
        return self._replies.pop(0)


async def test_retries_once_after_invalid_json() -> None:
    chat = ScriptedChat(["not json", '{"intent": true, "name": "Ana"}'])

    result = await BookingExtractor(chat, hours=HOURS).extract("book me", None, TODAY)

    assert chat.calls == 2
    assert result.intent is True
    assert result.name == "Ana"


async def test_gives_up_quietly_after_two_invalid_replies() -> None:
    chat = ScriptedChat(["nope", "still nope"])

    result = await BookingExtractor(chat, hours=HOURS).extract("hello", None, TODAY)

    assert chat.calls == 2
    assert result.intent is False
