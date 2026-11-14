"""Integration test for the booking flow: four messages from first request to a saved row.

Uses the real PostgreSQL, Redis, and Qdrant containers. Both providers are faked, so no
paid API is called. Requires `docker compose up -d` and migrations.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import cast

import httpx
import pytest
from sqlalchemy import delete
from starlette.types import ASGIApp, Receive, Scope, Send

from app.container import Container
from app.db.models import InterviewBooking
from app.llm.base import ChatMessage
from app.main import app as fastapi_app
from app.services.booking_extractor import EXTRACTION_INSTRUCTIONS

EMAIL = "ana.booking@example.com"
INTENT_MESSAGE = "I want to book an interview"
DETAILS_MESSAGE = f"My name is Ana Test and my email is {EMAIL}"
SLOT_MESSAGE = "March 10 2099 at 10:00"
CONFIRM_MESSAGE = "yes"

EXTRACTIONS: dict[str, dict[str, object]] = {
    INTENT_MESSAGE: {"intent": True},
    DETAILS_MESSAGE: {"name": "Ana Test", "email": EMAIL},
    SLOT_MESSAGE: {"date": "2099-03-10", "time": "10:00"},
    CONFIRM_MESSAGE: {"is_confirmed": True},
}


class FakeEmbeddingProvider:
    def __init__(self, client: object, model: str, dimension: int) -> None:
        self._dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dimension for _ in texts]


class FakeChatModel:
    """Answers only booking-extraction requests. Any other call is a test failure."""

    def __init__(self, client: object, model: str, temperature: float) -> None:
        return None

    async def complete(self, messages: list[ChatMessage]) -> str:
        if not messages[0].content.startswith(EXTRACTION_INSTRUCTIONS):
            raise AssertionError("Only booking extraction is expected in this test.")
        message = messages[-1].content.rsplit("Message: ", 1)[-1]
        return json.dumps(EXTRACTIONS[message])


class _StateInjector:
    """Passes lifespan state to requests. httpx's ASGI transport does not do this itself."""

    def __init__(self, app: ASGIApp, state: dict[str, object]) -> None:
        self._app = app
        self._state = state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {}).update(self._state)
        await self._app(scope, receive, send)


@pytest.fixture
async def stack(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[httpx.AsyncClient, Container]]:
    monkeypatch.setattr("app.container.OpenAIEmbeddingProvider", FakeEmbeddingProvider)
    monkeypatch.setattr("app.container.OpenAIChatModel", FakeChatModel)
    async with fastapi_app.router.lifespan_context(fastapi_app) as state:
        assert state is not None
        container = cast(Container, state["container"])
        transport = httpx.ASGITransport(app=_StateInjector(fastapi_app, dict(state)))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http, container


async def test_booking_is_saved_after_confirmation(
    stack: tuple[httpx.AsyncClient, Container],
) -> None:
    http, container = stack
    first = await http.post("/api/v1/chat", json={"message": INTENT_MESSAGE})
    session_id = first.json()["session_id"]
    try:
        replies = [first]
        for message in (DETAILS_MESSAGE, SLOT_MESSAGE, CONFIRM_MESSAGE):
            replies.append(
                await http.post(
                    "/api/v1/chat",
                    json={"session_id": session_id, "message": message},
                )
            )

        statuses = [reply.json()["booking_status"] for reply in replies]
        assert statuses == ["collecting", "collecting", "awaiting_confirmation", "confirmed"]

        listing = await http.get("/api/v1/bookings")
        emails = [item["email"] for item in listing.json()["items"]]
        assert EMAIL in emails
    finally:
        async with container.session_factory() as db:
            await db.execute(delete(InterviewBooking).where(InterviewBooking.email == EMAIL))
            await db.commit()
        await http.delete(f"/api/v1/chat/{session_id}")
