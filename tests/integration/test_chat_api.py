"""Integration tests for the chat endpoints: multi-turn Q&A, history, and clearing.

Uses the real PostgreSQL, Redis, and Qdrant containers. Embedding and chat providers are
faked, so no paid API is called. Requires `docker compose up -d` and migrations.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping

import httpx
import pytest
from starlette.types import ASGIApp, Receive, Scope, Send

from app.llm.base import ChatMessage
from app.main import app as fastapi_app
from app.services.prompt_builder import CONDENSE_INSTRUCTIONS

DOCUMENT = (
    "Refunds are processed within 14 days of the item being received. "
    "Customers request a refund by contacting support with their order number.\n\n"
    "Interview bookings are available on weekdays between 09:00 and 18:00."
)


class FakeEmbeddingProvider:
    def __init__(self, client: object, model: str, dimension: int) -> None:
        self._dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dimension for _ in texts]


class FakeChatModel:
    """Condenses follow-ups into a marked query. Answers report how many earlier turns it saw."""

    def __init__(self, client: object, model: str, temperature: float) -> None:
        return None

    async def complete(self, messages: list[ChatMessage]) -> str:
        if messages[0].content == CONDENSE_INSTRUCTIONS:
            latest = messages[-1].content.rsplit("Latest message: ", 1)[-1]
            return f"STANDALONE[{latest}]"
        earlier_turns = sum(1 for message in messages if message.role == "assistant")
        return f"ANSWER(turns_seen={earlier_turns}): {messages[-1].content}"


class _StateInjector:
    """Passes lifespan state to requests. httpx's ASGI transport does not do this itself."""

    def __init__(self, app: ASGIApp, state: Mapping[str, object] | None) -> None:
        self._app = app
        self._state = dict(state or {})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {}).update(self._state)
        await self._app(scope, receive, send)


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[httpx.AsyncClient]:
    monkeypatch.setattr("app.container.OpenAIEmbeddingProvider", FakeEmbeddingProvider)
    monkeypatch.setattr("app.container.OpenAIChatModel", FakeChatModel)
    async with fastapi_app.router.lifespan_context(fastapi_app) as state:
        transport = httpx.ASGITransport(app=_StateInjector(fastapi_app, state))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


@pytest.fixture
async def document_id(client: httpx.AsyncClient) -> AsyncIterator[str]:
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("policy.txt", DOCUMENT.encode(), "text/plain")},
        data={"chunking_strategy": "recursive"},
    )
    assert response.status_code == 201
    created = response.json()["id"]
    yield created
    await client.delete(f"/api/v1/documents/{created}")


async def test_follow_up_is_answered_with_the_first_turn_in_context(
    client: httpx.AsyncClient, document_id: str
) -> None:
    first = await client.post(
        "/api/v1/chat",
        json={"message": "How long do refunds take?", "document_id": document_id},
    )
    session_id = first.json()["session_id"]

    second = await client.post(
        "/api/v1/chat",
        json={
            "session_id": session_id,
            "message": "And what about the cost?",
            "document_id": document_id,
        },
    )

    assert first.status_code == 200
    assert first.json()["reply"].startswith("ANSWER(turns_seen=0)")
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id
    assert second.json()["reply"].startswith("ANSWER(turns_seen=1)")
    assert second.json()["sources"]
    assert all(source["document_id"] == document_id for source in second.json()["sources"])


async def test_history_returns_original_messages_in_order(
    client: httpx.AsyncClient, document_id: str
) -> None:
    first = await client.post(
        "/api/v1/chat", json={"message": "How long do refunds take?", "document_id": document_id}
    )
    session_id = first.json()["session_id"]
    await client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "And the cost?", "document_id": document_id},
    )

    history = await client.get(f"/api/v1/chat/{session_id}/history")

    assert history.status_code == 200
    assert [turn["user"] for turn in history.json()["turns"]] == [
        "How long do refunds take?",
        "And the cost?",
    ]


async def test_delete_clears_the_session(client: httpx.AsyncClient, document_id: str) -> None:
    first = await client.post(
        "/api/v1/chat", json={"message": "How long do refunds take?", "document_id": document_id}
    )
    session_id = first.json()["session_id"]

    deleted = await client.delete(f"/api/v1/chat/{session_id}")
    missing = await client.get(f"/api/v1/chat/{session_id}/history")

    assert deleted.status_code == 204
    assert missing.status_code == 404
    assert missing.json()["error"] == "session_not_found"


async def test_invalid_session_id_is_rejected(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/chat/not-a-uuid/history")

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_session_id"
