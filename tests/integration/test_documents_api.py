"""Integration tests for the document endpoints.

Uses the real PostgreSQL, Redis, and Qdrant containers. Only the embedding provider is
faked, so no paid API is called. Requires `docker compose up -d` and
`uv run alembic upgrade head` before running.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx
import pytest
from starlette.types import Receive, Scope, Send

from app.core.config import settings
from app.main import app as fastapi_app

TXT_BODY = "\n\n".join(
    f"Paragraph {i} explains the refund policy in enough words to be split into chunks."
    for i in range(40)
).encode()

PDF_LINES = [
    f"Line {i} describes the refund policy so that the text spans several chunks."
    for i in range(40)
]


class FakeEmbeddingProvider:
    """Stands in for OpenAIEmbeddingProvider and accepts the same constructor arguments."""

    def __init__(self, client: object, model: str, dimension: int) -> None:
        self._dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dimension for _ in texts]


class _LifespanStateInjector:
    """Passes lifespan state to requests. httpx's ASGI transport does not do this itself."""

    def __init__(self, app: object, state: dict[str, object]) -> None:
        self._app = app
        self._state = state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {}).update(self._state)
        await self._app(scope, receive, send)  # type: ignore[operator]


@dataclass
class Api:
    """HTTP client plus the ids of documents created by a test, so they can be removed."""

    http: httpx.AsyncClient
    created: list[uuid.UUID] = field(default_factory=list)

    async def upload(
        self,
        filename: str,
        content_type: str,
        data: bytes,
        strategy: str = "recursive",
    ) -> httpx.Response:
        response = await self.http.post(
            "/api/v1/documents",
            files={"file": (filename, data, content_type)},
            data={"chunking_strategy": strategy},
        )
        if response.status_code == 201:
            self.created.append(uuid.UUID(response.json()["id"]))
        return response


@pytest.fixture
async def api(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Api]:
    monkeypatch.setattr("app.container.OpenAIEmbeddingProvider", FakeEmbeddingProvider)
    async with fastapi_app.router.lifespan_context(fastapi_app) as state:
        transport = httpx.ASGITransport(app=_LifespanStateInjector(fastapi_app, state))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            wrapper = Api(http)
            yield wrapper
            for document_id in wrapper.created:
                await http.delete(f"/api/v1/documents/{document_id}")


def _text_pdf(lines: list[str]) -> bytes:
    """Build a minimal, valid, text-based PDF so the extractor has real text to read."""
    ops = "BT /F1 12 Tf 72 720 Td " + "".join(f"({line}) Tj 0 -14 Td " for line in lines) + "ET"
    stream = ops.encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    out += trailer.encode()
    return bytes(out)


@pytest.mark.parametrize("strategy", ["semantic", "recursive"])
async def test_txt_upload_completes_with_each_strategy(api: Api, strategy: str) -> None:
    response = await api.upload("notes.txt", "text/plain", TXT_BODY, strategy)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert body["chunking_strategy"] == strategy
    assert body["chunk_count"] > 1
    assert body["error"] is None


@pytest.mark.parametrize("strategy", ["semantic", "recursive"])
async def test_pdf_upload_completes_with_each_strategy(api: Api, strategy: str) -> None:
    response = await api.upload("report.pdf", "application/pdf", _text_pdf(PDF_LINES), strategy)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert body["chunking_strategy"] == strategy
    assert body["chunk_count"] >= 1


async def test_uploaded_document_can_be_read_back(api: Api) -> None:
    created = await api.upload("notes.txt", "text/plain", TXT_BODY)
    document_id = created.json()["id"]

    response = await api.http.get(f"/api/v1/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["id"] == document_id


async def test_deleted_document_is_gone(api: Api) -> None:
    created = await api.upload("notes.txt", "text/plain", TXT_BODY)
    document_id = created.json()["id"]

    deleted = await api.http.delete(f"/api/v1/documents/{document_id}")
    missing = await api.http.get(f"/api/v1/documents/{document_id}")

    assert deleted.status_code == 204
    assert missing.status_code == 404
    assert missing.json()["error"] == "document_not_found"
    api.created.remove(uuid.UUID(document_id))


@pytest.mark.parametrize(
    ("filename", "content_type", "data", "expected_error"),
    [
        ("photo.png", "image/png", b"\x89PNG\r\n", "unsupported_file_type"),
        ("fake.pdf", "application/pdf", b"just some text", "unsupported_file_type"),
        ("binary.txt", "text/plain", b"abc\x00def", "unsupported_file_type"),
        ("empty.txt", "text/plain", b"", "unsupported_file_type"),
    ],
)
async def test_rejects_invalid_file_content(
    api: Api, filename: str, content_type: str, data: bytes, expected_error: str
) -> None:
    response = await api.upload(filename, content_type, data)

    assert response.status_code == 415
    assert response.json()["error"] == expected_error


async def test_rejects_file_over_size_limit(api: Api) -> None:
    oversized = b"x" * (settings.max_upload_bytes + 1)

    response = await api.upload("big.txt", "text/plain", oversized)

    assert response.status_code == 413
    assert response.json()["error"] == "file_too_large"


async def test_rejects_unknown_chunking_strategy(api: Api) -> None:
    response = await api.upload("notes.txt", "text/plain", TXT_BODY, strategy="fixed")

    assert response.status_code == 422
    assert api.created == []