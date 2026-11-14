"""Unit tests for chunk ids and payload shape. Qdrant calls are covered by integration tests."""

from __future__ import annotations

import uuid

from app.vectorstore.base import ChunkPayload, chunk_point_id


def test_point_id_is_deterministic_per_document_and_index() -> None:
    document = uuid.uuid4()
    assert chunk_point_id(document, 3) == chunk_point_id(document, 3)
    assert chunk_point_id(document, 3) != chunk_point_id(document, 4)
    assert chunk_point_id(document, 3) != chunk_point_id(uuid.uuid4(), 3)


def test_payload_dict_has_exactly_the_agreed_fields() -> None:
    payload = ChunkPayload(
        document_id="doc",
        chunk_index=0,
        text="hello",
        source_filename="a.txt",
        strategy="recursive",
    )
    assert set(payload.to_dict()) == {
        "document_id",
        "chunk_index",
        "text",
        "source_filename",
        "strategy",
    }
