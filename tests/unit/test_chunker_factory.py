"""The factory maps each strategy value to the right chunker."""

from __future__ import annotations

import pytest

from app.chunking.factory import ChunkingStrategyKind, create_chunker
from app.chunking.recursive import RecursiveChunker
from app.chunking.semantic import SemanticChunker
from app.core.config import Settings


class _UnusedEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("The factory must not call the embedder.")


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    [
        (ChunkingStrategyKind.SEMANTIC, SemanticChunker),
        (ChunkingStrategyKind.RECURSIVE, RecursiveChunker),
    ],
)
def test_maps_kind_to_chunker(kind: ChunkingStrategyKind, expected_type: type) -> None:
    chunker = create_chunker(kind, Settings(), _UnusedEmbedder())
    assert isinstance(chunker, expected_type)
    assert chunker.name == kind.value
