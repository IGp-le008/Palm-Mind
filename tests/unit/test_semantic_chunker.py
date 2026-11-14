"""Unit tests for the semantic chunker, using a fake embedder that encodes topic."""

from __future__ import annotations

import pytest

from app.chunking.semantic import SemanticChunker


class TopicEmbedder:
    """Sentences mentioning 'cat' embed to one axis, everything else to the other."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "cat" in text else [0.0, 1.0] for text in texts]


CAT_THEN_STOCKS = (
    "The cat sleeps all day. The cat eats fish. The cat purrs loudly. "
    "Stock prices rose sharply. Stock markets fell today. Stock traders stayed busy."
)


async def test_splits_where_topic_changes() -> None:
    chunker = SemanticChunker(
        TopicEmbedder(), breakpoint_percentile=80, chunk_size=800, chunk_overlap=0
    )
    chunks = await chunker.chunk(CAT_THEN_STOCKS)
    assert [chunk.text for chunk in chunks] == [
        "The cat sleeps all day. The cat eats fish. The cat purrs loudly.",
        "Stock prices rose sharply. Stock markets fell today. Stock traders stayed busy.",
    ]


async def test_oversized_group_is_split_within_limit() -> None:
    long_topic = " ".join(["The cat sits quietly by the window."] * 20)
    chunks = await SemanticChunker(
        TopicEmbedder(), breakpoint_percentile=95, chunk_size=60, chunk_overlap=0
    ).chunk(long_topic)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 60 for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


async def test_empty_input_returns_no_chunks() -> None:
    chunker = SemanticChunker(
        TopicEmbedder(), breakpoint_percentile=95, chunk_size=800, chunk_overlap=0
    )
    assert await chunker.chunk("   ") == []


@pytest.mark.parametrize("percentile", [0, 100])
def test_rejects_percentile_outside_range(percentile: int) -> None:
    with pytest.raises(ValueError):
        SemanticChunker(
            TopicEmbedder(), breakpoint_percentile=percentile, chunk_size=800, chunk_overlap=0
        )
