"""Unit tests for the recursive chunker."""

from __future__ import annotations

import pytest

from app.chunking.base import ChunkingStrategy
from app.chunking.recursive import RecursiveChunker

PARAGRAPHS = [
    f"Paragraph {i} contains some filler words that keep going for a while here." for i in range(20)
]
TEXT = "\n\n".join(PARAGRAPHS)


def test_conforms_to_chunking_strategy_protocol() -> None:
    strategy: ChunkingStrategy = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    assert strategy.name == "recursive"


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)


async def test_empty_and_blank_input_returns_no_chunks() -> None:
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    assert await chunker.chunk("") == []
    assert await chunker.chunk("  \n\n  \t ") == []


async def test_short_text_becomes_single_chunk() -> None:
    chunks = await RecursiveChunker(chunk_size=100, chunk_overlap=10).chunk("hello world")
    assert [(c.index, c.text) for c in chunks] == [(0, "hello world")]


async def test_no_chunk_exceeds_size() -> None:
    chunks = await RecursiveChunker(chunk_size=200, chunk_overlap=40).chunk(TEXT)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 for chunk in chunks)


async def test_indexes_are_sequential() -> None:
    chunks = await RecursiveChunker(chunk_size=200, chunk_overlap=40).chunk(TEXT)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


async def test_prefers_paragraph_boundaries() -> None:
    text = "\n\n".join(["A" * 50, "B" * 50, "C" * 50])
    chunks = await RecursiveChunker(chunk_size=110, chunk_overlap=0).chunk(text)
    assert [chunk.text for chunk in chunks] == ["A" * 50 + "\n\n" + "B" * 50 + "\n\n", "C" * 50]


async def test_zero_overlap_loses_no_text() -> None:
    chunks = await RecursiveChunker(chunk_size=200, chunk_overlap=0).chunk(TEXT)
    assert "".join(chunk.text for chunk in chunks) == TEXT


async def test_neighbouring_chunks_share_overlap() -> None:
    chunks = await RecursiveChunker(chunk_size=120, chunk_overlap=30).chunk(TEXT)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        shared = any(
            current.text.startswith(previous.text[-k:])
            for k in range(5, min(len(previous.text), 30) + 1)
        )
        assert shared, f"no overlap between chunk {previous.index} and {current.index}"


async def test_overlap_never_exceeds_configured_limit() -> None:
    chunker = RecursiveChunker(chunk_size=120, chunk_overlap=30)
    chunks = await chunker.chunk(TEXT)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        overlap = _longest_overlap(previous.text, current.text)
        assert overlap <= 30


async def test_overlap_starts_on_word_boundary() -> None:
    chunks = await RecursiveChunker(chunk_size=120, chunk_overlap=30).chunk(TEXT)
    for current in chunks[1:]:
        assert not current.text[0].isspace()
        first_word = current.text.split()[0]
        assert first_word in TEXT.split()


async def test_falls_back_to_character_windows() -> None:
    chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0, separators=())
    chunks = await chunker.chunk("abcdefghijklmnopqrstuvwxyz")
    assert [chunk.text for chunk in chunks] == ["abcdefghij", "klmnopqrst", "uvwxyz"]


def _longest_overlap(left: str, right: str) -> int:
    """Length of the longest suffix of `left` that is also a prefix of `right`."""
    limit = min(len(left), len(right))
    return max((k for k in range(1, limit + 1) if right.startswith(left[-k:])), default=0)
