"""Unit tests for the Chunk value type."""

from __future__ import annotations

import pytest

from app.chunking.base import Chunk


def test_chunk_holds_index_and_text() -> None:
    chunk = Chunk(index=0, text="hello")
    assert chunk.index == 0
    assert chunk.text == "hello"


def test_chunk_is_immutable() -> None:
    chunk = Chunk(index=0, text="hello")
    with pytest.raises(AttributeError):
        chunk.text = "changed"  # type: ignore[misc]


def test_negative_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Chunk(index=-1, text="hello")


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_text_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="visible characters"):
        Chunk(index=0, text=text)
