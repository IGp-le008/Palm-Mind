"""Shared protocol and helpers for text extraction."""

from __future__ import annotations

from typing import Protocol

from app.core.exceptions import IngestionFailed


class TextExtractor(Protocol):
    """Turns raw file bytes into plain text. Runs in a worker thread, so it may block."""

    def extract(self, data: bytes) -> str:
        """Return the document text. Raise IngestionFailed if no usable text exists."""
        ...


def require_text(text: str, empty_message: str) -> str:
    """Return the text if it has visible content, otherwise raise IngestionFailed."""
    if not text.strip():
        raise IngestionFailed(empty_message)
    return text


