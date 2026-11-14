"""Chunking protocol and the Chunk value type shared by all strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous piece of document text.

    `index` is the chunk's position in the document, starting at 0. It is used to build
    the deterministic chunk id, so re-ingesting the same document overwrites its vectors.
    """

    index: int
    text: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Chunk index must be non-negative.")
        if not self.text.strip():
            raise ValueError("Chunk text must contain visible characters.")


class ChunkingStrategy(Protocol):
    """Splits document text into ordered chunks.

    Implementations are async so that strategies which call an embedding provider
    (semantic chunking) do not block the event loop. Strategies that do no I/O
    (recursive chunking) simply return immediately.
    """

    @property
    def name(self) -> str:
        """Stable strategy name stored in the `documents.strategy` column."""
        ...

    async def chunk(self, text: str) -> list[Chunk]:
        """Return ordered, non-empty chunks covering the text. Empty input returns []."""
        ...