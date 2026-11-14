"""Vector store protocol and the value types shared with the rest of the application."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ChunkPayload:
    """Metadata stored alongside each chunk vector in Qdrant."""

    document_id: str
    chunk_index: int
    text: str
    source_filename: str
    strategy: str

    def to_dict(self) -> dict[str, str | int]:
        """Return the payload in the flat form Qdrant stores."""
        return {
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "source_filename": self.source_filename,
            "strategy": self.strategy,
        }


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """One chunk ready to be written: its deterministic id, embedding, and payload."""

    point_id: uuid.UUID
    vector: list[float]
    payload: ChunkPayload


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A stored chunk returned by a similarity search. Higher score means more similar."""

    point_id: uuid.UUID
    score: float
    payload: ChunkPayload


def chunk_point_id(document_id: uuid.UUID, chunk_index: int) -> uuid.UUID:
    """Deterministic id for a chunk. Re-ingesting a document overwrites its points."""
    return uuid.uuid5(document_id, str(chunk_index))


class VectorStore(Protocol):
    """Stores and searches chunk vectors. Implementations raise ProviderError on failure."""

    async def ensure_collection(self) -> None:
        """Create the collection and its document_id index if they do not exist."""
        ...

    async def upsert(self, records: list[ChunkRecord]) -> None:
        """Write records, replacing any existing point with the same id."""
        ...

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        document_id: uuid.UUID | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        """Return up to `limit` hits, best first, optionally restricted to one document."""
        ...

    async def delete_document(self, document_id: uuid.UUID) -> None:
        """Remove every point belonging to the document."""
        ...
