"""Embeds a query and searches the vector store."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.llm.base import EmbeddingProvider
from app.vectorstore.base import VectorStore


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by search, with the fields the prompt and the client need."""

    document_id: uuid.UUID
    chunk_index: int
    text: str


class Retriever:
    """Runs one similarity search per query."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        *,
        top_k: int,
        score_threshold: float,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k
        self._score_threshold = score_threshold

    async def retrieve(
        self,
        query: str,
        document_id: uuid.UUID | None = None,
    ) -> list[RetrievedChunk]:
        """Return up to `top_k` chunks above the score threshold, best first."""
        [vector] = await self._embedder.embed([query])
        hits = await self._vector_store.search(
            vector,
            limit=self._top_k,
            document_id=document_id,
            score_threshold=self._score_threshold,
        )
        return [
            RetrievedChunk(
                document_id=uuid.UUID(hit.payload.document_id),
                chunk_index=hit.payload.chunk_index,
                text=hit.payload.text,
            )
            for hit in hits
        ]
