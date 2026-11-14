"""Qdrant implementation of the VectorStore protocol."""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.vectorstore.base import ChunkPayload, ChunkRecord, SearchHit

logger = get_logger(__name__)

_DOCUMENT_FIELD = "document_id"


class QdrantVectorStore:
    """Cosine-distance collection. Scores returned are cosine similarity, so higher is closer."""

    def __init__(self, client: AsyncQdrantClient, collection: str, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        self._client = client
        self._collection = collection
        self._dimension = dimension

    @classmethod
    def from_settings(cls, config: Settings) -> QdrantVectorStore:
        """Build the store from environment-backed settings."""
        client = AsyncQdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key or None)
        return cls(client, config.collection_name, config.embedding_dim)

    async def ensure_collection(self) -> None:
        try:
            if await self._client.collection_exists(self._collection):
                return
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
            )
            # Index the filter field so per-document searches and deletes stay fast.
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=_DOCUMENT_FIELD,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            raise _provider_error("ensure_collection", exc) from exc

    async def upsert(self, records: list[ChunkRecord]) -> None:
        if not records:
            return
        points = [
            PointStruct(
                id=str(record.point_id),
                vector=record.vector,
                payload=record.payload.to_dict(),
            )
            for record in records
        ]
        try:
            await self._client.upsert(collection_name=self._collection, points=points)
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            raise _provider_error("upsert", exc) from exc

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        document_id: uuid.UUID | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        query_filter = None if document_id is None else _document_filter(document_id)
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            raise _provider_error("search", exc) from exc

        return [
            SearchHit(
                point_id=uuid.UUID(str(point.id)),
                score=point.score,
                payload=_payload_from(point.payload),
            )
            for point in response.points
        ]

    async def delete_document(self, document_id: uuid.UUID) -> None:
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=FilterSelector(filter=_document_filter(document_id)),
            )
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            raise _provider_error("delete", exc) from exc


def _document_filter(document_id: uuid.UUID) -> Filter:
    return Filter(
        must=[FieldCondition(key=_DOCUMENT_FIELD, match=MatchValue(value=str(document_id)))]
    )


def _payload_from(raw: Mapping[str, object] | None) -> ChunkPayload:
    """Rebuild a ChunkPayload from Qdrant's loose dict. Raises if a field is missing or mistyped."""
    if raw is None:
        raise ProviderError("Stored chunk has no payload.")
    text_fields = ("document_id", "text", "source_filename", "strategy")
    chunk_index = raw.get("chunk_index")
    if not all(isinstance(raw.get(name), str) for name in text_fields) or not isinstance(
        chunk_index, int
    ):
        raise ProviderError("Stored chunk payload is malformed.")
    return ChunkPayload(
        document_id=str(raw["document_id"]),
        chunk_index=chunk_index,
        text=str(raw["text"]),
        source_filename=str(raw["source_filename"]),
        strategy=str(raw["strategy"]),
    )


def _provider_error(operation: str, exc: Exception) -> ProviderError:
    """Log the operation and error type only. Qdrant error bodies can echo chunk text."""
    logger.warning("Qdrant %s failed: %s", operation, type(exc).__name__)
    return ProviderError("Vector store request failed.")