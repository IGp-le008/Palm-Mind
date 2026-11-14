"""Orchestrates document ingestion: validate, save metadata, extract, chunk, embed, store."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.chunking.base import Chunk, ChunkingStrategy
from app.chunking.factory import ChunkingStrategyKind
from app.core.exceptions import AppError, IngestionFailed, ProviderError
from app.core.logging import get_logger
from app.db.models import DocumentStatus
from app.extraction.base import TextExtractor
from app.llm.base import EmbeddingProvider
from app.repositories.document_repository import DocumentRepository
from app.services.file_validator import FileKind, ValidatedFile, validate_upload
from app.vectorstore.base import ChunkPayload, ChunkRecord, VectorStore, chunk_point_id

logger = get_logger(__name__)

_MAX_EMBED_ATTEMPTS: Final = 3


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Outcome of a successful ingestion, returned to the API layer."""

    document_id: uuid.UUID
    status: str
    chunk_count: int


class IngestionService:
    """Runs the ingestion pipeline for one uploaded file.

    Each stage that changes the database uses its own short session and commits right away,
    so a failure leaves the `failed` status and error message visible to later reads.
    Files rejected by validation never create a document row.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        extractors: Mapping[FileKind, TextExtractor],
        chunker_for: Callable[[ChunkingStrategyKind], ChunkingStrategy],
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        max_upload_bytes: int,
        embedding_batch_size: int,
        retry_multiplier: float = 0.5,
    ) -> None:
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive.")
        self._session_factory = session_factory
        self._extractors = extractors
        self._chunker_for = chunker_for
        self._embedder = embedder
        self._vector_store = vector_store
        self._max_upload_bytes = max_upload_bytes
        self._batch_size = embedding_batch_size
        self._retry_multiplier = retry_multiplier

    async def ingest(
        self,
        *,
        filename: str,
        declared_mime: str | None,
        data: bytes,
        strategy: ChunkingStrategyKind,
    ) -> IngestionResult:
        """Ingest one file. Raises an AppError subclass if the file is rejected or fails."""
        upload = validate_upload(
            filename=filename,
            declared_mime=declared_mime,
            data=data,
            max_bytes=self._max_upload_bytes,
        )
        document_id = await self._create_pending(upload, data, strategy)
        await self._update(document_id, DocumentStatus.PROCESSING)
        try:
            chunk_count = await self._run_pipeline(document_id, upload, data, strategy)
        except AppError as exc:
            await self._update(document_id, DocumentStatus.FAILED, error=exc.message)
            raise
        except Exception as exc:
            logger.error("Ingestion %s failed: %s", document_id, type(exc).__name__)
            message = "Unexpected error during ingestion."
            await self._update(document_id, DocumentStatus.FAILED, error=message)
            raise IngestionFailed(message) from exc

        await self._update(document_id, DocumentStatus.COMPLETED, chunk_count=chunk_count)
        return IngestionResult(document_id, DocumentStatus.COMPLETED.value, chunk_count)

    async def _run_pipeline(
        self,
        document_id: uuid.UUID,
        upload: ValidatedFile,
        data: bytes,
        strategy: ChunkingStrategyKind,
    ) -> int:
        """Extract, chunk, embed, and store. Returns the number of chunks written."""
        text = await asyncio.to_thread(self._extractors[upload.kind].extract, data)
        chunker = self._chunker_for(strategy)
        chunks = await chunker.chunk(text)
        if not chunks:
            raise IngestionFailed("The document produced no chunks.")

        vectors = await self._embed_in_batches([chunk.text for chunk in chunks])
        records = [
            _to_record(document_id, upload.filename, chunker.name, chunk, vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await self._vector_store.ensure_collection()
        await self._vector_store.upsert(records)
        return len(records)

    async def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        """Embed in fixed-size batches so one large document does not exceed provider limits."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await self._embed_with_retry(batch))
        return vectors

    async def _embed_with_retry(self, batch: list[str]) -> list[list[float]]:
        """Retry transient provider failures with exponential backoff, then give up."""
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(ProviderError),
            stop=stop_after_attempt(_MAX_EMBED_ATTEMPTS),
            wait=wait_exponential(multiplier=self._retry_multiplier, max=8),
            reraise=True,
        )
        return await retrying(self._embedder.embed, batch)

    async def _create_pending(
        self,
        upload: ValidatedFile,
        data: bytes,
        strategy: ChunkingStrategyKind,
    ) -> uuid.UUID:
        """Insert the metadata row in the `pending` state and return its id."""
        async with self._session_factory() as session:
            document = await DocumentRepository(session).create(
                filename=upload.filename,
                content_type=upload.content_type,
                size_bytes=upload.size_bytes,
                sha256=hashlib.sha256(data).hexdigest(),
                strategy=strategy.value,
            )
            await session.commit()
            return document.id

    async def _update(
        self,
        document_id: uuid.UUID,
        status: DocumentStatus,
        *,
        chunk_count: int | None = None,
        error: str | None = None,
    ) -> None:
        """Persist a status change in its own committed transaction."""
        async with self._session_factory() as session:
            await DocumentRepository(session).update_status(
                document_id,
                status.value,
                chunk_count=chunk_count,
                error=error,
            )
            await session.commit()


def _to_record(
    document_id: uuid.UUID,
    source_filename: str,
    strategy_name: str,
    chunk: Chunk,
    vector: list[float],
) -> ChunkRecord:
    """Pair a chunk with its vector and deterministic id.

    Re-ingestion overwrites the same points.
    """
    return ChunkRecord(
        point_id=chunk_point_id(document_id, chunk.index),
        vector=vector,
        payload=ChunkPayload(
            document_id=str(document_id),
            chunk_index=chunk.index,
            text=chunk.text,
            source_filename=source_filename,
            strategy=strategy_name,
        ),
    )
