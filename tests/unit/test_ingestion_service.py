"""Ingestion pipeline tests with a temporary SQLite database and fake providers."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401  registers tables on Base.metadata
from app.chunking.factory import ChunkingStrategyKind
from app.chunking.recursive import RecursiveChunker
from app.core.exceptions import ProviderError
from app.db.base import Base
from app.db.models import Document, DocumentStatus
from app.extraction.pdf_extractor import PdfExtractor
from app.extraction.txt_extractor import TxtExtractor
from app.services.file_validator import FileKind
from app.services.ingestion_service import IngestionService
from app.vectorstore.base import ChunkRecord, SearchHit, chunk_point_id

TEXT = "\n\n".join(
    f"Paragraph {i} has enough filler words to force several chunks in the test." for i in range(12)
)


class FakeEmbedder:
    """Fails the first `failures` calls with a ProviderError, then returns fixed vectors."""

    def __init__(self, failures: int = 0) -> None:
        self._failures_left = failures
        self.batch_sizes: list[int] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        if self._failures_left > 0:
            self._failures_left -= 1
            raise ProviderError("Embedding provider request failed.")
        return [[float(len(text)), 1.0] for text in texts]


class FakeVectorStore:
    """Records what would be written to Qdrant."""

    def __init__(self) -> None:
        self.records: list[ChunkRecord] = []

    async def ensure_collection(self) -> None:
        return None

    async def upsert(self, records: list[ChunkRecord]) -> None:
        self.records.extend(records)

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        document_id: uuid.UUID | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        return []

    async def delete_document(self, document_id: uuid.UUID) -> None:
        return None


@pytest.fixture
async def session_factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


def _service(
    session_factory: async_sessionmaker[AsyncSession],
    store: FakeVectorStore,
    embedder: FakeEmbedder,
) -> IngestionService:
    return IngestionService(
        session_factory=session_factory,
        extractors={FileKind.TXT: TxtExtractor(), FileKind.PDF: PdfExtractor()},
        chunker_for=lambda _kind: RecursiveChunker(chunk_size=200, chunk_overlap=20),
        embedder=embedder,
        vector_store=store,
        max_upload_bytes=1024 * 1024,
        embedding_batch_size=4,
        retry_multiplier=0,
    )


async def test_completed_document_is_stored_and_indexed(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    store = FakeVectorStore()
    service = _service(session_factory, store, FakeEmbedder(failures=1))

    result = await service.ingest(
        filename="notes.txt",
        declared_mime="text/plain",
        data=TEXT.encode(),
        strategy=ChunkingStrategyKind.RECURSIVE,
    )

    assert result.status == DocumentStatus.COMPLETED.value
    assert result.chunk_count > 4  # more than one embedding batch
    assert {r.point_id for r in store.records} == {
        chunk_point_id(result.document_id, i) for i in range(result.chunk_count)
    }
    async with session_factory() as session:
        document = await session.get(Document, result.document_id)
    assert document is not None
    assert document.status == DocumentStatus.COMPLETED.value
    assert document.chunk_count == result.chunk_count


async def test_permanent_provider_failure_marks_document_failed(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    service = _service(session_factory, FakeVectorStore(), FakeEmbedder(failures=10))

    with pytest.raises(ProviderError):
        await service.ingest(
            filename="notes.txt",
            declared_mime="text/plain",
            data=TEXT.encode(),
            strategy=ChunkingStrategyKind.RECURSIVE,
        )

    async with session_factory() as session:
        document = (await session.execute(select(Document))).scalar_one()
    assert document.status == DocumentStatus.FAILED.value
    assert document.error == "Embedding provider request failed."


async def test_embeds_chunks_in_requests_of_batch_size(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    embedder = FakeEmbedder()
    service = _service(session_factory, FakeVectorStore(), embedder)

    result = await service.ingest(
        filename="notes.txt",
        declared_mime="text/plain",
        data=TEXT.encode(),
        strategy=ChunkingStrategyKind.RECURSIVE,
    )

    full, remainder = divmod(result.chunk_count, 4)
    expected = [4] * full + ([remainder] if remainder else [])
    assert embedder.batch_sizes == expected
