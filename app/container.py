"""Dependency wiring.

Builds shared clients and services once at startup and closes the clients on shutdown.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.chunking.factory import create_chunker
from app.core.config import Settings
from app.db.session import create_db_engine, create_session_factory
from app.extraction.pdf_extractor import PdfExtractor
from app.extraction.txt_extractor import TxtExtractor
from app.llm.base import EmbeddingProvider
from app.llm.openai_embeddings import OpenAIEmbeddingProvider
from app.services.document_service import DocumentService
from app.services.file_validator import FileKind
from app.services.ingestion_service import IngestionService
from app.vectorstore.base import VectorStore
from app.vectorstore.qdrant_store import QdrantVectorStore


@dataclass(frozen=True)
class Container:
    """Shared clients and services for the lifetime of the application."""

    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    redis: Redis[str]
    qdrant: AsyncQdrantClient
    openai: AsyncOpenAI
    vector_store: VectorStore
    ingestion_service: IngestionService
    document_service: DocumentService

    @classmethod
    def create(cls, config: Settings) -> Container:
        """Validate configuration and construct all clients. Does not open connections yet."""
        config.validate_for_storage()
        config.validate_for_embeddings()

        engine = create_db_engine(config.database_url)
        session_factory = create_session_factory(engine)
        redis = Redis.from_url(config.redis_url, decode_responses=True)
        qdrant = AsyncQdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key or None)
        openai = AsyncOpenAI(api_key=config.effective_embedding_key, base_url=config.base_url)

        embedder: EmbeddingProvider = OpenAIEmbeddingProvider(
            openai, config.embedding_model, config.embedding_dim
        )
        vector_store = QdrantVectorStore(qdrant, config.collection_name, config.embedding_dim)
        ingestion_service = IngestionService(
            session_factory=session_factory,
            extractors={FileKind.PDF: PdfExtractor(), FileKind.TXT: TxtExtractor()},
            chunker_for=lambda kind: create_chunker(kind, config, embedder),
            embedder=embedder,
            vector_store=vector_store,
            max_upload_bytes=config.max_upload_bytes,
            embedding_batch_size=config.embedding_batch_size,
        )
        document_service = DocumentService(session_factory, vector_store)

        return cls(
            settings=config,
            engine=engine,
            session_factory=session_factory,
            redis=redis,
            qdrant=qdrant,
            openai=openai,
            vector_store=vector_store,
            ingestion_service=ingestion_service,
            document_service=document_service,
        )

    async def close(self) -> None:
        """Release all connections. Call once on application shutdown."""
        await self.redis.aclose()
        await self.openai.close()
        await self.qdrant.close()
        await self.engine.dispose()