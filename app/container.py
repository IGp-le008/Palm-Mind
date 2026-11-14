"""Dependency wiring.

Builds shared clients and services once at startup and closes the clients on shutdown.
This is the composition root: it sits at the top of app/ and may import from every layer.
"""

from __future__ import annotations

import datetime as dt
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
from app.llm.base import ChatModel, EmbeddingProvider
from app.llm.openai_chat import OpenAIChatModel
from app.llm.openai_embeddings import OpenAIEmbeddingProvider
from app.memory.base import ChatMemory
from app.memory.redis_booking_state import RedisBookingState
from app.memory.redis_memory import RedisChatMemory
from app.services.booking_extractor import BookingExtractor
from app.services.booking_service import BookingService
from app.services.document_service import DocumentService
from app.services.file_validator import FileKind
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RagService
from app.services.retriever import Retriever
from app.vectorstore.base import VectorStore
from app.vectorstore.qdrant_store import QdrantVectorStore


@dataclass(frozen=True)
class Container:
    """Shared clients and services for the lifetime of the application."""

    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    redis: Redis
    qdrant: AsyncQdrantClient
    openai: AsyncOpenAI
    vector_store: VectorStore
    ingestion_service: IngestionService
    document_service: DocumentService
    booking_service: BookingService
    rag_service: RagService

    @classmethod
    def create(cls, config: Settings) -> Container:
        """Validate configuration and construct all clients. Does not open connections yet."""
        config.validate_for_storage()
        config.validate_for_embeddings()
        config.validate_for_chat()

        engine = create_db_engine(config.database_url)
        session_factory = create_session_factory(engine)
        redis = Redis.from_url(config.redis_url, decode_responses=True)
        qdrant = AsyncQdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key or None)
        openai = AsyncOpenAI(
            api_key=config.effective_embedding_key,
            base_url=config.base_url,
            timeout=config.llm_timeout_seconds,
        )

        embedder: EmbeddingProvider = OpenAIEmbeddingProvider(
            openai, config.embedding_model, config.embedding_dim
        )
        chat_model: ChatModel = OpenAIChatModel(openai, config.chat_model, config.temperature)
        vector_store = QdrantVectorStore(qdrant, config.collection_name, config.embedding_dim)
        memory: ChatMemory = RedisChatMemory(
            redis,
            max_turns=config.chat_history_turns,
            ttl_seconds=config.chat_ttl_seconds,
        )
        retriever = Retriever(
            embedder,
            vector_store,
            top_k=config.retrieval_k,
            score_threshold=config.retrieval_score_threshold,
        )

        hours = (config.business_hours_start, config.business_hours_end)
        booking_service = BookingService(
            session_factory=session_factory,
            extractor=BookingExtractor(chat_model, hours=hours),
            state=RedisBookingState(redis, ttl_seconds=config.chat_ttl_seconds),
            clock=dt.date.today,
            hours=hours,
        )

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
        rag_service = RagService(
            chat=chat_model,
            memory=memory,
            retriever=retriever,
            booking=booking_service,
        )

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
            booking_service=booking_service,
            rag_service=rag_service,
        )

    async def close(self) -> None:
        """Release all connections. Call once on application shutdown."""
        await self.redis.aclose()
        await self.openai.close()
        await self.qdrant.close()
        await self.engine.dispose()
