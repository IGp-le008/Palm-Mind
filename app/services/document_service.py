"""Read and delete operations for documents."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import DocumentNotFound
from app.db.models import Document
from app.repositories.document_repository import DocumentRepository
from app.vectorstore.base import VectorStore


class DocumentService:
    """Serves document metadata and removes documents from SQL and the vector store."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        vector_store: VectorStore,
    ) -> None:
        self._session_factory = session_factory
        self._vector_store = vector_store

    async def get(self, document_id: uuid.UUID) -> Document:
        """Return the document or raise DocumentNotFound."""
        async with self._session_factory() as session:
            document = await DocumentRepository(session).get_by_id(document_id)
        if document is None:
            raise DocumentNotFound("Document not found.")
        return document

    async def list(self, *, limit: int, offset: int) -> Sequence[Document]:
        """Return one page of documents, newest first."""
        async with self._session_factory() as session:
            return await DocumentRepository(session).list(limit=limit, offset=offset)

    async def delete(self, document_id: uuid.UUID) -> None:
        """Delete vectors first, then the row, so a failed vector delete can be retried."""
        await self.get(document_id)
        await self._vector_store.delete_document(document_id)
        async with self._session_factory() as session:
            await DocumentRepository(session).delete(document_id)
            await session.commit()
