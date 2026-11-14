"""Data access for document metadata."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document


class DocumentRepository:
    """Reads and writes `Document` rows.

    The repository flushes changes but does not commit. The calling service owns the
    transaction, so a failed ingestion step can roll back cleanly.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        filename: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
        strategy: str,
    ) -> Document:
        """Insert a new document in the `pending` state."""
        document = Document(
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=sha256,
            strategy=strategy,
        )
        self._session.add(document)
        await self._session.flush()
        await self._session.refresh(document)
        return document

    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        """Return the document with this id, or None if it does not exist."""
        return await self._session.get(Document, document_id)

    async def update_status(
        self,
        document_id: uuid.UUID,
        status: str,
        *,
        chunk_count: int | None = None,
        error: str | None = None,
    ) -> Document | None:
        """Set the ingestion status. Returns None if the document does not exist."""
        document = await self.get_by_id(document_id)
        if document is None:
            return None
        document.status = status
        if chunk_count is not None:
            document.chunk_count = chunk_count
        document.error = error
        await self._session.flush()
        await self._session.refresh(document)
        return document

    async def list(self, *, limit: int = 20, offset: int = 0) -> Sequence[Document]:
        """Return documents newest first, one page at a time."""
        statement = (
            select(Document)
            .order_by(Document.created_at.desc(), Document.id)
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(statement)
        return result.scalars().all()

    async def delete(self, document_id: uuid.UUID) -> bool:
        """Delete the document row. Returns False if it does not exist."""
        document = await self.get_by_id(document_id)
        if document is None:
            return False
        await self._session.delete(document)
        await self._session.flush()
        return True
