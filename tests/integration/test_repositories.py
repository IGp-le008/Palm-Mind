"""Repository tests against the Postgres test database."""

from __future__ import annotations

import uuid
from datetime import date, time

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentStatus
from app.repositories.booking_repository import BookingRepository
from app.repositories.document_repository import DocumentRepository


async def make_document(repo: DocumentRepository, filename: str = "a.txt") -> Document:
    return await repo.create(
        filename=filename,
        content_type="text/plain",
        size_bytes=10,
        sha256="0" * 64,
        strategy="recursive",
    )


async def test_create_document_starts_pending(session: AsyncSession) -> None:
    document = await make_document(DocumentRepository(session))

    assert document.status == DocumentStatus.PENDING.value
    assert document.chunk_count == 0


async def test_get_by_id_returns_none_for_unknown_id(session: AsyncSession) -> None:
    assert await DocumentRepository(session).get_by_id(uuid.uuid4()) is None


async def test_update_status_sets_status_and_chunk_count(session: AsyncSession) -> None:
    repo = DocumentRepository(session)
    document = await make_document(repo)

    updated = await repo.update_status(
        document.id, DocumentStatus.COMPLETED.value, chunk_count=7
    )

    assert updated is not None
    assert updated.status == DocumentStatus.COMPLETED.value
    assert updated.chunk_count == 7


async def test_update_status_returns_none_for_unknown_id(session: AsyncSession) -> None:
    repo = DocumentRepository(session)
    assert await repo.update_status(uuid.uuid4(), DocumentStatus.FAILED.value) is None


async def test_list_respects_limit_and_offset(session: AsyncSession) -> None:
    repo = DocumentRepository(session)
    for i in range(3):
        await make_document(repo, f"doc-{i}.txt")

    assert len(await repo.list(limit=2, offset=0)) == 2
    assert len(await repo.list(limit=2, offset=2)) == 1


async def test_booking_exists_after_create(session: AsyncSession) -> None:
    repo = BookingRepository(session)
    await repo.create(
        session_id="s1",
        name="Ana",
        email="ana@example.com",
        date=date(2030, 1, 10),
        time=time(10, 0),
    )

    assert await repo.exists(email="ana@example.com", date=date(2030, 1, 10), time=time(10, 0))
    assert not await repo.exists(
        email="ana@example.com", date=date(2030, 1, 10), time=time(11, 0)
    )


async def test_duplicate_slot_raises_integrity_error(session: AsyncSession) -> None:
    repo = BookingRepository(session)
    await repo.create(
        session_id="s1",
        name="Ana",
        email="ana@example.com",
        date=date(2030, 1, 10),
        time=time(10, 0),
    )

    with pytest.raises(IntegrityError):
        await repo.create(
            session_id="s2",
            name="Ana",
            email="ana@example.com",
            date=date(2030, 1, 10),
            time=time(10, 0),
        )


async def test_list_by_session_returns_only_that_session(session: AsyncSession) -> None:
    repo = BookingRepository(session)
    await repo.create(
        session_id="s1",
        name="Ana",
        email="ana@example.com",
        date=date(2030, 1, 10),
        time=time(10, 0),
    )
    await repo.create(
        session_id="s2",
        name="Ben",
        email="ben@example.com",
        date=date(2030, 1, 11),
        time=time(10, 0),
    )

    bookings = await repo.list_by_session("s1")

    assert [b.name for b in bookings] == ["Ana"]