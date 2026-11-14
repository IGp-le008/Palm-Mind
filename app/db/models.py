"""ORM models for PostgreSQL."""

from __future__ import annotations

import uuid
from datetime import date as dt_date
from datetime import datetime
from datetime import time as dt_time
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Integer,
    String,
    Text,
    Time,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class DocumentStatus(StrEnum):
    """Ingestion lifecycle: pending -> processing -> completed | failed."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BookingStatus(StrEnum):
    """Lifecycle of an interview booking. Rows are written only after the user confirms."""

    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


class ChunkingStrategyName(StrEnum):
    """Chunking strategies accepted by the ingestion API."""

    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


class Document(Base):
    """Metadata for an uploaded document. Chunk vectors live in Qdrant, not here."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    strategy: Mapped[str] = mapped_column(String(20))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default=DocumentStatus.PENDING.value)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class InterviewBooking(Base):
    """A confirmed interview slot created through the chat booking flow."""

    __tablename__ = "interview_bookings"
    __table_args__ = (UniqueConstraint("email", "date", "time", name="uq_interview_booking_slot"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254))
    date: Mapped[dt_date] = mapped_column(Date)
    time: Mapped[dt_time] = mapped_column(Time)
    status: Mapped[str] = mapped_column(String(20), default=BookingStatus.CONFIRMED.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
