"""Data access for interview bookings."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date as dt_date
from datetime import time as dt_time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import InterviewBooking


class BookingRepository:
    """Reads and writes `InterviewBooking` rows.

    Like `DocumentRepository`, this flushes but does not commit. The service commits
    once the booking is confirmed.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        session_id: str,
        name: str,
        email: str,
        date: dt_date,
        time: dt_time,
    ) -> InterviewBooking:
        """Insert a confirmed booking. Raises IntegrityError if the slot is already taken."""
        booking = InterviewBooking(
            session_id=session_id,
            name=name,
            email=email,
            date=date,
            time=time,
        )
        self._session.add(booking)
        await self._session.flush()
        await self._session.refresh(booking)
        return booking

    async def exists(self, *, email: str, date: dt_date, time: dt_time) -> bool:
        """Return True if this email already has a booking at this date and time."""
        statement = (
            select(InterviewBooking.id)
            .where(
                InterviewBooking.email == email,
                InterviewBooking.date == date,
                InterviewBooking.time == time,
            )
            .limit(1)
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none() is not None

    async def list_by_session(self, session_id: str) -> Sequence[InterviewBooking]:
        """Return all bookings made in a chat session, in date and time order."""
        statement = (
            select(InterviewBooking)
            .where(InterviewBooking.session_id == session_id)
            .order_by(InterviewBooking.date, InterviewBooking.time)
        )
        result = await self._session.execute(statement)
        return result.scalars().all()
