"""Interview booking flow: detects intent, collects details, confirms, and saves."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import BookingValidationError
from app.db.models import InterviewBooking
from app.memory.base import BookingStateStore
from app.repositories.booking_repository import BookingRepository
from app.schemas.booking import (
    BookingDraft,
    ConfirmedBooking,
    merge_draft,
    validate_values,
)
from app.services.booking_extractor import BookingExtractor

DUPLICATE_MESSAGE = "That slot is already booked."

_ASK_LABELS: dict[str, str] = {
    "name": "your name",
    "email": "your email address",
    "date": "the date",
    "time": "the time",
}


@dataclass(frozen=True, slots=True)
class BookingTurn:
    """The reply for one booking message and where the booking stands.

    `status` is one of: collecting, awaiting_confirmation, confirmed, cancelled.
    """

    reply: str
    status: str


class BookingService:
    """Handles booking messages. Returns None when a message is an ordinary question."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        extractor: BookingExtractor,
        state: BookingStateStore,
        clock: Callable[[], dt.date],
        hours: tuple[dt.time, dt.time],
    ) -> None:
        self._session_factory = session_factory
        self._extractor = extractor
        self._state = state
        self._clock = clock
        self._hours = hours

    async def handle_message(self, session_id: str, message: str) -> BookingTurn | None:
        """Route one message through the booking flow, or return None for a normal question.

        A booking in progress always takes the message. Otherwise a message starts a booking
        only when the model reports booking intent.
        """
        today = self._clock()
        draft = await self._state.get(session_id)
        extraction = await self._extractor.extract(message, draft, today)

        if draft is None:
            if not extraction.intent:
                return None
            draft = BookingDraft()

        if extraction.is_confirmed is False and draft.awaiting_confirmation:
            await self._state.clear(session_id)
            return BookingTurn("Okay, I've cancelled the booking.", "cancelled")

        merged = merge_draft(draft, extraction)
        confirmed = extraction.is_confirmed is True
        return await self._advance(session_id, merged, confirmed=confirmed, today=today)

    async def list_bookings(self, *, limit: int, offset: int) -> list[InterviewBooking]:
        """Return one page of saved bookings, newest first."""
        async with self._session_factory() as session:
            rows = await BookingRepository(session).list(limit=limit, offset=offset)
            return list(rows)

    async def _advance(
        self,
        session_id: str,
        draft: BookingDraft,
        *,
        confirmed: bool,
        today: dt.date,
    ) -> BookingTurn:
        booking = draft.to_confirmed()
        if booking is None:
            await self._state.save(session_id, draft)
            return BookingTurn(_ask_for(draft.missing_fields()), "collecting")

        problems = validate_values(draft, today, self._hours)
        if problems:
            return await self._reset_slot(session_id, draft, " ".join(problems))

        if not draft.awaiting_confirmation:
            awaiting = draft.model_copy(update={"awaiting_confirmation": True})
            await self._state.save(session_id, awaiting)
            return BookingTurn(_summary(booking), "awaiting_confirmation")

        if not confirmed:
            return BookingTurn(_summary(booking), "awaiting_confirmation")

        return await self._book(session_id, booking, draft)

    async def _book(
        self,
        session_id: str,
        booking: ConfirmedBooking,
        draft: BookingDraft,
    ) -> BookingTurn:
        try:
            await self._save(session_id, booking)
        except BookingValidationError as exc:
            return await self._reset_slot(session_id, draft, exc.message)

        await self._state.clear(session_id)
        reply = (
            f"Your interview is booked for {booking.date:%A %d %B %Y} at {booking.time:%H:%M}. "
            f"We have recorded your details under {booking.email}."
        )
        return BookingTurn(reply, "confirmed")

    async def _reset_slot(self, session_id: str, draft: BookingDraft, problem: str) -> BookingTurn:
        """Clear the date and time so the user can choose again, and keep the name and email."""
        cleared = draft.model_copy(
            update={"date": None, "time": None, "awaiting_confirmation": False}
        )
        await self._state.save(session_id, cleared)
        return BookingTurn(f"{problem} {_ask_for(['date', 'time'])}", "collecting")

    async def _save(self, session_id: str, booking: ConfirmedBooking) -> None:
        """Write the booking. Duplicates are detected by lookup and by the unique constraint."""
        async with self._session_factory() as session:
            repository = BookingRepository(session)
            email = str(booking.email)
            if await repository.exists(email=email, date=booking.date, time=booking.time):
                raise BookingValidationError(DUPLICATE_MESSAGE)
            try:
                await repository.create(
                    session_id=session_id,
                    name=booking.name,
                    email=email,
                    date=booking.date,
                    time=booking.time,
                )
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise BookingValidationError(DUPLICATE_MESSAGE) from exc


def _summary(booking: ConfirmedBooking) -> str:
    return (
        f"Please confirm: {booking.name}, {booking.email}, "
        f"{booking.date:%A %d %B %Y} at {booking.time:%H:%M}. "
        "Reply yes to book or no to cancel."
    )


def _ask_for(missing: list[str]) -> str:
    """Ask for at most two missing fields at a time."""
    asked = " and ".join(_ASK_LABELS[field] for field in missing[:2])
    return f"Could you share {asked}?"
