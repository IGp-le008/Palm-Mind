"""Booking models: what the LLM extracts, what is kept between turns, and what is saved."""

from __future__ import annotations

import datetime as dt
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr


class BookingExtraction(BaseModel):
    """Structured output the LLM returns for one message. Unexpected keys are ignored."""

    model_config = ConfigDict(extra="ignore")

    intent: bool = False
    name: str | None = None
    email: EmailStr | None = None
    date: dt.date | None = None
    time: dt.time | None = None
    is_confirmed: bool | None = None


class ConfirmedBooking(BaseModel):
    """A booking with every required field present and valid. Only this is persisted."""

    name: str
    email: EmailStr
    date: dt.date
    time: dt.time


class BookingDraft(BaseModel):
    """Partial booking kept in Redis between turns."""

    name: str | None = None
    email: EmailStr | None = None
    date: dt.date | None = None
    time: dt.time | None = None
    awaiting_confirmation: bool = False

    def missing_fields(self) -> list[str]:
        """Required fields still empty, in the order they should be asked for."""
        missing: list[str] = []
        if self.name is None or not self.name.strip():
            missing.append("name")
        if self.email is None:
            missing.append("email")
        if self.date is None:
            missing.append("date")
        if self.time is None:
            missing.append("time")
        return missing

    def to_confirmed(self) -> ConfirmedBooking | None:
        """Return the booking if every field is present, otherwise None."""
        if self.name is None or not self.name.strip():
            return None
        if self.email is None or self.date is None or self.time is None:
            return None
        return ConfirmedBooking(
            name=self.name.strip(), email=self.email, date=self.date, time=self.time
        )


def merge_draft(draft: BookingDraft, extraction: BookingExtraction) -> BookingDraft:
    """Overlay the values from this message onto the draft.

    Fields the message did not mention keep their earlier values. Changing any field
    means the summary must be confirmed again.
    """
    updates: dict[str, object] = {}
    if extraction.name and extraction.name.strip():
        updates["name"] = extraction.name.strip()
    if extraction.email is not None:
        updates["email"] = extraction.email
    if extraction.date is not None:
        updates["date"] = extraction.date
    if extraction.time is not None:
        updates["time"] = extraction.time
    if not updates:
        return draft
    changed = any(getattr(draft, key) != value for key, value in updates.items())
    awaiting = draft.awaiting_confirmation and not changed
    return draft.model_copy(update={**updates, "awaiting_confirmation": awaiting})


def validate_values(
    draft: BookingDraft,
    today: dt.date,
    hours: tuple[dt.time, dt.time],
) -> list[str]:
    """Problems with the date or time, worded for the user. An empty list means valid."""
    start, end = hours
    problems: list[str] = []
    if draft.date is not None and draft.date < today:
        problems.append("That date has already passed.")
    if draft.time is not None and not start <= draft.time < end:
        problems.append(f"Interviews are available between {start:%H:%M} and {end:%H:%M}.")
    return problems


class BookingResponse(BaseModel):
    """Public view of a stored booking."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: str
    name: str
    email: str
    date: dt.date
    time: dt.time
    status: str
    created_at: datetime


class BookingListResponse(BaseModel):
    """One page of bookings."""

    items: list[BookingResponse]
    limit: int
    offset: int
