"""Booking rules: merging keeps earlier values, and validation covers dates and hours."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from app.schemas.booking import BookingDraft, BookingExtraction, merge_draft, validate_values

TODAY = dt.date(2099, 1, 1)
HOURS = (dt.time(9), dt.time(18))


def test_merge_keeps_earlier_values_when_a_message_is_silent() -> None:
    merged = merge_draft(BookingDraft(name="Ana"), BookingExtraction(intent=True))

    assert merged.name == "Ana"


def test_changing_a_field_after_the_summary_requires_confirmation_again() -> None:
    draft = BookingDraft(
        name="Ana",
        email="ana@example.com",
        date=TODAY,
        time=dt.time(10),
        awaiting_confirmation=True,
    )

    merged = merge_draft(draft, BookingExtraction(time=dt.time(11)))

    assert merged.time == dt.time(11)
    assert merged.awaiting_confirmation is False


def test_missing_fields_are_listed_in_the_order_they_are_asked() -> None:
    assert BookingDraft(name=" ").missing_fields() == ["name", "email", "date", "time"]


def test_invalid_email_is_rejected_by_the_model() -> None:
    with pytest.raises(ValidationError):
        BookingDraft(email="not-an-email")


def test_past_date_is_rejected() -> None:
    draft = BookingDraft(date=TODAY - dt.timedelta(days=1))

    assert validate_values(draft, TODAY, HOURS) == ["That date has already passed."]


def test_time_outside_business_hours_is_rejected() -> None:
    draft = BookingDraft(time=dt.time(20))

    problems = validate_values(draft, TODAY, HOURS)

    assert len(problems) == 1
    assert "09:00" in problems[0]


def test_valid_date_and_time_have_no_problems() -> None:
    draft = BookingDraft(date=TODAY, time=dt.time(10))

    assert validate_values(draft, TODAY, HOURS) == []
