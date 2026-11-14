"""Asks the chat model for structured booking details and validates the reply."""

from __future__ import annotations

import datetime as dt

from pydantic import ValidationError

from app.core.logging import get_logger
from app.llm.base import ChatMessage, ChatModel
from app.schemas.booking import BookingDraft, BookingExtraction

logger = get_logger(__name__)

EXTRACTION_INSTRUCTIONS = (
    "You extract interview booking details from one user message. "
    "Reply with a single JSON object and nothing else, using exactly these keys: "
    '"intent" (true only if the user wants to book an interview or is answering a booking '
    'question), "name" (string or null), "email" (string or null), "date" (YYYY-MM-DD or null), '
    '"time" (HH:MM in 24-hour format, or null), and "is_confirmed" (true for a clear yes, '
    "false for a clear no, otherwise null). Use null for anything the user did not state. "
    "Never guess."
)

_RETRY_HINT = (
    "Your last reply was not a valid JSON object with those keys. Reply with only the JSON object."
)
_ATTEMPTS = 2


class BookingExtractor:
    """Returns the fields a message contains. Never raises on malformed model output."""

    def __init__(self, chat: ChatModel, *, hours: tuple[dt.time, dt.time]) -> None:
        self._chat = chat
        self._hours = hours

    async def extract(
        self,
        message: str,
        draft: BookingDraft | None,
        today: dt.date,
    ) -> BookingExtraction:
        """Return the extracted fields. After two invalid replies, returns an empty result."""
        messages = self._prompt(message, draft, today)
        for attempt in range(1, _ATTEMPTS + 1):
            reply = await self._chat.complete(messages)
            try:
                return BookingExtraction.model_validate_json(_strip_fences(reply))
            except ValidationError:
                logger.info("Booking extraction attempt %d returned invalid JSON", attempt)
                messages = [*messages, ChatMessage("user", _RETRY_HINT)]
        return BookingExtraction()

    def _prompt(
        self,
        message: str,
        draft: BookingDraft | None,
        today: dt.date,
    ) -> list[ChatMessage]:
        start, end = self._hours
        system = (
            f"{EXTRACTION_INSTRUCTIONS}\n\n"
            f"Today is {today.isoformat()}. Business hours are {start:%H:%M} to {end:%H:%M}."
        )
        return [
            ChatMessage("system", system),
            ChatMessage("user", f"Booking so far: {_describe(draft)}\nMessage: {message}"),
        ]


def _describe(draft: BookingDraft | None) -> str:
    if draft is None:
        return "nothing yet"
    known = [
        f"{label}={value}"
        for label, value in (
            ("name", draft.name),
            ("email", draft.email),
            ("date", draft.date),
            ("time", draft.time),
        )
        if value is not None
    ]
    status = f"awaiting confirmation={draft.awaiting_confirmation}"
    return ", ".join([*known, status])


def _strip_fences(text: str) -> str:
    """Remove a Markdown code fence if the model added one around the JSON."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").removeprefix("json").strip()
    return stripped
