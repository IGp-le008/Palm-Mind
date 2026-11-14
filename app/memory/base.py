"""Memory protocols and the value types they store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.schemas.booking import BookingDraft


@dataclass(frozen=True, slots=True)
class Turn:
    """One exchange: what the user asked and what the assistant replied."""

    user: str
    assistant: str


class ChatMemory(Protocol):
    """Stores recent turns per session."""

    async def get_turns(self, session_id: str) -> list[Turn]:
        """Return the stored turns, oldest first. Unknown sessions return []."""
        ...

    async def append_turn(self, session_id: str, turn: Turn) -> None:
        """Add a turn, keeping only the most recent ones and refreshing the expiry."""
        ...

    async def clear(self, session_id: str) -> None:
        """Remove all turns for the session."""
        ...


class BookingStateStore(Protocol):
    """Holds the partial booking for a session between turns."""

    async def get(self, session_id: str) -> BookingDraft | None:
        """Return the draft, or None if no booking is in progress."""
        ...

    async def save(self, session_id: str, draft: BookingDraft) -> None:
        """Store the draft, refreshing its expiry."""
        ...

    async def clear(self, session_id: str) -> None:
        """Remove the draft."""
        ...
