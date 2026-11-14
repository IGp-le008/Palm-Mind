"""Chat memory protocol and the Turn value type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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
