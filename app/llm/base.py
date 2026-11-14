"""Protocols for model providers. Services depend on these, never on a vendor SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a prompt sent to the chat model."""

    role: Role
    content: str


class EmbeddingProvider(Protocol):
    """Turns text into vectors. Implementations raise ProviderError on upstream failure."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order. Empty input returns []."""
        ...


class ChatModel(Protocol):
    """Generates a reply from messages. Implementations raise ProviderError on failure."""

    async def complete(self, messages: list[ChatMessage]) -> str:
        """Return the reply text, stripped. An empty reply raises ProviderError."""
        ...
