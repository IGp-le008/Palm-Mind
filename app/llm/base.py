"""Protocols for model providers. Services depend on these, never on a vendor SDK."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Turns text into vectors. Implementations raise ProviderError on upstream failure."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order. Empty input returns []."""
        ...
