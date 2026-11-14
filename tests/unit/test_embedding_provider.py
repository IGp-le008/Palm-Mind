"""Checks the embedding provider's response handling without calling the network."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from openai import AsyncOpenAI

from app.core.exceptions import ProviderError
from app.llm.openai_embeddings import OpenAIEmbeddingProvider


class _FakeEmbeddings:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    async def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in self._vectors])


class _FakeClient:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embeddings = _FakeEmbeddings(vectors)


def _provider(vectors: list[list[float]], dimension: int) -> OpenAIEmbeddingProvider:
    client = cast(AsyncOpenAI, _FakeClient(vectors))
    return OpenAIEmbeddingProvider(client, model="test-model", dimension=dimension)


async def test_returns_vectors_in_order() -> None:
    provider = _provider([[0.1, 0.2], [0.3, 0.4]], dimension=2)
    assert await provider.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]


async def test_rejects_vector_size_mismatch() -> None:
    provider = _provider([[0.1, 0.2, 0.3]], dimension=2)
    with pytest.raises(ProviderError, match="EMBEDDING_DIM"):
        await provider.embed(["a"])
