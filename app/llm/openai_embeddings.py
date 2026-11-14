"""Embedding provider backed by an OpenAI-compatible API (OpenRouter by default)."""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAIError

from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAIEmbeddingProvider:
    """Calls the embeddings endpoint and checks each vector against the configured size.

    The dimension check catches a mismatched EMBEDDING_DIM before any vector reaches Qdrant,
    where a wrong size would otherwise fail later with a less clear error.
    """

    def __init__(self, client: AsyncOpenAI, model: str, dimension: int) -> None:
        self._client = client
        self._model = model
        self._dimension = dimension

    @classmethod
    def from_settings(cls, config: Settings) -> OpenAIEmbeddingProvider:
        """Build the provider from environment-backed settings."""
        config.validate_for_embeddings()
        client = AsyncOpenAI(api_key=config.effective_embedding_key, base_url=config.base_url)
        return cls(client, config.embedding_model, config.embedding_dim)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.embeddings.create(model=self._model, input=texts)
        except OpenAIError as exc:
            logger.warning("Embedding request failed: %s", type(exc).__name__)
            raise ProviderError("Embedding provider request failed.") from exc

        vectors = [item.embedding for item in response.data]
        if len(vectors) != len(texts):
            raise ProviderError("Embedding provider returned an unexpected number of vectors.")
        if any(len(vector) != self._dimension for vector in vectors):
            raise ProviderError("Embedding size does not match EMBEDDING_DIM.")
        return vectors
