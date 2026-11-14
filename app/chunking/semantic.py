"""Semantic chunking: split where the topic shifts, detected by embedding distance."""

from __future__ import annotations

import math
import re
from statistics import quantiles

from app.chunking.base import Chunk
from app.chunking.recursive import RecursiveChunker
from app.llm.base import EmbeddingProvider

# A sentence ends at . ! or ? followed by whitespace, or at the end of the text.
_SENTENCE = re.compile(r".+?(?:[.!?]+(?=\s)|\Z)", re.DOTALL)


class SemanticChunker:
    """Groups consecutive sentences until the embedding distance between them spikes.

    A break is placed where the distance between neighbouring sentences exceeds the
    `breakpoint_percentile` of all such distances in the document. Any group still longer
    than `chunk_size` is split by the recursive chunker, so no chunk exceeds the limit.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        *,
        breakpoint_percentile: int,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        if not 0 < breakpoint_percentile < 100:
            raise ValueError("breakpoint_percentile must be between 1 and 99.")
        self._embedder = embedder
        self._percentile = breakpoint_percentile
        self._chunk_size = chunk_size
        self._fallback = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @property
    def name(self) -> str:
        """Strategy name stored on the document row."""
        return "semantic"

    async def chunk(self, text: str) -> list[Chunk]:
        sentences = [s for s in _SENTENCE.findall(text) if s.strip()]
        if not sentences:
            return []
        pieces: list[str] = []
        for group in await self._group(sentences):
            if len(group) <= self._chunk_size:
                pieces.append(group)
            else:
                pieces.extend(chunk.text for chunk in await self._fallback.chunk(group))
        return [Chunk(index=i, text=piece) for i, piece in enumerate(pieces)]

    async def _group(self, sentences: list[str]) -> list[str]:
        """Return the text of each topic group, in document order."""
        if len(sentences) < 2:
            return ["".join(sentences).strip()]

        vectors = await self._embedder.embed([s.strip() for s in sentences])
        distances = [
            _cosine_distance(left, right) for left, right in zip(vectors, vectors[1:], strict=False)
        ]
        threshold = _percentile(distances, self._percentile)

        groups: list[str] = []
        current: list[str] = [sentences[0]]
        for sentence, distance in zip(sentences[1:], distances, strict=True):
            if distance > threshold:
                groups.append("".join(current).strip())
                current = []
            current.append(sentence)
        groups.append("".join(current).strip())
        return groups


def _cosine_distance(left: list[float], right: list[float]) -> float:
    """1 minus cosine similarity. Zero-length vectors count as identical."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return 0.0 if norm == 0.0 else 1.0 - dot / norm


def _percentile(values: list[float], percent: int) -> float:
    """Interpolated percentile. With a single value there is nothing to compare against."""
    if len(values) < 2:
        return max(values)
    return quantiles(values, n=100, method="inclusive")[percent - 1]
