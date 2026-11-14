"""Maps the strategy requested at upload time to a concrete chunker."""

from __future__ import annotations

from enum import StrEnum
from typing import assert_never

from app.chunking.base import ChunkingStrategy
from app.chunking.recursive import RecursiveChunker
from app.chunking.semantic import SemanticChunker
from app.core.config import Settings
from app.llm.base import EmbeddingProvider


class ChunkingStrategyKind(StrEnum):
    """Values accepted by the `chunking_strategy` form field."""

    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


def create_chunker(
    kind: ChunkingStrategyKind,
    config: Settings,
    embedder: EmbeddingProvider,
) -> ChunkingStrategy:
    """Build the chunker for `kind`. `assert_never` makes mypy flag any unhandled value."""
    match kind:
        case ChunkingStrategyKind.SEMANTIC:
            return SemanticChunker(
                embedder,
                breakpoint_percentile=config.semantic_breakpoint_percentile,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        case ChunkingStrategyKind.RECURSIVE:
            return RecursiveChunker(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        case _:
            assert_never(kind)
