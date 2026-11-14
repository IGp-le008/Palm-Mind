"""Recursive chunking: split on paragraphs, then lines, sentences, and finally characters."""

from __future__ import annotations

from app.chunking.base import Chunk

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "")


class RecursiveChunker:
    """Packs text into chunks of at most `chunk_size` characters.

    The text is split at the strongest separator available: paragraphs first, then lines,
    then sentences, then fixed character windows. Each separator stays attached to the
    text before it, so joining the pieces reproduces the original text exactly. Neighbouring
    chunks share up to `chunk_overlap` characters, trimmed to a word boundary.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: tuple[str, ...] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators

    @property
    def name(self) -> str:
        """Strategy name stored on the document row."""
        return "recursive"

    async def chunk(self, text: str) -> list[Chunk]:
        """Return ordered, non-blank chunks. Empty or whitespace-only input returns []."""
        pieces = _split(text, self._chunk_size, self._separators)
        merged = _merge(pieces, self._chunk_size, self._chunk_overlap)
        visible = [value for value in merged if value.strip()]
        return [Chunk(index=i, text=value) for i, value in enumerate(visible)]


def _split(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    """Break text into pieces no longer than `size`, preferring the strongest separator."""
    if not text:
        return []
    if len(text) <= size:
        return [text]
    if not separators or separators[0] == "":
        return _hard_split(text, size)

    separator, remaining = separators[0], separators[1:]
    parts = text.split(separator)
    if len(parts) == 1:
        return _split(text, size, remaining)

    attached = [part + separator for part in parts[:-1]] + [parts[-1]]
    pieces: list[str] = []
    for piece in attached:
        pieces.extend(_split(piece, size, remaining))
    return pieces


def _hard_split(text: str, size: int) -> list[str]:
    """Cut text into fixed windows of `size` characters."""
    return [text[start : start + size] for start in range(0, len(text), size)]


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack pieces into chunks, starting each new chunk with a tail of the last one."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > size:
            chunks.append(current)
            # Keep the new chunk within `size` even after the overlap is added.
            current = _overlap_tail(current, min(overlap, size - len(piece)))
        current += piece
    if current:
        chunks.append(current)
    return chunks


def _overlap_tail(chunk: str, limit: int) -> str:
    """Return up to `limit` trailing characters of `chunk`, starting after a whitespace."""
    if limit <= 0:
        return ""
    tail = chunk[-limit:]
    boundary = next((i for i, char in enumerate(tail) if char.isspace()), None)
    if boundary is None:
        return tail
    return tail[boundary + 1 :]