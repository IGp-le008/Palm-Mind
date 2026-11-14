"""Extraction for plain text files."""

from __future__ import annotations

from app.extraction.base import require_text

_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "cp1252")


class TxtExtractor:
    """Decodes a text file as UTF-8, falling back to Windows-1252, then Latin-1.

    Latin-1 maps every byte, so decoding always succeeds. Files that reach it
    may show mojibake, but they are still ingested rather than rejected.
    """

    def extract(self, data: bytes) -> str:
        return require_text(self._decode(data), "The text file is empty.")

    @staticmethod
    def _decode(data: bytes) -> str:
        for encoding in _ENCODINGS:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("latin-1")
