"""Extraction for PDF files using pypdf."""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import IngestionFailed
from app.extraction.base import require_text

_NO_TEXT_MESSAGE = (
    "The PDF contains no extractable text. It may be scanned or image-only, "
    "and OCR is not supported."
)


class PdfExtractor:
    """Extracts text page by page and joins pages with blank lines."""

    def extract(self, data: bytes) -> str:
        pages = self._read_pages(data)
        text = "\n\n".join(page.strip() for page in pages if page.strip())
        return require_text(text, _NO_TEXT_MESSAGE)

    @staticmethod
    def _read_pages(data: bytes) -> list[str]:
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise IngestionFailed("Password-protected PDFs are not supported.")
            return [page.extract_text() or "" for page in reader.pages]
        except (PdfReadError, ValueError) as exc:
            raise IngestionFailed("The PDF file could not be read.") from exc