"""Unit tests for text and PDF extraction."""

from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfWriter

from app.core.exceptions import IngestionFailed
from app.extraction.pdf_extractor import PdfExtractor
from app.extraction.txt_extractor import TxtExtractor


def test_txt_decodes_utf8() -> None:
    assert TxtExtractor().extract("héllo wörld".encode()) == "héllo wörld"


def test_txt_falls_back_to_cp1252() -> None:
    assert TxtExtractor().extract("café".encode("cp1252")) == "café"


def test_txt_rejects_whitespace_only() -> None:
    with pytest.raises(IngestionFailed):
        TxtExtractor().extract(b"   \n\t  ")


def test_pdf_without_text_is_rejected() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)

    with pytest.raises(IngestionFailed, match="no extractable text"):
        PdfExtractor().extract(buffer.getvalue())


def test_pdf_garbage_bytes_are_rejected() -> None:
    with pytest.raises(IngestionFailed, match="could not be read"):
        PdfExtractor().extract(b"not a pdf")
