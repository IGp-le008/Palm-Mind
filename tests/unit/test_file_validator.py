"""Unit tests for upload validation."""

from __future__ import annotations

import pytest

from app.core.exceptions import FileTooLarge, UnsupportedFileType
from app.services.file_validator import FileKind, validate_upload

MAX = 1024
PDF_BYTES = b"%PDF-1.4\n% minimal body"


def _validate(
    filename: str = "doc.txt",
    mime: str | None = "text/plain",
    data: bytes = b"hello",
    max_bytes: int = MAX,
) -> None:
    validate_upload(filename=filename, declared_mime=mime, data=data, max_bytes=max_bytes)


def test_accepts_valid_txt() -> None:
    result = validate_upload(
        filename="notes.TXT", declared_mime="text/plain; charset=utf-8", data=b"hi", max_bytes=MAX
    )
    assert result.kind is FileKind.TXT
    assert result.content_type == "text/plain"
    assert result.size_bytes == 2


def test_accepts_valid_pdf() -> None:
    result = validate_upload(
        filename="report.pdf", declared_mime="application/pdf", data=PDF_BYTES, max_bytes=MAX
    )
    assert result.kind is FileKind.PDF


def test_accepts_generic_mime_when_extension_and_content_agree() -> None:
    _validate(filename="doc.pdf", mime="application/octet-stream", data=PDF_BYTES)
    _validate(filename="doc.txt", mime=None, data=b"plain")


def test_rejects_file_over_size_limit() -> None:
    with pytest.raises(FileTooLarge):
        _validate(data=b"x" * (MAX + 1))


def test_accepts_file_exactly_at_size_limit() -> None:
    _validate(data=b"x" * MAX)


def test_rejects_empty_file() -> None:
    with pytest.raises(UnsupportedFileType, match="empty"):
        _validate(data=b"")


@pytest.mark.parametrize("filename", ["image.png", "archive.zip", "no_extension", "pdf"])
def test_rejects_unsupported_extension(filename: str) -> None:
    with pytest.raises(UnsupportedFileType, match=r"\.pdf and \.txt"):
        _validate(filename=filename)


def test_rejects_double_extension_trick() -> None:
    with pytest.raises(UnsupportedFileType):
        _validate(filename="evil.pdf.exe", mime="application/pdf", data=PDF_BYTES)


def test_rejects_mime_that_does_not_match_extension() -> None:
    with pytest.raises(UnsupportedFileType, match="does not match"):
        _validate(filename="doc.pdf", mime="image/png", data=PDF_BYTES)


def test_rejects_pdf_extension_without_pdf_magic() -> None:
    with pytest.raises(UnsupportedFileType, match="not a valid PDF"):
        _validate(filename="doc.pdf", mime="application/pdf", data=b"just text")


def test_rejects_txt_containing_nul_bytes() -> None:
    with pytest.raises(UnsupportedFileType, match="binary"):
        _validate(filename="doc.txt", mime="text/plain", data=b"abc\x00def")
