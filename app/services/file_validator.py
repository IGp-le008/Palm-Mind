"""Validation for uploaded document files: size, extension, MIME type, and magic bytes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from app.core.exceptions import FileTooLarge, UnsupportedFileType

_GENERIC_MIME = "application/octet-stream"
_PDF_MAGIC = b"%PDF-"
_NUL_BYTE = b"\x00"


class FileKind(StrEnum):
    """File types the ingestion API accepts."""

    PDF = "pdf"
    TXT = "txt"


_KIND_BY_EXTENSION: dict[str, FileKind] = {".pdf": FileKind.PDF, ".txt": FileKind.TXT}

_ALLOWED_MIME: dict[FileKind, frozenset[str]] = {
    FileKind.PDF: frozenset({"application/pdf", _GENERIC_MIME}),
    FileKind.TXT: frozenset({"text/plain", _GENERIC_MIME}),
}

_CANONICAL_MIME: dict[FileKind, str] = {
    FileKind.PDF: "application/pdf",
    FileKind.TXT: "text/plain",
}


@dataclass(frozen=True)
class ValidatedFile:
    """An upload that passed every check. Carries the canonical content type."""

    filename: str
    kind: FileKind
    content_type: str
    size_bytes: int


def validate_upload(
    *,
    filename: str,
    declared_mime: str | None,
    data: bytes,
    max_bytes: int,
) -> ValidatedFile:
    """Run all upload checks in order and return the validated file.

    Order: size, emptiness, extension, MIME type, magic bytes. The first failing
    check raises, so callers get one clear error per request.
    """
    _check_size(len(data), max_bytes)
    _check_not_empty(data)
    kind = _kind_from_extension(filename)
    _check_mime(kind, declared_mime)
    _check_magic_bytes(kind, data)
    return ValidatedFile(
        filename=filename,
        kind=kind,
        content_type=_CANONICAL_MIME[kind],
        size_bytes=len(data),
    )


def _check_size(size: int, max_bytes: int) -> None:
    if size > max_bytes:
        raise FileTooLarge(f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")


def _check_not_empty(data: bytes) -> None:
    if not data:
        raise UnsupportedFileType("The uploaded file is empty.")


def _kind_from_extension(filename: str) -> FileKind:
    extension = PurePosixPath(filename).suffix.lower()
    kind = _KIND_BY_EXTENSION.get(extension)
    if kind is None:
        raise UnsupportedFileType("Only .pdf and .txt files are accepted.")
    return kind


def _check_mime(kind: FileKind, declared_mime: str | None) -> None:
    mime = (declared_mime or _GENERIC_MIME).split(";")[0].strip().lower()
    if mime not in _ALLOWED_MIME[kind]:
        raise UnsupportedFileType(f"Content type '{mime}' does not match a {kind.value} file.")


def _check_magic_bytes(kind: FileKind, data: bytes) -> None:
    if kind is FileKind.PDF and not data.startswith(_PDF_MAGIC):
        raise UnsupportedFileType("The file is named .pdf but is not a valid PDF.")
    if kind is FileKind.TXT and _NUL_BYTE in data:
        raise UnsupportedFileType("The file is named .txt but contains binary data.")