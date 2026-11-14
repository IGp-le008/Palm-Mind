"""Domain exceptions.

Each exception carries the HTTP status code and a stable machine-readable error code.
A single FastAPI exception handler (registered in app/main.py) converts them into responses,
so routes never build error responses by hand and clients never see stack traces.
"""

from __future__ import annotations

from typing import ClassVar


class AppError(Exception):
    """Base class for all domain errors."""

    status_code: ClassVar[int] = 500
    error_code: ClassVar[str] = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFileType(AppError):
    """Uploaded file is not a supported .pdf or .txt file."""

    status_code: ClassVar[int] = 415
    error_code: ClassVar[str] = "unsupported_file_type"


class FileTooLarge(AppError):
    """Uploaded file exceeds the configured MAX_UPLOAD_MB limit."""

    status_code: ClassVar[int] = 413
    error_code: ClassVar[str] = "file_too_large"


class IngestionFailed(AppError):
    """Text extraction, chunking, embedding, or storage failed for a document."""

    status_code: ClassVar[int] = 500
    error_code: ClassVar[str] = "ingestion_failed"


class SessionNotFound(AppError):
    """No chat session exists for the given session id."""

    status_code: ClassVar[int] = 404
    error_code: ClassVar[str] = "session_not_found"


class BookingValidationError(AppError):
    """Booking fields are missing or invalid, or the requested slot is unavailable."""

    status_code: ClassVar[int] = 422
    error_code: ClassVar[str] = "booking_validation_error"


class ProviderError(AppError):
    """An upstream model provider (OpenRouter, embeddings, or chat) failed."""

    status_code: ClassVar[int] = 502
    error_code: ClassVar[str] = "provider_error"
