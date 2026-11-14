"""Request and response DTOs for the document endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentResponse(BaseModel):
    """Public view of a document. Never exposes ORM objects directly."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    chunking_strategy: str = Field(validation_alias="strategy")
    chunk_count: int
    status: str
    error: str | None
    created_at: datetime


class DocumentListResponse(BaseModel):
    """One page of documents."""

    items: list[DocumentResponse]
    limit: int
    offset: int
