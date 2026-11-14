"""Request and response DTOs for the chat endpoints."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """A user message. Omit `session_id` to start a new conversation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)
    document_id: uuid.UUID | None = None


class SourceRef(BaseModel):
    """The chunk an answer relied on. The text itself is not returned."""

    document_id: uuid.UUID
    chunk_index: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    sources: list[SourceRef]
    booking_status: str | None = None


class TurnResponse(BaseModel):
    user: str
    assistant: str


class HistoryResponse(BaseModel):
    session_id: str
    turns: list[TurnResponse]
