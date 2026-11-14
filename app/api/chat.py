"""HTTP routes for conversational RAG."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Response, status

from app.api.deps import RagDep
from app.core.exceptions import SessionNotFound
from app.schemas.chat import ChatRequest, ChatResponse, HistoryResponse, SourceRef, TurnResponse
from app.services.chat_session import resolve_session_id

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

SessionPath = Annotated[str, Path(description="Session id returned by POST /api/v1/chat")]


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, rag: RagDep) -> ChatResponse:
    """Answer a message, or continue the interview booking when one is in progress."""
    session_id = resolve_session_id(request.session_id)
    answer = await rag.answer(session_id, request.message, request.document_id)
    return ChatResponse(
        session_id=answer.session_id,
        reply=answer.reply,
        sources=[
            SourceRef(document_id=chunk.document_id, chunk_index=chunk.chunk_index)
            for chunk in answer.sources
        ],
        booking_status=answer.booking_status,
    )


@router.get("/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: SessionPath, rag: RagDep) -> HistoryResponse:
    """Return the stored turns. Expired or never-created sessions return 404."""
    normalised = resolve_session_id(session_id)
    turns = await rag.history(normalised)
    if not turns:
        raise SessionNotFound("No chat history exists for this session.")
    return HistoryResponse(
        session_id=normalised,
        turns=[TurnResponse(user=turn.user, assistant=turn.assistant) for turn in turns],
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_chat(session_id: SessionPath, rag: RagDep) -> Response:
    """Clear the session's memory. Idempotent, so clearing twice is not an error."""
    await rag.clear(resolve_session_id(session_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
