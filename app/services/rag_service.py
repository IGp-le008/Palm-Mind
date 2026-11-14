"""The custom RAG pipeline: booking routing, history, condensation, retrieval, and generation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.llm.base import ChatModel
from app.memory.base import ChatMemory, Turn
from app.services.booking_service import BookingService
from app.services.prompt_builder import build_answer_messages, build_condense_messages
from app.services.retriever import RetrievedChunk, Retriever


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """Result of one chat turn. `booking_status` is set only for booking turns."""

    session_id: str
    reply: str
    sources: list[RetrievedChunk]
    booking_status: str | None = None


class RagService:
    """Runs one chat turn, routing to the booking flow when it applies."""

    def __init__(
        self,
        *,
        chat: ChatModel,
        memory: ChatMemory,
        retriever: Retriever,
        booking: BookingService | None = None,
    ) -> None:
        self._chat = chat
        self._memory = memory
        self._retriever = retriever
        self._booking = booking

    async def answer(
        self,
        session_id: str,
        message: str,
        document_id: uuid.UUID | None = None,
    ) -> RagAnswer:
        """Answer one message. The session id must already be validated."""
        if self._booking is not None:
            booking_turn = await self._booking.handle_message(session_id, message)
            if booking_turn is not None:
                await self._memory.append_turn(
                    session_id, Turn(user=message, assistant=booking_turn.reply)
                )
                return RagAnswer(
                    session_id=session_id,
                    reply=booking_turn.reply,
                    sources=[],
                    booking_status=booking_turn.status,
                )

        history = await self._memory.get_turns(session_id)
        search_query = await self._standalone_query(history, message)
        chunks = await self._retriever.retrieve(search_query, document_id)
        reply = await self._chat.complete(build_answer_messages(history, chunks, message))
        await self._memory.append_turn(session_id, Turn(user=message, assistant=reply))
        return RagAnswer(session_id=session_id, reply=reply, sources=chunks)

    async def history(self, session_id: str) -> list[Turn]:
        """Return stored turns for the session, oldest first."""
        return await self._memory.get_turns(session_id)

    async def clear(self, session_id: str) -> None:
        """Forget the session's turns. Clearing an unknown session is not an error."""
        await self._memory.clear(session_id)

    async def _standalone_query(self, history: list[Turn], message: str) -> str:
        """Condense only when there is history. A first question is already standalone."""
        if not history:
            return message
        return await self._chat.complete(build_condense_messages(history, message))
