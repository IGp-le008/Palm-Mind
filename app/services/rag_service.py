"""The custom RAG pipeline: history, condensation, retrieval, prompt, generation, memory."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.llm.base import ChatModel
from app.memory.base import ChatMemory, Turn
from app.services.prompt_builder import build_answer_messages, build_condense_messages
from app.services.retriever import RetrievedChunk, Retriever


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """Result of one chat turn."""

    session_id: str
    reply: str
    sources: list[RetrievedChunk]


class RagService:
    """Runs one chat turn. Booking routing is added in Phase 5."""

    def __init__(self, *, chat: ChatModel, memory: ChatMemory, retriever: Retriever) -> None:
        self._chat = chat
        self._memory = memory
        self._retriever = retriever

    async def answer(
        self,
        session_id: str,
        message: str,
        document_id: uuid.UUID | None = None,
    ) -> RagAnswer:
        """Answer one message. The session id must already be validated."""
        history = await self._memory.get_turns(session_id)
        search_query = await self._standalone_query(history, message)
        chunks = await self._retriever.retrieve(search_query, document_id)
        reply = await self._chat.complete(build_answer_messages(history, chunks, message))
        await self._memory.append_turn(session_id, Turn(user=message, assistant=reply))
        return RagAnswer(session_id=session_id, reply=reply, sources=chunks)

    async def _standalone_query(self, history: list[Turn], message: str) -> str:
        """Condense only when there is history. A first question is already standalone."""
        if not history:
            return message
        return await self._chat.complete(build_condense_messages(history, message))
