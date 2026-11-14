"""Prompt builder: labelled context, history order, and the condensation prompt."""

from __future__ import annotations

import uuid

from app.llm.base import ChatMessage
from app.memory.base import Turn
from app.services.prompt_builder import build_answer_messages, build_condense_messages
from app.services.retriever import RetrievedChunk

DOC = uuid.UUID("11111111-1111-1111-1111-111111111111")
CHUNK = RetrievedChunk(document_id=DOC, chunk_index=3, text="Refunds take 14 days.")


def test_context_is_labelled_with_its_source() -> None:
    messages = build_answer_messages([], [CHUNK], "How long do refunds take?")

    assert f"[{DOC}#3]" in messages[0].content
    assert "Refunds take 14 days." in messages[0].content
    assert messages[-1] == ChatMessage("user", "How long do refunds take?")


def test_history_is_replayed_in_order_before_the_question() -> None:
    history = [Turn("first", "reply one"), Turn("second", "reply two")]

    messages = build_answer_messages(history, [], "third")

    assert [m.role for m in messages[1:]] == ["user", "assistant", "user", "assistant", "user"]
    assert [m.content for m in messages[1:]] == [
        "first",
        "reply one",
        "second",
        "reply two",
        "third",
    ]


def test_empty_context_is_stated_explicitly() -> None:
    messages = build_answer_messages([], [], "anything")

    assert "(no excerpts found)" in messages[0].content


def test_condensation_prompt_contains_history_and_latest_message() -> None:
    messages = build_condense_messages(
        [Turn("What is the refund policy?", "14 days.")],
        "And the cost?",
    )

    assert messages[0].role == "system"
    assert "What is the refund policy?" in messages[1].content
    assert "Latest message: And the cost?" in messages[1].content
