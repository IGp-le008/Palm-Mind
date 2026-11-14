"""Chat model backed by an OpenAI-compatible API (OpenRouter by default)."""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.llm.base import ChatMessage

logger = get_logger(__name__)


class OpenAIChatModel:
    """Sends prompts to the chat completions endpoint."""

    def __init__(self, client: AsyncOpenAI, model: str, temperature: float) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    async def complete(self, messages: list[ChatMessage]) -> str:
        payload = [_to_param(message) for message in messages]
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=payload,
                temperature=self._temperature,
            )
        except OpenAIError as exc:
            logger.warning("Chat request failed: %s", type(exc).__name__)
            raise ProviderError("Chat provider request failed.") from exc

        reply = response.choices[0].message.content if response.choices else None
        if reply is None or not reply.strip():
            raise ProviderError("Chat provider returned an empty reply.")
        return reply.strip()


def _to_param(message: ChatMessage) -> ChatCompletionMessageParam:
    """Build the SDK's typed message for the role. Each role has its own TypedDict."""
    if message.role == "system":
        return ChatCompletionSystemMessageParam(role="system", content=message.content)
    if message.role == "assistant":
        return ChatCompletionAssistantMessageParam(role="assistant", content=message.content)
    return ChatCompletionUserMessageParam(role="user", content=message.content)
