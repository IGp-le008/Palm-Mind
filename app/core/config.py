"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    app_env: str = os.getenv("APP_ENV", "local").lower()

    # Model providers (OpenRouter)
    chat_api_key: str = os.getenv("CHAT_API_KEY", "")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    base_url: str = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")
    chat_model: str = os.getenv("CHAT_MODEL", "nvidia/nemotron-3.5-lightning:free")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nvidia/nemotron-3.5-embed-1b:free")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "2048"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.4"))

    # PostgreSQL
    database_url: str = os.getenv("DATABASE_URL", "")

    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    chat_history_turns: int = int(os.getenv("CHAT_HISTORY_TURNS", "10"))
    chat_ttl_seconds: int = int(os.getenv("CHAT_TTL_SECONDS", "86400"))

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "documents")

    # Ingestion
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    semantic_breakpoint_percentile: int = int(os.getenv("SEMANTIC_BREAKPOINT_PERCENTILE", "95"))

    # Retrieval
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))

    # Interview booking
    business_hours_start: time = _time(os.getenv("BUSINESS_HOURS_START", "09:00"))
    business_hours_end: time = _time(os.getenv("BUSINESS_HOURS_END", "18:00"))

    def __post_init__(self) -> None:
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise RuntimeError("LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL.")

        if self.chunk_overlap >= self.chunk_size:
            raise RuntimeError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        if self.business_hours_end <= self.business_hours_start:
            raise RuntimeError("BUSINESS_HOURS_END must be later than BUSINESS_HOURS_START.")

    @property
    def effective_embedding_key(self) -> str:
        return self.embedding_api_key or self.chat_api_key

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def validate_for_chat(self) -> None:
        if not self.chat_api_key:
            raise RuntimeError(
                "CHAT_API_KEY is missing. Copy .env.example to .env and set your key."
            )

    def validate_for_embeddings(self) -> None:
        if not self.effective_embedding_key:
            raise RuntimeError("Set EMBEDDING_API_KEY or CHAT_API_KEY in .env.")

        if self.embedding_dim <= 0:
            raise RuntimeError("EMBEDDING_DIM must match the embedding model's vector size.")

    def validate_for_storage(self) -> None:
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is missing in .env.")

        if not self.redis_url:
            raise RuntimeError("REDIS_URL is missing in .env.")


settings = Settings()
