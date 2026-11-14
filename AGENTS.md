# AGENTS.md

Instructions for AI coding agents (Claude Code, Cursor, Copilot, Codex, etc.) working in this repository.
Read this file fully before making any change.

## 1. Project goal

Backend-only Python service with two REST APIs, built with FastAPI:

1. **Document Ingestion API**: upload `.pdf` / `.txt`, extract text, chunk with one of two selectable strategies, embed, store vectors in a vector DB (Qdrant), store metadata in a SQL DB (PostgreSQL / SQLite).
2. **Conversational RAG API**: custom RAG pipeline (no `RetrievalQAChain`), multi-turn chat with memory stored in Redis, and LLM-driven interview booking (name, email, date, time) persisted to the SQL DB.

## 2. Hard constraints (never violate)

- **Forbidden**: FAISS, Chroma, `RetrievalQAChain` (and any `*QAChain` / `RetrievalQA` class). Do not add a UI of any kind (no HTML templates, no Streamlit, no frontend).
- Vector store must be one of: Pinecone, Qdrant, Weaviate, Milvus. Default choice: **Qdrant**.
- Chat memory must use **Redis**. Do not use in-process dicts or files for chat history.
- The RAG orchestration must be written by us: retrieval, prompt assembly, history handling and generation are explicit functions/classes in `app/services/`. LangChain loaders/splitters/SDK wrappers are allowed, but the chain itself must not be a prebuilt QA chain.
- Code must be modular and follow industry standards for typing and annotations: every function and method has full type hints, `mypy --strict` must pass.
- Never hardcode secrets. Current configuration is loaded by `app/core/config.py` from dotenv/environment values into the `Settings` dataclass.

## 3. Architecture rules

Layered structure. Dependencies flow downward only:

```
api  ->  services  ->  repositories / vectorstore / llm / chunking  ->  db / core
```

- `app/api/`: FastAPI routers and dependency wiring only. No business logic, no direct DB or vector calls.
- `app/schemas/`: Pydantic models for request/response DTOs. Never return ORM objects directly.
- `app/services/`: business logic (ingestion orchestration, RAG pipeline, booking logic, chat session handling).
- `app/repositories/`: SQL access (SQLAlchemy 2.x style). One repository per aggregate.
- `app/vectorstore/`: an abstract `VectorStore` protocol plus a `QdrantVectorStore` implementation.
- `app/llm/`: `EmbeddingProvider` and `ChatModel` protocols plus concrete providers.
- `app/chunking/`: `ChunkingStrategy` protocol with two implementations (see section 5).
- `app/extraction/`: text extraction protocol plus PDF and TXT implementations.
- `app/core/`: settings, logging, and exceptions. Dependency wiring is in `app/container.py`.
- Depend on abstractions (`Protocol`s) in services, not on concrete providers.

## 4. Coding standards

- Python 3.11+.
- Format and lint with **ruff** (`ruff format`, `ruff check --fix`). Keep lines <= 100 chars.
- Type check with **mypy --strict**. No `Any` unless justified in a comment. No bare `dict` / `list`; use `dict[str, str]`, `list[Chunk]`, etc.
- Use `from __future__ import annotations` only if needed for forward refs.
- Async all the way for I/O (FastAPI routes, Redis via `redis.asyncio`, SQLAlchemy async engine, httpx for HTTP). Do not mix blocking calls into async routes; use `asyncio.to_thread` or `run_in_threadpool` for CPU-bound work like PDF parsing.
- Docstrings: Google style on public classes and functions.
- Errors: define domain exceptions in `app/core/exceptions.py`; map them to HTTP responses in one exception handler. Never leak stack traces to clients.
- Logging: use `logging.getLogger(__name__)`. No `print`. Never log raw document content, API keys, or full emails at INFO level.
- Config: read everything through `app/core/config.py` (`Settings` class); application modules do not read environment variables directly.
- Keep functions small (aim for < 40 lines) and classes single-purpose.
- No dead code, no commented-out code, no TODO without a linked item in `TODO.md`.

## 5. Domain-specific rules

### Ingestion
- Accepted types: `application/pdf` and `text/plain`. Validate by extension AND MIME/magic bytes. Enforce a max size (default 10 MB, configurable).
- Two chunking strategies, selected by a request field `chunking_strategy` with values `semantic` and `recursive`. Both must be implemented behind the same `ChunkingStrategy` protocol.
  - `semantic`: semantic token/character window with overlap (default 800 chars, 100 overlap). - **Semantic Chunk Size Limit**: Semantic chunks exceeding the maximum size **must** be recursively split using the recursive chunker.
  - `recursive` (paragraph/sentence-aware): split on `\n\n`, then `\n`, then `. `, falling back to characters, merging up to a max size.
- Each chunk gets a deterministic id (`uuid5(document_id, chunk_index)`) so re-ingestion is idempotent.
- Vector payload stores: `document_id`, `chunk_index`, `text`, `source_filename`, `strategy`. Metadata row in SQL stores: document id, filename, content type, size, sha256, strategy, chunk count, status, created_at.
- Ingestion status flow: `pending -> processing -> completed | failed`. Failures must update the status row and return a clear error.

### RAG and chat
- Pipeline order per user turn: load history from Redis -> condense question into standalone query (LLM, only if history exists) -> embed query -> vector search (top_k, optional document filter) -> build prompt with history + context + question -> generate -> append turn to Redis.
- Redis keys: `chat:{session_id}:messages`, stored as a JSON list. Keep the last N turns (default 10). Set a TTL (default 24h), refreshed on each turn.
- Session ids are server-generated UUID4 if the client does not supply one.
- The system prompt must instruct the model to answer only from context and say so when context is insufficient.

### Interview booking
- Booking intent is detected by the LLM (tool/function calling or structured output with a `booking_intent` flag), not by keyword matching.
- Required fields: `name` (non-empty), `email` (validated with Pydantic `EmailStr`), `date` (ISO `YYYY-MM-DD`, must be today or later), `time` (`HH:MM` 24h, within configured business hours).
- The assistant asks only for the missing fields, one or two at a time. Never invent values.
- Persist only after all fields are valid AND the user confirms the summary.
- Store in SQL table `interview_bookings` with `session_id`, the four fields, `status`, `created_at`. Add a unique constraint on (`email`, `date`, `time`) to prevent duplicates.
- Booking state (partial fields) lives in Redis under `chat:{session_id}:booking`, not in the message history.

### Testing
- pytest with `pytest-asyncio`. Unit tests for chunking, schemas, booking validation, and prompt building. Integration tests for the two APIs using `httpx.AsyncClient`, with Qdrant, Postgres and Redis run via Docker Compose.
- Mock external LLM and embedding providers in tests. Never call a paid API in CI.
- Coverage target: >= 80% on `app/services` and `app/chunking`.

## 6. Workflow for agents

1. Read `TODO.md` and pick the first unchecked item. Work on one item at a time.
2. Before writing code, check whether a protocol or utility already exists.
3. After each change run: `ruff format . && ruff check . && mypy app && pytest -q`.
4. Update `TODO.md` (tick the item) in the same change.
5. Keep commits small and conventional: `feat(ingestion): add recursive chunker`, `fix(chat): ...`, `test(booking): ...`.
6. Never commit `.env`, secrets, uploaded files, or `__pycache__`.
7. If a requirement is ambiguous, prefer the simplest option that meets the written constraints and note the decision in `docs/DECISIONS.md`.

## 7. Definition of done

- Both endpoints work end to end with `docker compose up` followed by `uvicorn`.
- `ruff`, `mypy --strict`, and `pytest` all pass.
- README contains setup steps and example `curl` calls for every endpoint.
- No forbidden libraries (FAISS, Chroma, RetrievalQAChain) appear anywhere in the code (`grep -R` check).
- No UI code exists in the repo.
