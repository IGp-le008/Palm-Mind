# TODO

Work top to bottom. Tick items as they are completed (`- [x]`). Each item should be a single reviewable change.

## Phase 0: Setup and planning
- [x] Confirm submission requirements with the recruiter (repo link vs zip, deadline, demo video, Postman collection)
- [x] Create GitHub repo `palm-mind-rag` (private until submission)
- [x] Choose Python version (3.11 or 3.12) and install `uv`
- [x] Initialise project with `uv init` and create `pyproject.toml` with the dependency groups listed in README
- [x] Add `.gitignore`, `.env.example`, `ruff.toml`, `mypy` config (strict)
- [x] Write `docker-compose.yml` for PostgreSQL, Redis, and Qdrant
- [x] Create `docs/DECISIONS.md` and record the stack choices (Qdrant, Redis, PostgreSQL, embedding and LLM provider)
- [x] Verify all three containers start and are reachable

## Phase 1: Core scaffolding
- [x] `app/core/config.py`: `Settings` via pydantic-settings, with validation for all env vars
- [x] `app/core/logging.py`: structured logging setup
- [x] `app/core/exceptions.py`: domain exceptions (`UnsupportedFileType`, `FileTooLarge`, `IngestionFailed`, `SessionNotFound`, `BookingValidationError`, `ProviderError`)
- [x] `app/main.py`: FastAPI app factory, router registration, exception handlers, `/health` endpoint
- [x] `app/core/container.py`: dependency wiring (settings, DB session, Redis client, vector store, providers)
- [x] Database engine and session factory (`app/db/session.py`)
- [x] Alembic initialised, first migration created

## Phase 2: Persistence layer
- [x] ORM model `Document` (id, filename, content_type, size_bytes, sha256, strategy, chunk_count, status, error, created_at, updated_at)
- [x] ORM model `InterviewBooking` (id, session_id, name, email, date, time, status, created_at) with unique (email, date, time)
- [x] `DocumentRepository` with create, update_status, get_by_id, list
- [x] `BookingRepository` with create, exists, list_by_session
- [x] Repository unit tests against a test database

## Phase 3: Document ingestion
- [x] Text extraction for `.txt` (UTF-8 with fallback encoding detection)
- [x] Text extraction for `.pdf` (pypdf; handle empty or scanned PDFs with a clear error)
- [x] File validation: extension, MIME, magic bytes, size limit
- [x] `ChunkingStrategy` protocol and `Chunk` dataclass
- [x] `SemanticChunker` (size and overlap configurable) with unit tests
- [x] `RecursiveChunker` (paragraph, sentence, then character fallback) with unit tests
- [x] Chunker factory that maps the request value to an implementation
- [x] `EmbeddingProvider` protocol and first implementation (OpenAI `nvidia/nemotron-3.5-embed-1b:free` or local `sentence-transformers`)
- [x] `VectorStore` protocol and `QdrantVectorStore` (create collection if missing, upsert, search with optional document filter, delete by document id)
- [x] `IngestionService` orchestrating validate -> save metadata (pending) -> extract -> chunk -> embed -> upsert -> mark completed or failed
- [x] Batch embedding (configurable batch size) with retry and backoff on provider errors
- [x] `POST /api/v1/documents` endpoint (multipart upload, `chunking_strategy` form field) returning document id and status
- [x] `GET /api/v1/documents/{id}` endpoint returning metadata and status
- [x] `GET /api/v1/documents` endpoint with pagination
- [x] `DELETE /api/v1/documents/{id}` removes SQL row and vector points
- [x] Integration tests for upload of a txt and a pdf, for both strategies, and for invalid input

## Phase 4: Custom RAG pipeline
- [x] `ChatModel` protocol and implementation (`Nvidia` `nvidia/nemotron-3.5-lightning:free` or OpenAI / Groq / Gemini equivalent)
- [x] `ChatMemory` protocol and `RedisChatMemory` (append turn, get last N turns, TTL refresh, clear)
- [x] Session id generation and validation
- [x] Query condensation prompt: rewrite follow-up question into a standalone query using history
- [x] Retriever service: embed query, search, optional `document_id` filter, score threshold
- [x] Prompt builder: system prompt, history block, context block with source markers, user question
- [x] `RagService.answer(session_id, message, document_id?)` tying the steps together
- [x] Unit tests for prompt builder and condensation logic with a fake ChatModel
- [x] Return sources (document id, chunk index) in the response

## Phase 5: Interview booking
- [x] Booking tool schema (name, email, date, time, plus `intent` and `is_confirmed`) defined as a Pydantic model
- [x] LLM extraction call that returns structured JSON (function calling or JSON mode) with validation and one retry on failure
- [x] Booking state stored in Redis per session, merged across turns
- [x] Validation rules: name non-empty, email format, date >= today, time within business hours
- [x] Missing-field prompting: ask only for what is missing, one or two fields at a time
- [x] Confirmation step: summarise the booking and wait for explicit yes/no
- [x] `BookingService.save` writes to SQL only after confirmation; handles duplicate slot with a clear message
- [x] Routing in `RagService`: if booking intent is active, go to booking flow, otherwise RAG flow
- [x] Unit tests for validation and state merge
- [x] Integration test: multi-turn booking conversation ends with a row in `interview_bookings`

## Phase 6: Conversational RAG API
- [x] `POST /api/v1/chat` accepts `{session_id?, message, document_id?}` and returns `{session_id, reply, sources, booking_status}`
- [x] `GET /api/v1/chat/{session_id}/history` returns stored turns (optional but useful for demo)
- [x] `DELETE /api/v1/chat/{session_id}` clears Redis memory
- [x] `GET /api/v1/bookings` lists bookings (optional, helps the reviewer verify storage)
- [ ] Rate limiting or simple request timeout on LLM calls
- [x] Integration tests: multi-turn Q&A where the second question depends on the first

## Phase 7: Quality and hardening
- [x] `mypy --strict app` passes with zero errors
- [x] `ruff check` and `ruff format --check` pass
- [x] Test coverage >= 80% on `app/services` and `app/chunking`
- [x] Remove any leftover TODOs, prints, or dead code
- [x] Verify no forbidden libraries: `grep -RniE "faiss|chroma|RetrievalQA" .` returns nothing
- [x] Verify no UI code: no templates, no Streamlit, no static frontend
- [x] Secrets only via env; `.env` is git-ignored
- [x] Input size limits and request validation on every endpoint

## Phase 8: Documentation and submission
- [x] README: overview, architecture diagram, setup, env vars, run, test
- [x] README: example `curl` for every endpoint with sample responses
- [x] `docs/DECISIONS.md`: chunking strategy trade-offs, why Qdrant, why Redis memory design, booking design
- [x] Postman collection or `scripts/demo.http` covering the full flow
- [x] Record a short screen demo (optional, check requirements)
- [x] Fresh-clone test: follow README from scratch on a clean machine or folder
- [x] Final review of AGENTS.md compliance
- [x] Submit to Palm Mind AI before the deadline
