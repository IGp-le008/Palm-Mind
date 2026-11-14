# Architecture Decision Records

This document records the architectural decisions that shape Palm Mind RAG. The
decisions are written against the project constraints in `AGENTS.md` and the
completed implementation items in `TODO.md`.

## ADR-001: Use FastAPI with layered application boundaries

- **Status:** Accepted
- **Decision:** Use FastAPI for the two REST APIs and keep dependencies flowing
  from `app/api/` to services, repositories/providers, and infrastructure.
- **Context:** The service needs typed asynchronous HTTP endpoints while keeping
  ingestion, retrieval, booking, and persistence independently testable.
- **Implementation:** `app/api/` contains routers and dependency wiring only;
  `app/services/` contains business logic; `app/repositories/` contains SQL
  access; and `app/container.py` owns dependency construction. Protocols in the
  provider packages keep services independent of concrete implementations.
- **Consequences:** Business logic can be tested with fake providers, but the
  application requires explicit dependency wiring and protocol maintenance.

## ADR-002: Use PostgreSQL-compatible SQL persistence

- **Status:** Accepted
- **Decision:** Use PostgreSQL as the deployment database and support SQLite for
  quick local runs and tests where configured.
- **Context:** Document metadata and confirmed interview bookings require durable,
  transactional relational storage.
- **Implementation:** SQLAlchemy 2.x async sessions and Alembic migrations are
  used through `app/db/` and `migrations/`. The `Document` and
  `InterviewBooking` aggregates have separate repositories. Interview bookings
  enforce uniqueness on `(email, date, time)`.
- **Consequences:** Relational constraints protect data integrity, while local
  SQLite compatibility keeps development lightweight. Production deployments
  must use the configured PostgreSQL connection.

## ADR-003: Use Qdrant for vector storage

- **Status:** Accepted
- **Decision:** Use Qdrant as the vector database.
- **Context:** Document chunks need similarity search, optional document
  filtering, deterministic upserts, and deletion by document id.
- **Implementation:** `QdrantVectorStore` implements the `VectorStore` protocol,
  creates the collection when needed, stores the required chunk payload, and
  uses deterministic UUID5 point ids derived from the document id and chunk
  index.
- **Consequences:** Qdrant provides a simple Docker-based local deployment and
  preserves idempotent re-ingestion. Qdrant remains an external runtime
  dependency for end-to-end operation.

## ADR-004: Use Redis for chat memory and booking drafts

- **Status:** Accepted
- **Decision:** Store conversation history and partial booking state in Redis,
  never in process memory or files.
- **Context:** Chat state must work across workers and expire without manual
  cleanup.
- **Implementation:** Messages use `chat:{session_id}:messages` and are stored
  as a JSON list. Booking drafts use `chat:{session_id}:booking` and remain
  separate from message history. Both stores refresh a configurable TTL, and
  message history keeps the last configurable number of turns.
- **Consequences:** Sessions are shared across application instances and expire
  predictably. Redis availability is required for chat and booking flows.

## ADR-005: Use configurable OpenRouter-compatible model providers

- **Status:** Accepted
- **Decision:** Use OpenAI-compatible clients configured through environment
  settings, with OpenRouter model identifiers as the current defaults.
- **Context:** Embeddings and chat generation are external capabilities and must
  be replaceable without changing application services or hardcoding secrets.
- **Implementation:** Embeddings default to
  `nvidia/nemotron-3.5-embed-1b:free`; chat defaults to
  `nvidia/nemotron-3.5-lightning:free`. `EmbeddingProvider` and `ChatModel`
  protocols isolate provider details. URLs, model names, dimensions, and
  credentials are loaded by `app/core/config.py`.
- **Consequences:** Provider changes are configuration-level when the protocol
  remains compatible. External calls must be mocked in CI; paid APIs are never
  called by tests.

## ADR-006: Implement custom RAG orchestration

- **Status:** Accepted
- **Decision:** Keep retrieval, question condensation, prompt assembly, history
  handling, and generation as explicit application code.
- **Context:** The project requires transparent control over the RAG pipeline
  and forbids prebuilt QA-chain abstractions.
- **Implementation:** Each turn loads Redis history, condenses only when history
  exists, embeds the standalone query, searches Qdrant, builds a context-bound
  prompt, generates the answer, and appends the turn to Redis. The system prompt
  requires answers to use only retrieved context and to state when context is
  insufficient.
- **Consequences:** Pipeline behavior is explicit and unit-testable, but the
  project owns the orchestration, prompt contracts, and provider error handling.

## ADR-007: Support semantic and recursive chunking behind one protocol

- **Status:** Accepted
- **Decision:** Expose `semantic` and `recursive` strategies through the same
  `ChunkingStrategy` protocol and select them per ingestion request.
- **Context:** Different documents benefit from either fixed semantic windows or
  structure-aware splitting, while services should not depend on implementations.
- **Implementation:** Semantic chunks use a configurable 800-character window
  with 100-character overlap by default. Chunks that exceed the maximum size are
  recursively split. Recursive chunking prefers paragraphs, lines, sentences,
  and then character boundaries.
- **Consequences:** Clients can choose a strategy without changing the pipeline.
  Chunk counts and retrieval quality vary by document and strategy.

## ADR-008: Make ingestion idempotent and status-driven

- **Status:** Accepted
- **Decision:** Track ingestion with `pending -> processing -> completed` or
  `failed`, and use deterministic chunk point ids.
- **Context:** Uploads can fail during extraction, embedding, or vector writes;
  failures must be visible and safe to retry.
- **Implementation:** The ingestion service validates extension, MIME/magic bytes,
  and the configurable size limit; saves metadata; extracts and chunks text;
  embeds and upserts vectors; then updates status. Failures update the metadata
  row with a clear error. Deletion removes vectors before the SQL row.
- **Consequences:** Re-ingestion is deterministic and failed work is observable.
  A crash after a vector write and before the final status update can leave a
  document in `processing`; retry or deletion reconciles that state.

## ADR-009: Require LLM-driven, confirmed booking persistence

- **Status:** Accepted
- **Decision:** Detect booking intent and extract fields with structured LLM
  output, then persist only after validation and explicit user confirmation.
- **Context:** Booking intent cannot rely on keyword matching, and invented or
  incomplete appointment details must never be stored.
- **Implementation:** Required fields are non-empty `name`, validated `email`, a
  current-or-future ISO date, and a business-hours `HH:MM` time. The assistant
  asks only for missing fields, one or two at a time, stores partial state in
  Redis, summarizes the completed booking, and writes to SQL only after
  confirmation.
- **Consequences:** The flow is safer and auditable, but it requires an extra
  conversational confirmation step and robust handling of structured-model
  failures.

## ADR-010: Keep unresolved operational hardening visible

- **Status:** Open
- **Decision:** Track rate limiting or a simple LLM request timeout as a separate
  hardening item rather than silently treating it as complete.
- **Context:** The quality checklist in `TODO.md` still contains this unchecked
  item, while the rest of the documented architecture is implemented.
- **Consequence:** The service should not be considered fully production-hardened
  until the item is implemented, tested, and marked complete in `TODO.md`.
