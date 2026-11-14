# Architecture Decisions

## Storage and infrastructure

- **Vector database:** Qdrant
  - Stores document chunk embeddings and searchable payload metadata.
- **Chat memory:** Redis
  - Stores per-session conversation history and booking state with TTL support.
- **SQL database:** PostgreSQL
  - Stores document metadata and confirmed interview bookings.

## AI providers

- **Embedding model:** `nvidia/nemotron-3.5-embed-1b:free`
  - Used to create vectors for document chunks and user queries.
- **Chat model:** `nvidia/nemotron-3.5-lightning:free`
  - Used for response generation, query condensation, and structured booking intent extraction.

The model identifiers are kept in environment configuration so they can be changed without
modifying application code.

## Application architecture

The service uses FastAPI with explicit ingestion and RAG orchestration. Retrieval, prompt assembly, conversation history handling, and response generation remain application-level operations rather than a prebuilt question-answering chain.

The current implementation is organised as follows: `app/api/` contains routers; `app/services/` contains orchestration; `app/repositories/` contains SQL access; `app/vectorstore/`, `app/llm/`, `app/memory/`, `app/chunking/`, and `app/extraction/` contain protocols and implementations; `app/db/` contains models and async sessions; and `migrations/` contains Alembic migrations. `app/container.py` creates shared dependencies.

Configuration is read by `app/core/config.py` from environment variables, with optional dotenv loading for local development. The provider URL defaults to OpenRouter, while model names, vector dimensions, upload limits, retrieval settings, Redis TTL, and booking hours remain configurable.

## Phase 8 documentation decisions

- **Chunking:** semantic chunking uses embedding-based breakpoints; recursive chunking splits paragraphs, lines, sentences, and finally characters. Oversized semantic chunks are bounded by the recursive chunker.
- **Qdrant:** local Docker support, payload filtering, deterministic point upserts, and vector deletion keep retrieval simple and idempotent.
- **Redis memory:** chat turns and partial booking drafts use separate per-session keys with a refreshed TTL, so memory is shared across workers and expires cleanly.
- **Booking design:** typed LLM extraction is merged into Redis state, validated, summarized for explicit confirmation, and persisted only after confirmation. The SQL unique constraint prevents duplicate slots.


Ingestion runs synchronously within the upload request, so the response is completed or an error.
Delete removes vectors before the row, so a Qdrant failure can be retried.
A crash between the Qdrant write and the final status commit leaves a document in processing with its points present. A delete cleans both up.
The container sits at `app/container.py` and imports services. Keeping dependency wiring outside `app/core/` matches the current module structure.


## Trade-offs from phase 4
condensation runs only when history exists, memory is last-write-wins per session, and chat calls have no retry yet.
