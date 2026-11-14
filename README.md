## Introduction 

A production-style backend AI assistant engineered with custom Retrieval-Augmented Generation (RAG), persistent conversational memory, and automated interview scheduling. Built from the ground up without high-level abstractions, the system ingests multi-format company documentation for context-aware Q&A while using structured LLM extraction to validate and persist interview bookings via a REST API.

This RAG will be responsible for:
1. Document processing and ingestion
2. Vector database and semantic search
3. Custom RAG for ground answering
4. Conversational memory
5. Interview appointment booking system

## Schema

```mermaid
flowchart TD
    U["Client / API Consumer"]

    U --> API["FastAPI Backend"]

    API --> ING["Document Ingestion API"]
    API --> CHAT["Conversational RAG API"]

    ING --> EXT["PDF / TXT Extraction"]
    EXT --> CH["Chunking Strategy"]
    CH --> EMB["Embedding Model"]
    EMB --> QD[("Qdrant")]
    ING --> PG[("PostgreSQL")]

    CHAT --> RED[("Redis")]
    RED --> INT["Intent Detection / Query Contextualization"]

    INT -->|Document Query| RET["Qdrant Retrieval"]
    RET --> PROMPT["Custom RAG Prompt"]
    PROMPT --> LLM["LLM"]
    LLM --> RESP["Response"]
    RESP --> RED

    INT -->|Interview Booking| COL["Collect Missing Fields"]
    COL --> VAL["Validate + Confirm"]
    VAL --> BOOK[("PostgreSQL Booking Table")]
    BOOK --> RESP
  ```

# Palm Mind RAG Backend

A FastAPI backend with two REST APIs:

1. **Document Ingestion API**: upload PDF/TXT files, chunk them with a selectable strategy, embed, store vectors in Qdrant, and save metadata in PostgreSQL.
2. **Conversational RAG API**: custom multi-turn RAG with Redis chat memory and LLM-driven interview booking.

No FAISS, no Chroma, no RetrievalQAChain, no UI.



---

## 1. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Standard for ML/LLM backends |
| Web framework | FastAPI | Async, typed, automatic OpenAPI docs |
| Validation and config | Pydantic v2, dotenv-backed `Settings` | Typed DTOs and environment config |
| SQL database | PostgreSQL (SQLite for quick local runs) | Metadata and bookings |
| ORM and migrations | SQLAlchemy 2.x (async), Alembic | Industry standard |
| Vector database | Qdrant (Docker) | Free, local, simple API. Pinecone/Weaviate/Milvus also allowed |
| Chat memory | Redis (`redis.asyncio`) | Fast TTL-based session storage |
| Embeddings | OpenRouter model configured by `EMBEDDING_MODEL` | Configurable provider |
| LLM | OpenRouter model configured by `CHAT_MODEL` | Response generation and structured booking extraction |
| PDF parsing | pypdf | Pure Python, no system deps |
| HTTP client | httpx | Async, used for provider calls |
| Testing | pytest, pytest-asyncio, httpx.AsyncClient | Standard |
| Lint and format | ruff | Fast, replaces flake8/isort/black |
| Type checking | mypy (strict) | Required by the brief's typing standard |
| Dependency management | uv | Fast, lockfile-based |
| Containers | Docker Compose | Reproducible Postgres, Redis, Qdrant |


---

## 2. Project structure

```
palm-mind-rag/
├── app/
│   │── container.py           # Dependency wiring
│   ├── main.py                    # FastAPI app factory, routers, exception handlers
│   ├── api/
│   │   ├── deps.py                # FastAPI dependencies (services, sessions)
│   │   ├── documents.py           # /api/v1/documents routes
│   │   ├── chat.py                # /api/v1/chat routes
│   │   └── bookings.py            # /api/v1/bookings routes
│   ├── core/
│   │   ├── config.py            # Settings (pydantic-settings)
│   │   ├── exceptions.py          # Domain exceptions + HTTP mapping
│   │   └── logging.py
│   ├── schemas/
│   │   ├── documents.py           # Upload/response DTOs
│   │   ├── chat.py                # Chat request/response DTOs
│   │   └── booking.py             # Booking DTOs and validation
│   ├── services/
│   │   ├── ingestion_service.py   # Orchestrates the ingestion pipeline
│   │   ├── rag_service.py         # Custom RAG: condense, retrieve, prompt, generate
│   │   ├── retriever.py           # Embed query + vector search
│   │   ├── prompt_builder.py      # System, history, context prompts
│   │   ├── booking_service.py     # Booking state, validation, persistence
│   │   └── chat_session.py        # Session id handling
│   ├── repositories/
│   │   ├── document_repository.py
│   │   └── booking_repository.py
│   ├── vectorstore/
│   │   ├── base.py                # VectorStore Protocol
│   │   └── qdrant_store.py        # QdrantVectorStore
│   ├── llm/
│   │   ├── base.py                # EmbeddingProvider, ChatModel Protocols
│   │   ├── openai_embeddings.py
│   │   └── openai_chat.py
│   ├── memory/
│   │   ├── base.py                # ChatMemory Protocol
│   │   └── redis_memory.py        # RedisChatMemory
│   ├── chunking/
│   │   ├── base.py                # ChunkingStrategy Protocol, Chunk dataclass
│   │   ├── semantic.py            # Strategy 1: semantic aware chunking
│   │   ├── recursive.py           # Strategy 2: paragraph/sentence-aware
│   │   └── factory.py
│   ├── extraction/
│   │   ├── pdf_extractor.py
│   │   └── txt_extractor.py
│   └── db/
│       ├── base.py                # DeclarativeBase
│       ├── models.py              # Document, InterviewBooking
│       └── session.py             # Engine and session factory
├── migrations/                    # Alembic
├── tests/
│   ├── unit/
│   │   ├── test_chunking.py
│   │   ├── test_prompt_builder.py
│   │   ├── test_booking_validation.py
│   │   └── test_memory.py
│   └── integration/
│       ├── test_documents_api.py
│       └── test_chat_api.py
├── scripts/
│   └── demo.http                  # Sample requests for the full flow
├── docs/
│   └── DECISIONS.md               # Design decisions and trade-offs
├── AGENTS.md
├── README.md
├── docker-compose.yml
├── .env.example
├── .gitignore
├── pyproject.toml
├── ruff.toml
└── alembic.ini
```

---

## 3. Requirements

### System
- Python 3.11 or 3.12
- Docker and Docker Compose (for Postgres, Redis, Qdrant)
- `uv` package manager (`pip install uv` or see docs.astral.sh/uv)
- Git
- An OpenRouter-compatible API key (chat and embedding keys may be separate)

### Python dependencies

Runtime:
- `fastapi`, `uvicorn[standard]`, `python-multipart` (file uploads)
- `pydantic`, `pydantic-settings`, `email-validator`
- `sqlalchemy[asyncio]`, `asyncpg`, `alembic`
- `redis`
- `qdrant-client`
- `openai`
- `pypdf`
- `httpx`
- `tenacity` (retry and backoff)

Dev:
- `pytest`, `pytest-asyncio`, `pytest-cov`
- `mypy`, `types-redis` (if needed), `ruff`
- `aiosqlite` (optional, for fast tests)

### Environment variables

| Variable | Example | Purpose |
|---|---|---|
| `APP_ENV` | `local` | Environment name |
| `DATABASE_URL` | `postgresql+asyncpg://app:app@localhost:5432/ragdb` | SQL connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Chat memory |
| `QDRANT_URL` | `http://localhost:6333` | Vector DB |
| `QDRANT_COLLECTION` | `documents` | Collection name |
| `CHAT_API_KEY` | `...` | Chat model access |
| `EMBEDDING_API_KEY` | `...` | Embedding model access; falls back to `CHAT_API_KEY` |
| `BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible provider URL |
| `EMBEDDING_MODEL` | `nvidia/nemotron-3.5-embed-1b:free` | Embedding model |
| `EMBEDDING_DIM` | `2048` | Must match the configured embedding model |
| `CHAT_MODEL` | `nvidia/nemotron-3.5-lightning:free` | Generation model |
| `MAX_UPLOAD_MB` | `10` | Upload size limit |
| `CHAT_HISTORY_TURNS` | `10` | Turns kept in Redis |
| `CHAT_TTL_SECONDS` | `86400` | Session expiry |
| `CHUNK_SIZE` | `800` | Maximum chunk size |
| `CHUNK_OVERLAP` | `100` | Chunk overlap |
| `EMBEDDING_BATCH_SIZE` | `32` | Embeddings per provider request |
| `RETRIEVAL_K` | `5` | Number of chunks retrieved |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.0` | Minimum retrieval score |
| `LLM_TIMEOUT_SECONDS` | `30` | Provider request timeout |
| `BUSINESS_HOURS_START` | `09:00` | Booking window start |
| `BUSINESS_HOURS_END` | `18:00` | Booking window end |

---

## 4. Setup

### 4.1 Clone and install

```bash
git clone <your-repo-url> palm-mind-rag
cd palm-mind-rag
cp .env.example .env          # then fill in OPENAI_API_KEY
uv sync                       # installs runtime and dev deps from pyproject.toml
```

### 4.2 Start infrastructure

```bash
docker compose up -d          # Postgres, Redis, Qdrant
docker compose ps             # all three should be "running"
```

### 4.3 Run migrations

```bash
uv run alembic upgrade head
```

### 4.4 Start the API

```bash
uv run uvicorn app.main:app --reload --port 8000
```

- Health check: `GET http://localhost:8000/health`
- Interactive docs: `http://localhost:8000/docs`

### 4.5 Quality checks

```bash
uv run ruff format .
uv run ruff check .
uv run mypy app
uv run pytest -q
uv run pytest --cov=app/services --cov=app/chunking
```

---

## 5. API reference

### 5.1 Document ingestion

**Upload a document**

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@./sample.pdf" \
  -F "chunking_strategy=recursive"
```

`chunking_strategy` is `semantic` or `recursive`.

Response `201 Created`:
```json
{
  "id": "3f2c9e1a-7b4d-4c1e-9a8f-2d1b6e0c5a11",
  "filename": "sample.pdf",
  "status": "completed",
  "chunking_strategy": "recursive",
  "chunk_count": 42
}
```

**Get document status**
```bash
curl http://localhost:8000/api/v1/documents/3f2c9e1a-7b4d-4c1e-9a8f-2d1b6e0c5a11
```

**List documents**
```bash
curl "http://localhost:8000/api/v1/documents?limit=20&offset=0"
```

**Delete a document** (removes SQL row and Qdrant points)
```bash
curl -X DELETE http://localhost:8000/api/v1/documents/3f2c9e1a-7b4d-4c1e-9a8f-2d1b6e0c5a11
```

### 5.2 Conversational RAG and booking

**Ask a question (starts a new session)**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the refund policy?"}'
```

Response:
```json
{
  "session_id": "b7e1...",
  "reply": "According to the document, refunds are processed within 14 days.",
  "sources": [{"document_id": "3f2c...", "chunk_index": 5}],
  "booking_status": null
}
```

**Follow-up in the same session** (uses Redis memory)
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "b7e1...", "message": "And how long does it take to get the money back?"}'
```

**Book an interview**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "b7e1...", "message": "I want to book an interview"}'
# assistant asks for name, email, date, time; then asks for confirmation
```

**Other chat endpoints**
- `GET /api/v1/chat/{session_id}/history`
- `DELETE /api/v1/chat/{session_id}` (clears Redis memory)
- `GET /api/v1/bookings` (lists stored bookings)

**List bookings**
```bash
curl "http://localhost:8000/api/v1/bookings?limit=20&offset=0"
```

Response:
```json
{"items": [], "limit": 20, "offset": 0}
```

---

## 6. How the pieces fit

**Ingestion:** upload -> validate -> save metadata (`pending`) -> extract text -> chunk (selected strategy) -> embed in batches -> upsert to Qdrant with payload -> mark `completed`. Any failure marks `failed` with the error message.

**RAG turn:** load last N turns from Redis -> if history exists, condense the question into a standalone query -> embed -> search Qdrant (optionally filtered by `document_id`) -> build prompt (system + history + context + question) -> call LLM -> append turn to Redis -> return reply and sources.

**Booking turn:** the LLM extracts booking fields and intent from the message -> merge with partial booking state in Redis -> validate -> if anything is missing, ask for it -> if complete, show summary and ask to confirm -> on "yes", insert into `interview_bookings` and clear the booking state.

See `docs/DECISIONS.md` for trade-offs.

