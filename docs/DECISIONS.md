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


Ingestion runs synchronously within the upload request, so the response is completed or an error.
Delete removes vectors before the row, so a Qdrant failure can be retried.
A crash between the Qdrant write and the final status commit leaves a document in processing with its points present. A delete cleans both up.
The container sits in app/core and imports services. Moving it up a layer is possible later.