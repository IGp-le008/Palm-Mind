# scripts/check_qdrant.py  (temporary, delete after checking)
import asyncio
import uuid

from app.core.config import settings
from app.vectorstore.base import ChunkPayload, ChunkRecord, chunk_point_id
from app.vectorstore.qdrant_store import QdrantVectorStore


async def main() -> None:
    store = QdrantVectorStore.from_settings(settings)
    await store.ensure_collection()
    doc = uuid.uuid4()
    vector = [0.1] * settings.embedding_dim
    record = ChunkRecord(
        point_id=chunk_point_id(doc, 0),
        vector=vector,
        payload=ChunkPayload(str(doc), 0, "hello", "a.txt", "recursive"),
    )
    await store.upsert([record])
    hits = await store.search(vector, limit=3, document_id=doc)
    print(hits)
    await store.delete_document(doc)


asyncio.run(main())