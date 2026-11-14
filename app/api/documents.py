"""HTTP routes for document ingestion and management."""

from __future__ import annotations

import uuid
from pathlib import PurePosixPath
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Response, UploadFile, status

from app.api.deps import DocumentDep, IngestionDep, SettingsDep
from app.chunking.factory import ChunkingStrategyKind
from app.schemas.documents import DocumentListResponse, DocumentResponse

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=DocumentResponse)
async def upload_document(
    ingestion: IngestionDep,
    documents: DocumentDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File()],
    chunking_strategy: Annotated[ChunkingStrategyKind, Form()],
) -> DocumentResponse:
    """Upload, extract, chunk, embed, and index a .pdf or .txt file. Runs synchronously."""
    # Read one byte past the limit so an oversized file is detected without buffering it all.
    data = await file.read(settings.max_upload_bytes + 1)
    filename = PurePosixPath(file.filename or "").name
    result = await ingestion.ingest(
        filename=filename,
        declared_mime=file.content_type,
        data=data,
        strategy=chunking_strategy,
    )
    document = await documents.get(result.document_id)
    return DocumentResponse.model_validate(document)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: uuid.UUID, documents: DocumentDep) -> DocumentResponse:
    """Return metadata and status for one document."""
    document = await documents.get(document_id)
    return DocumentResponse.model_validate(document)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    documents: DocumentDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListResponse:
    """Return one page of documents, newest first."""
    rows = await documents.list(limit=limit, offset=offset)
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(row) for row in rows],
        limit=limit,
        offset=offset,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, documents: DocumentDep) -> Response:
    """Remove the document's vectors and its metadata row."""
    await documents.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)