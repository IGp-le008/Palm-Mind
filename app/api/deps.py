"""FastAPI dependencies that expose container services to the routes."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from app.core.config import Settings
from app.container import Container
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService


def get_container(request: Request) -> Container:
    """Return the container that the lifespan handed to Starlette at startup."""
    return cast(Container, request.state.container)


ContainerDep = Annotated[Container, Depends(get_container)]


def get_settings(container: ContainerDep) -> Settings:
    return container.settings


def get_ingestion_service(container: ContainerDep) -> IngestionService:
    return container.ingestion_service


def get_document_service(container: ContainerDep) -> DocumentService:
    return container.document_service


SettingsDep = Annotated[Settings, Depends(get_settings)]
IngestionDep = Annotated[IngestionService, Depends(get_ingestion_service)]
DocumentDep = Annotated[DocumentService, Depends(get_document_service)]