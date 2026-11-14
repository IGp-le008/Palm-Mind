"""Application entry point: app factory, routers, exception handlers, and health check."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.exceptions import AppError
from app.core.logging import get_logger, setup_logging
from app.container import Container

from app.api.documents import router as documents_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[dict[str, Container]]:
    """Build shared clients at startup, expose them to requests, and release them on shutdown."""
    container = Container.create(settings)
    logger.info("Application started (env=%s)", settings.app_env)
    try:
        yield {"container": container}
    finally:
        await container.close()
        logger.info("Application stopped")


def register_exception_handlers(app: FastAPI) -> None:
    """Map domain exceptions to JSON responses. Clients never see stack traces."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("%s on %s: %s", exc.error_code, request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error_code, "message": exc.message},
        )


def register_routers(app: FastAPI) -> None:
    """Attach API routers. Each router is added here as it is implemented."""
    health_router = APIRouter(tags=["health"])

    @health_router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(health_router)
    app.include_router(documents_router)


def create_app() -> FastAPI:
    """Build the FastAPI application. Used by uvicorn as `app.main:app`."""
    setup_logging()

    app = FastAPI(
        title="Palm Mind RAG Backend",
        version="0.1.0",
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    register_routers(app)
    return app


app = create_app()
