"""Session id creation and validation for chat requests."""

from __future__ import annotations

import uuid

from app.core.exceptions import InvalidSessionId


def resolve_session_id(raw: str | None) -> str:
    """Return a normalised session id. Creates a new UUID4 when the client sent none."""
    if raw is None or not raw.strip():
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(raw.strip()))
    except ValueError as exc:
        raise InvalidSessionId("session_id must be a UUID.") from exc
