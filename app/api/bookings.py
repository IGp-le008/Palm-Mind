"""HTTP routes for saved interview bookings."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import BookingDep
from app.schemas.booking import BookingListResponse, BookingResponse

router = APIRouter(prefix="/api/v1/bookings", tags=["bookings"])


@router.get("", response_model=BookingListResponse)
async def list_bookings(
    bookings: BookingDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> BookingListResponse:
    """Return one page of confirmed bookings, newest first."""
    rows = await bookings.list_bookings(limit=limit, offset=offset)
    return BookingListResponse(
        items=[BookingResponse.model_validate(row) for row in rows],
        limit=limit,
        offset=offset,
    )
