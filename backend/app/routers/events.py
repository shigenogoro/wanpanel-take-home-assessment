"""Event listing, detail, lifecycle transitions and snapshots."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..db import get_session
from ..events import service
from ..events.hub import hub
from ..models import Event, EventStatus
from ..schemas import (
    EventListOut,
    EventOut,
    EventStatusUpdate,
    ManualTestEventIn,
)

router = APIRouter(prefix="/api/events", tags=["events"])


def _get_or_404(session: Session, event_id: uuid.UUID) -> Event:
    event = session.get(Event, event_id)
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no event with id {event_id}",
        )
    return event


@router.get("", response_model=EventListOut)
def list_events(
    status_filter: EventStatus | None = Query(default=None, alias="status"),
    type_filter: str | None = Query(default=None, alias="type"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
) -> EventListOut:
    rows, total = service.list_events(
        session,
        status=status_filter,
        event_type=type_filter,
        limit=limit,
        offset=offset,
    )
    return EventListOut(
        items=[service.to_out(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{event_id}", response_model=EventOut)
def get_event(
    event_id: uuid.UUID, session: Session = Depends(get_session)
) -> EventOut:
    return service.to_out(_get_or_404(session, event_id))


@router.patch("/{event_id}", response_model=EventOut)
async def update_event_status(
    event_id: uuid.UUID,
    payload: EventStatusUpdate,
    session: Session = Depends(get_session),
) -> EventOut:
    """Move an event along its lifecycle.

    Illegal transitions are rejected with 422 rather than silently applied.
    """
    event = _get_or_404(session, event_id)
    try:
        updated = service.update_status(session, event, payload.status)
    except service.InvalidStatusTransition as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    out = service.to_out(updated)
    await hub.broadcast(
        {"type": "event.updated", "payload": out.model_dump(mode="json")}
    )
    return out


@router.post(
    "/test", response_model=EventOut, status_code=status.HTTP_201_CREATED
)
async def create_test_event(
    payload: ManualTestEventIn, session: Session = Depends(get_session)
) -> EventOut:
    """Debugging tool only.

    The brief requires the *primary* event to come from model output; this
    endpoint exists to exercise the dashboard's event plumbing and tags every
    row it writes as `manual_test` so the two can never be confused.
    """
    event = service.create_manual_test_event(session, payload.label)
    out = service.to_out(event)
    await hub.broadcast(
        {"type": "event.created", "payload": out.model_dump(mode="json")}
    )
    return out


@router.get("/{event_id}/snapshot")
def get_snapshot(
    event_id: uuid.UUID, session: Session = Depends(get_session)
) -> FileResponse:
    event = _get_or_404(session, event_id)
    if not event.snapshot_path or not Path(event.snapshot_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no snapshot stored for this event",
        )
    return FileResponse(event.snapshot_path, media_type="image/jpeg")
