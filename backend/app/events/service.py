"""Turning rule output into durable, reviewable events.

This is the only place that writes to the `events` table. It owns three
things the rule engine deliberately does not: persistence, the annotated
snapshot on disk, and the broadcast to connected dashboards.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import ALLOWED_TRANSITIONS, Event, EventStatus, EventStatusHistory
from ..schemas import EventOut
from .rules import EventDraft, EventType

logger = logging.getLogger(__name__)


class InvalidStatusTransition(ValueError):
    """Raised when a caller asks for a lifecycle move the state machine forbids."""


def snapshot_path_for(event_id: uuid.UUID) -> str:
    settings.snapshot_dir.mkdir(parents=True, exist_ok=True)
    return str(settings.snapshot_dir / f"{event_id}.jpg")


def write_snapshot(
    event_id: uuid.UUID,
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float] | None,
    polygon: list[list[float]] | None,
) -> str | None:
    """Save an annotated JPEG showing why the event fired.

    Best-effort: a snapshot failure must never lose the event itself, so this
    logs and returns None rather than raising.
    """
    try:
        annotated = frame_bgr.copy()
        height, width = annotated.shape[:2]

        if polygon and len(polygon) >= 3:
            points = np.array(
                [[int(x * width), int(y * height)] for x, y in polygon],
                dtype=np.int32,
            )
            cv2.polylines(annotated, [points], True, (0, 0, 255), 2)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(
                annotated,
                (int(x1 * width), int(y1 * height)),
                (int(x2 * width), int(y2 * height)),
                (0, 165, 255),
                2,
            )

        path = snapshot_path_for(event_id)
        if cv2.imwrite(path, annotated):
            return path
        logger.warning("cv2.imwrite returned False for %s", path)
    except Exception:  # noqa: BLE001 - never lose an event over a snapshot
        logger.exception("failed to write snapshot for event %s", event_id)
    return None


def create_event(
    session: Session,
    draft: EventDraft,
    *,
    source: str,
    zone_id: int | None,
    frame_bgr: np.ndarray | None = None,
    polygon: list[list[float]] | None = None,
) -> Event:
    """Persist one rule-raised event, with provenance and an audit entry."""
    event_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    snapshot = None
    if frame_bgr is not None:
        snapshot = write_snapshot(event_id, frame_bgr, draft.bbox, polygon)

    event = Event(
        id=event_id,
        event_time=now,
        created_at=now,
        type=str(draft.type),
        label=draft.label,
        confidence=draft.confidence,
        source=source,
        model_name=settings.model_name,
        model_version=settings.model_version,
        status=EventStatus.OPEN,
        zone_id=zone_id,
        track_id=draft.track_id,
        bbox=list(draft.bbox) if draft.bbox else None,
        reason=draft.reason,
        snapshot_path=snapshot,
    )
    session.add(event)
    session.add(
        EventStatusHistory(
            event_id=event_id,
            from_status=None,
            to_status=EventStatus.OPEN,
            changed_at=now,
        )
    )
    session.commit()
    session.refresh(event)

    # The rule inputs are logged verbatim so any alert can be explained later.
    logger.info(
        "event fired type=%s id=%s track=%s conf=%.3f source=%s reason=%s",
        draft.type,
        event_id,
        draft.track_id,
        draft.confidence,
        source,
        draft.reason,
    )
    return event


def create_manual_test_event(session: Session, label: str) -> Event:
    """A debugging aid, tagged so it can never pass as a model-derived alert."""
    draft = EventDraft(
        type=EventType.MANUAL_TEST,
        label=label,
        confidence=1.0,
        track_id=None,
        bbox=None,
        reason={"origin": "manual debug endpoint, not model output"},
    )
    return create_event(session, draft, source="manual_test", zone_id=None)


def update_status(
    session: Session, event: Event, new_status: EventStatus
) -> Event:
    """Move an event through Open -> Acknowledged -> Resolved.

    Rejects illegal moves (and no-ops) instead of writing them, so the
    lifecycle is guaranteed server-side rather than trusted from the client.
    """
    current = EventStatus(event.status)
    if new_status not in ALLOWED_TRANSITIONS[current]:
        raise InvalidStatusTransition(
            f"cannot move an event from '{current}' to '{new_status}'; "
            f"allowed from here: "
            f"{sorted(ALLOWED_TRANSITIONS[current]) or 'nothing (terminal)'}"
        )

    now = datetime.now(timezone.utc)
    event.status = new_status
    if new_status is EventStatus.ACKNOWLEDGED:
        event.acknowledged_at = now
    elif new_status is EventStatus.RESOLVED:
        event.resolved_at = now

    session.add(
        EventStatusHistory(
            event_id=event.id,
            from_status=current,
            to_status=new_status,
            changed_at=now,
        )
    )
    session.commit()
    session.refresh(event)
    logger.info("event %s %s -> %s", event.id, current, new_status)
    return event


def to_out(event: Event) -> EventOut:
    """ORM row -> API model, computing `has_snapshot` from the stored path."""
    payload = EventOut.model_validate(event)
    payload.has_snapshot = bool(event.snapshot_path)
    return payload


def list_events(
    session: Session,
    *,
    status: EventStatus | None = None,
    event_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Event], int]:
    filters = []
    if status is not None:
        filters.append(Event.status == status)
    if event_type is not None:
        filters.append(Event.type == event_type)

    total = session.scalar(
        select(func.count()).select_from(Event).where(*filters)
    )
    rows = (
        session.scalars(
            select(Event)
            .where(*filters)
            .order_by(Event.event_time.desc())
            .limit(limit)
            .offset(offset)
        )
        .unique()
        .all()
    )
    return list(rows), int(total or 0)
