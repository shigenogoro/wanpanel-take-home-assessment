"""SQLAlchemy ORM models.

Three tables:

* `zones`  -- the configurable ROI, stored as a normalised polygon so it is
  independent of camera resolution.
* `events` -- one row per safety event, carrying enough provenance
  (model name/version, confidence, source, the rule inputs) to answer "why did
  this fire, and what produced it?" long after the fact.
* `event_status_history` -- an append-only audit trail of the
  Open -> Acknowledged -> Resolved lifecycle.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

#: JSONB on Postgres (indexable, typed) but plain JSON elsewhere, so the test
#: suite can run against SQLite without a database server.
JsonColumn = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class EventStatus(StrEnum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


#: Legal lifecycle moves. Anything else is rejected with HTTP 422 rather than
#: silently written, so the state machine is enforced server-side.
ALLOWED_TRANSITIONS: dict[EventStatus, set[EventStatus]] = {
    EventStatus.OPEN: {EventStatus.ACKNOWLEDGED, EventStatus.RESOLVED},
    EventStatus.ACKNOWLEDGED: {EventStatus.RESOLVED},
    EventStatus.RESOLVED: set(),
}


class Zone(Base):
    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    #: [[x, y], ...] with every component normalised to [0, 1].
    polygon: Mapped[list] = mapped_column(JsonColumn, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class Event(Base):
    __tablename__ = "events"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    #: When the rule fired (model time), as distinct from when the row landed.
    event_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    type: Mapped[str] = mapped_column(String(50), nullable=False)
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    #: "webcam", "video:<name>" or "manual_test".
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)

    status: Mapped[str] = mapped_column(
        String(20), default=EventStatus.OPEN, nullable=False
    )

    zone_id: Mapped[int | None] = mapped_column(
        ForeignKey("zones.id", ondelete="SET NULL"), nullable=True
    )
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Normalised [x1, y1, x2, y2] of the person that triggered the event.
    bbox: Mapped[list | None] = mapped_column(JsonColumn, nullable=True)
    #: The rule inputs that caused the fire -- thresholds, dwell, foot point.
    reason: Mapped[dict | None] = mapped_column(JsonColumn, nullable=True)

    snapshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    acknowledged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    history: Mapped[list["EventStatusHistory"]] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
        order_by="EventStatusHistory.changed_at",
    )

    __table_args__ = (
        Index("ix_events_status_event_time", "status", "event_time"),
        Index("ix_events_event_time", "event_time"),
    )


class EventStatusHistory(Base):
    __tablename__ = "event_status_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("events.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    to_status: Mapped[str] = mapped_column(String(20), nullable=False)
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    event: Mapped[Event] = relationship(back_populates="history")
