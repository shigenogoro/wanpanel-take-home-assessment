"""Pydantic request/response models -- the validated API contract."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import EventStatus

Normalised = Annotated[float, Field(ge=0.0, le=1.0)]
BBox = Annotated[list[Normalised], Field(min_length=4, max_length=4)]


# --- detections ----------------------------------------------------------


class DetectionOut(BaseModel):
    track_id: int
    label: str
    confidence: float
    bbox: list[float] = Field(description="normalised [x1, y1, x2, y2]")
    foot: list[float] = Field(description="normalised [x, y] anchor point")
    in_zone: bool


class ZoneStateOut(BaseModel):
    zone_id: int | None
    occupied_track_ids: list[int]
    dwell_seconds: float


class InferenceResult(BaseModel):
    """What `/ws/inference` returns for each frame."""

    type: Literal["inference"] = "inference"
    frame_id: int
    ts: datetime
    latency_ms: float = Field(description="model time only, excludes transport")
    fps: float
    detections: list[DetectionOut]
    zone_state: ZoneStateOut
    events_fired: list[uuid.UUID]


class WsError(BaseModel):
    """A recoverable per-frame error. The socket stays open."""

    type: Literal["error"] = "error"
    code: str
    message: str


# --- events ---------------------------------------------------------------


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: uuid.UUID
    event_time: datetime
    created_at: datetime
    type: str
    label: str
    confidence: float
    source: str
    model_name: str
    model_version: str
    status: EventStatus
    zone_id: int | None
    track_id: int | None
    bbox: list[float] | None
    reason: dict | None
    has_snapshot: bool = False
    acknowledged_at: datetime | None
    resolved_at: datetime | None


class EventListOut(BaseModel):
    items: list[EventOut]
    total: int
    limit: int
    offset: int


class EventStatusUpdate(BaseModel):
    status: EventStatus


class ManualTestEventIn(BaseModel):
    """Debugging aid only. Events created here are tagged `manual_test` so they
    can never be mistaken for a model-derived alert."""

    label: str = Field(default="manual test event", max_length=200)


# --- zones ----------------------------------------------------------------


class ZoneOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    polygon: list[list[float]]
    active: bool
    updated_at: datetime


class ZoneUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    polygon: list[list[Normalised]] = Field(min_length=3)

    @field_validator("polygon")
    @classmethod
    def each_vertex_is_a_pair(
        cls, value: list[list[float]]
    ) -> list[list[float]]:
        for vertex in value:
            if len(vertex) != 2:
                raise ValueError("each polygon vertex must be exactly [x, y]")
        return value


# --- status / config ------------------------------------------------------


class ModelStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    version: str
    license: str
    source: str
    loaded: bool
    provider: str
    error: str | None = None


class StatusOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: ModelStatus
    database_connected: bool
    active_source: str | None
    inference_clients: int
    dashboard_clients: int
    metrics: dict
    zone: ZoneOut | None


class ConfigOut(BaseModel):
    """Live thresholds, so the dashboard and README cannot drift from code."""

    conf_threshold: float
    dwell_seconds: float
    hysteresis_grace_seconds: float
    track_cooldown_seconds: float
    zone_cooldown_seconds: float
    absence_seconds: float
    absence_cooldown_seconds: float
    track_iou_threshold: float
    track_max_age_frames: int
    max_frame_bytes: int
