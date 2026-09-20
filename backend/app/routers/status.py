"""Health, system status and live rule configuration."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_session, ping
from ..events.hub import hub
from ..inference.metrics import metrics
from ..models import Zone
from ..runtime import runtime
from ..schemas import ConfigOut, ModelStatus, StatusOut, ZoneOut

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness only -- deliberately does not touch the model or database."""
    return {"status": "ok"}


@router.get("/status", response_model=StatusOut)
def status(session: Session = Depends(get_session)) -> StatusOut:
    """Everything the dashboard needs to describe the system's condition.

    Always returns 200, even when the model failed to load or the database is
    down: a monitoring tool that goes dark when something breaks is worse than
    useless, so degradation is reported in the body instead.
    """
    zone = session.scalar(select(Zone).where(Zone.active.is_(True)))
    return StatusOut(
        model=ModelStatus(
            name=settings.model_name,
            version=settings.model_version,
            license=settings.model_license,
            source=settings.model_source,
            loaded=runtime.detector.is_loaded,
            provider=runtime.detector.provider,
            error=runtime.detector.load_error,
        ),
        database_connected=ping(),
        active_source=runtime.active_source,
        inference_clients=runtime.inference_clients,
        dashboard_clients=hub.client_count,
        metrics=metrics.snapshot(),
        zone=ZoneOut.model_validate(zone) if zone else None,
    )


@router.get("/config", response_model=ConfigOut)
def get_config() -> ConfigOut:
    """The thresholds actually in force, so docs and UI cannot drift."""
    return ConfigOut(
        conf_threshold=settings.conf_threshold,
        dwell_seconds=settings.dwell_seconds,
        hysteresis_grace_seconds=settings.hysteresis_grace_seconds,
        track_cooldown_seconds=settings.track_cooldown_seconds,
        zone_cooldown_seconds=settings.zone_cooldown_seconds,
        absence_seconds=settings.absence_seconds,
        absence_cooldown_seconds=settings.absence_cooldown_seconds,
        track_iou_threshold=settings.track_iou_threshold,
        track_max_age_frames=settings.track_max_age_frames,
        max_frame_bytes=settings.max_frame_bytes,
    )
