"""The configurable region of interest."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Zone
from ..runtime import runtime
from ..schemas import ZoneOut, ZoneUpdate

router = APIRouter(prefix="/api/zones", tags=["zones"])


@router.get("", response_model=list[ZoneOut])
def list_zones(session: Session = Depends(get_session)) -> list[Zone]:
    return list(session.scalars(select(Zone).order_by(Zone.id)).all())


@router.get("/active", response_model=ZoneOut)
def get_active_zone(session: Session = Depends(get_session)) -> Zone:
    zone = session.scalar(select(Zone).where(Zone.active.is_(True)))
    if zone is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="no active zone"
        )
    return zone


@router.put("/{zone_id}", response_model=ZoneOut)
def update_zone(
    zone_id: int,
    payload: ZoneUpdate,
    session: Session = Depends(get_session),
) -> Zone:
    """Redefine the ROI polygon.

    Dwell state is reset, because timers accumulated against the old geometry
    are meaningless once the boundary moves -- keeping them would let someone
    who was already inside trigger instantly against a zone they never entered.
    """
    zone = session.get(Zone, zone_id)
    if zone is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no zone with id {zone_id}",
        )

    zone.polygon = [[float(x), float(y)] for x, y in payload.polygon]
    if payload.name is not None:
        zone.name = payload.name
    session.commit()
    session.refresh(zone)

    runtime.rule_state.reset()
    return zone
