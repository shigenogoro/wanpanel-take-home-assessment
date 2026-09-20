"""The safety-event rule engine.

This module is deliberately **pure**: it performs no I/O, opens no database
connection and never calls `time.time()` -- the caller passes `now` in. That is
what makes the rules unit-testable against synthetic detection sequences, and
what lets a reviewer change a threshold and see the effect without running a
camera.

Primary event -- `restricted_zone_entry`
    A tracked person's foot point stays inside the configured ROI polygon for
    `dwell_seconds` of continuous presence. Brief dropouts shorter than
    `hysteresis_grace_seconds` do not reset the timer, so a single missed frame
    or a momentary occlusion cannot restart the clock.

Secondary event -- `person_absence`
    No person is detected anywhere in frame for `absence_seconds` after at
    least one person has been seen.

Duplicate suppression works at two levels: a per-track cooldown stops the same
person re-alerting, and a per-zone cooldown catches the case where the tracker
loses and re-acquires someone under a fresh id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ..inference.tracker import TrackedDetection

Point = tuple[float, float]
Polygon = list[Point]


class EventType(StrEnum):
    RESTRICTED_ZONE_ENTRY = "restricted_zone_entry"
    PERSON_ABSENCE = "person_absence"
    MANUAL_TEST = "manual_test"


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray-casting point-in-polygon test.

    Works for any simple polygon, so a ROI is not restricted to a rectangle.
    """
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


@dataclass
class RuleConfig:
    """Snapshot of the tunables the engine needs. Mirrors `app.config`."""

    dwell_seconds: float = 2.0
    hysteresis_grace_seconds: float = 0.5
    track_cooldown_seconds: float = 15.0
    zone_cooldown_seconds: float = 10.0
    absence_seconds: float = 10.0
    absence_cooldown_seconds: float = 30.0


@dataclass
class TrackZoneState:
    """Per-person dwell bookkeeping for one zone."""

    inside_since: float | None = None
    last_inside_at: float | None = None
    last_fired_at: float | None = None


@dataclass
class RuleState:
    """All mutable state the engine carries between frames."""

    tracks: dict[int, TrackZoneState] = field(default_factory=dict)
    zone_last_fired_at: float | None = None
    last_person_seen_at: float | None = None
    absence_last_fired_at: float | None = None

    def reset(self) -> None:
        self.tracks.clear()
        self.zone_last_fired_at = None
        self.last_person_seen_at = None
        self.absence_last_fired_at = None


@dataclass
class EventDraft:
    """An event the rule decided to raise, before it is persisted.

    `reason` carries the exact numbers that triggered it. It is logged verbatim
    and stored with the event, so "why did this fire?" is always answerable
    after the fact.
    """

    type: EventType
    label: str
    confidence: float
    track_id: int | None
    bbox: tuple[float, float, float, float] | None
    reason: dict[str, object]


@dataclass
class ZoneEvaluation:
    """What the engine concluded about this frame."""

    events: list[EventDraft] = field(default_factory=list)
    inside_track_ids: list[int] = field(default_factory=list)
    max_dwell_seconds: float = 0.0


def evaluate(
    state: RuleState,
    detections: list[TrackedDetection],
    polygon: Polygon,
    now: float,
    config: RuleConfig,
) -> ZoneEvaluation:
    """Advance the state machine by one frame and return any events raised.

    `state` is mutated in place; `now` is a timestamp in seconds.
    """
    result = ZoneEvaluation()

    if detections:
        state.last_person_seen_at = now

    live_ids = {d.track_id for d in detections}

    for det in detections:
        track_state = state.tracks.setdefault(det.track_id, TrackZoneState())
        inside = point_in_polygon(det.foot, polygon)

        if inside:
            result.inside_track_ids.append(det.track_id)
            track_state.last_inside_at = now
            if track_state.inside_since is None:
                track_state.inside_since = now

            dwell = now - track_state.inside_since
            result.max_dwell_seconds = max(result.max_dwell_seconds, dwell)

            if dwell >= config.dwell_seconds and _may_fire(
                state, track_state, now, config
            ):
                result.events.append(
                    EventDraft(
                        type=EventType.RESTRICTED_ZONE_ENTRY,
                        label="person in restricted zone",
                        confidence=det.confidence,
                        track_id=det.track_id,
                        bbox=det.bbox,
                        reason={
                            "dwell_seconds": round(dwell, 3),
                            "dwell_threshold": config.dwell_seconds,
                            "confidence": det.confidence,
                            "foot_point": [round(v, 4) for v in det.foot],
                            "polygon_vertices": len(polygon),
                        },
                    )
                )
                track_state.last_fired_at = now
                state.zone_last_fired_at = now
                # Restart the dwell clock: continuous presence re-alerts once
                # the cooldown lapses, rather than firing once and going quiet.
                track_state.inside_since = now
        else:
            # Hysteresis: only clear the dwell timer once the person has been
            # outside for longer than the grace window.
            outside_for = (
                None
                if track_state.last_inside_at is None
                else now - track_state.last_inside_at
            )
            if (
                outside_for is not None
                and outside_for <= config.hysteresis_grace_seconds
            ):
                if track_state.inside_since is not None:
                    result.max_dwell_seconds = max(
                        result.max_dwell_seconds,
                        now - track_state.inside_since,
                    )
            else:
                track_state.inside_since = None
                track_state.last_inside_at = None

    _evaluate_absence(state, detections, now, config, result)

    # Forget tracks the tracker has retired, but keep any still in cooldown so
    # a recycled id cannot bypass duplicate suppression.
    state.tracks = {
        tid: ts
        for tid, ts in state.tracks.items()
        if tid in live_ids
        or (
            ts.last_fired_at is not None
            and now - ts.last_fired_at < config.track_cooldown_seconds
        )
    }
    return result


def _may_fire(
    state: RuleState,
    track_state: TrackZoneState,
    now: float,
    config: RuleConfig,
) -> bool:
    """Both cooldowns must be clear before a zone event is allowed."""
    if (
        track_state.last_fired_at is not None
        and now - track_state.last_fired_at < config.track_cooldown_seconds
    ):
        return False
    if (
        state.zone_last_fired_at is not None
        and now - state.zone_last_fired_at < config.zone_cooldown_seconds
    ):
        return False
    return True


def _evaluate_absence(
    state: RuleState,
    detections: list[TrackedDetection],
    now: float,
    config: RuleConfig,
    result: ZoneEvaluation,
) -> None:
    if detections or state.last_person_seen_at is None:
        return
    gap = now - state.last_person_seen_at
    if gap < config.absence_seconds:
        return
    if (
        state.absence_last_fired_at is not None
        and now - state.absence_last_fired_at < config.absence_cooldown_seconds
    ):
        return
    result.events.append(
        EventDraft(
            type=EventType.PERSON_ABSENCE,
            label="no person detected",
            confidence=1.0,
            track_id=None,
            bbox=None,
            reason={
                "seconds_since_last_person": round(gap, 3),
                "absence_threshold": config.absence_seconds,
            },
        )
    )
    state.absence_last_fired_at = now
