"""Unit tests for the safety-event rule engine.

These drive the engine with synthetic detection sequences and a fake clock, so
they are fast, deterministic and independent of the model. They are the
evidence for the threshold / timing-window / duplicate-suppression claims in
the README.
"""

from __future__ import annotations

import pytest

from app.events.rules import (
    EventType,
    RuleConfig,
    RuleState,
    evaluate,
    point_in_polygon,
)
from app.inference.tracker import TrackedDetection

# A ROI occupying the middle of the frame.
ZONE = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.9), (0.3, 0.9)]

CONFIG = RuleConfig(
    dwell_seconds=2.0,
    hysteresis_grace_seconds=0.5,
    track_cooldown_seconds=15.0,
    zone_cooldown_seconds=10.0,
    absence_seconds=10.0,
    absence_cooldown_seconds=30.0,
)


def person(track_id: int, foot_x: float, foot_y: float) -> TrackedDetection:
    """A detection whose foot point sits exactly at (foot_x, foot_y)."""
    half_w = 0.05
    return TrackedDetection(
        track_id=track_id,
        label="person",
        confidence=0.9,
        bbox=(foot_x - half_w, foot_y - 0.4, foot_x + half_w, foot_y),
    )


INSIDE = (0.5, 0.6)
OUTSIDE = (0.1, 0.6)


# --- geometry ------------------------------------------------------------


def test_point_in_polygon_basic():
    assert point_in_polygon((0.5, 0.6), ZONE)
    assert not point_in_polygon((0.1, 0.6), ZONE)
    assert not point_in_polygon((0.5, 0.95), ZONE)


def test_point_in_polygon_rejects_degenerate_polygon():
    assert not point_in_polygon((0.5, 0.5), [])
    assert not point_in_polygon((0.5, 0.5), [(0.0, 0.0), (1.0, 1.0)])


def test_point_in_polygon_handles_non_rectangular_roi():
    triangle = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    assert point_in_polygon((0.2, 0.2), triangle)
    assert not point_in_polygon((0.9, 0.9), triangle)


# --- primary event: restricted-zone entry --------------------------------


def test_fires_only_after_dwell_threshold():
    """The POSITIVE scenario: standing in the zone past the dwell window."""
    state = RuleState()
    fired: list[float] = []
    for step in range(30):  # 3.0 s at 10 fps
        now = step * 0.1
        result = evaluate(state, [person(1, *INSIDE)], ZONE, now, CONFIG)
        fired.extend(now for _ in result.events)

    assert fired, "a person standing in the zone must raise an event"
    # Nothing before 2.0 s; the first alert lands on the dwell boundary.
    assert fired[0] == pytest.approx(2.0, abs=0.05)


def test_walk_past_outside_zone_never_fires():
    """The NEGATIVE scenario: crossing the frame outside the ROI."""
    state = RuleState()
    events = []
    for step in range(50):
        now = step * 0.1
        x = 0.05 + step * 0.004  # travels 0.05 -> 0.25, never reaches 0.3
        result = evaluate(state, [person(1, x, 0.6)], ZONE, now, CONFIG)
        events.extend(result.events)
    assert events == []


def test_brief_visit_shorter_than_dwell_does_not_fire():
    state = RuleState()
    events = []
    for step in range(15):  # 1.5 s inside -- below the 2.0 s threshold
        now = step * 0.1
        result = evaluate(state, [person(1, *INSIDE)], ZONE, now, CONFIG)
        events.extend(result.events)
    for step in range(15, 40):  # then leaves
        now = step * 0.1
        result = evaluate(state, [person(1, *OUTSIDE)], ZONE, now, CONFIG)
        events.extend(result.events)
    assert events == []


# --- temporal smoothing ---------------------------------------------------


def test_hysteresis_survives_a_short_dropout():
    """A one-frame detection glitch must not restart the dwell timer."""
    state = RuleState()
    events = []
    for step in range(30):
        now = step * 0.1
        # At 1.0 s the person is briefly mis-placed outside for 0.2 s,
        # well within the 0.5 s grace window.
        outside = 10 <= step < 12
        det = person(1, *(OUTSIDE if outside else INSIDE))
        events.extend(evaluate(state, [det], ZONE, now, CONFIG).events)

    assert events, "a sub-grace dropout must not prevent the alert"


def test_long_absence_from_zone_resets_dwell():
    """A gap longer than the grace window starts the clock over."""
    state = RuleState()
    events = []
    for step in range(15):  # 1.5 s inside
        events.extend(
            evaluate(state, [person(1, *INSIDE)], ZONE, step * 0.1, CONFIG).events
        )
    for step in range(15, 30):  # 1.5 s outside -- far beyond the 0.5 s grace
        events.extend(
            evaluate(
                state, [person(1, *OUTSIDE)], ZONE, step * 0.1, CONFIG
            ).events
        )
    for step in range(30, 45):  # 1.5 s back inside: still under 2.0 s
        events.extend(
            evaluate(state, [person(1, *INSIDE)], ZONE, step * 0.1, CONFIG).events
        )
    assert events == []


# --- duplicate suppression ------------------------------------------------


def test_cooldown_suppresses_immediate_realert():
    state = RuleState()
    fired: list[float] = []
    for step in range(100):  # 10 s of continuous presence
        now = step * 0.1
        result = evaluate(state, [person(1, *INSIDE)], ZONE, now, CONFIG)
        fired.extend(now for _ in result.events)

    assert len(fired) == 1, (
        "continuous presence inside a 15 s track cooldown must alert once, "
        f"got {len(fired)} at {fired}"
    )


def test_second_person_is_held_by_the_zone_cooldown():
    """A different track still cannot alert inside the per-zone cooldown."""
    state = RuleState()
    for step in range(25):  # person 1 dwells and fires at 2.0 s
        evaluate(state, [person(1, *INSIDE)], ZONE, step * 0.1, CONFIG)

    events = []
    for step in range(25, 70):  # person 2 dwells during the 10 s zone cooldown
        now = step * 0.1
        events.extend(
            evaluate(
                state,
                [person(1, *INSIDE), person(2, 0.6, 0.7)],
                ZONE,
                now,
                CONFIG,
            ).events
        )
    assert events == [], "zone cooldown must absorb a second track's alert"


def test_realerts_after_cooldown_lapses():
    state = RuleState()
    fired: list[float] = []
    for step in range(400):  # 40 s of continuous presence
        now = step * 0.1
        result = evaluate(state, [person(1, *INSIDE)], ZONE, now, CONFIG)
        fired.extend(now for _ in result.events)

    assert len(fired) >= 2, "sustained intrusion should re-alert eventually"
    assert fired[1] - fired[0] >= CONFIG.track_cooldown_seconds


# --- secondary event: person absence --------------------------------------


def test_absence_fires_after_threshold_and_only_once():
    state = RuleState()
    evaluate(state, [person(1, *OUTSIDE)], ZONE, 0.0, CONFIG)

    events = []
    for step in range(1, 200):  # 20 s with nobody in frame
        events.extend(evaluate(state, [], ZONE, step * 0.1, CONFIG).events)

    absence = [e for e in events if e.type is EventType.PERSON_ABSENCE]
    assert len(absence) == 1
    assert absence[0].reason["seconds_since_last_person"] >= 10.0


def test_absence_does_not_fire_before_anyone_is_seen():
    """An idle camera at startup is not an absence event."""
    state = RuleState()
    events = []
    for step in range(200):
        events.extend(evaluate(state, [], ZONE, step * 0.1, CONFIG).events)
    assert events == []


# --- event payload --------------------------------------------------------


def test_event_carries_the_numbers_that_caused_it():
    state = RuleState()
    drafts = []
    for step in range(25):
        drafts.extend(
            evaluate(state, [person(1, *INSIDE)], ZONE, step * 0.1, CONFIG).events
        )

    draft = drafts[0]
    assert draft.type is EventType.RESTRICTED_ZONE_ENTRY
    assert draft.track_id == 1
    assert draft.confidence == 0.9
    assert draft.bbox is not None
    assert draft.reason["dwell_threshold"] == 2.0
    assert draft.reason["dwell_seconds"] >= 2.0
