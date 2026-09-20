"""Replay evaluation: run clips through the real detector and rule engine.

This is the Phase III evaluation the brief asks for. It exercises the exact
production code path -- same ONNX session, same tracker, same rule engine, same
thresholds -- against one clip that should raise an event and one that should
not, and reports latency, throughput and PASS/FAIL.

Nothing here touches the database or the network, so it is safe to run
repeatedly and gives the README its performance numbers.

    python backend/scripts/make_fixtures.py     # once, to build the clips
    python backend/scripts/evaluate.py
    python backend/scripts/evaluate.py --video path/to/clip.mp4 --expect-min 1
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.events.rules import (  # noqa: E402
    EventType,
    RuleConfig,
    RuleState,
    evaluate as evaluate_rules,
)
from app.inference.detector import PersonDetector  # noqa: E402
from app.inference.tracker import IouTracker  # noqa: E402

FIXTURES = BACKEND_DIR / "fixtures"

#: The ROI the fixture clips were authored against. Kept beside the clips
#: rather than read from the database so evaluation is hermetic.
EVAL_ZONE = [(0.08, 0.55), (0.48, 0.55), (0.48, 1.0), (0.08, 1.0)]

#: Clips are replayed at their authored rate so wall-clock dwell thresholds
#: mean the same thing here as they do on a live 10 fps webcam stream.
ASSUMED_FPS = 10.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100.0 * len(ordered)) - 1))
    return ordered[idx]


@dataclass
class ScenarioResult:
    name: str
    frames: int
    frames_with_person: int
    events: int
    zone_events: int
    latencies: list[float]
    wall_seconds: float
    expectation: str
    passed: bool

    @property
    def mean_ms(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0


def run_clip(
    detector: PersonDetector,
    path: Path,
    *,
    name: str,
    expect_min: int,
    expect_max: int | None,
) -> ScenarioResult:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"could not open {path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or ASSUMED_FPS
    if fps <= 0:
        fps = ASSUMED_FPS

    tracker = IouTracker(
        iou_threshold=settings.track_iou_threshold,
        max_age_frames=settings.track_max_age_frames,
    )
    state = RuleState()
    config = RuleConfig(
        dwell_seconds=settings.dwell_seconds,
        hysteresis_grace_seconds=settings.hysteresis_grace_seconds,
        track_cooldown_seconds=settings.track_cooldown_seconds,
        zone_cooldown_seconds=settings.zone_cooldown_seconds,
        absence_seconds=settings.absence_seconds,
        absence_cooldown_seconds=settings.absence_cooldown_seconds,
    )

    latencies: list[float] = []
    frames = with_person = zone_events = total_events = 0
    started = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        detections, latency_ms = detector.detect(frame)
        latencies.append(latency_ms)
        frames += 1
        if detections:
            with_person += 1

        tracked = tracker.update(detections)
        # Virtual clock derived from the clip's frame rate: replay speed then
        # has no effect on whether a 2 s dwell threshold is met.
        now = frames / fps
        result = evaluate_rules(state, tracked, EVAL_ZONE, now, config)

        total_events += len(result.events)
        zone_events += sum(
            1
            for e in result.events
            if e.type is EventType.RESTRICTED_ZONE_ENTRY
        )

    capture.release()
    wall = time.perf_counter() - started

    passed = zone_events >= expect_min and (
        expect_max is None or zone_events <= expect_max
    )
    expectation = (
        f">= {expect_min}"
        if expect_max is None
        else (
            f"== {expect_min}"
            if expect_min == expect_max
            else f"{expect_min}..{expect_max}"
        )
    )

    return ScenarioResult(
        name=name,
        frames=frames,
        frames_with_person=with_person,
        events=total_events,
        zone_events=zone_events,
        latencies=latencies,
        wall_seconds=wall,
        expectation=expectation,
        passed=passed,
    )


def print_report(results: list[ScenarioResult]) -> None:
    print()
    print(f"Model     : {settings.model_name} {settings.model_version}")
    print(f"Provider  : CPUExecutionProvider")
    print(
        f"Thresholds: conf>={settings.conf_threshold} "
        f"dwell>={settings.dwell_seconds}s "
        f"grace={settings.hysteresis_grace_seconds}s "
        f"cooldown(track/zone)={settings.track_cooldown_seconds:.0f}/"
        f"{settings.zone_cooldown_seconds:.0f}s"
    )
    print(f"ROI       : {EVAL_ZONE}")
    print()

    header = (
        f"{'scenario':<26}{'frames':>7}{'person':>8}{'zone ev':>9}"
        f"{'expect':>9}{'mean ms':>9}{'p50':>7}{'p95':>7}{'fps':>7}  result"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        throughput = r.frames / r.wall_seconds if r.wall_seconds else 0.0
        print(
            f"{r.name:<26}{r.frames:>7}{r.frames_with_person:>8}"
            f"{r.zone_events:>9}{r.expectation:>9}"
            f"{r.mean_ms:>9.1f}{percentile(r.latencies, 50):>7.1f}"
            f"{percentile(r.latencies, 95):>7.1f}{throughput:>7.1f}"
            f"  {'PASS' if r.passed else 'FAIL'}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="evaluate a single clip")
    parser.add_argument("--expect-min", type=int, default=0)
    parser.add_argument("--expect-max", type=int, default=None)
    args = parser.parse_args()

    detector = PersonDetector(
        model_path=settings.model_path,
        person_class_id=settings.person_class_id,
        conf_threshold=settings.conf_threshold,
    )
    detector.load()
    if not detector.is_loaded:
        print(f"!!  {detector.load_error}", file=sys.stderr)
        return 1

    if args.video:
        scenarios = [
            (args.video, args.video.stem, args.expect_min, args.expect_max)
        ]
    else:
        scenarios = [
            (
                FIXTURES / "positive_zone_entry.mp4",
                "positive: enters zone",
                1,
                None,
            ),
            (
                FIXTURES / "negative_walk_by.mp4",
                "negative: passes outside",
                0,
                0,
            ),
        ]

    missing = [path for path, *_ in scenarios if not path.exists()]
    if missing:
        for path in missing:
            print(f"!!  missing clip: {path}", file=sys.stderr)
        print(
            "\n    Generate the fixtures first:\n"
            "      python backend/scripts/make_fixtures.py",
            file=sys.stderr,
        )
        return 1

    results = [
        run_clip(
            detector, path, name=name, expect_min=lo, expect_max=hi
        )
        for path, name, lo, hi in scenarios
    ]
    print_report(results)

    if all(r.passed for r in results):
        print("All scenarios behaved as expected.")
        return 0
    print("At least one scenario did NOT behave as expected.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
