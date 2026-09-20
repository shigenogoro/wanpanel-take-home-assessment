"""Detector preprocessing/postprocessing and tracker identity tests.

The model itself is not under test here -- these cover the code around it,
which is where coordinate-convention bugs actually live.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import settings
from app.inference.detector import (
    Detection,
    InvalidFrameError,
    ModelNotLoadedError,
    PersonDetector,
)
from app.inference.tracker import IouTracker, iou


@pytest.fixture(scope="module")
def detector() -> PersonDetector:
    d = PersonDetector(
        model_path=settings.model_path,
        person_class_id=settings.person_class_id,
        conf_threshold=settings.conf_threshold,
    )
    d.load()
    return d


# --- postprocessing -------------------------------------------------------


def test_postprocess_keeps_only_confident_people(detector):
    """Class filter and confidence threshold, in one pass.

    The graph emits boxes as (ymin, xmin, ymax, xmax); everything downstream
    expects (x1, y1, x2, y2). This pins that conversion down.
    """
    boxes = np.array(
        [
            [0.10, 0.20, 0.80, 0.40],  # person, confident      -> kept
            [0.10, 0.20, 0.80, 0.40],  # person, low confidence -> dropped
            [0.10, 0.20, 0.80, 0.40],  # not a person           -> dropped
        ]
    )
    classes = np.array([1.0, 1.0, 17.0])
    scores = np.array([0.90, 0.10, 0.99])

    results = detector._postprocess(boxes, classes, scores)

    assert len(results) == 1
    kept = results[0]
    assert kept.label == "person"
    assert kept.confidence == pytest.approx(0.90)
    # (ymin=0.10, xmin=0.20, ymax=0.80, xmax=0.40) -> (0.20, 0.10, 0.40, 0.80)
    assert kept.bbox == pytest.approx((0.20, 0.10, 0.40, 0.80))


def test_postprocess_clips_boxes_to_the_frame(detector):
    boxes = np.array([[-0.20, -0.10, 1.30, 1.40]])
    results = detector._postprocess(boxes, np.array([1.0]), np.array([0.9]))
    x1, y1, x2, y2 = results[0].bbox
    assert all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2))


def test_foot_point_is_the_bottom_centre():
    """The zone rule tests this point, so its definition matters."""
    detection = Detection(label="person", confidence=0.9, bbox=(0.2, 0.1, 0.6, 0.8))
    assert detection.foot == pytest.approx((0.4, 0.8))


# --- decoding and failure modes -------------------------------------------


def test_decode_jpeg_round_trips_a_real_image():
    frame = np.full((120, 160, 3), 77, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    decoded = PersonDetector.decode_jpeg(buf.tobytes())
    assert decoded.shape == (120, 160, 3)


@pytest.mark.parametrize("payload", [b"", b"not a jpeg at all", b"\x00\x01\x02"])
def test_decode_rejects_junk(payload):
    with pytest.raises(InvalidFrameError):
        PersonDetector.decode_jpeg(payload)


def test_detect_without_a_loaded_model_raises():
    unloaded = PersonDetector(
        model_path="does-not-exist.onnx", person_class_id=1, conf_threshold=0.5
    )
    unloaded.load()
    assert not unloaded.is_loaded
    assert "fetch_model.py" in (unloaded.load_error or "")
    with pytest.raises(ModelNotLoadedError):
        unloaded.detect(np.zeros((10, 10, 3), dtype=np.uint8))


def test_detect_reports_latency_and_finds_nobody_in_a_blank_frame(detector):
    detections, latency_ms = detector.detect(np.zeros((240, 320, 3), np.uint8))
    assert detections == []
    assert latency_ms > 0


# --- tracker --------------------------------------------------------------


def test_iou_of_identical_and_disjoint_boxes():
    assert iou((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)) == pytest.approx(1.0)
    assert iou((0.0, 0.0, 0.1, 0.1), (0.9, 0.9, 1.0, 1.0)) == 0.0


def test_tracker_keeps_one_id_for_a_moving_person():
    tracker = IouTracker()
    ids = []
    for step in range(6):
        offset = step * 0.02
        detection = Detection(
            "person", 0.9, (0.1 + offset, 0.1, 0.3 + offset, 0.6)
        )
        ids.append(tracker.update([detection])[0].track_id)
    assert len(set(ids)) == 1


def test_tracker_separates_two_people():
    tracker = IouTracker()
    tracked = tracker.update(
        [
            Detection("person", 0.9, (0.05, 0.1, 0.25, 0.6)),
            Detection("person", 0.9, (0.70, 0.1, 0.90, 0.6)),
        ]
    )
    assert len({t.track_id for t in tracked}) == 2


def test_tracker_survives_short_occlusion_but_retires_stale_tracks():
    detection = Detection("person", 0.9, (0.1, 0.1, 0.3, 0.6))

    brief = IouTracker(max_age_frames=3)
    first = brief.update([detection])[0].track_id
    for _ in range(2):
        brief.update([])
    assert brief.update([detection])[0].track_id == first

    stale = IouTracker(max_age_frames=3)
    stale.update([detection])
    for _ in range(6):
        stale.update([])
    assert stale.update([detection])[0].track_id != first
