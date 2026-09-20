"""Greedy IoU tracker: assigns stable ids to detections across frames.

Deliberately simple -- no Kalman filter, no appearance embedding. Its only job
is to give the event rule a stable handle on "the same person" so that a dwell
timer and a per-person cooldown mean something. A frame-to-frame IoU match is
sufficient at the frame rates and person counts this demo targets; the
limitation (id switches when two people cross, or after a long occlusion) is
documented in the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Detection

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass
class TrackedDetection:
    """A detection with the identity the tracker assigned to it."""

    track_id: int
    label: str
    confidence: float
    bbox: Box

    @property
    def foot(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


@dataclass
class _Track:
    track_id: int
    bbox: Box
    missed: int = 0


@dataclass
class IouTracker:
    iou_threshold: float = 0.30
    max_age_frames: int = 15

    _tracks: list[_Track] = field(default_factory=list)
    _next_id: int = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        """Match detections to existing tracks, then age out the unmatched."""
        pairs: list[tuple[float, int, int]] = []
        for d_idx, det in enumerate(detections):
            for t_idx, track in enumerate(self._tracks):
                score = iou(det.bbox, track.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, d_idx, t_idx))

        # Highest-IoU pairs win; each detection and each track is used once.
        pairs.sort(key=lambda p: p[0], reverse=True)
        det_to_track: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _score, d_idx, t_idx in pairs:
            if d_idx in det_to_track or t_idx in used_tracks:
                continue
            det_to_track[d_idx] = t_idx
            used_tracks.add(t_idx)

        results: list[TrackedDetection] = []
        for d_idx, det in enumerate(detections):
            t_idx = det_to_track.get(d_idx)
            if t_idx is None:
                track = _Track(track_id=self._next_id, bbox=det.bbox)
                self._next_id += 1
                self._tracks.append(track)
            else:
                track = self._tracks[t_idx]
                track.bbox = det.bbox
                track.missed = 0
            results.append(
                TrackedDetection(
                    track_id=track.track_id,
                    label=det.label,
                    confidence=det.confidence,
                    bbox=det.bbox,
                )
            )

        # Anything that did not end up in `results` went unseen this frame.
        seen_ids = {r.track_id for r in results}
        for track in self._tracks:
            if track.track_id not in seen_ids:
                track.missed += 1
        self._tracks = [
            t for t in self._tracks if t.missed <= self.max_age_frames
        ]
        return results
