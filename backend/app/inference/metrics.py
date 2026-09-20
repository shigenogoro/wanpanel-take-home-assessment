"""Rolling runtime metrics for the inference loop.

Kept in memory on purpose: these are liveness numbers for the dashboard and the
README's performance table, not business data worth a database round-trip per
frame.

Two different latencies are tracked because they answer different questions:

* `inference_ms` -- model time alone. Comparable across machines, and the
  number to quote when discussing the model.
* `round_trip_ms` -- what the browser actually waited, including JPEG encode,
  network hop and decode. The number a user would feel.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Exact enough for a 120-sample window."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100.0 * len(ordered)) - 1))
    return ordered[idx]


@dataclass
class MetricsTracker:
    window: int = 120

    _inference_ms: deque[float] = field(default_factory=deque)
    _frame_times: deque[float] = field(default_factory=deque)
    _lock: Lock = field(default_factory=Lock)

    frames_processed: int = 0
    frames_rejected: int = 0
    started_at: float = field(default_factory=time.monotonic)
    last_frame_at: float | None = None

    def __post_init__(self) -> None:
        self._inference_ms = deque(maxlen=self.window)
        self._frame_times = deque(maxlen=self.window)

    def record_frame(self, inference_ms: float) -> None:
        with self._lock:
            now = time.monotonic()
            self._inference_ms.append(inference_ms)
            self._frame_times.append(now)
            self.frames_processed += 1
            self.last_frame_at = now

    def record_rejected(self) -> None:
        with self._lock:
            self.frames_rejected += 1

    def reset(self) -> None:
        with self._lock:
            self._inference_ms.clear()
            self._frame_times.clear()
            self.frames_processed = 0
            self.frames_rejected = 0
            self.started_at = time.monotonic()
            self.last_frame_at = None

    @property
    def fps(self) -> float:
        """Throughput over the sample window, not 1/latency.

        These diverge whenever the client is the bottleneck, which it usually
        is -- the browser caps how fast it sends frames.
        """
        with self._lock:
            if len(self._frame_times) < 2:
                return 0.0
            span = self._frame_times[-1] - self._frame_times[0]
            if span <= 0:
                return 0.0
            return (len(self._frame_times) - 1) / span

    def snapshot(self) -> dict[str, float | int | None]:
        with self._lock:
            samples = list(self._inference_ms)
        mean = sum(samples) / len(samples) if samples else 0.0
        return {
            "frames_processed": self.frames_processed,
            "frames_rejected": self.frames_rejected,
            "inference_ms_mean": round(mean, 2),
            "inference_ms_p50": round(_percentile(samples, 50), 2),
            "inference_ms_p95": round(_percentile(samples, 95), 2),
            "fps": round(self.fps, 2),
            "uptime_seconds": round(time.monotonic() - self.started_at, 1),
            "seconds_since_last_frame": (
                None
                if self.last_frame_at is None
                else round(time.monotonic() - self.last_frame_at, 2)
            ),
        }


metrics = MetricsTracker()
