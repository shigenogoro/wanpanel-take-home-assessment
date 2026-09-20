"""Process-wide inference runtime: the model, the tracker and the rule state.

Held in one place so `/api/status` can report it and the WebSocket handler can
use it without re-loading a 29 MB graph per connection.
"""

from __future__ import annotations

import logging
import threading

from .config import settings
from .events.rules import RuleConfig, RuleState
from .inference.detector import PersonDetector
from .inference.tracker import IouTracker

logger = logging.getLogger(__name__)


def rule_config_from_settings() -> RuleConfig:
    return RuleConfig(
        dwell_seconds=settings.dwell_seconds,
        hysteresis_grace_seconds=settings.hysteresis_grace_seconds,
        track_cooldown_seconds=settings.track_cooldown_seconds,
        zone_cooldown_seconds=settings.zone_cooldown_seconds,
        absence_seconds=settings.absence_seconds,
        absence_cooldown_seconds=settings.absence_cooldown_seconds,
    )


class Runtime:
    """Single-stream runtime.

    This demo intentionally supports one active camera at a time: a second
    inference client would otherwise interleave frames into the same tracker
    and corrupt every dwell timer. The WebSocket handler enforces that and
    says so explicitly rather than silently degrading.
    """

    def __init__(self) -> None:
        self.detector = PersonDetector(
            model_path=settings.model_path,
            person_class_id=settings.person_class_id,
            conf_threshold=settings.conf_threshold,
        )
        self.tracker = IouTracker(
            iou_threshold=settings.track_iou_threshold,
            max_age_frames=settings.track_max_age_frames,
        )
        self.rule_state = RuleState()
        self.rule_config = rule_config_from_settings()
        self.active_source: str | None = None
        self.inference_clients = 0
        self.lock = threading.Lock()

    def load_model(self) -> None:
        self.detector.load()

    def begin_stream(self, source: str) -> None:
        """Reset per-stream state so a new session starts from a clean slate."""
        self.tracker.reset()
        self.rule_state.reset()
        self.active_source = source
        self.inference_clients += 1
        logger.info("stream started source=%s", source)

    def end_stream(self) -> None:
        self.inference_clients = max(0, self.inference_clients - 1)
        if self.inference_clients == 0:
            self.active_source = None
            self.tracker.reset()
            self.rule_state.reset()
        logger.info("stream ended")


runtime = Runtime()
