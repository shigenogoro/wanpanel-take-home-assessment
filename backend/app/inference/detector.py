"""Person detector: ONNX Runtime + SSD-MobileNetV1 (COCO), CPU only.

Why this model
--------------
* Apache-2.0 and published by the ONNX organisation itself, so the download URL
  is stable and the licence is unencumbered (YOLOv8n is AGPL-3.0 and has no
  official pre-exported ONNX artifact).
* ~25 ms/frame on a laptop CPU at 640x480 -- roughly 3x the headroom of
  YOLOv8n, which matters because the whole pipeline is CPU-bound.
* Non-max suppression is baked into the graph and boxes come out already
  normalised to [0, 1], so postprocessing is a class filter and a threshold
  rather than a hand-rolled NMS.

Trade-off, stated plainly: SSD-MobileNetV1 is weaker than YOLOv8n on small or
distant people. For a zone-entry demo with a person a few metres from the
camera that is an acceptable exchange for licence cleanliness and speed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted without a usable ONNX session."""


class InvalidFrameError(ValueError):
    """Raised when an incoming frame cannot be decoded as an image."""


@dataclass(frozen=True)
class Detection:
    """One detected person, in normalised frame coordinates.

    `bbox` is (x1, y1, x2, y2) with each component in [0, 1], so the frontend
    can draw it at any canvas size without knowing the capture resolution.
    `foot` is the bottom-centre anchor -- the point the zone rule tests,
    because a person's feet are where they actually *are* on the floor plane.
    """

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]

    @property
    def foot(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


class PersonDetector:
    """Wraps an ONNX Runtime session and exposes a single `detect` call."""

    def __init__(
        self,
        model_path: Path,
        person_class_id: int,
        conf_threshold: float,
    ) -> None:
        self.model_path = Path(model_path)
        self.person_class_id = person_class_id
        self.conf_threshold = conf_threshold
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self.load_error: str | None = None

    # -- lifecycle --------------------------------------------------------

    def load(self) -> None:
        """Load the model. Records, rather than raises, a failure reason so the
        API can stay up and report *why* inference is unavailable."""
        if not self.model_path.exists():
            self.load_error = (
                f"Model file not found at {self.model_path}. "
                "Run: python backend/scripts/fetch_model.py"
            )
            logger.error(self.load_error)
            return
        try:
            self._session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            self.load_error = None
            logger.info(
                "model loaded path=%s providers=%s",
                self.model_path,
                self._session.get_providers(),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via /api/status
            self._session = None
            self.load_error = f"Failed to load ONNX model: {exc}"
            logger.exception("model load failed")

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def provider(self) -> str:
        return self._session.get_providers()[0] if self._session else "none"

    # -- inference --------------------------------------------------------

    @staticmethod
    def decode_jpeg(payload: bytes) -> np.ndarray:
        """Decode JPEG bytes to a BGR frame, or raise InvalidFrameError.

        Every failure mode is funnelled into InvalidFrameError so one bad frame
        is a recoverable per-frame error rather than something that tears down
        the stream. Note that cv2.imdecode *raises* on an empty buffer instead
        of returning None, so the empty case is checked before calling it.
        """
        if not payload:
            raise InvalidFrameError("empty frame payload")
        try:
            frame = cv2.imdecode(
                np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
            )
        except cv2.error as exc:
            raise InvalidFrameError(f"could not decode frame: {exc}") from exc
        if frame is None or frame.size == 0:
            raise InvalidFrameError("payload is not a decodable image")
        return frame

    def detect(self, frame_bgr: np.ndarray) -> tuple[list[Detection], float]:
        """Run inference on one BGR frame.

        Returns the surviving person detections and the pure inference latency
        in milliseconds (model time only -- decode and drawing are excluded so
        the number means something when compared across machines).
        """
        if self._session is None or self._input_name is None:
            raise ModelNotLoadedError(self.load_error or "model is not loaded")

        # The graph takes uint8 NHWC RGB at any resolution: no letterboxing and
        # no /255 normalisation, which removes a whole class of coordinate bugs.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)[None]

        started = time.perf_counter()
        boxes, classes, scores, _count = self._session.run(
            None, {self._input_name: rgb}
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        return self._postprocess(boxes[0], classes[0], scores[0]), latency_ms

    def _postprocess(
        self, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray
    ) -> list[Detection]:
        """Keep confident `person` boxes and convert to (x1, y1, x2, y2).

        The graph emits boxes as (ymin, xmin, ymax, xmax); everything else in
        this codebase uses (x1, y1, x2, y2), so the swap happens exactly here
        and nowhere else.
        """
        keep = (classes.astype(int) == self.person_class_id) & (
            scores >= self.conf_threshold
        )
        detections: list[Detection] = []
        for (ymin, xmin, ymax, xmax), score in zip(boxes[keep], scores[keep]):
            detections.append(
                Detection(
                    label="person",
                    confidence=round(float(score), 4),
                    bbox=(
                        float(np.clip(xmin, 0.0, 1.0)),
                        float(np.clip(ymin, 0.0, 1.0)),
                        float(np.clip(xmax, 0.0, 1.0)),
                        float(np.clip(ymax, 0.0, 1.0)),
                    ),
                )
            )
        return detections
