"""Central configuration. Every tunable the reviewer may ask about lives here.

Nothing else in the codebase should hard-code a threshold; `GET /api/config`
serves this object verbatim so the dashboard and README cannot drift from the
behaviour that is actually running.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # our fields legitimately start with `model_`; opt out of pydantic's
        # protected namespace so they don't collide with BaseModel internals.
        protected_namespaces=(),
    )

    # --- Model -----------------------------------------------------------
    model_path: Path = BACKEND_DIR / "models" / "ssd_mobilenet_v1_10.onnx"
    model_name: str = "ssd_mobilenet_v1"
    model_version: str = "10 (ONNX opset 10, tf2onnx 1.7.0)"
    model_license: str = "Apache-2.0"
    model_source: str = (
        "https://github.com/onnx/models/tree/main/validated/vision/"
        "object_detection_segmentation/ssd-mobilenetv1"
    )
    # COCO label id for `person` in the TF Object Detection label map.
    person_class_id: int = 1

    # --- Detection thresholds -------------------------------------------
    conf_threshold: float = 0.45
    max_frame_bytes: int = 2 * 1024 * 1024

    # --- Tracking --------------------------------------------------------
    track_iou_threshold: float = 0.30
    track_max_age_frames: int = 15

    # --- Event rule: restricted-zone entry -------------------------------
    dwell_seconds: float = 2.0
    hysteresis_grace_seconds: float = 0.5
    track_cooldown_seconds: float = 15.0
    zone_cooldown_seconds: float = 10.0

    # --- Event rule: person absence --------------------------------------
    absence_seconds: float = 10.0
    absence_cooldown_seconds: float = 30.0

    # --- Storage ---------------------------------------------------------
    database_url: str = "sqlite:///./wanpanel.db"
    snapshot_dir: Path = BACKEND_DIR / "snapshots"

    # --- Server ----------------------------------------------------------
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
