"""FastAPI application entry point.

    uvicorn app.main:app --reload --port 8000     (run from backend/)

Startup deliberately never aborts. A monitoring system that refuses to boot
because its model file or database is missing gives the operator nothing to
look at; instead the failure is recorded and surfaced through `/api/status`,
and the dashboard renders an actionable banner.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import create_all, ensure_default_zone
from .routers import events, status, ws, zones

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .runtime import runtime

    settings.snapshot_dir.mkdir(parents=True, exist_ok=True)

    try:
        create_all()
        ensure_default_zone()
        logger.info("database ready at %s", _safe_dsn(settings.database_url))
    except Exception:  # noqa: BLE001 - reported via /api/status
        logger.exception(
            "database unavailable at startup; the API will stay up and report "
            "database_connected=false"
        )

    runtime.load_model()
    if not runtime.detector.is_loaded:
        logger.error("MODEL NOT LOADED: %s", runtime.detector.load_error)

    yield
    logger.info("shutting down")


def _safe_dsn(dsn: str) -> str:
    """Render a connection string without leaking the password into logs."""
    if "@" not in dsn or "://" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    credentials, host = rest.rsplit("@", 1)
    user = credentials.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


app = FastAPI(
    title="Local AI Safety Monitoring System",
    description=(
        "Local CPU person detection (SSD-MobileNetV1 via ONNX Runtime) turned "
        "into restricted-zone safety events, with a live monitoring dashboard."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(status.router)
app.include_router(events.router)
app.include_router(zones.router)
app.include_router(ws.router)
