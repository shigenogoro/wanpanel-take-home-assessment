"""Database engine, session factory and startup bootstrap."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .models import Base, Zone

logger = logging.getLogger(__name__)

IS_SQLITE = settings.database_url.startswith("sqlite")

# SQLite needs two accommodations that Postgres does not. Inference runs in a
# worker thread (see routers/ws.py), so a pooled connection legitimately moves
# between threads -- hence check_same_thread=False. And SQLite locks the whole
# database on write, so a busy timeout stops a snapshot write colliding with a
# dashboard query and raising "database is locked".
_connect_args = (
    {"check_same_thread": False, "timeout": 15} if IS_SQLITE else {}
)

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # survive Postgres restarts without a stale-conn error
    connect_args=_connect_args,
    future=True,
)


@event.listens_for(Engine, "connect")
def _configure_sqlite(dbapi_connection, _record) -> None:
    """Put SQLite in WAL mode so reads never block on a write.

    Without this, the dashboard polling /api/status every 2s can be locked out
    by the inference thread committing an event.
    """
    if not IS_SQLITE:
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

#: Bottom-anchored rectangle covering the lower-left of the frame. Anchored to
#: the bottom edge because the rule tests a person's *foot* point, which sits
#: near y=1.0 whenever someone is close enough to the camera to be cut off at
#: the knee -- a zone floating above y=0.95 would never trigger in practice.
#:
#: The x range matches where the bundled sample source puts its two people once
#: they settle, so a fresh database demonstrates a trigger without redrawing the
#: ROI first. Matches EVAL_ZONE in the fixture and evaluation scripts.
DEFAULT_POLYGON = [[0.08, 0.55], [0.48, 0.55], [0.48, 1.0], [0.08, 1.0]]


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def ping() -> bool:
    """True if the database answers. Used by /api/status."""
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True
    except Exception:  # noqa: BLE001 - reported as status, never raised to UI
        logger.warning("database ping failed", exc_info=True)
        return False


def ensure_default_zone() -> None:
    """Create the starter ROI on first run so the demo is never zone-less."""
    with SessionLocal() as session:
        existing = session.scalar(select(Zone).where(Zone.active.is_(True)))
        if existing is not None:
            return
        session.add(
            Zone(name="Restricted Zone", polygon=DEFAULT_POLYGON, active=True)
        )
        session.commit()
        logger.info("seeded default zone")


def create_all() -> None:
    """Create tables directly from the ORM metadata.

    Alembic remains the documented path for schema changes; this exists so the
    test suite and a first-time reviewer can get a working database without a
    migration step.
    """
    Base.metadata.create_all(engine)
