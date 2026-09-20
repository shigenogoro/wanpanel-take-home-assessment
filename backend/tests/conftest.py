"""Test fixtures.

The suite runs against a throwaway SQLite file rather than Postgres, so a
reviewer can run `pytest` immediately after `pip install` without provisioning
a database. The ORM uses dialect-portable column types specifically to make
this possible.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

TMP_DIR = Path(tempfile.mkdtemp(prefix="wanpanel-tests-"))

# Must be set before `app.config` is imported anywhere.
os.environ["DATABASE_URL"] = f"sqlite:///{TMP_DIR / 'test.db'}"
os.environ["SNAPSHOT_DIR"] = str(TMP_DIR / "snapshots")


@pytest.fixture()
def client():
    """A TestClient with a fresh schema and the default zone seeded."""
    from fastapi.testclient import TestClient

    from app.db import engine
    from app.models import Base

    Base.metadata.drop_all(engine)

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
