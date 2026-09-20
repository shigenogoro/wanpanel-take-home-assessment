"""API-level tests: status reporting, the event lifecycle, ROI validation and
the WebSocket error contract."""

from __future__ import annotations

import cv2
import numpy as np
import pytest


def jpeg_bytes(width: int = 320, height: int = 240) -> bytes:
    """A small, valid JPEG. Contains no people, which is the point: it proves
    the transport works without depending on model accuracy."""
    frame = np.full((height, width, 3), 120, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


# --- status ---------------------------------------------------------------


def test_health_is_cheap_and_always_ok(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_status_reports_model_database_and_zone(client):
    body = client.get("/api/status").json()
    assert body["model"]["name"] == "ssd_mobilenet_v1"
    assert body["model"]["license"] == "Apache-2.0"
    assert body["database_connected"] is True
    assert body["zone"]["polygon"]
    assert "inference_ms_p95" in body["metrics"]


def test_config_exposes_live_thresholds(client):
    body = client.get("/api/config").json()
    assert body["dwell_seconds"] == 2.0
    assert body["conf_threshold"] == 0.45


# --- event lifecycle ------------------------------------------------------


def test_event_moves_open_to_acknowledged_to_resolved(client):
    created = client.post("/api/events/test", json={"label": "lifecycle"})
    assert created.status_code == 201
    event_id = created.json()["id"]
    assert created.json()["status"] == "open"
    # A manual event must be unmistakably tagged as non-model output.
    assert created.json()["source"] == "manual_test"

    ack = client.patch(f"/api/events/{event_id}", json={"status": "acknowledged"})
    assert ack.status_code == 200
    assert ack.json()["status"] == "acknowledged"
    assert ack.json()["acknowledged_at"] is not None

    res = client.patch(f"/api/events/{event_id}", json={"status": "resolved"})
    assert res.status_code == 200
    assert res.json()["resolved_at"] is not None


def test_status_survives_a_fresh_read(client):
    """Persistence: the value comes back from the database, not memory."""
    event_id = client.post("/api/events/test", json={}).json()["id"]
    client.patch(f"/api/events/{event_id}", json={"status": "acknowledged"})
    assert client.get(f"/api/events/{event_id}").json()["status"] == "acknowledged"


@pytest.mark.parametrize(
    "first,second",
    [("resolved", "acknowledged"), ("resolved", "resolved")],
)
def test_illegal_transitions_are_rejected_with_422(client, first, second):
    event_id = client.post("/api/events/test", json={}).json()["id"]
    client.patch(f"/api/events/{event_id}", json={"status": first})
    bad = client.patch(f"/api/events/{event_id}", json={"status": second})
    assert bad.status_code == 422


def test_reopening_an_acknowledged_event_is_rejected(client):
    event_id = client.post("/api/events/test", json={}).json()["id"]
    client.patch(f"/api/events/{event_id}", json={"status": "acknowledged"})
    assert (
        client.patch(f"/api/events/{event_id}", json={"status": "open"}).status_code
        == 422
    )


def test_unknown_event_is_404(client):
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/api/events/{missing}").status_code == 404


def test_events_can_be_filtered_and_paginated(client):
    for i in range(3):
        client.post("/api/events/test", json={"label": f"e{i}"})
    ack_id = client.get("/api/events").json()["items"][0]["id"]
    client.patch(f"/api/events/{ack_id}", json={"status": "acknowledged"})

    open_only = client.get("/api/events", params={"status": "open"}).json()
    assert open_only["total"] == 2
    assert all(item["status"] == "open" for item in open_only["items"])

    page = client.get("/api/events", params={"limit": 1, "offset": 0}).json()
    assert len(page["items"]) == 1 and page["total"] == 3


def test_missing_snapshot_is_404_not_a_crash(client):
    event_id = client.post("/api/events/test", json={}).json()["id"]
    assert client.get(f"/api/events/{event_id}/snapshot").status_code == 404


# --- zones ----------------------------------------------------------------


def test_roi_can_be_reconfigured(client):
    zone_id = client.get("/api/zones/active").json()["id"]
    polygon = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.5], [0.1, 0.5]]
    updated = client.put(f"/api/zones/{zone_id}", json={"polygon": polygon})
    assert updated.status_code == 200
    assert updated.json()["polygon"] == polygon
    assert client.get("/api/zones/active").json()["polygon"] == polygon


@pytest.mark.parametrize(
    "polygon",
    [
        [[0.1, 0.1], [0.4, 0.1]],  # fewer than 3 vertices
        [[0.1, 0.1], [0.4, 0.1], [1.4, 0.5]],  # outside [0, 1]
        [[0.1, 0.1], [0.4, 0.1], [0.4, 0.5, 0.2]],  # not an (x, y) pair
    ],
)
def test_invalid_roi_is_rejected(client, polygon):
    zone_id = client.get("/api/zones/active").json()["id"]
    assert (
        client.put(f"/api/zones/{zone_id}", json={"polygon": polygon}).status_code
        == 422
    )


# --- websocket contract ---------------------------------------------------


def test_inference_socket_returns_structured_output(client):
    with client.websocket_connect("/ws/inference?source=webcam") as socket:
        socket.send_bytes(jpeg_bytes())
        message = socket.receive_json()

    assert message["type"] == "inference"
    assert message["frame_id"] == 1
    assert message["latency_ms"] > 0
    assert message["detections"] == []  # a flat grey frame has no people
    assert "occupied_track_ids" in message["zone_state"]
    assert message["events_fired"] == []


def test_corrupt_frame_is_reported_without_dropping_the_stream(client):
    with client.websocket_connect("/ws/inference") as socket:
        socket.send_bytes(b"this is definitely not a jpeg")
        error = socket.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "invalid_frame"

        # The socket must still be usable afterwards.
        socket.send_bytes(jpeg_bytes())
        assert socket.receive_json()["type"] == "inference"


def test_oversized_frame_is_rejected(client):
    with client.websocket_connect("/ws/inference") as socket:
        socket.send_bytes(b"\xff" * (2 * 1024 * 1024 + 1))
        error = socket.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "frame_too_large"


def test_text_on_the_binary_channel_is_reported_not_fatal(client):
    with client.websocket_connect("/ws/inference") as socket:
        socket.send_text("hello")
        assert socket.receive_json()["code"] == "expected_binary"


def test_second_inference_stream_is_refused(client):
    """One camera at a time, stated explicitly rather than silently degrading."""
    with client.websocket_connect("/ws/inference"):
        with client.websocket_connect("/ws/inference") as second:
            assert second.receive_json()["code"] == "stream_busy"
