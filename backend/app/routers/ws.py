"""WebSocket endpoints.

`/ws/inference`
    Binary JPEG frames in, structured detections out. The browser sends one
    frame at a time and waits for the reply before sending the next, which
    gives natural backpressure: a slow CPU lowers the frame rate instead of
    building an unbounded queue.

`/ws/events`
    Read-only broadcast of event creations and status changes to every open
    dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from ..config import settings
from ..db import SessionLocal
from ..events import service
from ..events.hub import hub
from ..events.rules import evaluate
from ..inference.detector import InvalidFrameError, ModelNotLoadedError
from ..inference.metrics import metrics
from ..models import Zone
from ..runtime import runtime
from ..schemas import (
    DetectionOut,
    InferenceResult,
    WsError,
    ZoneStateOut,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])

#: Close code used when the server cannot do its job at all (RFC 6455 1011).
INTERNAL_ERROR = 1011


@router.websocket("/ws/events")
async def events_socket(websocket: WebSocket) -> None:
    await hub.connect(websocket)
    try:
        while True:
            # The channel is push-only; this read just parks until the client
            # goes away, which is how we learn to prune the connection.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.debug("dashboard socket error", exc_info=True)
    finally:
        await hub.disconnect(websocket)


@router.websocket("/ws/inference")
async def inference_socket(websocket: WebSocket) -> None:
    source = websocket.query_params.get("source", "webcam")

    await websocket.accept()

    # Fail loudly and specifically rather than accepting frames we cannot use.
    if not runtime.detector.is_loaded:
        await websocket.send_json(
            WsError(
                code="model_not_loaded",
                message=runtime.detector.load_error
                or "model is not loaded; run python backend/scripts/fetch_model.py",
            ).model_dump()
        )
        await websocket.close(code=INTERNAL_ERROR, reason="model not loaded")
        return

    if runtime.inference_clients > 0:
        await websocket.send_json(
            WsError(
                code="stream_busy",
                message=(
                    "another inference stream is already active; this demo "
                    "tracks one camera at a time so dwell timers stay coherent"
                ),
            ).model_dump()
        )
        await websocket.close(code=INTERNAL_ERROR, reason="stream busy")
        return

    runtime.begin_stream(source)
    frame_id = 0

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            payload = message.get("bytes")
            if payload is None:
                # Text frames are not part of the protocol, but a stray one
                # should not kill the stream.
                await websocket.send_json(
                    WsError(
                        code="expected_binary",
                        message="send JPEG bytes, not text",
                    ).model_dump()
                )
                continue

            if len(payload) > settings.max_frame_bytes:
                metrics.record_rejected()
                await websocket.send_json(
                    WsError(
                        code="frame_too_large",
                        message=(
                            f"frame is {len(payload)} bytes, limit is "
                            f"{settings.max_frame_bytes}"
                        ),
                    ).model_dump()
                )
                continue

            frame_id += 1
            try:
                result = await asyncio.to_thread(
                    _process_frame, payload, frame_id, source
                )
            except InvalidFrameError as exc:
                metrics.record_rejected()
                await websocket.send_json(
                    WsError(code="invalid_frame", message=str(exc)).model_dump()
                )
                continue
            except ModelNotLoadedError as exc:
                await websocket.send_json(
                    WsError(
                        code="model_not_loaded", message=str(exc)
                    ).model_dump()
                )
                await websocket.close(
                    code=INTERNAL_ERROR, reason="model not loaded"
                )
                break

            result_payload, broadcasts = result
            await websocket.send_json(result_payload)
            for message_out in broadcasts:
                await hub.broadcast(message_out)

    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 - log, then close cleanly
        logger.exception("inference socket failed")
        try:
            await websocket.close(code=INTERNAL_ERROR, reason="internal error")
        except Exception:  # noqa: BLE001
            pass
    finally:
        runtime.end_stream()


def _process_frame(
    payload: bytes, frame_id: int, source: str
) -> tuple[dict, list[dict]]:
    """Decode, detect, track, evaluate and persist. Runs off the event loop.

    Inference is CPU-bound and holds the GIL inside ONNX Runtime's native code,
    so it is dispatched to a worker thread; otherwise a ~25 ms model call would
    stall every dashboard socket on the same loop.
    """
    frame = runtime.detector.decode_jpeg(payload)
    detections, latency_ms = runtime.detector.detect(frame)
    metrics.record_frame(latency_ms)

    tracked = runtime.tracker.update(detections)

    with SessionLocal() as session:
        zone = session.scalar(select(Zone).where(Zone.active.is_(True)))
        # Copy out what we need now; `zone` is detached once the session closes.
        zone_id = zone.id if zone else None
        zone_polygon = list(zone.polygon) if zone else []
        polygon = [(float(x), float(y)) for x, y in zone_polygon]

        evaluation = evaluate(
            runtime.rule_state,
            tracked,
            polygon,
            time.monotonic(),
            runtime.rule_config,
        )

        broadcasts: list[dict] = []
        fired_ids = []
        for draft in evaluation.events:
            event = service.create_event(
                session,
                draft,
                source=source,
                zone_id=zone_id,
                frame_bgr=frame,
                polygon=zone_polygon or None,
            )
            fired_ids.append(event.id)
            broadcasts.append(
                {
                    "type": "event.created",
                    "payload": service.to_out(event).model_dump(mode="json"),
                }
            )

    inside = set(evaluation.inside_track_ids)
    result = InferenceResult(
        frame_id=frame_id,
        ts=datetime.now(timezone.utc),
        latency_ms=round(latency_ms, 2),
        fps=round(metrics.fps, 2),
        detections=[
            DetectionOut(
                track_id=d.track_id,
                label=d.label,
                confidence=d.confidence,
                bbox=list(d.bbox),
                foot=list(d.foot),
                in_zone=d.track_id in inside,
            )
            for d in tracked
        ],
        zone_state=ZoneStateOut(
            zone_id=zone_id,
            occupied_track_ids=sorted(inside),
            dwell_seconds=round(evaluation.max_dwell_seconds, 2),
        ),
        events_fired=fired_ids,
    )
    return result.model_dump(mode="json"), broadcasts
