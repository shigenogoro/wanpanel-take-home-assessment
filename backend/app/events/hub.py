"""Fan-out hub for the `/ws/events` dashboard channel.

Every connected dashboard gets every event the moment it is created or its
status changes -- this is what satisfies "show new events without requiring a
full page refresh" without polling.

A dead or slow socket must never take down the inference loop, so `broadcast`
swallows per-client failures and prunes the connection instead.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class EventHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
        logger.info("dashboard connected (%d total)", len(self._clients))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)
        logger.info("dashboard disconnected (%d left)", len(self._clients))

    async def broadcast(self, message: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._clients)

        dead: list[WebSocket] = []
        for client in targets:
            try:
                await client.send_json(message)
            except Exception:  # noqa: BLE001 - a broken dashboard is not fatal
                dead.append(client)

        if dead:
            async with self._lock:
                for client in dead:
                    self._clients.discard(client)
            logger.info("pruned %d dead dashboard socket(s)", len(dead))


hub = EventHub()
