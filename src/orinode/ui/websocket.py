"""WebSocket endpoint: stream training events to the frontend."""

from __future__ import annotations

import asyncio

from fastapi import WebSocket, WebSocketDisconnect

from orinode.ui.progress_store import ProgressStore

_store = ProgressStore()
_POLL_INTERVAL = 1.0  # seconds between polls


async def ws_run_events(websocket: WebSocket, run_id: str) -> None:
    """Stream new events for ``run_id`` as they arrive on disk.

    Polls the JSONL file every second and forwards any new events (those with
    ``ts > last_seen_ts``) as JSON messages.  Closes cleanly on disconnect.

    Args:
        websocket: Accepted WebSocket connection.
        run_id: Training run identifier to follow.
    """
    await websocket.accept()
    last_ts = 0.0
    try:
        while True:
            new_events = _store.tail_events(run_id, after_ts=last_ts)
            for ev in new_events:
                await websocket.send_json(ev.model_dump())
                last_ts = max(last_ts, ev.ts)
            await asyncio.sleep(_POLL_INTERVAL)
    except WebSocketDisconnect:
        pass
