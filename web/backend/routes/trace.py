"""WebSocket trace stream.

Connect to ``ws://host/ws/trace``. Send a JSON command:

    {"action": "recommend", "user_id": "u1", "directive": {...}, "k": 6}

The server streams an ordered sequence of trace events, then the final
recommendation as ``{"event": "result", "data": {...}}``.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

log = logging.getLogger("margo.web.trace")
router = APIRouter()


@router.websocket("/trace")
async def trace_ws(ws: WebSocket):
    await ws.accept()
    runner = ws.app.state.runner

    try:
        while True:
            raw = await ws.receive_text()
            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"event": "error", "message": "invalid JSON"})
                continue

            if cmd.get("action") != "recommend":
                await ws.send_json({"event": "error", "message": "unknown action"})
                continue

            user_id = cmd.get("user_id")
            if not user_id:
                await ws.send_json({"event": "error", "message": "user_id required"})
                continue

            await ws.send_json({"event": "phase", "data": {"phase": "phase1", "label": "Agent Initialization (cached)"}})
            await asyncio.sleep(0.2)

            async def emit(evt):
                await ws.send_json({"event": "trace", "data": evt.to_dict()})

            try:
                result = await runner.recommend(
                    user_id=user_id,
                    directive=cmd.get("directive"),
                    k=int(cmd.get("k", 6)),
                    emit=emit,
                )
                await ws.send_json({"event": "result", "data": result})
            except Exception as e:
                log.exception("trace_ws recommend failed")
                await ws.send_json({"event": "error", "message": str(e)})

    except WebSocketDisconnect:
        log.info("trace ws disconnected")
