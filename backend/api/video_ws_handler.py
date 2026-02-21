"""
VideoWebSocketHandler — Streams processed video frames to the browser.

Protocol (client → server, JSON):
  { type: 'video_play' }
  { type: 'video_pause' }
  { type: 'video_seek', frame: N }
  { type: 'video_click', x: N, y: N, frame: N }
  { type: 'video_reset_tracking' }
  { type: 'video_settings', blur_strength: 0-1, depth_bias: -0.5..0.5,
                             show_depth: bool }

Protocol (server → client, JSON):
  { type: 'video_metadata', ...meta }
  { type: 'video_frame', frame: b64, frame_index: N, total: N,
                          fps: F, state: str }
  { type: 'tracking_update', bbox, state }
  { type: 'export_progress', progress: 0-100 }
  { type: 'export_done', url: '/video/exports/<id>' }
  { type: 'error', message: str }
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid
import cv2
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict

from core.video_session import VideoSession


# How long (seconds) to sleep between frames during playback
_MIN_INTERVAL = 1.0 / 30   # cap at 30 fps stream


class VideoWebSocketHandler:
    """Manages WebSocket sessions for video upload mode."""

    def __init__(self, sessions: Dict[str, VideoSession]) -> None:
        self._sessions = sessions   # shared dict: session_id → VideoSession

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    async def handle_connection(
        self,
        websocket: WebSocket,
        session_id: str,
    ) -> None:
        await websocket.accept()
        session = self._sessions.get(session_id)
        if session is None:
            await websocket.send_json({"type": "error", "message": "Session not found"})
            await websocket.close()
            return

        # Send metadata immediately
        await websocket.send_json({
            "type": "video_metadata",
            **session.metadata,
            "session_id": session.session_id,
        })

        print(f"[VideoWS] Connected  session={session_id}  "
              f"frames={session.frame_count}  fps={session.fps:.1f}")

        try:
            while True:
                loop_start = time.monotonic()

                # ----- Incoming messages (non-blocking 1ms) -----
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(), timeout=0.001
                    )
                    await self._dispatch(websocket, session, raw)
                except asyncio.TimeoutError:
                    pass

                # ----- Stream one frame if playing -----
                if session.is_playing:
                    await self._send_frame(websocket, session)
                    advanced = session.advance()
                    if not advanced:
                        session.is_playing = False   # reached end

                # ----- Throttle -----
                elapsed   = time.monotonic() - loop_start
                target    = max(_MIN_INTERVAL, 1.0 / max(session.fps, 1))
                sleep_for = max(0.0, target - elapsed)
                await asyncio.sleep(sleep_for if sleep_for > 0 else 0.001)

        except WebSocketDisconnect:
            print(f"[VideoWS] Disconnected  session={session_id}")
        except Exception as e:
            print(f"[VideoWS] Error: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # Frame sender
    # ------------------------------------------------------------------
    async def _send_frame(
        self,
        ws: WebSocket,
        session: VideoSession,
    ) -> None:
        frame = session.get_rendered_frame()
        if frame is None:
            return

        _, buf   = cv2.imencode('.jpg', frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 82])
        b64      = base64.b64encode(buf).decode('utf-8')
        tk_state = session.tracker.state

        await ws.send_json({
            "type":        "video_frame",
            "frame":       b64,
            "frame_index": session.current_frame,
            "total":       session.frame_count,
            "fps":         round(session.fps, 1),
            "state":       tk_state,
        })

    # ------------------------------------------------------------------
    # Message dispatcher
    # ------------------------------------------------------------------
    async def _dispatch(
        self,
        ws: WebSocket,
        session: VideoSession,
        raw: str,
    ) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return

        t = msg.get("type", "")

        if t == "video_play":
            session.is_playing = True
            # Immediately send first frame so UI doesn't blank
            await self._send_frame(ws, session)

        elif t == "video_pause":
            session.is_playing = False
            await self._send_frame(ws, session)

        elif t == "video_seek":
            fi = int(msg.get("frame", 0))
            session.seek(fi)
            session.is_playing = False
            await self._send_frame(ws, session)

        elif t == "video_click":
            x  = int(msg.get("x", 0))
            y  = int(msg.get("y", 0))
            fi = int(msg.get("frame", session.current_frame))
            bbox = session.on_click(fi, x, y)
            await ws.send_json({
                "type":  "tracking_update",
                "bbox":  list(bbox) if bbox else None,
                "state": session.tracker.state,
            })
            await self._send_frame(ws, session)

        elif t == "video_reset_tracking":
            session.reset_tracking()
            await ws.send_json({"type": "tracking_update",
                                "bbox": None, "state": "idle"})
            await self._send_frame(ws, session)

        elif t == "video_settings":
            if "blur_strength" in msg:
                session.blur_strength = float(msg["blur_strength"])
            if "depth_bias" in msg:
                session.depth_bias = float(msg["depth_bias"])
            if "show_depth" in msg:
                session.show_depth_debug = bool(msg["show_depth"])
            await self._send_frame(ws, session)

        elif t == "video_export":
            await self._start_export(ws, session)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    async def _start_export(
        self,
        ws: WebSocket,
        session: VideoSession,
    ) -> None:
        from core.video_pipeline.exporter import VideoExporter

        loop = asyncio.get_event_loop()
        export_dir = os.path.join(
            os.path.dirname(session.file_path), "exports"
        )
        os.makedirs(export_dir, exist_ok=True)
        exporter = VideoExporter(tmp_dir=export_dir)

        # Snapshot current tracker state for export
        bbox_ema = session.tracker.bbox_ema
        init_bbox = (
            (int(bbox_ema[0]), int(bbox_ema[1]),
             int(bbox_ema[2]), int(bbox_ema[3]))
            if bbox_ema else None
        )

        last_sent = [0]

        def progress_cb(cur, total):
            pct = int(cur / max(total, 1) * 100)
            if pct != last_sent[0]:
                last_sent[0] = pct
                asyncio.run_coroutine_threadsafe(
                    ws.send_json({"type": "export_progress", "progress": pct}),
                    loop,
                )

        await ws.send_json({"type": "export_progress", "progress": 0})

        out_path = await loop.run_in_executor(
            None,
            lambda: exporter.export(
                session.file_path,
                init_bbox,
                blur_strength=session.blur_strength,
                depth_bias=session.depth_bias,
                progress_cb=progress_cb,
            ),
        )

        if out_path:
            export_id = os.path.basename(out_path)
            await ws.send_json({
                "type": "export_done",
                "url":  f"/video/exports/{export_id}",
            })
        else:
            await ws.send_json({"type": "error", "message": "Export failed"})
