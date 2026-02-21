"""
VideoTracker — Frame-by-frame tracker for video pipeline.

Features:
  • CSRT tracker (MIL fallback) — same as the live autofocus tracker
  • Exponential-Moving-Average (EMA) smoothing on bbox coords (α=0.35)
  • Full-body expansion identical to live mode (2.5×W, 4.5×H, 80% downward)
  • Occlusion grace period: holds last known position for 1.0 s at video fps
  • Unique tracking ID per session
"""
from __future__ import annotations

import uuid
import time
import cv2
import numpy as np
from typing import Optional, Tuple


_EMA_ALPHA = 0.35          # Smoothing factor:  0=frozen, 1=raw tracker output
_GRACE_FRAMES = 30         # Frames to hold position on tracker failure


def _create_tracker() -> cv2.Tracker:
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, 'TrackerMIL_create'):
        return cv2.TrackerMIL_create()
    raise RuntimeError("No OpenCV tracker available (need opencv-contrib or MIL)")


def _expand_to_body(
    bbox: Tuple[int, int, int, int],
    frame_h: int,
    frame_w: int,
) -> Tuple[int, int, int, int]:
    """Expand face bbox to approx full-body region (matches live pipeline)."""
    x, y, bw, bh = bbox
    cx = x + bw // 2
    cy = y + bh // 2

    aspect = bh / max(bw, 1)
    if aspect < 1.2:
        h_mult = 4.5 * 1.3
    elif aspect > 2.0:
        h_mult = 4.5 * 0.75
    else:
        h_mult = 4.5

    new_w = int(bw * 2.5)
    new_h = int(bh * h_mult)

    # Cap to 70% of frame
    max_area = frame_h * frame_w * 0.70
    if new_w * new_h > max_area:
        scale = (max_area / (new_w * new_h)) ** 0.5
        new_w = int(new_w * scale)
        new_h = int(new_h * scale)

    x1 = max(0,       cx - new_w // 2)
    y1 = max(0,       cy - int(new_h * 0.20))
    x2 = min(frame_w, x1 + new_w)
    y2 = min(frame_h, y1 + new_h)
    return x1, y1, x2, y2


class VideoTracker:
    """Single-subject tracker for offline video processing."""

    def __init__(self) -> None:
        self.track_id   = str(uuid.uuid4())[:8]
        self._tracker   = None
        self._bbox_ema  = None          # (x, y, w, h) float
        self._grace     = 0             # frames remaining in grace period
        self._state     = "idle"        # idle | tracking | grace
        self._frame_h   = 0
        self._frame_w   = 0

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def initialize(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> bool:
        """
        Initialize tracker with the given (x, y, w, h) bounding box.
        Returns True on success.
        """
        self._frame_h, self._frame_w = frame.shape[:2]
        self._tracker = _create_tracker()
        ok = self._tracker.init(frame, bbox)
        if ok is False:
            print("[VideoTracker] Tracker init failed")
            return False

        self._bbox_ema = tuple(float(v) for v in bbox)
        self._state    = "tracking"
        self._grace    = 0
        print(f"[VideoTracker] Initialized  id={self.track_id}  bbox={bbox}")
        return True

    def reset(self) -> None:
        self._tracker  = None
        self._bbox_ema = None
        self._grace    = 0
        self._state    = "idle"

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(
        self, frame: np.ndarray
    ) -> Tuple[str, Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
        """
        Update tracker for the current frame.

        Returns
        -------
        (state, tracker_bbox, body_bbox)
          state       : "tracking" | "grace" | "idle"
          tracker_bbox: raw (x,y,w,h) or None
          body_bbox   : expanded (x1,y1,x2,y2) or None
        """
        if self._tracker is None or self._state == "idle":
            return "idle", None, None

        h, w = frame.shape[:2]
        ok, raw_bbox = self._tracker.update(frame)

        if ok is not False and ok:
            # Valid update — EMA smooth
            rx, ry, rw, rh = raw_bbox
            ex, ey, ew, eh = self._bbox_ema
            a = _EMA_ALPHA
            self._bbox_ema = (
                a * rx + (1-a) * ex,
                a * ry + (1-a) * ey,
                a * rw + (1-a) * ew,
                a * rh + (1-a) * eh,
            )
            self._grace = 0
            self._state = "tracking"
        else:
            # Tracker lost
            self._grace += 1
            if self._grace > _GRACE_FRAMES:
                self._state = "grace"
            # Keep last known EMA position

        if self._bbox_ema is None:
            return self._state, None, None

        # Integer bbox for return
        ix, iy, iw, ih = (int(v) for v in self._bbox_ema)
        tracker_bbox = (ix, iy, iw, ih)
        body_bbox    = _expand_to_body(tracker_bbox, h, w)

        return self._state, tracker_bbox, body_bbox

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def bbox_ema(self) -> Optional[Tuple[float, ...]]:
        return self._bbox_ema
