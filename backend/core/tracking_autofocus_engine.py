"""
TrackingAutofocusEngine
=======================
CSRT-based subject tracker that drives dynamic autofocus blur.

Pipeline per frame:
  frame → tracker.update() → new bbox center → blur mask → composite

No deep-learning models. No detection dependencies. Pure OpenCV.
"""
import time
import threading
import cv2
import numpy as np
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_soft_mask(
    h: int, w: int,
    cx: int, cy: int,
    radius: int,
    feather: int,
) -> np.ndarray:
    """
    Float32 H×W mask: 1.0 inside focus circle, 0.0 outside.
    Linear feather over `feather` pixels at the boundary.
    """
    Y, X = np.ogrid[:h, :w]
    dist = np.hypot(X - cx, Y - cy).astype(np.float32)
    inner = float(radius)
    outer = float(radius + max(feather, 1))
    return np.clip((outer - dist) / (outer - inner), 0.0, 1.0)


def _create_tracker() -> cv2.Tracker:
    """
    Return the best available OpenCV tracker.
    Preference: CSRT (opencv-contrib) → MIL (built-in).
    """
    if hasattr(cv2, 'TrackerCSRT_create'):
        print("[TRACKER] Using CSRT tracker")
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, 'TrackerMIL_create'):
        print("[TRACKER] CSRT unavailable, using MIL tracker")
        return cv2.TrackerMIL_create()
    raise RuntimeError(
        "No suitable OpenCV tracker found. "
        "Install opencv-contrib-python for CSRT, or ensure opencv-python>=4.x for MIL."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TrackingAutofocusEngine
# ─────────────────────────────────────────────────────────────────────────────

class TrackingAutofocusEngine:
    """
    Maintains a CSRT tracker; updates the focus region each frame.

    State machine
    ─────────────
    IDLE        → no tracker, no blur
    TRACKING    → tracker active, blur follows bbox
    GRACE       → tracker lost, hold last position for `grace_period` seconds
    LOST        → grace expired, revert to IDLE
    """

    # Public states for the websocket payload / UI badge
    STATE_IDLE     = "idle"
    STATE_TRACKING = "tracking"
    STATE_GRACE    = "grace"
    STATE_LOST     = "lost"

    def __init__(self, config: dict = None):
        cfg = config or {}

        # Tracker init box
        self.bbox_size:    int   = cfg.get('bbox_size', 120)      # px, square
        # Blur mask — tight radius so blur is clearly visible around subject
        self.focus_radius: int   = cfg.get('focus_radius', 75)    # px radius (tight)
        self.feather:      int   = cfg.get('feather', 30)         # px softness
        self.blur_ksize:   int   = cfg.get('blur_ksize', 101)     # heavy GaussianBlur
        # Grace period on tracker loss
        self.grace_period: float = cfg.get('grace_period', 0.8)   # seconds

        # Internal state
        self._lock             = threading.Lock()
        self._state            = self.STATE_IDLE
        self._tracker                          = None
        self._bbox: Optional[Tuple[int,int,int,int]] = None  # x,y,w,h
        self._center: Optional[Tuple[int,int]]       = None
        self._loss_time: Optional[float]             = None
        self._frame_count                            = 0

        # Mask cache (avoid rebuilding every frame when center unchanged)
        self._mask_cache: Optional[np.ndarray] = None
        self._mask_key:   tuple                = ()

        # Pending click — set by on_click(), consumed in init_tracker()
        self._pending_click: Optional[Tuple[int,int]] = None

        print(f"✓ TrackingAutofocusEngine  "
              f"bbox={self.bbox_size}px  radius={self.focus_radius}px  "
              f"blur_k={self.blur_ksize}  grace={self.grace_period}s")

    # ──────────────────────────────────────────────────── public API ──────────

    def on_click(self, x: int, y: int):
        """Called by websocket handler when user clicks on video."""
        with self._lock:
            self._pending_click = (int(x), int(y))
        print(f"[TRACKER] Click queued → ({x}, {y})")

    def on_double_click(self):
        """Reset tracker and remove blur."""
        with self._lock:
            self._reset_locked()
        print("[TRACKER] Reset (double-click)")

    def set_focus_radius(self, radius: int):
        with self._lock:
            self.focus_radius = int(radius)
            self._mask_cache = None          # force rebuild
        print(f"[TRACKER] focus_radius → {radius}px")

    def set_blur_strength(self, strength: float):
        k = max(3, int(strength * 20) | 1)
        with self._lock:
            self.blur_ksize  = k
            self._mask_cache = None
        print(f"[TRACKER] blur_ksize → {k}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Main per-frame entry point.
        Returns blur-composited frame (or original if IDLE).
        """
        self._frame_count += 1
        h, w = frame.shape[:2]

        with self._lock:
            pending = self._pending_click
            if pending is not None:
                self._pending_click = None

        # ── Handle pending click: (re)init tracker ────────────────────
        if pending is not None:
            self._init_tracker(frame, pending[0], pending[1])

        # ── Update tracker ────────────────────────────────────────────
        with self._lock:
            state = self._state

        if state == self.STATE_TRACKING:
            self._update_tracker(frame)

        # ── Composite ─────────────────────────────────────────────────
        with self._lock:
            state  = self._state
            center = self._center
            bbox   = self._bbox

        if state in (self.STATE_TRACKING, self.STATE_GRACE) and center is not None:
            output = self._composite_blur(frame, center)
            output = self._draw_overlay(output, bbox, center, state)
        else:
            # IDLE preview: blur whole frame 50% + hint text so
            # user can immediately see the effect is ready
            k = self.blur_ksize | 1
            blurred  = cv2.GaussianBlur(frame, (k, k), 0)
            blurred  = cv2.GaussianBlur(blurred, (31, 31), 0)
            output   = cv2.addWeighted(blurred, 0.55, frame, 0.45, 0)
            _put_label(output, "Tap a subject to focus",
                       frame.shape[1] // 2, frame.shape[0] - 24,
                       (200, 220, 255), scale=0.6)

        return output

    def get_status(self) -> dict:
        with self._lock:
            return {
                'state':        self._state,
                'center':       self._center,
                'bbox':         self._bbox,
                'focus_radius': self.focus_radius,
                'blur_ksize':   self.blur_ksize,
            }

    def cleanup(self):
        with self._lock:
            self._reset_locked()
        print("✓ TrackingAutofocusEngine cleanup")

    # ──────────────────────────────────────────────── internal ───────────────

    def _init_tracker(self, frame: np.ndarray, cx: int, cy: int):
        """Create / reinitialize CSRT tracker centered at (cx, cy)."""
        h, w = frame.shape[:2]
        half  = self.bbox_size // 2
        x1    = max(0, cx - half)
        y1    = max(0, cy - half)
        bw    = min(self.bbox_size, w - x1)
        bh    = min(self.bbox_size, h - y1)

        # Ensure minimum viable bbox
        if bw < 10 or bh < 10:
            print(f"[TRACKER] Click too close to edge, skipping init")
            return

        bbox = (x1, y1, bw, bh)

        tracker = _create_tracker()
        try:
            ok = tracker.init(frame, bbox)
        except Exception as e:
            print(f"[TRACKER] Init error: {e}")
            return

        if ok is False:
            print("[TRACKER] init() returned False")
            return

        with self._lock:
            self._tracker    = tracker
            self._bbox       = bbox
            self._center     = (cx, cy)
            self._state      = self.STATE_TRACKING
            self._loss_time  = None
            self._mask_cache = None   # force mask rebuild at new position

        print(f"[TRACKER] Initialized  center=({cx},{cy})  bbox={bbox}")

    def _update_tracker(self, frame: np.ndarray):
        """Run tracker.update() and advance state machine."""
        with self._lock:
            tracker = self._tracker

        if tracker is None:
            return

        try:
            success, bbox = tracker.update(frame)
        except Exception as e:
            print(f"[TRACKER] update() error: {e}")
            success = False
            bbox    = None

        with self._lock:
            if success:
                x, y, bw, bh = [int(v) for v in bbox]
                cx = x + bw // 2
                cy = y + bh // 2
                self._bbox       = (x, y, bw, bh)
                self._center     = (cx, cy)
                self._state      = self.STATE_TRACKING
                self._loss_time  = None

                if self._frame_count % 30 == 0:
                    print(f"[TRACKER] Updated  center=({cx},{cy})  bbox={bbox}")
            else:
                # Tracker failed — enter grace period if not already
                if self._state == self.STATE_TRACKING:
                    self._state     = self.STATE_GRACE
                    self._loss_time = time.time()
                    print("[TRACKER] Lost — grace period started")
                elif self._state == self.STATE_GRACE:
                    elapsed = time.time() - (self._loss_time or 0)
                    if elapsed >= self.grace_period:
                        print("[TRACKER] Grace period expired — IDLE")
                        self._reset_locked()

    def _composite_blur(
        self,
        frame:  np.ndarray,
        center: Tuple[int, int],
    ) -> np.ndarray:
        """Apply selective blur: sharp inside circle, blurred outside."""
        h, w = frame.shape[:2]
        cx   = int(np.clip(center[0], 0, w - 1))
        cy   = int(np.clip(center[1], 0, h - 1))

        # Full-frame heavy Gaussian blur (background)
        k = self.blur_ksize | 1          # ensure odd
        blurred = cv2.GaussianBlur(frame, (k, k), 0)
        # Apply a second lighter pass for extra cinematic softness
        blurred = cv2.GaussianBlur(blurred, (31, 31), 0)

        # Mask cache — invalidate if center or config changed
        key = (cx, cy, self.focus_radius, self.feather, h, w)
        with self._lock:
            if self._mask_cache is None or self._mask_key != key:
                self._mask_cache = _build_soft_mask(
                    h, w, cx, cy, self.focus_radius, self.feather)
                self._mask_key = key
            mask = self._mask_cache

        mask3     = mask[:, :, np.newaxis]
        composite = (frame.astype(np.float32) * mask3 +
                     blurred.astype(np.float32) * (1.0 - mask3))
        return np.clip(composite, 0, 255).astype(np.uint8)

    def _draw_overlay(
        self,
        frame:  np.ndarray,
        bbox:   Optional[Tuple[int,int,int,int]],
        center: Optional[Tuple[int,int]],
        state:  str,
    ) -> np.ndarray:
        """Draw animated focus ring and status indicator on frame."""
        out = frame.copy()

        if center is None:
            return out

        cx, cy = center

        if state == self.STATE_TRACKING:
            # Green focus ring — outer animated circle
            ring_r = self.focus_radius
            cv2.circle(out, (cx, cy), ring_r,      (0, 220, 80),  2, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), ring_r + 6,  (0, 180, 60),  1, cv2.LINE_AA)
            # Small bright centre dot
            cv2.circle(out, (cx, cy), 4, (0, 255, 120), -1, cv2.LINE_AA)

            # Tracking bbox (transparent overlay)
            if bbox:
                x, y, bw, bh = bbox
                cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 200, 80), 1, cv2.LINE_AA)

        elif state == self.STATE_GRACE:
            # Orange ring — subject temporarily lost
            ring_r = self.focus_radius
            cv2.circle(out, (cx, cy), ring_r, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 4,      (0, 165, 255), -1, cv2.LINE_AA)

            # "Searching…" label
            _put_label(out, "Searching...", cx, cy - ring_r - 14,
                       (0, 165, 255))

        return out

    def _reset_locked(self):
        """Reset all tracking state. Must be called with lock held."""
        self._tracker    = None
        self._state      = self.STATE_IDLE
        self._bbox       = None
        self._center     = None
        self._loss_time  = None
        self._mask_cache = None


# ─────────────────────────────────────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _put_label(
    img:   np.ndarray,
    text:  str,
    cx:    int,
    y:     int,
    color: tuple,
    scale: float = 0.55,
) -> None:
    """Centered text with black drop-shadow."""
    font      = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tx = cx - tw // 2
    ty = max(th + 4, y)
    # Shadow
    cv2.putText(img, text, (tx + 1, ty + 1), font, scale, (0, 0, 0),     thickness + 1, cv2.LINE_AA)
    # Text
    cv2.putText(img, text, (tx,     ty),     font, scale, color,          thickness,     cv2.LINE_AA)
