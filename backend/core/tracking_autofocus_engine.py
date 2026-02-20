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


def _build_body_mask(
    h: int, w: int,
    x1: int, y1: int, x2: int, y2: int,
    feather: int,
) -> np.ndarray:
    """
    Float32 H×W mask: 1.0 deep inside rectangle (x1,y1)-(x2,y2), 0.0 outside.
    Linearly feathered over `feather` pixels at every edge.
    Uses vectorised distance-to-edge — no per-pixel loops.
    """
    Y, X = np.ogrid[:h, :w]
    d_left   = (X - x1).astype(np.float32)
    d_right  = (x2 - X).astype(np.float32)
    d_top    = (Y - y1).astype(np.float32)
    d_bottom = (y2 - Y).astype(np.float32)
    # Minimum distance to any edge (negative = outside rectangle)
    d = np.minimum(np.minimum(d_left, d_right),
                   np.minimum(d_top,  d_bottom))
    return np.clip(d / max(float(feather), 1.0), 0.0, 1.0)


def _expand_bbox_to_body(
    bbox:           Tuple[int, int, int, int],
    frame_h:        int,
    frame_w:        int,
    expand_w:       float = 2.5,
    expand_h:       float = 4.5,
    max_area_frac:  float = 0.70,
) -> Tuple[int, int, int, int]:
    """
    Expand tracker bbox (face region) to approximate full-body coverage.

    Expansion rules:
      • Width  × expand_w  (default 2.5)
      • Height × expand_h  (default 4.5)
      • Vertical split: 20% upward from face centre, 80% downward
        (face sits near the top of the body).

    Adaptive behaviour:
      • If bbox is wide/square (h < w × 1.2) → head viewed flat or far away;
        push expand_h up by ×1.3 to ensure legs are captured.
      • If bbox is already tall (h > w × 2.0) → subject partially cropped;
        pull expand_h back by ×0.75 to avoid over-expanding.

    Returns (x1, y1, x2, y2) clamped to frame boundaries.
    """
    x, y, bw, bh = bbox
    cx = x + bw // 2
    cy = y + bh // 2

    # Adaptive height multiplier based on tracked bbox aspect ratio
    aspect = bh / max(bw, 1)          # > 1 → tall box, < 1 → wide/flat box
    if aspect < 1.2:                  # small/square head → expand aggressively
        h_mult = expand_h * 1.3
    elif aspect > 2.0:                # already tall box → soften expansion
        h_mult = expand_h * 0.75
    else:
        h_mult = expand_h

    new_w = int(bw * expand_w)
    new_h = int(bh * h_mult)

    # Cap to max_area_frac of frame
    max_area = frame_h * frame_w * max_area_frac
    if new_w * new_h > max_area:
        scale = (max_area / (new_w * new_h)) ** 0.5
        new_w = int(new_w * scale)
        new_h = int(new_h * scale)

    # Face is near top of body → 20% upward, 80% downward
    x1 = cx - new_w // 2
    y1 = cy - int(new_h * 0.20)
    x2 = x1 + new_w
    y2 = y1 + new_h

    # Clip to frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    return x1, y1, x2, y2


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
        # Background blur strength
        self.feather:      int   = cfg.get('feather', 30)         # px softness at body edge
        self.blur_ksize:   int   = cfg.get('blur_ksize', 101)     # heavy GaussianBlur
        # Grace period on tracker loss
        self.grace_period: float = cfg.get('grace_period', 0.8)   # seconds
        # Body-region expansion (face click → full-body sharp zone)
        self.body_expand_w:     float = cfg.get('body_expand_w',     2.5)
        self.body_expand_h:     float = cfg.get('body_expand_h',     4.5)
        self.body_max_frac:     float = cfg.get('body_max_frac',     0.70)
        # focus_radius kept for API compatibility (used by set_focus_radius)
        self.focus_radius: int   = cfg.get('focus_radius', 75)

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
            output = self._composite_blur(frame, center, bbox)
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
        bbox:   Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Apply selective blur.
        Sharp zone = expanded body region derived from tracker bbox.
        Falls back to circular mask when no bbox is available.
        """
        h, w = frame.shape[:2]

        # Full-frame heavy Gaussian blur (background)
        k = self.blur_ksize | 1
        blurred = cv2.GaussianBlur(frame, (k, k), 0)
        blurred = cv2.GaussianBlur(blurred, (31, 31), 0)

        if bbox is not None:
            # ── Body-region mask (rectangular, feathered) ──────────────
            ex1, ey1, ex2, ey2 = _expand_bbox_to_body(
                bbox, h, w,
                expand_w=self.body_expand_w,
                expand_h=self.body_expand_h,
                max_area_frac=self.body_max_frac,
            )
            cache_key = (ex1, ey1, ex2, ey2, self.feather, h, w)
            with self._lock:
                if self._mask_cache is None or self._mask_key != cache_key:
                    self._mask_cache = _build_body_mask(
                        h, w, ex1, ey1, ex2, ey2, self.feather)
                    self._mask_key = cache_key
                mask = self._mask_cache
        else:
            # ── Fallback: circular mask centred on click point ────────
            cx   = int(np.clip(center[0], 0, w - 1))
            cy   = int(np.clip(center[1], 0, h - 1))
            cache_key = (cx, cy, self.focus_radius, self.feather, h, w)
            with self._lock:
                if self._mask_cache is None or self._mask_key != cache_key:
                    self._mask_cache = _build_soft_mask(
                        h, w, cx, cy, self.focus_radius, self.feather)
                    self._mask_key = cache_key
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
        """Draw body-region outline and status indicator on frame."""
        out = frame.copy()
        if center is None:
            return out

        h, w = out.shape[:2]
        cx, cy = center

        if state == self.STATE_TRACKING:
            if bbox is not None:
                # Expanded body rectangle
                ex1, ey1, ex2, ey2 = _expand_bbox_to_body(
                    bbox, h, w,
                    expand_w=self.body_expand_w,
                    expand_h=self.body_expand_h,
                    max_area_frac=self.body_max_frac,
                )
                # Outer glow: slightly thicker, darker green
                cv2.rectangle(out, (ex1 - 2, ey1 - 2), (ex2 + 2, ey2 + 2),
                              (0, 140, 50), 2, cv2.LINE_AA)
                # Main body outline: bright green
                cv2.rectangle(out, (ex1, ey1), (ex2, ey2),
                              (0, 230, 90), 1, cv2.LINE_AA)
                # Corner accent marks
                corner = 14
                for px, py, dx, dy in [
                    (ex1, ey1,  1,  1), (ex2, ey1, -1,  1),
                    (ex1, ey2,  1, -1), (ex2, ey2, -1, -1),
                ]:
                    cv2.line(out, (px, py), (px + dx * corner, py), (0, 255, 120), 2, cv2.LINE_AA)
                    cv2.line(out, (px, py), (px, py + dy * corner), (0, 255, 120), 2, cv2.LINE_AA)
                # "Body Focus" label above box
                _put_label(out, "Body Focus", (ex1 + ex2) // 2, ey1 - 10,
                           (0, 230, 90), scale=0.50)
            # Centre dot at face click
            cv2.circle(out, (cx, cy), 4, (0, 255, 120), -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 7, (0, 200, 80),  1, cv2.LINE_AA)

        elif state == self.STATE_GRACE:
            if bbox is not None:
                ex1, ey1, ex2, ey2 = _expand_bbox_to_body(
                    bbox, h, w,
                    expand_w=self.body_expand_w,
                    expand_h=self.body_expand_h,
                    max_area_frac=self.body_max_frac,
                )
                cv2.rectangle(out, (ex1, ey1), (ex2, ey2),
                              (0, 165, 255), 1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 6, (0, 165, 255), -1, cv2.LINE_AA)
            _put_label(out, "Searching...", cx,
                       (ey1 - 10) if bbox else (cy - self.focus_radius - 14),
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
