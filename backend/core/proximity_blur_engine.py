"""
ProximityBlurEngine
===================
Lightweight, real-time cinematic blur engine.

  Feature 1 — Proximity-Based Adaptive Blur
    Distance from selected subject → linear blur strength mapping.
    Selected subject = 0 blur.  Nearby objects = light blur.  Far = heavy.

  Feature 2 — Smooth Transition Engine
    current_blur = 0.8 × prev_blur + 0.2 × new_blur  (per pixel)
    Rack-focus animation over N_TRANSITION_FRAMES when subject changes.

  Feature 3 — Smart Auto Re-Focus
    If tracker is lost / confidence low: scan all detections, lock onto
    the person nearest to the last known position. HUD badge shown.

  Feature 4 — Focus Stability
    Bbox exponential smoothing: 0.7 × prev_box + 0.3 × new_box.
    Prevents tracker jitter.

  Feature 5 — Performance Guardrail
    All processing at ≤ 640 × 480.
    One pre-computed 3-layer blur pass per frame.
    No segmentation or depth model.
    Target: 20-30 FPS on CPU.

Design:
  • State machine identical to TrackingAutofocusEngine (IDLE/TRACKING/GRACE/LOST).
  • Accepts optional YOLO-style detection list via feed_detections().
  • Falls back gracefully if no detections (uses track bbox only).
"""
from __future__ import annotations

import time
import threading
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── constants ────────────────────────────────────────────────────────────────
PROC_W, PROC_H   = 640, 480          # Feature 5: max processing resolution
MIN_BLUR_K       = 5                  # smallest kernel (adjacent subject)
MAX_BLUR_K       = 35                 # largest kernel (far background)
BBOX_SMOOTH_ALPHA = 0.70              # Feature 4: bbox temporal blend weight
BLUR_TEMPORAL_ALPHA = 0.80            # Feature 2: blur-map temporal blend
N_TRANSITION_FRAMES = 20              # rack-focus ramp length (frames)
FEATHER_PX        = 10               # Feature 1 step 5: edge feather (px)
AUTO_REFOCUS_BADGE_FRAMES = 60        # how long "AI Re-Focused" badge shows


def _create_tracker() -> cv2.Tracker:
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, 'TrackerMIL_create'):
        return cv2.TrackerMIL_create()
    raise RuntimeError("No OpenCV tracker available.")


def _feathered_rect_mask(
    h: int, w: int,
    x1: int, y1: int, x2: int, y2: int,
    feather: int,
) -> np.ndarray:
    """
    Float32 H×W mask: 1.0 inside rect, fades to 0 over `feather` px.
    Feature 5, step 5: prevents hard subject-rectangle cutout.
    """
    Y, X = np.ogrid[:h, :w]
    d_left   = (X - x1).astype(np.float32)
    d_right  = (x2 - X).astype(np.float32)
    d_top    = (Y - y1).astype(np.float32)
    d_bottom = (y2 - Y).astype(np.float32)
    d = np.minimum(np.minimum(d_left, d_right), np.minimum(d_top, d_bottom))
    return np.clip(d / max(float(feather), 1.0), 0.0, 1.0)


def _put_label(
    img: np.ndarray, text: str,
    cx: int, y: int, color: tuple,
    scale: float = 0.55,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    tw, _ = cv2.getTextSize(text, font, scale, 1)[0]
    tx, ty = cx - tw // 2, max(14, y)
    cv2.putText(img, text, (tx+1, ty+1), font, scale, (0,0,0),   2, cv2.LINE_AA)
    cv2.putText(img, text, (tx,   ty  ), font, scale, color,     1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────

class ProximityBlurEngine:
    """
    Lightweight proximity-based cinematic blur.
    No segmentation · no depth model · single Gaussian pass per frame.
    """

    STATE_IDLE     = "idle"
    STATE_TRACKING = "tracking"
    STATE_GRACE    = "grace"
    STATE_LOST     = "lost"

    def __init__(self, config: dict = None):
        cfg = config or {}

        self.grace_period:    float = cfg.get('grace_period',    0.8)
        self.bbox_size:       int   = cfg.get('bbox_size',       120)
        self.min_blur_k:      int   = cfg.get('min_blur_k',      MIN_BLUR_K)
        self.max_blur_k:      int   = cfg.get('max_blur_k',      MAX_BLUR_K)
        self.feather:         int   = cfg.get('feather',         FEATHER_PX)
        self.bbox_alpha:      float = cfg.get('bbox_alpha',      BBOX_SMOOTH_ALPHA)
        self.blur_alpha:      float = cfg.get('blur_alpha',      BLUR_TEMPORAL_ALPHA)
        self.transition_frames: int = cfg.get('transition_frames', N_TRANSITION_FRAMES)

        # Internal state
        self._lock            = threading.Lock()
        self._state           = self.STATE_IDLE
        self._tracker                             = None
        self._bbox: Optional[Tuple[int,int,int,int]] = None   # x,y,w,h (proc coords)
        self._smooth_bbox: Optional[np.ndarray]      = None   # float x,y,w,h
        self._center: Optional[Tuple[int,int]]        = None   # proc coords
        self._loss_time: Optional[float]              = None
        self._frame_count                             = 0

        # Detections fed each frame (compatible with YOLO-seg format)
        self._detections: List[dict] = []

        # Temporal blur map (Feature 2)
        self._prev_blur_map: Optional[np.ndarray] = None
        # Rack-focus ramp counter (Feature 2)
        self._ramp_frame: int = 0

        # Auto re-focus badge (Feature 3)
        self._refocus_badge_counter: int = 0

        # Pending click
        self._pending_click: Optional[Tuple[int,int]] = None

        # Scale from original frame → proc frame (updated each process_frame call)
        self._last_scale:  float = 1.0
        self._last_orig_w: int   = PROC_W
        self._last_orig_h: int   = PROC_H
        self._last_proc_w: int   = PROC_W
        self._last_proc_h: int   = PROC_H

        print(f"✓ ProximityBlurEngine  "
              f"blur=[{self.min_blur_k}-{self.max_blur_k}]px  "
              f"feather={self.feather}px  grace={self.grace_period}s")

    # ── public API (mirrors TrackingAutofocusEngine) ─────────────────────────

    def on_click(self, x: int, y: int):
        with self._lock:
            self._pending_click = (int(x), int(y))
        print(f"[PROX] Click queued → ({x}, {y})")

    def on_double_click(self):
        with self._lock:
            self._reset_locked()
        print("[PROX] Reset (double-click)")

    def set_focus_radius(self, radius: int):
        pass   # API compatibility — not used in proximity mode

    def set_blur_strength(self, strength: float):
        # Map 0-1 to kernel range
        k = max(3, int(strength * self.max_blur_k) | 1)
        with self._lock:
            self.max_blur_k = k
        print(f"[PROX] max_blur_k → {k}")

    def feed_detections(self, detections: list):
        """Accept YOLO/any bbox detections for this frame."""
        with self._lock:
            self._detections = list(detections)

    # alias so callers can use same API as TrackingAutofocusEngine
    def feed_seg_detections(self, detections: list):
        self.feed_detections(detections)

    def get_status(self) -> dict:
        with self._lock:
            return {
                'state':        self._state,
                'center':       self._center,
                'bbox':         self._bbox,
                'focus_radius': 0,
                'blur_ksize':   self.max_blur_k,
            }

    def cleanup(self):
        with self._lock:
            self._reset_locked()
        print("✓ ProximityBlurEngine cleanup")

    # ── main entry point ─────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return blur-composited frame (original when IDLE)."""
        self._frame_count += 1
        orig_h, orig_w = frame.shape[:2]

        # ── Feature 5: downscale for processing ─────────────────────
        scale_x = scale_y = 1.0
        if orig_w > PROC_W or orig_h > PROC_H:
            scale_x = PROC_W / orig_w
            scale_y = PROC_H / orig_h
            scale   = min(scale_x, scale_y)
            proc    = cv2.resize(frame, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_LINEAR)
            scale_x = scale_y = scale
        else:
            proc = frame

        ph, pw = proc.shape[:2]
        # Store for coord-space helpers (detect bboxes come in original coords)
        with self._lock:
            self._last_scale  = scale_x       # same in both axes
            self._last_orig_w = orig_w
            self._last_orig_h = orig_h
            self._last_proc_w = pw
            self._last_proc_h = ph
        # ── Consume pending click ────────────────────────────────────
        with self._lock:
            pending = self._pending_click
            if pending is not None:
                self._pending_click = None

        if pending is not None:
            # Scale click to proc coords
            pcx = int(pending[0] * scale_x)
            pcy = int(pending[1] * scale_y)
            self._init_tracker(proc, pcx, pcy)

        # ── Update tracker ───────────────────────────────────────────
        with self._lock:
            state = self._state

        if state == self.STATE_TRACKING:
            self._update_tracker(proc)

        # ── Read state ───────────────────────────────────────────────
        with self._lock:
            state  = self._state
            center = self._center
            bbox   = self._bbox

        # ── Feature 3: auto re-focus if lost ─────────────────────────
        if state in (self.STATE_GRACE, self.STATE_LOST) or (
                state == self.STATE_IDLE and self._prev_blur_map is not None):
            self._try_auto_refocus(proc)
            with self._lock:
                state  = self._state
                center = self._center
                bbox   = self._bbox

        # ── Render ──────────────────────────────────────────────────
        if state in (self.STATE_TRACKING, self.STATE_GRACE) and center is not None:
            proc_out = self._apply_proximity_blur(proc, center, bbox, ph, pw)
            proc_out = self._draw_overlay(proc_out, bbox, center, state)
        else:
            # IDLE: subtle whole-frame vignette so user can see mode is ready
            k     = (self.max_blur_k | 1)
            light = cv2.GaussianBlur(proc, (k, k), 0)
            proc_out = cv2.addWeighted(light, 0.40, proc, 0.60, 0)
            _put_label(proc_out, "Tap a subject to focus",
                       pw // 2, ph - 18, (200, 220, 255), scale=0.55)

        # ── Feature 5: upscale back to original resolution ──────────
        if proc_out.shape[:2] != (orig_h, orig_w):
            proc_out = cv2.resize(proc_out, (orig_w, orig_h),
                                  interpolation=cv2.INTER_LINEAR)

        return proc_out

    # ── core blur rendering ──────────────────────────────────────────────────

    def _apply_proximity_blur(
        self,
        frame:  np.ndarray,
        center: Tuple[int, int],
        bbox:   Optional[Tuple[int, int, int, int]],
        h: int, w: int,
    ) -> np.ndarray:
        """
        Feature 1 + 2 + 4 + 5:
          1. Compute per-region blur strength from proximity.
          2. Pre-compute 3 blur layers (light/medium/heavy).
          3. Build spatial weight map via feathered rects.
          4. Temporal smooth the map.
          5. Piecewise blend between layers.
        """
        frame_f = frame.astype(np.float32)
        diag    = (h * h + w * w) ** 0.5   # frame diagonal for normalising

        # ── Feature 5, step 4: pre-compute 3 blur layers ────────────
        k_h = max(3, self.max_blur_k | 1)
        k_m = max(3, (self.max_blur_k // 2) | 1)
        k_l = max(3, (self.max_blur_k // 5) | 1)

        blur_heavy  = cv2.GaussianBlur(frame, (k_h, k_h), 0).astype(np.float32)
        blur_medium = cv2.GaussianBlur(frame, (k_m, k_m), 0).astype(np.float32)
        blur_light  = cv2.GaussianBlur(frame, (k_l, k_l), 0).astype(np.float32)

        # Start with maximum blur everywhere (background = full blur)
        # blur_weight: 0.0 = sharp, 1.0 = max blur
        blur_w = np.ones((h, w), dtype=np.float32)

        # ── Feature 1, step 2-3: per-detection proximity blur ────────
        with self._lock:
            dets = list(self._detections)
            bbox_alpha = self.bbox_alpha

        for det in dets:
            # Map detection bbox to proc space unconditionally
            dbx1, dby1, dbx2, dby2 = self._det_bbox_to_proc(det['bbox'])
            dcx = (dbx1 + dbx2) // 2
            dcy = (dby1 + dby2) // 2

            # Distance from detection centre to selected subject centre
            dist   = ((dcx - center[0]) ** 2 + (dcy - center[1]) ** 2) ** 0.5
            norm_d = min(dist / max(diag, 1.0), 1.0)   # [0, 1]

            # Linear map: close → min_blur_ratio, far → 1.0
            min_ratio = self.min_blur_k / max(self.max_blur_k, 1)
            strength  = min_ratio + (1.0 - min_ratio) * norm_d    # [min_ratio, 1.0]

            # Feature 1 step 5: feathered rect mask for this detection
            det_mask = _feathered_rect_mask(h, w, dbx1, dby1, dbx2, dby2,
                                            self.feather)
            # Where det_mask == 1 → override blur_w with this detection's strength
            blur_w = blur_w * (1.0 - det_mask) + strength * det_mask

        # ── Selected subject itself → 0 blur (fully sharp) ───────────
        if bbox is not None:
            x, y, bw, bh = [int(v) for v in bbox]
            subj_mask = _feathered_rect_mask(h, w, x, y, x + bw, y + bh,
                                             self.feather)
            blur_w = blur_w * (1.0 - subj_mask)   # force 0 on subject

        # ── Feature 2: temporal blend ─────────────────────────────────
        with self._lock:
            prev = self._prev_blur_map
            ramp = self._ramp_frame

        if prev is not None and prev.shape == (h, w):
            # Rack-focus ramp: gradual increase over transition_frames
            ramp_t = min(1.0, ramp / max(self.transition_frames, 1))
            alpha  = self.blur_alpha * ramp_t + (1.0 - ramp_t) * 0.5
            blur_w = alpha * prev + (1.0 - alpha) * blur_w
        with self._lock:
            self._prev_blur_map = blur_w
            self._ramp_frame    = min(ramp + 1, self.transition_frames)

        # ── Feature 5, step 4: piecewise 3-layer blend ───────────────
        # blur_w in [0,    0.4)  → lerp   original  → light
        # blur_w in [0.4,  0.7)  → lerp   light     → medium
        # blur_w in [0.7,  1.0]  → lerp   medium    → heavy
        t = blur_w[:, :, np.newaxis]

        a1   = np.clip(t / 0.4, 0.0, 1.0)
        out1 = frame_f * (1.0 - a1) + blur_light  * a1

        a2   = np.clip((t - 0.4) / 0.3, 0.0, 1.0)
        out2 = blur_light * (1.0 - a2) + blur_medium * a2

        a3   = np.clip((t - 0.7) / 0.3, 0.0, 1.0)
        out3 = blur_medium * (1.0 - a3) + blur_heavy * a3

        c1 = (blur_w <  0.4).astype(np.float32)[:, :, np.newaxis]
        c2 = ((blur_w >= 0.4) & (blur_w < 0.7)).astype(np.float32)[:, :, np.newaxis]
        c3 = (blur_w >= 0.7).astype(np.float32)[:, :, np.newaxis]

        composite = out1 * c1 + out2 * c2 + out3 * c3
        return np.clip(composite, 0, 255).astype(np.uint8)

    # ── tracker management ───────────────────────────────────────────────────

    def _init_tracker(self, frame: np.ndarray, cx: int, cy: int):
        h, w = frame.shape[:2]

        # Snap to nearest detection if available
        snap = self._snap_to_detection(cx, cy)
        if snap is not None:
            bx1, by1, bx2, by2 = self._det_bbox_to_proc(snap['bbox'])
            # Widen box slightly for CSRT stability
            pad_x = int((bx2 - bx1) * 0.15)
            pad_y = int((by2 - by1) * 0.15)
            bx1 = max(0, bx1 - pad_x); by1 = max(0, by1 - pad_y)
            bx2 = min(w, bx2 + pad_x); by2 = min(h, by2 + pad_y)
            bw, bh = bx2 - bx1, by2 - by1
            bbox    = (bx1, by1, bw, bh)
            cx, cy  = (bx1 + bx2) // 2, (by1 + by2) // 2
            print(f"[PROX] Snapped to detection  bbox={bbox}")
        else:
            half = self.bbox_size // 2
            x1 = max(0, cx - half); y1 = max(0, cy - half)
            bw = min(self.bbox_size, w - x1)
            bh = min(self.bbox_size, h - y1)
            bbox = (x1, y1, bw, bh)

        if bbox[2] < 8 or bbox[3] < 8:
            return

        tracker = _create_tracker()
        try:
            ok = tracker.init(frame, bbox)
        except Exception as e:
            print(f"[PROX] Tracker init error: {e}")
            return
        if ok is False:
            return

        with self._lock:
            self._tracker     = tracker
            self._bbox        = bbox
            self._smooth_bbox = np.array(bbox, dtype=np.float32)
            self._center      = (cx, cy)
            self._state       = self.STATE_TRACKING
            self._loss_time   = None
            # Feature 2: reset ramp on new subject
            self._prev_blur_map = None
            self._ramp_frame    = 0
            self._refocus_badge_counter = 0

        print(f"[PROX] Tracker initialized  center=({cx},{cy})")

    def _update_tracker(self, frame: np.ndarray):
        with self._lock:
            tracker = self._tracker

        if tracker is None:
            return
        try:
            success, raw_bbox = tracker.update(frame)
        except Exception:
            success = False

        with self._lock:
            if success:
                x, y, bw, bh = [int(v) for v in raw_bbox]

                # ── Feature 4: bbox exponential smoothing ─────────────
                if self._smooth_bbox is None:
                    self._smooth_bbox = np.array([x, y, bw, bh], dtype=np.float32)
                else:
                    self._smooth_bbox = (
                        self.bbox_alpha * self._smooth_bbox +
                        (1.0 - self.bbox_alpha) * np.array([x, y, bw, bh],
                                                            dtype=np.float32)
                    )
                sx, sy, sbw, sbh = [int(v) for v in self._smooth_bbox]
                self._bbox   = (sx, sy, sbw, sbh)
                self._center = (sx + sbw // 2, sy + sbh // 2)
                self._state  = self.STATE_TRACKING
                self._loss_time = None
            else:
                if self._state == self.STATE_TRACKING:
                    self._state     = self.STATE_GRACE
                    self._loss_time = time.time()
                elif self._state == self.STATE_GRACE:
                    if time.time() - (self._loss_time or 0) >= self.grace_period:
                        self._reset_locked()

    # ── Feature 3: auto re-focus ─────────────────────────────────────────────

    def _try_auto_refocus(self, frame: np.ndarray):
        """
        Find the person nearest to the last known centre and re-init tracker.
        Only fires when state is GRACE, LOST, or IDLE with prior subject.
        """
        with self._lock:
            last_center = self._center
            dets        = list(self._detections)

        if not dets or last_center is None:
            return

        h, w = frame.shape[:2]
        lx, ly = last_center
        best, best_dist = None, float('inf')
        for det in dets:
            bx1, by1, bx2, by2 = self._scale_bbox_to_proc(
                det['bbox'], None, h, w, already_proc=True)
            dcx = (bx1 + bx2) // 2
            dcy = (by1 + by2) // 2
            d = ((dcx - lx) ** 2 + (dcy - ly) ** 2) ** 0.5
            if d < best_dist:
                best_dist, best = d, det

        if best is None:
            return

        bx1, by1, bx2, by2 = self._scale_bbox_to_proc(
            best['bbox'], None, h, w, already_proc=True)
        self._init_tracker(frame, (bx1 + bx2) // 2, (by1 + by2) // 2)

        with self._lock:
            self._refocus_badge_counter = AUTO_REFOCUS_BADGE_FRAMES
        print("[PROX] Auto re-focused on nearest subject")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _det_bbox_to_proc(
        self, bbox: Tuple[int,int,int,int]
    ) -> Tuple[int,int,int,int]:
        """Map a detection bbox from original-frame coords to proc coords."""
        with self._lock:
            s  = self._last_scale
            pw = self._last_proc_w
            ph = self._last_proc_h
        x1, y1, x2, y2 = bbox
        return (
            max(0, int(x1 * s)), max(0, int(y1 * s)),
            min(pw, int(x2 * s)), min(ph, int(y2 * s)),
        )

    def _snap_to_detection(self, cx: int, cy: int) -> Optional[dict]:
        """Return detection containing or nearest click (proc coords)."""
        with self._lock:
            dets = list(self._detections)
        if not dets:
            return None

        # Prefer detections containing click (coords mapped to proc space)
        containing = []
        for d in dets:
            px1, py1, px2, py2 = self._det_bbox_to_proc(d['bbox'])
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                containing.append(d)
        pool = containing if containing else dets
        best, bd = None, float('inf')
        for d in pool:
            px1, py1, px2, py2 = self._det_bbox_to_proc(d['bbox'])
            dcx, dcy = (px1 + px2) // 2, (py1 + py2) // 2
            dist = ((cx - dcx)**2 + (cy - dcy)**2) ** 0.5
            if dist < bd:
                bd, best = dist, d
        return best

    @staticmethod
    def _scale_bbox_to_proc(
        bbox: Tuple[int, int, int, int],
        frame_shape,          # original frame shape (h,w,c) — unused if already_proc
        proc_h: int,
        proc_w: int,
        already_proc: bool = False,
    ) -> Tuple[int, int, int, int]:
        """
        Convert bbox [x1,y1,x2,y2] from original to proc resolution.
        If already_proc=True returns as-is (clamped).
        """
        x1, y1, x2, y2 = bbox
        if already_proc or frame_shape is None:
            return (max(0, x1), max(0, y1),
                    min(proc_w, x2), min(proc_h, y2))
        orig_h, orig_w = frame_shape[:2]
        sx = proc_w / max(orig_w, 1)
        sy = proc_h / max(orig_h, 1)
        return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

    def _draw_overlay(
        self,
        frame: np.ndarray,
        bbox:  Optional[Tuple[int,int,int,int]],
        center: Optional[Tuple[int,int]],
        state: str,
    ) -> np.ndarray:
        out = frame.copy()
        if center is None:
            return out
        h, w = out.shape[:2]
        cx, cy = center

        # Feature 3 badge
        with self._lock:
            badge_ct = self._refocus_badge_counter
            if badge_ct > 0:
                self._refocus_badge_counter -= 1

        if badge_ct > 0:
            _put_label(out, "AI Re-Focused", w // 2, 28,
                       (0, 255, 180), scale=0.65)

        if state == self.STATE_TRACKING and bbox is not None:
            x, y, bw, bh = [int(v) for v in bbox]
            x2_, y2_ = x + bw, y + bh
            # Outer glow
            cv2.rectangle(out, (x-2, y-2), (x2_+2, y2_+2),
                          (0, 140, 50), 2, cv2.LINE_AA)
            # Main rect
            cv2.rectangle(out, (x, y), (x2_, y2_),
                          (0, 230, 90), 1, cv2.LINE_AA)
            # Corner marks
            for px_, py_, dx_, dy_ in [
                (x, y, 1, 1), (x2_, y, -1, 1),
                (x, y2_, 1, -1), (x2_, y2_, -1, -1),
            ]:
                cv2.line(out, (px_, py_), (px_+dx_*12, py_), (0,255,120), 2, cv2.LINE_AA)
                cv2.line(out, (px_, py_), (px_, py_+dy_*12), (0,255,120), 2, cv2.LINE_AA)
            _put_label(out, "Focus Lock", (x+x2_)//2, y-10,
                       (0, 230, 90), scale=0.48)
            cv2.circle(out, (cx, cy), 4, (0,255,120), -1, cv2.LINE_AA)

        elif state == self.STATE_GRACE:
            if bbox is not None:
                x, y, bw, bh = [int(v) for v in bbox]
                cv2.rectangle(out, (x,y), (x+bw,y+bh), (0,165,255), 1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 6, (0,165,255), -1, cv2.LINE_AA)
            _put_label(out, "Searching...", cx, cy - 20, (0, 165, 255))

        return out

    def _reset_locked(self):
        """Must be called with lock held."""
        self._tracker         = None
        self._state           = self.STATE_IDLE
        self._bbox            = None
        self._smooth_bbox     = None
        self._center          = None
        self._loss_time       = None
        self._prev_blur_map   = None
        self._ramp_frame      = 0
