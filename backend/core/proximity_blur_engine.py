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

from core.motion_utils import SpringPhysics, EASE_CINEMATIC

# ── constants ────────────────────────────────────────────────────────────────
PROC_W, PROC_H   = 640, 480          # Feature 5: max processing resolution
MIN_BLUR_K       = 5                  # smallest kernel (adjacent subject)
MAX_BLUR_K       = 35                 # largest kernel (far background)
FEATHER_PX        = 20               # Feature 4: soft edge feather (px)
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
    Uses distance transform-like logic for correct corner rounding.
    """
    Y, X = np.ogrid[:h, :w]
    
    # Calculate distance to the rectangle edge
    # interior is positive, exterior is negative
    # x dist
    dx = np.maximum(x1 - X, X - x2)
    # y dist
    dy = np.maximum(y1 - Y, Y - y2)
    
    # max(dx, dy) gives the L-inf distance to the box
    # If both are negative, we are inside.
    # If one is positive, we are outside.
    
    dist = np.maximum(dx, dy)
    
    # If dist < 0 (inside), mask = 1.0
    # If dist > 0 (outside), mask fades from 1.0 to 0.0 over `feather` pixels
    
    # Invert so inside is high value
    # We want 1.0 at dist <= 0
    # We want 0.0 at dist >= feather
    
    mask = np.clip(1.0 - (dist / feather), 0.0, 1.0)
    
    # Apply cubic ease out for smoother falloff
    mask = mask * mask * (3 - 2 * mask)
    
    return mask


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
    "Cinematic Focus Engine" (formerly ProximityBlurEngine)
    Ultra-premium rack focus system with Apple-style spring physics.
    
    Architecture:
    - Uses dual-layer masking for "rack focus" effect (crossfading between subjects).
    - Uses SpringPhysics for smoothbbox tracking to avoid jitter.
    - Implements cubic-bezier blur interpolation.
    """

    STATE_IDLE     = "idle"
    STATE_TRACKING = "tracking"
    STATE_GRACE    = "grace"
    STATE_LOST     = "lost"
    STATE_RACKING  = "racking"  # New state for switching subjects

    def __init__(self, config: dict = None):
        cfg = config or {}

        self.grace_period:    float = cfg.get('grace_period',    0.8)
        self.bbox_size:       int   = cfg.get('bbox_size',       120)
        self.min_blur_k:      int   = cfg.get('min_blur_k',      MIN_BLUR_K)
        self.max_blur_k:      int   = cfg.get('max_blur_k',      MAX_BLUR_K)
        self.feather:         int   = cfg.get('feather',         FEATHER_PX)

        # Internal state
        self._lock            = threading.Lock()
        self._state           = self.STATE_IDLE
        self._tracker         = None
        
        # Physics Engines for BBox (Current active subject)
        # Using Apple-style spring parameters (stiffness 170, damping 26)
        self._phys_x = SpringPhysics(stiffness=200, damping=25)
        self._phys_y = SpringPhysics(stiffness=200, damping=25)
        self._phys_w = SpringPhysics(stiffness=180, damping=30)
        self._phys_h = SpringPhysics(stiffness=180, damping=30)
        
        # Blur transition state
        self._current_blur_map: Optional[np.ndarray] = None
        
        # Rack Focus State
        # When switching subjects, we crossfade from _prev_mask to _curr_mask
        self._rack_start_time = 0
        self._rack_duration = 0.5 # seconds
        self._is_racking = False
        self._prev_focus_snapshot: Optional[Tuple[int,int,int,int]] = None # The box where we WERE focused
        
        # Raw tracker output
        self._raw_bbox: Optional[Tuple[int,int,int,int]] = None 

        self._frame_count                             = 0

        # Detections fed each frame (compatible with YOLO-seg format)
        self._detections: List[dict] = []

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

        print(f"✓ CinematicFocusEngine Initialized (Spring Physics + Cubic Bezier)")

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
        # Store for coord-space helpers
        with self._lock:
            self._last_scale  = scale_x
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
        if state in (self.STATE_GRACE, self.STATE_LOST):
            self._try_auto_refocus(proc)
            with self._lock:
                state  = self._state
                bbox   = self._bbox
                center = self._center

        # ── Render ──────────────────────────────────────────────────
        if state in (self.STATE_TRACKING, self.STATE_GRACE) and center is not None:
            proc_out = self._apply_proximity_blur(proc, center, bbox, ph, pw)
            # Optional overlay could go here, omitting for pure cinematic look
            # proc_out = self._draw_overlay(proc_out, bbox, center, state)
            
            # Draw minimal "Searching" UI only in grace state
            if state == self.STATE_GRACE:
                _put_label(proc_out, "Searching...", center[0], center[1]-20, (0, 165, 255))
        else:
            # IDLE: subtle whole-frame vignette
            k     = (self.max_blur_k | 1)
            blur_bg = cv2.GaussianBlur(proc, (k, k), 0)
            proc_out = cv2.addWeighted(blur_bg, 0.3, proc, 0.7, 0)
            _put_label(proc_out, "Tap subject to focus",
                       pw // 2, ph - 20, (255, 255, 255), scale=0.6)

        # ── Feature 5: upscale back to original resolution ──────────
        if proc_out.shape[:2] != (orig_h, orig_w):
            proc_out = cv2.resize(proc_out, (orig_w, orig_h),
                                  interpolation=cv2.INTER_LINEAR)

        return proc_out

    # ── core blur rendering ──────────────────────────────────────────────────

    def _apply_proximity_blur(
        self,
        frame:  np.ndarray,
        center: Optional[Tuple[int, int]],
        bbox:   Optional[Tuple[int, int, int, int]],
        h: int, w: int,
    ) -> np.ndarray:
        """
        Ultra-Premium Cinematic Blur Renderer (Spring + Cubic Bezier)
        """
        # 1. Update Physics
        if bbox is not None:
             tx, ty, tw, th = bbox
             # Spring Physics Step
             cx = self._phys_x.update(tx)
             cy = self._phys_y.update(ty)
             cw = self._phys_w.update(tw)
             ch = self._phys_h.update(th)
             draw_box = (cx, cy, cw, ch)
        else:
             draw_box = None

        # 2. Handle Rack Focus State / Mask Generation
        now = time.time()
        
        # Calculate primary mask (Current Target)
        if draw_box:
            bx, by, bw, bh = draw_box
            mask_main = _feathered_rect_mask(h, w, int(bx), int(by), int(bx+bw), int(by+bh), self.feather)
        else:
            mask_main = np.zeros((h, w), dtype=np.float32)

        # Rack Focus Interpolation
        if self._is_racking and self._prev_focus_snapshot:
            progress = (now - self._rack_start_time) / self._rack_duration
            if progress >= 1.0:
                self._is_racking = False
            else:
                # Cubic Bezier Ease
                t = EASE_CINEMATIC.solve(progress)
                
                # Render Old Focal Plane
                ox, oy, ow, oh = self._prev_focus_snapshot
                mask_old = _feathered_rect_mask(h, w, ox, oy, ox+ow, oy+oh, self.feather)
                
                # Cinematic Cross-Dissolve: Linear mix of the masks
                mask_main = mask_old * (1.0 - t) + mask_main * t

        # 3. Render Blur Layers (Tri-Linear Blend)
        k_h = self.max_blur_k | 1
        k_m = (self.max_blur_k // 2) | 1
        k_l = (self.max_blur_k // 5) | 1
        k_m = max(1, k_m); k_l = max(1, k_l)

        frame_f = frame.astype(np.float32)
        b_heavy  = cv2.GaussianBlur(frame, (k_h, k_h), 0).astype(np.float32)
        b_medium = cv2.GaussianBlur(frame, (k_m, k_m), 0).astype(np.float32)
        b_light  = cv2.GaussianBlur(frame, (k_l, k_l), 0).astype(np.float32)

        # 4. Composite
        # Blur Map: 0.0=Sharp, 1.0=Heavy Blur
        blur_map = 1.0 - mask_main 
        blur_map = np.clip(blur_map, 0.0, 1.0)
        blur_map_3c = blur_map[:, :, np.newaxis]

        # Vectorized tri-linear blend
        w_sharp_light = np.clip(blur_map_3c * 3.0, 0.0, 1.0)
        out = frame_f * (1.0 - w_sharp_light) + b_light * w_sharp_light
        
        w_light_med = np.clip((blur_map_3c - 0.333) * 3.0, 0.0, 1.0)
        out = out * (1.0 - w_light_med) + b_medium * w_light_med
        
        w_med_heavy = np.clip((blur_map_3c - 0.666) * 3.0, 0.0, 1.0)
        out = out * (1.0 - w_med_heavy) + b_heavy * w_med_heavy
        
        return np.clip(out, 0, 255).astype(np.uint8)

    # ── tracker management ───────────────────────────────────────────────────

    def _init_tracker(self, frame: np.ndarray, cx: int, cy: int):
        h, w = frame.shape[:2]
        
        # Snapshot for rack focus if we are ALREADY tracking something valid
        with self._lock:
            if self._state == self.STATE_TRACKING and self._bbox is not None:
                # Capture where the physics engine currently IS
                self._prev_focus_snapshot = (
                    int(self._phys_x.value), int(self._phys_y.value),
                    int(self._phys_w.value), int(self._phys_h.value)
                )
                self._is_racking = True
                self._rack_start_time = time.time()
                print(f"[CINE] Rack focus start...")
            else:
                self._is_racking = False

        # Snap logic
        snap = self._snap_to_detection(cx, cy)
        if snap:
            bx1, by1, bx2, by2 = self._det_bbox_to_proc(snap['bbox'])
            # Cinematic padding
            pad_x, pad_y = int((bx2-bx1)*0.25), int((by2-by1)*0.15)
            bx1 = max(0, bx1-pad_x); by1 = max(0, by1-pad_y)
            bx2 = min(w, bx2+pad_x); by2 = min(h, by2+pad_y)
            bbox = (bx1, by1, bx2-bx1, by2-by1)
        else:
            half = self.bbox_size // 2
            bbox = (max(0, cx-half), max(0, cy-half), self.bbox_size, self.bbox_size)

        if bbox[2] < 10 or bbox[3] < 10: return

        tracker = _create_tracker()
        if not tracker.init(frame, bbox): return

        with self._lock:
            self._tracker = tracker
            self._bbox = bbox
            self._center = (cx, cy)
            self._state = self.STATE_TRACKING
            self._loss_time = None
            
            # If NOT racking (fresh start), snap physics instantly
            if not self._is_racking:
                self._phys_x.reset(bbox[0]); self._phys_y.reset(bbox[1])
                self._phys_w.reset(bbox[2]); self._phys_h.reset(bbox[3])
                print("[CINE] Physics reset")

    def _update_tracker(self, frame: np.ndarray):
        with self._lock:
            tracker = self._tracker
        if not tracker: return

        ok, raw = tracker.update(frame)
        with self._lock:
            if ok:
                x, y, w, h = [int(v) for v in raw]
                self._bbox   = (x, y, w, h)
                self._center = (x + w//2, y + h//2)
                self._state  = self.STATE_TRACKING
                self._loss_time = None
                # Physics updates happen in _apply_proximity_blur
            else:
                if self._state == self.STATE_TRACKING:
                    self._state, self._loss_time = self.STATE_GRACE, time.time()
                elif self._state == self.STATE_GRACE and time.time() - (self._loss_time or 0) >= self.grace_period:
                    self._reset_locked()

    def _try_auto_refocus(self, frame: np.ndarray):
        with self._lock:
            last_c = self._center
            dets = list(self._detections)
        
        if not dets or not last_c: return
        
        lx, ly = last_c
        best_d, min_dist = None, float('inf')
        for d in dets:
            bx1, by1, bx2, by2 = self._det_bbox_to_proc(d['bbox'])
            dcx, dcy = (bx1+bx2)//2, (by1+by2)//2
            dist = np.hypot(dcx-lx, dcy-ly)
            if dist < min_dist:
                min_dist = dist
                best_d = d
        
        if best_d and min_dist < 150:
            bx1, by1, bx2, by2 = self._det_bbox_to_proc(best_d['bbox'])
            self._init_tracker(frame, (bx1+bx2)//2, (by1+by2)//2)
            print("[CINE] Auto-recover focus")

    def _det_bbox_to_proc(self, bbox):
        with self._lock:
            s, pw, ph = self._last_scale, self._last_proc_w, self._last_proc_h
        x1, y1, x2, y2 = bbox
        return (
            max(0, int(x1 * s)), max(0, int(y1 * s)),
            min(pw, int(x2 * s)), min(ph, int(y2 * s))
        )

    def _snap_to_detection(self, cx, cy):
        with self._lock: dets = list(self._detections)
        for d in dets:
            x1, y1, x2, y2 = self._det_bbox_to_proc(d['bbox'])
            if x1 <= cx <= x2 and y1 <= cy <= y2: return d
        return None

    def _draw_overlay(self, *args):
        # Kept for compatibility if called, but unused in new cinematic render
        pass

    def _reset_locked(self):
        """Must be called with lock held."""
        self._tracker         = None
        self._state           = self.STATE_IDLE
        self._bbox            = None
        self._center          = None
        self._loss_time       = None
        self._is_racking      = False
