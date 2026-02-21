"""
TrackingAutofocusEngine
=======================
CSRT-based subject tracker with PIXEL-ACCURATE SEGMENTATION for blur masking.

UPGRADE: Full Silhouette Subject Lock Mode
- Uses AI segmentation (MediaPipe/DeepLabV3) instead of bounding boxes
- Entire person silhouette stays sharp (arms, legs, hair, clothes)
- Only background gets blurred
- Morphological edge refinement prevents halo artifacts

Pipeline per frame:
  frame → tracker.update() → bbox → segmentation → binary mask → composite blur

Falls back to geometric masks if segmentation unavailable.
"""
import time
import threading
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from core.motion_utils import CubicBezier
EASE_CINEMATIC = CubicBezier(0.22, 1.0, 0.36, 1.0)


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

        # ── SEGMENTATION UPGRADE ──────────────────────────────────────
        # Pixel-accurate person segmentation for Full Silhouette Lock
        self.use_segmentation:  bool  = cfg.get('use_segmentation', True)   # Enable segmentation
        self.segmentation_threshold:  float = cfg.get('seg_threshold', 0.5) # Mask confidence
        self.seg_edge_dilation:       int   = cfg.get('seg_dilation', 2)    # Morphological dilation (px)
        self.seg_edge_feather:        int   = cfg.get('seg_feather', 2)     # Edge smoothing (px)
        self.seg_frame_skip:          int   = cfg.get('seg_frame_skip', 0)  # Run seg every N frames (0=every frame)
        self.seg_fallback_expand_pct: float = cfg.get('seg_fallback_expand', 0.20)  # Bbox expand on seg failure
        
        # Segmenter instance (injected via set_segmenter)
        self._segmenter = None
        self._seg_frame_counter = 0
        self._last_seg_mask: Optional[np.ndarray] = None

        # ── INSTANCE SEGMENTATION RESULTS (Phase 2) ──────────────────
        # Fed each frame from YOLOv8SegModule before process_frame()
        self._seg_detections: list             = []   # current frame dets
        self._selected_track_id: Optional[int] = None # locked person ID
        self._last_instance_mask: Optional[np.ndarray] = None  # most recent valid mask
        self._seg_fail_count: int              = 0    # consecutive seg failures
        self.seg_fail_max: int                 = 5    # frames before fallback kicks in

        # ── DEPTH-BASED SMOOTH BLUR (Tasks 1-7) ──────────────────────
        # DepthEstimator injected by AutofocusModule after initialize()
        self._depth_estimator                        = None
        # Per-frame temporal state
        self._last_blur_map: Optional[np.ndarray]    = None   # Task 6 smoothing
        # Tuning
        self.depth_blur_k:          float = cfg.get('depth_blur_k',          3.5)
        self.spatial_threshold_px:  float = cfg.get('spatial_threshold_px',  180.0)
        self.spatial_reduce_frac:   float = cfg.get('spatial_reduce_frac',   0.45)
        self.temporal_alpha:        float = cfg.get('temporal_alpha',         0.80)
        self.rack_focus_duration:   float = cfg.get('rack_focus_duration',    0.40)
        # Rack-focus animation state (Task 7)
        self._depth_ref:            float = 0.5
        self._prev_depth_ref:       float = 0.5
        self._target_depth_ref:     float = 0.5
        self._rack_focus_start:     float = 0.0
        self._rack_focus_active:    bool  = False

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
              f"blur_k={self.blur_ksize}  grace={self.grace_period}s  "
              f"segmentation={'ON' if self.use_segmentation else 'OFF'}")

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

    def set_segmenter(self, segmenter):
        """
        Inject PersonSegmentation module for pixel-accurate masking.
        
        Args:
            segmenter: PersonSegmentation instance
        """
        with self._lock:
            self._segmenter = segmenter
            self._mask_cache = None  # Invalidate cache
        print(f"[TRACKER] Segmenter {'injected' if segmenter else 'removed'}")

    def set_segmentation_enabled(self, enabled: bool):
        """
        Enable/disable segmentation-based masking.
        Falls back to geometric masks when disabled.
        
        Args:
            enabled: True = use segmentation, False = use bbox masks
        """
        with self._lock:
            self.use_segmentation = enabled
            self._mask_cache = None
            self._last_seg_mask = None
        print(f"[TRACKER] Segmentation {'enabled' if enabled else 'disabled'}")

    def set_depth_estimator(self, estimator) -> None:
        """
        Inject a DepthEstimator instance for depth-based cinematic blur.
        Call after initialize() from AutofocusModule.
        """
        with self._lock:
            self._depth_estimator = estimator
        print(f"[TRACKER] DepthEstimator {'injected' if estimator else 'removed'}")

    def feed_seg_detections(self, detections: list):
        """
        Phase 2 — Called by AutofocusModule BEFORE process_frame().

        Stores YOLOv8-seg detection results for the current frame so
        _init_tracker() can snap to the correct person and _composite_blur()
        can use the pixel-accurate instance mask.

        Args:
            detections: list of dicts from YOLOv8SegModule.detect()
                        Each dict: {class, confidence, bbox, track_id, mask}
        """
        with self._lock:
            self._seg_detections = list(detections)

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
        """
        Create / reinitialize CSRT tracker.

        Phase 2 upgrade: if YOLO-seg detections are available for this frame,
        snap the initial bbox to the nearest detected person (gives the tracker
        a clean full-body region instead of a tiny click square) and store the
        track_id so the mask pipeline can lock onto it every frame.
        """
        h, w = frame.shape[:2]

        # ── Try to snap to a YOLO-seg detection ────────────────────────
        snap = self._snap_to_detection(cx, cy)

        if snap is not None:
            # Use the full detected person bbox (much better than click square)
            bx1, by1, bx2, by2 = snap['bbox']
            bw    = bx2 - bx1
            bh    = by2 - by1
            # Apply body padding (20 % W / 30 % H)
            pad_x = int(bw * 0.20)
            pad_y = int(bh * 0.30)
            bx1   = max(0, bx1 - pad_x)
            by1   = max(0, by1 - pad_y)
            bx2   = min(w, bx2 + pad_x)
            by2   = min(h, by2 + pad_y)
            bw, bh = bx2 - bx1, by2 - by1
            bbox   = (bx1, by1, bw, bh)
            tid    = snap.get('track_id')
            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            print(f"[TRACKER] Snapped to detection  track_id={tid}  bbox={bbox}")
        else:
            # Fallback: use fixed-size square centred on click
            half = self.bbox_size // 2
            x1   = max(0, cx - half)
            y1   = max(0, cy - half)
            bw   = min(self.bbox_size, w - x1)
            bh   = min(self.bbox_size, h - y1)
            bbox = (x1, y1, bw, bh)
            tid  = None
            print(f"[TRACKER] No detection found, using click square  bbox={bbox}")

        if bw < 10 or bh < 10:
            print("[TRACKER] bbox too small, skipping init")
            return

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
            self._tracker            = tracker
            self._bbox               = bbox
            self._center             = (cx, cy)
            self._state              = self.STATE_TRACKING
            self._loss_time          = None
            self._mask_cache         = None
            self._selected_track_id  = tid    # lock onto this person
            self._last_instance_mask = None   # clear stale mask
            self._seg_fail_count     = 0
            # Task 7 — rack focus: start animation to new depth reference
            self._prev_depth_ref    = self._depth_ref
            self._target_depth_ref  = self._depth_ref  # refined in first frame
            self._rack_focus_start  = time.time()
            self._rack_focus_active = True
            self._last_blur_map     = None              # clear temporal state

        print(f"[TRACKER] Initialized  center=({cx},{cy})  bbox={bbox}  track_id={tid}")

    def _snap_to_detection(self, click_x: int, click_y: int) -> Optional[Dict]:
        """
        Phase 2 — Find YOLO-seg detection nearest/containing click point.

        Returns the detection dict or None if no detections available.
        """
        with self._lock:
            dets = list(self._seg_detections)

        if not dets:
            return None

        # Prefer detections that CONTAIN the click point
        containing = [
            d for d in dets
            if (d['bbox'][0] <= click_x <= d['bbox'][2] and
                d['bbox'][1] <= click_y <= d['bbox'][3])
        ]
        candidates = containing if containing else dets

        # Among candidates, pick closest centre
        best, best_dist = None, float('inf')
        for d in candidates:
            x1, y1, x2, y2 = d['bbox']
            dc = ((click_x - (x1+x2)//2)**2 + (click_y - (y1+y2)//2)**2) ** 0.5
            if dc < best_dist:
                best_dist, best = dc, d

        return best

    def _get_instance_mask(
        self,
        frame_h: int,
        frame_w: int,
    ) -> Optional[np.ndarray]:
        """
        Phase 2 — Look up the segmentation mask for the selected track_id.

        Returns floating-point mask (H×W) 0.0-1.0 or None.
        """
        with self._lock:
            tid  = self._selected_track_id
            dets = self._seg_detections

        if tid is None or not dets:
            return None

        # Find the detection with matching track id
        match = next((d for d in dets if d.get('track_id') == tid), None)

        if match is None:
            # track momentarily absent — try to keep last mask for one extra frame
            if self._last_instance_mask is not None and self._seg_fail_count < self.seg_fail_max:
                self._seg_fail_count += 1
                return self._last_instance_mask
            return None   # give up; caller will fall back to geometric

        self._seg_fail_count = 0  # reset failure counter on successful match

        raw_mask = match.get('mask')
        if raw_mask is None:
            return None

        # Convert uint8 0/255 → float32 0.0/1.0
        if raw_mask.dtype == np.uint8:
            mask = raw_mask.astype(np.float32) / 255.0
        else:
            mask = raw_mask.astype(np.float32)

        # Resize to frame dimensions if needed
        if mask.shape[:2] != (frame_h, frame_w):
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

        # ── Phase 4: Edge recovery & anti-halo refinement ─────────────
        # 1. Morphological dilation — recover arms, hair, finger edges
        if self.seg_edge_dilation > 0:
            d_k  = self.seg_edge_dilation * 2 + 1
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_k, d_k))
            m8   = (mask * 255).astype(np.uint8)
            mask = cv2.dilate(m8, kern, iterations=1).astype(np.float32) / 255.0

        # 2. Morphological closing — fill small internal holes
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        m8      = (mask * 255).astype(np.uint8)
        mask    = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, close_k).astype(np.float32) / 255.0

        # 3. Remove small isolated noise blobs
        m8_bin = (mask > 0.5).astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m8_bin, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            clean = np.zeros_like(m8_bin)
            clean[labels == largest_label] = 255
            # Also keep any other blobs > 5% of main blob area
            main_area = stats[largest_label, cv2.CC_STAT_AREA]
            for lbl in range(1, num_labels):
                if lbl != largest_label and stats[lbl, cv2.CC_STAT_AREA] > main_area * 0.05:
                    clean[labels == lbl] = 255
            mask = clean.astype(np.float32) / 255.0

        # 4. Edge feathering — prevent hard cutoff (anti-halo, max 2px)
        if self.seg_edge_feather > 0:
            f_k  = max(3, self.seg_edge_feather * 2 + 1)
            mask = cv2.GaussianBlur(mask, (f_k, f_k), 0)

        # ── Phase 5: Coverage guarantee ───────────────────────────────
        # Validate mask area relative to reported bbox
        x1, y1, x2, y2 = match['bbox']
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        mask_area = float(np.sum(mask > 0.5))
        coverage  = mask_area / bbox_area

        if coverage < 0.25:
            # Under-segmented — temporarily disable blur inside bbox,
            # return None to trigger the geometric fallback expansion
            print(f"[TRACKER] Low mask coverage ({coverage:.1%}) — using fallback")
            return None

        self._last_instance_mask = mask
        return mask

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

    def _build_segmentation_mask(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        Build pixel-accurate person segmentation mask.
        
        FULL SILHOUETTE SUBJECT LOCK MODE:
        - Uses AI segmentation (MediaPipe/DeepLabV3)
        - Returns binary mask: 1.0 = person, 0.0 = background
        - Applies morphological dilation to recover edges (arms, fingers, hair)
        - Feathers edges to prevent hard cutoff
        
        Args:
            frame: BGR image (H×W×3)
            bbox: Tracker bbox (x, y, w, h) - used as ROI hint
        
        Returns:
            Binary mask (H×W float32) or None if segmentation fails
        """
        if self._segmenter is None:
            return None
        
        h, w = frame.shape[:2]
        x, y, bw, bh = bbox
        
        # Convert bbox from (x, y, w, h) to (x1, y1, x2, y2)
        x1, y1 = x, y
        x2, y2 = x + bw, y + bh
        
        # Expand bbox by fallback percentage to ensure full body captured
        expand_pct = self.seg_fallback_expand_pct
        expand_w = int(bw * expand_pct)
        expand_h = int(bh * expand_pct)
        
        x1_exp = max(0, x1 - expand_w)
        y1_exp = max(0, y1 - expand_h)
        x2_exp = min(w, x2 + expand_w)
        y2_exp = min(h, y2 + expand_h)
        
        roi_bbox = (x1_exp, y1_exp, x2_exp, y2_exp)
        
        try:
            # Run segmentation
            mask = self._segmenter.segment_person(
                frame,
                bbox=roi_bbox,
                threshold=self.segmentation_threshold,
            )
            
            # Ensure mask shape matches frame
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # ── EDGE RECOVERY: Morphological dilation ──────────────────
            # Expands mask by 1-3px to recover arms, fingers, hair edges
            if self.seg_edge_dilation > 0:
                kernel_size = self.seg_edge_dilation * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
                mask = mask_dilated.astype(np.float32) / 255.0
            
            # ── ANTI-HALO REFINEMENT: Morphological closing + feather ──
            # Close small holes inside silhouette
            mask = self._segmenter.refine_mask(
                mask,
                kernel_size=5,
                feather=self.seg_edge_feather,
            )
            
            # ── FULL BODY COVERAGE GUARANTEE ──────────────────────────
            # Check if segmented area is reasonable relative to bbox
            mask_area = np.sum(mask > 0.5)
            bbox_area = bw * bh
            
            if mask_area < bbox_area * 0.30:
                # Segmentation likely failed (too small)
                print(f"[TRACKER] Segmentation coverage low ({mask_area}/{bbox_area:.0f}), using fallback")
                return None
            
            # Store for potential frame skip interpolation
            self._last_seg_mask = mask
            
            return mask
            
        except Exception as e:
            print(f"[TRACKER] Segmentation error: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────
    #  Task 1-7: Depth-aware cinematic blur pipeline
    # ─────────────────────────────────────────────────────────────────

    def _compute_depth_blur_map(
        self,
        frame:        np.ndarray,
        subject_mask: Optional[np.ndarray],
        h: int, w: int,
    ) -> np.ndarray:
        """
        Returns H×W float32 blur-weight map, range [0.0 … 1.0].
          0.0 = razor sharp (on the focal plane)
          1.0 = maximum blur (far from focal plane)
        """
        with self._lock:
            depth_est        = self._depth_estimator
            sel_center       = self._center
            prev_ref         = self._depth_ref
            target_ref       = self._target_depth_ref
            rack_start       = self._rack_focus_start
            rack_active      = self._rack_focus_active
            rack_dur         = 0.50  # 500ms transition
            k_blur           = self.depth_blur_k
            sp_thresh        = self.spatial_threshold_px
            tmp_alpha        = self.temporal_alpha
            prev_blur_map    = self._last_blur_map

        # ── Task 1: Depth Map Integration & Normalization ──────────────
        raw_depth = depth_est.infer(frame)                  # H×W float32 [0,1]
        # MiDaS outputs 1.0 for near, 0.0 for far. Invert to 0 (near) to 1 (far).
        depth_map = 1.0 - raw_depth
        
        # Contrast stretching (Normalization) to maximize depth range
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            depth_map = (depth_map - d_min) / (d_max - d_min)

        # ── Task 2: Focal Plane Definition ────────────────────────────
        raw_depth_mean: float = 0.5
        if subject_mask is not None:
            pixels = depth_map[subject_mask > 0.5]
            if len(pixels) >= 10:
                raw_depth_mean = float(np.mean(pixels))

        # ── Task 7: Transition Behavior (Rack Focus) ──────────────────
        if rack_active:
            if raw_depth_mean != target_ref:
                with self._lock:
                    self._target_depth_ref = raw_depth_mean
                target_ref = raw_depth_mean

            elapsed   = time.time() - rack_start
            t         = min(1.0, elapsed / max(rack_dur, 0.01))
            
            # Use cubic-bezier(0.22,1,0.36,1) for cinematic transition
            t_ease    = EASE_CINEMATIC.solve(t)
            new_ref   = prev_ref + t_ease * (target_ref - prev_ref)

            if t >= 1.0:
                with self._lock:
                    self._rack_focus_active = False
            with self._lock:
                self._depth_ref = new_ref
            focal_depth = new_ref
        else:
            focal_depth = self._depth_ref

        # ── Task 3: Aggressive Focal Plane Isolation & Amplification ───
        distance_from_focal = np.abs(depth_map - focal_depth).astype(np.float32)
        
        # Parameters for f/1.4 equivalent simulation
        epsilon = 0.03      # Sharpness band (Zone 1)
        k_logistic = 12.0   # Sharpness of transition
        threshold = 0.15    # Focal separation
        gamma = 2.5         # Non-linear depth scaling
        
        # Logistic S-Curve for depth amplification
        s_curve = 1.0 / (1.0 + np.exp(-k_logistic * (distance_from_focal - threshold)))
        
        # Normalize S-curve so it starts at 0 for distance <= epsilon
        s_min = 1.0 / (1.0 + np.exp(-k_logistic * (epsilon - threshold)))
        s_max = 1.0 / (1.0 + np.exp(-k_logistic * (1.0 - threshold)))
        
        normalized_s = np.clip((s_curve - s_min) / (s_max - s_min), 0.0, 1.0)
        
        # Non-linear power scaling (Gamma) to exaggerate depth differences
        blur_strength = np.power(normalized_s, gamma)
        
        # Force 0 blur inside the narrow sharpness band
        blur_strength[distance_from_focal < epsilon] = 0.0

        # ── Task 4: Spatial Falloff Enhancement ────────────────────────
        if sel_center is not None:
            scx, scy = sel_center
            Y, X = np.ogrid[:h, :w]
            dist_map = np.hypot(X - scx, Y - scy).astype(np.float32)
            
            # radial_falloff decreases blur for objects spatially near the selected subject
            # 0 at center, 1 at spatial_threshold_px
            radial_falloff = np.clip(dist_map / sp_thresh, 0.0, 1.0)
            # Apply smoothstep to radial falloff for natural transition
            radial_falloff = radial_falloff * radial_falloff * (3.0 - 2.0 * radial_falloff)
            
            # Combine depth blur and spatial falloff
            # Objects close in depth AND close in space get least blur
            # We map radial_falloff from [0, 1] to [0.2, 1.0] so background objects right behind subject still get some blur
            radial_falloff = 0.2 + 0.8 * radial_falloff
            
            final_blur = blur_strength * radial_falloff
        else:
            final_blur = blur_strength

        # ── Task 8: Temporal smoothing ─────────────────────────────────
        if prev_blur_map is not None and prev_blur_map.shape == final_blur.shape:
            final_blur = tmp_alpha * prev_blur_map + (1.0 - tmp_alpha) * final_blur

        with self._lock:
            self._last_blur_map = final_blur

        return final_blur.astype(np.float32)

    def _composite_blur(
        self,
        frame:  np.ndarray,
        center: Tuple[int, int],
        bbox:   Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        DSLR-style cinematic blur — full pipeline:
        """
        h, w   = frame.shape[:2]
        frame_f = frame.astype(np.float32)

        # ── Task 6: Multi-Layer Blur Strategy ─────────────────────────
        # Split scene into zones: Focal/Near/Mid/Far
        # We use 3 blur layers + original frame = 4 layers total
        # Increased max blur radius for f/1.4 simulation (30-45px)
        k_heavy  = max(45, self.blur_ksize | 1)
        k_medium = max(21, (self.blur_ksize // 2) | 1)
        k_light  = max(7, (self.blur_ksize // 4) | 1)

        blur_light  = cv2.GaussianBlur(frame, (k_light,  k_light),  0).astype(np.float32)
        blur_medium = cv2.GaussianBlur(frame, (k_medium, k_medium), 0).astype(np.float32)
        blur_heavy  = cv2.GaussianBlur(frame, (k_heavy,  k_heavy),  0)
        # Extra pass for extreme background softness
        blur_heavy  = cv2.GaussianBlur(blur_heavy, (45, 45),        0).astype(np.float32)

        # ── Resolve subject mask (silhouette lock) ─────────────────────
        mask = None

        if self.use_segmentation and bbox is not None:
            run_now = (self.seg_frame_skip == 0 or
                       self._seg_frame_counter % (self.seg_frame_skip + 1) == 0)
            self._seg_frame_counter += 1
            if run_now:
                mask = self._get_instance_mask(h, w)
                if mask is None and self._segmenter is not None:
                    mask = self._build_segmentation_mask(frame, bbox)
            else:
                mask = self._last_instance_mask or self._last_seg_mask

        if mask is None:
            if bbox is not None:
                ex1, ey1, ex2, ey2 = _expand_bbox_to_body(
                    bbox, h, w,
                    expand_w=self.body_expand_w,
                    expand_h=self.body_expand_h,
                    max_area_frac=self.body_max_frac,
                )
                if self.use_segmentation:
                    fac  = 1.0 + self.seg_fallback_expand_pct
                    ecx  = (ex1 + ex2) // 2
                    ecy  = (ey1 + ey2) // 2
                    ew   = int((ex2 - ex1) * fac)
                    eh_  = int((ey2 - ey1) * fac)
                    ex1  = max(0, ecx - ew // 2);   ey1 = max(0, ecy - eh_ // 2)
                    ex2  = min(w, ecx + ew // 2);   ey2 = min(h, ecy + eh_ // 2)
                cache_key = (ex1, ey1, ex2, ey2, self.feather, h, w)
                with self._lock:
                    if self._mask_cache is None or self._mask_key != cache_key:
                        self._mask_cache = _build_body_mask(
                            h, w, ex1, ey1, ex2, ey2, self.feather)
                        self._mask_key = cache_key
                    mask = self._mask_cache
            else:
                cx_ = int(np.clip(center[0], 0, w - 1))
                cy_ = int(np.clip(center[1], 0, h - 1))
                ck  = (cx_, cy_, self.focus_radius, self.feather, h, w)
                with self._lock:
                    if self._mask_cache is None or self._mask_key != ck:
                        self._mask_cache = _build_soft_mask(
                            h, w, cx_, cy_, self.focus_radius, self.feather)
                        self._mask_key = ck
                    mask = self._mask_cache

        # ── Task 5: Edge Feathering ────────────────────────────────────
        # Apply 25px Gaussian feathering around mask boundaries to avoid halos
        if mask is not None:
            mask = cv2.GaussianBlur(mask, (25, 25), 0)

        # ── Tasks 1-4: Per-pixel blur weight from depth ─────────────
        with self._lock:
            has_depth = self._depth_estimator is not None
        if has_depth:
            blur_w = self._compute_depth_blur_map(frame, mask, h, w)
        else:
            # Fallback: binary (subject=sharp, background=full blur)
            blur_w = (1.0 - mask).astype(np.float32)

        # Subject silhouette ALWAYS stays sharp (mask overrides depth)
        blur_w = blur_w * (1.0 - mask)          # zero blur where mask==1

        # ── Task 6: Multi-Layer Blur Strategy (4 layers) ──────────────
        # blur_w in [0.0, 0.33) → lerp original → light
        # blur_w in [0.33, 0.66) → lerp light    → medium
        # blur_w in [0.66, 1.0] → lerp medium   → heavy
        t  = blur_w[:, :, np.newaxis]            # (H,W,1)

        # Segment 1: original → light  (t: 0 → 0.33)
        a1    = np.clip(t / 0.333, 0.0, 1.0)
        out1  = frame_f * (1.0 - a1) + blur_light * a1

        # Segment 2: light → medium   (t: 0.33 → 0.66)
        a2    = np.clip((t - 0.333) / 0.333, 0.0, 1.0)
        out2  = blur_light * (1.0 - a2) + blur_medium * a2

        # Segment 3: medium → heavy   (t: 0.66 → 1.0)
        a3    = np.clip((t - 0.666) / 0.334, 0.0, 1.0)
        out3  = blur_medium * (1.0 - a3) + blur_heavy * a3

        # Select segment per pixel
        c1 = (blur_w < 0.333).astype(np.float32)[:, :, np.newaxis]
        c2 = ((blur_w >= 0.333) & (blur_w < 0.666)).astype(np.float32)[:, :, np.newaxis]
        c3 = (blur_w >= 0.666).astype(np.float32)[:, :, np.newaxis]

        composite = out1 * c1 + out2 * c2 + out3 * c3

        # ── Task 9: Cinematic Luminance Emphasis ──────────────────────
        # Slight luminance emphasis on focal subject
        if mask is not None:
            mask_3c = mask[:, :, np.newaxis]
            composite = composite * (1.0 + 0.08 * mask_3c)
            
        return np.clip(composite, 0, 255).astype(np.uint8)

    def _draw_overlay(
        self,
        frame:  np.ndarray,
        bbox:   Optional[Tuple[int,int,int,int]],
        center: Optional[Tuple[int,int]],
        state:  str,
    ) -> np.ndarray:
        """
        Cinematic overlay — INVISIBLE INTELLIGENCE MODE.

        Rules:
          • NO bounding boxes.
          • NO text labels.
          • NO confidence scores.
          • NO green rectangles.
          • Only a single, thin, semi-transparent focus ring at the
            face-click point — visible for ~800ms then fades out.
          • All cinematic logic runs invisibly in the background.
        """
        # Return frame unchanged — all debug elements stripped.
        # The depth-of-field composite IS the visual output.
        return frame

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
