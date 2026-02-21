"""
YOLOv11 Instance Segmentation Module — Production Grade
=======================================================
Uses YOLOv11x-seg (highest accuracy variant) with CUDA acceleration.

Per-frame output per detected object:
    {
        'class':      str,
        'confidence': float,
        'bbox':       [x1, y1, x2, y2],       # full-frame pixel coords
        'track_id':   int,                      # stable BoT-SORT track id
        'mask':       np.ndarray (H×W uint8),   # 0 or 255, full-frame size
        'center':     (cx, cy),                 # exponentially smoothed center
    }

Key features:
  - YOLOv11x-seg (most accurate) → yolo11l-seg → yolo11x (det) → demo
  - GPU/CUDA acceleration with automatic fallback to CPU
  - Detection frequency control: run every N frames, interpolate between
  - IoU-based track matching + exponential smoothing (Kalman-style)
  - Hidden mode: no visual overlays, no bounding boxes rendered
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional

from core.base_module import BaseAIModule, InferenceResult


class YOLOv8SegModule(BaseAIModule):
    """
    Production-grade object detection + segmentation using YOLOv11x-seg.

    Model priority chain (highest accuracy first):
        1. yolo11x-seg.pt   — YOLOv11 Extra-Large Segmentation (primary)
        2. yolo11l-seg.pt   — YOLOv11 Large Segmentation (fallback)
        3. yolo11x.pt       — YOLOv11 Extra-Large Detection-only (fallback)
        4. demo mode        — synthetic detections (no ultralytics)
    """

    # Model priority chain — highest accuracy first
    MODEL_CHAIN = [
        ('yolo11x-seg.pt', True),   # YOLOv11-X segmentation (best)
        ('yolo11l-seg.pt', True),   # YOLOv11-L segmentation
        ('yolo11x.pt',     False),  # YOLOv11-X detection only
        ('yolo11n-seg.pt', True),   # YOLOv11-N seg (last resort before demo)
    ]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        cfg = config or {}

        self.confidence_threshold: float = cfg.get('confidence_threshold', 0.35)
        self.iou_threshold:        float = cfg.get('iou_threshold', 0.50)
        self.detect_only_person:   bool  = cfg.get('detect_only_person', False)
        self.device:               str   = cfg.get('device', 'auto')  # auto = try CUDA first

        self.model           = None
        self._has_seg: bool  = False   # True when seg model loaded
        self._demo:    bool  = False
        self._resolved_device: str = 'cpu'  # set after CUDA probing

        # ── Exponential smoothing for bbox centers ──────────────────
        self._smooth_centers:  Dict[int, tuple] = {}
        self._smooth_alpha:    float = cfg.get('smooth_alpha', 0.40)  # lower = smoother

        # ── Velocity for sub-frame interpolation (Kalman-lite) ──────
        # key: track_id → (vx, vy) pixels/frame
        self._velocities:      Dict[int, tuple] = {}

        # ── Detection frequency / frame-skip settings ───────────────
        # Run inference every N frames; interpolate in between.
        self.detection_interval: int = cfg.get('detection_interval', 2)  # 2 = every 2nd frame
        self._frame_idx:         int = 0
        self._last_detections:   List[Dict[str, Any]] = []

        print(f"✓ YOLOv11SegModule created  "
              f"conf={self.confidence_threshold}  device={self.device}  "
              f"person_only={self.detect_only_person}  "
              f"det_interval={self.detection_interval}")

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def initialize(self) -> bool:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("⚠ ultralytics not installed → demo mode")
            print("  Run:  pip install ultralytics")
            self._demo = True
            self.is_initialized = True
            return True

        # ── Resolve CUDA device ──────────────────────────────────────
        if self.device == 'auto':
            try:
                import torch
                self._resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self._resolved_device = 'cpu'
        else:
            self._resolved_device = self.device
        print(f"[YOLOv11] Compute device: {self._resolved_device.upper()}")

        # ── Try model chain (highest accuracy first) ─────────────────
        for model_name, has_seg in self.MODEL_CHAIN:
            try:
                print(f"⏳ Loading {model_name} on {self._resolved_device} …")
                m = YOLO(model_name)
                # Warm-up pass to load weights onto GPU
                import numpy as _np
                dummy = _np.zeros((320, 320, 3), dtype=_np.uint8)
                m(dummy, verbose=False, device=self._resolved_device)
                self.model    = m
                self._has_seg = has_seg
                print(f"✓ {model_name} ready  "
                      f"seg={'YES' if has_seg else 'NO'}  "
                      f"device={self._resolved_device.upper()}")
                self.is_initialized = True
                return True
            except Exception as e:
                print(f"⚠ {model_name} failed: {e}")

        print("⚠ All YOLOv11 models failed → demo mode")
        self._demo = True
        self.is_initialized = True
        return True

    # ------------------------------------------------------------------ #
    #  BaseAIModule.process_frame  (regular detection mode for UI)         #
    # ------------------------------------------------------------------ #

    def process_frame(self, frame: np.ndarray) -> InferenceResult:
        t0 = time.time()
        self.frame_count += 1

        detections = self.detect(frame)

        ms = (time.time() - t0) * 1000
        self.total_inference_time += ms

        alert = 'critical' if len(detections) >= 5 else \
                'warning'  if len(detections) >= 3 else 'normal'

        metrics = {
            'object_count':  len(detections),
            'avg_inference': round(self.get_average_inference_time(), 2),
            'seg_active':    self._has_seg,
        }
        return self._create_result(detections, metrics, alert, ms)

    # ------------------------------------------------------------------ #
    #  detect() — called each frame by the autofocus pipeline              #
    # ------------------------------------------------------------------ #

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run segmentation (or detection) inference with frame-skip interpolation.

        Detection runs every `detection_interval` frames. On skipped frames
        the last detections are returned with centers interpolated using
        the per-track velocity estimate — no flicker, no jump.

        Returns:
            List of dicts with: class, confidence, bbox, track_id, mask, center.
        """
        self._frame_idx += 1

        if self._demo:
            return self._demo_detections(frame)

        run_full = (self._frame_idx % max(1, self.detection_interval) == 0)

        if run_full:
            if self._has_seg:
                detections = self._seg_inference(frame)
            else:
                detections = self._det_inference(frame)
            self._update_velocities(detections)
            self._last_detections = detections
        else:
            # Interpolate positions using velocity (Kalman-lite)
            detections = self._interpolate_detections(self._last_detections)

        return detections

    # ------------------------------------------------------------------ #
    #  Private: model inference                                             #
    # ------------------------------------------------------------------ #

    def _seg_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """YOLOv11-seg with BoT-SORT tracking → per-instance masks."""
        h, w = frame.shape[:2]

        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            persist=True,        # keep track IDs across frames
            tracker='botsort.yaml',
            device=self._resolved_device,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            boxes  = r.boxes
            masks  = r.masks   # None if no objects / no seg model

            for i, box in enumerate(boxes):
                cls_id     = int(box.cls[0])
                class_name = r.names[cls_id]

                if self.detect_only_person and class_name != 'person':
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]

                # Track ID: YOLO assigns it when tracking is active
                track_id = int(box.id[0]) if box.id is not None else self._fallback_id(x1, y1, x2, y2)

                # Smoothed center
                cx, cy = self._smooth_center(track_id, (x1 + x2) // 2, (y1 + y2) // 2)

                # Instance mask (H×W uint8, full-frame)
                instance_mask = None
                if masks is not None and i < len(masks.data):
                    # YOLO returns masks at model resolution; resize to frame size
                    raw_mask = masks.data[i].cpu().numpy()  # float32 [0-1], model HW
                    raw_resized = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    instance_mask = (raw_resized > 0.5).astype(np.uint8) * 255

                detections.append({
                    'class':      class_name,
                    'confidence': conf,
                    'bbox':       [x1, y1, x2, y2],
                    'track_id':   track_id,
                    'mask':       instance_mask,
                    'center':     (cx, cy),
                })

        return detections

    def _det_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """YOLOv11 detection-only fallback (no masks)."""
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            persist=True,
            device=self._resolved_device,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id     = int(box.cls[0])
                class_name = r.names[cls_id]

                if self.detect_only_person and class_name != 'person':
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                track_id = int(box.id[0]) if box.id is not None else self._fallback_id(x1, y1, x2, y2)
                cx, cy   = self._smooth_center(track_id, (x1 + x2) // 2, (y1 + y2) // 2)

                detections.append({
                    'class':      class_name,
                    'confidence': conf,
                    'bbox':       [x1, y1, x2, y2],
                    'track_id':   track_id,
                    'mask':       None,
                    'center':     (cx, cy),
                })

        return detections

    def _demo_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Synthetic detections when ultralytics is not available."""
        if frame is None:
            return []
        h, w = frame.shape[:2]
        import random
        detections = []
        if np.mean(frame) > 40:
            for i in range(random.randint(0, 1)):
                x1 = random.randint(w // 4, w // 2)
                y1 = random.randint(h // 8, h // 4)
                x2 = min(x1 + w // 4, w - 1)
                y2 = min(y1 + h // 2, h - 1)
                conf = random.uniform(0.55, 0.90)
                track_id = i + 1
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Synthetic ellipse mask for demo
                mask = np.zeros((h, w), dtype=np.uint8)
                rw, rh_m = (x2 - x1) // 2, (y2 - y1) // 2
                cv2.ellipse(mask, (cx, cy), (rw, rh_m), 0, 0, 360, 255, -1)

                detections.append({
                    'class':      'person',
                    'confidence': conf,
                    'bbox':       [x1, y1, x2, y2],
                    'track_id':   track_id,
                    'mask':       mask,
                    'center':     (cx, cy),
                })
        return detections

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _smooth_center(self, track_id: int, raw_cx: int, raw_cy: int) -> tuple:
        """
        Exponential moving average of bbox center (Phase 5).

        Prevents jitter from frame-to-frame detection fluctuation.
        """
        alpha = self._smooth_alpha
        if track_id in self._smooth_centers:
            px, py = self._smooth_centers[track_id]
            scx = int(alpha * raw_cx + (1.0 - alpha) * px)
            scy = int(alpha * raw_cy + (1.0 - alpha) * py)
        else:
            scx, scy = raw_cx, raw_cy

        self._smooth_centers[track_id] = (scx, scy)
        return scx, scy

    # Fallback track id when YOLO tracking disabled
    _next_id = 1

    def _fallback_id(self, x1, y1, x2, y2) -> int:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        for tid, (px, py) in list(self._smooth_centers.items()):
            if abs(cx - px) < 80 and abs(cy - py) < 80:
                return tid
        tid = YOLOv8SegModule._next_id
        YOLOv8SegModule._next_id += 1
        return tid

    # ------------------------------------------------------------------ #
    #  Frame-skip interpolation helpers                                     #
    # ------------------------------------------------------------------ #

    def _update_velocities(self, detections: List[Dict[str, Any]]) -> None:
        """Compute per-track velocity (pixels/frame) from latest detection."""
        for det in detections:
            tid = det['track_id']
            cx, cy = det['center']
            if tid in self._velocities:
                px, py = self._smooth_centers.get(tid, (cx, cy))
                self._velocities[tid] = (cx - px, cy - py)
            else:
                self._velocities[tid] = (0, 0)

    def _interpolate_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Advance each tracked object's center by its velocity estimate."""
        interpolated = []
        for det in detections:
            tid = det['track_id']
            vx, vy = self._velocities.get(tid, (0, 0))
            cx, cy = det['center']
            new_cx = cx + vx
            new_cy = cy + vy
            # Shift bbox by same delta
            x1, y1, x2, y2 = det['bbox']
            new_bbox = [x1 + vx, y1 + vy, x2 + vx, y2 + vy]
            new_det = dict(det)
            new_det['center'] = (int(new_cx), int(new_cy))
            new_det['bbox']   = [int(v) for v in new_bbox]
            interpolated.append(new_det)
        return interpolated

    def cleanup(self):
        self.model = None
        self._smooth_centers.clear()
        self._velocities.clear()
        print("✓ YOLOv11SegModule cleanup")
