"""
YOLOv8 Instance Segmentation Module
=====================================
Replaces detection-only YOLO with yolov8n-seg.pt.

Per-frame output per detected person:
    {
        'class':      'person',
        'confidence': float,
        'bbox':       [x1, y1, x2, y2],       # full-frame pixel coords
        'track_id':   int,                      # stable YOLO botsort track id
        'mask':       np.ndarray (H×W uint8),   # 0 or 255, full-frame size
    }

Tracking uses YOLO's built-in BoT-SORT so track_ids stay consistent
across frames, exactly what the autofocus engine needs to lock on.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional

from core.base_module import BaseAIModule, InferenceResult


class YOLOv8SegModule(BaseAIModule):
    """
    Instance segmentation using YOLOv8n-seg (nano segmentation model).

    Falls back gracefully:
        yolov8n-seg.pt not found → try yolov8n.pt (no masks)
        ultralytics not installed → demo mode

    Output each frame: List of dicts with class/confidence/bbox/track_id/mask.
    """

    SEG_MODEL  = 'yolov8n-seg.pt'   # primary: nano segmentation
    DET_MODEL  = 'yolov8n.pt'        # fallback: detection only

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        cfg = config or {}

        self.confidence_threshold: float = cfg.get('confidence_threshold', 0.40)
        self.iou_threshold:        float = cfg.get('iou_threshold', 0.45)
        self.detect_only_person:   bool  = cfg.get('detect_only_person', True)
        self.device:               str   = cfg.get('device', 'cpu')

        self.model           = None
        self._has_seg: bool  = False   # True when seg model loaded
        self._demo:    bool  = False

        # ── Exponential smoothing for bbox centers (task 5) ──────────
        # key: track_id → smoothed (cx, cy)
        self._smooth_centers: Dict[int, tuple] = {}
        self._smooth_alpha: float = cfg.get('smooth_alpha', 0.55)

        print(f"✓ YOLOv8SegModule created  "
              f"conf={self.confidence_threshold}  device={self.device}  "
              f"person_only={self.detect_only_person}")

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

        # Try segmentation model first
        for model_name, has_seg in [(self.SEG_MODEL, True),
                                    (self.DET_MODEL,  False)]:
            try:
                print(f"⏳ Loading {model_name} …")
                self.model = YOLO(model_name)
                self._has_seg = has_seg
                print(f"✓ {model_name} loaded  (seg={'YES' if has_seg else 'NO (fallback)'})")
                self.is_initialized = True
                return True
            except Exception as e:
                print(f"⚠ {model_name} failed: {e}")

        print("⚠ All models failed → demo mode")
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
        Run segmentation (or detection) inference.

        Returns:
            List of dicts.  Each dict has:
                class, confidence, bbox, track_id, mask (or None).
        """
        if self._demo:
            return self._demo_detections(frame)
        if self._has_seg:
            return self._seg_inference(frame)
        return self._det_inference(frame)

    # ------------------------------------------------------------------ #
    #  Private: model inference                                             #
    # ------------------------------------------------------------------ #

    def _seg_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """YOLOv8-seg with BoT-SORT tracking → per-instance masks."""
        h, w = frame.shape[:2]

        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            persist=True,        # keep track IDs across frames
            tracker='botsort.yaml',
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
        """Detection-only fallback (no masks)."""
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            persist=True,
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

    def cleanup(self):
        self.model = None
        self._smooth_centers.clear()
        print("✓ YOLOv8SegModule cleanup")
