"""
VideoDetectionEngine — Run detection ONLY on the click frame to identify
the bounding box under the cursor.

Strategy (no heavy model required):
  1. Run the existing ObjectDetectionModule if available.
  2. Fall back to a contour/GrabCut patch approach when no model is loaded.

The result is a single (x, y, w, h) bounding box that VideoTracker will use
to initialise tracking.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple


class VideoDetectionEngine:
    """Detects the best bounding box under a user click on a single frame."""

    # Minimum blob area to be considered a meaningful object
    _MIN_AREA = 400

    def __init__(self) -> None:
        # Try to reuse the live detection pipeline if already imported
        self._live_module = None
        try:
            from modules.object_detection import ObjectDetectionModule
            m = ObjectDetectionModule({'confidence_threshold': 0.4})
            if m.initialize():
                self._live_module = m
                print("[VideoDetectionEngine] Using live YOLO module for detection")
        except Exception:
            pass

        if self._live_module is None:
            print("[VideoDetectionEngine] Falling back to GrabCut patch detector")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_at_click(
        self,
        frame: np.ndarray,
        click_x: int,
        click_y: int,
        patch_fraction: float = 0.20,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Return (x, y, w, h) of the object under the click, or None.

        Parameters
        ----------
        frame : BGR frame at native resolution
        click_x, click_y : pixel coords in frame space
        patch_fraction : fraction of frame size used as seed window
        """
        h, w = frame.shape[:2]

        # --- Try live model first ------------------------------------------
        if self._live_module is not None:
            bbox = self._detect_with_model(frame, click_x, click_y, w, h)
            if bbox is not None:
                return bbox

        # --- Fallback: GrabCut seed around click --------------------------
        return self._detect_grabcut(frame, click_x, click_y,
                                    patch_fraction, w, h)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _detect_with_model(
        self, frame, cx, cy, fw, fh
    ) -> Optional[Tuple[int, int, int, int]]:
        try:
            result = self._live_module.process(frame)
            best_iou = 0.0
            best_box = None
            for det in (result.detections if result else []):
                bx, by, bw, bh = det["bbox"]
                # Check if click is inside this box
                if bx <= cx <= bx + bw and by <= cy <= by + bh:
                    area = bw * bh
                    if area > best_iou:
                        best_iou = area
                        best_box = (bx, by, bw, bh)
            return best_box
        except Exception as e:
            print(f"[VideoDetectionEngine] Model detect error: {e}")
            return None

    def _detect_grabcut(
        self, frame, cx, cy, patch_frac, fw, fh
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        GrabCut seeded at click point → tight foreground bounding box.
        """
        half_pw = max(int(fw * patch_frac * 0.5), 40)
        half_ph = max(int(fh * patch_frac * 0.5), 40)

        # GrabCut seed rect around click
        gx1 = max(0,  cx - half_pw)
        gy1 = max(0,  cy - half_ph)
        gx2 = min(fw, cx + half_pw)
        gy2 = min(fh, cy + half_ph)

        if (gx2 - gx1) < 10 or (gy2 - gy1) < 10:
            # Degenerate rect — just return the seed rectangle itself
            return (gx1, gy1, gx2 - gx1, gy2 - gy1)

        try:
            mask   = np.zeros((fh, fw), np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect = (int(gx1), int(gy1), int(gx2 - gx1), int(gy2 - gy1))
            cv2.grabCut(frame, mask, rect, bgdModel, fgdModel,
                        iterCount=3, mode=cv2.GC_INIT_WITH_RECT)

            # Foreground mask
            fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                               255, 0).astype(np.uint8)

            # Find bounding rect of foreground
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return rect

            # Pick largest contour that contains the click
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
                br = cv2.boundingRect(cnt)
                bx, by, bw, bh = br
                if bx <= cx <= bx + bw and by <= cy <= by + bh:
                    if bw * bh >= self._MIN_AREA:
                        return br
            return cv2.boundingRect(max(contours, key=cv2.contourArea))

        except Exception as e:
            print(f"[VideoDetectionEngine] GrabCut error: {e}")
            return (gx1, gy1, gx2 - gx1, gy2 - gy1)
