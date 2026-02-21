"""
VideoBlurRenderer — Applies cinematic depth-of-field blur to a video frame.

Algorithm
---------
For each pixel at depth d, the blur radius is proportional to
|d - subject_depth| scaled by blur_strength.  A smooth Gaussian falloff
mask is built from the depth difference so edges feather naturally.

Two layers of compositing:
  1. Body-region mask  (tracker bbox expanded to full body) — always sharp
  2. Depth-difference mask — gradual blur for every pixel outside body

Both masks are multiplied so the body region is always protected even when
the depth estimate is noisy.

Parameters
----------
blur_strength : float [0, 1]   — how aggressively background is blurred
depth_bias    : float [-0.5, 0.5] — shift the focus depth plane
feather       : int  >=0       — px softness at body-region edge
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple


# Kernel sizes (must be odd) — [background, secondary]
_BLUR_K1 = 101
_BLUR_K2 = 31


def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def _build_body_mask(
    h: int, w: int,
    x1: int, y1: int, x2: int, y2: int,
    feather: int = 30,
) -> np.ndarray:
    """Float32 mask: 1.0 inside body rect, 0.0 outside, feathered edges."""
    Y, X = np.ogrid[:h, :w]
    d_l = (X - x1).astype(np.float32)
    d_r = (x2 - X).astype(np.float32)
    d_t = (Y - y1).astype(np.float32)
    d_b = (y2 - Y).astype(np.float32)
    d   = np.minimum(np.minimum(d_l, d_r), np.minimum(d_t, d_b))
    return np.clip(d / max(float(feather), 1.0), 0.0, 1.0)


class VideoBlurRenderer:
    """
    Composites a depth-of-field blur for each processed frame.
    Stateful: caches the pre-blurred frame to avoid redundant computation.
    """

    def __init__(self, feather: int = 30) -> None:
        self._feather   = feather
        self._last_id   = None          # id(frame) of cached blur
        self._blurred1  : Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Rack focus interpolation
    # ------------------------------------------------------------------
    def interpolate_depth(
        self,
        current: float,
        target: float,
        alpha: float = 0.08,
    ) -> float:
        """Smooth the focus depth plane toward *target* (rack focus feel)."""
        return current + alpha * (target - current)

    # ------------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------------
    def render(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        subject_depth: float,
        body_bbox: Optional[Tuple[int, int, int, int]],   # (x1,y1,x2,y2)
        blur_strength: float = 0.85,
        depth_bias: float = 0.0,
    ) -> np.ndarray:
        """
        Return a composited frame with depth-of-field effect.

        Parameters
        ----------
        frame         : BGR source frame
        depth_map     : float32 H×W depth, 1.0=closest (may be None)
        subject_depth : reference depth of tracked subject [0,1]
        body_bbox     : expanded body region (x1,y1,x2,y2) in frame coords
        blur_strength : 0=no blur, 1=full heavy blur
        depth_bias    : shifts focus plane up/down
        """
        h, w = frame.shape[:2]
        fid  = id(frame)

        # Cache heavy blur computation — same frame object reused in export
        if self._last_id != fid:
            k1 = _odd(int(_BLUR_K1 * max(0.3, blur_strength)))
            k2 = _odd(int(_BLUR_K2 * max(0.3, blur_strength)))
            b1 = cv2.GaussianBlur(frame, (k1, k1), 0)
            self._blurred1 = cv2.GaussianBlur(b1,  (k2, k2), 0)
            self._last_id  = fid

        blurred = self._blurred1

        # --- Layer 1: body region mask (always sharp) --------------------
        if body_bbox is not None:
            x1, y1, x2, y2 = [max(0, v) for v in body_bbox]
            x2 = min(w, x2);  y2 = min(h, y2)
            body_mask = _build_body_mask(h, w, x1, y1, x2, y2, self._feather)
        else:
            body_mask = np.zeros((h, w), np.float32)

        # --- Layer 2: depth-difference mask --------------------------------
        if depth_map is not None and blur_strength > 0.0:
            eff_depth = np.clip(subject_depth + depth_bias, 0.0, 1.0)
            diff      = np.abs(depth_map - eff_depth)
            # Falloff: pixels close to subject depth stay sharp
            sigma     = max(0.05, 0.25 * (1.0 - blur_strength))
            depth_mask = np.exp(-0.5 * (diff / sigma) ** 2).astype(np.float32)
        else:
            depth_mask = np.zeros((h, w), np.float32)

        # --- Combine masks (take max → most-sharp wins) ------------------
        alpha = np.maximum(body_mask, depth_mask)[:, :, np.newaxis]

        # --- Composite ---------------------------------------------------
        frame_f   = frame.astype(np.float32)
        blurred_f = blurred.astype(np.float32)
        out = (alpha * frame_f + (1.0 - alpha) * blurred_f).clip(0, 255)
        return out.astype(np.uint8)

    # ------------------------------------------------------------------
    # Debug overlay
    # ------------------------------------------------------------------
    def render_depth_heatmap(self, depth_map: np.ndarray) -> np.ndarray:
        """Render a colour-mapped depth heatmap for debugging."""
        d8 = (depth_map * 255).clip(0, 255).astype(np.uint8)
        return cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
