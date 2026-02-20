"""
Blur Compositor - CINEMATIC AUTOFOCUS
Multi-layer edge-aware blur blending
Produces cinematic depth-of-field effect from blur_map + original frame
"""
import cv2
import numpy as np
from typing import Optional
import time
from collections import deque


# --------------------------------------------------------------------------- #
#  Blur layer definitions                                                       #
# --------------------------------------------------------------------------- #

# (name, threshold_low, threshold_high, kernel_size)
_LAYERS = [
    ('sharp',  0.00, 0.15,  0),
    ('light',  0.15, 0.35,  5),
    ('medium', 0.35, 0.60, 15),
    ('heavy',  0.60, 0.80, 35),
    ('extreme',0.80, 1.00, 61),
]


class BlurCompositor:
    """
    High-quality, edge-aware cinematic blur compositor.

    Pipeline per frame:
      1. Pre-bilateral filter (edge preservation)
      2. Generate 4 blur layers (5→61 kernels)
      3. Alpha-composite by blur_map
      4. Feather focus boundary
      5. Return composited frame
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.bilateral_d       = cfg.get('bilateral_d', 7)
        self.bilateral_sigma_c = cfg.get('bilateral_sigma_c', 50.0)
        self.bilateral_sigma_s = cfg.get('bilateral_sigma_s', 50.0)
        self.feather_radius    = cfg.get('feather_radius', 30)
        self.enable_bilateral  = cfg.get('enable_bilateral', True)

        # Pre-build blur kernels
        self._kernels = self._build_kernels()

        # Performance
        self._comp_times: deque = deque(maxlen=30)

        print(f"✓ BlurCompositor  bilateral={self.enable_bilateral}  "
              f"feather={self.feather_radius}px")

    # ------------------------------------------------------------------ #
    #  Public                                                               #
    # ------------------------------------------------------------------ #

    def composite(
        self,
        frame: np.ndarray,
        blur_map: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Apply cinematic depth-of-field blur to *frame* guided by *blur_map*.

        Args:
            frame:    BGR uint8 H×W×3
            blur_map: float32 H×W in [0,1]. None = passthrough.

        Returns:
            BGR uint8 composited frame.
        """
        if blur_map is None or np.max(blur_map) < 0.01:
            return frame

        t0 = time.time()
        result = self._apply(frame, blur_map)
        self._comp_times.append((time.time() - t0) * 1000)
        return result

    def get_stats(self) -> dict:
        times = list(self._comp_times)
        return {
            'avg_ms': round(float(np.mean(times)), 1) if times else 0,
            'max_ms': round(float(np.max(times)), 1) if times else 0,
            'bilateral': self.enable_bilateral,
        }

    def cleanup(self):
        print("✓ BlurCompositor cleanup complete")

    # ------------------------------------------------------------------ #
    #  Internal                                                             #
    # ------------------------------------------------------------------ #

    def _build_kernels(self) -> list:
        """Pre-allocate blur kernels for each layer (except sharp=0)."""
        kernels = []
        for name, lo, hi, k in _LAYERS:
            if k > 0:
                # Ensure odd kernel size
                k = k | 1
                kernels.append((name, lo, hi, k))
            else:
                kernels.append((name, lo, hi, 0))
        return kernels

    def _apply(self, frame: np.ndarray, blur_map: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        # ── 1. Edge-preserving pre-filter ─────────────────────────────
        if self.enable_bilateral:
            base = cv2.bilateralFilter(
                frame,
                self.bilateral_d,
                self.bilateral_sigma_c,
                self.bilateral_sigma_s
            )
        else:
            base = frame

        # ── 2. Feather mask edges ──────────────────────────────────────
        blur_map_f = self._feather_map(blur_map, self.feather_radius)

        # ── 3. Build blurred layers ────────────────────────────────────
        blurred: dict[str, np.ndarray] = {}
        for name, lo, hi, k in self._kernels:
            if k == 0:
                blurred[name] = base
            else:
                blurred[name] = cv2.GaussianBlur(base, (k, k), 0)

        # ── 4. Layer composite from sharp → extreme ────────────────────
        output = base.astype(np.float32)

        for name, lo, hi, k in self._kernels:
            # Alpha = how much of this layer applies at each pixel
            alpha = np.clip((blur_map_f - lo) / max(hi - lo, 1e-5), 0.0, 1.0)
            alpha3 = alpha[:, :, np.newaxis]  # broadcast over BGR channels
            layer  = blurred[name].astype(np.float32)
            output = output * (1 - alpha3) + layer * alpha3

        # ── 5. Final clip ──────────────────────────────────────────────
        return np.clip(output, 0, 255).astype(np.uint8)

    def _feather_map(self, blur_map: np.ndarray, radius: int) -> np.ndarray:
        """Smooth the hard edges in blur_map with a Gaussian."""
        if radius <= 0:
            return blur_map
        k = (radius * 2 + 1) | 1  # ensure odd
        return cv2.GaussianBlur(blur_map, (k, k), radius / 2.0)
