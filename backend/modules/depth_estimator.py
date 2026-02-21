"""
DepthEstimator
==============
Wraps MiDaS-small (via torch.hub) for per-frame monocular depth estimation.

Output contract:
  infer(frame) → H×W float32 depth map, values in [0.0, 1.0].
  MiDaS produces *inverse* depth (higher value = CLOSER to camera).
  The map is normalised per-frame so 1.0 = nearest object, 0.0 = farthest.

Fallback chain (if PyTorch / internet unavailable):
  1. Luminance-proxy depth using edge density + vertical prior.
     (Highly textured regions tend to be closer; floor/ground is lower
      in typical fixed-camera views so we blend in a vertical gradient.)
  2. Pure vertical gradient (trivially cheap, always available).

Thread safety: infer() is NOT thread-safe.  Call from one thread only
  (or wrap externally).  The heavy model inference is ~20 ms on CPU.
"""
from __future__ import annotations

import cv2
import numpy as np
import time
from typing import Optional


class DepthEstimator:
    """Monocular depth estimation with graceful multi-level fallback."""

    # ── public tuning knobs ─────────────────────────────────────────────────
    DEPTH_FRAME_SKIP: int   = 3          # run model every N frames (0 = every frame)
    SMOOTHING_ALPHA:  float = 0.5        # temporal blend: α*prev + (1-α)*new

    def __init__(self, device: str = "cpu"):
        self.device          = device
        self._model          = None
        self._transform      = None
        self._has_torch      = False
        self._mode: str      = "uninit"  # "midas" | "proxy" | "gradient"

        self._last_depth: Optional[np.ndarray] = None   # temporally smoothed
        self._counter:  int = 0

        print("[Depth] DepthEstimator created (not yet initialised)")

    # ── public API ──────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """Try to load MiDaS-small; fall back to proxy mode."""
        if self._try_load_midas():
            print("✓ DepthEstimator: MiDaS-small loaded  (full depth)")
            return True
        # proxy fallback — no heavy deps
        self._mode = "proxy"
        print("⚠ DepthEstimator: MiDaS unavailable — using luminance proxy depth")
        return True   # always succeeds

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns H×W float32 in [0, 1].
        1.0 = nearest to camera, 0.0 = farthest.
        Temporally smoothed across frames.
        """
        h, w = frame.shape[:2]
        self._counter += 1

        # Frame-skip: reuse last depth if available
        if (self.DEPTH_FRAME_SKIP > 0 and
                self._counter % (self.DEPTH_FRAME_SKIP + 1) != 0 and
                self._last_depth is not None):
            return self._last_depth

        # Compute new depth
        if self._mode == "midas":
            raw = self._infer_midas(frame)
        elif self._mode == "proxy":
            raw = self._infer_proxy(frame)
        else:
            raw = self._infer_gradient(h, w)

        # Temporal smoothing
        if self._last_depth is not None and self._last_depth.shape == raw.shape:
            raw = (self.SMOOTHING_ALPHA * self._last_depth +
                   (1.0 - self.SMOOTHING_ALPHA) * raw)

        self._last_depth = raw
        return raw

    def reset(self):
        """Clear temporal state (call on subject change / scene cut)."""
        self._last_depth = None

    # ── private — model loading ─────────────────────────────────────────────

    def _try_load_midas(self) -> bool:
        try:
            import torch
            self._has_torch = True
        except ImportError:
            return False

        try:
            import torch
            model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small",
                trust_repo=True, verbose=False,
            )
            transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms",
                trust_repo=True, verbose=False,
            )
            dev = torch.device(self.device)
            model.to(dev).eval()
            self._model     = model
            self._transform = transforms.small_transform
            self._device    = dev
            self._mode      = "midas"
            return True
        except Exception as e:
            print(f"[Depth] MiDaS load failed: {e}")
            return False

    # ── private — inference ─────────────────────────────────────────────────

    def _infer_midas(self, frame: np.ndarray) -> np.ndarray:
        """Run MiDaS-small; returns normalised H×W float32."""
        import torch
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch  = self._transform(rgb).to(self._device)

        with torch.no_grad():
            pred = self._model(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = pred.cpu().numpy().astype(np.float32)

        # Normalise per-frame: [0,1], higher = closer
        mn, mx = depth.min(), depth.max()
        depth  = (depth - mn) / max(mx - mn, 1e-6)
        return depth

    def _infer_proxy(self, frame: np.ndarray) -> np.ndarray:
        """
        Luminance-proxy depth:
          - Laplacian edge density → high texture ≈ close.
          - Blended with vertical position prior (top=far, bottom=close)
            at 40% weight so it doesn't dominate indoors.
        Produces visually plausible depth in typical surveillance/studio shots.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge density (Laplacian magnitude, smoothed)
        lap  = cv2.Laplacian(gray, cv2.CV_32F)
        edge = np.abs(lap)
        edge = cv2.GaussianBlur(edge, (31, 31), 0)
        mn, mx = edge.min(), edge.max()
        edge_norm = (edge - mn) / max(mx - mn, 1e-6)

        # Vertical prior: y=0 → 0.0 (far), y=h → 1.0 (close)
        grad = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
        grad = np.broadcast_to(grad, (h, w)).copy()

        # Blend: 60% edge, 40% vertical
        depth = 0.60 * edge_norm + 0.40 * grad
        depth = np.clip(depth, 0.0, 1.0).astype(np.float32)

        # Smooth out noise
        depth = cv2.GaussianBlur(depth, (15, 15), 0)
        return depth

    @staticmethod
    def _infer_gradient(h: int, w: int) -> np.ndarray:
        """Cheapest fallback: pure vertical gradient."""
        grad = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
        return np.broadcast_to(grad, (h, w)).copy()
