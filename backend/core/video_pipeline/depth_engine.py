"""
VideoDepthEngine — Monocular depth estimation for video frames.

Two backends (auto-selected on first call):
  1. MiDaS / DPT via `torch` + `timm`  (best quality, GPU-accelerated)
  2. Edge-density fallback (CPU, no extra deps) — coarser but functional

Output is always a float32 H×W array normalised to [0, 1] where
1.0 = CLOSEST to camera.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple


class VideoDepthEngine:
    """Estimates a per-frame depth map and extracts subject depth."""

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._model  = None
        self._transform = None
        self._backend = "edge"   # "midas" | "edge"
        self._try_load_midas()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------
    def _try_load_midas(self) -> None:
        try:
            import torch
            import timm  # noqa: F401  – just check presence

            model_type = "MiDaS_small"
            self._model = torch.hub.load(
                "intel-isl/MiDaS", model_type, trust_repo=True, verbose=False
            )
            device = (
                torch.device("cuda")
                if self._device == "cuda" and torch.cuda.is_available()
                else torch.device("cpu")
            )
            self._model.to(device)
            self._model.eval()

            transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True, verbose=False
            )
            self._transform = transforms.small_transform
            self._device_obj = device
            self._torch = torch
            self._backend = "midas"
            print(f"[DepthEngine] MiDaS loaded on {device}")
        except Exception as e:
            print(f"[DepthEngine] MiDaS unavailable ({e}) — using edge-density fallback")
            self._backend = "edge"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a float32 depth map shaped (H, W), normalised [0,1].
        Values closer to 1.0 are nearer to the camera.
        """
        if self._backend == "midas":
            return self._midas_depth(frame)
        return self._edge_depth(frame)

    def get_subject_depth(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> float:
        """
        Average depth inside a tracker bbox (x,y,w,h).
        Returns a value in [0, 1].
        """
        x, y, w, h = bbox
        fh, fw = depth_map.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)
        if x2 <= x1 or y2 <= y1:
            return 0.5
        patch = depth_map[y1:y2, x1:x2]
        return float(np.mean(patch))

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------
    def _midas_depth(self, frame: np.ndarray) -> np.ndarray:
        import torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = self._transform(rgb).to(self._device_obj)

        with torch.no_grad():
            pred = self._model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        raw = pred.cpu().numpy().astype(np.float32)
        # MiDaS outputs inverse-depth (larger = closer)
        dmin, dmax = raw.min(), raw.max()
        if dmax > dmin:
            norm = (raw - dmin) / (dmax - dmin)
        else:
            norm = np.full_like(raw, 0.5)
        return norm  # 1.0 = closest

    def _edge_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Lightweight proxy: local edge density → sharpness → closeness.

        Logic: objects in sharp focus (high local edge density) are treated
        as closer to the camera.  Apply strong Gaussian blur to spread the
        depth gradient outward.
        """
        h, w = frame.shape[:2]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge density via Laplacian magnitude
        lap   = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        edges = np.abs(lap)

        # Spread edges with a large blur to create smooth depth gradient
        ksize = max(31, (min(h, w) // 8) | 1)   # odd, at least 31
        depth = cv2.GaussianBlur(edges, (ksize, ksize), 0)

        # Normalise to [0,1]
        dmin, dmax = depth.min(), depth.max()
        if dmax > dmin:
            depth = (depth - dmin) / (dmax - dmin)
        else:
            depth = np.full((h, w), 0.5, dtype=np.float32)

        return depth.astype(np.float32)

    @property
    def backend(self) -> str:
        return self._backend
