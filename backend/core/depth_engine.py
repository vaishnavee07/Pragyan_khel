"""
Depth Engine - CINEMATIC AUTOFOCUS
Monocular depth estimation using MiDaS
Produces normalized depth map [0,1] for every frame
"""
import cv2
import numpy as np
import time
import threading
from typing import Optional, Tuple
from collections import deque


class DepthEngine:
    """
    Lightweight monocular depth estimation
    Supports MiDaS-small (CPU) and DPT-Hybrid (GPU)
    Thread-safe with async inference support
    """

    MODELS = {
        'midas_small': {
            'type': 'MiDaS_small',
            'transform': 'small_transform',
            'input_size': 256,
        },
        'dpt_hybrid': {
            'type': 'DPT_Hybrid',
            'transform': 'dpt_transform',
            'input_size': 384,
        },
        'dpt_large': {
            'type': 'DPT_Large',
            'transform': 'dpt_transform',
            'input_size': 384,
        },
    }

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.model_name     = cfg.get('model', 'midas_small')
        self.device_pref    = cfg.get('device', 'auto')   # 'cpu' / 'cuda' / 'auto'
        self.input_size     = cfg.get('input_size', 256)
        self.temporal_alpha = cfg.get('temporal_alpha', 0.5)   # smoothing weight

        self.model      = None
        self.transform  = None
        self.device     = None
        self.is_ready   = False

        # Temporal averaging buffer
        self._prev_depth: Optional[np.ndarray] = None

        # Thread-safety for background inference
        self._lock        = threading.Lock()
        self._latest_map: Optional[np.ndarray] = None

        # Performance
        self._infer_times: deque = deque(maxlen=30)

        print(f"✓ DepthEngine created  model={self.model_name}  device={self.device_pref}")

    # ------------------------------------------------------------------ #
    #  Initialization                                                       #
    # ------------------------------------------------------------------ #

    def initialize(self) -> bool:
        """Load MiDaS model.  Falls back to radial-blur sentinel on failure."""
        try:
            import torch

            # Resolve device
            if self.device_pref == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(self.device_pref)

            print(f"⏳ Loading MiDaS [{self.model_name}] on {self.device} …")

            self.model = torch.hub.load(
                'intel-isl/MiDaS',
                self.MODELS[self.model_name]['type'],
                trust_repo=True
            )
            self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load(
                'intel-isl/MiDaS',
                'transforms',
                trust_repo=True
            )
            tf_attr = self.MODELS[self.model_name]['transform']
            self.transform = getattr(midas_transforms, tf_attr)

            # Use half precision on CUDA for speed
            if self.device.type == 'cuda':
                self.model = self.model.half()

            self.is_ready = True
            print(f"✓ MiDaS loaded  device={self.device}")
            return True

        except Exception as e:
            print(f"⚠ MiDaS unavailable ({e}) — radial-fallback mode active")
            self.is_ready = False
            return False

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns depth map (float32, [0,1]) same resolution as *frame*.
        Falls back to radial blur if model unavailable.
        """
        if not self.is_ready:
            return self._radial_fallback(frame)

        t0 = time.time()
        depth = self._run_midas(frame)
        self._infer_times.append((time.time() - t0) * 1000)

        # Temporal smoothing
        if self._prev_depth is not None and self._prev_depth.shape == depth.shape:
            depth = (self.temporal_alpha * depth +
                     (1.0 - self.temporal_alpha) * self._prev_depth)
        self._prev_depth = depth.copy()

        with self._lock:
            self._latest_map = depth

        return depth

    def get_depth_at(self, depth_map: np.ndarray, x: int, y: int) -> float:
        """Sample depth value at pixel (x, y), clamped to valid range."""
        h, w = depth_map.shape[:2]
        px = int(np.clip(x, 0, w - 1))
        py = int(np.clip(y, 0, h - 1))
        return float(depth_map[py, px])

    def get_stats(self) -> dict:
        times = list(self._infer_times)
        return {
            'model': self.model_name,
            'device': str(self.device),
            'ready': self.is_ready,
            'avg_ms': round(float(np.mean(times)), 1) if times else 0,
            'min_ms': round(float(np.min(times)), 1) if times else 0,
            'max_ms': round(float(np.max(times)), 1) if times else 0,
        }

    def cleanup(self):
        """Release model from memory."""
        self.model     = None
        self.transform = None
        self.is_ready  = False
        self._prev_depth = None
        print("✓ DepthEngine cleanup complete")

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run_midas(self, frame: np.ndarray) -> np.ndarray:
        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)

        if self.device.type == 'cuda':
            inp = inp.half()

        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=frame.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()

        depth = pred.cpu().float().numpy()

        # Normalize to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-5:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        return depth.astype(np.float32)

    def _radial_fallback(self, frame: np.ndarray) -> np.ndarray:
        """
        Radial depth map centred in the frame (closer = higher value).
        Used when MiDaS is unavailable.
        """
        h, w = frame.shape[:2]
        cy, cx = h / 2, w / 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        # Invert: centre = 1.0 (near), edges = 0.0 (far)
        depth = 1.0 - (dist / max_dist)
        return depth.astype(np.float32)
