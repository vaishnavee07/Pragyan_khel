"""
AutofocusModule — Simple Selective Blur
Keeps a circular region sharp; everything else is blurred.
No depth model. No external sub-engines. Direct cv2 operations.
"""
import time
import threading
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

from core.base_module import BaseAIModule, InferenceResult


class AutofocusModule(BaseAIModule):
    """
    Selective-blur cinematic autofocus.
    Click → focus point stored.
    Each frame → sharp circle + blurred background.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        cfg = config or {}

        self.focus_radius: int  = cfg.get('focus_radius', 150)   # pixels
        self.blur_ksize:   int  = cfg.get('blur_ksize', 51)       # GaussianBlur kernel
        self.feather:      int  = cfg.get('feather', 40)          # edge softness (px)

        # State
        self._focus_point: Optional[Tuple[int, int]] = None
        self._lock = threading.Lock()

        # Cached mask (rebuilt only when frame size or focus point changes)
        self._mask_cache:  Optional[np.ndarray] = None
        self._mask_key: tuple = ()                                # (x, y, r, H, W)

        # Output
        self._last_composited: Optional[np.ndarray] = None

        print(f"✓ AutofocusModule  radius={self.focus_radius}px  "
              f"blur_k={self.blur_ksize}  feather={self.feather}")

    # ------------------------------------------------------------------ #
    #  BaseAIModule interface                                               #
    # ------------------------------------------------------------------ #

    def initialize(self) -> bool:
        self.is_initialized = True
        print("✓ AutofocusModule.initialize() — ready (direct cv2 blur)")
        return True

    def process_frame(self, frame: np.ndarray) -> InferenceResult:
        t0 = time.time()
        self.frame_count += 1

        with self._lock:
            fp = self._focus_point

        if fp is None:
            # No focus set — pass through original frame unchanged
            self._last_composited = frame
        else:
            self._last_composited = self._apply_selective_blur(frame, fp)

        inference_ms = (time.time() - t0) * 1000
        self.total_inference_time += inference_ms

        metrics = {
            'active':       fp is not None,
            'focus_point':  fp,
            'focus_radius': self.focus_radius,
            'blur_ksize':   self.blur_ksize,
        }

        return self._create_result([], metrics, 'normal', inference_ms)

    def cleanup(self):
        self._focus_point = None
        self._mask_cache  = None
        self.is_initialized = False
        print("✓ AutofocusModule cleanup")

    # ------------------------------------------------------------------ #
    #  Click events                                                         #
    # ------------------------------------------------------------------ #

    def on_click(self, x: int, y: int):
        with self._lock:
            self._focus_point = (int(x), int(y))
            # Invalidate mask cache so it rebuilds next frame
            self._mask_cache = None
        print(f"[AUTOFOCUS] Focus set to: ({x}, {y})")

    def on_double_click(self):
        with self._lock:
            self._focus_point = None
            self._mask_cache  = None
        print("[AUTOFOCUS] Focus reset")

    def set_focus_radius(self, radius: int):
        with self._lock:
            self.focus_radius = int(radius)
            self._mask_cache  = None   # force rebuild
        print(f"[AUTOFOCUS] radius → {radius}px")

    def set_blur_strength(self, strength: float):
        # Map strength [0.2..5.0] → kernel size (must be odd, ≥3)
        k = max(3, int(strength * 20) | 1)
        with self._lock:
            self.blur_ksize  = k
            self._mask_cache = None
        print(f"[AUTOFOCUS] blur_ksize → {k}")

    def get_composited_frame(self) -> Optional[np.ndarray]:
        return self._last_composited

    # ------------------------------------------------------------------ #
    #  Core blur logic                                                      #
    # ------------------------------------------------------------------ #

    def _apply_selective_blur(
        self,
        frame: np.ndarray,
        focus_point: Tuple[int, int],
    ) -> np.ndarray:
        """
        Returns a frame where everything OUTSIDE the focus circle is blurred.
        Pipeline:
          1. Blur entire frame with strong Gaussian kernel
          2. Build soft circular mask centred at focus_point
          3. Composite: sharp × mask  +  blurred × (1 - mask)
        """
        h, w = frame.shape[:2]
        fx = int(np.clip(focus_point[0], 0, w - 1))
        fy = int(np.clip(focus_point[1], 0, h - 1))

        # ── 1. Blur the full frame ─────────────────────────────────────
        k = self.blur_ksize | 1             # ensure odd
        blurred = cv2.GaussianBlur(frame, (k, k), 0)

        # ── 2. Build or retrieve cached mask ──────────────────────────
        key = (fx, fy, self.focus_radius, h, w)
        if self._mask_cache is None or self._mask_key != key:
            self._mask_cache = self._build_mask(h, w, fx, fy,
                                                self.focus_radius, self.feather)
            self._mask_key = key

        mask = self._mask_cache          # float32 H×W [0..1], 1 = sharp

        # ── 3. Composite ──────────────────────────────────────────────
        # Expand mask to 3 channels for broadcasting
        mask3 = mask[:, :, np.newaxis]

        sharp_f   = frame.astype(np.float32)
        blurred_f = blurred.astype(np.float32)

        composite = sharp_f * mask3 + blurred_f * (1.0 - mask3)
        result = np.clip(composite, 0, 255).astype(np.uint8)

        if self.frame_count % 30 == 0:
            print(f"[AUTOFOCUS] Composited  focus=({fx},{fy})  "
                  f"radius={self.focus_radius}  blur_k={k}  "
                  f"mask_mean={mask.mean():.3f}")

        return result

    @staticmethod
    def _build_mask(
        h: int, w: int,
        cx: int, cy: int,
        radius: int,
        feather: int,
    ) -> np.ndarray:
        """
        Create a soft circular mask.
        Pixel value = 1.0 inside the focus radius (sharp)
                    = 0.0 well outside (blurred)
        Transition zone = 'feather' pixels wide.
        """
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx).astype(np.float32) ** 2 +
                       (Y - cy).astype(np.float32) ** 2)

        inner = float(radius)
        outer = float(radius + max(feather, 1))

        # Linear feather: 1.0 inside, 0.0 outside, smooth ramp in between
        mask = np.clip((outer - dist) / (outer - inner), 0.0, 1.0)
        return mask.astype(np.float32)
