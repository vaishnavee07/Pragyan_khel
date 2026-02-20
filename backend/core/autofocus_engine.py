"""
Autofocus Engine - CINEMATIC AUTOFOCUS
Manages active focus point, smooth transition animation,
and per-pixel blur-strength map computation.
"""
import time
import numpy as np
from typing import Optional, Tuple


# --------------------------------------------------------------------------- #
#  Easing helpers                                                               #
# --------------------------------------------------------------------------- #

def _ease_in_out(t: float) -> float:
    """Cubic ease-in-out: t in [0,1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- #
#  AutofocusEngine                                                              #
# --------------------------------------------------------------------------- #

class AutofocusEngine:
    """
    Maintains:
    - active_focus_point: current target (x, y)
    - smooth transition when focus changes
    - blur-strength map from depth difference
    """

    def __init__(self, config: dict = None):
        cfg = config or {}

        # Focus region
        self.focus_radius: int   = cfg.get('focus_radius', 120)  # pixels
        self.max_blur_strength   = cfg.get('max_blur_strength', 3.0)  # multiplier
        self.depth_sensitivity   = cfg.get('depth_sensitivity', 2.0)  # sharpens falloff

        # Transition
        self.transition_ms: float = cfg.get('transition_ms', 300.0)

        # State
        self.active_focus_point: Optional[Tuple[int, int]] = None
        self._prev_focus_point:  Optional[Tuple[int, int]] = None
        self._transition_start:  float = 0.0
        self._transition_progress: float = 1.0  # 1 = complete

        # Cached depth at focus
        self._focus_depth: float = 0.5
        self._prev_focus_depth: float = 0.5

        # Mode
        self.is_active: bool = False

        print(f"✓ AutofocusEngine  radius={self.focus_radius}px  "
              f"transition={self.transition_ms}ms")

    # ------------------------------------------------------------------ #
    #  Click handling                                                       #
    # ------------------------------------------------------------------ #

    def handle_click(self, x: int, y: int):
        """Set new focus point, begin transition animation."""
        self._prev_focus_point  = self.active_focus_point
        self._prev_focus_depth  = self._focus_depth
        self.active_focus_point = (int(x), int(y))
        self._transition_start  = time.time()
        self._transition_progress = 0.0
        self.is_active = True
        print(f"[AUTOFOCUS] Focus set to: ({x}, {y})  is_active={self.is_active}")

    def handle_double_click(self):
        """Reset focus — no blur applied."""
        self.active_focus_point   = None
        self._prev_focus_point    = None
        self._transition_progress = 1.0
        self.is_active            = False
        print("[AUTOFOCUS] Focus reset")

    # ------------------------------------------------------------------ #
    #  Frame processing                                                     #
    # ------------------------------------------------------------------ #

    def compute_blur_map(
        self,
        depth_map: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Returns float32 array [0,1] same H×W as frame.
        0 = no blur (in focus) · 1 = maximum blur
        Returns zeros if no focus point set.
        """
        if not self.is_active or self.active_focus_point is None:
            return np.zeros(frame_shape[:2], dtype=np.float32)

        h, w = frame_shape[:2]
        fx, fy = self.active_focus_point

        # Update transition
        self._tick_transition()

        # Interpolate effective focus point during transition
        eff_fx, eff_fy, eff_depth = self._interpolated_focus(depth_map, h, w)

        # ── depth difference map ──────────────────────────────────────
        depth_diff = np.abs(depth_map - eff_depth)
        # Sharpen falloff
        depth_diff = np.power(depth_diff, 1.0 / self.depth_sensitivity).astype(np.float32)

        # ── Gaussian spatial mask around focus region ──────────────────
        Y, X     = np.ogrid[:h, :w]
        spatial  = np.sqrt((X - eff_fx) ** 2 + (Y - eff_fy) ** 2)
        sigma    = max(self.focus_radius, 1)
        gauss    = np.exp(-(spatial ** 2) / (2 * sigma ** 2)).astype(np.float32)
        # outside_mask: 0 at focus centre, 1 far away
        outside_mask = (1.0 - gauss)

        # ── BLUR MAP: spatial is primary driver, depth modulates edges ──
        # outside_mask alone guarantees blur beyond the focus radius
        # depth_diff boosts blur where depth differs (sharpens subject isolation)
        depth_boost = np.clip(0.4 + depth_diff * 0.6, 0.0, 1.0)
        blur_map = outside_mask * depth_boost
        blur_map = np.clip(blur_map * self.max_blur_strength, 0.0, 1.0)

        print(f"[AUTOFOCUS] blur_map  max={blur_map.max():.3f}  mean={blur_map.mean():.3f}  "
              f"focus=({self.active_focus_point})  radius={self.focus_radius}")

        return blur_map.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Internal                                                             #
    # ------------------------------------------------------------------ #

    def _tick_transition(self):
        """Advance transition progress based on elapsed time."""
        if self._transition_progress >= 1.0:
            return
        elapsed = (time.time() - self._transition_start) * 1000
        self._transition_progress = min(1.0, elapsed / max(self.transition_ms, 1))

    def _interpolated_focus(
        self,
        depth_map: np.ndarray,
        h: int,
        w: int,
    ) -> Tuple[float, float, float]:
        """Ease-interpolated focus position + depth value."""
        tx = _ease_in_out(self._transition_progress)

        # Clamp current target
        fx = int(np.clip(self.active_focus_point[0], 0, w - 1))
        fy = int(np.clip(self.active_focus_point[1], 0, h - 1))
        current_depth = float(depth_map[fy, fx])

        if self._prev_focus_point is None or self._transition_progress >= 1.0:
            self._focus_depth = current_depth
            return float(fx), float(fy), current_depth

        # Prev position
        px = int(np.clip(self._prev_focus_point[0], 0, w - 1))
        py = int(np.clip(self._prev_focus_point[1], 0, h - 1))

        eff_x     = px + tx * (fx - px)
        eff_y     = py + tx * (fy - py)
        eff_depth = self._prev_focus_depth + tx * (current_depth - self._prev_focus_depth)

        self._focus_depth = eff_depth
        return eff_x, eff_y, eff_depth

    # ------------------------------------------------------------------ #
    #  Status                                                               #
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        return {
            'active': self.is_active,
            'focus_point': self.active_focus_point,
            'focus_depth': round(self._focus_depth, 3),
            'transition_pct': round(self._transition_progress * 100),
            'focus_radius': self.focus_radius,
        }

    def set_focus_radius(self, radius: int):
        self.focus_radius = max(20, int(radius))

    def set_blur_strength(self, strength: float):
        self.max_blur_strength = float(np.clip(strength, 0.1, 5.0))

    def cleanup(self):
        self.active_focus_point = None
        self.is_active = False
        print("✓ AutofocusEngine cleanup complete")
