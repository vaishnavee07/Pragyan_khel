"""
AutofocusModule — Dynamic Tracking-Based Autofocus
Delegates all tracking + blur logic to TrackingAutofocusEngine.
Thin BaseAIModule wrapper; no other changes to the pipeline needed.
"""
import time
import numpy as np
from typing import Optional, Dict, Any

from core.base_module import BaseAIModule, InferenceResult
from core.tracking_autofocus_engine import TrackingAutofocusEngine


class AutofocusModule(BaseAIModule):
    """
    Autofocus mode driven by OpenCV CSRT tracker.
    Click a subject → tracker locks on → blur follows the subject.
    Double-click → reset.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        cfg = config or {}

        self.engine = TrackingAutofocusEngine({
            'bbox_size':    cfg.get('bbox_size',    120),
            'focus_radius': cfg.get('focus_radius', 150),
            'feather':      cfg.get('feather',       40),
            'blur_ksize':   cfg.get('blur_ksize',    51),
            'grace_period': cfg.get('grace_period',  0.8),
        })

        self._last_composited: Optional[np.ndarray] = None
        print("✓ AutofocusModule (CSRT tracking) created")

    # ------------------------------------------------------------------ #
    #  BaseAIModule interface                                               #
    # ------------------------------------------------------------------ #

    def initialize(self) -> bool:
        self.is_initialized = True
        print("✓ AutofocusModule.initialize() — CSRT tracker ready")
        return True

    def process_frame(self, frame: np.ndarray) -> InferenceResult:
        t0 = time.time()
        self.frame_count += 1

        # Delegate full tracking + blur + overlay to engine
        composited = self.engine.process_frame(frame)
        self._last_composited = composited

        inference_ms = (time.time() - t0) * 1000
        self.total_inference_time += inference_ms

        status = self.engine.get_status()
        metrics = {
            'active':       status['state'] != 'idle',
            'state':        status['state'],
            'focus_point':  status['center'],
            'focus_radius': status['focus_radius'],
            'blur_ksize':   status['blur_ksize'],
        }

        return self._create_result([], metrics, 'normal', inference_ms)

    def cleanup(self):
        self.engine.cleanup()
        self._last_composited = None
        self.is_initialized = False
        print("✓ AutofocusModule cleanup")

    # ------------------------------------------------------------------ #
    #  Click events (called by websocket handler)                          #
    # ------------------------------------------------------------------ #

    def on_click(self, x: int, y: int):
        self.engine.on_click(x, y)

    def on_double_click(self):
        self.engine.on_double_click()

    def set_focus_radius(self, radius: int):
        self.engine.set_focus_radius(radius)

    def set_blur_strength(self, strength: float):
        self.engine.set_blur_strength(strength)

    def get_composited_frame(self) -> Optional[np.ndarray]:
        return self._last_composited

