"""
AutofocusModule — Dynamic Tracking-Based Autofocus + Subject Isolation
Two modes:
1. BLUR mode: Tracking + Gaussian blur background (TrackingAutofocusEngine)
2. ISOLATION mode: Body focus + hard subject isolation (BodyFocusEngine)

Thin BaseAIModule wrapper; no other changes to the pipeline needed.
"""
import time
import numpy as np
from typing import Optional, Dict, Any, Literal

from core.base_module import BaseAIModule, InferenceResult
from core.tracking_autofocus_engine import TrackingAutofocusEngine
from core.body_focus_engine import BodyFocusEngine, create_body_focus_engine
from core.person_segmentation import PersonSegmentation, create_segmenter
from core.subject_isolation_renderer import SubjectIsolationRenderer, create_isolation_renderer
from modules.yolo_seg_module import YOLOv8SegModule
from modules.depth_estimator import DepthEstimator
from core.proximity_blur_engine import ProximityBlurEngine


class AutofocusModule(BaseAIModule):
    """
    Autofocus mode driven by OpenCV CSRT tracker.
    
    BLUR MODE:
      Click a subject → tracker locks on → blur follows the subject.
    
    ISOLATION MODE:
      Click a person → detect full body → track → segment → hard isolation (no blur).
    
    Double-click → reset.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        cfg = config or {}

        # Mode: 'blur' or 'isolation'
        self.mode: Literal['blur', 'isolation'] = cfg.get('mode', 'blur')

        # Engine selection:
        #   proximity_mode=True  → ProximityBlurEngine  (lightweight, 20-30 FPS,
        #                           no seg/depth, spatial distance blur)
        #   proximity_mode=False → TrackingAutofocusEngine (depth + YOLO-seg)
        self.proximity_mode: bool = cfg.get('proximity_mode', True)

        # ── Proximity engine (Features 1-5: lightweight cinematic blur) ──
        self.proximity_engine = ProximityBlurEngine({
            'bbox_size':         cfg.get('bbox_size',         120),
            'min_blur_k':        cfg.get('min_blur_k',        5),
            'max_blur_k':        cfg.get('max_blur_k',        35),
            'feather':           cfg.get('feather',           10),
            'grace_period':      cfg.get('grace_period',      0.8),
            'bbox_alpha':        cfg.get('bbox_alpha',        0.70),
            'blur_alpha':        cfg.get('blur_alpha',        0.80),
            'transition_frames': cfg.get('transition_frames', 20),
        })

        # Blur engine (TrackingAutofocusEngine) — full depth+seg pipeline
        self.blur_engine = TrackingAutofocusEngine({
            'bbox_size':    cfg.get('bbox_size',    120),
            'focus_radius': cfg.get('focus_radius', 150),
            'feather':      cfg.get('feather',       40),
            'blur_ksize':   cfg.get('blur_ksize',    51),
            'grace_period': cfg.get('grace_period',  0.8),
            'use_segmentation': cfg.get('use_segmentation', True),
            'seg_threshold':    cfg.get('seg_threshold', 0.5),
            'seg_dilation':     cfg.get('seg_dilation', 2),
            'seg_feather':      cfg.get('seg_feather', 2),
            'seg_frame_skip':   cfg.get('seg_frame_skip', 0),
            # Depth blur tuning
            'depth_blur_k':         cfg.get('depth_blur_k',         3.5),
            'spatial_threshold_px': cfg.get('spatial_threshold_px', 180.0),
            'spatial_reduce_frac':  cfg.get('spatial_reduce_frac',  0.45),
            'temporal_alpha':       cfg.get('temporal_alpha',        0.80),
            'rack_focus_duration':  cfg.get('rack_focus_duration',   0.40),
        })

        # Isolation engine (BodyFocusEngine) — lazy init
        self.isolation_engine: Optional[BodyFocusEngine] = None
        self.person_segmenter: Optional[PersonSegmentation] = None
        self.isolation_renderer: Optional[SubjectIsolationRenderer] = None

        # Detector reference (injected later via set_detector)
        self.detector = None

        # YOLOv8-seg module — instance segmentation for Full Silhouette Lock
        self.yolo_seg_module: Optional[YOLOv8SegModule] = None

        # DepthEstimator — per-pixel depth-based cinematic blur (Tasks 1-7)
        self.depth_estimator: Optional[DepthEstimator] = None

        self._last_composited: Optional[np.ndarray] = None
        eng_name = 'ProximityBlurEngine' if self.proximity_mode else 'TrackingAutofocusEngine'
        print(f"✓ AutofocusModule (mode={self.mode}  engine={eng_name}) created")

    # ------------------------------------------------------------------ #
    #  BaseAIModule interface                                               #
    # ------------------------------------------------------------------ #

    def initialize(self) -> bool:
        if self.proximity_mode:
            # ── LIGHTWEIGHT PATH: ProximityBlurEngine ────────────────────
            # No depth model, no segmentation, no extra inference.
            # Still optionally load YOLO detections for proximity map.
            if self.yolo_seg_module is None:
                self.yolo_seg_module = YOLOv8SegModule()
            try:
                self.yolo_seg_module.initialize()
                print("✓ YOLOv8SegModule initialised (proximity detection source)")
            except Exception as e:
                print(f"⚠ YOLOv8SegModule init failed: {e} — proximity blur uses track bbox only")
                self.yolo_seg_module = None

            if self.mode == 'isolation':
                self._init_isolation_engine()
            self.is_initialized = True
            print("✓ AutofocusModule → ProximityBlurEngine (20-30 FPS lightweight mode)")
            return True

        # ── FULL PATH: TrackingAutofocus + Depth + YOLO-seg ─────────
        # ── YOLOv8-seg (primary segmentation path) ──────────────────────
        if self.yolo_seg_module is None:
            self.yolo_seg_module = YOLOv8SegModule()
        try:
            self.yolo_seg_module.initialize()
            print("✓ YOLOv8SegModule initialised for Full Silhouette Lock Mode")
        except Exception as e:
            print(f"⚠ YOLOv8SegModule init failed: {e} — will use geometric fallback")
            self.yolo_seg_module = None

        # No MediaPipe segmenter needed (YOLO-seg replaces it)
        self.blur_engine.set_segmenter(None)

        # ── DepthEstimator (Tasks 1-7) ──────────────────────────────
        if self.depth_estimator is None:
            self.depth_estimator = DepthEstimator()
        try:
            self.depth_estimator.initialize()
            self.blur_engine.set_depth_estimator(self.depth_estimator)
            print("✓ DepthEstimator wired to blur engine (depth-based cinematic blur ON)")
        except Exception as e:
            print(f"⚠ DepthEstimator init failed: {e} — falling back to binary mask")
            self.depth_estimator = None

        # Initialize isolation engine if in isolation mode
        if self.mode == 'isolation':
            self._init_isolation_engine()

        self.is_initialized = True
        print(f"✓ AutofocusModule.initialize() — mode={self.mode}")
        return True

    def _init_isolation_engine(self):
        """Lazy initialization of isolation components."""
        if self.isolation_engine is not None:
            return  # Already initialized
        
        # Reuse shared segmenter (already created in initialize())
        if self.person_segmenter is None:
            try:
                self.person_segmenter = create_segmenter()
            except Exception as e:
                print(f"⚠ Failed to create segmenter: {e}")
                self.person_segmenter = None
        
        # Create isolation renderer
        self.isolation_renderer = create_isolation_renderer(background_mode='black')
        
        # Create body focus engine
        self.isolation_engine = create_body_focus_engine(
            detector=self.detector,
            segmenter=self.person_segmenter,
            isolation_renderer=self.isolation_renderer,
        )
        
        print("✓ Isolation engine initialized")

    def process_frame(self, frame: np.ndarray) -> InferenceResult:
        t0 = time.time()
        self.frame_count += 1

        # Route to appropriate engine based on mode
        if self.mode == 'blur':
            composited = self._process_blur_mode(frame)
        elif self.mode == 'isolation':
            composited = self._process_isolation_mode(frame)
        else:
            composited = frame  # Fallback: no effect
        
        self._last_composited = composited

        inference_ms = (time.time() - t0) * 1000
        self.total_inference_time += inference_ms

        # Get status from active engine
        status = self._get_status()
        metrics = {
            'active':       status['state'] != 'idle',
            'state':        status['state'],
            'mode':         self.mode,
            'focus_point':  status.get('center'),
            'focus_radius': status.get('focus_radius', 0),
            'blur_ksize':   status.get('blur_ksize', 0),
            'person_id':    status.get('person_id'),
        }

        return self._create_result([], metrics, 'normal', inference_ms)
    
    def set_yolo_seg(self, module: 'YOLOv8SegModule'):
        """Inject an already-initialised YOLOv8SegModule (called from main.py)."""
        self.yolo_seg_module = module
        print("✓ YOLOv8SegModule injected into AutofocusModule")

    def _active_blur_engine(self):
        """Return whichever blur engine is currently active."""
        return self.proximity_engine if self.proximity_mode else self.blur_engine

    def _process_blur_mode(self, frame: np.ndarray) -> np.ndarray:
        """Route to proximity or depth engine."""
        detections: list = []
        if self.yolo_seg_module is not None:
            try:
                detections = self.yolo_seg_module.detect(frame)
            except Exception as e:
                print(f"⚠ YOLO detect error: {e}")

        if self.proximity_mode:
            self.proximity_engine.feed_detections(detections)
            return self.proximity_engine.process_frame(frame)
        else:
            self.blur_engine.feed_seg_detections(detections)
            return self.blur_engine.process_frame(frame)
    
    def _process_isolation_mode(self, frame: np.ndarray) -> np.ndarray:
        """Process frame in isolation mode (BodyFocusEngine)."""
        if self.isolation_engine is None:
            self._init_isolation_engine()
        
        # Run detection if detector available
        detections = None
        if self.detector is not None:
            try:
                detections = self.detector.detect(frame)
            except Exception as e:
                print(f"[AutofocusModule] Detection failed: {e}")
        
        # Process with body focus engine
        result = self.isolation_engine.process_frame(frame, detections=detections)
        
        # Return isolated frame if available, else original
        return result.get('isolated_frame') or frame
    
    def _get_status(self) -> Dict[str, Any]:
        """Get status from active engine."""
        if self.mode == 'blur':
            return self._active_blur_engine().get_status()
        elif self.mode == 'isolation' and self.isolation_engine is not None:
            result = self.isolation_engine.process_frame.__self__.__dict__
            return {
                'state': self.isolation_engine.state.value,
                'center': None,
                'person_id': self.isolation_engine.selected_person_id,
                'focus_radius': 0,
                'blur_ksize': 0,
            }
        else:
            return {'state': 'idle'}

    def cleanup(self):
        self.blur_engine.cleanup()
        self.proximity_engine.cleanup()
        
        if self.isolation_engine is not None:
            self.isolation_engine.cleanup()
        
        if self.person_segmenter is not None:
            self.person_segmenter.cleanup()
        
        self._last_composited = None
        self.is_initialized = False
        print("✓ AutofocusModule cleanup")

    # ------------------------------------------------------------------ #
    #  Click events (called by websocket handler)                          #
    # ------------------------------------------------------------------ #

    def on_click(self, x: int, y: int):
        """Route click event to active engine."""
        if self.mode == 'blur':
            self._active_blur_engine().on_click(x, y)
        elif self.mode == 'isolation':
            if self.isolation_engine is None:
                self._init_isolation_engine()
            self.isolation_engine.on_click(x, y)

    def on_double_click(self):
        """Route double-click event to active engine."""
        if self.mode == 'blur':
            self._active_blur_engine().on_double_click()
        elif self.mode == 'isolation' and self.isolation_engine is not None:
            self.isolation_engine.on_double_click()

    def set_focus_radius(self, radius: int):
        """Only affects blur mode."""
        self._active_blur_engine().set_focus_radius(radius)

    def set_blur_strength(self, strength: float):
        """Only affects blur mode."""
        self._active_blur_engine().set_blur_strength(strength)
    
    def set_mode(self, mode: Literal['blur', 'isolation']):
        """
        Switch between blur and isolation modes.
        
        Args:
            mode: 'blur' or 'isolation'
        """
        if mode not in ['blur', 'isolation']:
            print(f"⚠ Invalid mode: {mode}. Use 'blur' or 'isolation'.")
            return
        
        if mode == self.mode:
            return  # Already in this mode
        
        self.mode = mode
        # Reset both engines when switching
        self.blur_engine.on_double_click()
        self.proximity_engine.on_double_click()
        if self.isolation_engine is not None:
            self.isolation_engine.on_double_click()
        print(f"✓ AutofocusModule mode switched to: {mode}")
    
    def set_detector(self, detector):
        """
        Inject detector module for isolation mode.
        
        Args:
            detector: ObjectDetection module (YOLO/RT-DETR)
        """
        self.detector = detector
        
        # Update isolation engine if already initialized
        if self.isolation_engine is not None:
            self.isolation_engine.detector = detector
        
        print("✓ Detector injected into AutofocusModule")

    def get_composited_frame(self) -> Optional[np.ndarray]:
        return self._last_composited

