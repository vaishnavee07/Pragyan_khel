"""
VideoSession — In-memory state for one video upload/edit session.

Holds:
  • VideoLoader     (file access)
  • VideoTracker    (single subject)
  • VideoDepthEngine
  • VideoBlurRenderer

Manages play/pause, seek, and per-frame rendering so the WebSocket
handler can stay thin.
"""
from __future__ import annotations

import threading
import time
import uuid
import cv2
import numpy as np
from typing import Callable, Optional, Tuple

from .video_pipeline.video_loader   import VideoLoader
from .video_pipeline.video_tracker  import VideoTracker
from .video_pipeline.depth_engine   import VideoDepthEngine
from .video_pipeline.blur_renderer  import VideoBlurRenderer


class VideoSession:
    """One session = one uploaded video + all processing state."""

    def __init__(self, file_path: str) -> None:
        self.session_id       = uuid.uuid4().hex[:10]
        self.file_path        = file_path

        self.loader           = VideoLoader(file_path)
        self.tracker          = VideoTracker()
        self.depther          = VideoDepthEngine()
        self.renderer         = VideoBlurRenderer()

        # Playback state
        self.current_frame    = 0
        self.is_playing       = False
        self._lock            = threading.Lock()

        # Tracking / rendering settings
        self.blur_strength    = 0.85
        self.depth_bias       = 0.0
        self.show_depth_debug = False

        # Rack focus: smoothed depth plane
        self._depth_plane     = 0.5
        self._subject_depth   = 0.5

        # Initialised flag
        self._open            = False

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------
    def open(self) -> bool:
        ok = self.loader.open()
        self._open = ok
        return ok

    def close(self) -> None:
        self.loader.release()
        self.tracker.reset()
        self._open = False

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------
    @property
    def metadata(self) -> dict:
        return self.loader.get_metadata()

    @property
    def frame_count(self) -> int:
        return self.loader.frame_count

    @property
    def fps(self) -> float:
        return self.loader.fps

    # ------------------------------------------------------------------
    # Click → init tracker
    # ------------------------------------------------------------------
    def on_click(
        self,
        frame_index: int,
        click_x: int,
        click_y: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Called when the user clicks on a frame.
        Runs GrabCut detection to find a bbox, then initialises tracker.
        Returns the detected (x,y,w,h) or None.
        """
        frame = self.loader.read_frame(frame_index)
        if frame is None:
            return None

        from .video_pipeline.detection_engine import VideoDetectionEngine
        det = VideoDetectionEngine()
        bbox = det.detect_at_click(frame, click_x, click_y)
        if bbox is None:
            # Fallback: 120×120 centred at click
            hw = 60
            x1 = max(0, click_x - hw)
            y1 = max(0, click_y - hw)
            w  = min(frame.shape[1] - x1, hw * 2)
            h  = min(frame.shape[0] - y1, hw * 2)
            bbox = (x1, y1, w, h)

        self.tracker.reset()
        # Seek loader to frame_index and initialise
        frame_init = self.loader.read_frame(frame_index)
        if frame_init is not None:
            self.tracker.initialize(frame_init, bbox)
            # Set seek position
            self.current_frame = frame_index
        return bbox

    def reset_tracking(self) -> None:
        self.tracker.reset()

    # ------------------------------------------------------------------
    # Render single frame
    # ------------------------------------------------------------------
    def get_rendered_frame(
        self,
        frame_index: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Render frame at *frame_index* (default: current_frame).
        Advances tracker if playing forward.
        Returns composited BGR frame.
        """
        if not self._open:
            return None

        idx = frame_index if frame_index is not None else self.current_frame
        frame = self.loader.read_frame(idx)
        if frame is None:
            return None

        # Track update
        state, raw_bbox, body_bbox = self.tracker.update(frame)

        # Depth estimate
        depth_map = self.depther.estimate(frame)

        # Smooth depth plane (rack focus)
        if raw_bbox is not None:
            sd = self.depther.get_subject_depth(depth_map, raw_bbox)
            self._depth_plane = self.renderer.interpolate_depth(
                self._depth_plane, sd
            )
            self._subject_depth = self._depth_plane

        # Debug overlay
        if self.show_depth_debug:
            return self.renderer.render_depth_heatmap(depth_map)

        return self.renderer.render(
            frame,
            depth_map,
            self._subject_depth,
            body_bbox,
            blur_strength=self.blur_strength,
            depth_bias=self.depth_bias,
        )

    def advance(self) -> bool:
        """Advance current_frame by 1.  Returns False at end of video."""
        if self.current_frame < self.frame_count - 1:
            self.current_frame += 1
            return True
        return False

    def seek(self, frame_index: int) -> None:
        self.current_frame = max(0, min(frame_index, self.frame_count - 1))
        # Re-generate tracker state from scratch at new position
        # (tracker only tracks forward — reset to idle on seek)
        self.tracker.reset()
