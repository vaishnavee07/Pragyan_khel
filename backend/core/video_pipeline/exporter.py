"""
VideoExporter — Re-processes the entire video at full resolution and
writes out a rendered MP4.

Pipeline per frame:
  1. Read frame from source via VideoLoader
  2. Update VideoTracker
  3. Estimate depth via VideoDepthEngine
  4. Composite blur via VideoBlurRenderer
  5. Write to cv2.VideoWriter

Audio is preserved via a separate FFmpeg pass (optional).
If FFmpeg is unavailable the export is video-only.

Progress is reported via a callback:  progress_cb(current_frame, total_frames)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import uuid
import cv2
import numpy as np
from typing import Callable, Optional, Tuple

from .video_loader   import VideoLoader
from .video_tracker  import VideoTracker
from .depth_engine   import VideoDepthEngine
from .blur_renderer  import VideoBlurRenderer


class VideoExporter:
    """Renders and exports a processed video file."""

    def __init__(self, tmp_dir: str = "/tmp/sentravision_exports") -> None:
        os.makedirs(tmp_dir, exist_ok=True)
        self._tmp_dir = tmp_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def export(
        self,
        source_path:   str,
        init_bbox:     Optional[Tuple[int, int, int, int]],
        blur_strength: float = 0.85,
        depth_bias:    float = 0.0,
        progress_cb:   Optional[Callable[[int, int], None]] = None,
    ) -> Optional[str]:
        """
        Re-render *source_path* with cinematic blur and save to a new file.

        Parameters
        ----------
        source_path   : absolute path to the uploaded video
        init_bbox     : (x, y, w, h) of the tracked subject on frame-0;
                        None → no tracking, full frame is sharp
        blur_strength : 0.0–1.0
        depth_bias    : focus plane offset  (−0.5 … +0.5)
        progress_cb   : called with (current_frame, total_frames) periodically

        Returns
        -------
        Path to rendered file, or None on failure.
        """
        loader = VideoLoader(source_path)
        if not loader.open():
            return None

        meta        = loader.get_metadata()
        fps         = meta["fps"]
        fw, fh      = meta["width"], meta["height"]
        frame_count = meta["frame_count"]

        # Output path
        out_id   = uuid.uuid4().hex[:10]
        out_name = f"export_{out_id}.mp4"
        out_path = os.path.join(self._tmp_dir, out_name)
        tmp_path = out_path.replace(".mp4", "_novid.mp4")  # before audio mix

        # Writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (fw, fh))
        if not writer.isOpened():
            print("[Exporter] VideoWriter failed to open")
            loader.release()
            return None

        # Pipeline objects
        tracker  = VideoTracker()
        depther  = VideoDepthEngine()
        renderer = VideoBlurRenderer()

        initialized = False
        subject_depth  = 0.5
        depth_plane    = 0.5        # for rack-focus interpolation

        print(f"[Exporter] Starting  {frame_count} frames  {fw}×{fh}  {fps:.1f}fps")

        try:
            for idx, frame in loader.frame_generator():
                # Init tracker on first frame
                if not initialized and init_bbox is not None:
                    ok = tracker.initialize(frame, init_bbox)
                    if ok:
                        initialized = True

                # Track
                body_bbox = None
                raw_bbox  = None
                if initialized:
                    state, raw_bbox, body_bbox = tracker.update(frame)
                    if state == "idle":
                        initialized = False

                # Depth
                depth_map = depther.estimate(frame)

                # Subject depth (smooth via rack-focus EMA)
                if raw_bbox is not None:
                    sd = depther.get_subject_depth(depth_map, raw_bbox)
                    depth_plane = renderer.interpolate_depth(depth_plane, sd)
                    subject_depth = depth_plane

                # Render
                out_frame = renderer.render(
                    frame, depth_map, subject_depth, body_bbox,
                    blur_strength=blur_strength,
                    depth_bias=depth_bias,
                )
                writer.write(out_frame)

                # Progress callback every 10 frames
                if progress_cb and idx % 10 == 0:
                    progress_cb(idx, frame_count)

        except Exception as e:
            print(f"[Exporter] Error at frame {idx}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            writer.release()
            loader.release()

        # Try to mux original audio back with FFmpeg
        final_path = self._mux_audio(source_path, tmp_path, out_path)
        if progress_cb:
            progress_cb(frame_count, frame_count)

        print(f"[Exporter] Done → {final_path}")
        return final_path

    # ------------------------------------------------------------------
    # Audio mux
    # ------------------------------------------------------------------
    def _mux_audio(
        self,
        source_path: str,
        video_only:  str,
        output:      str,
    ) -> str:
        """
        If FFmpeg is available, pull audio from source and merge with
        the rendered video.  Otherwise just rename video_only → output.
        """
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            os.replace(video_only, output)
            return output

        cmd = [
            ffmpeg, "-y",
            "-i", video_only,
            "-i", source_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            output,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            if os.path.isfile(video_only):
                os.remove(video_only)
            return output
        except Exception as e:
            print(f"[Exporter] FFmpeg audio mux failed ({e}) — returning video-only")
            os.replace(video_only, output)
            return output
