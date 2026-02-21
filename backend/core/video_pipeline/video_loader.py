"""
VideoLoader — Opens an uploaded video file and streams frames on-demand.

Design constraints:
  • Never loads the whole video into RAM.
  • Caller iterates via frame_generator() or reads individual frames with
    read_frame(index).
  • Thread-safe per-instance (each session owns its own VideoLoader).
"""
from __future__ import annotations

import os
import cv2
from typing import Generator, Optional, Tuple


class VideoLoader:
    """Lazy OpenCV-based video decoder."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._width: int = 0
        self._height: int = 0
        self._frame_count: int = 0
        self._duration: float = 0.0
        self._opened: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> bool:
        """Open the video file.  Returns True on success."""
        if not os.path.isfile(self._path):
            print(f"[VideoLoader] File not found: {self._path}")
            return False

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            print(f"[VideoLoader] OpenCV could not open: {self._path}")
            return False

        self._fps         = float(self._cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self._width       = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height      = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration    = self._frame_count / max(self._fps, 1.0)
        self._opened      = True
        print(
            f"[VideoLoader] Opened  {os.path.basename(self._path)}  "
            f"{self._width}×{self._height}  {self._fps:.2f}fps  "
            f"{self._frame_count}f  {self._duration:.1f}s"
        )
        return True

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def get_metadata(self) -> dict:
        return {
            "width":       self._width,
            "height":      self._height,
            "fps":         round(self._fps, 3),
            "frame_count": self._frame_count,
            "duration":    round(self._duration, 2),
            "path":        self._path,
        }

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------
    def read_frame(self, index: int) -> Optional[cv2.Mat]:
        """Seek to *index* and return that single frame (BGR)."""
        if not self._opened or self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ok, frame = self._cap.read()
        return frame if ok else None

    def frame_generator(
        self,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Generator[Tuple[int, cv2.Mat], None, None]:
        """
        Yield (frame_index, frame_bgr) from *start* to *end* (exclusive).
        Seeks to *start* once; then reads sequentially — no seek on each frame.
        """
        if not self._opened or self._cap is None:
            return

        stop = end if end is not None else self._frame_count
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))

        for idx in range(start, stop):
            ok, frame = self._cap.read()
            if not ok:
                break
            yield idx, frame

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def is_open(self) -> bool:
        return self._opened
