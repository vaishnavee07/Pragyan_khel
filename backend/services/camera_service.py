"""
Camera Service - Enhanced camera management with performance tracking
"""
import cv2
import time
from typing import Optional

class CameraService:
    """Enhanced camera capture with FPS tracking and performance optimization"""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update = time.time()
        self.is_active = False
        
    def open(self) -> bool:
        """Open camera connection"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                self.is_active = True
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"✓ Camera {self.camera_index} opened: {width}x{height}")
                return True
            return False
        except Exception as e:
            print(f"✗ Failed to open camera: {e}")
            return False
    
    def read(self):
        """Read frame from camera"""
        if not self.cap or not self.is_active:
            return None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            self._update_fps()
            return frame
        return None
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        elapsed = current_time - self.last_fps_update
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps
    
    def get_resolution(self) -> tuple:
        """Get camera resolution"""
        if not self.cap:
            return (0, 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def is_opened(self) -> bool:
        """Check if camera is opened"""
        return self.cap is not None and self.cap.isOpened() and self.is_active
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
            self.is_active = False
            self.fps = 0
            self.frame_count = 0
            print("✓ Camera released")
