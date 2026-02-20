"""
Camera capture and management
"""
import cv2
import time

class Camera:
    """Webcam capture with FPS tracking"""
    
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        # Use DirectShow backend on Windows for better compatibility
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update = time.time()
        
        if self.cap.isOpened():
            print(f"✓ Camera {camera_index} opened: {self.get_resolution()}")
        else:
            print(f"✗ Failed to open camera {camera_index}")
    
    def is_opened(self):
        """Check if camera is opened"""
        return self.cap.isOpened()
    
    def read(self):
        """Read a frame from camera"""
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
    
    def get_fps(self):
        """Get current FPS"""
        return self.fps
    
    def get_resolution(self):
        """Get camera resolution"""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return f"{width}x{height}"
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            print("✓ Camera released")
