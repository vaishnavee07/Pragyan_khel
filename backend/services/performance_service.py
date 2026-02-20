"""
Performance Service - Monitor system performance
"""
import psutil
import time
from collections import deque
from typing import Dict, Any

class PerformanceService:
    """Monitor and manage system performance"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.fps_history = deque(maxlen=window_size)
        self.inference_history = deque(maxlen=window_size)
        self.cpu_history = deque(maxlen=window_size)
        self.memory_history = deque(maxlen=window_size)
        self.last_update = time.time()
        self.frame_skip = 1
        
    def update(self, fps: float, inference_time: float):
        """Update performance metrics"""
        self.fps_history.append(fps)
        self.inference_history.append(inference_time)
        
        # Update system metrics every second
        current_time = time.time()
        if current_time - self.last_update >= 1.0:
            self.cpu_history.append(psutil.cpu_percent())
            self.memory_history.append(psutil.virtual_memory().percent)
            self.last_update = current_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'fps': {
                'current': self.fps_history[-1] if self.fps_history else 0,
                'average': sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0,
                'min': min(self.fps_history) if self.fps_history else 0,
                'max': max(self.fps_history) if self.fps_history else 0
            },
            'inference': {
                'current': self.inference_history[-1] if self.inference_history else 0,
                'average': sum(self.inference_history) / len(self.inference_history) if self.inference_history else 0
            },
            'system': {
                'cpu': self.cpu_history[-1] if self.cpu_history else 0,
                'memory': self.memory_history[-1] if self.memory_history else 0
            },
            'frame_skip': self.frame_skip
        }
    
    def should_skip_frame(self, frame_count: int) -> bool:
        """Determine if frame should be skipped based on performance"""
        return frame_count % self.frame_skip != 0
    
    def adjust_frame_skip(self):
        """Dynamically adjust frame skip based on performance"""
        if not self.inference_history:
            return
        
        avg_inference = sum(self.inference_history) / len(self.inference_history)
        
        # If inference is slow, increase frame skip
        if avg_inference > 100:  # >100ms
            self.frame_skip = min(self.frame_skip + 1, 5)
        elif avg_inference < 50 and self.frame_skip > 1:  # <50ms
            self.frame_skip = max(self.frame_skip - 1, 1)
    
    def reset(self):
        """Reset all metrics"""
        self.fps_history.clear()
        self.inference_history.clear()
        self.cpu_history.clear()
        self.memory_history.clear()
        self.frame_skip = 1
