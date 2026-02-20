"""
RT-DETR Detection Module - Real-Time Detection Transformer
Replaces YOLO while maintaining tracking compatibility
"""
import cv2
import numpy as np
import time
import psutil
from typing import List, Dict, Any, Optional
from core.base_module import BaseAIModule, InferenceResult
from modules.tracking_adapter import TrackingAdapter

class RTDETRDetectionModule(BaseAIModule):
    """Object detection using RT-DETR pretrained model"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = None
        self.confidence_threshold = config.get('confidence_threshold', 0.4) if config else 0.4
        self.iou_threshold = config.get('iou_threshold', 0.6) if config else 0.6
        self.device = config.get('device', 'cpu')  # 'cpu' or 'cuda'
        self.detect_only_person = config.get('detect_only_person', False)
        self.person_class_id = 0  # COCO person class
        
        # Tracking adapter integration
        self.tracker = TrackingAdapter()
        
        # Performance tracking
        self.fps_history = []
        self.fps_window = 30  # Rolling window for FPS calculation
        self.last_frame_time = time.time()
        self.inference_times = []
        
        # GPU monitoring
        self.gpu_available = False
        
        print(f"✓ RTDETRDetectionModule initialized")
        print(f"  Model: RT-DETR")
        print(f"  Confidence: {self.confidence_threshold}")
        print(f"  IOU: {self.iou_threshold}")
        print(f"  Device: {self.device}")
        print(f"  Person-only: {self.detect_only_person}")
        
    def initialize(self) -> bool:
        """Initialize RT-DETR model"""
        try:
            from ultralytics import RTDETR
            import torch
            
            # Check GPU availability
            if torch.cuda.is_available() and self.device == 'cuda':
                self.gpu_available = True
                print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                print("⚠ Using CPU (GPU not available or not requested)")
            
            print("⏳ Loading RT-DETR model...")
            # RT-DETR-L model - good balance of speed/accuracy
            self.model = RTDETR('rtdetr-l.pt')
            
            # Move to device
            if self.gpu_available:
                self.model.to('cuda')
            
            # Initialize tracking
            self.tracker.initialize_tracker()
            
            self.is_initialized = True
            print("✓ RT-DETR model loaded successfully")
            return True
            
        except ImportError as e:
            print(f"✗ RT-DETR not installed: {e}")
            print("  Run: pip install ultralytics  \u2014 running in demo mode")
            self.model = "demo"
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"\u2717 Model initialization failed: {e}  \u2014 running in demo mode")
            self.model = "demo"
            self.is_initialized = True
            return True
    
    def process_frame(self, frame) -> InferenceResult:
        """Process frame with RT-DETR"""
        start_time = time.time()
        
        # Run detection
        detections = self._rtdetr_detection(frame)
        
        # Filter to person only if enabled
        if self.detect_only_person:
            detections = [d for d in detections if d.get('class_id') == self.person_class_id]
        
        # Add tracking IDs
        if detections and frame is not None:
            detections = self.tracker.update_tracks(detections, frame.shape[:2])
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Track inference times for statistics
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        # Update FPS tracking
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        instant_fps = 1.0 / frame_time if frame_time > 0 else 0
        
        self.fps_history.append(instant_fps)
        if len(self.fps_history) > self.fps_window:
            self.fps_history.pop(0)
        
        rolling_fps = sum(self.fps_history) / len(self.fps_history)
        
        # Update counters
        self.frame_count += 1
        self.total_inference_time += inference_time
        
        # Performance logging (every 30 frames)
        if self.frame_count % 30 == 0:
            self._log_performance(inference_time, rolling_fps)
        
        # Alert logic based on object count
        alert_level = "normal"
        if len(detections) >= 5:
            alert_level = "critical"
        elif len(detections) >= 3:
            alert_level = "warning"
        
        # Gather metrics
        metrics = {
            'object_count': len(detections),
            'avg_inference': round(self.get_average_inference_time(), 2),
            'rolling_fps': round(rolling_fps, 1),
            'gpu_memory': self._get_gpu_memory() if self.gpu_available else 0,
            'cpu_percent': round(psutil.cpu_percent(interval=0), 1)
        }
        
        return self._create_result(detections, metrics, alert_level, inference_time)
    
    def _rtdetr_detection(self, frame) -> List[Dict]:
        """Run RT-DETR inference (or demo mode if ultralytics not installed)."""
        if frame is None or self.model is None:
            return []

        if self.model == "demo":
            return []

        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )
            
            detections = []
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = r.names[cls]
                    
                    detections.append({
                        'class': class_name,
                        'class_id': cls,
                        'confidence': conf,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
            
            return detections
            
        except Exception as e:
            print(f"✗ RT-DETR inference error: {e}")
            return []
    
    def get_detections_for_tracker(self, detections: List[Dict]) -> np.ndarray:
        """
        Convert detections to ByteTrack-compatible format
        Output: np.array([[x1, y1, x2, y2, confidence], ...])
        """
        if not detections:
            return np.empty((0, 5))
        
        tracker_format = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            tracker_format.append([x1, y1, x2, y2, conf])
        return np.array(tracker_format, dtype=np.float32)

    def _log_performance(self, current_inference: float, current_fps: float):
        """Log performance metrics"""
        avg_inference = np.mean(self.inference_times) if self.inference_times else 0
        min_inference = min(self.inference_times) if self.inference_times else 0
        max_inference = max(self.inference_times) if self.inference_times else 0

        print(f"[RT-DETR Performance]")
        print(f"  Model: RT-DETR")
        print(f"  Inference: {current_inference:.1f} ms (avg: {avg_inference:.1f}, min: {min_inference:.1f}, max: {max_inference:.1f})")
        print(f"  FPS: {current_fps:.1f}")
        if self.gpu_available:
            print(f"  GPU Memory: {self._get_gpu_memory():.1f} MB")
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory usage in MB"""
        try:
            import torch
            if torch.cuda.is_available():
                return round(torch.cuda.memory_allocated() / 1024**2, 1)
        except:
            pass
        return 0.0
    
    def cleanup(self):
        """Release resources"""
        if self.model is not None:
            del self.model
            self.model = None
        
        # Clear GPU cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        # Cleanup tracker
        self.tracker.cleanup()
        
        self.fps_history.clear()
        print("✓ RT-DETR module cleaned up")
