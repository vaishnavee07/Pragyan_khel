"""
Object Detection Module - YOLOv8 Integration
"""
import cv2
import numpy as np
import time
import random
from typing import List, Dict, Any
from core.base_module import BaseAIModule, InferenceResult

class ObjectDetectionModule(BaseAIModule):
    """Object detection using YOLOv8 pretrained model"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = None
        self.confidence_threshold = config.get('confidence_threshold', 0.4) if config else 0.4
        self.track_history = {}
        self.next_track_id = 1
        
        print(f"✓ ObjectDetectionModule initialized")
        print(f"  Confidence: {self.confidence_threshold}")
        
    def initialize(self) -> bool:
        """Initialize YOLOv8 model"""
        try:
            from ultralytics import YOLO
            
            print("⏳ Loading YOLOv8 model...")
            self.model = YOLO('yolov8n.pt')  # Nano model - fastest
            self.is_initialized = True
            
            print("✓ YOLOv8 model loaded successfully")
            return True
            
        except ImportError:
            print("⚠ YOLOv8 not installed - using demo mode")
            print("  Run: pip install ultralytics")
            self.model = "demo"
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"✗ Model initialization failed: {e}")
            self.model = "demo"
            self.is_initialized = True
            return True
    
    def process_frame(self, frame) -> InferenceResult:
        """Process frame with YOLOv8"""
        start_time = time.time()
        
        if self.model == "demo":
            detections = self._demo_detection(frame)
        else:
            detections = self._yolo_detection(frame)
        
        inference_time = (time.time() - start_time) * 1000
        
        self.frame_count += 1
        self.total_inference_time += inference_time
        
        # Alert logic
        alert_level = "normal"
        if len(detections) >= 5:
            alert_level = "critical"
        elif len(detections) >= 3:
            alert_level = "warning"
        
        metrics = {
            'object_count': len(detections),
            'avg_inference': round(self.get_average_inference_time(), 2)
        }
        
        return self._create_result(detections, metrics, alert_level, inference_time)
    
    def _yolo_detection(self, frame) -> List[Dict]:
        """Run YOLOv8 inference"""
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = r.names[cls]
                
                track_id = self._assign_track_id(x1, y1, x2, y2)
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'track_id': track_id
                })
        
        return detections
    
    def _demo_detection(self, frame) -> List[Dict]:
        """Demo mode detection"""
        if frame is None:
            return []
        
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        detections = []
        if brightness > 50:
            for i in range(random.randint(1, 2)):
                obj_class = random.choice(['person', 'laptop', 'cell phone'])
                conf = random.uniform(0.5, 0.9)
                x1 = random.randint(50, w // 2)
                y1 = random.randint(50, h // 2)
                x2 = min(x1 + 120, w - 10)
                y2 = min(y1 + 120, h - 10)
                
                track_id = self._assign_track_id(x1, y1, x2, y2)
                
                detections.append({
                    'class': obj_class,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'track_id': track_id
                })
        
        return detections
    
    def _assign_track_id(self, x1, y1, x2, y2) -> int:
        """Simple centroid tracking"""
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Find closest existing track
        min_dist = 100
        matched_id = None
        
        for track_id, (prev_cx, prev_cy) in self.track_history.items():
            dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            if dist < min_dist:
                min_dist = dist
                matched_id = track_id
        
        if matched_id:
            self.track_history[matched_id] = (cx, cy)
            return matched_id
        else:
            new_id = self.next_track_id
            self.next_track_id += 1
            self.track_history[new_id] = (cx, cy)
            return new_id
    
    def cleanup(self):
        """Release resources"""
        self.model = None
        self.track_history.clear()