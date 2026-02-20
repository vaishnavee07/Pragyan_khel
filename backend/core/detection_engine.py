"""
Detection Engine - PHASE 1
Modular detection engine that loads and manages detection models
Provides clean separation between detection and business logic
"""
from typing import List, Dict, Any, Optional
import time


class DetectionEngine:
    """
    Core detection engine that manages detection models
    Independent from attendance, tracking, or any business logic
    """
    
    def __init__(self, detector_config: Dict[str, Any] = None):
        """
        Initialize detection engine
        
        Args:
            detector_config: Configuration for detector
                - model_type: 'yolo', 'rtdetr', 'face'
                - confidence_threshold: float (default 0.4)
                - device: 'cpu' or 'cuda'
        """
        self.config = detector_config or {}
        self.model_type = self.config.get('model_type', 'yolo')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.4)
        self.device = self.config.get('device', 'cpu')
        
        self.model = None
        self.is_initialized = False
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0
        
        print(f"✓ DetectionEngine created")
        print(f"  Model Type: {self.model_type}")
        print(f"  Confidence: {self.confidence_threshold}")
        print(f"  Device: {self.device}")
    
    def load_model(self, model_type: str = None) -> bool:
        """
        Load detection model based on type
        
        Args:
            model_type: 'yolo', 'rtdetr', 'face'
            
        Returns:
            bool: Success status
        """
        if model_type:
            self.model_type = model_type
        
        try:
            if self.model_type == 'yolo':
                return self._load_yolo()
            elif self.model_type == 'rtdetr':
                return self._load_rtdetr()
            elif self.model_type == 'face':
                return self._load_face_detector()
            else:
                print(f"✗ Unknown model type: {self.model_type}")
                return False
                
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            return False
    
    def _load_yolo(self) -> bool:
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO
            print("⏳ Loading YOLOv8 model...")
            self.model = YOLO('yolov8n.pt')
            self.is_initialized = True
            print("✓ YOLOv8 loaded successfully")
            return True
        except ImportError:
            print("✗ YOLOv8 not installed. Run: pip install ultralytics")
            return False
    
    def _load_rtdetr(self) -> bool:
        """Load RT-DETR model"""
        try:
            from ultralytics import RTDETR
            print("⏳ Loading RT-DETR model...")
            self.model = RTDETR('rtdetr-l.pt')
            self.is_initialized = True
            print("✓ RT-DETR loaded successfully")
            return True
        except ImportError:
            print("✗ RT-DETR not installed. Run: pip install ultralytics")
            return False
    
    def _load_face_detector(self) -> bool:
        """Load face detection model"""
        try:
            import cv2
            print("⏳ Loading Haar Cascade face detector...")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.model = cv2.CascadeClassifier(cascade_path)
            self.is_initialized = True
            print("✓ Face detector loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Face detector loading failed: {e}")
            return False
    
    def detect(self, frame) -> List[Dict[str, Any]]:
        """
        Run detection on frame
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            List of detections with structure:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "class_id": int,
                    "class_name": str
                }
            ]
        """
        if not self.is_initialized or self.model is None:
            print("✗ Model not initialized")
            return []
        
        start_time = time.time()
        
        try:
            if self.model_type in ['yolo', 'rtdetr']:
                detections = self._detect_objects(frame)
            elif self.model_type == 'face':
                detections = self._detect_faces(frame)
            else:
                detections = []
            
            inference_time = (time.time() - start_time) * 1000
            
            self.frame_count += 1
            self.total_inference_time += inference_time
            
            return detections
            
        except Exception as e:
            print(f"✗ Detection failed: {e}")
            return []
    
    def _detect_objects(self, frame) -> List[Dict[str, Any]]:
        """Detect objects using YOLO or RT-DETR"""
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = r.names[cls]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': round(conf, 3),
                    'class_id': cls,
                    'class_name': class_name
                })
        
        return detections
    
    def _detect_faces(self, frame) -> List[Dict[str, Any]]:
        """Detect faces using Haar Cascade"""
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detections = []
        for i, (x, y, w, h) in enumerate(faces):
            detections.append({
                'bbox': [int(x), int(y), int(x + w), int(y + h)],
                'confidence': 0.99,  # Haar cascade doesn't provide confidence
                'class_id': 0,
                'class_name': 'face'
            })
        
        return detections
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        avg_inference = (
            self.total_inference_time / self.frame_count
            if self.frame_count > 0 else 0
        )
        
        return {
            'model_type': self.model_type,
            'frame_count': self.frame_count,
            'avg_inference_ms': round(avg_inference, 2),
            'confidence_threshold': self.confidence_threshold
        }
    
    def cleanup(self):
        """Release detector resources"""
        self.model = None
        self.is_initialized = False
        print("✓ DetectionEngine cleanup complete")
