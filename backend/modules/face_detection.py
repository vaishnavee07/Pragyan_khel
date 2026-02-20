"""
Face Detection Module - PHASE 1
Dedicated face detection for attendance mode
Uses Haar Cascade or dlib for face detection
"""
import cv2
import numpy as np
import time
from typing import List, Dict, Any
from core.base_module import BaseAIModule, InferenceResult


class FaceDetectionModule(BaseAIModule):
    """Face detection module for attendance system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None
        self.scale_factor = config.get('scale_factor', 1.1) if config else 1.1
        self.min_neighbors = config.get('min_neighbors', 5) if config else 5
        self.min_face_size = config.get('min_face_size', 30) if config else 30
        
        print(f"✓ FaceDetectionModule initialized")
        print(f"  Scale Factor: {self.scale_factor}")
        print(f"  Min Neighbors: {self.min_neighbors}")
    
    def initialize(self) -> bool:
        """Initialize face detector"""
        try:
            # Load Haar Cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                print("✗ Failed to load Haar Cascade")
                return False
            
            self.is_initialized = True
            print("✓ Face detector loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Face detector initialization failed: {e}")
            return False
    
    def process_frame(self, frame) -> InferenceResult:
        """
        Detect faces in frame
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            InferenceResult with face detections
        """
        start_time = time.time()
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        # Convert to standard detection format
        detections = []
        for i, (x, y, w, h) in enumerate(faces):
            detections.append({
                'bbox': [int(x), int(y), int(x + w), int(y + h)],
                'confidence': 0.95,  # Haar cascade doesn't provide confidence
                'class_id': 0,
                'class_name': 'face',
                'face_id': i  # Face identifier for tracking
            })
        
        inference_time = (time.time() - start_time) * 1000
        
        self.frame_count += 1
        self.total_inference_time += inference_time
        
        # Metrics
        metrics = {
            'face_count': len(detections),
            'avg_inference': round(self.get_average_inference_time(), 2)
        }
        
        # Alert level based on face count
        alert_level = "normal"
        if len(detections) >= 3:
            alert_level = "warning"
        elif len(detections) >= 5:
            alert_level = "critical"
        
        return self._create_result(detections, metrics, alert_level, inference_time)
    
    def extract_face_embedding(self, frame, bbox: List[int]) -> np.ndarray:
        """
        Extract face embedding for recognition
        
        Args:
            frame: BGR image
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Face embedding vector (placeholder for now)
        """
        x1, y1, x2, y2 = bbox
        face_crop = frame[y1:y2, x1:x2]
        
        # Placeholder: In production, use FaceNet or ArcFace
        # For now, return simple histogram features
        if face_crop.size == 0:
            return np.zeros(128)
        
        # Resize to standard size
        face_resized = cv2.resize(face_crop, (128, 128))
        
        # Simple feature extraction (replace with deep learning model)
        gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        features = cv2.calcHist([gray_face], [0], None, [128], [0, 256]).flatten()
        
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features
    
    def cleanup(self):
        """Release resources"""
        self.face_cascade = None
        self.is_initialized = False
        print("✓ FaceDetectionModule cleanup complete")
