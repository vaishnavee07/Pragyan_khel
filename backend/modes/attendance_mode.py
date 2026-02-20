"""
Attendance Mode - PHASE 1
Business logic for attendance tracking
Exclusively uses face detection module
"""
import time
from typing import Dict, Any, List
from core.base_module import BaseAIModule, InferenceResult
from modules.face_detection import FaceDetectionModule


class AttendanceMode(BaseAIModule):
    """
    Attendance tracking mode
    Uses face detection module exclusively
    Manages attendance records and recognition logic
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_detector = None
        self.attendance_records = {}
        self.recognition_threshold = config.get('recognition_threshold', 0.6) if config else 0.6
        self.known_faces = {}  # face_id -> person_name mapping
        
        print(f"✓ AttendanceMode initialized")
        print(f"  Recognition Threshold: {self.recognition_threshold}")
    
    def initialize(self) -> bool:
        """Initialize attendance mode with face detector"""
        try:
            # Create dedicated face detection module
            self.face_detector = FaceDetectionModule({
                'scale_factor': 1.1,
                'min_neighbors': 5,
                'min_face_size': 30
            })
            
            if not self.face_detector.initialize():
                print("✗ Failed to initialize face detector")
                return False
            
            self.is_initialized = True
            print("✓ AttendanceMode ready")
            return True
            
        except Exception as e:
            print(f"✗ AttendanceMode initialization failed: {e}")
            return False
    
    def process_frame(self, frame) -> InferenceResult:
        """
        Process frame for attendance tracking
        
        Args:
            frame: BGR image
            
        Returns:
            InferenceResult with attendance data
        """
        if not self.is_initialized or not self.face_detector:
            return self._create_empty_result()
        
        start_time = time.time()
        
        # Detect faces using dedicated face detector
        face_result = self.face_detector.process_frame(frame)
        
        # Process each detected face for recognition
        attendance_data = []
        for detection in face_result.detections:
            bbox = detection['bbox']
            
            # Extract face embedding
            embedding = self.face_detector.extract_face_embedding(frame, bbox)
            
            # Attempt recognition
            person_name, confidence = self._recognize_face(embedding)
            
            # Add to attendance if recognized
            if person_name and confidence >= self.recognition_threshold:
                self._mark_attendance(person_name)
                detection['person_name'] = person_name
                detection['recognition_confidence'] = confidence
                detection['attendance_status'] = 'present'
            else:
                detection['person_name'] = 'Unknown'
                detection['recognition_confidence'] = confidence
                detection['attendance_status'] = 'unknown'
            
            attendance_data.append(detection)
        
        inference_time = (time.time() - start_time) * 1000
        
        self.frame_count += 1
        self.total_inference_time += inference_time
        
        # Metrics
        metrics = {
            'faces_detected': len(attendance_data),
            'recognized_count': sum(1 for d in attendance_data if d['attendance_status'] == 'present'),
            'unknown_count': sum(1 for d in attendance_data if d['attendance_status'] == 'unknown'),
            'total_attendance': len(self.attendance_records),
            'avg_inference': round(self.get_average_inference_time(), 2)
        }
        
        # Alert level
        alert_level = "normal"
        unknown_count = metrics['unknown_count']
        if unknown_count >= 3:
            alert_level = "warning"
        elif unknown_count >= 5:
            alert_level = "critical"
        
        return self._create_result(attendance_data, metrics, alert_level, inference_time)
    
    def _recognize_face(self, embedding) -> tuple:
        """
        Recognize face from embedding
        
        Args:
            embedding: Face feature vector
            
        Returns:
            (person_name, confidence)
        """
        if not self.known_faces:
            return None, 0.0
        
        # Placeholder: In production, use cosine similarity with known embeddings
        # For now, return mock recognition
        import numpy as np
        
        best_match = None
        best_similarity = 0.0
        
        for person_id, known_embedding in self.known_faces.items():
            # Cosine similarity
            similarity = np.dot(embedding, known_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(known_embedding) + 1e-7
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_id
        
        return best_match, float(best_similarity)
    
    def _mark_attendance(self, person_name: str):
        """Mark person as present"""
        current_time = time.time()
        
        if person_name not in self.attendance_records:
            self.attendance_records[person_name] = {
                'first_seen': current_time,
                'last_seen': current_time,
                'total_frames': 1
            }
            print(f"✓ Attendance marked: {person_name}")
        else:
            self.attendance_records[person_name]['last_seen'] = current_time
            self.attendance_records[person_name]['total_frames'] += 1
    
    def register_person(self, person_name: str, face_embedding):
        """
        Register a new person for recognition
        
        Args:
            person_name: Person identifier
            face_embedding: Face feature vector
        """
        self.known_faces[person_name] = face_embedding
        print(f"✓ Registered person: {person_name}")
    
    def get_attendance_report(self) -> Dict[str, Any]:
        """Get attendance report"""
        return {
            'total_present': len(self.attendance_records),
            'records': self.attendance_records,
            'timestamp': time.time()
        }
    
    def _create_empty_result(self) -> InferenceResult:
        """Create empty result when detector not initialized"""
        return self._create_result([], {'error': 'Not initialized'}, 'normal', 0)
    
    def cleanup(self):
        """Release resources"""
        if self.face_detector:
            self.face_detector.cleanup()
            self.face_detector = None
        
        self.attendance_records = {}
        self.is_initialized = False
        print("✓ AttendanceMode cleanup complete")
