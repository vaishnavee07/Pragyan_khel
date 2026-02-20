"""
PHASE 1-2-3 INTEGRATION EXAMPLE
Demonstrates how to wire Detection → Tracking → Selection pipeline
"""
import cv2
import numpy as np
from typing import Optional

# Import Phase 1: Detection Engine
from core.detection_engine import DetectionEngine

# Import Phase 2: Tracking Engine
from core.tracking_engine import TrackingEngine

# Import Phase 3: Selection Engine
from core.selection_engine import SelectionEngine


class VisionPipeline:
    """
    Complete vision pipeline integrating:
    - Phase 1: Modular Detection
    - Phase 2: ByteTrack Tracking
    - Phase 3: Tap-to-Select Selection
    """
    
    def __init__(self):
        # Phase 1: Detection
        self.detector = DetectionEngine({
            'model_type': 'yolo',  # or 'rtdetr' or 'face'
            'confidence_threshold': 0.4,
            'device': 'cpu'
        })
        
        # Phase 2: Tracking
        self.tracker = TrackingEngine({
            'track_thresh': 0.5,
            'track_buffer': 30,
            'match_thresh': 0.8,
            'fps': 30
        })
        
        # Phase 3: Selection
        self.selector = SelectionEngine({
            'timeout': 1.0,
            'auto_reset': True
        })
        
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize all pipeline components"""
        print("\n" + "="*60)
        print("INITIALIZING VISION PIPELINE")
        print("="*60)
        
        # Initialize detector
        if not self.detector.load_model():
            print("✗ Failed to load detection model")
            return False
        
        # Initialize tracker
        if not self.tracker.initialize():
            print("✗ Failed to initialize tracker")
            return False
        
        # Selector needs no initialization
        
        self.is_initialized = True
        print("✓ Vision pipeline ready")
        print("="*60 + "\n")
        return True
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process single frame through complete pipeline
        
        Args:
            frame: BGR image
            
        Returns:
            dict with detections, tracked_objects, and focus info
        """
        if not self.is_initialized:
            return {
                'detections': [],
                'tracked_objects': [],
                'focus_object': None,
                'error': 'Pipeline not initialized'
            }
        
        frame_height, frame_width = frame.shape[:2]
        
        # PHASE 1: DETECTION
        detections = self.detector.detect(frame)
        
        # PHASE 2: TRACKING
        tracked_objects = self.tracker.update(detections, (frame_height, frame_width))
        
        # PHASE 3: SELECTION - Get active focus object
        focus_object = self.selector.get_active_focus_object(tracked_objects)
        
        return {
            'detections': detections,
            'tracked_objects': tracked_objects,
            'focus_object': focus_object,
            'focus_status': self.selector.get_focus_status()
        }
    
    def handle_click(self, x: int, y: int, tracked_objects: list) -> Optional[int]:
        """
        Handle click event from frontend
        
        Args:
            x, y: Click coordinates
            tracked_objects: Current tracked objects
            
        Returns:
            track_id of selected object
        """
        return self.selector.handle_click(x, y, tracked_objects)
    
    def get_stats(self) -> dict:
        """Get pipeline statistics"""
        return {
            'detection': self.detector.get_stats(),
            'tracking': self.tracker.get_stats(),
            'selection': self.selector.get_stats()
        }
    
    def cleanup(self):
        """Cleanup all resources"""
        self.detector.cleanup()
        self.tracker.cleanup()
        self.selector.cleanup()


# ============================================================
# EXAMPLE 1: Basic Usage
# ============================================================
def example_basic_usage():
    """Example: Basic pipeline usage"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Create pipeline
    pipeline = VisionPipeline()
    
    # Initialize
    if not pipeline.initialize():
        print("Failed to initialize pipeline")
        return
    
    # Simulate frame processing
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    result = pipeline.process_frame(frame)
    
    print(f"Detections: {len(result['detections'])}")
    print(f"Tracked Objects: {len(result['tracked_objects'])}")
    print(f"Focus Object: {result['focus_object']}")
    
    # Cleanup
    pipeline.cleanup()
    print("✓ Example complete\n")


# ============================================================
# EXAMPLE 2: With Click Handling
# ============================================================
def example_with_click():
    """Example: Pipeline with click handling"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Click Handling")
    print("="*60)
    
    pipeline = VisionPipeline()
    pipeline.initialize()
    
    # Simulate frames
    for i in range(3):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame)
        
        print(f"\nFrame {i+1}:")
        print(f"  Tracked objects: {len(result['tracked_objects'])}")
        
        # Simulate click on first object
        if result['tracked_objects'] and i == 1:
            obj = result['tracked_objects'][0]
            bbox = obj['bbox']
            click_x = (bbox[0] + bbox[2]) // 2
            click_y = (bbox[1] + bbox[3]) // 2
            
            track_id = pipeline.handle_click(click_x, click_y, result['tracked_objects'])
            print(f"  Clicked object: track_id={track_id}")
        
        # Check focus
        if result['focus_object']:
            print(f"  Current focus: {result['focus_object']}")
    
    pipeline.cleanup()
    print("✓ Example complete\n")


# ============================================================
# EXAMPLE 3: WebSocket Integration Pattern
# ============================================================
def example_websocket_integration():
    """Example: How to integrate with WebSocket handler"""
    print("\n" + "="*60)
    print("EXAMPLE 3: WebSocket Integration Pattern")
    print("="*60)
    
    print("""
# In your WebSocket handler:

class WebSocketHandler:
    def __init__(self):
        self.pipeline = VisionPipeline()
        self.pipeline.initialize()
    
    async def handle_connection(self, websocket):
        while True:
            # Receive message
            message = await websocket.receive_json()
            
            if message['type'] == 'frame':
                # Process frame
                frame = decode_frame(message['frame'])
                result = self.pipeline.process_frame(frame)
                
                # Send back results
                await websocket.send_json({
                    'type': 'detections',
                    'tracked_objects': result['tracked_objects'],
                    'focus_id': result['focus_status']['focus_id']
                })
            
            elif message['type'] == 'click':
                # Handle click
                x, y = message['x'], message['y']
                
                # Get current tracked objects (from last frame)
                tracked_objects = self.last_tracked_objects
                
                track_id = self.pipeline.handle_click(x, y, tracked_objects)
                
                await websocket.send_json({
                    'type': 'focus_changed',
                    'focus_id': track_id
                })
    """)
    
    print("✓ Example complete\n")


# ============================================================
# EXAMPLE 4: Attendance Mode Integration
# ============================================================
def example_attendance_mode():
    """Example: Using attendance mode with face detection"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Attendance Mode")
    print("="*60)
    
    from modes.attendance_mode import AttendanceMode
    
    # Create attendance mode
    attendance = AttendanceMode({
        'recognition_threshold': 0.6
    })
    
    if not attendance.initialize():
        print("Failed to initialize attendance mode")
        return
    
    # Process frames
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = attendance.process_frame(frame)
    
    print(f"Faces detected: {result.metrics.get('faces_detected', 0)}")
    print(f"Recognized: {result.metrics.get('recognized_count', 0)}")
    print(f"Unknown: {result.metrics.get('unknown_count', 0)}")
    
    # Get attendance report
    report = attendance.get_attendance_report()
    print(f"Total present: {report['total_present']}")
    
    attendance.cleanup()
    print("✓ Example complete\n")


# ============================================================
# EXAMPLE 5: Statistics and Monitoring
# ============================================================
def example_statistics():
    """Example: Getting pipeline statistics"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Statistics and Monitoring")
    print("="*60)
    
    pipeline = VisionPipeline()
    pipeline.initialize()
    
    # Process some frames
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame)
    
    # Get statistics
    stats = pipeline.get_stats()
    
    print("\nDetection Stats:")
    print(f"  Model: {stats['detection']['model_type']}")
    print(f"  Frames: {stats['detection']['frame_count']}")
    print(f"  Avg Inference: {stats['detection']['avg_inference_ms']} ms")
    
    print("\nTracking Stats:")
    print(f"  Active Tracks: {stats['tracking']['active_tracks']}")
    print(f"  Total Tracks: {stats['tracking']['total_tracks']}")
    print(f"  Rolling FPS: {stats['tracking']['rolling_fps']}")
    
    print("\nSelection Stats:")
    print(f"  Has Focus: {stats['selection']['has_focus']}")
    print(f"  Focus ID: {stats['selection']['active_focus_id']}")
    print(f"  Total Focus Events: {stats['selection']['total_focus_events']}")
    
    pipeline.cleanup()
    print("✓ Example complete\n")


# ============================================================
# RUN ALL EXAMPLES
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PHASE 1-2-3 INTEGRATION EXAMPLES")
    print("="*60)
    
    try:
        example_basic_usage()
        example_with_click()
        example_websocket_integration()
        example_attendance_mode()
        example_statistics()
        
        print("\n" + "="*60)
        print("✓ ALL EXAMPLES COMPLETE")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
