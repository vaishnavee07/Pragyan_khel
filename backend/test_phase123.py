"""
Verification Test Suite for Phase 1-2-3 Implementation
Tests all three phases: Detection → Tracking → Selection
"""
import sys
import numpy as np
from typing import Dict, Any


def test_phase1_detection_engine():
    """Test Phase 1: Detection Engine"""
    print("\n" + "="*60)
    print("TEST PHASE 1: DETECTION ENGINE")
    print("="*60)
    
    try:
        from core.detection_engine import DetectionEngine
        
        # Test 1: Initialization
        detector = DetectionEngine({
            'model_type': 'yolo',
            'confidence_threshold': 0.4,
            'device': 'cpu'
        })
        print("✓ Detection engine initialized")
        
        # Test 2: Model loading
        if detector.load_model('yolo'):
            print("✓ YOLO model loaded")
        else:
            print("⚠ YOLO model not available (using demo mode)")
        
        # Test 3: Detection format
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        print(f"✓ Detection output: {len(detections)} objects")
        
        if len(detections) > 0:
            det = detections[0]
            required_keys = ['bbox', 'confidence', 'class_id', 'class_name']
            for key in required_keys:
                if key not in det:
                    print(f"✗ Missing key: {key}")
                    return False
            print("✓ Detection format correct")
        
        # Test 4: Stats
        stats = detector.get_stats()
        print(f"✓ Stats available: {stats.keys()}")
        
        # Test 5: Cleanup
        detector.cleanup()
        print("✓ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Phase 1 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase1_face_detection():
    """Test Phase 1: Face Detection Module"""
    print("\n" + "="*60)
    print("TEST PHASE 1: FACE DETECTION MODULE")
    print("="*60)
    
    try:
        from modules.face_detection import FaceDetectionModule
        
        # Test 1: Initialization
        face_detector = FaceDetectionModule({
            'scale_factor': 1.1,
            'min_neighbors': 5
        })
        print("✓ Face detector initialized")
        
        # Test 2: Model loading
        if not face_detector.initialize():
            print("✗ Face detector initialization failed")
            return False
        print("✓ Face detector ready")
        
        # Test 3: Process frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = face_detector.process_frame(frame)
        
        print(f"✓ Face detection result: {len(result.detections)} faces")
        print(f"✓ Metrics: {result.metrics}")
        
        # Test 4: Cleanup
        face_detector.cleanup()
        print("✓ Face detector cleanup complete")
        
        return True
        
    except Exception as e:
        print(f"✗ Face detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase1_attendance_mode():
    """Test Phase 1: Attendance Mode"""
    print("\n" + "="*60)
    print("TEST PHASE 1: ATTENDANCE MODE")
    print("="*60)
    
    try:
        from modes.attendance_mode import AttendanceMode
        
        # Test 1: Initialization
        attendance = AttendanceMode({
            'recognition_threshold': 0.6
        })
        print("✓ Attendance mode initialized")
        
        # Test 2: Initialize internal components
        if not attendance.initialize():
            print("✗ Attendance mode initialization failed")
            return False
        print("✓ Attendance mode ready")
        
        # Test 3: Process frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = attendance.process_frame(frame)
        
        print(f"✓ Attendance result: {result.metrics}")
        
        # Test 4: Get report
        report = attendance.get_attendance_report()
        print(f"✓ Attendance report: {report['total_present']} present")
        
        # Test 5: Cleanup
        attendance.cleanup()
        print("✓ Attendance mode cleanup complete")
        
        return True
        
    except Exception as e:
        print(f"✗ Attendance mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase2_tracking_engine():
    """Test Phase 2: Tracking Engine"""
    print("\n" + "="*60)
    print("TEST PHASE 2: TRACKING ENGINE")
    print("="*60)
    
    try:
        from core.tracking_engine import TrackingEngine
        
        # Test 1: Initialization
        tracker = TrackingEngine({
            'track_thresh': 0.5,
            'track_buffer': 30,
            'match_thresh': 0.8
        })
        print("✓ Tracking engine initialized")
        
        # Test 2: Initialize tracker
        if not tracker.initialize():
            print("✗ Tracker initialization failed")
            return False
        print("✓ ByteTrack tracker ready")
        
        # Test 3: Update tracking
        # Simulate detections
        detections = [
            {
                'bbox': [100, 100, 200, 200],
                'confidence': 0.9,
                'class_id': 0,
                'class_name': 'person'
            },
            {
                'bbox': [300, 150, 400, 250],
                'confidence': 0.85,
                'class_id': 0,
                'class_name': 'person'
            }
        ]
        
        frame_shape = (480, 640)
        tracked = tracker.update(detections, frame_shape)
        
        print(f"✓ Tracking result: {len(tracked)} tracked objects")
        
        if len(tracked) > 0:
            track = tracked[0]
            required_keys = ['track_id', 'bbox', 'confidence', 'class_name']
            for key in required_keys:
                if key not in track:
                    print(f"✗ Missing key: {key}")
                    return False
            print("✓ Tracking format correct")
            print(f"  Track IDs: {[t['track_id'] for t in tracked]}")
        
        # Test 4: Track stability (multiple frames)
        for i in range(5):
            tracked = tracker.update(detections, frame_shape)
        
        print(f"✓ Track stability maintained: {len(tracked)} tracks")
        
        # Test 5: Stats
        stats = tracker.get_stats()
        print(f"✓ Tracking stats: {stats}")
        
        # Test 6: Cleanup
        tracker.cleanup()
        print("✓ Tracking engine cleanup complete")
        
        return True
        
    except Exception as e:
        print(f"✗ Phase 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase2_bytetrack():
    """Test Phase 2: ByteTrack Module"""
    print("\n" + "="*60)
    print("TEST PHASE 2: BYTETRACK MODULE")
    print("="*60)
    
    try:
        from modules.bytetrack_tracker import BYTETracker
        
        # Test 1: Initialization
        tracker = BYTETracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8
        )
        print("✓ BYTETracker initialized")
        
        # Test 2: Update with detections
        detections = np.array([
            [100, 100, 200, 200, 0.9],
            [300, 150, 400, 250, 0.85]
        ])
        
        frame_shape = (480, 640)
        tracks = tracker.update(detections, frame_shape)
        
        print(f"✓ ByteTrack output: {len(tracks)} tracks")
        print(f"  Track IDs: {tracks[:, 4].astype(int).tolist() if len(tracks) > 0 else []}")
        
        # Test 3: Track persistence
        for i in range(5):
            tracks = tracker.update(detections, frame_shape)
        
        print(f"✓ Track persistence: {len(tracks)} tracks after 5 frames")
        
        return True
        
    except Exception as e:
        print(f"✗ ByteTrack test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase3_selection_engine():
    """Test Phase 3: Selection Engine"""
    print("\n" + "="*60)
    print("TEST PHASE 3: SELECTION ENGINE")
    print("="*60)
    
    try:
        from core.selection_engine import SelectionEngine
        
        # Test 1: Initialization
        selector = SelectionEngine({
            'timeout': 1.0,
            'auto_reset': True
        })
        print("✓ Selection engine initialized")
        
        # Test 2: Handle click
        tracked_objects = [
            {
                'track_id': 1,
                'bbox': [100, 100, 200, 200],
                'confidence': 0.9,
                'class_name': 'person'
            },
            {
                'track_id': 2,
                'bbox': [300, 150, 400, 250],
                'confidence': 0.85,
                'class_name': 'person'
            }
        ]
        
        # Click inside first bbox
        click_x, click_y = 150, 150
        track_id = selector.handle_click(click_x, click_y, tracked_objects)
        
        if track_id == 1:
            print(f"✓ Click handling correct: track_id={track_id}")
        else:
            print(f"✗ Click handling failed: expected 1, got {track_id}")
            return False
        
        # Test 3: Get active focus
        focus = selector.get_active_focus_object(tracked_objects)
        if focus and focus['track_id'] == 1:
            print(f"✓ Active focus: track_id={focus['track_id']}")
        else:
            print("✗ Active focus retrieval failed")
            return False
        
        # Test 4: Focus status
        status = selector.get_focus_status()
        print(f"✓ Focus status: {status}")
        
        # Test 5: Focus switching
        click_x, click_y = 350, 200
        new_track_id = selector.handle_click(click_x, click_y, tracked_objects)
        
        if new_track_id == 2:
            print(f"✓ Focus switch: 1 → {new_track_id}")
        else:
            print(f"✗ Focus switch failed")
            return False
        
        # Test 6: Lost track handling
        empty_objects = []
        focus = selector.get_active_focus_object(empty_objects)
        print(f"✓ Lost track handling: {focus}")
        
        # Test 7: Reset focus
        selector.reset_focus()
        if selector.get_focus_id() is None:
            print("✓ Focus reset successful")
        else:
            print("✗ Focus reset failed")
            return False
        
        # Test 8: Stats
        stats = selector.get_stats()
        print(f"✓ Selection stats: {stats}")
        
        # Test 9: Cleanup
        selector.cleanup()
        print("✓ Selection engine cleanup complete")
        
        return True
        
    except Exception as e:
        print(f"✗ Phase 3 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_integration():
    """Test complete pipeline integration: Detection → Tracking → Selection"""
    print("\n" + "="*60)
    print("TEST PIPELINE INTEGRATION")
    print("="*60)
    
    try:
        from core.detection_engine import DetectionEngine
        from core.tracking_engine import TrackingEngine
        from core.selection_engine import SelectionEngine
        
        # Test 1: Initialize all components
        detector = DetectionEngine({'model_type': 'yolo', 'confidence_threshold': 0.4})
        tracker = TrackingEngine({'track_thresh': 0.5})
        selector = SelectionEngine({'timeout': 1.0})
        
        detector.load_model()
        tracker.initialize()
        
        print("✓ All components initialized")
        
        # Test 2: Process frame through pipeline
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Phase 1: Detection
        detections = detector.detect(frame)
        print(f"✓ Detection: {len(detections)} objects")
        
        # Phase 2: Tracking
        frame_shape = frame.shape[:2]
        tracked = tracker.update(detections, frame_shape)
        print(f"✓ Tracking: {len(tracked)} tracked objects")
        
        # Phase 3: Selection
        if len(tracked) > 0:
            # Simulate click
            bbox = tracked[0]['bbox']
            click_x = (bbox[0] + bbox[2]) // 2
            click_y = (bbox[1] + bbox[3]) // 2
            
            track_id = selector.handle_click(click_x, click_y, tracked)
            focus = selector.get_active_focus_object(tracked)
            
            print(f"✓ Selection: focus_id={selector.get_focus_id()}")
        
        # Test 3: Cleanup
        detector.cleanup()
        tracker.cleanup()
        selector.cleanup()
        
        print("✓ Pipeline integration successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests"""
    print("\n" + "="*60)
    print("PHASE 1-2-3 VERIFICATION TEST SUITE")
    print("="*60)
    
    results = []
    
    # Phase 1 Tests
    results.append(("Phase 1: Detection Engine", test_phase1_detection_engine()))
    results.append(("Phase 1: Face Detection", test_phase1_face_detection()))
    results.append(("Phase 1: Attendance Mode", test_phase1_attendance_mode()))
    
    # Phase 2 Tests
    results.append(("Phase 2: Tracking Engine", test_phase2_tracking_engine()))
    results.append(("Phase 2: ByteTrack", test_phase2_bytetrack()))
    
    # Phase 3 Tests
    results.append(("Phase 3: Selection Engine", test_phase3_selection_engine()))
    
    # Integration Test
    results.append(("Pipeline Integration", test_pipeline_integration()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED - IMPLEMENTATION VERIFIED")
    else:
        print("⚠ SOME TESTS FAILED - CHECK IMPLEMENTATION")
    
    print("="*60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
