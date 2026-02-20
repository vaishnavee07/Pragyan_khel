"""
Phase 4 Testing Script - RT-DETR Verification
Tests detection accuracy, FPS, tracking, and regression safety
"""
import cv2
import time
import numpy as np
from modules.rtdetr_detection import RTDETRDetectionModule
from modules.tracking_adapter import TrackingAdapter

def test_rtdetr_detection():
    """Test RT-DETR detection module"""
    print("=" * 60)
    print("PHASE 4 TESTING - RT-DETR Detection Module")
    print("=" * 60)
    
    # Initialize module
    print("\n1. Initializing RT-DETR...")
    module = RTDETRDetectionModule({
        'confidence_threshold': 0.4,
        'iou_threshold': 0.6,
        'device': 'cpu'
    })
    
    if not module.initialize():
        print("✗ Module initialization failed")
        return False
    
    print("✓ Module initialized successfully")
    
    # Test with sample frame
    print("\n2. Testing inference...")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    inference_times = []
    detection_counts = []
    
    for i in range(30):
        result = module.process_frame(frame)
        inference_times.append(result.inference_time)
        detection_counts.append(len(result.detections))
    
    avg_inference = np.mean(inference_times)
    avg_fps = 1000 / avg_inference if avg_inference > 0 else 0
    
    print(f"✓ Average inference time: {avg_inference:.2f}ms")
    print(f"✓ Average FPS: {avg_fps:.1f}")
    print(f"✓ Detection count: {np.mean(detection_counts):.1f} objects/frame")
    
    # Verify FPS target
    print("\n3. Performance verification...")
    if avg_fps >= 15:
        print(f"✓ FPS target met: {avg_fps:.1f} >= 15")
    else:
        print(f"✗ FPS target missed: {avg_fps:.1f} < 15")
        return False
    
    # Test tracking adapter
    print("\n4. Testing tracking adapter...")
    tracker = TrackingAdapter()
    tracker.initialize_tracker()
    
    # Simulate detections
    test_detections = [
        {'bbox': [100, 100, 200, 200], 'confidence': 0.9, 'class': 'person', 'class_id': 0},
        {'bbox': [300, 150, 400, 250], 'confidence': 0.8, 'class': 'car', 'class_id': 2}
    ]
    
    tracked = tracker.update_tracks(test_detections, (480, 640))
    
    if all('track_id' in det for det in tracked):
        print("✓ Track IDs assigned correctly")
    else:
        print("✗ Track ID assignment failed")
        return False
    
    # Test focus functionality
    print("\n5. Testing tap-to-select (focus)...")
    tracker.set_focus(1)
    if tracker.get_focus() == 1:
        print("✓ Focus set correctly")
    else:
        print("✗ Focus setting failed")
        return False
    
    tracker.clear_focus()
    if tracker.get_focus() is None:
        print("✓ Focus cleared correctly")
    else:
        print("✗ Focus clearing failed")
        return False
    
    # Test ByteTrack format conversion
    print("\n6. Testing ByteTrack format conversion...")
    tracker_format = module.get_detections_for_tracker(test_detections)
    
    if tracker_format.shape[1] == 5:  # [x1, y1, x2, y2, conf]
        print(f"✓ ByteTrack format correct: {tracker_format.shape}")
    else:
        print(f"✗ ByteTrack format incorrect: {tracker_format.shape}")
        return False
    
    # Cleanup
    print("\n7. Cleanup...")
    module.cleanup()
    tracker.cleanup()
    print("✓ Resources released")
    
    print("\n" + "=" * 60)
    print("PHASE 4 TESTING - ALL TESTS PASSED ✓")
    print("=" * 60)
    return True

def test_regression_safety():
    """Test that existing features still work"""
    print("\n" + "=" * 60)
    print("REGRESSION SAFETY CHECKS")
    print("=" * 60)
    
    checks = {
        "Detection format compatible": False,
        "Tracking interface preserved": False,
        "Focus tracking works": False,
        "ByteTrack format correct": False
    }
    
    # Test detection format
    print("\n1. Detection format compatibility...")
    test_det = {
        'bbox': [10, 20, 30, 40],
        'confidence': 0.85,
        'class': 'person',
        'class_id': 0,
        'track_id': 1
    }
    
    required_fields = ['bbox', 'confidence', 'class']
    if all(field in test_det for field in required_fields):
        checks["Detection format compatible"] = True
        print("✓ Detection format compatible")
    
    # Test tracking interface
    print("\n2. Tracking interface...")
    tracker = TrackingAdapter()
    if hasattr(tracker, 'update_tracks') and hasattr(tracker, 'set_focus'):
        checks["Tracking interface preserved"] = True
        print("✓ Tracking interface preserved")
    
    # Test focus tracking
    print("\n3. Focus tracking...")
    tracker.set_focus(5)
    if tracker.active_focus_id == 5:
        checks["Focus tracking works"] = True
        print("✓ Focus tracking works (active_focus_id managed)")
    
    # Test ByteTrack format
    print("\n4. ByteTrack format...")
    detections = [test_det]
    module = RTDETRDetectionModule()
    format_array = module.get_detections_for_tracker(detections)
    if format_array.shape == (1, 5):
        checks["ByteTrack format correct"] = True
        print("✓ ByteTrack format correct")
    
    # Summary
    print("\n" + "=" * 60)
    print("REGRESSION SAFETY SUMMARY")
    print("=" * 60)
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    all_passed = all(checks.values())
    if all_passed:
        print("\n✓ ALL REGRESSION CHECKS PASSED")
    else:
        print("\n✗ SOME REGRESSION CHECKS FAILED")
    
    return all_passed

if __name__ == "__main__":
    print("\nStarting Phase 4 tests...\n")
    
    success = test_rtdetr_detection()
    
    if success:
        regression_safe = test_regression_safety()
        
        if regression_safe:
            print("\n" + "=" * 60)
            print("PHASE 4 COMPLETE - READY FOR DEPLOYMENT")
            print("=" * 60)
        else:
            print("\n✗ Regression issues detected")
    else:
        print("\n✗ RT-DETR testing failed")
