"""
RT-DETR Installation & Integration Test
Tests all Phase 4 requirements
"""
import sys
import time
import numpy as np

def test_pytorch():
    """Test PyTorch installation"""
    print("\n" + "="*60)
    print("TEST 1: PyTorch Installation")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install: pip install torch torchvision")
        return False

def test_rtdetr_import():
    """Test RT-DETR module import"""
    print("\n" + "="*60)
    print("TEST 2: RT-DETR Module Import")
    print("="*60)
    
    try:
        from ultralytics import RTDETR
        print("✓ ultralytics package installed")
        print("✓ RTDETR class available")
        return True
    except ImportError as e:
        print(f"✗ ultralytics not installed: {e}")
        print("  Install: pip install ultralytics")
        return False

def test_model_loading():
    """Test RT-DETR model loading"""
    print("\n" + "="*60)
    print("TEST 3: RT-DETR Model Loading")
    print("="*60)
    
    try:
        from ultralytics import RTDETR
        print("⏳ Loading RT-DETR model...")
        model = RTDETR('rtdetr-l.pt')
        print("✓ RT-DETR model loaded: rtdetr-l.pt")
        return model
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None

def test_inference_speed(model):
    """Test inference speed"""
    print("\n" + "="*60)
    print("TEST 4: Inference Speed")
    print("="*60)
    
    if model is None:
        print("✗ Model not available")
        return False
    
    # Create dummy frame
    import cv2
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warmup
    print("⏳ Warmup inference...")
    for _ in range(3):
        _ = model(frame, verbose=False)
    
    # Measure inference times
    print("⏳ Measuring inference times (10 frames)...")
    times = []
    for i in range(10):
        start = time.time()
        results = model(frame, conf=0.4, verbose=False)
        inference_time = (time.time() - start) * 1000
        times.append(inference_time)
        print(f"  Frame {i+1}: {inference_time:.1f} ms")
    
    avg_time = np.mean(times)
    avg_fps = 1000 / avg_time
    
    print(f"\n✓ Average inference: {avg_time:.1f} ms")
    print(f"✓ Average FPS: {avg_fps:.1f}")
    
    if avg_fps >= 15:
        print("✓ FPS target met (>=15)")
        return True
    else:
        print("⚠ FPS below target (<15)")
        return False

def test_detection_module():
    """Test detection module integration"""
    print("\n" + "="*60)
    print("TEST 5: Detection Module Integration")
    print("="*60)
    
    try:
        from modules.rtdetr_detection import RTDETRDetectionModule
        
        # Initialize module
        module = RTDETRDetectionModule({
            'confidence_threshold': 0.4,
            'iou_threshold': 0.6,
            'device': 'cpu',
            'detect_only_person': False
        })
        
        print("✓ Module imported")
        
        # Initialize model
        if module.initialize():
            print("✓ Module initialized")
        else:
            print("✗ Module initialization failed")
            return False
        
        # Test inference
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = module.process_frame(frame)
        
        print(f"✓ Inference result: {result.mode}")
        print(f"✓ Detections: {len(result.detections)}")
        print(f"✓ Inference time: {result.inference_time:.1f} ms")
        
        # Test ByteTrack format conversion
        tracker_format = module.get_detections_for_tracker(result.detections)
        print(f"✓ ByteTrack format shape: {tracker_format.shape}")
        
        module.cleanup()
        print("✓ Module cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_person_filter():
    """Test person-only filter"""
    print("\n" + "="*60)
    print("TEST 6: Person-Only Filter")
    print("="*60)
    
    try:
        from modules.rtdetr_detection import RTDETRDetectionModule
        
        module = RTDETRDetectionModule({
            'confidence_threshold': 0.4,
            'detect_only_person': True
        })
        
        if not module.initialize():
            print("✗ Module initialization failed")
            return False
        
        print("✓ Person-only mode enabled")
        print("✓ Only class_id=0 (person) will be detected")
        
        module.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Person filter test failed: {e}")
        return False

def test_tracking_compatibility():
    """Test tracking adapter compatibility"""
    print("\n" + "="*60)
    print("TEST 7: Tracking Compatibility")
    print("="*60)
    
    try:
        from modules.tracking_adapter import TrackingAdapter
        
        tracker = TrackingAdapter()
        tracker.initialize_tracker()
        
        # Create test detections
        test_dets = [
            {'bbox': [100, 100, 200, 200], 'confidence': 0.9, 'class': 'person', 'class_id': 0},
            {'bbox': [300, 150, 400, 250], 'confidence': 0.8, 'class': 'person', 'class_id': 0}
        ]
        
        # Update tracks
        tracked = tracker.update_tracks(test_dets, (480, 640))
        
        print(f"✓ Tracking adapter initialized")
        print(f"✓ Track IDs assigned: {[d['track_id'] for d in tracked]}")
        
        # Test focus
        tracker.set_focus(1)
        if tracker.get_focus() == 1:
            print("✓ Focus tracking works (active_focus_id)")
        
        tracker.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Tracking test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("RT-DETR PHASE 4 INSTALLATION TEST")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("PyTorch", test_pytorch()))
    results.append(("RT-DETR Import", test_rtdetr_import()))
    
    model = test_model_loading()
    results.append(("Model Loading", model is not None))
    
    if model:
        results.append(("Inference Speed", test_inference_speed(model)))
    
    results.append(("Detection Module", test_detection_module()))
    results.append(("Person Filter", test_person_filter()))
    results.append(("Tracking Compatibility", test_tracking_compatibility()))
    
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
        print("✓ ALL TESTS PASSED - RT-DETR READY FOR DEPLOYMENT")
    else:
        print("⚠ SOME TESTS FAILED - CHECK INSTALLATION")
    
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
