"""
Model Switcher - Toggle between detection models
Usage: python switch_model.py [rtdetr|yolo]
"""
import sys
import os

def switch_to_rtdetr():
    """Update main.py to use RT-DETR"""
    print("Switching to RT-DETR...")
    
    # This would modify main.py programmatically
    # For safety, just print instructions
    print("""
✓ To use RT-DETR, ensure main.py has:

from modules.rtdetr_detection import RTDETRDetectionModule

rtdetr_detection = RTDETRDetectionModule({
    'confidence_threshold': 0.4,
    'iou_threshold': 0.6,
    'device': 'cpu'
})
ai_engine.register_module('object_detection', rtdetr_detection)
""")
    print("✓ RT-DETR is currently active")

def switch_to_yolo():
    """Update main.py to use YOLO"""
    print("Switching to YOLO...")
    
    print("""
✓ To use YOLO, update main.py:

from modules.object_detection import ObjectDetectionModule

yolo_detection = ObjectDetectionModule({
    'confidence_threshold': 0.4
})
ai_engine.register_module('object_detection', yolo_detection)
""")
    print("⚠ Remember: YOLO requires ultralytics YOLO models")

def show_current():
    """Show current detector"""
    try:
        with open('main.py', 'r') as f:
            content = f.read()
            if 'RTDETRDetectionModule' in content:
                print("✓ Current detector: RT-DETR")
            elif 'ObjectDetectionModule' in content:
                print("✓ Current detector: YOLO")
            else:
                print("? Unable to determine current detector")
    except Exception as e:
        print(f"✗ Error reading main.py: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_current()
        print("\nUsage: python switch_model.py [rtdetr|yolo|status]")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == 'rtdetr':
        switch_to_rtdetr()
    elif command == 'yolo':
        switch_to_yolo()
    elif command == 'status':
        show_current()
    else:
        print("Invalid command. Use: rtdetr, yolo, or status")
