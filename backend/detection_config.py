# SentraVision RT-DETR Configuration

# Detection settings
MODEL_TYPE = "rtdetr"  # "rtdetr" or "yolo"
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.6
DEVICE = "cpu"  # "cpu" or "cuda"

# Person-only detection filter
DETECT_ONLY_PERSON = False  # True = only detect persons, False = all classes

# Performance settings
TARGET_FPS = 15
ENABLE_GPU = True  # Auto-detect GPU if available

# COCO class IDs
PERSON_CLASS_ID = 0
