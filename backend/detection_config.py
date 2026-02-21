# SentraVision — YOLOv11 Production Configuration

# Detection settings
MODEL_TYPE = "yolo"              # "yolo" = YOLOv11 pipeline
CONFIDENCE_THRESHOLD = 0.35     # Slightly lower for max recall
IOU_THRESHOLD = 0.50             # Standard NMS IoU
DEVICE = "auto"                  # "auto" = CUDA if available, else CPU

# Person-only detection filter
DETECT_ONLY_PERSON = False       # False = detect all objects (full scene awareness)

# Performance settings
TARGET_FPS = 30
ENABLE_GPU = True                # Auto-detect GPU via torch.cuda.is_available()

# Detection frequency — run inference every N frames, interpolate in between
# Higher N = better FPS; Lower N = more responsive tracking
DETECTION_INTERVAL = 2          # Run YOLOv11 every 2nd frame

# COCO class IDs
PERSON_CLASS_ID = 0
