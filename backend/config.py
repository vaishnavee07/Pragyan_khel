"""
Configuration settings
"""
import os

class Config:
    """Application configuration"""
    
    # Camera settings
    CAMERA_INDEX = 0
    
    # Model settings
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "detect.tflite")
    CONFIDENCE_THRESHOLD = 0.5
    
    # Performance settings
    FRAME_SKIP = 2  # Process every Nth frame (1 = no skip, 2 = process every 2nd frame)
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000
