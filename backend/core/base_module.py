"""
Base AI Module - Abstract class for all AI modes
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass
import time

@dataclass
class InferenceResult:
    """Standardized inference result structure"""
    mode: str
    fps: float
    detections: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    alert_level: str  # NORMAL, WARNING, CRITICAL
    inference_time: float
    timestamp: float

class BaseAIModule(ABC):
    """Abstract base class for all AI modules"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.mode_name = self.__class__.__name__.replace('Module', '').lower()
        self.is_initialized = False
        self.frame_count = 0
        self.total_inference_time = 0
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the AI module (load models, allocate resources)"""
        pass
    
    @abstractmethod
    def process_frame(self, frame) -> InferenceResult:
        """Process a single frame and return structured results"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Release resources and cleanup"""
        pass
    
    def get_average_inference_time(self) -> float:
        """Calculate average inference time"""
        if self.frame_count == 0:
            return 0
        return self.total_inference_time / self.frame_count
    
    def _create_result(self, detections: List[Dict], metrics: Dict, 
                      alert_level: str, inference_time: float) -> InferenceResult:
        """Helper to create standardized inference result"""
        return InferenceResult(
            mode=self.mode_name,
            fps=0,  # Will be set by engine
            detections=detections,
            metrics=metrics,
            alert_level=alert_level,
            inference_time=inference_time,
            timestamp=time.time()
        )
