"""
Alert Engine - System-wide alert management
"""
from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass
import time

class AlertLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    level: AlertLevel
    message: str
    timestamp: float
    metadata: Dict[str, Any]

class AlertEngine:
    """Manages system-wide alerts and state"""
    
    def __init__(self):
        self.current_level = AlertLevel.NORMAL
        self.alert_history: List[Alert] = []
        self.max_history = 100
        self.rules = {}
        
    def evaluate(self, inference_result) -> AlertLevel:
        """Evaluate inference result and determine alert level"""
        detection_count = len(inference_result.detections)
        
        # Critical conditions
        if detection_count >= 5:
            return AlertLevel.CRITICAL
        
        # Warning conditions
        if detection_count >= 3:
            return AlertLevel.WARNING
        
        # Check for specific object types
        critical_objects = {'knife', 'gun', 'fire'}
        warning_objects = {'person'}
        
        for det in inference_result.detections:
            obj_class = det.get('class', '').lower()
            if obj_class in critical_objects:
                return AlertLevel.CRITICAL
            if obj_class in warning_objects and detection_count >= 2:
                return AlertLevel.WARNING
        
        return AlertLevel.NORMAL
    
    def add_rule(self, name: str, condition_fn):
        """Add custom alert rule"""
        self.rules[name] = condition_fn
    
    def create_alert(self, level: AlertLevel, message: str, metadata: Dict = None):
        """Create new alert"""
        alert = Alert(
            level=level,
            message=message,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        self.current_level = level
        return alert
    
    def get_alert_color(self, level: AlertLevel) -> str:
        """Get UI color for alert level"""
        colors = {
            AlertLevel.NORMAL: "#10b981",
            AlertLevel.WARNING: "#f59e0b",
            AlertLevel.CRITICAL: "#ef4444"
        }
        return colors[level]
    
    def reset(self):
        """Reset to normal state"""
        self.current_level = AlertLevel.NORMAL
