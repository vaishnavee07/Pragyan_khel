"""
AI Engine - Core controller for AI mode management
"""
from typing import Dict, Optional
import threading
from core.base_module import BaseAIModule, InferenceResult
from core.alert_engine import AlertEngine, AlertLevel

class AIEngine:
    """Central AI engine managing multiple AI modules"""
    
    def __init__(self):
        self.modules: Dict[str, BaseAIModule] = {}
        self.active_module: Optional[BaseAIModule] = None
        self.active_mode: Optional[str] = None
        self.alert_engine = AlertEngine()
        self.lock = threading.Lock()
        self.is_running = False
        
    def register_module(self, mode_name: str, module: BaseAIModule):
        """Register an AI module"""
        with self.lock:
            self.modules[mode_name] = module
            print(f"✓ Registered AI module: {mode_name}")
    
    def switch_mode(self, mode_name: str) -> bool:
        """Switch to a different AI mode"""
        with self.lock:
            if mode_name not in self.modules:
                print(f"✗ Unknown mode: {mode_name}")
                return False
            
            # Cleanup current module
            if self.active_module:
                try:
                    self.active_module.cleanup()
                    print(f"✓ Cleaned up mode: {self.active_mode}")
                except Exception as e:
                    print(f"✗ Error cleaning up {self.active_mode}: {e}")
            
            # Initialize new module
            new_module = self.modules[mode_name]
            try:
                if new_module.initialize():
                    self.active_module = new_module
                    self.active_mode = mode_name
                    self.alert_engine.reset()
                    print(f"✓ Switched to mode: {mode_name}")
                    return True
                else:
                    print(f"✗ Failed to initialize mode: {mode_name}")
                    return False
            except Exception as e:
                print(f"✗ Error initializing {mode_name}: {e}")
                return False
    
    def process_frame(self, frame, fps: float) -> Optional[InferenceResult]:
        """Process frame with active AI module"""
        if not self.active_module:
            return None
        
        try:
            result = self.active_module.process_frame(frame)
            result.fps = fps
            
            # Evaluate alert level
            alert_level = self.alert_engine.evaluate(result)
            result.alert_level = alert_level.value
            
            return result
            
        except Exception as e:
            print(f"✗ Error processing frame: {e}")
            return None
    
    def get_available_modes(self) -> list:
        """Get list of available AI modes"""
        return list(self.modules.keys())
    
    def get_active_mode(self) -> Optional[str]:
        """Get currently active mode"""
        return self.active_mode
    
    def shutdown(self):
        """Shutdown engine and cleanup all modules"""
        with self.lock:
            if self.active_module:
                try:
                    self.active_module.cleanup()
                except Exception as e:
                    print(f"✗ Error during shutdown: {e}")
            
            self.active_module = None
            self.active_mode = None
            self.is_running = False
            print("✓ AI Engine shutdown complete")
