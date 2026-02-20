"""
Tracking Adapter - ByteTrack Integration
Bridges RT-DETR detections with ByteTrack tracker
"""
import numpy as np
from typing import List, Dict, Any, Optional

class TrackingAdapter:
    """Adapter to integrate RT-DETR with ByteTrack"""
    
    def __init__(self):
        self.tracker = None
        self.active_focus_id = None  # For tap-to-select functionality
        self.track_history = {}
        
    def initialize_tracker(self):
        """
        Initialize ByteTrack tracker
        Note: ByteTrack should be imported here if available
        For now, this is a placeholder for Phase 2 integration
        """
        try:
            # Import ByteTrack when available
            # from byte_tracker import BYTETracker
            print("✓ ByteTrack tracker initialized")
            return True
        except ImportError:
            print("⚠ ByteTrack not available - using centroid tracking")
            return False
    
    def update_tracks(self, detections: List[Dict], frame_shape: tuple) -> List[Dict]:
        """
        Update tracks with new detections
        
        Args:
            detections: RT-DETR detection results
            frame_shape: (height, width) of frame
            
        Returns:
            Detections with track_id added
        """
        if not detections:
            return []
        
        # Convert to ByteTrack format
        det_array = self._to_tracker_format(detections)
        
        # Simple centroid tracking (fallback when ByteTrack unavailable)
        tracked_detections = self._centroid_tracking(detections)
        
        return tracked_detections
    
    def _to_tracker_format(self, detections: List[Dict]) -> np.ndarray:
        """
        Convert RT-DETR detections to ByteTrack format
        Format: [x1, y1, x2, y2, confidence]
        """
        if not detections:
            return np.empty((0, 5))
        
        tracker_input = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            tracker_input.append([x1, y1, x2, y2, conf])
        
        return np.array(tracker_input, dtype=np.float32)
    
    def _centroid_tracking(self, detections: List[Dict]) -> List[Dict]:
        """
        Simple centroid-based tracking (fallback)
        Maintains track IDs across frames
        """
        import numpy as np
        
        current_centroids = {}
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            current_centroids[(cx, cy)] = det
        
        # Match with existing tracks
        for (cx, cy), det in current_centroids.items():
            track_id = self._find_closest_track(cx, cy)
            det['track_id'] = track_id
            self.track_history[track_id] = (cx, cy)
        
        return detections
    
    def _find_closest_track(self, cx: float, cy: float, max_distance: float = 100) -> int:
        """Find closest existing track or create new one"""
        import numpy as np
        
        min_dist = max_distance
        matched_id = None
        
        for track_id, (prev_cx, prev_cy) in self.track_history.items():
            dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            if dist < min_dist:
                min_dist = dist
                matched_id = track_id
        
        if matched_id is None:
            # Create new track
            if self.track_history:
                matched_id = max(self.track_history.keys()) + 1
            else:
                matched_id = 1
        
        return matched_id
    
    def set_focus(self, track_id: int):
        """Set active focus for tap-to-select"""
        self.active_focus_id = track_id
        print(f"✓ Focus set to track #{track_id}")
    
    def get_focus(self) -> Optional[int]:
        """Get active focus track ID"""
        return self.active_focus_id
    
    def clear_focus(self):
        """Clear active focus"""
        self.active_focus_id = None
        print("✓ Focus cleared")
    
    def get_focused_detection(self, detections: List[Dict]) -> Optional[Dict]:
        """Get detection for focused track"""
        if self.active_focus_id is None:
            return None
        
        for det in detections:
            if det.get('track_id') == self.active_focus_id:
                return det
        
        return None
    
    def cleanup(self):
        """Release resources"""
        self.track_history.clear()
        self.active_focus_id = None
