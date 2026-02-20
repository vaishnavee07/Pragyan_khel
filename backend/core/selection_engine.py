"""
Selection Engine - PHASE 3
Tap-to-select active focus system
Handles click events and manages active subject focus
"""
import time
from typing import List, Dict, Any, Optional, Tuple


class SelectionEngine:
    """
    Selection engine for tap-to-select functionality
    Independent logic layer for subject selection
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize selection engine
        
        Args:
            config: Configuration dict
                - timeout: Seconds to maintain focus after subject lost (default 1.0)
                - auto_reset: Auto-reset focus when subject lost (default True)
        """
        self.config = config or {}
        self.timeout = self.config.get('timeout', 1.0)
        self.auto_reset = self.config.get('auto_reset', True)
        
        self.active_focus_id = None
        self.last_seen_time = None
        self.focus_history = []
        
        print(f"✓ SelectionEngine initialized")
        print(f"  Timeout: {self.timeout}s")
        print(f"  Auto-reset: {self.auto_reset}")
    
    def handle_click(
        self, 
        click_x: int, 
        click_y: int, 
        tracked_objects: List[Dict[str, Any]]
    ) -> Optional[int]:
        """
        Handle click event and determine which object was clicked
        
        Args:
            click_x: X coordinate of click
            click_y: Y coordinate of click
            tracked_objects: List of tracked objects with bboxes and track_ids
            
        Returns:
            track_id of clicked object, or None if no object clicked
        """
        if not tracked_objects:
            return None
        
        # Find which bounding box contains the click
        clicked_track_id = None
        
        for obj in tracked_objects:
            bbox = obj.get('bbox')
            track_id = obj.get('track_id')
            
            if not bbox or track_id is None:
                continue
            
            if self._point_in_bbox(click_x, click_y, bbox):
                clicked_track_id = track_id
                break
        
        # Update active focus
        if clicked_track_id is not None:
            self._set_focus(clicked_track_id)
            print(f"[SELECTION] ✓ Focus set to track_id={clicked_track_id}")
        else:
            print(f"[SELECTION] ⚠ Click at ({click_x}, {click_y}) - no object found")
        
        return clicked_track_id
    
    def _point_in_bbox(self, x: int, y: int, bbox: List[int]) -> bool:
        """
        Check if point is inside bounding box
        
        Args:
            x, y: Point coordinates
            bbox: [x1, y1, x2, y2]
            
        Returns:
            True if point is inside bbox
        """
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def _set_focus(self, track_id: int):
        """Set active focus to track_id"""
        if self.active_focus_id != track_id:
            # Log focus change
            if self.active_focus_id is not None:
                print(f"[SELECTION] Focus switched: {self.active_focus_id} → {track_id}")
            
            self.active_focus_id = track_id
            self.last_seen_time = time.time()
            
            # Add to history
            self.focus_history.append({
                'track_id': track_id,
                'timestamp': self.last_seen_time,
                'event': 'focus_set'
            })
    
    def get_active_focus_object(
        self, 
        tracked_objects: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Get currently focused object from tracked objects list
        
        Args:
            tracked_objects: List of tracked objects
            
        Returns:
            Focused object dict, or None if not found
        """
        if self.active_focus_id is None:
            return None
        
        # Find object with matching track_id
        for obj in tracked_objects:
            if obj.get('track_id') == self.active_focus_id:
                self.last_seen_time = time.time()
                return obj
        
        # Object not found in current frame
        return self._handle_lost_focus()
    
    def _handle_lost_focus(self) -> Optional[Dict[str, Any]]:
        """
        Handle case when focused object is lost
        Maintains focus for timeout period, then resets
        
        Returns:
            None (focus maintained or reset)
        """
        if self.last_seen_time is None:
            self.reset_focus()
            return None
        
        time_since_seen = time.time() - self.last_seen_time
        
        if time_since_seen > self.timeout:
            # Timeout exceeded - reset focus
            if self.auto_reset:
                print(f"[SELECTION] ⚠ Track {self.active_focus_id} lost (timeout)")
                
                self.focus_history.append({
                    'track_id': self.active_focus_id,
                    'timestamp': time.time(),
                    'event': 'focus_lost_timeout'
                })
                
                self.reset_focus()
            
            return None
        else:
            # Still within timeout - maintain focus
            return {
                'track_id': self.active_focus_id,
                'status': 'tracking_lost',
                'time_since_seen': round(time_since_seen, 2)
            }
    
    def reset_focus(self):
        """Clear active focus"""
        if self.active_focus_id is not None:
            print(f"[SELECTION] Focus reset (was: track_id={self.active_focus_id})")
            
            self.focus_history.append({
                'track_id': self.active_focus_id,
                'timestamp': time.time(),
                'event': 'focus_reset'
            })
        
        self.active_focus_id = None
        self.last_seen_time = None
    
    def get_focus_id(self) -> Optional[int]:
        """Get current active focus ID"""
        return self.active_focus_id
    
    def is_focused(self, track_id: int) -> bool:
        """Check if given track_id is currently focused"""
        return self.active_focus_id == track_id
    
    def get_focus_status(self) -> Dict[str, Any]:
        """
        Get current focus status
        
        Returns:
            Status dict with focus information
        """
        status = {
            'has_focus': self.active_focus_id is not None,
            'focus_id': self.active_focus_id,
            'timeout': self.timeout
        }
        
        if self.active_focus_id is not None and self.last_seen_time is not None:
            time_since_seen = time.time() - self.last_seen_time
            status['time_since_seen'] = round(time_since_seen, 2)
            status['timeout_remaining'] = max(0, round(self.timeout - time_since_seen, 2))
        
        return status
    
    def get_focus_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent focus history
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of focus events
        """
        return self.focus_history[-limit:]
    
    def handle_click_with_tolerance(
        self,
        click_x: int,
        click_y: int,
        tracked_objects: List[Dict[str, Any]],
        tolerance: int = 10
    ) -> Optional[int]:
        """
        Handle click with tolerance for near-misses
        Expands bboxes by tolerance pixels
        
        Args:
            click_x, click_y: Click coordinates
            tracked_objects: Tracked objects list
            tolerance: Pixel tolerance for bbox expansion
            
        Returns:
            track_id of clicked object
        """
        if not tracked_objects:
            return None
        
        # First try exact match
        for obj in tracked_objects:
            bbox = obj.get('bbox')
            track_id = obj.get('track_id')
            
            if not bbox or track_id is None:
                continue
            
            if self._point_in_bbox(click_x, click_y, bbox):
                self._set_focus(track_id)
                return track_id
        
        # Try with tolerance
        for obj in tracked_objects:
            bbox = obj.get('bbox')
            track_id = obj.get('track_id')
            
            if not bbox or track_id is None:
                continue
            
            # Expand bbox by tolerance
            expanded_bbox = [
                bbox[0] - tolerance,
                bbox[1] - tolerance,
                bbox[2] + tolerance,
                bbox[3] + tolerance
            ]
            
            if self._point_in_bbox(click_x, click_y, expanded_bbox):
                self._set_focus(track_id)
                print(f"[SELECTION] ✓ Focus set (with tolerance) to track_id={track_id}")
                return track_id
        
        print(f"[SELECTION] ⚠ Click at ({click_x}, {click_y}) - no object found even with tolerance")
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get selection statistics"""
        return {
            'active_focus_id': self.active_focus_id,
            'has_focus': self.active_focus_id is not None,
            'total_focus_events': len(self.focus_history),
            'timeout': self.timeout,
            'auto_reset': self.auto_reset
        }
    
    def cleanup(self):
        """Cleanup selection engine"""
        self.reset_focus()
        self.focus_history.clear()
        print("✓ SelectionEngine cleanup complete")
