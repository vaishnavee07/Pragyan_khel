"""
Tracking Engine - PHASE 2
Enterprise-grade tracking layer using ByteTrack
Independent from detection and business logic
"""
import time
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque


class TrackingEngine:
    """
    Centralized tracking engine
    Accepts detections and assigns stable track IDs
    Uses ByteTrack for robust multi-object tracking
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize tracking engine
        
        Args:
            config: Configuration dict
                - track_thresh: Detection score threshold for tracking (default 0.5)
                - track_buffer: Number of frames to keep lost tracks (default 30)
                - match_thresh: Matching threshold for IoU (default 0.8)
                - fps: Expected FPS for trajectory prediction (default 30)
        """
        self.config = config or {}
        self.track_thresh = self.config.get('track_thresh', 0.5)
        self.track_buffer = self.config.get('track_buffer', 30)
        self.match_thresh = self.config.get('match_thresh', 0.8)
        self.fps = self.config.get('fps', 30)
        
        self.tracker = None
        self.is_initialized = False
        
        # Performance tracking
        self.frame_count = 0
        self.total_tracking_time = 0
        self.fps_history = deque(maxlen=30)
        self.frame_times = deque(maxlen=30)
        
        # Lifecycle tracking
        self.active_tracks = set()
        self.track_history = {}  # track_id -> metadata
        
        print(f"✓ TrackingEngine created")
        print(f"  Track Threshold: {self.track_thresh}")
        print(f"  Track Buffer: {self.track_buffer} frames")
        print(f"  Match Threshold: {self.match_thresh}")
    
    def initialize(self) -> bool:
        """Initialize ByteTrack tracker"""
        try:
            from modules.bytetrack_tracker import BYTETracker
            
            # Initialize ByteTrack with config
            self.tracker = BYTETracker(
                track_thresh=self.track_thresh,
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
                frame_rate=self.fps
            )
            
            self.is_initialized = True
            print("✓ ByteTrack tracker initialized")
            return True
            
        except ImportError:
            print("⚠ ByteTrack not available - using fallback tracker")
            self.tracker = self._create_fallback_tracker()
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"✗ Tracker initialization failed: {e}")
            return False
    
    def update(self, detections: List[Dict[str, Any]], frame_shape: tuple) -> List[Dict[str, Any]]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of detections from detection engine
                Each detection: {bbox, confidence, class_id, class_name}
            frame_shape: (height, width) of frame
            
        Returns:
            List of tracked objects with added track_id:
            [
                {
                    "track_id": int,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "class_name": str,
                    "class_id": int
                }
            ]
        """
        if not self.is_initialized or not self.tracker:
            print("✗ Tracker not initialized")
            return detections
        
        start_time = time.time()
        
        # Convert detections to tracker format
        det_array = self._to_tracker_format(detections)
        
        # Update tracker
        tracked_objects = self.tracker.update(det_array, frame_shape)
        
        # Convert back to standard format and add track IDs
        tracked_detections = self._from_tracker_format(
            tracked_objects, 
            detections,
            frame_shape
        )
        
        # Update lifecycle tracking
        self._update_lifecycle(tracked_detections)
        
        tracking_time = (time.time() - start_time) * 1000
        
        self.frame_count += 1
        self.total_tracking_time += tracking_time
        
        # Calculate FPS
        current_time = time.time()
        self.frame_times.append(current_time)
        if len(self.frame_times) >= 2:
            fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
            self.fps_history.append(fps)
        
        # Log every 30 frames
        if self.frame_count % 30 == 0:
            self._log_tracking_stats()
        
        return tracked_detections
    
    def _to_tracker_format(self, detections: List[Dict]) -> np.ndarray:
        """
        Convert detections to ByteTrack format
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
    
    def _from_tracker_format(
        self, 
        tracked_objects: np.ndarray,
        original_detections: List[Dict],
        frame_shape: tuple
    ) -> List[Dict[str, Any]]:
        """
        Convert ByteTrack output back to standard format
        ByteTrack output: [x1, y1, x2, y2, track_id, score, class_id]
        """
        if len(tracked_objects) == 0:
            return []
        
        tracked_detections = []
        
        for track in tracked_objects:
            if len(track) >= 5:
                x1, y1, x2, y2 = track[:4]
                track_id = int(track[4])
                
                # Find matching detection for class info
                bbox_match = self._find_matching_detection(
                    [x1, y1, x2, y2],
                    original_detections
                )
                
                if bbox_match:
                    tracked_detections.append({
                        'track_id': track_id,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': bbox_match['confidence'],
                        'class_name': bbox_match['class_name'],
                        'class_id': bbox_match.get('class_id', 0)
                    })
        
        return tracked_detections
    
    def _find_matching_detection(self, bbox: List, detections: List[Dict]) -> Optional[Dict]:
        """Find detection that matches tracked bbox (using IoU)"""
        best_match = None
        best_iou = 0.0
        
        for det in detections:
            iou = self._calculate_iou(bbox, det['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_match = det
        
        return best_match if best_iou > 0.3 else None
    
    def _calculate_iou(self, box1: List, box2: List) -> float:
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-7)
    
    def _update_lifecycle(self, tracked_detections: List[Dict]):
        """Track lifecycle events: new tracks, lost tracks, recovered tracks"""
        current_tracks = {det['track_id'] for det in tracked_detections}
        
        # Detect new tracks
        new_tracks = current_tracks - self.active_tracks
        for track_id in new_tracks:
            print(f"[TRACKING] ✓ New track: ID={track_id}")
            self.track_history[track_id] = {
                'created': self.frame_count,
                'last_seen': self.frame_count
            }
        
        # Detect lost tracks
        lost_tracks = self.active_tracks - current_tracks
        for track_id in lost_tracks:
            if track_id in self.track_history:
                frames_alive = self.frame_count - self.track_history[track_id]['created']
                print(f"[TRACKING] ✗ Lost track: ID={track_id} (alive {frames_alive} frames)")
        
        # Update active tracks
        for track_id in current_tracks:
            if track_id in self.track_history:
                self.track_history[track_id]['last_seen'] = self.frame_count
        
        self.active_tracks = current_tracks
    
    def _log_tracking_stats(self):
        """Log tracking performance statistics"""
        avg_tracking_time = (
            self.total_tracking_time / self.frame_count
            if self.frame_count > 0 else 0
        )
        
        rolling_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        print("\n" + "="*50)
        print("[TRACKING PERFORMANCE]")
        print(f"  Frame: {self.frame_count}")
        print(f"  Active Tracks: {len(self.active_tracks)}")
        print(f"  Tracking Time: {avg_tracking_time:.1f} ms (avg)")
        print(f"  FPS: {rolling_fps:.1f}")
        print(f"  Total Tracked: {len(self.track_history)} objects")
        print("="*50 + "\n")
    
    def _create_fallback_tracker(self):
        """Create simple centroid tracker as fallback"""
        class SimpleCentroidTracker:
            def __init__(self):
                self.next_id = 1
                self.centroids = {}
                self.max_distance = 100
            
            def update(self, detections: np.ndarray, frame_shape: tuple):
                if len(detections) == 0:
                    return np.empty((0, 5))
                
                # Calculate centroids
                current_centroids = {}
                for det in detections:
                    x1, y1, x2, y2, conf = det
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    current_centroids[(cx, cy)] = det
                
                # Match with existing tracks
                tracked = []
                for (cx, cy), det in current_centroids.items():
                    track_id = self._find_closest_track(cx, cy)
                    x1, y1, x2, y2, conf = det
                    tracked.append([x1, y1, x2, y2, track_id])
                    self.centroids[track_id] = (cx, cy)
                
                return np.array(tracked)
            
            def _find_closest_track(self, cx, cy):
                min_dist = self.max_distance
                matched_id = None
                
                for track_id, (prev_cx, prev_cy) in self.centroids.items():
                    dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        matched_id = track_id
                
                if matched_id is None:
                    matched_id = self.next_id
                    self.next_id += 1
                
                return matched_id
        
        return SimpleCentroidTracker()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        avg_tracking_time = (
            self.total_tracking_time / self.frame_count
            if self.frame_count > 0 else 0
        )
        rolling_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        return {
            'frame_count': self.frame_count,
            'active_tracks': len(self.active_tracks),
            'total_tracks': len(self.track_history),
            'avg_tracking_ms': round(avg_tracking_time, 2),
            'rolling_fps': round(rolling_fps, 1)
        }
    
    def cleanup(self):
        """Release tracker resources"""
        self.tracker = None
        self.active_tracks.clear()
        self.track_history.clear()
        self.is_initialized = False
        print("✓ TrackingEngine cleanup complete")
