"""
Body Focus Engine
=================
Orchestrates detection → tracking → segmentation for full-body subject isolation.

Click Flow:
1. User clicks on face/torso
2. Detect all persons in frame (YOLO)
3. Find nearest person to click coords
4. Expand bbox to full body (20% width, 30% height padding)
5. Initialize tracker on full body region
6. Generate segmentation mask for selected person
7. Apply hard isolation rendering

Multi-person handling: Track selected person ID, mask only that person.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from enum import Enum


class FocusState(Enum):
    """Body focus tracking states"""
    IDLE = "idle"           # No target selected
    DETECTING = "detecting" # Running detection to find person
    TRACKING = "tracking"   # Actively tracking person
    LOST = "lost"          # Lost track of person


class BodyFocusEngine:
    """
    Full-body person tracking with segmentation-based isolation.
    
    Features:
    - Click → detect person → track full body
    - 20% width, 30% height padding around body
    - Dynamic re-centering (10% top headroom)
    - Person segmentation (MediaPipe/SAM/DeepLabv3)
    - Hard isolation rendering (no blur)
    """
    
    def __init__(
        self,
        detector=None,
        segmenter=None,
        isolation_renderer=None,
        padding_w: float = 0.20,
        padding_h: float = 0.30,
    ):
        """
        Initialize body focus engine.
        
        Args:
            detector: ObjectDetection module (YOLO/RT-DETR)
            segmenter: PersonSegmentation module
            isolation_renderer: SubjectIsolationRenderer module
            padding_w: Width padding around body (0.20 = 20%)
            padding_h: Height padding around body (0.30 = 30%)
        """
        self.detector = detector
        self.segmenter = segmenter
        self.iso_renderer = isolation_renderer
        
        self.padding_w = padding_w
        self.padding_h = padding_h
        
        # State
        self.state = FocusState.IDLE
        self.selected_person_id: Optional[int] = None
        self.tracked_bbox: Optional[Tuple[int, int, int, int]] = None
        self.tracker = None
        
        # Moving average for re-centering
        self.center_history = []
        self.max_history = 10
        
        # Pending click (queued until next frame)
        self.pending_click: Optional[Tuple[int, int]] = None
        
        print("✓ BodyFocusEngine initialized")
    
    def on_click(self, x: int, y: int):
        """
        User clicked at (x, y) to select person.
        Queues click for processing in next process_frame() call.
        """
        self.pending_click = (x, y)
        print(f"[BodyFocus] Click queued: ({x}, {y})")
    
    def on_double_click(self):
        """Double-click to deselect person."""
        self.reset()
        print("[BodyFocus] Double-click → Reset")
    
    def process_frame(
        self,
        frame: np.ndarray,
        detections: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Process single frame for body focus + isolation.
        
        Args:
            frame: BGR image (H×W×3)
            detections: Optional pre-computed detections from detector
                       Format: [{'class': str, 'confidence': float, 'bbox': [x1,y1,x2,y2], 'track_id': int}]
        
        Returns:
            {
                'isolated_frame': np.ndarray or None,
                'mask': np.ndarray or None,
                'bbox': Tuple[int,int,int,int] or None,
                'state': str,
                'person_id': int or None
            }
        """
        h, w = frame.shape[:2]
        
        # Handle pending click
        if self.pending_click is not None:
            click_x, click_y = self.pending_click
            self.pending_click = None
            
            # Find person at click location
            person_bbox, person_id = self._find_person_at_click(
                click_x, click_y, frame, detections
            )
            
            if person_bbox is not None:
                # Expand to full body with padding
                body_bbox = self._expand_to_body(person_bbox, w, h)
                
                # Initialize tracker
                self._init_tracker(frame, body_bbox)
                self.selected_person_id = person_id
                self.tracked_bbox = body_bbox
                self.state = FocusState.TRACKING
                
                print(f"[BodyFocus] Tracking person ID={person_id}, bbox={body_bbox}")
            else:
                print(f"[BodyFocus] No person found at click ({click_x}, {click_y})")
                return self._empty_result()
        
        # If not tracking, return original frame
        if self.state != FocusState.TRACKING or self.tracker is None:
            return self._empty_result()
        
        # Update tracker
        success, bbox_xywh = self.tracker.update(frame)
        
        if not success:
            print("[BodyFocus] Tracker lost target")
            self.state = FocusState.LOST
            return self._empty_result()
        
        # Convert bbox from (x, y, w, h) → (x1, y1, x2, y2)
        x, y, w_box, h_box = [int(v) for v in bbox_xywh]
        x1, y1 = x, y
        x2, y2 = x + w_box, y + h_box
        
        # Clamp to frame bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        self.tracked_bbox = (x1, y1, x2, y2)
        
        # Dynamic re-centering (moving average)
        self._update_center_moving_average(x1, y1, x2, y2)
        
        # Generate segmentation mask
        mask = None
        if self.segmenter is not None:
            mask = self.segmenter.segment_person(
                frame,
                bbox=self.tracked_bbox,
                threshold=0.5
            )
            # Refine mask: morphological ops + feather
            mask = self.segmenter.refine_mask(mask, kernel_size=5, feather=2)
        
        # Render isolated subject
        isolated_frame = None
        if self.iso_renderer is not None and mask is not None:
            isolated_frame = self.iso_renderer.render(frame, mask)
        
        return {
            'isolated_frame': isolated_frame,
            'mask': mask,
            'bbox': self.tracked_bbox,
            'state': self.state.value,
            'person_id': self.selected_person_id,
        }
    
    def _find_person_at_click(
        self,
        click_x: int,
        click_y: int,
        frame: np.ndarray,
        detections: Optional[list],
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[int]]:
        """
        Find person bbox nearest to click coordinates.
        
        Returns:
            (bbox, person_id) or (None, None) if no person found
        """
        # If no detector, return None
        if self.detector is None:
            print("[BodyFocus] No detector available")
            return None, None
        
        # If detections not provided, run detection
        if detections is None:
            detections = self.detector.detect(frame)
        
        # Filter for "person" class
        person_detections = [
            det for det in detections
            if det.get('class', '').lower() == 'person'
        ]
        
        if not person_detections:
            print("[BodyFocus] No persons detected in frame")
            return None, None
        
        # Find nearest person to click point
        min_dist = float('inf')
        nearest_person = None
        
        for det in person_detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            
            # Check if click is inside bbox
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                # Click inside → this is the person
                return tuple(bbox), det.get('track_id')
            
            # Otherwise, compute distance from click to bbox center
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = ((click_x - cx)**2 + (click_y - cy)**2)**0.5
            
            if dist < min_dist:
                min_dist = dist
                nearest_person = det
        
        # Return nearest person (even if click was outside)
        if nearest_person is not None:
            bbox = tuple(nearest_person['bbox'])
            person_id = nearest_person.get('track_id')
            print(f"[BodyFocus] Nearest person: bbox={bbox}, dist={min_dist:.1f}px")
            return bbox, person_id
        
        return None, None
    
    def _expand_to_body(
        self,
        bbox: Tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[int, int, int, int]:
        """
        Expand detection bbox to full body with padding.
        
        Args:
            bbox: (x1, y1, x2, y2) detected person bbox
            frame_w, frame_h: Frame dimensions
        
        Returns:
            Expanded bbox (x1, y1, x2, y2) with 20% width, 30% height padding
        """
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        
        # Apply padding
        new_w = w * (1 + 2 * self.padding_w)
        new_h = h * (1 + 2 * self.padding_h)
        
        # Re-center
        new_x1 = int(cx - new_w / 2)
        new_y1 = int(cy - new_h / 2)
        new_x2 = int(cx + new_w / 2)
        new_y2 = int(cy + new_h / 2)
        
        # Clamp to frame bounds
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(frame_w, new_x2)
        new_y2 = min(frame_h, new_y2)
        
        return (new_x1, new_y1, new_x2, new_y2)
    
    def _init_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Initialize OpenCV tracker on bbox."""
        x1, y1, x2, y2 = bbox
        bbox_xywh = (x1, y1, x2 - x1, y2 - y1)
        
        # Try CSRT first (most accurate), fallback to MIL
        try:
            self.tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            print("[BodyFocus] CSRT unavailable, using MIL tracker")
            self.tracker = cv2.TrackerMIL_create()
        
        self.tracker.init(frame, bbox_xywh)
        print(f"[BodyFocus] Tracker initialized: {bbox_xywh}")
    
    def _update_center_moving_average(
        self,
        x1: int, y1: int, x2: int, y2: int
    ):
        """
        Update moving average of bbox center for smooth re-centering.
        Stores last N centers for drift correction.
        """
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        self.center_history.append((cx, cy))
        
        # Keep only last N centers
        if len(self.center_history) > self.max_history:
            self.center_history.pop(0)
    
    def get_smoothed_center(self) -> Optional[Tuple[float, float]]:
        """Get smoothed center from moving average."""
        if not self.center_history:
            return None
        
        avg_cx = sum(c[0] for c in self.center_history) / len(self.center_history)
        avg_cy = sum(c[1] for c in self.center_history) / len(self.center_history)
        
        return (avg_cx, avg_cy)
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result when no tracking active."""
        return {
            'isolated_frame': None,
            'mask': None,
            'bbox': None,
            'state': self.state.value,
            'person_id': None,
        }
    
    def reset(self):
        """Reset tracking state."""
        self.state = FocusState.IDLE
        self.selected_person_id = None
        self.tracked_bbox = None
        self.tracker = None
        self.center_history = []
        self.pending_click = None
        print("[BodyFocus] Reset")
    
    def cleanup(self):
        """Release resources."""
        if self.segmenter is not None:
            self.segmenter.cleanup()
        print("✓ BodyFocusEngine cleanup")


def create_body_focus_engine(
    detector,
    segmenter=None,
    isolation_renderer=None,
) -> BodyFocusEngine:
    """
    Factory function to create body focus engine.
    
    Args:
        detector: ObjectDetection module (required)
        segmenter: PersonSegmentation module (optional)
        isolation_renderer: SubjectIsolationRenderer (optional)
    
    Returns:
        BodyFocusEngine instance
    """
    return BodyFocusEngine(
        detector=detector,
        segmenter=segmenter,
        isolation_renderer=isolation_renderer,
        padding_w=0.20,  # 20% width padding
        padding_h=0.30,  # 30% height padding
    )
