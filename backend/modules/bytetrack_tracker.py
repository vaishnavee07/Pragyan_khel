"""
ByteTrack Tracker Module - PHASE 2
ByteTrack: Multi-Object Tracking by Associating Every Detection Box
Paper: https://arxiv.org/abs/2110.06864
Windows-compatible implementation
"""
import numpy as np
from typing import List, Tuple
from collections import deque


class STrack:
    """Single target track"""
    
    shared_kalman = None
    track_id_count = 1
    
    def __init__(self, tlwh, score, cls_id=0):
        """
        Args:
            tlwh: [x, y, w, h]
            score: confidence score
            cls_id: class ID
        """
        # Convert tlwh to tlbr (x1, y1, x2, y2)
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.tlbr = self._tlwh_to_tlbr(self.tlwh)
        
        self.score = score
        self.cls_id = cls_id
        
        self.is_activated = False
        self.track_id = 0
        self.frame_id = 0
        self.tracklet_len = 0
        
        self.state = 'new'  # new, tracked, lost, removed
        
        # For smoothing
        self.smooth_feat = None
        self.features = deque([], maxlen=50)
        self.alpha = 0.9
    
    def activate(self, frame_id):
        """Activate track"""
        self.track_id = self.next_id()
        self.frame_id = frame_id
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
    
    def update(self, new_track, frame_id):
        """Update track with new detection"""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        self.tlwh = new_track.tlwh
        self.tlbr = new_track.tlbr
        self.score = new_track.score
        
        self.state = 'tracked'
        self.is_activated = True
    
    def mark_lost(self):
        """Mark track as lost"""
        self.state = 'lost'
    
    def mark_removed(self):
        """Mark track as removed"""
        self.state = 'removed'
    
    @staticmethod
    def next_id():
        """Get next track ID"""
        STrack.track_id_count += 1
        return STrack.track_id_count
    
    @staticmethod
    def _tlwh_to_tlbr(tlwh):
        """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret
    
    @staticmethod
    def _tlbr_to_tlwh(tlbr):
        """Convert [x1, y1, x2, y2] to [x, y, w, h]"""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret


class BYTETracker:
    """
    ByteTrack: Multi-object tracker
    Simplified Windows-compatible version
    """
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30
    ):
        """
        Args:
            track_thresh: Detection score threshold for tracking
            track_buffer: Frames to keep lost tracks
            match_thresh: IoU threshold for matching
            frame_rate: Expected FPS
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        self.frame_id = 0
        self.tracked_stracks = []  # Active tracks
        self.lost_stracks = []     # Lost tracks
        self.removed_stracks = []  # Removed tracks
        
        self.max_time_lost = self.track_buffer
        
        print(f"✓ BYTETracker initialized")
        print(f"  Track Threshold: {track_thresh}")
        print(f"  Match Threshold: {match_thresh}")
        print(f"  Track Buffer: {track_buffer}")
    
    def update(self, detections: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Update tracker with new detections
        
        Args:
            detections: [N, 5] array with [x1, y1, x2, y2, score]
            frame_shape: (height, width)
            
        Returns:
            [M, 7] array with [x1, y1, x2, y2, track_id, score, class_id]
        """
        self.frame_id += 1
        
        if len(detections) == 0:
            # Remove old lost tracks
            self.tracked_stracks = self._remove_duplicate_stracks(self.tracked_stracks)
            return np.empty((0, 7))
        
        # Separate high and low confidence detections
        det_high = detections[detections[:, 4] >= self.track_thresh]
        det_low = detections[detections[:, 4] < self.track_thresh]
        
        # Convert to STrack objects
        detections_high = [
            STrack(self._tlbr_to_tlwh(det[:4]), det[4], cls_id=0)
            for det in det_high
        ]
        
        detections_low = [
            STrack(self._tlbr_to_tlwh(det[:4]), det[4], cls_id=0)
            for det in det_low
        ]
        
        # Match high confidence detections with tracked tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if track.is_activated:
                tracked_stracks.append(track)
            else:
                unconfirmed.append(track)
        
        # First association: high confidence detections with tracked tracks
        matches, u_track, u_detection = self._associate(
            tracked_stracks, detections_high, self.match_thresh
        )
        
        # Update matched tracks
        for itracked, idet in matches:
            track = tracked_stracks[itracked]
            det = detections_high[idet]
            track.update(det, self.frame_id)
        
        # Process unmatched tracks
        for it in u_track:
            track = tracked_stracks[it]
            track.mark_lost()
            self.lost_stracks.append(track)
        
        # Process unmatched high-score detections
        new_tracks = []
        for idet in u_detection:
            det = detections_high[idet]
            det.activate(self.frame_id)
            new_tracks.append(det)
        
        # Second association: low confidence detections with lost tracks
        if len(detections_low) > 0:
            matches_low, u_track_low, u_detection_low = self._associate(
                self.lost_stracks, detections_low, 0.5
            )
            
            for itracked, idet in matches_low:
                track = self.lost_stracks[itracked]
                det = detections_low[idet]
                track.update(det, self.frame_id)
                tracked_stracks.append(track)
        
        # Update tracked tracks
        self.tracked_stracks = [t for t in tracked_stracks if t.state == 'tracked']
        self.tracked_stracks.extend(new_tracks)
        
        # Remove old lost tracks
        self.lost_stracks = [
            t for t in self.lost_stracks
            if self.frame_id - t.frame_id <= self.max_time_lost
        ]
        
        # Prepare output
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        # Convert to output format: [x1, y1, x2, y2, track_id, score, class_id]
        outputs = []
        for t in output_stracks:
            outputs.append([
                t.tlbr[0], t.tlbr[1], t.tlbr[2], t.tlbr[3],
                t.track_id, t.score, t.cls_id
            ])
        
        return np.array(outputs) if len(outputs) > 0 else np.empty((0, 7))
    
    def _associate(self, tracks, detections, thresh):
        """
        Associate tracks with detections using IoU
        
        Returns:
            matches: [(track_idx, det_idx), ...]
            unmatched_tracks: [track_idx, ...]
            unmatched_detections: [det_idx, ...]
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.tlbr, det.tlbr)
        
        # Greedy matching
        matches = []
        unmatched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        for i in range(len(tracks)):
            if len(unmatched_detections) == 0:
                unmatched_tracks.append(i)
                continue
            
            # Find best detection for this track
            j = unmatched_detections[0]
            max_iou = iou_matrix[i, j]
            
            for det_idx in unmatched_detections:
                if iou_matrix[i, det_idx] > max_iou:
                    max_iou = iou_matrix[i, det_idx]
                    j = det_idx
            
            if max_iou >= thresh:
                matches.append((i, j))
                unmatched_detections.remove(j)
            else:
                unmatched_tracks.append(i)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _iou(self, box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
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
    
    def _tlbr_to_tlwh(self, tlbr):
        """Convert [x1, y1, x2, y2] to [x, y, w, h]"""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret
    
    def _remove_duplicate_stracks(self, tracks):
        """Remove duplicate tracks"""
        # Simple implementation: keep tracks with higher scores
        if len(tracks) == 0:
            return []
        
        tracks_dict = {}
        for t in tracks:
            if t.track_id not in tracks_dict:
                tracks_dict[t.track_id] = t
            elif t.score > tracks_dict[t.track_id].score:
                tracks_dict[t.track_id] = t
        
        return list(tracks_dict.values())
