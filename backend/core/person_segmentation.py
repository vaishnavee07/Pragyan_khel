"""
Person Segmentation Engine
===========================
Uses MediaPipe Selfie Segmentation for real-time person masking.
Returns binary mask: 1.0 = person, 0.0 = background.

Fallback: If MediaPipe unavailable, uses simple GrabCut-based segmentation.
"""
import cv2
import numpy as np
from typing import Optional, Tuple


class PersonSegmentation:
    """
    Real-time person segmentation for subject isolation.
    
    Methods:
        segment_person(frame, bbox=None) -> mask (H×W float32, values 0-1)
    """
    
    def __init__(self, model_selection: int = 1):
        """
        Initialize person segmentation.
        
        Args:
            model_selection: 0 = general (256x256)
                           1 = landscape (256x144, faster, optimized for full-body)
        """
        self.backend = None
        self.model_selection = model_selection
        
        # Try MediaPipe first
        try:
            import mediapipe as mp
            self.mp_selfie = mp.solutions.selfie_segmentation
            self.segmenter = self.mp_selfie.SelfieSegmentation(
                model_selection=model_selection
            )
            self.backend = "mediapipe"
            print(f"✓ PersonSegmentation: MediaPipe backend (model={model_selection})")
        except ImportError:
            print("⚠ MediaPipe not found (pip install mediapipe)")
            print("  Falling back to GrabCut segmentation...")
            self.backend = "grabcut"
    
    def segment_person(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Segment person from background.
        
        Args:
            frame: BGR image (H×W×3)
            bbox: Optional region (x1, y1, x2, y2) to focus segmentation
            threshold: Confidence threshold for mask (0-1, default 0.5)
        
        Returns:
            Binary mask (H×W) float32, values 0.0-1.0
        """
        if self.backend == "mediapipe":
            return self._segment_mediapipe(frame, bbox, threshold)
        else:
            return self._segment_grabcut(frame, bbox)
    
    def _segment_mediapipe(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
        threshold: float,
    ) -> np.ndarray:
        """MediaPipe Selfie Segmentation"""
        h, w = frame.shape[:2]
        
        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run segmentation
        results = self.segmenter.process(rgb)
        
        if results.segmentation_mask is None:
            # Fallback: return full mask
            return np.ones((h, w), dtype=np.float32)
        
        # Get raw mask (values 0-1)
        mask = results.segmentation_mask.astype(np.float32)
        
        # Apply threshold
        mask = (mask > threshold).astype(np.float32)
        
        # If bbox provided, zero out mask outside bbox
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            bbox_mask = np.zeros((h, w), dtype=np.float32)
            bbox_mask[y1:y2, x1:x2] = 1.0
            mask = mask * bbox_mask
        
        return mask
    
    def _segment_grabcut(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """
        Fallback: GrabCut-based segmentation (slower, less accurate).
        Requires bbox to be provided.
        """
        h, w = frame.shape[:2]
        
        if bbox is None:
            # No bbox provided → return full mask
            return np.ones((h, w), dtype=np.float32)
        
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.ones((h, w), dtype=np.float32)
        
        # GrabCut requires (x, y, w, h) format
        rect = (x1, y1, x2 - x1, y2 - y1)
        
        # Initialize mask
        mask = np.zeros(frame.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(
                frame, mask, rect, bgd_model, fgd_model,
                iterCount=3,
                mode=cv2.GC_INIT_WITH_RECT
            )
            
            # Convert GrabCut mask to binary: 0,2 = bg, 1,3 = fg
            binary_mask = np.where((mask == 1) | (mask == 3), 1.0, 0.0).astype(np.float32)
            
            return binary_mask
        
        except Exception as e:
            print(f"[PersonSegmentation] GrabCut failed: {e}")
            # Return bbox-sized mask as fallback
            mask_fallback = np.zeros((h, w), dtype=np.float32)
            mask_fallback[y1:y2, x1:x2] = 1.0
            return mask_fallback
    
    def refine_mask(
        self,
        mask: np.ndarray,
        kernel_size: int = 5,
        feather: int = 2,
    ) -> np.ndarray:
        """
        Refine segmentation mask to remove noise and smooth edges.
        
        Args:
            mask: Binary mask (H×W float32)
            kernel_size: Morphological operation kernel size (odd number)
            feather: Edge blur radius (0 = no blur, >0 = softer edges)
        
        Returns:
            Refined mask (H×W float32)
        """
        # Ensure odd kernel
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        
        # Morphological closing: fill small holes
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Morphological opening: remove small noise
        mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        
        # Convert back to float
        mask_refined = mask_clean.astype(np.float32) / 255.0
        
        # Optional feathering (edge blur)
        if feather > 0:
            blur_k = feather * 2 + 1
            mask_refined = cv2.GaussianBlur(mask_refined, (blur_k, blur_k), 0)
        
        return mask_refined
    
    def cleanup(self):
        """Release resources"""
        if self.backend == "mediapipe" and hasattr(self, 'segmenter'):
            self.segmenter.close()
        print("✓ PersonSegmentation cleanup")


def create_segmenter() -> PersonSegmentation:
    """Factory function to create person segmentation engine."""
    return PersonSegmentation(model_selection=1)
