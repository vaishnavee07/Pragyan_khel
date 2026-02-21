"""
Subject Isolation Renderer
===========================
Hard subject isolation using binary segmentation mask.
Background is completely removed (black or transparent).

NO BLUR. Pure alpha cutout.
"""
import cv2
import numpy as np
from typing import Literal


class SubjectIsolationRenderer:
    """
    Render isolated subject using binary segmentation mask.
    
    Background options:
    - 'black': Background pixels set to (0, 0, 0)
    - 'transparent': Returns BGRA with alpha=0 for background
    """
    
    def __init__(self, background_mode: Literal['black', 'transparent'] = 'black'):
        """
        Initialize isolation renderer.
        
        Args:
            background_mode: 'black' or 'transparent'
        """
        self.background_mode = background_mode
        print(f"✓ SubjectIsolationRenderer: background={background_mode}")
    
    def render(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Apply hard isolation to frame using binary mask.
        
        Args:
            frame: BGR image (H×W×3)
            mask: Binary mask (H×W) float32, values 0.0-1.0
                  1.0 = show pixel, 0.0 = hide pixel
        
        Returns:
            Isolated frame (BGR or BGRA depending on background_mode)
        """
        h, w = frame.shape[:2]
        
        # Ensure mask matches frame dimensions
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Ensure mask is float32 [0, 1]
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        
        if self.background_mode == 'transparent':
            return self._render_transparent(frame, mask)
        else:
            return self._render_black(frame, mask)
    
    def _render_black(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Hard isolation with black background.
        
        For each pixel:
            if mask == 1: show original pixel
            if mask == 0: set to black (0, 0, 0)
        """
        # Create black background
        result = np.zeros_like(frame, dtype=np.uint8)
        
        # Expand mask to 3 channels: (H, W) → (H, W, 1) → (H, W, 3)
        mask_3ch = np.expand_dims(mask, axis=-1).repeat(3, axis=-1)
        
        # Apply mask: result = frame * mask + black * (1 - mask)
        # Since black = 0, this simplifies to: result = frame * mask
        result = (frame.astype(np.float32) * mask_3ch).astype(np.uint8)
        
        return result
    
    def _render_transparent(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Hard isolation with transparent background (BGRA output).
        
        Alpha channel:
            mask == 1: alpha = 255 (fully opaque)
            mask == 0: alpha = 0 (fully transparent)
        """
        h, w = frame.shape[:2]
        
        # Create BGRA frame
        result = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Copy BGR channels
        result[:, :, :3] = frame
        
        # Set alpha channel from mask
        result[:, :, 3] = (mask * 255).astype(np.uint8)
        
        return result
    
    def render_with_background(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        background_color: tuple = (0, 0, 0),
    ) -> np.ndarray:
        """
        Isolation with custom background color.
        
        Args:
            frame: BGR image
            mask: Binary mask (H×W float32)
            background_color: (B, G, R) tuple for background
        
        Returns:
            Isolated frame with colored background (BGR)
        """
        h, w = frame.shape[:2]
        
        # Ensure mask matches frame
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Create background
        background = np.full_like(frame, background_color, dtype=np.uint8)
        
        # Expand mask to 3 channels
        mask_3ch = np.expand_dims(mask, axis=-1).repeat(3, axis=-1)
        
        # Alpha blend: result = frame * mask + background * (1 - mask)
        result = (
            frame.astype(np.float32) * mask_3ch +
            background.astype(np.float32) * (1.0 - mask_3ch)
        ).astype(np.uint8)
        
        return result
    
    def render_side_by_side(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Debug view: Original | Mask | Isolated side-by-side.
        
        Returns:
            Wide frame (H×(3W)×3) showing all three views
        """
        h, w = frame.shape[:2]
        
        # Resize if too large
        max_width = 800
        if w > max_width:
            scale = max_width / w
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            mask = cv2.resize(mask, (new_w, new_h))
            h, w = new_h, new_w
        
        # Create views
        original = frame.copy()
        
        # Mask visualization (grayscale → BGR)
        mask_vis = (mask * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        
        # Isolated view
        isolated = self.render(frame, mask)
        
        # Concatenate horizontally
        combined = np.hstack([original, mask_bgr, isolated])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Mask", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Isolated", (2*w + 10, 30), font, 0.7, (255, 255, 255), 2)
        
        return combined


def create_isolation_renderer(background_mode: str = 'black') -> SubjectIsolationRenderer:
    """
    Factory function to create subject isolation renderer.
    
    Args:
        background_mode: 'black' or 'transparent'
    
    Returns:
        SubjectIsolationRenderer instance
    """
    return SubjectIsolationRenderer(background_mode=background_mode)
