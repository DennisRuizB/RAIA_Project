"""
Image segmentation module for extracting individual letters from word images.

Implements projection-based segmentation to separate characters.

Author: Senior ML Engineer
Date: 2025
"""

import numpy as np
from typing import List, Tuple, Optional
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage import filters
from scipy import ndimage

from src.config import SEGMENTATION_CONFIG, SEGMENTED_DIR
from src.logger import LoggerMixin


class ImageSegmenter(LoggerMixin):
    """
    Segments word images into individual character images.
    
    Uses vertical projection profile analysis to detect character boundaries.
    Each segmented character is normalized to 28x28 for EMNIST compatibility.
    
    Attributes:
        config: Segmentation configuration dictionary
    """
    
    def __init__(self):
        """Initialize the image segmenter."""
        self._setup_logger()
        self.config = SEGMENTATION_CONFIG
        self.logger.info("ImageSegmenter initialized")
    
    def segment_word(
        self,
        image: np.ndarray,
        image_id: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Segment a word image into individual character images.
        
        Args:
            image: Input image (grayscale or RGB)
            image_id: Optional identifier for debugging/saving
        
        Returns:
            List of character images, each of shape (28, 28)
        
        Raises:
            ValueError: If image is invalid
        """
        if image.size == 0:
            raise ValueError("Empty image provided")
        
        self.logger.info(f"Segmenting image (ID: {image_id or 'N/A'})")
        
        # Step 1: Preprocess image
        binary_image = self._preprocess_image(image)
        
        # Step 2: Find character boundaries
        boundaries = self._find_character_boundaries(binary_image)
        
        if len(boundaries) == 0:
            self.logger.warning("No characters detected in image")
            return []
        
        self.logger.info(f"Detected {len(boundaries)} character boundaries")
        
        # Step 3: Extract and normalize characters
        characters = []
        for i, (x_start, x_end) in enumerate(boundaries):
            char_image = self._extract_character(
                binary_image,
                x_start,
                x_end,
                char_index=i
            )
            
            # Normalize to 28x28
            normalized_char = self._normalize_character(char_image)
            characters.append(normalized_char)
            
            # Save individual character if configured
            if self.config["save_individual_chars"] and image_id:
                self._save_character(normalized_char, image_id, i)
        
        return characters
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for segmentation.
        
        Steps:
        1. Convert to grayscale if needed
        2. Binarize (Otsu's method)
        3. Invert (ensure text is white on black)
        4. Apply morphological operations
        
        Args:
            image: Input image
        
        Returns:
            Binary image (text=white, background=black)
        """
        # Convert to grayscale
        if image.ndim == 3:
            gray = rgb2gray(image)
        else:
            gray = image.astype(np.float32) / 255.0 if image.max() > 1 else image
        
        # Binarize
        method = self.config["binarization_method"]
        
        if method == "otsu":
            threshold = filters.threshold_otsu(gray)
            binary = gray < threshold  # Text is True (white)
        elif method == "adaptive":
            # Use simple mean for adaptive threshold
            kernel_size = 15
            local_mean = ndimage.uniform_filter(gray, size=kernel_size)
            binary = gray < local_mean - 0.05
        else:  # Fixed threshold
            threshold = self.config["threshold_value"] / 255.0
            binary = gray < threshold
        
        # Apply morphological operations if configured
        if self.config["apply_morphology"]:
            kernel_size = self.config["morph_kernel_size"]
            kernel = np.ones(kernel_size, dtype=np.uint8)
            
            for operation in self.config["morph_operations"]:
                if operation == "dilate":
                    binary = ndimage.binary_dilation(binary, structure=kernel)
                elif operation == "erode":
                    binary = ndimage.binary_erosion(binary, structure=kernel)
        
        return binary.astype(np.uint8) * 255
    
    def _find_character_boundaries(
        self,
        binary_image: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Find character boundaries using vertical projection profile.
        
        The projection profile counts white pixels in each column.
        Character boundaries are detected where the profile drops below threshold.
        
        Args:
            binary_image: Binary image (255=text, 0=background)
        
        Returns:
            List of tuples (x_start, x_end) for each character
        """
        # Calculate vertical projection (sum along height axis)
        projection = np.sum(binary_image > 0, axis=0)
        
        # Normalize
        if projection.max() > 0:
            projection = projection / projection.max()
        
        # Threshold to detect non-empty columns
        threshold = self.config["projection_threshold"]
        is_text = projection > threshold
        
        # Find transitions (boundaries)
        boundaries = []
        in_char = False
        start = 0
        
        for i, is_text_col in enumerate(is_text):
            if is_text_col and not in_char:
                # Start of character
                start = i
                in_char = True
            elif not is_text_col and in_char:
                # End of character
                end = i
                width = end - start
                
                # Validate width
                if (self.config["min_char_width"] <= width <= 
                    self.config["max_char_width"]):
                    boundaries.append((start, end))
                
                in_char = False
        
        # Handle case where character extends to image edge
        if in_char:
            end = len(is_text)
            width = end - start
            if (self.config["min_char_width"] <= width <= 
                self.config["max_char_width"]):
                boundaries.append((start, end))
        
        return boundaries
    
    def _extract_character(
        self,
        binary_image: np.ndarray,
        x_start: int,
        x_end: int,
        char_index: int
    ) -> np.ndarray:
        """
        Extract a single character from the binary image.
        
        Args:
            binary_image: Binary image
            x_start: Starting x coordinate
            x_end: Ending x coordinate
            char_index: Index of character (for logging)
        
        Returns:
            Extracted character image
        """
        # Add padding
        padding = self.config["padding"]
        x_start = max(0, x_start - padding)
        x_end = min(binary_image.shape[1], x_end + padding)
        
        # Extract character region
        char_image = binary_image[:, x_start:x_end]
        
        # Crop vertically to remove excess whitespace
        row_projection = np.sum(char_image > 0, axis=1)
        nonzero_rows = np.where(row_projection > 0)[0]
        
        if len(nonzero_rows) > 0:
            y_start = max(0, nonzero_rows[0] - padding)
            y_end = min(char_image.shape[0], nonzero_rows[-1] + 1 + padding)
            char_image = char_image[y_start:y_end, :]
        
        return char_image
    
    def _normalize_character(self, char_image: np.ndarray) -> np.ndarray:
        """
        Normalize character to 28x28 EMNIST format.
        
        Preserves aspect ratio by:
        1. Scaling to fit within 28x28
        2. Centering in 28x28 canvas
        
        Args:
            char_image: Character image of arbitrary size
        
        Returns:
            Normalized 28x28 character image
        """
        target_size = 28
        
        if char_image.size == 0:
            return np.zeros((target_size, target_size), dtype=np.float32)
        
        h, w = char_image.shape
        
        # Calculate scale to fit in 28x28 while preserving aspect ratio
        scale = min(target_size / h, target_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize using zoom (nearest neighbor for binary images)
        if new_h > 0 and new_w > 0:
            scaled = ndimage.zoom(
                char_image,
                (new_h / h, new_w / w),
                order=1  # Bilinear interpolation
            )
        else:
            scaled = char_image
        
        # Create 28x28 canvas and center the character
        canvas = np.zeros((target_size, target_size), dtype=np.float32)
        
        # Calculate centering offsets
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        # Place character in center
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = scaled
        
        # Invert to match EMNIST format (black text on white background)
        canvas = 255 - canvas
        
        return canvas
    
    def _save_character(
        self,
        char_image: np.ndarray,
        image_id: str,
        char_index: int
    ) -> None:
        """
        Save individual character image for debugging.
        
        Args:
            char_image: Character image
            image_id: Image identifier
            char_index: Character index in word
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            filename = SEGMENTED_DIR / f"{image_id}_char_{char_index:02d}.png"
            plt.imsave(filename, char_image, cmap='gray')
            self.logger.debug(f"Saved character to: {filename}")
            
        except ImportError:
            self.logger.debug("Matplotlib not available for saving characters")
    
    def visualize_segmentation(
        self,
        image: np.ndarray,
        boundaries: List[Tuple[int, int]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize segmentation with boundary lines.
        
        Args:
            image: Original image
            boundaries: List of character boundaries
            save_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Show image
            if image.ndim == 3:
                ax.imshow(image)
            else:
                ax.imshow(image, cmap='gray')
            
            # Draw boundary lines
            for x_start, x_end in boundaries:
                ax.axvline(x=x_start, color='r', linestyle='--', linewidth=1)
                ax.axvline(x=x_end, color='b', linestyle='--', linewidth=1)
            
            ax.set_title(f"Character Segmentation ({len(boundaries)} characters)")
            ax.axis('off')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                self.logger.info(f"Segmentation visualization saved to: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for visualization")
