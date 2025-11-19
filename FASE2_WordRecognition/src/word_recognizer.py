"""
Word recognizer module - combines segmentation and character recognition.

Orchestrates the complete word recognition pipeline.

Author: Senior ML Engineer
Date: 2025
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Add FASE1 src to path for InferenceEngine
FASE1_SRC = Path(__file__).parent.parent.parent / "FASE1_SingleCharacterRecognition" / "src"
sys.path.insert(0, str(FASE1_SRC))

from inference_engine import InferenceEngine

# Import FASE2 modules from src
from src.config import WORD_RECOGNITION_CONFIG, FASE1_MODEL_PATH, FASE1_PREPROCESSOR_PATH
from src.image_segmenter import ImageSegmenter
from src.logger import LoggerMixin


class WordRecognizer(LoggerMixin):
    """
    High-level word recognition system.
    
    Combines:
    1. Image segmentation (character extraction)
    2. Character recognition (Phase 1 model)
    3. Word assembly
    
    Attributes:
        segmenter: ImageSegmenter instance
        char_classifier: InferenceEngine from Phase 1
        config: Word recognition configuration
    """
    
    def __init__(self):
        """Initialize the word recognizer."""
        self._setup_logger()
        self.config = WORD_RECOGNITION_CONFIG
        
        # Initialize components
        self.segmenter = ImageSegmenter()
        self.char_classifier: Optional[InferenceEngine] = None
        
        self.logger.info("WordRecognizer initialized")
    
    def load_model(self) -> None:
        """
        Load the Phase 1 character classifier.
        
        Raises:
            FileNotFoundError: If Phase 1 model not found
        """
        self.logger.info("Loading Phase 1 character classifier...")
        
        if not FASE1_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Phase 1 model not found: {FASE1_MODEL_PATH}\n"
                f"Please train Phase 1 first by running: "
                f"python ../FASE1_SingleCharacterRecognition/main.py"
            )
        
        try:
            self.char_classifier = InferenceEngine(
                model_path=FASE1_MODEL_PATH,
                preprocessor_path=FASE1_PREPROCESSOR_PATH
            )
            self.char_classifier.load()
            
            self.logger.info("Phase 1 model loaded successfully")
            self.logger.info(f"Model info: {self.char_classifier.get_model_info()}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Phase 1 model: {str(e)}")
            raise
    
    def recognize_word(
        self,
        image: np.ndarray,
        image_id: Optional[str] = None,
        return_details: bool = False
    ) -> str:
        """
        Recognize a word from an image.
        
        Args:
            image: Input word image (grayscale or RGB)
            image_id: Optional identifier for debugging
            return_details: If True, return detailed recognition info
        
        Returns:
            Recognized word as string (or tuple if return_details=True)
        
        Raises:
            ValueError: If model not loaded
        """
        if self.char_classifier is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.logger.info(f"Recognizing word (ID: {image_id or 'N/A'})")
        
        # Step 1: Segment image into characters
        char_images = self.segmenter.segment_word(image, image_id=image_id)
        
        if len(char_images) == 0:
            self.logger.warning("No characters segmented")
            if return_details:
                return "", [], []
            return ""
        
        # Step 2: Recognize each character
        letters, confidences = self._recognize_characters(char_images)
        
        # Step 3: Assemble word
        word = self._assemble_word(letters, confidences)
        
        self.logger.info(f"Recognized word: '{word}'")
        
        if return_details:
            return word, letters, confidences
        return word
    
    def _recognize_characters(
        self,
        char_images: List[np.ndarray]
    ) -> Tuple[List[str], List[float]]:
        """
        Recognize individual characters.
        
        Args:
            char_images: List of character images (28x28)
        
        Returns:
            Tuple of (letters, confidences)
        """
        letters = []
        confidences = []
        
        for i, char_image in enumerate(char_images):
            # Flatten for prediction
            char_flat = char_image.flatten().astype(np.float32)
            
            # Predict
            letter, confidence = self.char_classifier.predict_single(
                char_flat,
                return_confidence=True
            )
            
            # Apply confidence threshold if configured
            if self.config["use_confidence_threshold"]:
                min_conf = self.config["min_confidence"]
                if confidence is not None and confidence < min_conf:
                    letter = self.config["unknown_char_placeholder"]
                    self.logger.warning(
                        f"Char {i}: Low confidence {confidence:.2f} < {min_conf}"
                    )
            
            letters.append(letter)
            confidences.append(confidence if confidence is not None else 0.0)
            
            conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
            self.logger.debug(
                f"Char {i}: '{letter}' (confidence: {conf_str})"
            )
        
        return letters, confidences
    
    def _assemble_word(
        self,
        letters: List[str],
        confidences: List[float]
    ) -> str:
        """
        Assemble recognized letters into a word.
        
        Args:
            letters: List of recognized letters
            confidences: List of confidence scores
        
        Returns:
            Assembled word string
        """
        # Simple left-to-right concatenation
        word = "".join(letters)
        
        # Apply post-processing
        if self.config["force_uppercase"]:
            word = word.upper()
        
        return word
    
    def recognize_batch(
        self,
        images: List[np.ndarray],
        image_ids: Optional[List[str]] = None
    ) -> List[Dict[str, any]]:
        """
        Recognize multiple word images.
        
        Args:
            images: List of word images
            image_ids: Optional list of image identifiers
        
        Returns:
            List of dictionaries with recognition results
        """
        if image_ids is None:
            image_ids = [f"image_{i:04d}" for i in range(len(images))]
        
        results = []
        
        for image, image_id in zip(images, image_ids):
            try:
                word, letters, confidences = self.recognize_word(
                    image,
                    image_id=image_id,
                    return_details=True
                )
                
                results.append({
                    "image_id": image_id,
                    "word": word,
                    "letters": letters,
                    "confidences": confidences,
                    "n_chars": len(letters),
                    "avg_confidence": np.mean(confidences) if confidences else 0.0,
                    "status": "success"
                })
                
            except Exception as e:
                self.logger.error(f"Failed to recognize {image_id}: {str(e)}")
                results.append({
                    "image_id": image_id,
                    "word": "",
                    "status": "failed",
                    "error": str(e)
                })
        
        return results
    
    def get_detailed_prediction(
        self,
        image: np.ndarray,
        top_k: int = 3
    ) -> Dict[str, any]:
        """
        Get detailed prediction with top-k alternatives for each character.
        
        Args:
            image: Word image
            top_k: Number of top predictions per character
        
        Returns:
            Dictionary with detailed predictions
        """
        if self.char_classifier is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Segment
        char_images = self.segmenter.segment_word(image)
        
        if len(char_images) == 0:
            return {"word": "", "characters": [], "alternatives": []}
        
        # Get top-k predictions for each character
        char_details = []
        
        for i, char_image in enumerate(char_images):
            char_flat = char_image.flatten().astype(np.float32)
            
            # Get top-k predictions
            top_k_preds = self.char_classifier.predict_with_top_k(char_flat, k=top_k)
            
            char_details.append({
                "char_index": i,
                "top_prediction": top_k_preds[0][0],
                "confidence": top_k_preds[0][1],
                "alternatives": top_k_preds[1:] if len(top_k_preds) > 1 else []
            })
        
        # Assemble word from top predictions
        word = "".join([cd["top_prediction"] for cd in char_details])
        
        if self.config["force_uppercase"]:
            word = word.upper()
        
        return {
            "word": word,
            "n_characters": len(char_details),
            "characters": char_details
        }
