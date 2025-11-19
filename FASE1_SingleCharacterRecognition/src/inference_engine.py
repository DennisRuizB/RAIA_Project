"""
Inference engine module for single character prediction.

Provides a clean interface for making predictions on new images.

Author: Senior ML Engineer
Date: 2025
"""

import numpy as np
import pickle
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from config import MODELS_DIR, TRAINING_CONFIG, IMAGE_SIZE
from logger import LoggerMixin


class InferenceEngine(LoggerMixin):
    """
    High-level inference engine for letter recognition.
    
    Handles:
    - Model and preprocessor loading
    - Single image prediction
    - Batch prediction
    - Confidence scores
    
    Attributes:
        model: Trained classifier model
        preprocessor: Fitted image preprocessor
        label_mapping: Dictionary mapping labels to letters
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to saved model. If None, uses default from config
            preprocessor_path: Path to saved preprocessor. If None, uses default
        """
        self._setup_logger()
        
        self.model = None
        self.preprocessor = None
        self.label_mapping: Optional[Dict[int, str]] = None
        
        # Set default paths
        if model_path is None:
            model_path = MODELS_DIR / TRAINING_CONFIG["model_filename"]
        if preprocessor_path is None:
            preprocessor_path = MODELS_DIR / TRAINING_CONFIG["scaler_filename"]
        
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        
        self.logger.info("InferenceEngine initialized")
    
    def load(self) -> None:
        """
        Load the trained model and preprocessor from disk.
        
        Raises:
            FileNotFoundError: If model or preprocessor files don't exist
        """
        self.logger.info("Loading model and preprocessor...")
        
        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.logger.info(f"Model loaded from: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
        
        # Load preprocessor
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {self.preprocessor_path}")
        
        try:
            with open(self.preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            self.logger.info(f"Preprocessor loaded from: {self.preprocessor_path}")
        except Exception as e:
            self.logger.error(f"Failed to load preprocessor: {str(e)}")
            raise
        
        # Load label mapping (from data_loader)
        try:
            from data_loader import EMNISTDataLoader
            loader = EMNISTDataLoader()
            self.label_mapping = loader.label_mapping
            self.logger.info(f"Label mapping loaded: {len(self.label_mapping)} classes")
        except Exception as e:
            self.logger.error(f"Failed to load label mapping: {str(e)}")
            raise
    
    def predict_single(
        self,
        image: np.ndarray,
        return_confidence: bool = False
    ) -> Tuple[str, Optional[float]]:
        """
        Predict a single letter from an image.
        
        Args:
            image: Image array of shape (28, 28) or (784,)
            return_confidence: Whether to return confidence score
        
        Returns:
            Tuple of (predicted_letter, confidence) or just predicted_letter
        
        Raises:
            ValueError: If model hasn't been loaded
            ValueError: If image has invalid shape
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model and preprocessor must be loaded first. Call load() method.")
        
        # Validate and reshape image
        image = self._prepare_image(image)
        
        # Preprocess
        X = self.preprocessor.preprocess_single_image(image, is_flat=True)
        
        # Predict
        label = self.model.predict(X)[0]
        letter = self.label_mapping[label]
        
        # Get confidence if requested
        confidence = None
        if return_confidence:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)[0]
                confidence = float(np.max(proba))
        
        self.logger.debug(f"Predicted: {letter} (label={label}, confidence={confidence})")
        
        return letter, confidence
    
    def predict_batch(
        self,
        images: np.ndarray,
        return_confidence: bool = False
    ) -> Tuple[List[str], Optional[List[float]]]:
        """
        Predict multiple letters from a batch of images.
        
        Args:
            images: Image array of shape (n_samples, 784) or (n_samples, 28, 28)
            return_confidence: Whether to return confidence scores
        
        Returns:
            Tuple of (predicted_letters, confidences) or just predicted_letters
        
        Raises:
            ValueError: If model hasn't been loaded
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model and preprocessor must be loaded first. Call load() method.")
        
        # Ensure images are flattened
        if images.ndim == 3:
            images = images.reshape(images.shape[0], -1)
        
        if images.shape[1] != IMAGE_SIZE[0] * IMAGE_SIZE[1]:
            raise ValueError(
                f"Expected images of size {IMAGE_SIZE[0] * IMAGE_SIZE[1]}, "
                f"got {images.shape[1]}"
            )
        
        # Preprocess
        X = self.preprocessor.transform(images)
        
        # Predict
        labels = self.model.predict(X)
        letters = [self.label_mapping[label] for label in labels]
        
        # Get confidences if requested
        confidences = None
        if return_confidence:
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(X)
                confidences = [float(np.max(proba)) for proba in probas]
            else:
                confidences = [None] * len(letters)
        
        self.logger.info(f"Predicted {len(letters)} letters")
        
        return letters, confidences
    
    def predict_with_top_k(
        self,
        image: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Predict top-k most likely letters with probabilities.
        
        Args:
            image: Image array of shape (28, 28) or (784,)
            k: Number of top predictions to return
        
        Returns:
            List of tuples (letter, probability) sorted by probability
        
        Raises:
            ValueError: If model doesn't support probability predictions
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model and preprocessor must be loaded first. Call load() method.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model doesn't support probability predictions")
        
        # Prepare image
        image = self._prepare_image(image)
        X = self.preprocessor.preprocess_single_image(image, is_flat=True)
        
        # Get probabilities
        probas = self.model.predict_proba(X)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(probas)[-k:][::-1]
        
        # Map to letters and probabilities
        top_k_predictions = [
            (self.label_mapping[idx + 1], float(probas[idx]))  # +1 because labels are 1-26
            for idx in top_k_indices
        ]
        
        return top_k_predictions
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Validate and prepare image for prediction.
        
        Args:
            image: Input image array
        
        Returns:
            Flattened image array of shape (784,)
        
        Raises:
            ValueError: If image has invalid shape
        """
        # Handle 2D image
        if image.ndim == 2:
            if image.shape != IMAGE_SIZE:
                raise ValueError(
                    f"Expected 2D image of shape {IMAGE_SIZE}, got {image.shape}"
                )
            image = image.flatten()
        
        # Handle 1D image
        elif image.ndim == 1:
            expected_size = IMAGE_SIZE[0] * IMAGE_SIZE[1]
            if image.shape[0] != expected_size:
                raise ValueError(
                    f"Expected 1D image of size {expected_size}, got {image.shape[0]}"
                )
        
        else:
            raise ValueError(f"Expected 1D or 2D image, got {image.ndim}D")
        
        return image
    
    def is_ready(self) -> bool:
        """
        Check if the engine is ready for inference.
        
        Returns:
            True if model and preprocessor are loaded
        """
        return (
            self.model is not None and 
            self.preprocessor is not None and 
            self.label_mapping is not None
        )
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_ready():
            return {"status": "Not loaded"}
        
        return {
            "status": "Ready",
            "model_type": type(self.model).__name__,
            "n_classes": str(len(self.label_mapping)),
            "model_path": str(self.model_path),
            "preprocessor_path": str(self.preprocessor_path)
        }
