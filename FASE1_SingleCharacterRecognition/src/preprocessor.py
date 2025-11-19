"""
Preprocessing module for EMNIST image data.

Handles image transformations, normalization, and feature extraction.

Author: Senior ML Engineer
Date: 2025
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage.feature import hog
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray

from config import PREPROCESSING_CONFIG, IMAGE_SIZE
from logger import LoggerMixin


class ImagePreprocessor(LoggerMixin):
    """
    Handles preprocessing of EMNIST letter images.
    
    Preprocessing pipeline:
    1. Reshape flat pixel arrays to 28x28 images
    2. Correct EMNIST orientation (rotation + flip)
    3. Optional: Extract HOG features
    4. Normalize features
    
    Attributes:
        scaler: Fitted scaler for feature normalization
        use_hog: Whether to use HOG feature extraction
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self._setup_logger()
        self.scaler: Optional[StandardScaler] = None
        self.use_hog: bool = PREPROCESSING_CONFIG["use_hog"]
        self.hog_params: dict = PREPROCESSING_CONFIG["hog_params"]
        
        # Initialize scaler based on config
        normalization_method = PREPROCESSING_CONFIG["normalization_method"]
        if normalization_method == "standard":
            self.scaler = StandardScaler()
        elif normalization_method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.logger.warning(f"Unknown normalization method: {normalization_method}")
            self.scaler = StandardScaler()
        
        self.logger.info(f"Preprocessor initialized with HOG={self.use_hog}")
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the preprocessor on training data and transform it.
        
        Args:
            X: Raw feature array of shape (n_samples, 784)
        
        Returns:
            Preprocessed feature array of shape (n_samples, n_features)
            where n_features depends on whether HOG is used
        """
        self.logger.info("Fitting preprocessor on training data...")
        
        # Apply transformations
        X_processed = self._transform_images(X)
        
        # Extract features
        if self.use_hog:
            X_features = self._extract_hog_features(X_processed)
        else:
            # Flatten images back to vectors
            X_features = X_processed.reshape(X_processed.shape[0], -1)
        
        # Fit and transform with scaler
        if PREPROCESSING_CONFIG["normalize"]:
            X_features = self.scaler.fit_transform(X_features)
            self.logger.info("Scaler fitted and applied")
        
        self.logger.info(f"Preprocessed training data shape: {X_features.shape}")
        return X_features
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor (for test/inference).
        
        Args:
            X: Raw feature array of shape (n_samples, 784)
        
        Returns:
            Preprocessed feature array of shape (n_samples, n_features)
        
        Raises:
            ValueError: If preprocessor hasn't been fitted yet
        """
        if PREPROCESSING_CONFIG["normalize"] and self.scaler is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        self.logger.info("Transforming data...")
        
        # Apply transformations
        X_processed = self._transform_images(X)
        
        # Extract features
        if self.use_hog:
            X_features = self._extract_hog_features(X_processed)
        else:
            X_features = X_processed.reshape(X_processed.shape[0], -1)
        
        # Transform with fitted scaler
        if PREPROCESSING_CONFIG["normalize"]:
            X_features = self.scaler.transform(X_features)
        
        self.logger.info(f"Transformed data shape: {X_features.shape}")
        return X_features
    
    def _transform_images(self, X: np.ndarray) -> np.ndarray:
        """
        Apply image transformations to correct EMNIST orientation.
        
        EMNIST images need to be rotated and flipped to appear correctly.
        
        Args:
            X: Raw pixel array of shape (n_samples, 784)
        
        Returns:
            Transformed image array of shape (n_samples, 28, 28)
        """
        n_samples = X.shape[0]
        
        # Reshape to images
        images = X.reshape(n_samples, IMAGE_SIZE[0], IMAGE_SIZE[1])
        
        # Apply rotation
        rotation_k = PREPROCESSING_CONFIG["rotation_k"]
        images = np.rot90(images, k=rotation_k, axes=(1, 2))
        
        # Apply horizontal flip
        if PREPROCESSING_CONFIG["flip_lr"]:
            images = np.fliplr(images)
        
        return images
    
    def _extract_hog_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract HOG (Histogram of Oriented Gradients) features.
        
        HOG is a robust feature descriptor for image recognition that
        captures edge and gradient structure.
        
        Args:
            images: Image array of shape (n_samples, 28, 28)
        
        Returns:
            HOG feature array of shape (n_samples, n_hog_features)
        """
        self.logger.info("Extracting HOG features...")
        
        n_samples = images.shape[0]
        hog_features_list = []
        
        # Process each image
        for i in range(n_samples):
            img = images[i]
            
            # Extract HOG features
            # Note: multichannel=False since images are grayscale
            features = hog(
                img,
                orientations=self.hog_params["orientations"],
                pixels_per_cell=self.hog_params["pixels_per_cell"],
                cells_per_block=self.hog_params["cells_per_block"],
                transform_sqrt=self.hog_params["transform_sqrt"],
                feature_vector=self.hog_params["feature_vector"],
                visualize=False
            )
            hog_features_list.append(features)
            
            # Progress logging
            if (i + 1) % 10000 == 0:
                self.logger.debug(f"Processed {i + 1}/{n_samples} images")
        
        hog_features = np.array(hog_features_list)
        self.logger.info(f"HOG features extracted: {hog_features.shape}")
        
        return hog_features
    
    def preprocess_single_image(
        self,
        image: np.ndarray,
        is_flat: bool = True
    ) -> np.ndarray:
        """
        Preprocess a single image for inference.
        
        Args:
            image: Single image as flat array (784,) or 2D array (28, 28)
            is_flat: Whether the image is already flattened
        
        Returns:
            Preprocessed feature vector of shape (1, n_features)
        
        Raises:
            ValueError: If preprocessor hasn't been fitted yet
        """
        if PREPROCESSING_CONFIG["normalize"] and self.scaler is None:
            raise ValueError("Preprocessor must be fitted before inference")
        
        # Ensure image is in correct format
        if is_flat:
            if image.shape != (IMAGE_SIZE[0] * IMAGE_SIZE[1],):
                raise ValueError(f"Expected flat image of size 784, got {image.shape}")
            image = image.reshape(1, -1)
        else:
            if image.shape != IMAGE_SIZE:
                raise ValueError(f"Expected image size {IMAGE_SIZE}, got {image.shape}")
            image = image.flatten().reshape(1, -1)
        
        # Transform
        return self.transform(image)
    
    def get_preprocessed_image_shape(self) -> Tuple[int, int]:
        """
        Get the shape of preprocessed images.
        
        Returns:
            Tuple of (height, width) after transformation
        """
        return IMAGE_SIZE
    
    def visualize_preprocessing(
        self,
        X: np.ndarray,
        indices: Optional[list] = None,
        n_samples: int = 5
    ) -> None:
        """
        Visualize the preprocessing steps for debugging.
        
        Args:
            X: Raw feature array
            indices: Specific indices to visualize
            n_samples: Number of samples to show if indices not provided
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("Matplotlib not available for visualization")
            return
        
        if indices is None:
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
        
        # Transform images
        images_transformed = self._transform_images(X[indices])
        
        # Create visualization
        fig, axes = plt.subplots(1, len(indices), figsize=(15, 3))
        if len(indices) == 1:
            axes = [axes]
        
        for i, (idx, ax) in enumerate(zip(indices, axes)):
            ax.imshow(images_transformed[i], cmap='gray')
            ax.axis('off')
            ax.set_title(f"Sample {idx}")
        
        plt.tight_layout()
        plt.show()
        
        self.logger.info(f"Visualized {len(indices)} preprocessed images")
