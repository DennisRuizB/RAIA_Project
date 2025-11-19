"""
Model training module for EMNIST letter recognition.

Handles model creation, training, and persistence.

Author: Senior ML Engineer
Date: 2025
"""

import numpy as np
import pickle
from typing import Any, Optional, Tuple
from pathlib import Path
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from config import MODEL_CONFIG, MODELS_DIR, TRAINING_CONFIG
from logger import LoggerMixin


class ModelTrainer(LoggerMixin):
    """
    Trains and manages ML models for letter recognition.
    
    Supports multiple model architectures:
    - SVM (Support Vector Machine) - Default, best for image classification
    - MLP (Multi-Layer Perceptron) - Neural network alternative
    - KNN (K-Nearest Neighbors) - Simple baseline
    
    Attributes:
        model: The trained scikit-learn model
        model_type: String identifier of model architecture
    """
    
    def __init__(self, model_type: Optional[str] = None):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Model architecture to use. Options: "svm", "mlp", "knn"
                       If None, uses config default
        """
        self._setup_logger()
        self.model_type = model_type or MODEL_CONFIG["model_type"]
        self.model: Optional[Any] = None
        
        self.logger.info(f"ModelTrainer initialized with model_type='{self.model_type}'")
    
    def create_model(self) -> Any:
        """
        Create and return a model instance based on configuration.
        
        Returns:
            Untrained scikit-learn model instance
        
        Raises:
            ValueError: If model_type is not supported
        """
        self.logger.info(f"Creating {self.model_type.upper()} model...")
        
        if self.model_type == "svm":
            model = SVC(**MODEL_CONFIG["svm"])
            self.logger.info(f"SVM Config: C={MODEL_CONFIG['svm']['C']}, "
                           f"kernel={MODEL_CONFIG['svm']['kernel']}")
        
        elif self.model_type == "mlp":
            model = MLPClassifier(**MODEL_CONFIG["mlp"])
            self.logger.info(f"MLP Config: hidden_layers={MODEL_CONFIG['mlp']['hidden_layer_sizes']}, "
                           f"max_iter={MODEL_CONFIG['mlp']['max_iter']}")
        
        elif self.model_type == "knn":
            model = KNeighborsClassifier(**MODEL_CONFIG["knn"])
            self.logger.info(f"KNN Config: n_neighbors={MODEL_CONFIG['knn']['n_neighbors']}")
        
        else:
            raise ValueError(
                f"Unsupported model_type: {self.model_type}. "
                f"Choose from: svm, mlp, knn"
            )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Train the model on provided data.
        
        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training labels of shape (n_samples,)
            X_val: Optional validation features
            y_val: Optional validation labels
        
        Raises:
            ValueError: If training data is invalid
        """
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have same number of samples")
        
        self.logger.info("=" * 70)
        self.logger.info("STARTING MODEL TRAINING")
        self.logger.info("=" * 70)
        self.logger.info(f"Training samples: {X_train.shape[0]}")
        self.logger.info(f"Features dimension: {X_train.shape[1]}")
        self.logger.info(f"Unique classes: {len(np.unique(y_train))}")
        
        try:
            # Create model
            self.model = self.create_model()
            
            # Train model
            self.logger.info("Fitting model... (this may take several minutes)")
            self.model.fit(X_train, y_train)
            self.logger.info("Model training completed successfully!")
            
            # Training accuracy
            train_score = self.model.score(X_train, y_train)
            self.logger.info(f"Training Accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
            
            # Validation accuracy if provided
            if X_val is not None and y_val is not None:
                val_score = self.model.score(X_val, y_val)
                self.logger.info(f"Validation Accuracy: {val_score:.4f} ({val_score*100:.2f}%)")
            
            self.logger.info("=" * 70)
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
        
        Returns:
            Predicted labels of shape (n_samples,)
        
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        self.logger.debug(f"Generated {len(predictions)} predictions")
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
        
        Returns:
            Probability array of shape (n_samples, n_classes)
        
        Raises:
            ValueError: If model doesn't support probability predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.model_type} model doesn't support probability predictions")
        
        probabilities = self.model.predict_proba(X)
        return probabilities
    
    def save_model(self, filename: Optional[str] = None) -> Path:
        """
        Save the trained model to disk.
        
        Args:
            filename: Custom filename. If None, uses config default
        
        Returns:
            Path where model was saved
        
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        if filename is None:
            filename = TRAINING_CONFIG["model_filename"]
        
        model_path = MODELS_DIR / filename
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            self.logger.info(f"Model saved successfully to: {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load_model(self, filename: Optional[str] = None) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filename: Model filename. If None, uses config default
        
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if filename is None:
            filename = TRAINING_CONFIG["model_filename"]
        
        model_path = MODELS_DIR / filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.logger.info(f"Model loaded successfully from: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def create_validation_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_size: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split training data into train and validation sets.
        
        Args:
            X: Feature array
            y: Label array
            validation_size: Fraction for validation (0.0 to 1.0)
        
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        if validation_size is None:
            validation_size = TRAINING_CONFIG["validation_size"]
        
        self.logger.info(f"Creating validation split: {validation_size*100:.0f}% for validation")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_size,
            random_state=TRAINING_CONFIG["random_state"],
            stratify=y
        )
        
        self.logger.info(f"Training set size: {len(X_train)}")
        self.logger.info(f"Validation set size: {len(X_val)}")
        
        return X_train, X_val, y_train, y_val
