"""
Configuration module for EMNIST Letter Recognition System.

This module centralizes all configuration parameters including paths,
model hyperparameters, and preprocessing settings.

Author: Senior ML Engineer
Date: 2025
"""

from pathlib import Path
from typing import Dict, Any


# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "RAIA_Project-main"

TRAIN_DATA_PATH = DATA_DIR / "emnist-letters-train.csv"
TEST_DATA_PATH = DATA_DIR / "emnist-letters-test.csv"
MAPPING_PATH = DATA_DIR / "emnist-letters-mapping.txt"

# Output directories
MODELS_DIR = Path(__file__).parent.parent / "models"
LOGS_DIR = Path(__file__).parent.parent / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA CONFIGURATION
# ============================================================================
IMAGE_SIZE = (28, 28)  # EMNIST images are 28x28 pixels
N_PIXELS = IMAGE_SIZE[0] * IMAGE_SIZE[1]  # 784
N_CLASSES = 26  # 26 letters (A-Z)


# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================
PREPROCESSING_CONFIG: Dict[str, Any] = {
    # Image transformations
    "rotation_k": 3,  # Rotate 270 degrees (k=3 means 3*90°)
    "flip_lr": True,   # Flip left-right to correct EMNIST orientation
    
    # Normalization
    "normalize": True,
    "normalization_method": "standard",  # Options: "standard", "minmax"
    
    # Feature extraction
    "use_hog": True,  # Use Histogram of Oriented Gradients
    "hog_params": {
        "orientations": 9,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2),
        "transform_sqrt": True,
        "feature_vector": True,
    }
}


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_CONFIG: Dict[str, Any] = {
    "model_type": "svm",  # Options: "svm", "mlp", "knn"
    
    # SVM Hyperparameters (default)
    "svm": {
        "C": 10.0,
        "kernel": "rbf",
        "gamma": "scale",
        "cache_size": 1000,
        "verbose": False,
        "random_state": 42,
    },
    
    # Alternative: MLP (uncomment to use)
    "mlp": {
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": 256,
        "learning_rate": "adaptive",
        "max_iter": 100,
        "random_state": 42,
        "verbose": True,
    },
    
    # Alternative: KNN (uncomment to use)
    "knn": {
        "n_neighbors": 5,
        "weights": "distance",
        "algorithm": "auto",
        "n_jobs": -1,
    }
}


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG: Dict[str, Any] = {
    # Data sampling (for faster experiments, set to None for full dataset)
    "train_sample_size": None,  # Example: 10000 for quick tests
    "test_sample_size": None,   # Example: 2000 for quick tests
    "random_state": 42,
    
    # Validation
    "use_validation_split": True,
    "validation_size": 0.2,
    
    # Model persistence
    "save_model": True,
    "model_filename": "emnist_letter_classifier.pkl",
    "scaler_filename": "feature_scaler.pkl",
}


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG: Dict[str, Any] = {
    "level": "INFO",  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_to_file": True,
    "log_filename": "training.log",
}


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
EVALUATION_CONFIG: Dict[str, Any] = {
    "metrics": ["accuracy", "precision", "recall", "f1"],
    "average": "weighted",  # For multi-class metrics
    "generate_confusion_matrix": True,
    "generate_classification_report": True,
    "top_k_errors": 10,  # Show top k misclassified classes
}
