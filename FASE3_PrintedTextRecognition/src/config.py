"""
Configuration for Printed Text Recognition System.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Image configuration (compatible with FASE1)
IMAGE_SIZE = (28, 28)
N_PIXELS = IMAGE_SIZE[0] * IMAGE_SIZE[1]  # 784
N_CLASSES = 26  # A-Z

# Preprocessing configuration (for FASE1 compatibility)
PREPROCESSING_CONFIG = {
    "rotation_k": 0,  # No rotation for printed text
    "flip_lr": False,  # No flip needed
    "normalize": True,
    "normalization_method": "standard",
    "use_hog": True,
    "hog_params": {
        "orientations": 9,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2),
        "transform_sqrt": True,
        "feature_vector": True,
    }
}

# Evaluation configuration (for FASE1 compatibility)
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1"],
    "average": "weighted",
    "generate_confusion_matrix": True,
    "generate_classification_report": True,
    "top_k_errors": 10,
}

# Logging configuration (for FASE1 compatibility)
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_to_file": True,
    "log_filename": "training.log",
    "log_dir": LOGS_DIR,
}

# Dataset Generation Config
DATASET_CONFIG = {
    "letters": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "samples_per_letter_per_variation": 100,  # 100 samples x 26 letters x ~40 variations = ~104,000 total
    
    # Fonts (instala las fuentes que tengas en Windows)
    "fonts": [
        "Arial",
        "Times New Roman",
        "Comic Sans MS",
        "Courier New",
        "Calibri",
        "Verdana",
        "Georgia",
        "Tahoma",
    ],
    
    # Font sizes
    "font_sizes": [16, 18, 20, 22, 24],
    
    # Font styles
    "styles": [
        ("normal", False, False),      # (name, bold, italic)
        ("bold", True, False),
        ("italic", False, True),
        ("bold_italic", True, True),
    ],
    
    # Image settings
    "image_size": (28, 28),
    "background_color": 255,  # White
    "text_color": 0,          # Black
    
    # Augmentations
    "apply_rotation": True,
    "rotation_range": (-5, 5),  # degrees
    
    "apply_noise": True,
    "noise_probability": 0.1,
    "noise_std": 10,
    
    "apply_blur": True,
    "blur_probability": 0.1,
    
    # Train/test split
    "test_size": 0.2,
    "random_seed": 42,
}

# Model Training Config
MODEL_CONFIG = {
    "model_type": "svm",  # "svm", "mlp", or "knn"
    
    "svm": {
        "C": 10.0,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": True,
        "random_state": 42,
    },
    
    "mlp": {
        "hidden_layer_sizes": (256, 128, 64),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 100,
        "random_state": 42,
        "early_stopping": True,
        "validation_fraction": 0.1,
    },
    
    "knn": {
        "n_neighbors": 5,
        "weights": "distance",
        "algorithm": "auto",
    },
}

# Preprocessing Config (duplicated but kept for clarity)
# This is already defined above for FASE1 compatibility

# Training Config
TRAINING_CONFIG = {
    "validation_size": 0.2,
    "random_state": 42,
    "save_model": True,
    "model_filename": "printed_letter_classifier.pkl",
    "preprocessor_filename": "printed_feature_scaler.pkl",
}
