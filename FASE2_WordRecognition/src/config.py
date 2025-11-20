"""
Configuration module for Word Recognition System - Phase 2.

Centralizes all configuration for word-level handwriting recognition.

Author: Senior ML Engineer
Date: 2025
"""

from pathlib import Path
from typing import Dict, Any


# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
FASE1_DIR = BASE_DIR.parent / "FASE1_SingleCharacterRecognition"

# Phase 1 model paths (reusing trained character classifier)
# Use augmented model for better word recognition performance
FASE1_MODEL_PATH = FASE1_DIR / "models" / "emnist_letter_classifier_augmented.pkl"
FASE1_PREPROCESSOR_PATH = FASE1_DIR / "models" / "feature_scaler_augmented.pkl"

# Fallback to original model if augmented not available
if not FASE1_MODEL_PATH.exists():
    FASE1_MODEL_PATH = FASE1_DIR / "models" / "emnist_letter_classifier.pkl"
    FASE1_PREPROCESSOR_PATH = FASE1_DIR / "models" / "feature_scaler.pkl"

# Output directories
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = OUTPUT_DIR / "results"
SEGMENTED_DIR = OUTPUT_DIR / "segmented_letters"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# IMAGE SEGMENTATION CONFIGURATION
# ============================================================================
SEGMENTATION_CONFIG: Dict[str, Any] = {
    # Image preprocessing
    "resize_height": 28,  # Standard EMNIST height
    "binarization_method": "otsu",  # Options: "otsu", "adaptive", "threshold"
    "threshold_value": 127,  # Used if method="threshold"
    
    # Morphological operations
    "apply_morphology": True,
    "morph_kernel_size": (3, 3),
    "morph_operations": ["dilate", "erode"],  # Clean up noise
    
    # Character segmentation
    "segmentation_method": "projection",  # Options: "projection", "contours"
    "min_char_width": 5,   # Minimum pixels for valid character
    "max_char_width": 50,  # Maximum pixels for valid character
    "min_char_height": 10,
    "char_spacing_threshold": 3,  # Pixels between characters
    
    # Projection profile parameters
    "projection_threshold": 0.1,  # Minimum % of white pixels to consider non-empty
    
    # Padding for segmented characters
    "padding": 2,  # Pixels to add around segmented character
    
    # Debug/visualization
    "save_segmentation_steps": True,
    "save_individual_chars": True,
}


# ============================================================================
# WORD RECOGNITION CONFIGURATION
# ============================================================================
WORD_RECOGNITION_CONFIG: Dict[str, Any] = {
    # Character recognition
    "use_confidence_threshold": True,
    "min_confidence": 0.3,  # Minimum confidence to accept prediction
    "unknown_char_placeholder": "?",  # Used for low-confidence predictions
    
    # Word assembly
    "merge_strategy": "left_to_right",  # Simple left-to-right assembly
    "preserve_spacing": False,  # Whether to preserve detected spaces
    
    # Post-processing (optional - can be expanded in future)
    "apply_spell_check": False,  # Not implemented yet (requires dictionary)
    "force_uppercase": True,  # Convert all letters to uppercase
}


# ============================================================================
# SLIDING WINDOW CONFIGURATION (Alternative Segmentation)
# ============================================================================
SLIDING_WINDOW_CONFIG: Dict[str, Any] = {
    "enabled": False,  # Use projection-based by default
    "window_width": 28,
    "window_height": 28,
    "stride": 5,  # Pixels to move window
    "overlap_threshold": 0.5,  # IoU threshold for merging detections
}


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG: Dict[str, Any] = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_to_file": True,
    "log_filename": "word_recognition.log",
}


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
EVALUATION_CONFIG: Dict[str, Any] = {
    # Character-level metrics
    "compute_char_accuracy": True,
    
    # Word-level metrics
    "compute_word_accuracy": True,
    "case_sensitive": False,
    
    # Error analysis
    "log_misrecognitions": True,
    "save_failed_segmentations": True,
}


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
VISUALIZATION_CONFIG: Dict[str, Any] = {
    "plot_segmentation": True,
    "plot_predictions": True,
    "save_plots": True,
    "dpi": 150,
}
