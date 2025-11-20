"""
Data augmentation and retraining script.

Generates augmented training data with transformations similar to
word segmentation pipeline (resize, padding, noise) to improve
model robustness for Phase 2 word recognition.

Author: Senior ML Engineer
Date: 2025-11-20
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from scipy import ndimage

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import TRAIN_DATA_PATH, MODELS_DIR, LOGS_DIR
from data_loader import EMNISTDataLoader
from preprocessor import ImagePreprocessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from logger import setup_logger


def augment_character(image: np.ndarray, target_size: int = 28) -> np.ndarray:
    """
    Apply augmentations similar to word segmentation pipeline.
    
    Transformations:
    - Random resize (80-100% of original)
    - Random padding/cropping
    - Center in canvas
    - Small rotation (-5 to +5 degrees)
    
    Args:
        image: Original 28x28 character image
        target_size: Output size (28)
    
    Returns:
        Augmented 28x28 image
    """
    # Random resize factor (simulate segmentation resize artifacts)
    scale = np.random.uniform(0.75, 0.95)
    
    # Calculate new size
    new_size = int(target_size * scale)
    new_size = max(10, min(26, new_size))  # Clamp to reasonable range
    
    # Resize
    if new_size != target_size:
        zoom_factor = new_size / target_size
        resized = ndimage.zoom(image, zoom_factor, order=1)
    else:
        resized = image.copy()
    
    # Create canvas and center
    canvas = np.zeros((target_size, target_size), dtype=np.float32)
    
    h, w = resized.shape
    y_offset = (target_size - h) // 2
    x_offset = (target_size - w) // 2
    
    # Add small random shift
    y_offset += np.random.randint(-2, 3)
    x_offset += np.random.randint(-2, 3)
    
    # Clamp offsets
    y_offset = max(0, min(target_size - h, y_offset))
    x_offset = max(0, min(target_size - w, x_offset))
    
    # Place on canvas
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = resized
    
    # Small rotation (simulate slight segmentation misalignment)
    angle = np.random.uniform(-3, 3)
    canvas = ndimage.rotate(canvas, angle, reshape=False, order=1, mode='constant', cval=0)
    
    # Add slight Gaussian noise (simulate compression artifacts)
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 2, canvas.shape)
        canvas = np.clip(canvas + noise, 0, 255)
    
    return canvas.astype(np.uint8)


def create_augmented_dataset(
    X: np.ndarray,
    y: np.ndarray,
    augmentation_factor: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create augmented dataset by applying transformations.
    
    Args:
        X: Original training data (n_samples, 784)
        y: Labels
        augmentation_factor: How many augmented copies per original
    
    Returns:
        Tuple of (X_augmented, y_augmented) including originals
    """
    logger = setup_logger("DataAugmentation")
    
    n_samples = X.shape[0]
    n_augmented = n_samples * augmentation_factor
    
    logger.info(f"Creating augmented dataset...")
    logger.info(f"Original samples: {n_samples}")
    logger.info(f"Augmentation factor: {augmentation_factor}")
    logger.info(f"Total augmented samples: {n_augmented}")
    
    # Prepare output arrays
    X_aug = np.zeros((n_samples + n_augmented, 784), dtype=np.float32)
    y_aug = np.zeros(n_samples + n_augmented, dtype=int)
    
    # Copy originals
    X_aug[:n_samples] = X
    y_aug[:n_samples] = y
    
    # Generate augmented samples
    idx = n_samples
    for i in range(n_samples):
        # Reshape to 28x28
        img = X[i].reshape(28, 28)
        label = y[i]
        
        # Create augmented versions
        for _ in range(augmentation_factor):
            aug_img = augment_character(img)
            X_aug[idx] = aug_img.flatten()
            y_aug[idx] = label
            idx += 1
        
        # Progress
        if (i + 1) % 10000 == 0:
            logger.info(f"Processed {i + 1}/{n_samples} samples")
    
    logger.info(f"Augmentation complete! Total samples: {len(X_aug)}")
    
    return X_aug, y_aug


def main():
    """Main training pipeline with augmentation."""
    
    logger = setup_logger("AugmentedTraining")
    
    print("=" * 70)
    print("EMNIST LETTER RECOGNITION - AUGMENTED TRAINING")
    print("Phase 1: Enhanced for Word Recognition (Phase 2)")
    print("=" * 70)
    print()
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("[STEP 1/6] Loading Original Data...")
    loader = EMNISTDataLoader()
    
    X_train, y_train, _ = loader.load_train_data()
    X_test, y_test, _ = loader.load_test_data()
    
    logger.info(f"Original training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # ========================================================================
    # STEP 2: Create Augmented Dataset
    # ========================================================================
    print("\n[STEP 2/6] Creating Augmented Dataset...")
    print("This will take a few minutes...")
    
    # Augment with factor of 1 (doubles the dataset)
    X_train_aug, y_train_aug = create_augmented_dataset(
        X_train, 
        y_train,
        augmentation_factor=1  # 1 augmented copy per original = 2x data
    )
    
    logger.info(f"Training set size after augmentation: {len(X_train_aug)}")
    
    # ========================================================================
    # STEP 3: Preprocess
    # ========================================================================
    print("\n[STEP 3/6] Preprocessing Images...")
    preprocessor = ImagePreprocessor()
    
    # Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train_aug)
    
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    # ========================================================================
    # STEP 4: Train Model
    # ========================================================================
    print("\n[STEP 4/6] Training Model on Augmented Data...")
    print("This may take 15-20 minutes...")
    
    trainer = ModelTrainer()
    trainer.train(X_train_processed, y_train_aug)
    
    # ========================================================================
    # STEP 5: Evaluate
    # ========================================================================
    print("\n[STEP 5/6] Evaluating Model...")
    print("Generating predictions on test set...")
    y_test_pred = trainer.predict(X_test_processed)
    
    evaluator = ModelEvaluator(loader.label_mapping)
    test_results = evaluator.evaluate(y_test, y_test_pred, dataset_name="Test")
    
    # ========================================================================
    # STEP 6: Save Model
    # ========================================================================
    print("\n[STEP 6/6] Saving Augmented Model...")
    
    model_filename = "emnist_letter_classifier_augmented.pkl"
    scaler_filename = "feature_scaler_augmented.pkl"
    
    # Save model
    trainer.save_model(filename=model_filename)
    
    # Save preprocessor separately
    import pickle
    with open(MODELS_DIR / scaler_filename, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    logger.info(f"Augmented model saved: {model_filename}")
    logger.info(f"Augmented scaler saved: {scaler_filename}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("AUGMENTED TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Training samples (augmented): {len(X_train_aug):,}")
    print(f"Test accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"Model saved as: {model_filename}")
    print()
    print("Next steps:")
    print("1. Update FASE2 config to use augmented model")
    print("2. Run: python main.py --demo")
    print("=" * 70)


if __name__ == "__main__":
    main()
