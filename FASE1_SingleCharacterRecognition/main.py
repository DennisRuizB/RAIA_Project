"""
Main training pipeline for EMNIST Letter Recognition - Phase 1.

This script orchestrates the complete training and evaluation workflow:
1. Data loading
2. Preprocessing
3. Model training
4. Evaluation
5. Model persistence

Author: Senior ML Engineer
Date: 2025
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import EMNISTDataLoader
from preprocessor import ImagePreprocessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from logger import setup_logger
from config import MODELS_DIR, LOGS_DIR, TRAINING_CONFIG
import pickle


def main() -> None:
    """Execute the complete training pipeline."""
    
    # Setup logger
    logger = setup_logger("MainPipeline")
    
    logger.info("=" * 70)
    logger.info("EMNIST LETTER RECOGNITION - PHASE 1")
    logger.info("Single Character Recognition System")
    logger.info("=" * 70)
    
    try:
        # ====================================================================
        # STEP 1: DATA LOADING
        # ====================================================================
        logger.info("\n[STEP 1/5] Loading Data...")
        
        data_loader = EMNISTDataLoader()
        
        # Load training data
        X_train_raw, y_train, letters_train = data_loader.load_train_data()
        logger.info(f"Training data loaded: {X_train_raw.shape[0]} samples")
        
        # Load test data
        X_test_raw, y_test, letters_test = data_loader.load_test_data()
        logger.info(f"Test data loaded: {X_test_raw.shape[0]} samples")
        
        # Show class distribution
        train_dist = data_loader.get_class_distribution(y_train)
        logger.info(f"Class distribution (training): {train_dist}")
        
        # ====================================================================
        # STEP 2: PREPROCESSING
        # ====================================================================
        logger.info("\n[STEP 2/5] Preprocessing Images...")
        
        preprocessor = ImagePreprocessor()
        
        # Fit on training data and transform
        X_train = preprocessor.fit_transform(X_train_raw)
        logger.info(f"Training features shape after preprocessing: {X_train.shape}")
        
        # Transform test data
        X_test = preprocessor.transform(X_test_raw)
        logger.info(f"Test features shape after preprocessing: {X_test.shape}")
        
        # Save preprocessor
        preprocessor_path = MODELS_DIR / TRAINING_CONFIG["scaler_filename"]
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        logger.info(f"Preprocessor saved to: {preprocessor_path}")
        
        # ====================================================================
        # STEP 3: MODEL TRAINING
        # ====================================================================
        logger.info("\n[STEP 3/5] Training Model...")
        
        trainer = ModelTrainer()
        
        # Optional: Create validation split
        if TRAINING_CONFIG["use_validation_split"]:
            X_train_split, X_val, y_train_split, y_val = trainer.create_validation_split(
                X_train, y_train
            )
            
            # Train with validation
            trainer.train(X_train_split, y_train_split, X_val, y_val)
        else:
            # Train on full training set
            trainer.train(X_train, y_train)
        
        # Save model
        if TRAINING_CONFIG["save_model"]:
            model_path = trainer.save_model()
            logger.info(f"Model saved to: {model_path}")
        
        # ====================================================================
        # STEP 4: EVALUATION
        # ====================================================================
        logger.info("\n[STEP 4/5] Evaluating Model...")
        
        evaluator = ModelEvaluator(data_loader.label_mapping)
        
        # Evaluate on test set
        y_test_pred = trainer.predict(X_test)
        test_results = evaluator.evaluate(y_test, y_test_pred, dataset_name="Test")
        
        # Analyze confusion pairs
        confusion_pairs = evaluator.get_confusion_pairs(y_test, y_test_pred, top_k=10)
        
        # Get per-class metrics
        per_class_df = evaluator.get_per_class_metrics(y_test, y_test_pred)
        logger.info("\nPer-Class Performance:")
        logger.info(f"\n{per_class_df.to_string()}")
        
        # ====================================================================
        # STEP 5: SAVE RESULTS
        # ====================================================================
        logger.info("\n[STEP 5/5] Saving Results...")
        
        # Save evaluation results
        results_path = LOGS_DIR / "evaluation_results.txt"
        evaluator.save_results(str(results_path))
        
        # Save confusion matrix visualization
        evaluator.visualize_confusion_matrix(y_test, y_test_pred, normalize=False)
        
        # Save per-class metrics
        metrics_path = LOGS_DIR / "per_class_metrics.csv"
        per_class_df.to_csv(metrics_path, index=False)
        logger.info(f"Per-class metrics saved to: {metrics_path}")
        
        # ====================================================================
        # SUMMARY
        # ====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"Test Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
        logger.info(f"Model saved: {MODELS_DIR / TRAINING_CONFIG['model_filename']}")
        logger.info(f"Results saved: {results_path}")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\nTraining failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
