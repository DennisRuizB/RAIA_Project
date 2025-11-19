"""
Prediction script for single character inference.

Allows prediction of individual letters from various input sources:
- Raw pixel arrays
- Image files
- CSV data

Author: Senior ML Engineer
Date: 2025
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from inference_engine import InferenceEngine
from logger import setup_logger


def predict_from_array(
    engine: InferenceEngine,
    image_array: np.ndarray,
    show_top_k: int = 5
) -> None:
    """
    Predict letter from a numpy array.
    
    Args:
        engine: Loaded inference engine
        image_array: Image as numpy array (28x28 or 784)
        show_top_k: Number of top predictions to show
    """
    logger = setup_logger("PredictArray")
    
    # Get top-k predictions
    if show_top_k > 1:
        predictions = engine.predict_with_top_k(image_array, k=show_top_k)
        
        logger.info(f"\nTop {show_top_k} Predictions:")
        for i, (letter, prob) in enumerate(predictions, 1):
            logger.info(f"{i}. '{letter}' - {prob*100:.2f}%")
    else:
        # Single prediction
        letter, confidence = engine.predict_single(image_array, return_confidence=True)
        
        logger.info(f"\nPredicted Letter: '{letter}'")
        if confidence is not None:
            logger.info(f"Confidence: {confidence*100:.2f}%")


def predict_from_csv(
    engine: InferenceEngine,
    csv_path: Path,
    n_samples: int = 10
) -> None:
    """
    Predict letters from a CSV file (e.g., test data).
    
    Args:
        engine: Loaded inference engine
        csv_path: Path to CSV file
        n_samples: Number of samples to predict
    """
    import pandas as pd
    
    logger = setup_logger("PredictCSV")
    
    logger.info(f"Loading data from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path, header=None, dtype=np.float32, nrows=n_samples)
    
    # Extract features (skip first column if it's labels)
    if df.shape[1] == 785:  # Has labels
        y_true = df.iloc[:, 0].values.astype(np.int32)
        X = df.iloc[:, 1:].values
        has_labels = True
    else:
        X = df.values
        has_labels = False
    
    logger.info(f"Loaded {len(X)} samples")
    
    # Predict
    letters, confidences = engine.predict_batch(X, return_confidence=True)
    
    # Display results
    logger.info("\nPrediction Results:")
    logger.info("-" * 50)
    
    for i, (letter, conf) in enumerate(zip(letters, confidences), 1):
        info = f"Sample {i}: '{letter}'"
        if conf is not None:
            info += f" ({conf*100:.1f}%)"
        
        if has_labels:
            true_label = int(y_true[i-1])
            # Convert label to letter (A=1, B=2, etc.)
            true_letter = chr(64 + true_label)
            correct = "✓" if letter == true_letter else "✗"
            info += f" | True: '{true_letter}' {correct}"
        
        logger.info(info)
    
    # Accuracy if labels available
    if has_labels:
        y_pred_labels = [ord(letter) - 64 for letter in letters]
        accuracy = np.mean(np.array(y_pred_labels) == y_true)
        logger.info(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


def predict_interactive(engine: InferenceEngine) -> None:
    """
    Interactive prediction mode - manually input pixel values.
    
    Args:
        engine: Loaded inference engine
    """
    logger = setup_logger("PredictInteractive")
    
    logger.info("\n" + "=" * 70)
    logger.info("INTERACTIVE PREDICTION MODE")
    logger.info("=" * 70)
    logger.info("Enter 784 pixel values (space-separated) or 'quit' to exit")
    logger.info("Pixel values should be in range [0, 255]")
    logger.info("-" * 70)
    
    while True:
        try:
            user_input = input("\nEnter pixel values (or 'quit'): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                logger.info("Exiting interactive mode")
                break
            
            # Parse input
            pixels = np.array([float(x) for x in user_input.split()])
            
            if len(pixels) != 784:
                logger.error(f"Expected 784 values, got {len(pixels)}")
                continue
            
            # Predict
            predict_from_array(engine, pixels, show_top_k=5)
            
        except ValueError as e:
            logger.error(f"Invalid input: {str(e)}")
        except KeyboardInterrupt:
            logger.info("\nExiting interactive mode")
            break


def main():
    """Main prediction script."""
    
    parser = argparse.ArgumentParser(
        description="Predict letters from EMNIST images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from CSV file (first 10 samples)
  python predict.py --csv ../RAIA_Project-main/emnist-letters-test.csv --samples 10
  
  # Interactive mode
  python predict.py --interactive
  
  # Predict specific array (passed as arguments)
  python predict.py --array 0 0 0 ... (784 values)
        """
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to CSV file with image data'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of samples to predict from CSV (default: 10)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Enter interactive prediction mode'
    )
    parser.add_argument(
        '--array',
        type=float,
        nargs=784,
        help='Predict from 784 pixel values passed as arguments'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to model file (optional, uses default if not specified)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions to show (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger("PredictMain")
    
    try:
        # Initialize and load inference engine
        logger.info("Initializing inference engine...")
        
        model_path = Path(args.model) if args.model else None
        engine = InferenceEngine(model_path=model_path)
        engine.load()
        
        logger.info("Inference engine ready!")
        logger.info(f"Model info: {engine.get_model_info()}")
        
        # Determine prediction mode
        if args.csv:
            csv_path = Path(args.csv)
            if not csv_path.exists():
                logger.error(f"CSV file not found: {csv_path}")
                sys.exit(1)
            predict_from_csv(engine, csv_path, n_samples=args.samples)
        
        elif args.interactive:
            predict_interactive(engine)
        
        elif args.array:
            image_array = np.array(args.array, dtype=np.float32)
            predict_from_array(engine, image_array, show_top_k=args.top_k)
        
        else:
            logger.error("No prediction mode specified. Use --csv, --interactive, or --array")
            parser.print_help()
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
