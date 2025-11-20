"""
Test the Phase 1 model with direct EMNIST samples (no segmentation).

This isolates whether the problem is:
1. The model itself
2. The segmentation/normalization pipeline
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "FASE1_SingleCharacterRecognition" / "src"))

from inference_engine import InferenceEngine


def main():
    print("=" * 70)
    print("DIRECT EMNIST TEST - No Segmentation")
    print("=" * 70)
    
    # Load EMNIST test data
    test_path = Path(__file__).parent.parent / "RAIA_Project-main" / "emnist-letters-test.csv"
    print(f"\nLoading test data from: {test_path}")
    
    df = pd.read_csv(test_path, header=None, dtype=np.float32, nrows=100)
    
    # Initialize inference engine
    print("\nInitializing inference engine...")
    engine = InferenceEngine()
    engine.load()  # Use load(), not load_model()
    
    # Get labels and pixels
    y_true = df.iloc[:, 0].values.astype(int)
    X = df.iloc[:, 1:].values
    
    # Map labels to letters
    letter_to_label = {chr(65 + i): i + 1 for i in range(26)}
    label_to_letter = {v: k for k, v in letter_to_label.items()}
    
    true_letters = [label_to_letter[label] for label in y_true]
    
    # Predict
    print(f"\nPredicting {len(X)} samples...")
    predicted_letters, confidences = engine.predict_batch(X, return_confidence=True)
    
    # Calculate accuracy
    correct = sum(1 for t, p in zip(true_letters, predicted_letters) if t == p)
    accuracy = correct / len(true_letters)
    
    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{len(true_letters)})")
    
    # Show first 20 results
    print("\nFirst 20 predictions:")
    print("-" * 70)
    for i in range(min(20, len(true_letters))):
        match = "✓" if true_letters[i] == predicted_letters[i] else "✗"
        conf_str = f"{confidences[i]:.2f}" if confidences[i] is not None else "N/A"
        print(f"Sample {i+1:2d}: True={true_letters[i]}, "
              f"Pred={predicted_letters[i]}, "
              f"Conf={conf_str} {match}")
    
    # Show errors
    errors = [(i, t, p) for i, (t, p) in enumerate(zip(true_letters, predicted_letters)) if t != p]
    
    if errors:
        print(f"\nErrors ({len(errors)} total):")
        print("-" * 70)
        for i, true, pred in errors[:10]:
            print(f"Sample {i+1}: Expected '{true}', got '{pred}'")


if __name__ == "__main__":
    main()
