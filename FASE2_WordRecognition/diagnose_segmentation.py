"""
Quick diagnostic script to visualize segmented letters and their predictions.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pickle

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "FASE1_SingleCharacterRecognition" / "src"))

from inference_engine import InferenceEngine

def diagnose_word(word_name):
    """Diagnose segmentation for a specific word."""
    
    # Load model
    print(f"\n{'='*70}")
    print(f"DIAGNOSING: {word_name}")
    print(f"{'='*70}")
    
    model_path = Path(__file__).parent.parent / "FASE1_SingleCharacterRecognition" / "models" / "emnist_letter_classifier_augmented.pkl"
    preprocessor_path = Path(__file__).parent.parent / "FASE1_SingleCharacterRecognition" / "models" / "feature_scaler_augmented.pkl"
    
    engine = InferenceEngine(
        model_path=model_path,
        preprocessor_path=preprocessor_path
    )
    engine.load()
    
    # Find segmented letters
    seg_dir = Path("output/segmented_letters")
    letter_files = sorted(seg_dir.glob(f"{word_name}_char_*.png"))
    
    if not letter_files:
        print(f"No segmented letters found for {word_name}")
        return
    
    print(f"\nFound {len(letter_files)} segmented characters\n")
    
    for i, letter_file in enumerate(letter_files):
        # Load image
        img = Image.open(letter_file).convert('L')
        img_array = np.array(img)
        
        print(f"Character {i}:")
        print(f"  File: {letter_file.name}")
        print(f"  Shape: {img_array.shape}")
        print(f"  Range: [{img_array.min()}, {img_array.max()}]")
        print(f"  Mean: {img_array.mean():.2f}")
        
        # Visualize as ASCII art (simplified)
        if img_array.shape == (28, 28):
            # Show small preview
            preview = img_array[::4, ::4]  # Downsample to 7x7
            print("  Preview (28x28 downsampled):")
            for row in preview:
                line = "    " + "".join("#" if p > 127 else "." for p in row)
                print(line)
        else:
            print(f"  ⚠️  Non-standard size: {img_array.shape} (expected 28x28)")
        
        # Predict
        try:
            if img_array.shape != (28, 28):
                print(f"  ❌ Cannot predict: wrong shape")
            else:
                img_flat = img_array.flatten().reshape(1, -1)
                predictions = engine.predict(img_flat, top_k=3)
                
                print(f"  🔮 Top predictions:")
                for pred in predictions:
                    print(f"     {pred['letter']}: {pred['confidence']:.2%}")
        except Exception as e:
            print(f"  ❌ Prediction error: {e}")
        
        print()


if __name__ == "__main__":
    # Diagnose recent words
    words = ["word_AAA", "word_ABC", "word_HELLO", "word_WORLD"]
    
    for word in words:
        diagnose_word(word)
        
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
