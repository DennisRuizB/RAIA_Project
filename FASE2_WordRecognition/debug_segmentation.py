"""
Debug script to visualize segmentation and compare with EMNIST format.

Saves intermediate images to understand the preprocessing mismatch.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "FASE1_SingleCharacterRecognition" / "src"))

from src.word_recognizer import WordRecognizer
from src.image_segmenter import ImageSegmenter


def load_emnist_sample(letter: str, df: pd.DataFrame) -> np.ndarray:
    """Load a single EMNIST sample for a given letter."""
    letter_to_label = {chr(65 + i): i + 1 for i in range(26)}
    label = letter_to_label[letter.upper()]
    
    samples = df[df.iloc[:, 0] == label]
    if len(samples) == 0:
        raise ValueError(f"No samples for {letter}")
    
    # Get pixel data (columns 1-784)
    sample = samples.sample(1).iloc[0, 1:].values
    img = sample.reshape(28, 28).astype(np.uint8)
    
    # Apply EMNIST corrections
    img = np.rot90(img, k=3)
    img = np.fliplr(img)
    
    return img


def create_comparison_figure(letter: str, original_emnist: np.ndarray, 
                             segmented_char: np.ndarray, 
                             output_path: Path):
    """Create side-by-side comparison of EMNIST vs segmented character."""
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Row 1: Original EMNIST
    axes[0, 0].imshow(original_emnist, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title(f'EMNIST Original\n{letter}')
    axes[0, 0].axis('off')
    
    axes[0, 1].hist(original_emnist.ravel(), bins=50, color='blue', alpha=0.7)
    axes[0, 1].set_title('EMNIST Histogram')
    axes[0, 1].set_xlabel('Pixel Value')
    
    axes[0, 2].text(0.1, 0.5, 
                    f'Shape: {original_emnist.shape}\n'
                    f'Min: {original_emnist.min()}\n'
                    f'Max: {original_emnist.max()}\n'
                    f'Mean: {original_emnist.mean():.1f}\n'
                    f'Dtype: {original_emnist.dtype}',
                    fontsize=10, family='monospace',
                    verticalalignment='center')
    axes[0, 2].axis('off')
    
    # Row 2: Segmented character
    axes[1, 0].imshow(segmented_char, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(f'Segmented Char\n(from word)')
    axes[1, 0].axis('off')
    
    axes[1, 1].hist(segmented_char.ravel(), bins=50, color='red', alpha=0.7)
    axes[1, 1].set_title('Segmented Histogram')
    axes[1, 1].set_xlabel('Pixel Value')
    
    axes[1, 2].text(0.1, 0.5,
                    f'Shape: {segmented_char.shape}\n'
                    f'Min: {segmented_char.min()}\n'
                    f'Max: {segmented_char.max()}\n'
                    f'Mean: {segmented_char.mean():.1f}\n'
                    f'Dtype: {segmented_char.dtype}',
                    fontsize=10, family='monospace',
                    verticalalignment='center')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Comparison: Letter "{letter}"', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison to: {output_path}")


def main():
    """Run diagnostic comparison."""
    
    print("=" * 70)
    print("SEGMENTATION DIAGNOSTIC")
    print("=" * 70)
    
    # Load EMNIST data
    emnist_path = Path(__file__).parent.parent / "RAIA_Project-main" / "emnist-letters-train.csv"
    print(f"\nLoading EMNIST from: {emnist_path}")
    df = pd.read_csv(emnist_path, header=None, dtype=np.float32, nrows=5000)
    
    # Test letters
    test_letters = ['A', 'H', 'E', 'L', 'O']
    
    # Create output directory
    output_dir = Path(__file__).parent / "output" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize segmenter
    segmenter = ImageSegmenter()
    
    for letter in test_letters:
        print(f"\n--- Analyzing letter: {letter} ---")
        
        # Get original EMNIST sample
        original_emnist = load_emnist_sample(letter, df)
        print(f"EMNIST: shape={original_emnist.shape}, "
              f"range=[{original_emnist.min()}, {original_emnist.max()}], "
              f"mean={original_emnist.mean():.1f}")
        
        # Create word with spacing (simulate main.py behavior)
        spacing = 5
        white_space = np.ones((28, spacing), dtype=np.uint8) * 255
        word_image = np.hstack([original_emnist, white_space, original_emnist])
        
        # Segment it
        segmented_chars = segmenter.segment_word(word_image, image_id=f"debug_{letter}")
        
        if len(segmented_chars) > 0:
            segmented_char = segmented_chars[0]
            print(f"Segmented: shape={segmented_char.shape}, "
                  f"range=[{segmented_char.min()}, {segmented_char.max()}], "
                  f"mean={segmented_char.mean():.1f}")
            
            # Create comparison figure
            output_path = output_dir / f"comparison_{letter}.png"
            create_comparison_figure(letter, original_emnist, segmented_char, output_path)
            
            # Check if they match
            difference = np.mean(np.abs(original_emnist.astype(float) - segmented_char))
            print(f"Mean absolute difference: {difference:.2f}")
            
            if difference > 50:
                print("⚠️  HIGH DIFFERENCE - Format mismatch detected!")
        else:
            print(f"❌ No characters segmented for {letter}")
    
    print("\n" + "=" * 70)
    print(f"Diagnostics saved to: {output_dir}")
    print("=" * 70)
    
    # Test with actual recognizer
    print("\n--- Testing with recognizer ---")
    recognizer = WordRecognizer()
    recognizer.load_model()
    
    # Create simple word
    word_chars = [load_emnist_sample('H', df), 
                  load_emnist_sample('E', df),
                  load_emnist_sample('L', df)]
    
    spacing = 5
    white_space = np.ones((28, spacing), dtype=np.uint8) * 255
    
    word_image = np.hstack([
        np.hstack([img, white_space])
        for img in word_chars[:-1]
    ] + [word_chars[-1]])
    
    print(f"Word image: shape={word_image.shape}, range=[{word_image.min()}, {word_image.max()}]")
    
    recognized, letters, confidences = recognizer.recognize_word(
        word_image,
        image_id="debug_HEL",
        return_details=True
    )
    
    print(f"Expected: HEL")
    print(f"Got: {recognized}")
    print(f"Letters: {letters}")
    print(f"Confidences: {[f'{c:.2f}' for c in confidences]}")


if __name__ == "__main__":
    main()
