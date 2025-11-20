"""
Main script for Word Recognition System - Phase 2.

Demonstrates word-level recognition using Phase 1 character classifier.

Author: Senior ML Engineer
Date: 2025
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.word_recognizer import WordRecognizer
from src.logger import setup_logger
from src.config import RESULTS_DIR, OUTPUT_DIR


def demo_from_synthetic() -> None:
    """
    Demo with synthetically created word image.
    
    Creates a simple synthetic word image for testing.
    """
    logger = setup_logger("DemoSynthetic")
    
    logger.info("=" * 70)
    logger.info("WORD RECOGNITION DEMO - Synthetic Image")
    logger.info("=" * 70)
    
    # Initialize recognizer
    recognizer = WordRecognizer()
    recognizer.load_model()
    
    # Create simple synthetic word image (placeholder)
    # In practice, you'd load real images here
    logger.info("\nCreating synthetic word image...")
    
    # Example: Create a simple 28x100 image with some patterns
    # This is just a placeholder - replace with real word images
    synthetic_image = np.random.randint(0, 255, (28, 100), dtype=np.uint8)
    
    # Recognize
    logger.info("Recognizing word...")
    word, letters, confidences = recognizer.recognize_word(
        synthetic_image,
        image_id="synthetic_demo",
        return_details=True
    )
    
    # Display results
    logger.info("\n" + "=" * 70)
    logger.info(f"Recognized Word: '{word}'")
    logger.info(f"Characters: {letters}")
    logger.info(f"Confidences: {[f'{c:.2f}' for c in confidences]}")
    logger.info(f"Average Confidence: {np.mean(confidences):.2f}")
    logger.info("=" * 70)


def demo_from_emnist_samples() -> None:
    """
    Demo by creating word from EMNIST test samples.
    
    Concatenates individual EMNIST letters to form synthetic words.
    """
    import pandas as pd
    
    logger = setup_logger("DemoEMNIST")
    
    logger.info("=" * 70)
    logger.info("WORD RECOGNITION DEMO - EMNIST Samples")
    logger.info("=" * 70)
    
    # Load EMNIST training data (has more diverse samples)
    emnist_path = Path(__file__).parent.parent / "RAIA_Project-main" / "emnist-letters-train.csv"
    
    if not emnist_path.exists():
        logger.error(f"EMNIST training data not found: {emnist_path}")
        return
    
    logger.info(f"Loading EMNIST samples from: {emnist_path}")
    # Load enough samples to have diversity across all 26 letters
    df = pd.read_csv(emnist_path, header=None, dtype=np.float32, nrows=5000)
    
    # Initialize recognizer
    recognizer = WordRecognizer()
    recognizer.load_model()
    
    # Create synthetic words by concatenating letters
    # Start with simple test, then more complex
    test_words = ["AAA", "ABC", "HELLO", "WORLD"]
    
    for test_word in test_words:
        logger.info(f"\n--- Creating word: {test_word} ---")
        
        # Sample random EMNIST images for each letter
        word_image = create_word_image_from_emnist(df, test_word, logger)
        
        if word_image is not None:
            # Recognize
            recognized, letters, confidences = recognizer.recognize_word(
                word_image,
                image_id=f"word_{test_word}",
                return_details=True
            )
            
            # Display results
            logger.info(f"Target Word:     {test_word}")
            logger.info(f"Recognized Word: {recognized}")
            logger.info(f"Characters:      {letters}")
            logger.info(f"Match: {'✓' if recognized == test_word else '✗'}")


def create_word_image_from_emnist(
    df: "pd.DataFrame",
    word: str,
    logger
) -> np.ndarray:
    """
    Create a word image by horizontally concatenating EMNIST letter samples.
    
    Args:
        df: DataFrame with EMNIST data
        word: Target word string
        logger: Logger instance
    
    Returns:
        Concatenated word image
    """
    # Map letters to EMNIST labels (A=1, B=2, ..., Z=26)
    letter_to_label = {chr(65 + i): i + 1 for i in range(26)}
    
    char_images = []
    
    for letter in word.upper():
        if letter not in letter_to_label:
            logger.warning(f"Letter '{letter}' not in EMNIST")
            continue
        
        label = letter_to_label[letter]
        
        # Find samples with this label
        samples = df[df.iloc[:, 0] == label]
        
        if len(samples) == 0:
            logger.warning(f"No samples found for letter '{letter}'")
            continue
        
        # Pick random sample
        sample = samples.sample(1).iloc[0, 1:].values
        
        # Reshape to 28x28
        char_img = sample.reshape(28, 28).astype(np.uint8)
        
        # Apply EMNIST orientation corrections for proper segmentation
        char_img = np.rot90(char_img, k=3)
        char_img = np.fliplr(char_img)
        
        char_images.append(char_img)
    
    if len(char_images) == 0:
        return None
    
    # Concatenate horizontally with BLACK spacing (EMNIST has white BG)
    spacing = 2
    black_space = np.zeros((28, spacing), dtype=np.uint8)  # BLACK not white!
    
    word_image = np.hstack([
        np.hstack([img, black_space])
        for img in char_images[:-1]
    ] + [char_images[-1]])
    
    # NO inversion! Keep EMNIST format: high values = text, low values = background
    
    # Save the created word image for debugging
    try:
        from skimage import io
        output_path = Path(__file__).parent / "output" / f"debug_word_{word}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        io.imsave(str(output_path), word_image)
        logger.info(f"Saved word image to: {output_path}")
        logger.info(f"Word image shape: {word_image.shape}, range: [{word_image.min()}, {word_image.max()}]")
    except Exception as e:
        logger.warning(f"Failed to save debug image: {e}")
    
    return word_image


def interactive_mode() -> None:
    """
    Interactive mode for recognizing word images.
    """
    logger = setup_logger("Interactive")
    
    logger.info("=" * 70)
    logger.info("INTERACTIVE WORD RECOGNITION")
    logger.info("=" * 70)
    
    # Initialize recognizer
    recognizer = WordRecognizer()
    recognizer.load_model()
    
    logger.info("\nReady for recognition!")
    logger.info("Commands:")
    logger.info("  - Enter image path to recognize")
    logger.info("  - 'demo' to run EMNIST demo")
    logger.info("  - 'quit' to exit")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                logger.info("Exiting...")
                break
            
            elif user_input.lower() == 'demo':
                demo_from_emnist_samples()
            
            else:
                # Try to load image
                image_path = Path(user_input)
                if not image_path.exists():
                    logger.error(f"File not found: {image_path}")
                    continue
                
                # Load and recognize
                try:
                    from skimage import io
                    image = io.imread(str(image_path))
                    
                    word, letters, confidences = recognizer.recognize_word(
                        image,
                        image_id=image_path.stem,
                        return_details=True
                    )
                    
                    logger.info(f"\nRecognized: '{word}'")
                    logger.info(f"Letters: {letters}")
                    logger.info(f"Avg Confidence: {np.mean(confidences):.2f}")
                    
                except Exception as e:
                    logger.error(f"Failed to process image: {str(e)}")
        
        except KeyboardInterrupt:
            logger.info("\nExiting...")
            break


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Word Recognition System - Phase 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run EMNIST-based demo
  python main.py --demo
  
  # Interactive mode
  python main.py --interactive
        """
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo with EMNIST samples'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Enter interactive mode'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Run synthetic image demo'
    )
    
    args = parser.parse_args()
    
    logger = setup_logger("MainScript")
    
    try:
        if args.demo:
            demo_from_emnist_samples()
        elif args.interactive:
            interactive_mode()
        elif args.synthetic:
            demo_from_synthetic()
        else:
            logger.info("No mode specified. Running EMNIST demo...")
            demo_from_emnist_samples()
    
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
