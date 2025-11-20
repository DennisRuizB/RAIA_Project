"""
Main script to generate printed text dataset.

Usage:
    python generate_dataset.py                # Generate full dataset
    python generate_dataset.py --samples 1000 # Generate smaller dataset
    python generate_dataset.py --preview      # Generate with preview samples
"""

import argparse
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset_generator import PrintedDatasetGenerator
from config import DATASET_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Generate printed text dataset")
    parser.add_argument("--samples", type=int, default=None,
                       help="Samples per letter per variation (default: from config)")
    parser.add_argument("--preview", action="store_true",
                       help="Save preview samples")
    args = parser.parse_args()
    
    # Update config if needed
    if args.samples:
        DATASET_CONFIG["samples_per_letter_per_variation"] = args.samples
        print(f"⚙️  Using {args.samples} samples per variation")
        print()
    
    # Create generator
    generator = PrintedDatasetGenerator()
    
    # Generate dataset
    X_data, y_labels = generator.generate_dataset(preview_samples=args.preview)
    
    # Split train/test
    print("🔀 Dividiendo en train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_data,
        y_labels,
        test_size=DATASET_CONFIG["test_size"],
        random_state=DATASET_CONFIG["random_seed"],
        stratify=y_labels
    )
    
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test:  {len(X_test):,} samples")
    print()
    
    # Save to CSV
    print("💾 Guardando archivos...")
    generator.save_to_csv(X_train, y_train, "train.csv")
    generator.save_to_csv(X_test, y_test, "test.csv")
    generator.create_mapping_file()
    
    print()
    print("=" * 70)
    print("✅ DATASET GENERADO EXITOSAMENTE!")
    print("=" * 70)
    print()
    print("📂 Archivos creados:")
    print(f"   • data/train.csv ({len(X_train):,} samples)")
    print(f"   • data/test.csv ({len(X_test):,} samples)")
    print(f"   • data/mapping.txt (26 labels)")
    if args.preview:
        print(f"   • data/samples/ (26 ejemplos)")
    print()
    print("🚀 Próximo paso:")
    print("   python train.py")
    print()


if __name__ == "__main__":
    main()
