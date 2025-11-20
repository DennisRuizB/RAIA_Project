"""
Prediction script for printed text recognition.

Usage:
    python predict.py --text "HELLO"
    python predict.py --image letter.png
    python predict.py --test
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pickle
from PIL import Image

# Add FASE3 src to path FIRST (priority)
FASE3_SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(FASE3_SRC))

# Import config from FASE3 (before FASE1 imports)
from config import MODELS_DIR, DATA_DIR

# Now add FASE1 to path for reusing components
FASE1_DIR = Path(__file__).parent.parent / "FASE1_SingleCharacterRecognition" / "src"
sys.path.insert(0, str(FASE1_DIR))

from preprocessor import ImagePreprocessor


class PrintedTextPredictor:
    """Predictor for printed text recognition."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.label_mapping = None
    
    def load_model(self):
        """Load trained model and preprocessor."""
        model_path = MODELS_DIR / "printed_letter_classifier.pkl"
        preprocessor_path = MODELS_DIR / "printed_feature_scaler.pkl"
        mapping_path = DATA_DIR / "mapping.txt"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado: {model_path}\n"
                "Ejecuta primero: python train.py"
            )
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load preprocessor
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        # Load mapping
        self.label_mapping = {}
        with open(mapping_path, 'r') as f:
            for line in f:
                label, letter = line.strip().split()
                self.label_mapping[int(label)] = letter
        
        print("✓ Modelo cargado")
    
    def predict_from_array(self, img_array: np.ndarray) -> str:
        """Predict letter from image array (28x28)."""
        # Flatten and preprocess
        img_flat = img_array.flatten().reshape(1, -1)
        features = self.preprocessor.transform(img_flat)
        
        # Predict
        label = self.model.predict(features)[0]
        letter = self.label_mapping[label]
        
        return letter
    
    def predict_from_image(self, image_path: str) -> str:
        """Predict letter from image file."""
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img, dtype=np.float32)
        
        return self.predict_from_array(img_array)
    
    def test_samples(self, n_samples=10):
        """Test on random samples from test set."""
        import pandas as pd
        
        test_path = DATA_DIR / "test.csv"
        df = pd.read_csv(test_path)
        
        # Random samples
        samples = df.sample(n=n_samples, random_state=42)
        
        print(f"\n📝 Probando {n_samples} muestras aleatorias:\n")
        
        correct = 0
        for idx, row in samples.iterrows():
            true_label = int(row.iloc[0])
            true_letter = self.label_mapping[true_label]
            
            img_data = row.iloc[1:].values.astype(np.float32)
            img_array = img_data.reshape(28, 28)
            
            pred_letter = self.predict_from_array(img_array)
            
            match = "✓" if pred_letter == true_letter else "✗"
            print(f"  {match} True: {true_letter} | Pred: {pred_letter}")
            
            if pred_letter == true_letter:
                correct += 1
        
        accuracy = correct / n_samples
        print(f"\n  Accuracy: {accuracy:.2%} ({correct}/{n_samples})")


def main():
    parser = argparse.ArgumentParser(description="Predict printed text")
    parser.add_argument("--text", type=str, help="Text to verify (e.g., 'HELLO')")
    parser.add_argument("--image", type=str, help="Image file path")
    parser.add_argument("--test", action="store_true", help="Test on random samples")
    args = parser.parse_args()
    
    # Load model
    predictor = PrintedTextPredictor()
    predictor.load_model()
    
    if args.test:
        predictor.test_samples(n_samples=20)
    
    elif args.text:
        print(f"\n✨ Predicción para texto: '{args.text}'")
        print("(Simulando que cada letra es una imagen)")
        print()
        for letter in args.text.upper():
            if letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                print(f"  {letter} → (predicción requiere imagen real)")
        print()
        print("💡 Tip: Usa --test para probar con muestras reales")
    
    elif args.image:
        letter = predictor.predict_from_image(args.image)
        print(f"\n✨ Letra detectada: {letter}\n")
    
    else:
        print("\n❌ Debes especificar --text, --image o --test")
        print("\nEjemplos:")
        print('  python predict.py --text "HELLO"')
        print('  python predict.py --image letter.png')
        print('  python predict.py --test')
        print()


if __name__ == "__main__":
    main()
