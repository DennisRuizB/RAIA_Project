"""
Quick augmentation training with progress bar.
Uses tqdm for visual progress tracking.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from scipy import ndimage
from tqdm import tqdm as progress_bar

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import TRAIN_DATA_PATH, MODELS_DIR
from data_loader import EMNISTDataLoader
from preprocessor import ImagePreprocessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator


def augment_character(image: np.ndarray) -> np.ndarray:
    """Apply augmentations to simulate segmentation pipeline."""
    scale = np.random.uniform(0.75, 0.95)
    new_size = int(28 * scale)
    new_size = max(10, min(26, new_size))
    
    if new_size != 28:
        zoom_factor = new_size / 28
        resized = ndimage.zoom(image, zoom_factor, order=1)
    else:
        resized = image.copy()
    
    canvas = np.zeros((28, 28), dtype=np.float32)
    h, w = resized.shape
    y_offset = (28 - h) // 2 + np.random.randint(-2, 3)
    x_offset = (28 - w) // 2 + np.random.randint(-2, 3)
    y_offset = max(0, min(28 - h, y_offset))
    x_offset = max(0, min(28 - w, x_offset))
    
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = resized
    
    angle = np.random.uniform(-3, 3)
    canvas = ndimage.rotate(canvas, angle, reshape=False, order=1, mode='constant', cval=0)
    
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 2, canvas.shape)
        canvas = np.clip(canvas + noise, 0, 255)
    
    return canvas.astype(np.uint8)


def create_augmented_dataset(X: np.ndarray, y: np.ndarray, 
                            aug_factor: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create augmented dataset with progress bar."""
    n_samples = X.shape[0]
    n_augmented = n_samples * aug_factor
    
    print(f"\n📊 Augmentación de datos:")
    print(f"  • Original: {n_samples:,} samples")
    print(f"  • Augmentación: {aug_factor}x")
    print(f"  • Total: {n_samples + n_augmented:,} samples\n")
    
    X_aug = np.zeros((n_samples + n_augmented, 784), dtype=np.float32)
    y_aug = np.zeros(n_samples + n_augmented, dtype=int)
    
    X_aug[:n_samples] = X
    y_aug[:n_samples] = y
    
    idx = n_samples
    
    with progress_bar(total=n_samples, desc="🔄 Augmentando", unit="imgs", 
              bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for i in range(n_samples):
            img = X[i].reshape(28, 28)
            label = y[i]
            
            for _ in range(aug_factor):
                aug_img = augment_character(img)
                X_aug[idx] = aug_img.flatten()
                y_aug[idx] = label
                idx += 1
            
            pbar.update(1)
    
    print(f"\n✅ Augmentación completa: {len(X_aug):,} samples totales\n")
    return X_aug, y_aug


def main():
    print("=" * 70)
    print("🚀 ENTRENAMIENTO AUGMENTADO CON BARRA DE PROGRESO")
    print("=" * 70)
    print()
    
    # STEP 1: Load Data
    print("📂 [1/6] Cargando datos...")
    loader = EMNISTDataLoader()
    X_train, y_train, _ = loader.load_train_data()
    X_test, y_test, _ = loader.load_test_data()
    print(f"✓ Datos cargados: {len(X_train):,} train, {len(X_test):,} test\n")
    
    # STEP 2: Augment
    print("🎨 [2/6] Generando datos augmentados...")
    X_train_aug, y_train_aug = create_augmented_dataset(X_train, y_train, aug_factor=1)
    
    # STEP 3: Preprocess
    print("⚙️  [3/6] Preprocesando (HOG features)...")
    preprocessor = ImagePreprocessor()
    
    with progress_bar(total=2, desc="   Procesando", unit="set",
              bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt}') as pbar:
        X_train_processed = preprocessor.fit_transform(X_train_aug)
        pbar.update(1)
        X_test_processed = preprocessor.transform(X_test)
        pbar.update(1)
    
    print(f"✓ Features extraídas: {X_train_processed.shape}\n")
    
    # STEP 4: Train
    print("🧠 [4/6] Entrenando modelo SVM...")
    print("⏳ Esto tomará ~12-15 minutos (sin barra por limitación de sklearn)...")
    
    trainer = ModelTrainer()
    trainer.train(X_train_processed, y_train_aug)
    print("✓ Entrenamiento completado!\n")
    
    # STEP 5: Evaluate
    print("📊 [5/6] Evaluando modelo...")
    print("⏳ Generando predicciones en test set...")
    y_pred = trainer.predict(X_test_processed)
    
    evaluator = ModelEvaluator(loader.label_mapping)
    results = evaluator.evaluate(y_test, y_pred)
    print(f"✓ Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
    
    # STEP 6: Save
    print("💾 [6/6] Guardando modelo augmentado...")
    model_filename = "emnist_letter_classifier_augmented.pkl"
    scaler_filename = "feature_scaler_augmented.pkl"
    
    trainer.save_model(
        model_path=MODELS_DIR / model_filename,
        preprocessor=preprocessor,
        preprocessor_path=MODELS_DIR / scaler_filename
    )
    print(f"✓ Guardado: {model_filename}\n")
    
    # Summary
    print("=" * 70)
    print("🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
    print("=" * 70)
    print(f"📈 Samples entrenados: {len(X_train_aug):,}")
    print(f"🎯 Test accuracy: {evaluator.overall_accuracy*100:.2f}%")
    print(f"💾 Modelo: {model_filename}")
    print()
    print("📝 Próximo paso:")
    print("   cd ..\\FASE2_WordRecognition")
    print("   python main.py --demo")
    print("=" * 70)


if __name__ == "__main__":
    try:
        # Try to import tqdm
        import tqdm
    except ImportError:
        print("⚠️  Instalando tqdm para barra de progreso...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
        print("✓ tqdm instalado\n")
    
    main()
