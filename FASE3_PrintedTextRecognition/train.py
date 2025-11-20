"""
Model trainer for printed text recognition.

Trains SVM/MLP/KNN classifier on generated dataset.
"""

import sys
from pathlib import Path

# Add FASE3 src to path FIRST (priority)
FASE3_SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(FASE3_SRC))

# Import config from FASE3 (before FASE1 imports)
from config import DATA_DIR, MODELS_DIR, MODEL_CONFIG, TRAINING_CONFIG, PREPROCESSING_CONFIG, IMAGE_SIZE

# Now add FASE1 to path for reusing components
FASE1_DIR = Path(__file__).parent.parent / "FASE1_SingleCharacterRecognition" / "src"
sys.path.insert(0, str(FASE1_DIR))

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# IMPORTANTE: Importar preprocessor y forzar uso de config FASE3
from preprocessor import ImagePreprocessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator

# Inyectar PREPROCESSING_CONFIG de FASE3 en el módulo config de FASE1
import config as fase1_config
fase1_config.PREPROCESSING_CONFIG = PREPROCESSING_CONFIG
fase1_config.IMAGE_SIZE = IMAGE_SIZE


def load_data():
    """Load train and test datasets."""
    print("[*] Cargando datos...")
    
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {train_path}\n"
            "Ejecuta primero: python generate_dataset.py"
        )
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    X_train = df_train.iloc[:, 1:].values.astype(np.float32)
    y_train = df_train.iloc[:, 0].values.astype(int)
    
    X_test = df_test.iloc[:, 1:].values.astype(np.float32)
    y_test = df_test.iloc[:, 0].values.astype(int)
    
    print(f"[OK] Train: {len(X_train):,} samples")
    print(f"[OK] Test: {len(X_test):,} samples")
    print()
    
    return X_train, y_train, X_test, y_test


def load_label_mapping():
    """Load label mapping."""
    mapping_path = DATA_DIR / "mapping.txt"
    mapping = {}
    
    with open(mapping_path, 'r') as f:
        for line in f:
            label, letter = line.strip().split()
            mapping[int(label)] = letter
    
    return mapping


def main():
    print("=" * 70)
    print("ENTRENAMIENTO - RECONOCIMIENTO DE TEXTO IMPRESO")
    print("=" * 70)
    print()
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    label_mapping = load_label_mapping()
    
    # Preprocess
    print("[*] Preprocesando (HOG features)...")
    preprocessor = ImagePreprocessor()
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"[OK] Features extraidas: {X_train_processed.shape}")
    print()
    
    # Train
    print("[*] Entrenando modelo...")
    print(f"   Tipo: {MODEL_CONFIG['model_type'].upper()}")
    print()
    
    trainer = ModelTrainer(model_type=MODEL_CONFIG["model_type"])
    trainer.train(X_train_processed, y_train)
    
    print()
    
    # Evaluate
    print("[*] Evaluando modelo...")
    y_pred = trainer.predict(X_test_processed)
    
    evaluator = ModelEvaluator(label_mapping)
    results = evaluator.evaluate(y_test, y_pred, dataset_name="Test")
    
    print()
    
    # Save model
    print("[*] Guardando modelo...")
    model_path = MODELS_DIR / TRAINING_CONFIG["model_filename"]
    preprocessor_path = MODELS_DIR / TRAINING_CONFIG["preprocessor_filename"]
    
    with open(model_path, 'wb') as f:
        pickle.dump(trainer.model, f)
    
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print(f"[OK] Modelo: {model_path}")
    print(f"[OK] Preprocessor: {preprocessor_path}")
    print()
    
    # Summary
    print("=" * 70)
    print("[OK] ENTRENAMIENTO COMPLETADO!")
    print("=" * 70)
    print(f"[*] Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"[*] Modelo guardado: {model_path.name}")
    print()
    print("[*] Proximo paso:")
    print('   python predict.py --text "HELLO"')
    print()


if __name__ == "__main__":
    main()
