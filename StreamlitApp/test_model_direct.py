"""Test directo del modelo con imágenes del dataset"""
import sys
from pathlib import Path

sys.path.insert(0, 'utils')
sys.path.insert(0, str(Path(__file__).parent.parent / 'FASE3_PrintedTextRecognition'))

from model_utils import load_printed_model
import pandas as pd
import numpy as np

# Cargar modelo
print("Cargando modelo...")
model, prep, mapping = load_printed_model()

# Cargar muestras del test
test_path = Path(__file__).parent.parent / 'FASE3_PrintedTextRecognition' / 'data' / 'test.csv'
df = pd.read_csv(test_path, nrows=20)

print("\nProbando 10 muestras del dataset de test:")
print("-" * 50)

correct = 0
for i in range(10):
    pixels = df.iloc[i, 1:].values.reshape(1, -1)
    true_label = int(df.iloc[i, 0])
    
    # Predecir
    features = prep.transform(pixels)
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    
    # Obtener confianza
    pred_idx = np.where(model.classes_ == pred)[0][0]
    confidence = proba[pred_idx]
    
    true_letter = mapping.get(true_label, '?')
    pred_letter = mapping.get(pred, '?')
    match = true_label == pred
    
    if match:
        correct += 1
    
    status = "✓" if match else "✗"
    print(f"{status} Sample {i}: True={true_letter} ({true_label}), Pred={pred_letter} ({pred}), Conf={confidence:.3f}")

print("-" * 50)
print(f"Accuracy: {correct}/10 = {correct*10}%")
