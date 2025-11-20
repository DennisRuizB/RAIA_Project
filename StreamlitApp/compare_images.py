"""Comparar letras segmentadas vs dataset"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, 'utils')
sys.path.insert(0, str(Path(__file__).parent.parent / 'FASE3_PrintedTextRecognition'))

from model_utils import load_printed_model
import pandas as pd

# Cargar modelo
print("Cargando modelo...")
model, prep, mapping = load_printed_model()

# 1. Cargar una letra del dataset
print("\n" + "="*60)
print("1. LETRA DEL DATASET (funciona bien)")
print("="*60)

test_path = Path(__file__).parent.parent / 'FASE3_PrintedTextRecognition' / 'data' / 'test.csv'
df = pd.read_csv(test_path, nrows=1)
dataset_pixels = df.iloc[0, 1:].values
dataset_label = int(df.iloc[0, 0])
dataset_letter = mapping[dataset_label]

print(f"Letra: {dataset_letter} (label={dataset_label})")
print(f"Forma: {dataset_pixels.shape}")
print(f"Rango: [{dataset_pixels.min():.0f}, {dataset_pixels.max():.0f}]")
print(f"Mean: {dataset_pixels.mean():.1f}")
print(f"Std: {dataset_pixels.std():.1f}")

# Predecir
dataset_processed = prep.transform(dataset_pixels.reshape(1, -1))
dataset_pred = model.predict(dataset_processed)[0]
dataset_pred_letter = mapping.get(dataset_pred, '?')
print(f"\n✓ Predicción: {dataset_pred_letter} (label={dataset_pred})")
print(f"  Correcto: {dataset_pred == dataset_label}")

# Guardar imagen
img = Image.fromarray(dataset_pixels.reshape(28, 28).astype(np.uint8))
img.save('debug_letters/comparison_dataset.png')
print(f"  Guardada en: debug_letters/comparison_dataset.png")

# 2. Cargar la letra segmentada guardada
print("\n" + "="*60)
print("2. LETRA SEGMENTADA (falla)")
print("="*60)

seg_img = Image.open('debug_letters/letter_0_before_preprocess.png')
seg_pixels = np.array(seg_img).flatten()

print(f"Forma: {seg_pixels.shape}")
print(f"Rango: [{seg_pixels.min():.0f}, {seg_pixels.max():.0f}]")
print(f"Mean: {seg_pixels.mean():.1f}")
print(f"Std: {seg_pixels.std():.1f}")

# Predecir
seg_processed = prep.transform(seg_pixels.reshape(1, -1))
seg_pred = model.predict(seg_processed)[0]
seg_pred_letter = mapping.get(seg_pred, '?')
seg_proba = model.predict_proba(seg_processed)[0]
seg_idx = np.where(model.classes_ == seg_pred)[0][0]
seg_conf = seg_proba[seg_idx]

print(f"\n✗ Predicción: {seg_pred_letter} (label={seg_pred})")
print(f"  Confianza: {seg_conf:.3f}")

# 3. COMPARACIÓN
print("\n" + "="*60)
print("3. COMPARACIÓN")
print("="*60)

diff_mean = abs(dataset_pixels.mean() - seg_pixels.mean())
diff_std = abs(dataset_pixels.std() - seg_pixels.std())

print(f"Diferencia en Mean: {diff_mean:.1f}")
print(f"Diferencia en Std: {diff_std:.1f}")

# Comparar píxeles directamente
pixel_diff = np.abs(dataset_pixels.astype(float) - seg_pixels.astype(float))
print(f"Diferencia promedio por píxel: {pixel_diff.mean():.1f}")
print(f"Diferencia máxima por píxel: {pixel_diff.max():.1f}")

# Ver los features HOG
print("\n" + "="*60)
print("4. FEATURES HOG (después del preprocessor)")
print("="*60)

print(f"Dataset HOG shape: {dataset_processed.shape}")
print(f"Dataset HOG mean: {dataset_processed.mean():.3f}, std: {dataset_processed.std():.3f}")
print(f"Dataset HOG range: [{dataset_processed.min():.3f}, {dataset_processed.max():.3f}]")

print(f"\nSegmentada HOG shape: {seg_processed.shape}")
print(f"Segmentada HOG mean: {seg_processed.mean():.3f}, std: {seg_processed.std():.3f}")
print(f"Segmentada HOG range: [{seg_processed.min():.3f}, {seg_processed.max():.3f}]")

hog_diff = np.abs(dataset_processed - seg_processed)
print(f"\nDiferencia HOG promedio: {hog_diff.mean():.3f}")
print(f"Diferencia HOG máxima: {hog_diff.max():.3f}")

print("\n" + "="*60)
print("CONCLUSIÓN")
print("="*60)
print("Si las diferencias son grandes, el problema está en el preprocesamiento")
print("de las letras segmentadas antes de llegar al preprocessor HOG.")
