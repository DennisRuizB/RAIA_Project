# FASE 3: Reconocimiento de Texto Impreso

Sistema de reconocimiento OCR para texto impreso de ordenador con múltiples fuentes.

## 🎯 Características

- **Dataset sintético generado automáticamente**
- **Múltiples fuentes:** Arial, Times New Roman, Comic Sans, Courier, Calibri, Verdana, etc.
- **Variaciones:** Tamaños, negrita, cursiva, rotaciones, ruido
- **Alta precisión:** 98-99%+ accuracy esperada
- **Rápido:** Generación de 100K imágenes en ~5 minutos

## 🚀 Inicio Rápido

```powershell
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Generar dataset (100K imágenes, ~5 min)
python generate_dataset.py

# 3. Entrenar modelo (~5 min)
python train.py

# 4. Probar predicciones
python predict.py --text "HELLO"
```

## 📊 Dataset Generado

```
data/
├── train.csv          (80,000 samples)
├── test.csv           (20,000 samples)
├── mapping.txt        (A-Z labels)
└── samples/           (100 ejemplos visuales)
```

**Fuentes incluidas:**
- Arial
- Times New Roman
- Comic Sans MS
- Courier New
- Calibri
- Verdana
- Georgia
- Tahoma

**Variaciones:**
- Tamaños: 16, 18, 20, 22, 24 pts
- Estilos: Normal, Bold, Italic, Bold+Italic
- Rotación: ±5 grados
- Ruido gaussiano: 10% de probabilidad
- Blur: 10% de probabilidad

## 📈 Resultados Esperados

| Métrica | Valor |
|---------|-------|
| **Training Accuracy** | 99.5%+ |
| **Test Accuracy** | 98-99% |
| **Tiempo generación** | ~5 min |
| **Tiempo entrenamiento** | ~5 min |

## 🔧 Configuración

Edita `src/config.py` para ajustar:
- Número de samples por letra
- Fuentes a usar
- Variaciones de estilo
- Parámetros del modelo

## 📝 Ejemplo de Uso

```python
from src.predictor import PrintedTextPredictor

# Cargar modelo
predictor = PrintedTextPredictor()
predictor.load_model()

# Predecir letra desde imagen
letter = predictor.predict_from_image("letter.png")
print(f"Letra detectada: {letter}")

# Predecir palabra
word = predictor.predict_word("word.png")
print(f"Palabra detectada: {word}")
```

## 🎨 Ventajas vs Manuscrito (EMNIST)

| Aspecto | Manuscrito | Impreso |
|---------|-----------|---------|
| Accuracy | 85-90% | **98-99%** |
| Consistencia | Baja | **Alta** |
| Segmentación | Difícil | **Fácil** |
| Dataset | Limitado | **Infinito** |
| Aplicaciones | Cheques, formularios | PDFs, screenshots, escaneos |

## 📚 Estructura del Proyecto

```
FASE3_PrintedTextRecognition/
├── README.md
├── requirements.txt
├── generate_dataset.py      # Generar dataset sintético
├── train.py                 # Entrenar modelo
├── predict.py               # Hacer predicciones
├── src/
│   ├── config.py           # Configuración
│   ├── dataset_generator.py # Generador de imágenes
│   ├── trainer.py          # Entrenador del modelo
│   └── predictor.py        # Motor de predicción
├── data/                   # Datasets generados
├── models/                 # Modelos entrenados
└── logs/                   # Logs de entrenamiento
```

## ⚡ Comandos Útiles

```powershell
# Generar dataset pequeño (rápido, testing)
python generate_dataset.py --samples 1000

# Generar dataset completo
python generate_dataset.py --samples 5000

# Entrenar con MLP (más rápido)
python train.py --model mlp

# Entrenar con SVM (mejor accuracy)
python train.py --model svm

# Ver muestras generadas
python generate_dataset.py --preview
```

## 🎯 Próximos Pasos

1. Genera el dataset con `generate_dataset.py`
2. Entrena el modelo con `train.py`
3. Prueba predicciones con `predict.py`
4. Integra en tu proyecto de Streamlit
5. ¡Disfruta del OCR con 99% accuracy! 🚀
