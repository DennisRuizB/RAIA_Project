"""
Pagina con informacion del modelo
"""

import streamlit as st
from utils.model_utils import get_model_info, model_exists

st.title("🤖 Informacion del Modelo")

if not model_exists():
    st.error("❌ Modelo no encontrado")
    st.stop()

model_info = get_model_info()

# Metricas principales
st.header("📊 Metricas del Modelo")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Tipo de Modelo", model_info['model_type'])

with col2:
    st.metric("Caracteristicas", model_info['features'])

with col3:
    st.metric("Muestras Train", f"{model_info['train_samples']:,}")

with col4:
    st.metric("Muestras Test", f"{model_info['test_samples']:,}")

# Arquitectura
st.divider()
st.header("🏗️ Arquitectura del Modelo")

st.markdown("""
### SVM (Support Vector Machine) con Caracteristicas HOG

El modelo utiliza una arquitectura clasica pero efectiva:

#### 1. **Preprocesamiento de Imagen**
   - Entrada: Imagen 28×28 pixeles en escala de grises
   - Normalizacion: Escala de 0-255 a rango normalizado

#### 2. **Extraccion de Caracteristicas HOG**
   - **HOG** (Histogram of Oriented Gradients)
   - **Parametros:**
     - Orientaciones: 9
     - Pixeles por celda: 8×8
     - Celdas por bloque: 2×2
   - **Salida:** 144 caracteristicas por imagen

#### 3. **Clasificador SVM**
   - **Kernel:** RBF (Radial Basis Function)
   - **C:** 10.0 (parametro de regularizacion)
   - **Gamma:** scale (automatico)
   - **Probabilidades:** Habilitado

#### 4. **Prediccion**
   - Salida: Clase predicha (0-25 → A-Z)
   - Probabilidades: Confianza para cada clase
""")

# Comparacion con CNN
st.divider()
st.header("🔄 SVM vs CNN")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 SVM + HOG (Este Modelo)")
    st.markdown("""
    **Ventajas:**
    - ✅ Entrenamiento rapido (~5-10 min)
    - ✅ Menos datos necesarios
    - ✅ Interpretable (caracteristicas HOG)
    - ✅ Sin GPU necesaria
    - ✅ Buen rendimiento en texto impreso
    
    **Desventajas:**
    - ⚠️ Requiere feature engineering
    - ⚠️ Menos flexible que CNNs
    """)

with col2:
    st.subheader("🧠 CNN (Deep Learning)")
    st.markdown("""
    **Ventajas:**
    - ✅ Aprende caracteristicas automaticamente
    - ✅ Mejor para datos complejos
    - ✅ State-of-the-art en vision
    
    **Desventajas:**
    - ⚠️ Entrenamiento lento (horas)
    - ⚠️ Necesita mas datos
    - ⚠️ Requiere GPU para ser eficiente
    - ⚠️ Menos interpretable
    """)

# Pipeline completo
st.divider()
st.header("🔄 Pipeline de Prediccion")

st.markdown("""
```
Imagen 28×28
    ↓
Normalizar (0-255 → valores estandar)
    ↓
Extraer caracteristicas HOG (144 features)
    ↓
Normalizar caracteristicas (StandardScaler)
    ↓
SVM Clasificador
    ↓
Prediccion + Probabilidades
```

**Tiempo de inferencia:** < 10 ms por imagen
""")

# Caracteristicas HOG
st.divider()
st.header("🔍 ¿Que es HOG?")

st.markdown("""
### Histogram of Oriented Gradients (HOG)

HOG es un descriptor de caracteristicas usado en vision por computadora para detectar objetos.

#### Como Funciona:

1. **Divide la imagen en celdas** (8×8 pixeles)
2. **Calcula gradientes** (direccion y magnitud de los cambios de intensidad)
3. **Crea histogramas de orientaciones** para cada celda
4. **Normaliza por bloques** (2×2 celdas)
5. **Concatena todos los histogramas** → Vector de caracteristicas

#### Ventajas para Reconocimiento de Letras:

- ✅ **Invariante a iluminacion:** Se basa en gradientes, no en valores absolutos
- ✅ **Captura la forma:** Los gradientes detectan bordes y formas
- ✅ **Robusto:** Funciona bien con pequenas variaciones
- ✅ **Eficiente:** Rapido de calcular

#### Resultado:

Para una imagen 28×28:
- Celdas: 3×3 (cada una de 8×8 pixeles)
- Bloques: 2×2
- Orientaciones: 9
- **Total: 144 caracteristicas**

Estas 144 caracteristicas capturan la esencia de la forma de la letra.
""")

# Entrenamiento
st.divider()
st.header("⏱️ Informacion de Entrenamiento")

st.markdown(f"""
### Proceso de Entrenamiento

El modelo fue entrenado con los siguientes parametros:

- **Dataset:** {model_info['train_samples']:,} imagenes de entrenamiento
- **Validacion:** {model_info['test_samples']:,} imagenes de prueba
- **Tiempo aproximado:** ~5-10 minutos (CPU)
- **Precision esperada:** >99% en texto impreso claro

### Archivos del Modelo

Los modelos entrenados se encuentran en:
```
FASE3_PrintedTextRecognition/models/
├── printed_letter_classifier.pkl (modelo SVM)
└── printed_feature_scaler.pkl (normalizador)
```

### Dataset

El dataset se encuentra en:
```
FASE3_PrintedTextRecognition/data/
├── train.csv (166,400 muestras, 773 MB)
├── test.csv (41,600 muestras, 193 MB)
├── mapping.txt (mapeo label → letra)
└── samples/ (imagenes de ejemplo)
```
""")
