"""
Aplicacion Streamlit para Reconocimiento de Letras Impresas
Integrado con el modelo de FASE3_PrintedTextRecognition
"""

import streamlit as st
from PIL import Image
import numpy as np
import sys
from pathlib import Path

# Configuracion de la pagina
st.set_page_config(
    page_title="Reconocimiento de Letras Impresas",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar utilidades personalizadas
from utils.model_utils import load_printed_model, predict_letter, model_exists, get_model_info

# Inicializar session_state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Titulo principal
st.title("🔤 Reconocimiento de Texto Impreso")
st.markdown("""
### 🎯 Sistema de Machine Learning con Texto Impreso

Esta aplicacion utiliza un **SVM con caracteristicas HOG** entrenado con 
**166,400 muestras de texto impreso** generadas sinteticamente.

**Caracteristicas:**
- 📝 Reconocimiento de texto completo (palabras y frases)
- 🔍 Segmentacion automatica de caracteres
- 📊 Multiples fuentes: Arial, Times New Roman, Comic Sans, etc.
- 🎨 Predicciones con alta precision (>99%)
- 📈 Metricas y estadisticas del modelo
""")

st.info("👈 Usa el **menu lateral** para navegar entre las diferentes paginas")

# Sidebar
with st.sidebar:
    st.header("📊 Informacion del Sistema")
    
    # Informacion del modelo
    st.subheader("🤖 Estado del Modelo")
    if model_exists():
        st.success("✅ Modelo entrenado disponible")
        model_info = get_model_info()
        if model_info:
            st.metric("Muestras de Entrenamiento", f"{model_info['train_samples']:,}")
            st.metric("Muestras de Prueba", f"{model_info['test_samples']:,}")
            st.metric("Clases", model_info['num_classes'])
    else:
        st.error("❌ Modelo no encontrado")
        st.info("El modelo debe estar en FASE3_PrintedTextRecognition/models/")
    
    st.divider()
    
    # Estadisticas de la sesion
    st.subheader("📈 Sesion Actual")
    st.metric("Predicciones realizadas", len(st.session_state.prediction_history))

# Contenido principal
st.header("🏠 Pagina Principal")

# Seccion de inicio rapido
col1, col2 = st.columns(2)

with col1:
    st.subheader("🚀 Inicio Rapido")
    st.markdown("""
    **1. Cargar Modelo**
    - El modelo se carga automaticamente al iniciar
    - Usa muestras pre-entrenadas de FASE3
    
    **2. Reconocer Texto**
    - Ve a la pagina **"🎨 Reconocer Texto"**
    - Carga una imagen con texto impreso
    - El sistema segmenta y reconoce automaticamente
    - Obten el texto completo con confianza por letra
    
    **3. Ver Dataset**
    - Explora las muestras del dataset
    - Visualiza diferentes fuentes y estilos
    - Analiza la distribucion de datos
    """)

with col2:
    st.subheader("📊 Sobre el Modelo")
    st.markdown("""
    **Arquitectura:**
    - SVM con kernel RBF (C=10.0)
    - Caracteristicas HOG (144 dimensiones)
    - Normalizacion estandar
    
    **Dataset:**
    - 8 fuentes diferentes
    - 5 tamanos de letra
    - 4 estilos (normal, negrita, cursiva, negrita+cursiva)
    - Augmentaciones: rotacion, ruido, desenfoque
    
    **Precision Esperada:**
    - ~99% en texto impreso claro
    - Robusto a diferentes fuentes y estilos
    """)

# Informacion del dataset
st.divider()
st.header("📚 Sobre el Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Clases",
        "26",
        help="Numero de letras (A-Z)"
    )

with col2:
    st.metric(
        "Tamano de Imagen",
        "28x28 pixeles",
        help="Imagenes en escala de grises"
    )

with col3:
    st.metric(
        "Muestras Totales",
        "208,000",
        help="166,400 entrenamiento + 41,600 prueba"
    )

st.markdown("""
### 📖 Acerca del Dataset Sintetico

Este dataset fue **generado sinteticamente** usando diferentes fuentes de Windows:

- **26 clases**: Una para cada letra del alfabeto (A-Z)
- **Imagenes en escala de grises**: 28x28 pixeles
- **8 fuentes**: Arial, Times New Roman, Comic Sans MS, Courier New, Calibri, Verdana, Georgia, Tahoma
- **Multiple estilos**: Normal, negrita, cursiva, negrita+cursiva
- **Augmentaciones**: Rotaciones (-5° a +5°), ruido gaussiano (10%), desenfoque (10%)

Este dataset es ideal para reconocimiento de texto impreso en documentos, capturas de pantalla, PDFs, etc.
""")

# Footer
st.divider()
st.caption("""
💡 **Proyecto RAIA - Reconocimiento de Letras Impresas**

✅ Modelo SVM entrenado | ✅ Dataset sintetico | ✅ Alta precision
""")
