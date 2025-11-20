"""
Pagina para visualizar el dataset
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.model_utils import load_sample_images, load_printed_model

st.title("📊 Visualizacion del Dataset")

st.markdown("""
### Explora el dataset de texto impreso sintetico

Este dataset contiene **208,000 imagenes** de letras impresas generadas con diferentes:
- **Fuentes**: Arial, Times New Roman, Comic Sans MS, Courier New, etc.
- **Estilos**: Normal, negrita, cursiva, negrita+cursiva
- **Augmentaciones**: Rotaciones, ruido, desenfoque
""")

# Cargar muestras
st.header("🎲 Muestras Aleatorias")

num_samples = st.slider("Numero de muestras a mostrar:", 6, 30, 12, 6)

if st.button("🔄 Cargar Nuevas Muestras"):
    with st.spinner("Cargando..."):
        images, labels = load_sample_images(num_samples=num_samples)
        
        if images is not None:
            st.session_state.viz_images = images
            st.session_state.viz_labels = labels
            st.success(f"✅ {len(images)} muestras cargadas!")

# Mostrar muestras si existen
if 'viz_images' in st.session_state:
    images = st.session_state.viz_images
    labels = st.session_state.viz_labels
    
    # Cargar mapeo
    _, _, label_mapping = load_printed_model()
    
    # Mostrar en grid
    cols_per_row = 6
    num_rows = (len(images) + cols_per_row - 1) // cols_per_row
    
    for row in range(num_rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            img_idx = row * cols_per_row + col_idx
            if img_idx < len(images):
                with cols[col_idx]:
                    letter = label_mapping[labels[img_idx]]
                    st.image(images[img_idx], caption=f"{letter}", width=100)

# Estadisticas del dataset
st.divider()
st.header("📈 Estadisticas del Dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Imagenes", "208,000")

with col2:
    st.metric("Entrenamiento", "166,400")

with col3:
    st.metric("Prueba", "41,600")

with col4:
    st.metric("Clases", "26 (A-Z)")

# Informacion adicional
st.divider()
st.header("ℹ️ Informacion del Dataset")

st.markdown("""
#### Generacion del Dataset

El dataset fue generado sinteticamente usando PIL (Python Imaging Library):

1. **Variaciones de Fuente y Estilo:**
   - 8 fuentes × 5 tamanos × 4 estilos = 160 combinaciones por letra
   
2. **Muestras por Combinacion:**
   - 100 muestras por cada combinacion
   - Total: 26 letras × 160 combinaciones × 100 = ~416,000 posibles
   
3. **Augmentaciones Aplicadas:**
   - **Rotaciones:** -5° a +5° (aleatoria)
   - **Ruido Gaussiano:** 10% de probabilidad
   - **Desenfoque:** 10% de probabilidad

4. **Division Train/Test:**
   - 80% entrenamiento (166,400 muestras)
   - 20% prueba (41,600 muestras)

#### Ventajas del Dataset Sintetico

✅ **Control Total:** Podemos generar exactamente las variaciones que necesitamos

✅ **Balanceado:** Todas las letras tienen la misma cantidad de muestras

✅ **Escalable:** Facil de generar mas datos si se necesita

✅ **Diverso:** Multiples fuentes, estilos y augmentaciones

✅ **Sin Problemas de Copyright:** Datos generados sinteticamente

#### Fuentes Incluidas

1. Arial
2. Times New Roman
3. Comic Sans MS
4. Courier New
5. Calibri
6. Verdana
7. Georgia
8. Tahoma

#### Casos de Uso

Este dataset es ideal para:
- Reconocimiento de texto en documentos impresos
- OCR de capturas de pantalla
- Procesamiento de PDFs escaneados
- Digitalizacion de documentos
- Reconocimiento de texto en imagenes
""")

# Comparacion con EMNIST
st.divider()
st.header("🔄 Comparacion con EMNIST")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 EMNIST (Manuscritas)")
    st.markdown("""
    - **Tipo:** Letras escritas a mano
    - **Muestras:** ~145,000
    - **Variabilidad:** Alta (diferentes personas)
    - **Precision:** ~89% (mas dificil)
    - **Uso:** Reconocimiento de escritura manual
    """)

with col2:
    st.subheader("🖨️ Dataset Impreso (FASE3)")
    st.markdown("""
    - **Tipo:** Letras impresas de ordenador
    - **Muestras:** 208,000
    - **Variabilidad:** Controlada (fuentes/estilos)
    - **Precision:** ~99% (mas facil)
    - **Uso:** Reconocimiento de texto impreso
    """)

st.info("""
💡 **Nota:** El dataset impreso tiene mayor precision porque las letras son mas 
consistentes y predecibles que la escritura manual.
""")
