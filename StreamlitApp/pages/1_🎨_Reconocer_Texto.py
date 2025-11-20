"""
Pagina para reconocer texto impreso completo
"""

import streamlit as st
from PIL import Image
import numpy as np

from utils.model_utils import model_exists
from utils.text_recognizer import TextRecognizer

st.title("🎨 Reconocer Texto Impreso")

if not model_exists():
    st.error("❌ Modelo no encontrado. Asegurate de que el modelo este entrenado en FASE3.")
    st.stop()

st.markdown("""
### Carga una imagen con texto impreso para reconocerlo

El sistema:
1. **Segmenta** la imagen en letras individuales
2. **Reconoce** cada letra usando el modelo entrenado
3. **Forma** el texto completo

Funciona con:
- **Capturas de pantalla** de texto
- **Documentos escaneados**
- **Fotos de texto impreso**
- **Imagenes de palabras o frases**
""")

# Inicializar el reconocedor (con cache)
@st.cache_resource
def get_recognizer():
    return TextRecognizer()

recognizer = get_recognizer()

# Tabs para diferentes opciones
tab1, tab2 = st.tabs(["📤 Cargar Imagen", "🎲 Usar Imagen de Prueba"])

with tab1:
    # Cargar imagen del usuario
    uploaded_file = st.file_uploader(
        "Selecciona una imagen con texto impreso (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        # Cargar y mostrar la imagen
        image = Image.open(uploaded_file).convert('L')  # Convertir a escala de grises
        image_array = np.array(image)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Imagen Original", width='stretch')
        
        with col2:
            # Boton para predecir
            if st.button("🔍 Reconocer Texto", type="primary", key="predict_upload"):
                with st.spinner("Segmentando y reconociendo..."):
                    try:
                        result = recognizer.recognize_with_details(image_array)
                        
                        if result['num_letters'] == 0:
                            st.warning(result.get('message', 'No se detectaron letras'))
                        else:
                            # Mostrar resultado principal
                            st.success("### Texto Reconocido:")
                            st.markdown(f"# `{result['text']}`")
                            
                            # Metricas
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Letras Detectadas", result['num_letters'])
                            with col_b:
                                avg_conf = np.mean(result['confidences'])
                                st.metric("Confianza Promedio", f"{avg_conf*100:.1f}%")
                            with col_c:
                                min_conf = np.min(result['confidences'])
                                st.metric("Confianza Minima", f"{min_conf*100:.1f}%")
                            
                            # Detalles por letra
                            st.markdown("#### 📝 Detalle por Letra:")
                            
                            # Mostrar letras segmentadas en grid
                            if 'segmented_images' in result:
                                cols = st.columns(min(len(result['segmented_images']), 8))
                                for i, (letter, conf, img) in enumerate(zip(
                                    result['letters'], 
                                    result['confidences'],
                                    result['segmented_images']
                                )):
                                    with cols[i % 8]:
                                        st.image(img, caption=f"{letter}\n{conf*100:.0f}%", width=60)
                            
                            # Tabla de detalles
                            with st.expander("Ver tabla de detalles"):
                                import pandas as pd
                                df = pd.DataFrame({
                                    'Posicion': range(1, len(result['letters']) + 1),
                                    'Letra': result['letters'],
                                    'Confianza': [f"{c*100:.2f}%" for c in result['confidences']]
                                })
                                st.dataframe(df, width='stretch')
                            
                            # Guardar en historial
                            if 'text_history' not in st.session_state:
                                st.session_state.text_history = []
                            
                            st.session_state.text_history.append({
                                'text': result['text'],
                                'num_letters': result['num_letters'],
                                'avg_confidence': avg_conf
                            })
                    
                    except Exception as e:
                        st.error(f"Error al procesar la imagen: {str(e)}")
                        st.exception(e)

with tab2:
    # Imagenes de prueba (ejemplos simples)
    st.markdown("### 🎲 Genera una imagen de prueba")
    
    test_text = st.text_input("Texto para generar:", "HOLA", max_chars=20)
    font_size = st.slider("Tamano de fuente:", 20, 60, 40)
    
    if st.button("🖼️ Generar Imagen"):
        from PIL import ImageDraw, ImageFont
        
        # Crear imagen con el texto
        img_width = len(test_text) * font_size
        img_height = font_size + 20
        
        test_image = Image.new('L', (img_width, img_height), color=255)
        draw = ImageDraw.Draw(test_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Dibujar texto
        draw.text((10, 10), test_text, fill=0, font=font)
        
        st.session_state.test_image = np.array(test_image)
    
    # Mostrar y reconocer imagen de prueba
    if 'test_image' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(st.session_state.test_image, caption="Imagen Generada", width='stretch')
        
        with col2:
            if st.button("🔍 Reconocer", type="primary", key="predict_test"):
                with st.spinner("Reconociendo..."):
                    try:
                        result = recognizer.recognize_with_details(st.session_state.test_image)
                        
                        # Mostrar info de debug
                        with st.expander("🔍 Info de Debug", expanded=False):
                            if 'image_info' in result:
                                st.json(result['image_info'])
                        
                        if result['num_letters'] > 0:
                            st.success(f"### Resultado: `{result['text']}`")
                            
                            avg_conf = np.mean(result['confidences'])
                            st.metric("Confianza", f"{avg_conf*100:.1f}%")
                            
                            # Comparacion
                            original = test_text.upper()
                            recognized = result['text']
                            
                            if original == recognized:
                                st.success("✅ ¡Perfecto! Reconocimiento correcto")
                            else:
                                st.warning(f"⚠️ Esperado: {original} | Obtenido: {recognized}")
                        else:
                            st.warning("No se detectaron letras")
                            st.info(result.get('message', 'Sin mensaje de error'))
                            
                            # Mostrar más debug
                            if 'image_info' in result:
                                st.write("**Información de la imagen:**")
                                st.json(result['image_info'])
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# Historial
st.divider()
st.header("📊 Historial de Reconocimientos")

if 'text_history' in st.session_state and st.session_state.text_history:
    st.markdown(f"**Total de reconocimientos:** {len(st.session_state.text_history)}")
    
    # Mostrar ultimos 10
    st.markdown("#### Ultimos reconocimientos:")
    recent = st.session_state.text_history[-10:][::-1]
    
    for i, rec in enumerate(recent):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.text(f"{i+1}. {rec['text']}")
        with col2:
            st.text(f"{rec['num_letters']} letras")
        with col3:
            st.text(f"{rec['avg_confidence']*100:.1f}%")
    
    if st.button("🗑️ Limpiar Historial"):
        st.session_state.text_history = []
        st.rerun()
else:
    st.info("No hay reconocimientos todavia. ¡Prueba reconocer tu primer texto arriba!")

# Consejos
st.divider()
st.markdown("""
### 💡 Consejos para Mejores Resultados

- ✅ Usa **texto impreso claro** (no manuscrito)
- ✅ Asegura **buena iluminacion** y contraste
- ✅ Evita **fondos ruidosos** o texturas
- ✅ Prefiere **imagenes horizontales** con texto en una linea
- ✅ El texto debe estar **bien enfocado**

**Fuentes que funcionan bien:**
Arial, Times New Roman, Calibri, Verdana, Comic Sans, Courier
""")
