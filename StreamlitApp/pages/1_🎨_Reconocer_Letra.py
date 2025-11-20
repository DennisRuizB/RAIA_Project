"""
Pagina para reconocer letras impresas
"""

import streamlit as st
from PIL import Image
import numpy as np
import io

from utils.model_utils import predict_letter, model_exists, load_sample_images

st.title("🎨 Reconocer Letra Impresa")

if not model_exists():
    st.error("❌ Modelo no encontrado. Asegurate de que el modelo este entrenado en FASE3.")
    st.stop()

st.markdown("""
### Carga una imagen de una letra impresa para reconocerla

Puedes usar:
- **Captura de pantalla** de una letra
- **Imagen de un documento** escaneado
- **Foto de texto impreso**
- **Muestras del dataset** (boton abajo)
""")

# Tabs para diferentes opciones
tab1, tab2 = st.tabs(["📤 Cargar Imagen", "🎲 Usar Muestra Aleatoria"])

with tab1:
    # Cargar imagen del usuario
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de una letra (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        # Cargar y mostrar la imagen
        image = Image.open(uploaded_file).convert('L')  # Convertir a escala de grises
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Imagen Original", use_container_width=True)
        
        with col2:
            # Redimensionar a 28x28
            image_resized = image.resize((28, 28), Image.LANCZOS)
            image_array = np.array(image_resized)
            
            st.image(image_resized, caption="Imagen Procesada (28x28)", width=150)
            
            # Predecir
            if st.button("🔍 Predecir", type="primary"):
                with st.spinner("Analizando..."):
                    result = predict_letter(image_array)
                    
                    # Mostrar resultado
                    st.success(f"### Prediccion: **{result['letter']}**")
                    st.metric("Confianza", f"{result['confidence']*100:.2f}%")
                    
                    # Mostrar top 5 probabilidades
                    if result['probabilities']:
                        st.markdown("#### Top 5 Probabilidades:")
                        sorted_probs = sorted(
                            result['probabilities'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:5]
                        
                        for letter, prob in sorted_probs:
                            st.progress(prob, text=f"{letter}: {prob*100:.2f}%")
                    
                    # Guardar en historial
                    st.session_state.prediction_history.append({
                        'letter': result['letter'],
                        'confidence': result['confidence']
                    })

with tab2:
    # Cargar muestras aleatorias
    st.markdown("### 🎲 Prueba con muestras del dataset")
    
    if st.button("🔄 Cargar Muestras Aleatorias"):
        with st.spinner("Cargando muestras..."):
            images, labels = load_sample_images(num_samples=12)
            
            if images is not None:
                st.session_state.sample_images = images
                st.session_state.sample_labels = labels
                st.success("✅ Muestras cargadas!")
    
    if 'sample_images' in st.session_state:
        images = st.session_state.sample_images
        labels = st.session_state.sample_labels
        
        # Crear mapeo de etiquetas
        from utils.model_utils import load_printed_model
        _, _, label_mapping = load_printed_model()
        
        # Mostrar en grid
        st.markdown("#### Selecciona una muestra para predecir:")
        
        cols = st.columns(6)
        for i in range(min(12, len(images))):
            with cols[i % 6]:
                true_letter = label_mapping[labels[i]]
                
                # Mostrar imagen (asegurar que esté en rango 0-255 uint8)
                img_display = images[i].astype(np.uint8)
                st.image(img_display, caption=f"Real: {true_letter}", width=80, clamp=True)
                
                # Boton para predecir
                if st.button(f"Predecir", key=f"pred_{i}"):
                    result = predict_letter(images[i])
                    
                    # Mostrar resultado en un expander
                    with st.expander(f"Resultado #{i+1}", expanded=True):
                        st.markdown(f"**Prediccion:** {result['letter']}")
                        st.markdown(f"**Real:** {true_letter}")
                        st.markdown(f"**Confianza:** {result['confidence']*100:.2f}%")
                        
                        if result['letter'] == true_letter:
                            st.success("✅ Correcto!")
                        else:
                            st.error("❌ Incorrecto")

# Historial de predicciones
st.divider()
st.header("📊 Historial de Predicciones")

if st.session_state.prediction_history:
    st.markdown(f"**Total de predicciones:** {len(st.session_state.prediction_history)}")
    
    # Calcular estadisticas
    avg_confidence = np.mean([p['confidence'] for p in st.session_state.prediction_history])
    st.metric("Confianza Promedio", f"{avg_confidence*100:.2f}%")
    
    # Mostrar ultimas 10
    st.markdown("#### Ultimas 10 predicciones:")
    recent = st.session_state.prediction_history[-10:][::-1]
    
    for i, pred in enumerate(recent):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(f"{i+1}. Letra: {pred['letter']}")
        with col2:
            st.text(f"{pred['confidence']*100:.1f}%")
    
    if st.button("🗑️ Limpiar Historial"):
        st.session_state.prediction_history = []
        st.rerun()
else:
    st.info("No hay predicciones todavia. ¡Haz tu primera prediccion arriba!")
