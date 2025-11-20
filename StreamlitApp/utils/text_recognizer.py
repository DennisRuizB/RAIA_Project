"""
Reconocedor de texto completo - Integra segmentacion con modelo de FASE3
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Importar segmentador simple (sin dependencias de FASE2)
from .simple_segmenter import SimpleImageSegmenter
from .model_utils import load_printed_model


class TextRecognizer:
    """Reconocedor de texto completo."""
    
    def __init__(self):
        self.segmenter = SimpleImageSegmenter()
        self.model, self.preprocessor, self.label_mapping = load_printed_model()
    
    def recognize_text(self, image_path_or_array, debug=False):
        """
        Reconocer texto completo de una imagen.
        
        Args:
            image_path_or_array: Ruta a imagen o numpy array
            debug: Si True, retorna informacion de debug
        
        Returns:
            dict con 'text', 'letters', 'confidences', y opcionalmente 'debug_info'
        """
        # Cargar imagen si es necesario
        if isinstance(image_path_or_array, (str, Path)):
            image = Image.open(image_path_or_array).convert('L')
            image_array = np.array(image)
        else:
            image_array = image_path_or_array
        
        # Segmentar caracteres
        letters_data = self.segmenter.segment_word(image_array)
        
        if not letters_data or len(letters_data) == 0:
            return {
                'text': '',
                'letters': [],
                'confidences': [],
                'num_letters': 0,
                'message': 'No se detectaron letras en la imagen'
            }
        
        # Predecir cada letra
        recognized_letters = []
        confidences = []
        debug_info = []
        
        for i, letter_img in enumerate(letters_data):
            # Asegurar que sea 28x28
            if letter_img.shape != (28, 28):
                img = Image.fromarray(letter_img.astype(np.uint8))
                img = img.resize((28, 28), Image.LANCZOS)
                letter_img = np.array(img)
            
            # IMPORTANTE: Invertir colores para que coincida con dataset
            # Segmentador: fondo negro (0), texto blanco (255)
            # Dataset: fondo blanco (255), texto negro (0)
            letter_img = 255 - letter_img
            
            # Ya está. No hacer nada más. El dataset tampoco tiene preprocesamiento extra.
            
            # DEBUG: Guardar la primera letra para inspección
            if debug and i == 0:
                from PIL import Image
                import os
                debug_dir = Path("debug_letters")
                debug_dir.mkdir(exist_ok=True)
                
                # Guardar imagen antes de preprocesar
                img_before = Image.fromarray(letter_img.astype(np.uint8))
                img_before.save(debug_dir / "letter_0_before_preprocess.png")
                print(f"[DEBUG] Letra guardada en {debug_dir / 'letter_0_before_preprocess.png'}")
                print(f"[DEBUG] Letra 0: mean={np.mean(letter_img):.1f}, min={np.min(letter_img)}, max={np.max(letter_img)}")
            
            # Aplanar y preprocesar
            letter_flat = letter_img.flatten().reshape(1, -1)
            letter_processed = self.preprocessor.transform(letter_flat)
            
            # Predecir
            prediction = self.model.predict(letter_processed)[0]
            proba = self.model.predict_proba(letter_processed)[0] if hasattr(self.model, 'predict_proba') else None
            
            # Obtener letra (manejar mapeo 1-26)
            letter = self.label_mapping.get(prediction, '?')
            
            # Obtener confianza usando model.classes_
            if proba is not None:
                pred_idx = np.where(self.model.classes_ == prediction)[0]
                confidence = proba[pred_idx[0]] if len(pred_idx) > 0 else 0.0
            else:
                confidence = 1.0
            
            recognized_letters.append(letter)
            confidences.append(confidence)
            
            if debug:
                debug_info.append({
                    'position': i,
                    'letter': letter,
                    'confidence': confidence,
                    'shape': letter_img.shape
                })
        
        # Formar texto completo
        text = ''.join(recognized_letters)
        
        result = {
            'text': text,
            'letters': recognized_letters,
            'confidences': confidences,
            'num_letters': len(recognized_letters)
        }
        
        if debug:
            result['debug_info'] = debug_info
            result['segmented_images'] = letters_data
        
        return result
    
    def recognize_with_details(self, image_path_or_array):
        """
        Reconocer texto con informacion detallada de cada letra.
        
        Returns:
            dict con informacion completa
        """
        # Activar debug en el segmentador
        self.segmenter.debug = True
        
        result = self.recognize_text(image_path_or_array, debug=True)
        
        # Agregar info de debug de la imagen original
        if isinstance(image_path_or_array, np.ndarray):
            result['image_info'] = {
                'shape': image_path_or_array.shape,
                'dtype': str(image_path_or_array.dtype),
                'min': int(np.min(image_path_or_array)),
                'max': int(np.max(image_path_or_array)),
                'mean': float(np.mean(image_path_or_array))
            }
        
        # Desactivar debug
        self.segmenter.debug = False
        
        return result
