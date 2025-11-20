"""
Utilidades para cargar y usar el modelo de FASE3
"""

import sys
from pathlib import Path
import pickle
import numpy as np
from PIL import Image

# Agregar rutas de FASE3 y FASE1
STREAMLIT_DIR = Path(__file__).parent.parent
FASE3_DIR = STREAMLIT_DIR.parent / "FASE3_PrintedTextRecognition"
FASE3_SRC = FASE3_DIR / "src"
FASE1_SRC = STREAMLIT_DIR.parent / "FASE1_SingleCharacterRecognition" / "src"

# Añadir rutas al path ANTES de importar
sys.path.insert(0, str(FASE3_SRC))
sys.path.insert(0, str(FASE1_SRC))

# Importar configuracion y componentes
from config import IMAGE_SIZE
from preprocessor import ImagePreprocessor

# Definir rutas directamente (para evitar problemas de importacion)
MODELS_DIR = FASE3_DIR / "models"
DATA_DIR = FASE3_DIR / "data"


# Cache global para modelos
_model_cache = {
    'model': None,
    'preprocessor': None,
    'label_mapping': None,
    'model_path': None  # Guardar path para detectar cambios
}


def model_exists():
    """Verificar si el modelo existe."""
    model_path = MODELS_DIR / "printed_letter_classifier.pkl"
    return model_path.exists()


def load_printed_model():
    """
    Cargar el modelo entrenado de FASE3.
    Usa cache para evitar recargar multiples veces.
    """
    global _model_cache
    
    model_path = MODELS_DIR / "printed_letter_classifier.pkl"
    preprocessor_path = MODELS_DIR / "printed_feature_scaler.pkl"
    mapping_path = DATA_DIR / "mapping.txt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    # Verificar si el modelo ha cambiado (timestamp diferente)
    import os
    current_mtime = os.path.getmtime(model_path)
    
    # Si ya esta cargado Y no ha cambiado, devolver del cache
    if (_model_cache['model'] is not None and 
        _model_cache['model_path'] == model_path and
        _model_cache.get('model_mtime') == current_mtime):
        print(f"[DEBUG] Usando modelo cacheado")
        return _model_cache['model'], _model_cache['preprocessor'], _model_cache['label_mapping']
    
    print(f"[DEBUG] Cargando modelo desde disco...")
    
    # Cargar modelo
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Cargar preprocessor
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # CRÍTICO: Forzar config de FASE3 (sin rotación)
    # El preprocessor puede tener el config de FASE1 embebido
    # Necesitamos sobrescribir el método _transform_images
    def no_transform_images(self, X):
        """Transform sin rotación ni flip para texto impreso."""
        n_samples = X.shape[0]
        images = X.reshape(n_samples, 28, 28)
        # NO aplicar rotación ni flip
        return images
    
    # Reemplazar el método en la instancia
    import types
    preprocessor._transform_images = types.MethodType(no_transform_images, preprocessor)
    print(f"[DEBUG] Transformaciones EMNIST desactivadas (rotation_k=0, flip_lr=False)")
    
    # DEBUG: Verificar configuración del preprocessor
    import os
    model_mtime = os.path.getmtime(model_path)
    print(f"[DEBUG] Modelo cargado: {model_path}")
    print(f"[DEBUG] Modelo modificado: {model_mtime}")
    print(f"[DEBUG] Preprocessor tiene rotation_k: {hasattr(preprocessor, 'rotation_k')}")
    
    # Verificar que el preprocessor NO tenga transformaciones EMNIST
    # El modelo nuevo fue entrenado sin rotación
    try:
        # Intentar acceder al config que usa el preprocessor
        import sys
        for module_name in list(sys.modules.keys()):
            if 'config' in module_name.lower():
                module = sys.modules[module_name]
                if hasattr(module, 'PREPROCESSING_CONFIG'):
                    config = module.PREPROCESSING_CONFIG
                    print(f"[DEBUG] Config encontrado en {module_name}")
                    print(f"[DEBUG] rotation_k = {config.get('rotation_k', 'N/A')}")
                    print(f"[DEBUG] flip_lr = {config.get('flip_lr', 'N/A')}")
                    break
    except Exception as e:
        print(f"[DEBUG] No se pudo verificar config: {e}")
    
    # Cargar mapeo de etiquetas
    label_mapping = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            label, letter = line.strip().split()
            label_mapping[int(label)] = letter
    
    # Guardar en cache
    _model_cache['model'] = model
    _model_cache['preprocessor'] = preprocessor
    _model_cache['label_mapping'] = label_mapping
    _model_cache['model_path'] = model_path
    _model_cache['model_mtime'] = current_mtime
    
    print(f"[DEBUG] Modelo cargado exitosamente")
    
    return model, preprocessor, label_mapping


def predict_letter(image_array):
    """
    Predecir la letra de una imagen.
    
    Args:
        image_array: Numpy array de la imagen (28x28 o similar)
    
    Returns:
        dict con 'letter' y 'probabilities'
    """
    model, preprocessor, label_mapping = load_printed_model()
    
    # Asegurar que la imagen es 28x28
    if image_array.shape != (28, 28):
        img = Image.fromarray(image_array.astype(np.uint8))
        img = img.resize((28, 28), Image.LANCZOS)
        image_array = np.array(img)
    
    # Aplanar la imagen
    image_flat = image_array.flatten().reshape(1, -1)
    
    # Preprocesar
    image_processed = preprocessor.transform(image_flat)
    
    # Predecir
    prediction = model.predict(image_processed)[0]
    probabilities = model.predict_proba(image_processed)[0] if hasattr(model, 'predict_proba') else None
    
    # El modelo predice clases 1-26, pero el índice del array es 0-25
    # Necesitamos obtener la posición correcta en el array de probabilidades
    predicted_letter = label_mapping.get(prediction, '?')
    
    # Crear diccionario de probabilidades por letra
    prob_dict = {}
    if probabilities is not None:
        # model.classes_ contiene las clases reales (1-26)
        for i, class_label in enumerate(model.classes_):
            letter = label_mapping.get(class_label, '?')
            prob_dict[letter] = probabilities[i]
        
        # La confianza de la predicción es la probabilidad de la clase predicha
        # Encontrar el índice de la clase predicha en model.classes_
        pred_idx = np.where(model.classes_ == prediction)[0][0]
        confidence = probabilities[pred_idx]
    else:
        confidence = 1.0
    
    return {
        'letter': predicted_letter,
        'confidence': confidence,
        'probabilities': prob_dict
    }


def get_model_info():
    """Obtener informacion del modelo."""
    if not model_exists():
        return None
    
    return {
        'train_samples': 166400,
        'test_samples': 41600,
        'num_classes': 26,
        'image_size': (28, 28),
        'model_type': 'SVM',
        'features': 'HOG'
    }


def load_sample_images(num_samples=10):
    """
    Cargar imagenes de muestra del dataset de prueba.
    
    Args:
        num_samples: Numero de muestras a cargar
    
    Returns:
        tuple (images, labels) o (None, None) si hay error
    """
    import pandas as pd
    
    test_path = DATA_DIR / "test.csv"
    
    if not test_path.exists():
        return None, None
    
    try:
        # Cargar solo las primeras filas necesarias
        df = pd.read_csv(test_path, nrows=num_samples)
        
        # Separar imagenes y etiquetas
        images = df.iloc[:, 1:].values.reshape(-1, 28, 28)
        labels = df.iloc[:, 0].values
        
        return images, labels
    except Exception as e:
        print(f"Error cargando muestras: {e}")
        return None, None
