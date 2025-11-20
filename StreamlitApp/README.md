# Aplicacion Streamlit - Reconocimiento de Letras Impresas

Aplicacion web integrada con el modelo de FASE3_PrintedTextRecognition.

## Caracteristicas

- 🎨 **Reconocimiento de letras**: Carga imagenes y obtén predicciones
- 📊 **Visualizacion del dataset**: Explora muestras del dataset sintetico
- 🤖 **Informacion del modelo**: Detalles de arquitectura y metricas
- 📈 **Alta precision**: >99% en texto impreso claro

## Instalacion

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Asegurate de que el modelo este entrenado en FASE3:
```bash
cd ../FASE3_PrintedTextRecognition
python train.py
```

2. Ejecutar la aplicacion:
```bash
streamlit run app.py
```

3. Navega a `http://localhost:8501` en tu navegador

## Estructura

```
StreamlitApp/
├── app.py                      # Pagina principal
├── requirements.txt            # Dependencias
├── README.md                   # Este archivo
├── pages/
│   ├── 1_🎨_Reconocer_Letra.py  # Predicciones
│   ├── 2_📊_Dataset.py          # Visualizacion
│   └── 3_🤖_Modelo.py           # Informacion
└── utils/
    └── model_utils.py          # Utilidades del modelo
```

## Requisitos

- Python 3.8+
- Modelo entrenado en `FASE3_PrintedTextRecognition/models/`
- Dataset en `FASE3_PrintedTextRecognition/data/`
