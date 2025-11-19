# FASE 2: Word Recognition System

## 📋 Project Overview

Advanced word-level handwriting recognition system that builds upon Phase 1's single-character classifier. Segments word images into individual letters and recognizes complete words.

**Author:** Senior ML Engineer  
**Date:** 2025  
**Status:** Production Ready  
**Dependencies:** Requires FASE1 (Phase 1) to be trained first

---

## 🎯 Features

- **Image Segmentation**: Projection-based algorithm to extract individual characters from word images
- **Word-Level Recognition**: Combines character predictions into complete words
- **Confidence Scoring**: Per-character and word-level confidence metrics
- **Robust Preprocessing**: Handles various image qualities and orientations
- **Reusable Architecture**: Leverages Phase 1 classifier without retraining
- **Comprehensive Logging**: Detailed tracking of segmentation and recognition steps

---

## 🏗️ Architecture

### Project Structure

```
FASE2_WordRecognition/
├── src/                           # Source code modules
│   ├── config.py                 # Configuration for segmentation & recognition
│   ├── logger.py                 # Logging utilities
│   ├── image_segmenter.py        # Character segmentation from word images
│   └── word_recognizer.py        # Complete word recognition pipeline
├── output/                        # Output directory
│   ├── results/                  # Recognition results
│   └── segmented_letters/        # Debug: individual character images
├── main.py                       # Main demo/testing script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### Design Philosophy

**1. Reuse Phase 1 Classifier**
- **Rationale**: No need to retrain - leverage existing high-accuracy model
- **Benefit**: Fast deployment, consistent predictions
- **Implementation**: Import FASE1's InferenceEngine directly

**2. Projection-Based Segmentation**
- **Rationale**: Simple, interpretable, works well for printed/clear handwriting
- **Method**: Vertical projection profile analysis
- **Alternative**: Contour-based (can be added via config)

**3. Modular Pipeline**
```
Word Image → Segmentation → Character Recognition → Word Assembly
```
Each stage is independent and testable.

**4. Configuration-Driven**
- All parameters (thresholds, padding, confidence) configurable in `config.py`
- Easy to tune without code changes

---

## 🚀 Quick Start

### Prerequisites

**CRITICAL:** Phase 1 must be trained first!

```powershell
# 1. Train Phase 1 model
cd ../FASE1_SingleCharacterRecognition
python main.py

# 2. Verify model exists
# Should see: models/emnist_letter_classifier.pkl
```

### Installation

```powershell
# Install dependencies
pip install -r requirements.txt
```

### Running Demos

**Demo 1: EMNIST-based word recognition**
```powershell
python main.py --demo
```

Creates synthetic words from EMNIST letters and recognizes them.

**Demo 2: Interactive mode**
```powershell
python main.py --interactive
```

Enter image paths or commands interactively.

---

## 📊 How It Works

### Step 1: Image Segmentation

```
Input: Word image (variable width x 28 height)
        ┌────────────────────────┐
        │  H  E  L  L  O        │
        └────────────────────────┘

Algorithm: Vertical Projection Profile
- Sum pixels vertically in each column
- Detect transitions (white → text → white)
- Extract character bounding boxes

Output: List of character images (28x28 each)
  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐
  │H │ │E │ │L │ │L │ │O │
  └──┘ └──┘ └──┘ └──┘ └──┘
```

### Step 2: Character Recognition

Each 28x28 character is:
1. Flattened to 784-pixel vector
2. Passed through Phase 1 preprocessor (HOG extraction, normalization)
3. Classified by Phase 1 SVM model
4. Returns letter + confidence score

### Step 3: Word Assembly

```python
letters = ['H', 'E', 'L', 'L', 'O']
confidences = [0.98, 0.95, 0.92, 0.88, 0.97]

# Simple concatenation
word = "HELLO"

# Optional: Filter low-confidence predictions
if confidence < threshold:
    letter = "?"
```

---

## ⚙️ Configuration

Edit `src/config.py`:

### Segmentation Parameters

```python
SEGMENTATION_CONFIG = {
    "binarization_method": "otsu",     # Thresholding method
    "min_char_width": 5,               # Minimum character width (px)
    "max_char_width": 50,              # Maximum character width (px)
    "char_spacing_threshold": 3,       # Space between chars (px)
    "projection_threshold": 0.1,       # Column emptiness threshold
    "padding": 2,                      # Extra padding around chars
}
```

### Recognition Parameters

```python
WORD_RECOGNITION_CONFIG = {
    "use_confidence_threshold": True,
    "min_confidence": 0.3,             # Below this → "?"
    "unknown_char_placeholder": "?",
    "force_uppercase": True,           # Convert to uppercase
}
```

---

## 🧪 Usage Examples

### Example 1: Recognize Custom Word Image

```python
from src.word_recognizer import WordRecognizer
from skimage import io
import numpy as np

# Initialize
recognizer = WordRecognizer()
recognizer.load_model()

# Load your word image
image = io.imread("path/to/word_image.png")

# Recognize
word = recognizer.recognize_word(image)
print(f"Recognized: {word}")
```

### Example 2: Detailed Predictions

```python
# Get detailed info with alternatives
details = recognizer.get_detailed_prediction(image, top_k=3)

print(f"Word: {details['word']}")
for char_info in details['characters']:
    print(f"  Char {char_info['char_index']}: {char_info['top_prediction']}")
    print(f"    Confidence: {char_info['confidence']:.2f}")
    print(f"    Alternatives: {char_info['alternatives']}")
```

### Example 3: Batch Processing

```python
# Recognize multiple word images
images = [image1, image2, image3]
image_ids = ["word1", "word2", "word3"]

results = recognizer.recognize_batch(images, image_ids)

for result in results:
    print(f"{result['image_id']}: {result['word']} "
          f"(confidence: {result['avg_confidence']:.2f})")
```

---

## 📈 Performance Considerations

### Segmentation Quality

**Works well on:**
- ✅ Printed-style text with clear spacing
- ✅ Consistent character heights
- ✅ High contrast images

**Challenges:**
- ⚠️ Cursive/connected handwriting (characters touch)
- ⚠️ Variable character sizes
- ⚠️ Low-quality/blurry images

**Solutions:**
- Adjust `char_spacing_threshold` in config
- Use morphological operations (dilate/erode)
- Pre-process images (denoise, contrast enhancement)

### Recognition Accuracy

- **Character-level**: Inherits Phase 1 accuracy (~94%)
- **Word-level**: Depends on segmentation quality
  - Perfect segmentation → ~94% word accuracy (for 4-letter words: 0.94^4 ≈ 78%)
  - Imperfect segmentation → lower

**Improvement strategies:**
1. Improve Phase 1 classifier (better features, more data)
2. Add language model / dictionary post-processing
3. Use ensemble predictions

---

## 🔧 Troubleshooting

### Issue: "Phase 1 model not found"
```
FileNotFoundError: Phase 1 model not found: .../models/emnist_letter_classifier.pkl
```

**Solution:** Train Phase 1 first:
```powershell
cd ../FASE1_SingleCharacterRecognition
python main.py
```

### Issue: "No characters segmented"
**Possible causes:**
1. Image is blank/too dark
2. `projection_threshold` too high

**Solution:**
- Lower `projection_threshold` in `config.py` (e.g., 0.05)
- Check image with `save_segmentation_steps=True`

### Issue: Poor recognition accuracy
**Checklist:**
1. ✓ Phase 1 model trained on full dataset?
2. ✓ Character spacing clear in input images?
3. ✓ Image orientation correct?

**Debug:**
```python
# Enable debug outputs
SEGMENTATION_CONFIG["save_individual_chars"] = True
SEGMENTATION_CONFIG["save_segmentation_steps"] = True

# Check output/segmented_letters/ for extracted characters
```

---

## 🚧 Future Enhancements

### Planned Features (Not Yet Implemented)

1. **Advanced Segmentation**
   - Contour-based character detection
   - Sliding window approach
   - Handle cursive text

2. **Language Model Integration**
   - Dictionary-based spell correction
   - N-gram language models
   - Context-aware predictions

3. **Confidence Calibration**
   - Temperature scaling for better confidence scores
   - Uncertainty estimation

4. **End-to-End Training** (Optional)
   - Fine-tune Phase 1 model on word-level data
   - Joint segmentation + recognition

---

## 📝 Integration with Phase 1

### Dependency Structure

```
FASE2_WordRecognition
│
├── imports → FASE1_SingleCharacterRecognition/src/
│   └── inference_engine.py
│
├── loads → FASE1_SingleCharacterRecognition/models/
│   ├── emnist_letter_classifier.pkl
│   └── feature_scaler.pkl
│
└── uses → Phase 1's preprocessing + prediction pipeline
```

### Path Configuration

In `src/config.py`:
```python
FASE1_DIR = BASE_DIR / "FASE1_SingleCharacterRecognition"
FASE1_MODEL_PATH = FASE1_DIR / "models" / "emnist_letter_classifier.pkl"
FASE1_PREPROCESSOR_PATH = FASE1_DIR / "models" / "feature_scaler.pkl"
```

Ensure directory structure:
```
ClaudeContent/
├── FASE1_SingleCharacterRecognition/
│   └── models/
│       ├── emnist_letter_classifier.pkl  ← Must exist
│       └── feature_scaler.pkl             ← Must exist
└── FASE2_WordRecognition/
    └── src/
```

---

## 🤝 Development Guidelines

### Adding New Segmentation Methods

1. Edit `src/image_segmenter.py`
2. Add method to `_find_character_boundaries()`
3. Update config: `segmentation_method = "your_method"`

### Adding Post-Processing

1. Edit `src/word_recognizer.py`
2. Modify `_assemble_word()` method
3. Add config parameters

---

## 📧 Contact & Support

For issues:
1. Check logs in `output/word_recognition_*.log`
2. Verify Phase 1 is working: `cd ../FASE1_SingleCharacterRecognition; python predict.py --demo`
3. Review configuration in `src/config.py`

---

## 🎓 Educational Notes

### Why Projection-Based Segmentation?

**Advantages:**
- Simple to implement and understand
- Fast (O(n) where n = image width)
- Works well for separated characters

**Limitations:**
- Struggles with touching/overlapping characters
- Requires consistent spacing

**Alternatives:**
- **Contour detection**: Find connected components
- **Sliding window**: Brute-force scan with classifier
- **Deep learning**: YOLO/R-CNN for character detection

### Why Reuse Phase 1 Model?

**Transfer Learning Approach:**
- Phase 1 model learned robust character features
- No need to retrain for word-level task
- Modular design: improve Phase 1 → Phase 2 improves automatically

**Production Benefit:**
- Fast deployment (no training time)
- Reduced complexity
- Easy to maintain

---

**Happy Recognizing! 🔤**
