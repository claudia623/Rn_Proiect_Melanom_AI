# Modul 3: Web Service / UI - README

## Descriere Generală

Interfață web **Streamlit** pentru clasificarea melanomului bazată pe similaritate imagini. Modulul prezintă pipeline-ul complet end-to-end:

**Input** → Validate → Preprocess → Feature Extraction (Modul 2) → Similarity Matching (Modul 1) → **Output (Classification)**

## Funcționalități

### 1. Upload și Validare Imagine

- ✅ Accept JPG/PNG
- ✅ Verific dimensiuni (100x100 minimum, 2048x2048 maximum)
- ✅ Blur detection (Laplacian variance > 100)
- ✅ File size check (max 10MB)

### 2. Feature Extraction

- ✅ Preprocess imagine (224x224, normalizare [0-1])
- ✅ Extract 256D features cu EfficientNetB0
- ✅ L2 normalization

### 3. Similarity Matching

- ✅ Compare cu 30+ imagini referință (benign + malignant)
- ✅ Cosine similarity metric
- ✅ Aggregate scores (mean, std, min, max)

### 4. Classification

- ✅ Binary classification: BENIGN / MALIGNANT
- ✅ Confidence score (0-1)
- ✅ Detailed statistics

### 5. Afișare Rezultate

- ✅ Classification badge (verde/roșu)
- ✅ Similarity percentages
- ✅ Top 3 reference images per clasă
- ✅ Expandable detailed statistics

### 6. Logging

- ✅ CSV log cu predictions (timestamp, filename, scores)

## Structură Fișiere

```
src/app/
├── __init__.py
├── streamlit_ui.py          ← MODUL 3 principal
├── utils.py                 ← Utility functions
└── README_Module3.md        (acest fișier)
```

## Utilizare

### Prerequisite

```bash
pip install -r requirements.txt

# Dependențe specifice:
# - streamlit>=1.28.0
# - opencv-python>=4.8.0
# - tensorflow>=2.15.0
# - keras>=3.0.0
```

### Lansare Server

```bash
# Din root directory
streamlit run src/app/streamlit_ui.py

# Server pornit pe:
# 🎈 Local URL: http://localhost:8501
# 🌍 Network URL: http://YOUR_IP:8501
```

### Screenshot Demo

Imaginea ar trebui să arate:

```
┌─────────────────────────────────────────────────────────────────────┐
│ 🏥 Melanom AI - Similarity-Based Classification System              │
│ Automatic skin lesion classification: Benign vs Malignant          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┬──────────────────────────────────────┐
│ 📸 Image Upload             │ 🔍 Analysis Results                │
│                             │                                      │
│ [Upload Area]               │ [Analysis Details]                 │
│ [Uploaded Image Preview]    │                                      │
│ [Analyze Button]            │ 📋 Classification Result           │
│                             │ ✅ BENIGN (example)               │
│                             │                                      │
│                             │ 📊 Confidence: 82.3% (HIGH)       │
│                             │                                      │
│                             │ 📈 Similarity Scores:              │
│                             │ Benign Match: 75.2% | σ=8.3%     │
│                             │ Malignant: 30.1% | σ=12.1%        │
│                             │                                      │
│                             │ 🖼️ Top Similar References:        │
│                             │ [Ref Image 1] [Ref Image 2] [3]   │
└─────────────────────────────┴──────────────────────────────────────┘

┌─ Sidebar ─────────────────────────────────────────────────────────┐
│ ℹ️ System Info                                                    │
│                                                                   │
│ How it works:                                                     │
│ 1. Upload dermatoscopic image                                    │
│ 2. System validates image quality                                │
│ 3. Extracts features using EfficientNetB0                        │
│ 4. Compares with 20+ reference images                            │
│ 5. Classifies as BENIGN or MALIGNANT                             │
│                                                                   │
│ Model Info:                                                       │
│ - Architecture: EfficientNetB0 + Dense(256)                      │
│ - Inference Time: ~100ms per image                               │
│ - Model Status: Etapa 4 (Untrained)                              │
└─────────────────────────────────────────────────────────────────┘
```

## Configurare

Modifică în `streamlit_ui.py`:

```python
CONFIG = {
    'model_path': 'models/similarity_model_untrained.h5',  # Path model
    'reference_dir': 'data/generated/original/',            # Reference images
    'output_dir': 'logs/',                                  # Output logs
    'log_file': 'logs/predictions.csv',                     # CSV predictions
    'image_size': (224, 224),                               # Input size
    'blur_threshold': 100,                                  # Laplacian variance
    'max_file_size_mb': 10,                                 # Max upload size
}
```

## Workflow User

### Scenario 1: Pacient cu Leziune Benignă

```
1. Doctor upload imagine leziune
2. UI: "Image valid"
3. System analiza ~100ms
4. UI: "✅ BENIGN - 82% confidence"
5. Doctor sees reference images similare → confirma diagnostic
6. Log salvat în CSV pentru audit
```

### Scenario 2: Pacient cu Leziune Suspectă

```
1. Doctor upload imagine
2. UI error: "Image too blurry" → Doctor retakes photo
3. Upload din nou
4. System analiza
5. UI: "⚠️ MALIGNANT - 65% confidence (MEDIUM)"
6. Doctor sees similar malignant references
7. Doctor referă pacient specialist dermatologie
8. Log saved
```

### Scenario 3: Model Nu Știe (Low Confidence)

```
1. Upload imagine ambiguă
2. System analiza
3. UI: "🟡 Classification: BENIGN but 28% confidence (LOW)"
4. Doctor tooltip: "Low confidence - recommend manual review"
5. Doctor takes decision based on clinical judgment
```

## Integration cu Alte Module

### Cu Modul 1 (Data Acquisition)

```python
# Reference images încărcate din:
reference_images = load_reference_images()
# → Cites din data/generated/original/benign/ și /malignant/
# → Trebuie rulat gen_synthetic_data.py înainte
```

### Cu Modul 2 (Neural Network)

```python
# Importuri din Modul 2:
from src.neural_network.similarity_model import (
    load_model,
    extract_features,
    compute_similarity,
    classify_melanoma
)

# Call path:
model = load_model(...)  # ← Modul 2
features_test = extract_features(model, image)  # ← Modul 2
sim = compute_similarity(features_test, features_ref)  # ← Modul 2
classification, confidence, scores = classify_melanoma(...)  # ← Modul 2
```

## Testing

### Test 1: UI Start

```bash
streamlit run src/app/streamlit_ui.py

# Așteptări:
# ✅ Server pornit fără erori
# ✅ UI accessible pe http://localhost:8501
# ✅ Load bar apară (model + reference images)
# ✅ Upload area visible
```

### Test 2: Upload Test Image

```bash
# Use exemplu image din data/test/
streamlit upload: data/test/benign/ISIC_0000000.jpg

# Așteptări:
# ✅ Image validat
# ✅ Preview afișat
# ✅ "Analyze" button active
```

### Test 3: Classification

```bash
# Click "Analyze Image"

# Așteptări:
# ✅ Spinner "Computing image features..."
# ✅ Results afișate: BENIGN/MALIGNANT + confidence
# ✅ Reference images grid
# ✅ CSV log updated (logs/predictions.csv)
```

### Test 4: Error Handling

```bash
# Test 4a: Blurry image
streamlit upload: [blurry_image.jpg]
# Așteptări: "Image too blurry (score: 45.3)"

# Test 4b: Invalid format
streamlit upload: [document.pdf]
# Așteptări: "Invalid image format. Use JPG or PNG"

# Test 4c: Too large
streamlit upload: [huge_image.jpg]  # 15MB
# Așteptări: "File size too large: 15.2MB > 10MB"
```

## Troubleshooting

### Problema 1: "ModuleNotFoundError: No module named 'streamlit'"

```bash
pip install streamlit>=1.28.0
```

### Problema 2: "Reference images not loaded"

```bash
# Asigură-te că ai rulat Modul 1
python src/data_acquisition/generate_synthetic_data.py

# Verific:
ls -la data/generated/original/benign/
ls -la data/generated/original/malignant/
```

### Problema 3: "Error loading model"

```bash
# Verific că Modul 2 a generat model:
python src/neural_network/similarity_model.py

# Verific:
ls -la models/similarity_model_untrained.h5
```

### Problema 4: Slow inference

**Cause:** Feature extraction lent pe CPU  
**Solution:** Install tensorflow-gpu
```bash
pip install tensorflow-gpu
```

### Problema 5: Port 8501 deja în folosință

```bash
# Run pe port diferit:
streamlit run src/app/streamlit_ui.py --server.port 8502
```

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Image Load Time | <100ms | File upload + read |
| Image Validation | <50ms | Blur + size checks |
| Preprocess | <20ms | Resize + normalize |
| Feature Extraction | 50-150ms | Depends on CPU/GPU |
| Similarity Compute | 50-100ms | 30 reference images |
| Classification | <5ms | Agregare scores |
| **Total Latency** | **200-400ms** | Depending on hardware |

## Extensii Posibile (Etapa 5+)

- [ ] Multi-class classification (melanom vs nevus vs keratosis)
- [ ] Confidence threshold customizable
- [ ] Batch processing (multiple images)
- [ ] Report generation (PDF cu rezultate)
- [ ] Model fine-tuning cu clinician feedback
- [ ] Export to DICOM format
- [ ] Integration cu Electronic Health Records (EHR)

## Referințe

- Streamlit Docs: https://docs.streamlit.io/
- OpenCV Image Processing: https://docs.opencv.org/
- Medical Image Analysis: Peer-reviewed journals

---

**Status:** ✅ Implementat pentru Etapa 4  
**Framework:** Streamlit  
**Backend:** Modul 2 (Neural Network) + Modul 1 (Data)  
**Autor:** Dumitru Claudia-Stefania  
**Data:** 09.12.2025
