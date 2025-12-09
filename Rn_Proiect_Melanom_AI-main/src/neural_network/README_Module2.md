# Modul 2: Neural Network - README

## Descriere Generală

Acest modul implementează o **rețea neuronală pentru feature extraction și similarity matching** în clasificarea melanomului. Modelul folosește **transfer learning** cu EfficientNetB0 pretreined pe ImageNet.

**Status Etapa 4:** Model definit, compilat, NU antrenat (weights random init din ImageNet).

## Arhitectura Modelului

### Componente

```
INPUT: RGB Image (224x224x3)
    ↓
PREPROCESSING: EfficientNet standardization (mean subtraction)
    ↓
EfficientNetB0 (Pretrained ImageNet, 154 layers)
    - Convolutional blocks cu depthwise separable convolutions
    - Efficient mobile architecture
    - Frozen layers (NU se antrenează în Etapa 4)
    ↓
GLOBAL AVERAGE POOLING: (7x7x1280) → (1280D vector)
    ↓
DENSE LAYER: (1280) → Dense(256, ReLU)
    - Feature extraction layer
    - ReLU activation pentru non-linearity
    ↓
DROPOUT: (256) → Dropout(0.5)
    - Regularizare pentru evitarea overfitting
    ↓
L2 NORMALIZATION: Normalize pe dimensiunea feature
    - Output vectori cu norm = 1.0
    ↓
OUTPUT: Feature Vector (256D, normalized)
    - Ready pentru similarity computation cu cosine distance
```

### Dimensiuni Layer-uri

| Layer | Output Shape | Parameters |
|-------|--------------|-----------|
| image_input | (None, 224, 224, 3) | 0 |
| preprocessing | (None, 224, 224, 3) | 0 |
| efficientnetb0 | (None, 7, 7, 1280) | 4,049,564 |
| global_avg_pool | (None, 1280) | 0 |
| feature_extraction | (None, 256) | 327,936 |
| dropout | (None, 256) | 0 |
| feature_output | (None, 256) | 0 |
| **TOTAL** | | **4,377,500** |

## Fișiere Principale

```
src/neural_network/
├── similarity_model.py        ← MODUL 2 principal (Etapa 4)
├── model.py                   (din Etapa 3, transfer learning generic)
├── train.py                   (din Etapa 3, preparare pentru antrenare)
├── evaluate.py                (din Etapa 3, metrici evaluare)
└── README_Module2.md          (acest fișier)
```

## Funcționalități

### 1. Feature Extraction

```python
from src.neural_network.similarity_model import create_similarity_model, extract_features

# Creează model
model = create_similarity_model(input_shape=(224, 224, 3), feature_dim=256)

# Extrage features din imagine
features = extract_features(model, image)  # Output: (256,)
```

### 2. Similarity Computation

```python
from src.neural_network.similarity_model import compute_similarity

# Comparare cu referință
sim = compute_similarity(features_test, features_reference)
# Output: similarity score [0, 1] (1 = identic, 0 = diferit)
```

### 3. Classification

```python
from src.neural_network.similarity_model import classify_melanoma

similarities_benign = [0.75, 0.70, 0.68]      # Cu imagini benigne
similarities_malignant = [0.30, 0.35, 0.25]   # Cu imagini maligne

classification, confidence, scores = classify_melanoma(
    similarities_benign,
    similarities_malignant
)
# Output:
# classification = "BENIGN"
# confidence = 0.403  (diferența dintre mean scores)
# scores = dict cu statistici detaliate
```

## Utilizare

### Prerequisite

```bash
pip install -r requirements.txt

# Dependențe specifice:
# - tensorflow>=2.15.0
# - keras>=3.0.0
# - numpy>=1.26.0
```

### Test Model Creation

```bash
python src/neural_network/similarity_model.py

# Output așteptat:
# - Model created successfully!
# - Model saved to models/similarity_model_untrained.h5
# - Feature extraction test passed
# - Classification test passed
```

### Încărcare Model în Cod

```python
from src.neural_network.similarity_model import (
    create_similarity_model,
    load_model,
    extract_features,
    compute_similarity,
    classify_melanoma
)

# Opțiune 1: Creează model nou
model = create_similarity_model()

# Opțiune 2: Încarcă din fișier salvat
model = load_model('models/similarity_model_untrained.h5')

# Folosire pentru inference
features = extract_features(model, preprocessed_image)
```

## Model Properties

### Características Importante

1. **Transfer Learning:**
   - ImageNet pretraining ✓
   - Frozen base model ✓
   - Fine-tuning ready pentru Etapa 5

2. **Feature Dimensionality:**
   - 256D features (balanță între expressiveness și efficiency)
   - L2-normalized (ready pentru cosine similarity)

3. **Efficiency:**
   - ~4.4M parameters
   - Inference time: ~50-100ms per image (CPU)
   - Forward pass optimized cu EfficientNet mobile architecture

4. **Inference-Only (Etapa 4):**
   - NU se antrenează
   - NU se calculează loss
   - NU se update-aza weights

## Integration cu Alte Module

### Cu Modul 1 (Data Acquisition)

```python
from src.data_acquisition.generate_synthetic_data import load_images_from_directory
from src.neural_network.similarity_model import (
    extract_features, compute_similarity, classify_melanoma
)

# Load imagini referință (benign/malignant)
reference_images = load_images_from_directory('data/generated/original/')

# Extract features
features_benign = [extract_features(model, img) for img in benign_refs]
features_malignant = [extract_features(model, img) for img in malignant_refs]
```

### Cu Modul 3 (Web UI)

```python
# Web UI va folosi similarity_model pentru backend inference
from src.neural_network.similarity_model import (
    load_model, extract_features, compute_similarity, classify_melanoma
)

model = load_model('models/similarity_model_untrained.h5')

# User upload imagine → preprocess → extract features → compare
features_test = extract_features(model, user_image)
similarity_benign = compute_similarity(features_test, features_benign_refs)
classification, confidence, scores = classify_melanoma(...)
```

## Testing

### Unit Test 1: Model Creation

```python
from src.neural_network.similarity_model import create_similarity_model

model = create_similarity_model()
assert model is not None
assert model.input_shape == (None, 224, 224, 3)
assert model.output_shape == (None, 256)
print("✅ Model creation test passed!")
```

### Unit Test 2: Feature Extraction

```python
import numpy as np
from src.neural_network.similarity_model import extract_features

dummy_image = np.random.randn(224, 224, 3).astype(np.float32)
dummy_image = np.clip(dummy_image / 255.0, 0, 1)

features = extract_features(model, dummy_image)
assert features.shape == (256,)
assert abs(np.linalg.norm(features) - 1.0) < 0.01  # L2 normalized
print("✅ Feature extraction test passed!")
```

### Unit Test 3: Similarity Computation

```python
from src.neural_network.similarity_model import compute_similarity

features1 = np.random.randn(256)
features1 = features1 / np.linalg.norm(features1)  # Normalize

# Same features → similarity ~1.0
sim_same = compute_similarity(features1, features1)
assert sim_same > 0.99

# Different features → similarity < 1.0
features2 = np.random.randn(256)
features2 = features2 / np.linalg.norm(features2)
sim_diff = compute_similarity(features1, features2)
assert sim_diff < 0.99

print("✅ Similarity computation test passed!")
```

### Unit Test 4: Classification

```python
from src.neural_network.similarity_model import classify_melanoma

similarities_benign = [0.85, 0.80, 0.78]  # High similarity with benign
similarities_malignant = [0.25, 0.20, 0.22]  # Low similarity with malignant

classification, confidence, scores = classify_melanoma(
    similarities_benign,
    similarities_malignant
)

assert classification == "BENIGN"
assert confidence > 0.5
print("✅ Classification test passed!")
```

## Model Comparison: Transfer Learning Approaches

### EfficientNetB0 (Ales)

| Aspect | EfficientNetB0 | ResNet50 | VGG16 |
|--------|---|---|---|
| Parameters | 4.0M | 25.6M | 138M |
| Inference Speed | Fast | Medium | Slow |
| Accuracy ImageNet | 77.1% | 76.0% | 71.3% |
| Mobile Ready | ✅ | ⚠️ | ❌ |
| Balanced | ✅ | ⚠️ | ❌ |

**Rație alegere:** EfficientNet oferă cel mai bun balance între accuracy și speed, ideal pentru aplicații medicale realtime.

## Probleme Cunoscute și Soluții

### Problema 1: Memory Error la Load Model

```
ResourceExhaustedError: OOM when allocating tensor
```

**Soluție:**
```python
# Reduce input batch size
model.predict(images_batch[0:1])  # Process 1 image la a time
```

### Problema 2: Slow Inference on CPU

**Soluție:**
```bash
# Install tensorflow-gpu (dacă GPU disponibil)
pip install tensorflow-gpu

# Alternativ, optimizează cu:
model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
```

### Problema 3: Similarity All Low (<0.5)

**Diagnostic:**
- Features din bază de date preprocesat diferit?
- Reference images de rezoluție diferită?

**Soluție:**
```python
# Verify normalization
assert np.allclose(np.linalg.norm(features), 1.0)
```

## Referințe

- EfficientNet paper: https://arxiv.org/abs/1905.11946
- Keras EfficientNetB0: https://keras.io/api/applications/efficientnet/
- Transfer Learning Guide: https://cs231n.github.io/transfer-learning/

---

**Status:** ✅ Implementat pentru Etapa 4 (Feature extraction only)  
**Arhitectură:** EfficientNetB0 + Dense(256) + L2 Normalization  
**Antrenare:** NU (weights pretrained ImageNet)  
**Autor:** Dumitru Claudia-Stefania  
**Data:** 09.12.2025
