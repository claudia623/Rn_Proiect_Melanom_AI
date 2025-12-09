# 📊 Descrierea Setului de Date - Melanom AI

## 1. Sursa Datelor

### 1.1 Origine
- **Dataset:** ISIC (International Skin Imaging Collaboration) Archive
- **URL:** https://www.isic-archive.com/
- **Alternative:** HAM10000 Dataset, Kaggle Skin Cancer MNIST

### 1.2 Modul de Achiziție
- ☑ Fișier extern (dataset public)
- Imagini dermatoscopice de înaltă calitate
- Etichetate de specialiști în dermatologie

### 1.3 Perioada Colectării
- Dataset-uri colectate între 2016-2023
- Validare clinică și histopatologică

---

## 2. Caracteristicile Dataset-ului

### 2.1 Statistici Generale
- **Număr total de imagini:** ~10,000+ (în funcție de subset)
- **Clase:** 2 (Benign, Malign/Melanom)
- **Format:** JPEG/PNG
- **Rezoluție originală:** Variabilă (600x450 până la 1024x1024)

### 2.2 Distribuția Claselor
| Clasă | Număr Imagini | Procent |
|-------|---------------|---------|
| Benign | ~7,000 | ~70% |
| Malign (Melanom) | ~3,000 | ~30% |

**⚠️ Notă:** Dataset dezechilibrat - necesită tehnici de balansare

### 2.3 Tipuri de Leziuni (Benigne)
- Nevus melanocitic
- Keratoză seboreică  
- Dermatofibrom
- Leziuni vasculare

### 2.4 Tipuri de Leziuni (Maligne)
- Melanom
- Carcinom bazocelular
- Carcinom scuamos

---

## 3. Descrierea Caracteristicilor Imaginilor

| Caracteristică | Tip | Descriere | Valori |
|----------------|-----|-----------|--------|
| Pixeli RGB | Numeric | Valori intensitate culoare | 0-255 |
| Dimensiune | Numeric | Lățime x Înălțime | Redimensionat la 224x224 |
| Contrast | Numeric | Diferența luminozitate | Variabil |
| Textură | Numeric | Pattern-uri suprafață | Extrase cu CNN |
| Formă | Categorial | Regulată/Neregulată | Extrasă automat |
| Culoare | Categorial | Uniformă/Variegată | Extrasă automat |

---

## 4. Structura Directoarelor

```
data/
├── raw/                    # Imagini originale nedeschise
│   ├── benign/
│   └── malignant/
├── processed/              # Imagini preprocesate
│   ├── benign/
│   └── malignant/
├── train/                  # 70% din date
│   ├── benign/
│   └── malignant/
├── validation/             # 15% din date
│   ├── benign/
│   └── malignant/
└── test/                   # 15% din date
    ├── benign/
    └── malignant/
```

---

## 5. Preprocesare Aplicată

### 5.1 Curățare Date
- Eliminarea imaginilor corupte
- Eliminarea duplicatelor
- Verificarea etichetelor

### 5.2 Transformări
- **Redimensionare:** 224x224 pixeli
- **Normalizare:** Pixeli scalați la [0, 1]
- **Standardizare:** Conform media/std ImageNet

### 5.3 Augmentare (doar pe train)
- Rotații: ±20°
- Flip orizontal/vertical
- Zoom: ±20%
- Shift: ±20%
- Ajustări luminozitate/contrast

---

## 6. Împărțirea Datelor

| Set | Procent | Număr Imagini | Scop |
|-----|---------|---------------|------|
| Train | 70% | ~7,000 | Antrenare model |
| Validation | 15% | ~1,500 | Validare și early stopping |
| Test | 15% | ~1,500 | Evaluare finală |

**Principii respectate:**
- ✅ Stratificare (păstrarea proporției claselor)
- ✅ Fără data leakage
- ✅ Reproducibilitate (seed fix)

---

## 7. Provocări și Soluții

| Problemă | Soluție |
|----------|---------|
| Class Imbalance | Weighted loss, Oversampling, SMOTE |
| Variabilitate luminozitate | Normalizare, Color jittering |
| Artefacte (păr, markere) | Preprocesare, Inpainting |
| Overfitting | Dropout, Data augmentation, Regularizare |

---

## 8. Surse de Descărcare

1. **ISIC Challenge:** https://challenge.isic-archive.com/
2. **Kaggle HAM10000:** https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
3. **Kaggle Melanoma:** https://www.kaggle.com/c/siim-isic-melanoma-classification

---

**Actualizat:** Noiembrie 2024
