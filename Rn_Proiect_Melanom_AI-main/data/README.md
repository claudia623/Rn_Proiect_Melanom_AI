## 1. Sursa Datelor

### 1.1 Origine
- **Dataset Principal:** ISIC (International Skin Imaging Collaboration) Archive
- **URL:** https://www.isic-archive.com/

### 1.2 Modul de Achiziție
-  Descărcat din ISIC Archive (dataset public)
- Imagini dermatoscopice de înaltă calitate
- Etichetate de specialiști în dermatologie
- Validare clinică și histopatologică

### 1.3 Perioada Colectării
- Dataset-uri colectate între 2016-2023
- Acces public pentru cercetare și educație

---

## 2. Caracteristicile Dataset-ului

### 2.1 Statistici Generale - STATUS ACTUAL
- **Imagini Procesate:** 202 imagini (101 benign + 101 malign)
- **Clase:** 2 (Benign și Malignant)
- **Format Stocare:** JPEG
- **Redimensionare Aplicată:** 224×224 pixeli
- **Rezoluție Originală:** 600×450 - 1024×1024 pixeli (variabilă)

### 2.2 Distribuția Claselor 

| Set | Benign | Malignant |  Total |  Procent |
|:------:|:---------:|:------------:|:-------:|:-----------:|
| **Processed** | 101 | 101 | **202** | **100%** |
| **Train** | 71 | 71 | **142** | **70%** |
| **Validation** | 16 | 16 | **32** | **15%** |
| **Test** | 16 | 16 | **32** | **15%** |

**Notă:** Dataset ECHILIBRAT (50/50) - nu necesită tehnici speciale de balansare.

### 2.3 Tipuri de Leziuni Incluse

**Clase Benigne (101 imagini):**
- Nevus melanocitic
- Keratoză seboreică  
- Dermatofibrom
- Leziuni vasculare
- Alte leziuni non-maligne

**Clase Maligne (101 imagini):**
- Melanom
- Carcinom bazocelular (BCC)
- Carcinom scuamos (SCC)

---

## 3. Structura Directoarelor 

```
data/
├── raw/                    # [GOL - Nu mai sunt imagini originale]
│   ├── benign/
│   └── malignant/
│
├── processed/              # 202 imagini PRELUCRATE
│   ├── benign/             # 101 imagini
│   └── malignant/          # 101 imagini
│
├── train/                  # 142 imagini (70%)
│   ├── benign/             # 71 imagini
│   └── malignant/          # 71 imagini
│
├── validation/             # 32 imagini (15%)
│   ├── benign/             # 16 imagini
│   └── malignant/          # 16 imagini
│
└── test/                   # 32 imagini (15%)
    ├── benign/             # 16 imagini
    └── malignant/          # 16 imagini
```

---

## 4. Preprocesare Aplicată 

### 4.1 Pași de Curățare și Validare
- Eliminarea imaginilor corupte
- Verificarea și corectarea etichetelor
- Validare format și integritate fișiere

### 4.2 Transformări Aplicate
- **Redimensionare:** 224×224 pixeli (standard pentru EfficientNetB0)
- **Normalizare:** Pixeli scalați la [0, 1]
- **Standardizare:** Media/std conform ImageNet pre-training

### 4.3 Operații Morfologice (Preprocessing Avansat)
- **Eliminare păr:** Operații morfologice (closing + opening)
- **Contrast Enhancement:** CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Ascuțire imagine:** Unsharp Masking
- **Detecție blur:** Validare calitate imagine

### 4.4 Augmentare Date (DOAR pe TRAIN)
- Rotații: ±20°
- Flip orizontal/vertical
- Zoom: ±20%
- Shift: ±20%
- Ajustări luminozitate/contrast
- Color jittering

---

## 5. Împărțirea Datelor

| Set |  Procent |  Benign |  Malignant |  Total |  Scop |
|:------:|:-----------:|:---------:|:------------:|:-------:|:---|
| **Train** | 70% | 71 | 71 | **142** | Antrenare model |
| **Validation** | 15% | 16 | 16 | **32** | Validare & early stopping |
| **Test** | 15% | 16 | 16 | **32** | Evaluare finală |

**Principii Respectate:**
-  Stratificare perfectă (50/50 pe fiecare set)
-  Fără data leakage
-  Reproducibilitate garantată
-  Echilibru perfect între clase

---

## 6. Provocări și Soluții Aplicate

|  Problemă |  Soluție Aplicată |
|:-----------|:---|
|  **Class Imbalance** | Dataset echilibrat 50/50 |
|  **Variabilitate luminozitate** | CLAHE + Color jittering |
|  **Artefacte (păr, markere)** | Morfologice operations + preprocessing |
|  **Overfitting** | Dropout, Augmentare, Regularizare L2 |
|  **Blur și calitate scazută** | Detecție Blur + filtrare |

---

## 7. Caracteristicile Imaginilor din Dataset

|  Caracteristică |  Tip |  Descriere |  Valori |
|:-:|:-:|:---|:---|
| **Pixeli RGB** | Numeric | Valori intensitate culoare | `0-255` → `[0, 1]` |
| **Dimensiune** | Numeric | Lățime × Înălțime | **224×224** px |
| **Contrast** | Numeric | Diferența luminozitate | Variabil → **CLAHE** |
| **Textură** | Numeric | Pattern-uri suprafață | **CNN (EfficientNetB0)** |
| **Formă** | Categorial | Regulată/Neregulată | Extrasă automat |
| **Culoare** | Categorial | Uniformă/Variegată | **Color jittering** |

---

## 8. Tehnologie și Instrumente Folosite

### 8.1 Framework-uri
- **TensorFlow/Keras** - Antrenare model neuronal
- **OpenCV** - Procesare imagini
- **NumPy/Pandas** - Manipulare date
- **Scikit-learn** - Metrici și evaluare

### 8.2 Model Neural
- **Arhitectură:** EfficientNetB0 (Transfer Learning)
- **Input:** 224×224×3 (imagini RGB)
- **Output:** 2 clase (Benign/Malignant)
- **Optimizator:** Adam
- **Loss:** Weighted Binary Crossentropy (pentru echilibru)

### 8.3 Scripturi de Procesare
- `preprocess_dataset.py` - Preprocesare imagini brute
- `split_processed_data.py` - Împărțire train/val/test
- `preprocess_test_data.py` - Preprocesare test
- `train.py` - Antrenare model (2 faze)
- `evaluate.py` - Evaluare și metrici

---

## 9. Surse și Referințe

### 9.1 Dataset Public
1. **ISIC Archive:** https://www.isic-archive.com/
2. **ISIC Challenge:** https://challenge.isic-archive.com/
3. **Kaggle HAM10000:** https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
4. **Kaggle Melanoma:** https://www.kaggle.com/c/siim-isic-melanoma-classification

### 9.2 Documente Referință în Proiect
- `../README.md` - Prezentarea proiectului
- `../src/preprocessing/preprocess_dataset.py` - Cod preprocesare
- `../config/config.yaml` - Configurații model și training

---

## 10. Statistici și Performanță Așteptată

### 10.1 Date de Antrenare Disponibile
- **Total imagini:** 206 (202 procesate + alte surse)
- **Dataset echilibrat:**  da (50% benign, 50% malignant)
- **Augmentare aplicată:**  da (pe setul de train)

### 10.2 Performanță Așteptată
Cu EfficientNetB0 și dataset-ul de 206 imagini:
- **Validation Accuracy:** ~75-85%
- **Test Accuracy:** ~70-80%
- **AUC Score:** ~0.85-0.92
- **Precision/Recall:** ~0.75-0.85 (per clasă)

### 10.3 Factori care Influențează Performanța
-  Dataset echilibrat (avantaj)
-  Dimensiune limitată (202 imagini) - necesită transfer learning
-  Augmentare aplicată - îmbunătățește generalizarea
-  Preprocessing avansat - îmbunătățește feature extraction

---

## 11. Note Importante

###  Limitări Actuale
- Dataset mic (206 imagini) - model va beneficia de transfer learning
- Single source (ISIC) - rezultatele pot varia cu date din alte surse
- Classes not fully diverse - testing cu date noi este esențial

###  Avantaje
- Dataset perfect echilibrat (50/50)
- Imagini de calitate înaltă
- Preprocessing complet (eliminare artefacte, etc.)

###  Îmbunătățiri Posibile
1. Augmentare dataset cu mai multe imagini din ISIC
2. Mixture de date din HAM10000 și alte surse
3. Cross-validation pentru evaluare mai robustă
4. Hyperparameter tuning pe dataset-ul complet

---

**Ultima Actualizare:** 11 februarie 2026  
