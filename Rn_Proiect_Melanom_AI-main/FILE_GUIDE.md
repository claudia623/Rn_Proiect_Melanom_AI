# 📖 GHID DETALIAT - Ce Face Fiecare Fișier

## 🗂️ FIȘIERE ROOT LEVEL (Directorul Principal)

### 📄 **README.md** (435 linii)
**Ce face?** Documentația principală a proiectului
- Descriere generală a proiectului (clasificare melanom cu RN)
- Tabel de conținut și navigare
- Overview tehnologic (EfficientNetB0, transfer learning)
- Performanța actuală (AUC 0.8889, Accuracy 80%)
- Instrucțiuni instalare și setup
- Ghid de utilizare complet
- Descriere arhitectură model
- Rezultate și metrici
- Link-uri la documentație suplimentară

**Când să-l citești?** Când vrei să înțelegi proiectul în ansamblu

---

### 📄 **START_HERE.md** (139 linii)
**Ce face?** Ghid rapid pentru a porni repede
- Instrucțiuni pentru primii 5 minute
- Comenzi simple pentru a rula pipeline-ul
- Explicații scurte pentru fiecare pas
- Link-uri la resurse mai detaliate
- Ghid rapid pentru troubleshooting

**Când să-l citești?** Când vrei să pornești proiectul rapid fără detalii

---

### 📄 **SETUP_INSTRUCTIONS.md** (302 linii)
**Ce face?** Ghid complet de instalare și configurare
- Prerequisite-uri (Python, OS, hardware)
- Crearea virtual environment-ului
- Instalarea dependențelor
- Pregătirea datelor (unde să pui imaginile)
- Rularea fiecărui pas al pipeline-ului
- Troubleshooting pentru probleme comune
- Configurare pentru GPU/CPU
- Descrierea fiecărei faze

**Când să-l citești?** Când trebuie să setezi mediul și datele

---

### 📄 **STRUCTURE.txt** (272 linii)
**Ce face?** Documentație detaliată a structurii proiectului
- Arborele complet cu descrieri
- Explicații pentru fiecare director
- Flux de date (data flow)
- Statistici proiect (755 imagini, 25 scripturi)
- Referință la fișiere cheie
- Informații despre modele și log-uri
- Configurație și constante

**Când să-l citești?** Când vrei să înțelegi organizarea foldere-lor

---

### 📄 **INDEX.md** (234 linii)
**Ce face?** Index principal cu navigare
- Link-uri rapide către toate documentele
- Ghid după rol (manager, developer, data scientist)
- Ghid după topic (setup, data, model, evaluation)
- Tabel cu răspunsuri la întrebări comune
- Ștări curente și statistici
- Learning path pentru diferiți utilizatori

**Când să-l citești?** Când cauți ceva specific și vrei link rapid

---

### 📄 **PROJECT_CHECKLIST.md** (273 linii)
**Ce face?** Lista de verificare și tracking al progresului
- Task-uri completate (✅) și pending (⏳)
- 13 faze majore ale proiectului
- Procent de completare pentru fiecare fază
- Milestone-uri și checkpoints
- Metrici de calitate (documentație, cod, model)
- Recomandări pentru next steps
- Probleme și soluții

**Când să-l citești?** Când vrei să vezi ce mai rămâne de făcut

---

### 📄 **PROJECT_STATUS_REPORT.md** (410 linii)
**Ce face?** Raport detaliat de status al proiectului
- Executive summary (rezumat executiv)
- Statistici proiect (755 imagini, 25 scripturi, 20+ docs)
- Arhitectură (EfficientNetB0, 4.84M parametri)
- Metrici performance (0.8889 AUC, 80% accuracy)
- Descriere detaliat a fiecărei faze
- Problemele întâlnite și soluțiile
- Experiență învățată (learnings)
- Recomandări pentru viitor

**Când să-l citești?** Când vrei raport complet de status

---

### 📄 **DOCUMENTATION_SUMMARY.md** (365 linii)
**Ce face?** Rezumat al sesiunii de documentare
- Fișierele create în această sesiune
- Statistici (2300+ linii documentație)
- Ghid de accesare al documentației
- Cale de învățare (learning path)
- Metrici calitate documentație
- Concluzie și next steps

**Când să-l citești?** Când vrei să cunoști documentația care a fost adăugată

---

### 📄 **requirements.txt**
**Ce face?** Lista dependințelor Python
```
tensorflow/keras    - Framework pentru neural networks
opencv-python      - Procesare imagini
numpy              - Operații numerice
pandas             - Manipulare date
scikit-learn       - Machine learning utilities
matplotlib/seaborn - Vizualizări
```

**Când să-l folosești?** Când rulezi `pip install -r requirements.txt`

---

### 📄 **.gitignore**
**Ce face?** Spune Git ce fișiere să ignore
- __pycache__/ - Fișiere compilate Python
- .venv/ - Virtual environment
- *.pyc - Fișiere compilate
- data/raw/*.jpg - Imagini brute (prea mari)
- models/*.keras - Modele mari (folosește Git LFS)

**Când să-l folosești?** Automat - nu trebuie să faci nimic

---

### 📄 **Makefile** (100+ linii)
**Ce face?** Automation targets pentru build și development
```bash
make install       # Instalează dependințe
make setup         # Cria directoare
make preprocess    # Preproceseaza imagini
make split         # Împarte datele
make train         # Antrenează model
make evaluate      # Evaluează pe test set
make full-pipeline # Rulează totul
```

**Când să-l folosești?** Când vrei să automatizezi build/test

---

### 📄 **config.yaml** (în config/)
**Ce face?** Configurare model și antrenare
```yaml
model:
  name: EfficientNetB0
  input_size: [224, 224, 3]
training:
  batch_size: 16
  epochs_phase1: 30
  epochs_phase2: 50
  learning_rate_phase1: 0.001
  learning_rate_phase2: 0.0001
```

**Când să-l folosești?** Când vrei să schimbi parametri antrenare

---

---

## 🐍 FIȘIERE PYTHON - SRC/ DIRECTORY

### **src/preprocessing/** - Preproceseaza imagini

#### **image_processing.py** (Core Functions)
**Ce face?**
- `preprocess_image()` - Principal pipeline:
  - Resize la 224×224 (ImageNet standard)
  - Îndepărtare păr (morphological operations)
  - Îmbunătățire contrast (CLAHE)
  - Ascuțire imagine (Unsharp Masking)
  - Validare calitate (Laplacian variance > 100)
- `apply_clahe()` - Contrast enhancement
- `sharpen_image()` - Edge enhancement
- `check_blur()` - Detect imagini nesigure
- `remove_hair()` - Îndepărtare păr din imagini

**Folosit de:** preprocess_dataset.py, preprocess_test_data.py

---

#### **preprocess_dataset.py** (Batch Processing)
**Ce face?**
1. Citește imagini din `data/raw/benign` și `data/raw/malignant`
2. Aplică pipeline de preprocesecare pe fiecare imagine
3. Salvează imagini prelucrate în `data/processed/benign` și `data/processed/malignant`
4. Validează calitate (reject dacă blur score < 100)
5. Log-uri progress

**Intrare:** 400 imagini brute din data/raw/  
**Ieșire:** 400 imagini prelucrate în data/processed/  
**Status:** ✅ COMPLETAT

---

#### **split_processed_data.py** (Data Splitting)
**Ce face?**
1. Citește imagini din `data/processed/`
2. Împarte în train/validation/test (70/15/15 ratio)
3. Mântine balansare clase în fiecare set
4. Muta imagini în directoarele corespunzatoare:
   - `data/train/` (140 imagini)
   - `data/validation/` (30 imagini)
   - `data/test/` (30 imagini)
5. Log-uri cu statistici

**Status:** ✅ COMPLETAT (140/30/30 split)

---

#### **preprocess_test_data.py** (Test Data)
**Ce face?**
1. Preproceseaza imagini din `data/test/`
2. Asigură consistență cu procesarea de antrenare
3. Validează calitate teste
4. Salvează imagini prelucrate

**Status:** ✅ COMPLETAT (60 imagini test prelucrate)

---

#### **split_data.py** (Legacy - Alternative)
**Ce face?** Versiune alternativă pentru split date (mai veche)

**Status:** ⏳ DEPRECATED (folosiți split_processed_data.py)

---

### **src/data_acquisition/** - Colectare/Organizare Date

#### **organize_images.py**
**Ce face?**
- Organizează imagini în subdirectoare după clasă
- Redenumeste imagini consistent
- Valideaza formatul imaginilor
- Log-uri cu statistici

**Folosit pentru:** Preprocesare date brute

---

### **src/neural_network/** - Model Training & Evaluation

#### **train.py** (Main Training Script)
**Ce face?**
1. **Phase 1 - Train Custom Head (Frozen Base)**
   - Încarcă EfficientNetB0 pre-antrenat
   - Îngheață backbone (nu modifi greutăți)
   - Antrenează doar custom head (256 → 128 → 2 units)
   - Epoca 1-22: AUC cresce la 0.8733
   
2. **Phase 2 - Fine-Tune**
   - Dezgheață ultimii 30 layeri din EfficientNetB0
   - Reduc learning rate (0.0001)
   - Antrenează 25 epoce
   - Final AUC: **0.8889** ⭐ (BEST)

3. **Callbacks Implementate:**
   - Early stopping (patience=10)
   - Learning rate reduction on plateau
   - Model checkpointing (saves best)
   - Real-time metrics

**Intrare:** Imagini din `data/train/` și `data/validation/`  
**Ieșire:** Model salvat în `models/melanom_efficientnetb0_best.keras`  
**Logs:** `logs/melanom_efficientnetb0_YYYYMMDD_HHMMSS/`  

**Status:** ✅ COMPLETAT (Training finalizat)

---

#### **evaluate.py** (Model Evaluation)
**Ce face?**
1. Încarcă modelul antrenat
2. Încarcă test set din `data/test/`
3. Generează predicții
4. Calculează metrici:
   - Confusion matrix
   - ROC-AUC
   - Precision, Recall, F1-Score
   - Cohen's Kappa
5. Salvează:
   - Confusion matrix (vizualizare)
   - ROC curve
   - Predictions CSV
   - Metrics CSV

**Rulare:** `python src/neural_network/evaluate.py --use-best`

**Ieșire:** Fișiere în `results/`

---

#### **model.py** (Model Architecture)
**Ce face?**
- Definiți arhitectura modelului
- Custom head: Dense 256 → Dense 128 → Dense 2 (softmax)
- Dropout layers (0.3, 0.2)
- Input: 224×224×3

---

#### **callbacks.py** (Training Callbacks)
**Ce face?**
- EarlyStopping - Oprește dacă nu se îmbunătățește
- ReduceLROnPlateau - Scade learning rate
- ModelCheckpoint - Salvează best model

---

#### **similarity_model.py** (Feature Extraction)
**Ce face?** Model pentru similarity-based inference (optional)

---

### **src/utils/** - Funcții Helper

#### **constants.py** (250+ linii)
**Ce face?**
- Definește paths (PROJECT_ROOT, DATA_DIR, MODELS_DIR, etc.)
- Constante preprocesecare (TARGET_SIZE=(224,224), CLAHE_CLIP_LIMIT=2.0)
- Constante model (BATCH_SIZE=16, EPOCHS_PHASE1=30)
- Split ratios (TRAIN=70%, VAL=15%, TEST=15%)
- Class labels (BENIGN=0, MALIGNANT=1)
- Best model metrics (AUC=0.8889)
- Helper functions:
  - `ensure_directories()` - Crează directoare
  - `get_class_path()` - Path la clasă
  - `get_model_path()` - Path la model
  - `get_config_path()` - Path la config

**Folosit de:** Toate scripturile

---

#### **helpers.py** (189 linii)
**Ce face?**
- `set_seed()` - Reproducibilitate
- `get_timestamp()` - Timestamp curent
- `get_available_gpus()` - Check GPU
- Alte funcții utility generale

---

#### **validators.py**
**Ce face?**
- Validare imagini (format, size)
- Validare dataset
- Validare configurare

---

### **src/app/** - Web/API Interface

#### **main.py** (Future)
**Ce face?** Va fi pentru serving model

---

---

## 📚 DOCUMENTAȚIE - DOCS/ DIRECTORY

### **docs/INDEX.md** (200 linii)
**Ce face?** Index al tuturor documentelor cu link-uri

---

### **docs/datasets/DATASET_INFO.md**
**Ce face?** Descriere dataset:
- Sursa datelor
- Statistici (755 imagini)
- Distribuție clase
- Calitate imagini

---

### **docs/error_analysis/**
**Ce face?** (Gol în prezent)
Va conține: Analiza erorilor, imagini misclassificate, pattern-uri

---

### **docs/README_Etapa4_Arhitectura_SIA.md**
**Ce face?** Descriere detaliată arhitectură:
- EfficientNetB0 backbone (4.04M params)
- Custom head layers
- Loss function (categorical crossentropy)
- Optimizer (Adam)

---

### **docs/README_Etapa5_Antrenare_RN.md**
**Ce face?** Ghid antrenare:
- Two-phase training strategy
- Hyperparameter-i
- Learning rate scheduling
- Early stopping logic

---

### **docs/README – Etapa 3 -...md**
**Ce face?** Procesare date:
- Preprocesecare pipeline
- Split strategy
- Data augmentation

---

### **docs/TESTING_GUIDE_ETAPA4.md**
**Ce face?** Ghid testing:
- Cum să evaluezi model
- Cum să interpretezi metrici
- Troubleshooting

---

### **docs/COMPLETION_REPORT.md**
**Ce face?** Raport finalizare fază

---

### **docs/generate_etapa5_visualizations.py**
**Ce face?**
- Genereaza grafice de training
- Confusion matrix plots
- ROC curves
- Performance comparisons

---

---

## 📊 CONFIG/ DIRECTORY

### **config/config.yaml**
**Ce face?** Configurare model
- Model architecture
- Training parameters
- Data paths
- Preprocessing settings

---

### **config/metadata.csv**
**Ce face?** Metadate dataset
- Imagini și clase
- Source information
- Image statistics

---

---

## 💾 DATA/ DIRECTORY

### **data/raw/**
**Ce face?** Stochează imagini brute
- benign/ (imagini benigne)
- malignant/ (imagini maligne)

**Conținut:** 400 imagini originale

---

### **data/processed/**
**Ce face?** Stochează imagini prelucrate
- benign/ (400 imagini prelucrate)
- malignant/ (400 imagini prelucrate)

**Size:** 224×224 pixels, normalized

---

### **data/train/**
**Ce face?** Training set (70%)
- benign/ (70 imagini)
- malignant/ (70 imagini)

**Total:** 140 imagini

---

### **data/validation/**
**Ce face?** Validation set (15%)
- benign/ (30 imagini)
- malignant/ (30 imagini)

**Total:** 30 imagini

---

### **data/test/**
**Ce face?** Test set (15%)
- benign/ (imagini test)
- malignant/ (imagini test)

**Total:** ~30 imagini

---

### **data/generated/**
**Ce face?** Augmented/generated data (Future use)

---

---

## 🤖 MODELS/ DIRECTORY

### **melanom_efficientnetb0_best.keras** (24 MB)
**Ce face?**
- Model antrenat complet
- AUC: 0.8889
- Accuracy: 80%
- Status: ✅ PRODUS PENTRU INFERENCE

---

### **melanom_efficientnetb0_last.keras**
**Ce face?** Ultima checkpoint din antrenare
- Pentru resume antrenare
- Alternative în caz de necesitate

---

---

## 📝 LOGS/ DIRECTORY

### **melanom_efficientnetb0_YYYYMMDD_HHMMSS/**
**Ce face?**
- `training_log.txt` - Log complet
- `metrics.csv` - Metrici pe epocă
- `best_model_info.txt` - Info model best

---

### **melanom_efficientnetb0_*_history.csv**
**Ce face?** History pe epocă:
- loss, accuracy, auc
- val_loss, val_accuracy, val_auc

---

---

## 📊 RESULTS/ DIRECTORY

### **predictions.csv**
**Ce face?** Predicții pe test set:
- true_label, predicted_label
- confidence scores

---

### **confusion_matrix_*.png**
**Ce face?** Vizualizare confusion matrix

---

### **roc_curve_*.png**
**Ce face?** ROC curve pentru model

---

---

## 🎯 REZUMAT RAPID

| Fișier | Scop | Status |
|--------|------|--------|
| README.md | Overview general | ✅ |
| SETUP_INSTRUCTIONS.md | Instalare | ✅ |
| src/preprocessing/ | Preproceseaza imagini | ✅ |
| src/neural_network/train.py | Antrenează model | ✅ |
| src/neural_network/evaluate.py | Evaluează model | ✅ |
| data/train/ | Date antrenare | ✅ |
| data/test/ | Date testare | ✅ |
| models/*.keras | Modele salvate | ✅ |
| logs/ | Log-uri antrenare | ✅ |
| docs/ | Documentație | ✅ |

---

**Status General:** 🟢 READY TO USE  
**Data:** 20 Ianuarie 2026
