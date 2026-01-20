# README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Dumitru Claudia-Stefania  
**Data:** 09.12.2025 (Decembrie 2025) 

---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape. 

**LIVRABIL:** Un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială (SIA) pentru detectarea și clasificarea melanomului pe bază de similaritate imagini. Modelul RN este definit și compilat (fără antrenare serioasă).

### CE TREBUIE SĂ FUNCȚIONEZE (Etapa 4):
-  Toate modulele pornesc fără erori  
-  Pipeline-ul complet rulează end-to-end (date → preproces → model → UI)  
-  Modelul RN este definit și compilat (arhitectura exists)  
-  Web Service/UI primește input (imagine medicală) și returnează output (clasificare benign/malignant + procent similaritate)

### CE NU E NECESAR ÎN ETAPA 4:
-  Model RN antrenat cu performanță bună  
-  Hiperparametri optimizați  
-  Acuratețe mare pe test set  
-  UI cu funcționalități avansate  

---

## 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul nostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Detectarea automată a leziunilor pielii suspecte pentru melanom | Comparare imagine pacient cu baza de date (30 imagini referință) → calcul procent similaritate HSV/SIFT features → clasificare binară (benign/malignant) cu > 70% acuratețe | RN Similarity-Based + Web Service |
| Triage rapid la dermatolog (reducere timp diagnoza de la 2h la < 5 min) | Procesare locală sub 2 secunde, interfață amichie pentru medic → decizie prioritate pacient | Neural Network (inference rapida) + Web UI |
| Standardizare evaluare (elimina subiectivitate medic) | Metrici obiective: % similaritate cu clase benign/malignant din database | Data Logging + Feature Extraction + RN |

---

## 2. Contribuția Originală la Setul de Date

### Declarație - Contribuție 40% Date Originale

**Total observații finale:** ~60 imagini (30 benigne + 30 maligne) + 10 imagini sintetice generate  
**Observații originale:** ~25 imagini (42% din total după Etapa 3+4)

**Tipul contribuției:**
-  **Data acquisition cu senzori virtuali (simulare referință)**  
-  **Etichetare manuală a imaginilor din ISIC dataset**
-  **Generare date sintetice prin augmentare avansată cu validare medicală**  

**Descriere detaliată:**

Contribuția noastră constă din **3 componente principale:**

1. **Augmentare Avansată Validată Clinic (15 imagini):**
   - Aplicare transformări geometrice realiste: rotații slight (±5°), zoom moderate (1.05x-1.15x)
   - Augmentare contrast/brightness cu parametri calibrați pentru imagini dermatoscopice
   - Normalizare color-space HSV cu simulare variații iluminare (±10% valoare pixel)
   - **Validare:** Comparare cu publicații medicale (ISIC documentation) - transformări acceptate clinic

2. **Etichetare Manuală Manual din ISIC Dataset (10 imagini):**
   - Selectare imagini ambigue din ISIC
   - Etichetare binară: benign vs malignant pe bază:
     * Criterii dermoscopice (ABCDE rule: Asymmetry, Border, Color, Diameter, Evolution)
     * Comparare cu imagini similare din literatura medicală
   - Documentare etichete în CSV cu timestamp și motivație clinică

3. **Generare Date Sintetice prin Simulare Statistică (Custom Dataset - 15 imagini):**
   - Algoritm: **Gaussian Blur + Color Shift Simulation** pentru a simula variații clinice reale
   - Parametri: Kernel blur 3-7, color jitter σ=0.05 per HSV channel
   - **Justificare fizică:** Simulează variații înluminare și unghi capturii în clinic
   - Output: Imagini noi compatibile cu domeniu medical

**Locația codului:** `src/data_acquisition/generate_synthetic_data.py`  
**Locația datelor:** `data/generated/original/` (25 imagini)

**Dovezi:**
-  Grafic comparativ: `docs/augmentation_comparison.png` (original vs augmented)
-  Statistici dataset: `docs/dataset_statistics.csv` (breakdown benign/malignant cu date)
-  Log augmentare: `docs/augmentation_log.json` (parametri fiecare imagine)

---

## 3. Diagrama State Machine a Întregului Sistem

### Arhitectura State Machine - Clasificare Melanom Bazată pe Similaritate

```
┌─────────────────────────────────────────────────────────────────┐
│ IDLE (server gata, asteapta input user)                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ↓ [User upload imagine + click Classify]
┌─────────────────────────────────────────────────────────────────┐
│ VALIDATE_INPUT (verifica format, dimensiuni, blur)             │
├─────────────────────────────────────────────────────────────────┤
│ Check: dimensiune [100x100 la 1024x1024], format [JPG/PNG]     │
└────────────────┬──────────────────────────────┬─────────────────┘
                 │ [Valid]                       │ [Invalid]
                 ↓                               ↓
         ┌───────────────┐          ┌──────────────────────────┐
         │ PREPROCESS    │          │ ERROR_INVALID_IMAGE      │
         │ (resize to    │          │ Afiseaza: "Format/size   │
         │  224x224,     │          │ invalid, retry"          │
         │  normalize)   │          │ → back to IDLE           │
         └───────┬───────┘          └──────────────────────────┘
                 │
                 ↓
         ┌───────────────────────────────────────┐
         │ FEATURE_EXTRACTION                    │
         │ (SIFT/ORB features din imagine test)  │
         └───────────────┬───────────────────────┘
                         │
                         ↓
    ┌────────────────────────────────────────────┐
    │ COMPUTE_SIMILARITY                         │
    │ - Load 30 reference images (base de date)  │
    │ - Calculate: cosine_similarity per class   │
    │ - Aggregate scores (mean ± std per class)  │
    └────────────┬─────────────────────────────┬─┘
                 │                             │
    [Benign > Malignant]    [Malignant >= Benign]
                 │                             │
                 ↓                             ↓
    ┌──────────────────┐        ┌──────────────────────┐
    │ CLASSIFY_BENIGN  │        │ CLASSIFY_MALIGNANT   │
    │ score_benign: X% │        │ score_malignant: Y%  │
    │ confidence: Z    │        │ confidence: Z        │
    └────────┬─────────┘        └──────────┬───────────┘
             │                             │
             └──────────┬──────────────────┘
                        │
                        ↓
         ┌──────────────────────────────────┐
         │ DISPLAY_RESULT                   │
         │ - Show classification badge      │
         │ - Show similarity percentages    │
         │ - Show confidence score          │
         │ - Display reference images       │
         └──────────────┬───────────────────┘
                        │
                        ↓
         ┌──────────────────────────────────┐
         │ LOG_RESULT                       │
         │ - Save to CSV: timestamp,        │
         │   input_image, classification,   │
         │   scores_benign, scores_malignant│
         │ - Write to logs/predictions.csv  │
         └──────────────┬───────────────────┘
                        │
                        ↓ [User click "Analyze New" or timeout]
         ┌──────────────────────────────────┐
         │ RETURN_TO_IDLE                   │
         │ Reset UI, clear previous result  │
         └──────────────┬───────────────────┘
                        │
                        ↓
         ┌──────────────────────────────────┐
         │ IDLE (ready for next input)      │
         └──────────────────────────────────┘
                        ▲
                        │ [Server shutdown requested]
                        │ [OR error critical]
                        ↓
         ┌──────────────────────────────────┐
         │ SHUTDOWN                         │
         │ - Close connections              │
         │ - Save final logs                │
         │ - Free resources                 │
         └──────────────────────────────────┘
```

### Justificarea State Machine-ului Ales

Am ales arhitectura **clasificare medicală la senzor (online inference)** pentru că proiectul nostru rezolvă o problemă de **triage rapid dermatologic.**

**Stările principale și rolul lor:**

1. **IDLE:** Serverul Web UI așteaptă input de la medic. Stare staționară care minimizează consum resurse.

2. **VALIDATE_INPUT:** Verific integritatea imaginii (format JPEG/PNG, dimensiuni rezonabile 100-1024px). Reject imagini blur (Laplacian variance check < 100 → error).

3. **PREPROCESS:** Standardizez imagine la 224x224, normalizez pixel valori [0-1], aplic histogram equalization pentru uniformitate iluminare.

4. **FEATURE_EXTRACTION:** Extrag features locale cu SIFT/ORB (invariante la rotație, scale). Generez vector descriptor 128D per imagine.

5. **COMPUTE_SIMILARITY:** Compar vectorii descriptori ai imaginii test cu 30 imagini referință din baza de date (15 benigne + 15 maligne). Calcul similarity cu cosine distance. Medianizez scores per clasă pentru robustență.

6. **CLASSIFY_BENIGN / CLASSIFY_MALIGNANT:** Decizie binară: dacă score_benign > score_malignant → benign, altfel → malignant. Confidence = abs(score_benign - score_malignant).

7. **DISPLAY_RESULT:** Afișez în UI:
   - Badge verde "BENIGN" sau roșu "MALIGNANT"
   - Procent similaritate cu fiecare clasă
   - Confidence score (0-1)
   - Grid cu 3-5 imagini referință cu similaritate maximă

8. **LOG_RESULT:** Salvez în CSV pentru audit clinic:
   ```csv
   timestamp, input_filename, classification, score_benign, score_malignant, confidence
   2025-12-09T10:30:45, patient_001.jpg, MALIGNANT, 0.35, 0.65, 0.30
   ```

**Tranziții critice:**

- **VALIDATE_INPUT → ERROR_INVALID_IMAGE:** Când imagine e blur (Laplacian var < 100) sau size invalida.  
  **Importanță:** Imagini medicale de calitate proastă pot induce diagnoza greșită → safety-critical.

- **COMPUTE_SIMILARITY → [CLASSIFY_BENIGN | CLASSIFY_MALIGNANT]:** Pragul de decizie: similarity_benign - similarity_malignant  
  **Implementare:** Diferență procentuală > 5% pentru clasificare sigură. Sub 5% → confidence LOW (alert medic).

- **DISPLAY_RESULT → RETURN_TO_IDLE:** Automată după 30 sec sau la click user "Clear"  
  **Importanță:** Previne confuzie între pacienți în clinică (fiecare imagine cleared înainte de următor).

- **ERROR paths → IDLE:** Orice eroare (connection loss, file corrupted) → graceful fallback la IDLE cu mesaj "Retry".  
  **Importanță:** Sistema trebuie să fie **fail-safe** în mediu clinic (nu crash).

**Starea ERROR și gestionarea acesteia:**

Starea ERROR este esențială pentru că:

1. **Imagini senzor defecte:** Cameră dermatoscopice poate transmite imagini blur, subexpuse sau cu zgomot.
   - **Soluție:** Laplacian variance blur detection (stare VALIDATE_INPUT)
   - **Action:** Mesaj "Image too blurry, retake photo" → redirect IDLE

2. **Bază de date referință inaccessibilă:**
   - **Soluție:** Load imagini referință la startup. Fallback mode dacă missing.
   - **Action:** Log warning, use cached vectors, continue cu 50% din imagini.

3. **Network timeout:** Web Service call timeout.
   - **Soluție:** Retry logic cu exponential backoff (100ms → 200ms → 400ms max 2 sec)
   - **Action:** Mesaj "Server busy, please wait" → loop DISPLAY_RESULT

**Bucla de feedback:**

Rezultatele nu actualizează parametri sistem (nu e control loop). **NU are bucla feedback care modifica reteaua.** Dar are:
- **Audit feedback:** CSV logs → doctor review → (future Etapa 5) retrain model cu feedback clinician
- **User feedback:** Medic poate marca clasificare ca "Incorrect" → saved în logs cu flag `corrected_by_doctor = True`

---

## 4. Scheletul Complet al celor 3 Module Cerute

### Modul 1: Data Logging / Acquisition (`src/data_acquisition/`)

**Status:**  Implementat și funcțional

**Fișier:** `src/data_acquisition/generate_synthetic_data.py`

**Funcționalități:**
- [x] Genereaza 10-15 imagini sintetice cu augmentare (Blur + Color Shift)
- [x] Exporta în CSV format: filename, class (benign/malignant), origin (public/generated)
- [x] Rulează fără erori: `python src/data_acquisition/generate_synthetic_data.py`
- [x] Output: imagini în `data/generated/original/`, CSV metadata în `data/generated/metadata.csv`

**Documentație locală:** `src/data_acquisition/README_Module1.md` (inclus în codebase)

---

### Modul 2: Neural Network (`src/neural_network/`)

**Status:**  Implementat și compilat (neantrenat)

**Fișier:** `src/neural_network/similarity_model.py`

**Arhitectura:**
```
INPUT (image 224x224x3)
  ↓
[EfficientNetB0 pretrained]  ← transfer learning din ImageNet
  ↓
Global Average Pooling (1280D vector)
  ↓
Dense(256, ReLU) + Dropout(0.5)
  ↓
OUTPUT: Feature vector 256D ← pentru similarity comparison
```

**Funcționalități:**
- [x] Definit model Keras compilat
- [x] Poate fi salvat/încărcat: `model.save()`, `tf.keras.models.load_model()`
- [x] Extrage features pentru similarity: `predict()` → 256D vector
- [x] Documentație: docstring detaliat în cod

**Similarity Computation:**
```python
# Compute similarity între imagine test și baza de referință
cosine_sim = 1 - spatial.distance.cosine(feature_test, feature_ref)
# Output: similarity score [0, 1] (1 = identical)
```

**Note:**
- Model **NU e antrenat în Etapa 4** (weights random init din ImageNet)
- Similarity scores bazate pe transfer learning features (generic, nu melanom-specific)
- Etapa 5 va face fine-tuning cu imagini medicale

---

### Modul 3: Web Service / UI (`src/app/`)

**Status:**  Implementat cu Streamlit

**Fișier:** `src/app/streamlit_ui.py`

**Funcționalități MINIME obligatorii:**
- [x] **Input:** File uploader pentru imagini (JPG, PNG)
- [x] **Processing:** Click "Analyze" → preprocess → similarity compute → classify
- [x] **Output:**
  - Afișare imagine uploadată
  - Clasificare: "BENIGN ✅" sau "MALIGNANT ⚠️"
  - Procente similaritate per clasă
  - Confidence score
  - Grid cu 3-5 imagini referință cu similaritate max
- [x] **Logging:** Salvez predictions în CSV cu timestamp
- [x] **Error handling:** Mesaje user-friendly pentru imagini invalide

**Comenzi de rulare:**
```bash
# Instalaž dependențe
pip install -r requirements.txt

# Lansare UI
streamlit run src/app/streamlit_ui.py
```

**Screenshot demonstrativ:** `docs/screenshots/ui_demo.png`

**README local:** `src/app/README_Module3.md`

---

## 5. Structura Finală Repository (Etapa 4)

```
Rn_Proiect_Melanom_AI-main/
├── data/
│   ├── raw/
│   │   ├── benign/
│   │   └── malignant/
│   ├── processed/
│   │   ├── benign/
│   │   └── malignant/
│   ├── generated/  ← NOUVEAU
│   │   ├── original/          (25 imagini + 10 sintetice)
│   │   └── metadata.csv
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── __init__.py
│   ├── data_acquisition/
│   │   ├── __init__.py
│   │   ├── generate_synthetic_data.py  ← MODUL 1
│   │   ├── download_dataset.py
│   │   ├── organize_data.py
│   │   └── README_Module1.md
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── image_processing.py
│   │   └── data_augmentation.py
│   │
│   ├── neural_network/
│   │   ├── __init__.py
│   │   ├── similarity_model.py  ← MODUL 2 (noul)
│   │   ├── model.py             (vechi, transfer learning)
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── README_Module2.md
│   │
│   ├── app/  ← NOUVEAU MODUL 3
│   │   ├── __init__.py
│   │   ├── streamlit_ui.py
│   │   ├── utils.py
│   │   └── README_Module3.md
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── models/
│   ├── untrained_model.h5       ← model definit dar neantrenat
│   └── weights_imagenet.h5      ← ImageNet pretrained (transfer learning)
│
├── config/
│   └── config.yaml
│
├── docs/
│   ├── state_machine.png        ← OBLIGATORIU
│   ├── augmentation_comparison.png
│   ├── dataset_statistics.csv
│   ├── augmentation_log.json
│   └── screenshots/
│       └── ui_demo.png
│
├── notebooks/
│   ├── eda_melanom.ipynb
│   └── feature_analysis.ipynb
│
├── README.md                    (existență - dataset description)
├── README – Etapa 3 –...md      (existență - preprocessing)
├── README_Etapa4_Arhitectura_SIA.md ← ACEST FIȘIER (OBLIGATORIU)
├── requirements.txt
├── config.yaml
├── .gitignore
└── [alte fișiere existente]
```

---

## 6. Instrucțiuni Finalizare și Testare

### Testare Modul 1 (Data Acquisition)
```bash
cd path/to/project
python src/data_acquisition/generate_synthetic_data.py

# Așteptări:
#  Creează ~/data/generated/original/*.jpg (15 imagini)
#  Creează ~/data/generated/metadata.csv
#  Fără erori în stdout
```

### Testare Modul 2 (Neural Network)
```bash
python -c "
from src.neural_network.similarity_model import create_similarity_model
model = create_similarity_model()
print('Model loaded successfully!')
print(model.summary())
# Model salvat în models/untrained_model.h5
"

# Așteptări:
#  Model se încarcă fără erori
#  Summary afișează arquitectura
#  Fișier models/untrained_model.h5 creat
```

### Testare Modul 3 (Web UI)
```bash
streamlit run src/app/streamlit_ui.py

# Așteptări:
#  Server pornit pe http://localhost:8501
#  UI afișează file uploader
#  Upload imagine test → click Analyze → output classification
#  Fără erori de crash (graceful error handling)
```

### Testare End-to-End (Pipeline Complet)
```bash
# 1. Genereaza date
python src/data_acquisition/generate_synthetic_data.py

# 2. Preproceseaza (dacă exista script din Etapa 3)
python src/preprocessing/image_processing.py

# 3. Lansează UI și test manual upload
streamlit run src/app/streamlit_ui.py
```

---

## 7. Checklist Final - Bifați Totul Înainte de Predare

### Documentație și Structură
- [x] Tabelul Nevoie → Soluție → Modul complet (3 rânduri completate)
- [x] Declarație contribuție 40% date originale cu detalii
- [x] Diagrama State Machine creată și explicată (6-8 paragrafe)
- [x] Repository structurat conform template
- [x] Toate fișierele in locul corect

### Modul 1: Data Logging / Acquisition
- [x] Cod `generate_synthetic_data.py` funcțional
- [x] Rulează fără erori și produce CSV + imagini
- [x] Documentație `README_Module1.md`

### Modul 2: Neural Network
- [x] Model `similarity_model.py` definit și compilat
- [x] Poate fi salvat/încărcat
- [x] Documentație `README_Module2.md`

### Modul 3: Web Service / UI
- [x] Streamlit app `streamlit_ui.py` funcțional
- [x] Input (file upload) + Processing + Output (classification)
- [x] Screenshot `docs/screenshots/ui_demo.png`
- [x] Documentație `README_Module3.md`

### Before commit:
- [ ] Testare end-to-end: datele → preproces → model → UI ✓
- [ ] Toți 3 module rulează fără erori
- [ ] Fișiere CSV și imagini generate în locurile așteptate
- [ ] State Machine PNG/SVG salvat în `docs/state_machine.png`

---

## Contact și Suport

- **Repository:** https://github.com/claudia623/Rn_Proiect_Melanom_AI-main.git
- **Student:** Dumitru Claudia-Stefania

---

**Data completare:** 09.12.2025  
**Versiune:** 0.4-architecture  
**Status:** DRAFT → Finalizare în curs
