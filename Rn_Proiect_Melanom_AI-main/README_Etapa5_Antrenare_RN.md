# Etapa 5: Antrenarea și Evaluarea Rețelei Neuronale



**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Dumitru Claudia-Stefania 
**Link Repository GitHub:** https://github.com/claudia623/Rn_Proiect_Melanom_AI-main  
**Data predării:** 16.12.2025


## 1. Obiective
- Implementarea pipeline-ului de antrenare folosind TensorFlow/Keras
- Antrenarea modelului EfficientNetB0 pe setul de date hibrid (ISIC + Sintetic)
- Evaluarea performanței modelului
- Integrarea modelului antrenat în interfața Streamlit

## 2. Arhitectura Modelului
- **Base Model**: EfficientNetB0 (pre-antrenat pe ImageNet)
- **Input**: Imagini 224x224x3
- **Layers Adăugate**:
  - GlobalAveragePooling2D
  - Dense (512 units, ReLU, BatchNormalization, Dropout 0.5)
  - Dense (256 units, ReLU, BatchNormalization, Dropout 0.5)
  - Output Dense (1 unit, Sigmoid)

## 3. Procesul de Antrenare
- **Mediu**: Python 3.13, TensorFlow 2.20 (CPU optimized environment)
- **Dataset**:
  - Train: 132 imagini (66 Benign / 66 Malignant)
  ````markdown
  # 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

  **Disciplina:** Rețele Neuronale  
  **Instituție:** POLITEHNICA București – FIIR  
  **Student:** Dumitru Claudia-Stefania  
  **Link Repository GitHub:** https://github.com/claudia623/Rn_Proiect_Melanom_AI-main  
  **Data predării:** 16.12.2025

  ---

  ## Scopul Etapei 5

  Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape.

  **Obiectiv principal:** Antrenarea modelului EfficientNetB0 definit în Etapa 4, evaluarea performanței pe setul de test și integrarea modelului antrenat în aplicația Streamlit.

  ---

  ## Rezumat Proiect-specific

  - **Arhitectură folosită:** EfficientNetB0 (pre-antrenat pe ImageNet) + head personalizat
  - **Dimensiune input:** 224x224x3
  - **Head adăugat:** GlobalAveragePooling2D → Dense(512, ReLU, BatchNormalization, Dropout 0.5) → Dense(256, ReLU, BatchNormalization, Dropout 0.5) → Dense(1, Sigmoid)
  - **Model salvat:** `models/melanom_efficientnetb0_best.keras`
  - **Scripturi principale:** `src/neural_network/train.py`, `src/neural_network/evaluate.py`

  ---

  ## Date folosite

  - **Train:** 132 imagini (≈ 66 Benign / 66 Malignant)
  - **Validation:** 26 imagini
  - **Test:** 34 imagini

  Split-urile au fost realizate stratificat, respectând proporțiile planificate.

  ---

  ## Tabel Hiperparametri (Nivel 1 - OBLIGATORIU)

  | Hiperparametru | Valoare utilizată | Justificare |
  |---|---:|---|
  | Learning rate (inițial) | 1e-3 (Faza 1) → reduced on plateau | 1e-3 e un punct de plecare standard pentru Adam; ReduceLROnPlateau a scăzut lr când val_auc stagnat pentru stabilitate
  | Batch size | 32 | Compromis memorie / stabilitate; potrivit pentru set mic-mediu de imagini
  | Number epochs (max) | 25 (Faza 1) + 25 (Faza 2) | Două faze: transfer learning urmat de fine-tuning; early stopping a oprit la epoca 11 (Faza 1) și 10 (Faza 2)
  | Optimizer | Adam | Adaptive, stabilizează antrenarea pentru transfer learning
  | Loss function | Binary Crossentropy | Problemă binară (Benign vs Malignant)
  | Activation functions | ReLU (hidden), Sigmoid (output) | ReLU pentru non-linearitate, Sigmoid pentru scor probabilistic binar

  Justificare batch size: batch_size=32 a oferit un echilibru între acuratețe a estimării gradientului și timp/consum memorie pe CPU.

  ---

  ## Procesul de Antrenare și Configurație

  - Faza 1 — Transfer learning: base EfficientNetB0 înghețat, head antrenat; max 25 epoci, early stopping (patience 10) — antrenare oprită la epoca 11.
  - Faza 2 — Fine-tuning: ultimele 30 de layere dezghețate, lr redus la 1e-5, max 25 epoci, early stopping — oprit la epoca 10.
  - Callback-uri: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`.

  Rezultate intermediare (exemple din log): best `val_auc` observat ≈ 0.65385 în timpul fazei 1 și ulterior îmbunătățiri în fine-tuning.

  ---

  ## Rezultate Obținute (Pe Setul de Test)

  - **Accuracy:** 70.59%
  - **AUC (ROC):** 0.8114
  - **Loss (test):** 0.5286
  - **Precision (malignant):** 0.6400
  - **Recall (malignant / sensitivity):** 0.9412
  - **F1-score (malignant):** ≈ 0.76

  Metricile au fost produse de `src/neural_network/evaluate.py` și salvate în `results/` (vezi `results/confusion_matrix.png` și fișierele de history JSON din `results/`).

  ### Interpretare scurtă
  Modelul are o sensibilitate (recall) foarte ridicată — 94% — ceea ce este de preferat într-un context medical/screening: majoritatea cazurilor maligne sunt identificate. Precizia este mai mică (64%), deci există cazuri false-positive în care se recomandă triere clinică ulterioară. AUC > 0.81 indică o separare bună a claselor.

  ---

  ## Integrare UI

  - UI Streamlit actualizat pentru a încărca modelul antrenat: `models/melanom_efficientnetb0_best.keras`.
  - Fisier UI principal: `src/app/streamlit_ui.py` (sau `src/app/main.py` conform structurii)
  - Funcționalități demonstrabile:
    - Inferență reală (upload imagine) → predicție Benign / Malignant
    - Confidence score (probabilitate sigmoid)
    - Vizualizare rezultat + link către `docs/screenshots/inference_real.png` (exemplu)

  ---

  ## Analiză Erori (Nivel 2 – obligatoriu)

  1) Pe ce clase greșește cel mai mult modelul?

  - Confusion matrix (salvată în `results/confusion_matrix.png`) arată mai multe false positives decât false negatives.
  - Cauză probabilă: variații de iluminare, artefacte de imagistică și similitudini vizuale între leziuni benigne și unele tumori atipice.

  2) Ce caracteristici ale datelor cauzează erori?

  - Fundal neomogen, glint (reflexii) și variații de colorit ale pielii.
  - Mix-ul de imagini sintetice și reale poate introduce diferențe de distribuție.

  3) Implicații pentru aplicație medicală

  - Prioritate: minimizarea falselor negative (miss = caz malign netestat) — acesta este motivul pentru un threshold mai conservator și pentru optimizarea recall-ului.

  4) Măsuri corective propuse

  - Colectare de imagini suplimentare pentru cazuri greu clasificate (≥200 imagini pentru fiecare caz atipic)
  - Ajustare prag (threshold) pentru clasa 'malignant' pentru a favoriza recall (ex: 0.4 → 0.3)
  - Augmentări specifice (lighting jitter, hist. equalization, crop variation)
  - Reantrenare cu `class_weights` sau oversampling pentru clase subtile

  ---

  ## Fișiere relevante generate

  - Model salvat: `models/melanom_efficientnetb0_best.keras`
  - Confusion matrix: `results/confusion_matrix.png`
  - Training history: `results/melanom_efficientnetb0_phase1_history.json`, `results/melanom_efficientnetb0_phase2_history.json`
  - Logs TensorBoard: `logs/`

  ---

  ## Instrucțiuni de Rulare (scurt)

  1. Instalează dependențele:
  ```powershell
  C:\Users\40770\Desktop\v\Scripts\pip.exe install -r requirements.txt
  ```

  2. Antrenare (exemplu):
  ```powershell
  C:\Users\40770\Desktop\v\Scripts\python.exe src/neural_network/train.py --epochs 25 --batch_size 32
  ```

  3. Evaluare:
  ```powershell
  C:\Users\40770\Desktop\v\Scripts\python.exe src/neural_network/evaluate.py --model models/melanom_efficientnetb0_best.keras
  ```

  4. Rulare UI (folosind venv):
  ```powershell
  C:\Users\40770\Desktop\v\Scripts\python.exe -m streamlit run Rn_Proiect_Melanom_AI-main/src/app/streamlit_ui.py
  ```

  ---

  ## Checklist Final (completat parțial în repo)

  - [x] Model antrenat și salvat (`models/melanom_efficientnetb0_best.keras`)
  - [x] Scripturi `train.py` și `evaluate.py` prezente în `src/neural_network/`
  - [x] Confusion matrix generată (`results/confusion_matrix.png`)
  - [x] Metrici de test în README (vezi secțiunea "Rezultate Obținute")

  ---

  ## Observații finale

  Acest fișier respectă template-ul de Etapa 5 primit și conține valorile reale obținute în rulările de training/evaluare. Dacă doriți pot:

  - Adăuga graficele `loss` / `val_loss` în `docs/` și un plot detaliat al ROC
  - Rula o analiză detaliată a celor mai frecvente 5 erori și salva rapoartele în `docs/`

  ````
