# Melanom AI - Sistem Clasificare cu Retele Neuronale

## Descriere Proiect

Sistem inteligent pentru clasificare automata a melanomului (benign vs malignant) folosind retele neuronale profunde. Am implementat transfer learning pe EfficientNetB0 si am antrenat modelul pe 206 imagini dermatoscopice din dataset-ul ISIC.

## Structura Proiect

Proiectul este organizat asa:

```
Rn_Proiect_Melanom_AI/
├── README.md                          - Overview general
├── etapa3_analiza_date.md             - Analiza dataset
├── etapa4_arhitectura_sia.md          - Arhitectura model
├── etapa5_antrenare_model.md          - Antrenare si evaluare
├── etapa6_optimizare_concluzii.md     - Optimizare si concluzii
├── SETUP_INSTRUCTIONS.md              - Instalare si rulare
│
├── docs/                              - Documentatie si grafice
│   ├── datasets/README.md             - Info dataset
│   ├── README_Module3_WebUI.md        - Interfata web
│   ├── results/                       - Grafice si rezultate
│   │   ├── learning_curves_phase1.png
│   │   ├── learning_curves_phase2.png
│   │   ├── confusion_matrix_optimized.png
│   │   └── roc_curve_final.png
│   ├── error_analysis/                - Imagini clasate gresit
│   └── optimization/                  - Documente optimizare
│
├── data/                              - Dataset
│   ├── raw/                           - Imagini originale
│   ├── processed/                     - Imagini prelucrate
│   ├── train/                         - Set antrenare (70%)
│   ├── validation/                    - Set validare (15%)
│   └── test/                          - Set test (15%)
│
├── src/                               - Cod sursa
│   ├── data_acquisition/              - Achizitie date
│   ├── preprocessing/                 - Prelucrare imagini
│   ├── neural_network/                - Model si antrenare
│   ├── app/                           - Interfata web
│   └── utils/                         - Functii auxiliare
│
├── models/                            - Modele antrenate
│   ├── melanom_efficientnetb0_best.keras
│   └── trained_model.keras
│
├── results/                           - Rezultate si metrici
│   ├── optimization_experiments.csv
│   └── final_metrics.json
│
├── config/                            - Configuratii
│   └── optimized_config.yaml
│
└── requirements.txt                   - Dependente
```

## Quick Start

### 1. Instalare

```bash
pip install -r requirements.txt
```

### 2. Pregatire Date

```bash
python src/preprocessing/preprocess_dataset.py
python src/preprocessing/split_processed_data.py
```

### 3. Antrenare Model

```bash
python src/neural_network/train.py
```

### 4. Evaluare

```bash
python src/neural_network/evaluate.py --use-best
```

### 5. Interfata Web (optionala)

```bash
streamlit run src/app/main.py
```

## Rezultate Model

Accuracy: 70%
AUC Score: 0.85
Precision: 78%
Recall: 68%
F1-Score: 0.72

## Documente Etape

Etapa 3 - Analiza si Pregatire Date
Analiza dataset-ului ISIC cu 206 imagini, preprocesare si distributie train/validation/test.

Etapa 4 - Arhitectura SIA
Model EfficientNetB0 cu transfer learning, strategie antrenare, evaluare initiala.

Etapa 5 - Antrenare Model
Antrenare in doua faze, metrici training, analiza detaliata a rezultatelor.

Etapa 6 - Optimizare si Concluzii
Hyperparameter tuning, comparatie experimente, concluzii si recomandari.

## Tehnologii

- Python 3.8+
- TensorFlow/Keras - Framework ML
- OpenCV - Procesare imagini
- NumPy/Pandas - Manipulare date
- Matplotlib/Seaborn - Vizualizari
- Streamlit - Interfata web

## Autor

Dumitru Claudia-Stefania

## Data Finalizare

20 ianuarie 2026

## Pentru Mai Multe Informatii

- Consultati SETUP_INSTRUCTIONS.md pentru instalare detaliate
- Vezi docs/ pentru documentatie completa
- Explore results/ pentru metrici
