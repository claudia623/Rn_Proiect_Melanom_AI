# VISUALIZĂRI ETAPA 5 - TRAINING & EVALUATION

## Descriere Fișiere Generate

### 1. phase1_loss_accuracy.png
**Path:** `docs/phase1_loss_accuracy.png`

Conține două subplots:
- **Stânga:** Loss curves (Train vs Validation)
  - X: Epoch (1-11)
  - Y: Loss (Binary Crossentropy)
  - Blue line: Training loss (descrescător din 0.693 → 0.476)
  - Red line: Validation loss (descrescător din 0.614 → 0.489)
  - **Interpretare:** Loss converge smooth, fără overfitting (train și val se apropie)

- **Dreapta:** Accuracy curves (Train vs Validation)
  - X: Epoch (1-11)
  - Y: Accuracy (0-1)
  - Blue line: Training accuracy (crescând din 56% → 86%)
  - Red line: Validation accuracy (crescând din 58% → 91%)
  - **Interpretare:** Model învață progresiv; val_acc > train_acc (normal pentru dataset mic)

**Status:** ✓ Generat de `generate_plots_simple.py`

---

### 2. phase2_loss_accuracy.png
**Path:** `docs/phase2_loss_accuracy.png`

Conține două subplots pentru Phase 2 (Fine-tuning):
- **Stânga:** Loss curves
  - X: Epoch (1-10)
  - Y: Loss (Binary Crossentropy)
  - Converge din 0.362 → 0.296
  - Validation loss: 0.468 → 0.423
  - **Interpretare:** Fine-tuning reușit; gap train-val stabil

- **Dreapta:** Accuracy curves
  - Accuracy: 88% → 89%
  - Val Accuracy: 86% → 91%
  - **Interpretare:** Creștere marginală dar stabilă

**Status:** ✓ Generat de `generate_plots_simple.py`

---

### 3. auc_curves_combined.png
**Path:** `docs/auc_curves_combined.png`

Subplot unique combinând ambele faze:
- **X-axis:** Epoch combined (1-21, cu linie separatoare la epoch 11-12)
- **Y-axis:** AUC (ROC) Score (0.6-1.0)

**Liniile:**
- Blue solid: Phase 1 Train AUC
- Blue dashed: Phase 1 Val AUC
- Green solid: Phase 2 Train AUC  
- Green dashed: Phase 2 Val AUC
- Red dotted line: Separare faze

**Evoluție:**
- Phase 1: Val AUC de 0.631 → 0.928
- Phase 2: Val AUC de 0.917 → 0.960 (BEST)
- Test AUC final: 0.8114 (gap = overfitting ușor pe val)

**Status:** ✓ Generat de `generate_plots_simple.py`

---

### 4. confusion_matrix_detailed.png
**Path:** `docs/confusion_matrix_detailed.png`

Heatmap colorat (Blue colormap):
```
                 Predicted
                Benign  Malignant
Actual Benign      7        7      (50% acc)
Actual Malignant   1       19      (95% acc)
```

Texto adnotat:
- Sensitivity (Recall): 95.00%
- Specificity: 50.00%
- Precision: 73.08%

**Interpretare:**
- ✓ Bună detectare melanom (95% sensitivity)
- ✗ Slabă specificitate benign (50%)
- **Implicație:** 7 false alarms din 14 cazuri benigne

**Status:** ✓ Generat de `generate_plots_simple.py`

---

## Instrucțiuni Regenerare

Dacă doriți să regenerați aceste grafice:

```bash
cd Rn_Proiect_Melanom_AI-main/
python generate_plots_simple.py
```

**Dependențe necesare:**
- matplotlib (2.3+)
- numpy (1.20+)
- pandas (1.2+)

---

## Metode Alternative Vizualizare

### TensorBoard
```bash
tensorboard --logdir=logs/
```
Accesează la `http://localhost:6006/`

Furnizează:
- Loss curves interactive
- Accuracy curves
- Histograme weight
- Distribution metrics

### Jupyter Notebook
Rulează `notebooks/visualization.ipynb` pentru:
- Ploturi interactive (plotly)
- Slider epoch
- Zoom/pan capability

---

## Note Tehnice

### Data Source
- Phase 1 history: `results/melanom_efficientnetb0_phase1_history.json`
- Phase 2 history: `results/melanom_efficientnetb0_phase2_history.json`
- Test set: `data/test/benign/` și `data/test/malignant/`

### Resolution
- DPI: 300 (publication quality)
- Format: PNG (lossless)
- Size typical: 1400x500 pentru 1x2 subplot

### Color Scheme
- Train: Blue (#1f77b4)
- Validation: Red (#ff7f0e)
- Phase 2: Green (#2ca02c)
- Grid: Light gray (alpha=0.3)

---

**Generat:** 12.01.2026  
**Versiune:** 1.0
