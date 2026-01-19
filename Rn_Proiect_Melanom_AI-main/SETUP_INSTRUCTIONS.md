# 🚀 COMPLETE SETUP INSTRUCTIONS

## Prerequisites

- **Python:** 3.8 or higher
- **Operating System:** Windows, macOS, or Linux
- **Disk Space:** ~5 GB (for models, data, and dependencies)
- **GPU (Optional):** NVIDIA GPU with CUDA support for faster training

---

## Step 1: Environment Setup

### 1.1 Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1.2 Verify Python Installation

```bash
python --version
python -c "import sys; print(f'Python {sys.version}')"
```

Expected output: `Python 3.8.x` or higher

---

## Step 2: Install Dependencies

### 2.1 Upgrade pip

```bash
pip install --upgrade pip
```

### 2.2 Install Project Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- TensorFlow/Keras
- OpenCV
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- And other required packages

### 2.3 Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
```

---

## Step 3: Prepare Project Structure

### 3.1 Automatic Setup (Recommended)

```bash
make setup
```

or manually:

```bash
python -c "from src.utils.constants import ensure_directories; ensure_directories()"
```

### 3.2 Verify Structure

```bash
# Linux/macOS
tree -L 2

# Windows (PowerShell)
Get-ChildItem -Recurse -Directory | Format-Table -AutoSize
```

---

## Step 4: Prepare Data

### 4.1 Place Raw Images

Place your melanoma images in:
```
data/raw/
├── benign/          (benign lesion images)
└── malignant/       (malignant lesion images)
```

**Expected structure:**
```
data/raw/
├── benign/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── malignant/
    ├── lesion_001.jpg
    ├── lesion_002.jpg
    └── ...
```

### 4.2 Data Requirements

- **Image Format:** JPG, PNG, BMP, or TIFF
- **Minimum Size:** 100×100 pixels
- **Image Quality:** Clear, well-focused images
- **Classes:** Must be organized in subdirectories by class (benign/malignant)

---

## Step 5: Run Preprocessing Pipeline

### 5.1 Preprocess Raw Images

```bash
python src/preprocessing/preprocess_dataset.py
```

**What it does:**
- Resizes images to 224×224 pixels
- Removes hair using morphological operations
- Applies CLAHE for contrast enhancement
- Sharpens images using Unsharp Masking
- Validates image quality (blur detection)
- Saves processed images to `data/processed/`

**Duration:** ~1-5 minutes (depending on image count)

### 5.2 Split Data into Train/Validation/Test

```bash
python src/preprocessing/split_processed_data.py
```

**What it does:**
- Splits images in 70/15/15 ratio
- Creates train/validation/test directories
- Ensures balanced class distribution

**Output:**
```
data/
├── train/       (70% of images)
├── validation/  (15% of images)
└── test/        (15% of images)
```

### 5.3 Preprocess Test Data (Optional)

```bash
python src/preprocessing/preprocess_test_data.py
```

Ensures test data is preprocessed consistently.

---

## Step 6: Train Model

### 6.1 Start Training

```bash
python src/neural_network/train.py
```

**Training Configuration:**
- **Model:** EfficientNetB0 (Transfer Learning)
- **Phase 1:** Train custom head with frozen base
  - Duration: ~5-10 minutes
  - Epochs: Up to 30 (with early stopping)
- **Phase 2:** Fine-tune last 30 layers
  - Duration: ~10-20 minutes
  - Epochs: Up to 50 (with early stopping)

**Total Training Time:** 15-30 minutes on GPU, 1-3 hours on CPU

### 6.2 Monitor Training

The script provides:
- Real-time metrics (loss, accuracy, AUC)
- Validation metrics every epoch
- Model checkpointing (saves best model)
- Training logs saved to `logs/`

**Expected Performance:**
- Validation Accuracy: ~80%
- Validation AUC: ~0.89

### 6.3 Resume Training (If Interrupted)

```bash
python src/neural_network/train.py --resume
```

---

## Step 7: Evaluate Model

### 7.1 Evaluate on Test Set

```bash
python src/neural_network/evaluate.py --use-best
```

**Generates:**
- Confusion Matrix
- ROC Curve
- Classification Metrics (Precision, Recall, F1-Score)
- Detailed predictions CSV

**Output Location:** `results/`

### 7.2 View Results

Results are saved as:
- `results/confusion_matrix_*.png` - Visualization
- `results/roc_curve_*.png` - ROC curve
- `results/predictions_*.csv` - Detailed predictions
- `results/evaluation_results_*.csv` - Summary metrics

---

## Step 8: Generate Visualizations

### 8.1 Create Comprehensive Visualizations

```bash
python docs/generate_etapa5_visualizations.py
```

Creates:
- Training history plots
- Confusion matrix heatmaps
- ROC curves
- Data distribution charts
- Model performance comparisons

---

## Complete Pipeline (One Command)

### Option 1: Using Make

```bash
make full-pipeline
```

### Option 2: Manual Execution

```bash
# Setup
python -c "from src.utils.constants import ensure_directories; ensure_directories()"

# Preprocess
python src/preprocessing/preprocess_dataset.py
python src/preprocessing/split_processed_data.py
python src/preprocessing/preprocess_test_data.py

# Train
python src/neural_network/train.py

# Evaluate
python src/neural_network/evaluate.py --use-best

# Visualize
python docs/generate_etapa5_visualizations.py
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Solution: Install CUDA and cuDNN if available
# Or the training will automatically use CPU
```

### Out of Memory

Reduce batch size in `config/config.yaml`:
```yaml
batch_size: 8  # Reduce from default 16
```

### Image Loading Issues

Verify image formats:
```bash
python -c "
import cv2
import os
for root, dirs, files in os.walk('data/raw'):
    for f in files:
        if f.endswith(('.jpg', '.png', '.bmp')):
            print(f'{f}: OK')
"
```

### Missing Dependencies

Reinstall all dependencies:
```bash
pip install -r requirements.txt --upgrade --force-reinstall
```

---

## Project Structure Reference

```
Melanoma-Detection/
├── data/
│   ├── raw/              ← Place your images here
│   ├── processed/        ← Preprocessed images
│   ├── train/            ← Training set
│   ├── validation/       ← Validation set
│   └── test/             ← Test set
├── src/
│   ├── preprocessing/    ← Image processing scripts
│   ├── neural_network/   ← Model training/evaluation
│   └── utils/            ← Helper functions
├── models/               ← Saved trained models
├── logs/                 ← Training logs
├── results/              ← Evaluation results
├── config/               ← Configuration files
├── docs/                 ← Documentation
├── requirements.txt      ← Python dependencies
└── README.md             ← Main documentation
```

---

## Key Configuration Files

### `config/config.yaml`
Contains model and training hyperparameters:
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

### `requirements.txt`
Lists all Python dependencies with specific versions.

---

## Next Steps

1. **After Setup:** Place images in `data/raw/`
2. **After Preprocessing:** Verify processed images look good
3. **After Training:** Check logs and best model metrics
4. **After Evaluation:** Review confusion matrix and error analysis
5. **For Deployment:** See API documentation in `docs/`

---

## Getting Help

- Check [README.md](../README.md) for project overview
- See [docs/INDEX.md](./INDEX.md) for documentation index
- Review specific phase guides in `docs/README_*.md`
- Check logs in `logs/` for detailed training information

---

## Performance Tips

1. **Use GPU:** Training is 10-50x faster with NVIDIA GPU
2. **Batch Size:** Increase to 32 if you have >8GB VRAM
3. **Data Augmentation:** Uncomment in `train.py` for better generalization
4. **Early Stopping:** Prevents overfitting automatically
5. **Learning Rate Scheduling:** Already implemented

---

**Last Updated:** 19 Ianuarie 2026  
**Status:** ✅ Complete - Ready to Use
