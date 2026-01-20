"""
Script pentru generarea tuturor visualizarilor si analizelor pentru Etapa 5
==========================================================================
Genereaza:
1. Loss/Val_Loss plots pentru fazele 1 si 2
2. AUC/Val_AUC plots
3. ROC curve plot
4. Confusion matrix detaliat
5. Imagini misclassified (false positives/negatives)
6. Raport detaliat al erorilor
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# Try to import required packages, handle missing dependencies gracefully
try:
    import numpy as np
except ImportError:
    print("✗ NumPy not found. Install with: pip install numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("✗ Matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("⚠ Seaborn not found. Continuing without seaborn...")
    sns = None

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    print("✗ TensorFlow not found. Install with: pip install tensorflow")
    sys.exit(1)

try:
    from sklearn.metrics import (
        confusion_matrix,
        roc_curve,
        auc,
        precision_recall_curve,
        classification_report
    )
except ImportError:
    print("✗ Scikit-learn not found. Install with: pip install scikit-learn")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("⚠ OpenCV not found. Error analysis images will be skipped...")
    cv2 = None

# Configuration
CONFIG = {
    'model_path': 'models/melanom_efficientnetb0_best.keras',
    'history_phase1': 'results/melanom_efficientnetb0_phase1_history.json',
    'history_phase2': 'results/melanom_efficientnetb0_phase2_history.json',
    'test_dir': 'data/test/',
    'docs_dir': 'docs/',
    'error_analysis_dir': 'docs/error_analysis/',
}

print("=" * 80)
print("INCEPE GENERAREA VISUALIZARILOR ETAPA 5")
print("=" * 80)

# ============================================================================
# 1. LOAD HISTORY
# ============================================================================
print("\n[1/6] Incarcare history files...")
try:
    with open(CONFIG['history_phase1'], 'r') as f:
        history_phase1 = json.load(f)
    print(f"✓ Phase 1 history incarcat ({len(history_phase1['accuracy'])} epoci)")
except Exception as e:
    print(f"✗ Eroare la incarcare phase1: {e}")
    history_phase1 = None

try:
    with open(CONFIG['history_phase2'], 'r') as f:
        history_phase2 = json.load(f)
    print(f"✓ Phase 2 history incarcat ({len(history_phase2['accuracy'])} epoci)")
except Exception as e:
    print(f"✗ Eroare la incarcare phase2: {e}")
    history_phase2 = None

# ============================================================================
# 2. CREATE DOCS DIRECTORIES
# ============================================================================
print("\n[2/6] Creare directoare...")
os.makedirs(CONFIG['docs_dir'], exist_ok=True)
os.makedirs(CONFIG['error_analysis_dir'], exist_ok=True)
print(f"✓ Directoare create: {CONFIG['docs_dir']}")

# ============================================================================
# 3. PLOT LOSS CURVES (PHASE 1 + 2)
# ============================================================================
print("\n[3/6] Generare Loss/Accuracy plots...")

if history_phase1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    epochs_phase1 = range(1, len(history_phase1['loss']) + 1)
    axes[0].plot(epochs_phase1, history_phase1['loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs_phase1, history_phase1['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (Binary Crossentropy)', fontsize=12)
    axes[0].set_title('Phase 1 - Loss Curves (Transfer Learning)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs_phase1, history_phase1['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs_phase1, history_phase1['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Phase 1 - Accuracy Curves', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["docs_dir"]}/phase1_loss_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"✓ Phase 1 plots salvate: {CONFIG['docs_dir']}/phase1_loss_accuracy.png")
    plt.close()

if history_phase2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    epochs_phase2 = range(1, len(history_phase2['loss']) + 1)
    axes[0].plot(epochs_phase2, history_phase2['loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs_phase2, history_phase2['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (Binary Crossentropy)', fontsize=12)
    axes[0].set_title('Phase 2 - Loss Curves (Fine-tuning)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs_phase2, history_phase2['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs_phase2, history_phase2['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Phase 2 - Accuracy Curves', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["docs_dir"]}/phase2_loss_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"✓ Phase 2 plots salvate: {CONFIG['docs_dir']}/phase2_loss_accuracy.png")
    plt.close()

# ============================================================================
# 4. LOAD MODEL AND GENERATE PREDICTIONS
# ============================================================================
print("\n[4/6] Incarcare model si generare predictii...")
try:
    model = tf.keras.models.load_model(CONFIG['model_path'])
    print(f"✓ Model incarcat din: {CONFIG['model_path']}")
    
    # Create test generator
    test_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )
    
    test_flow = test_generator.flow_from_directory(
        CONFIG['test_dir'],
        target_size=(224, 224),
        batch_size=32,
        shuffle=False,
        classes={'benign': 0, 'malignant': 1}
    )
    
    print(f"✓ Test generator creat ({test_flow.samples} imagini)")
    
    # Generate predictions
    predictions = model.predict(test_flow, verbose=0)
    predictions_binary = (predictions > 0.5).astype(int).flatten()
    y_true = test_flow.classes
    y_proba = predictions.flatten()
    
    print(f"✓ Predictii generate")
    print(f"  - Accuracy: {np.mean(predictions_binary == y_true):.2%}")
    
except Exception as e:
    print(f"✗ Eroare la incarcare model: {e}")
    predictions_binary = None
    y_true = None
    y_proba = None

# ============================================================================
# 5. GENERATE ROC CURVE
# ============================================================================
print("\n[5/6] Generare ROC curve...")
if y_true is not None and y_proba is not None:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificator Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Melanom Classifier (Test Set)', fontsize=13, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["docs_dir"]}/roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve salvat: {CONFIG['docs_dir']}/roc_curve.png")
    plt.close()

# ============================================================================
# 6. GENERATE CONFUSION MATRIX
# ============================================================================
print("\n[6/6] Generare Confusion Matrix...")
if y_true is not None and predictions_binary is not None:
    cm = confusion_matrix(y_true, predictions_binary)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                cbar_kws={'label': 'Count'},
                ax=ax, annot_kws={'size': 14})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - Test Set', fontsize=13, fontweight='bold')
    
    # Add metrics text
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    metrics_text = f'Sensitivity (Recall): {sensitivity:.2%}\nSpecificity: {specificity:.2%}\nPrecision: {precision:.2%}'
    ax.text(1.5, -0.3, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["docs_dir"]}/confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix salvat: {CONFIG['docs_dir']}/confusion_matrix_detailed.png")
    plt.close()

# ============================================================================
# 7. ANALYZE MISCLASSIFIED IMAGES
# ============================================================================
print("\n[ANALIZA ERORI] Identificare imagini misclassified...")
if y_true is not None and predictions_binary is not None:
    # Get misclassified indices
    misclassified_idx = np.where(predictions_binary != y_true)[0]
    correct_idx = np.where(predictions_binary == y_true)[0]
    
    print(f"✓ Imagini gresite: {len(misclassified_idx)} din {len(y_true)} ({len(misclassified_idx)/len(y_true):.1%})")
    print(f"  - False Positives (Benign -> Malignant): {np.sum((y_true == 0) & (predictions_binary == 1))}")
    print(f"  - False Negatives (Malignant -> Benign): {np.sum((y_true == 1) & (predictions_binary == 0))}")
    
    # Get file paths
    test_benign_dir = os.path.join(CONFIG['test_dir'], 'benign')
    test_malignant_dir = os.path.join(CONFIG['test_dir'], 'malignant')
    
    benign_files = sorted(os.listdir(test_benign_dir)) if os.path.exists(test_benign_dir) else []
    malignant_files = sorted(os.listdir(test_malignant_dir)) if os.path.exists(test_malignant_dir) else []
    
    # Create file mapping
    all_files = []
    for f in benign_files:
        all_files.append((os.path.join(test_benign_dir, f), 0, f))
    for f in malignant_files:
        all_files.append((os.path.join(test_malignant_dir, f), 1, f))
    
    # Sort to match generator order
    all_files.sort(key=lambda x: (x[1], x[2]))  # Sort by class then filename
    
    # Analyze top misclassified
    error_analysis = []
    for idx in misclassified_idx[:10]:  # Top 10 errors
        if idx < len(all_files):
            filepath, true_class, filename = all_files[idx]
            pred_class = predictions_binary[idx]
            confidence = y_proba[idx]
            
            error_type = 'False Positive' if true_class == 0 and pred_class == 1 else 'False Negative'
            error_analysis.append({
                'filename': filename,
                'filepath': filepath,
                'true_class': 'Benign' if true_class == 0 else 'Malignant',
                'pred_class': 'Benign' if pred_class == 0 else 'Malignant',
                'confidence': confidence,
                'error_type': error_type
            })
    
    # Save error analysis images
    print("\n[ERORI] Salvare imagini gresite...")
    os.makedirs(CONFIG['error_analysis_dir'], exist_ok=True)
    
    for i, error in enumerate(error_analysis[:5]):  # Save top 5
        try:
            img = cv2.imread(error['filepath'])
            if img is not None:
                # Create figure with annotations
                fig, ax = plt.subplots(figsize=(8, 6))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                
                title = f"{error['error_type']}\nTrue: {error['true_class']}, Predicted: {error['pred_class']}\nConfidence: {error['confidence']:.2%}"
                ax.set_title(title, fontsize=11, fontweight='bold', color='red')
                ax.axis('off')
                
                output_file = f"{CONFIG['error_analysis_dir']}/error_{i+1}_{error['filename']}"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ {error['filename']} ({error['error_type']})")
        except Exception as e:
            print(f"  ✗ Eroare salvare {error['filename']}: {e}")

# ============================================================================
# 8. GENERATE ERROR REPORT
# ============================================================================
print("\n[RAPORT] Generare raport detaliat erori...")

error_report = f"""
# RAPORT ANALIZA ERORI - ETAPA 5

## Rezumat Evaluare Test Set

- **Total imagini test:** {len(y_true)}
- **Imagini corecte:** {np.sum(predictions_binary == y_true)} ({np.mean(predictions_binary == y_true):.1%})
- **Imagini gresite:** {np.sum(predictions_binary != y_true)} ({np.mean(predictions_binary != y_true):.1%})

## Descompunere Erori

### False Positives (Benign clasificate ca Malignant)
- Count: {np.sum((y_true == 0) & (predictions_binary == 1))}
- Rata: {np.sum((y_true == 0) & (predictions_binary == 1)) / np.sum(y_true == 0):.1%} din imagini Benign
- **Implicatie:** Recomandare falsa de tratament / over-alarm

### False Negatives (Malignant clasificate ca Benign)
- Count: {np.sum((y_true == 1) & (predictions_binary == 0))}
- Rata: {np.sum((y_true == 1) & (predictions_binary == 0)) / np.sum(y_true == 1):.1%} din imagini Malignant
- **Implicatie:** CRITICA - cazuri maligne netestate (miss medical)

## Analiză Detaliat Confuzii

```
Confusion Matrix:
             Predicted Benign    Predicted Malignant
True Benign         {cm[0,0]:2d}                    {cm[0,1]:2d}
True Malignant      {cm[1,0]:2d}                    {cm[1,1]:2d}
```

## Metrici Derivate

- **Sensitivity (Recall pentru Malignant):** {np.sum((y_true == 1) & (predictions_binary == 1)) / np.sum(y_true == 1):.1%}
  - Abilitate de a detecta cazuri maligne reale
  
- **Specificity:** {np.sum((y_true == 0) & (predictions_binary == 0)) / np.sum(y_true == 0):.1%}
  - Abilitate de a identifica corect cazurile benigne
  
- **Precision (Positive Predictive Value):** {np.sum((y_true == 1) & (predictions_binary == 1)) / np.sum(predictions_binary == 1):.1%}
  - Proporție predicții malignant care sunt corecte

## Cauze Probabili ale Erorilor

1. **Variații Iluminare:** Imagini cu iluminare neomogenă, reflexii
   - Soluție: Data augmentation cu jitter iluminare
   
2. **Artefacte Imagistică:** Glint (reflexii), markeri de linie
   - Soluție: Filtrare preprocessing specifică
   
3. **Similitudine Vizuală:** Leziuni benigne care imita melanom (ex: nevus)
   - Soluție: Colectare imagini specifice pentru cazuri atipice
   
4. **Distribuție Dataset Hibrid:** Mix imagini sintetice + reale
   - Soluție: Validare separata pe imagini reale vs sintetice

## Recomandări Îmbunătățire

### Prioritate 1 (Critic) - Reducere False Negatives
1. Ajustare prag (threshold) de 0.5 → 0.35
   - Favorizează recall pentru Malignant
   - Cost: creștere False Positives acceptabilă în context medical
   
2. Reantrenare cu `class_weights`
   - Weight malignant mai mare (ex: 3:1)
   
3. Augmentări specifice pentru zone atipice
   - Histograma egalizare pentru contrast variabil
   - Crop aleator din diferite regiuni

### Prioritate 2 - Date Suplimentare
1. Colectare ≥50 imagini noi pentru cazuri atipice
2. Validare separata pe ImageNet-Only pretrained vs Altă bază

### Prioritate 3 - Optimizare Model
1. Explorare alte arhitecturi (ResNet, DenseNet)
2. Ensemble de modele cu thresholds diferite
3. Feature matching + similarity-based (fallback)

## Imagini Problematice

Vezi dosarul `docs/error_analysis/` pentru:
- Top 5 False Positive examples
- Top 5 False Negative examples
- Annotări cu confidence scores și predicții

---

**Generat:** 12.01.2026  
**Script:** generate_etapa5_visualizations.py
"""

with open(f'{CONFIG["error_analysis_dir"]}/error_analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(error_report)

print(f"✓ Raport salvat: {CONFIG['error_analysis_dir']}/error_analysis_report.md")

print("\n" + "=" * 80)
print("✓ GENERARE VISUALIZARI COMPLETATA")
print("=" * 80)
print("\nFisiere generate:")
print(f"  1. {CONFIG['docs_dir']}/phase1_loss_accuracy.png")
print(f"  2. {CONFIG['docs_dir']}/phase2_loss_accuracy.png")
print(f"  3. {CONFIG['docs_dir']}/roc_curve.png")
print(f"  4. {CONFIG['docs_dir']}/confusion_matrix_detailed.png")
print(f"  5. {CONFIG['error_analysis_dir']}/error_*.png (top 5 erori)")
print(f"  6. {CONFIG['error_analysis_dir']}/error_analysis_report.md")
