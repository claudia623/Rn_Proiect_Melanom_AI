"""
Script minimal pentru generarea visualizarilor Etapa 5
Genereaza ploturi din history JSON files
"""

import os
import json
import sys

print("=" * 80)
print("GENERARE VISUALIZARI ETAPA 5")
print("=" * 80)

# Check if matplotlib available
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("EROARE: Matplotlib nu este instalat")
    print("Instaleaza cu: pip install matplotlib numpy")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("EROARE: NumPy nu este instalat")
    sys.exit(1)

CONFIG = {
    'history_phase1': 'results/melanom_efficientnetb0_phase1_history.json',
    'history_phase2': 'results/melanom_efficientnetb0_phase2_history.json',
    'docs_dir': 'docs/',
}

os.makedirs(CONFIG['docs_dir'], exist_ok=True)

# ============================================================================
# LOAD AND PLOT PHASE 1
# ============================================================================
print("\n[1/2] Generare Phase 1 plots...")
try:
    with open(CONFIG['history_phase1'], 'r') as f:
        history_phase1 = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    epochs = range(1, len(history_phase1['loss']) + 1)
    axes[0].plot(epochs, history_phase1['loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history_phase1['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (Binary Crossentropy)', fontsize=12)
    axes[0].set_title('Phase 1 - Loss Curves (Transfer Learning)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history_phase1['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history_phase1['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Phase 1 - Accuracy Curves', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["docs_dir"]}/phase1_loss_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Phase 1 plots: {CONFIG['docs_dir']}/phase1_loss_accuracy.png")
    
except Exception as e:
    print(f"✗ Eroare Phase 1: {e}")

# ============================================================================
# LOAD AND PLOT PHASE 2
# ============================================================================
print("[2/2] Generare Phase 2 plots...")
try:
    with open(CONFIG['history_phase2'], 'r') as f:
        history_phase2 = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    epochs = range(1, len(history_phase2['loss']) + 1)
    axes[0].plot(epochs, history_phase2['loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history_phase2['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (Binary Crossentropy)', fontsize=12)
    axes[0].set_title('Phase 2 - Loss Curves (Fine-tuning)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history_phase2['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history_phase2['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Phase 2 - Accuracy Curves', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["docs_dir"]}/phase2_loss_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Phase 2 plots: {CONFIG['docs_dir']}/phase2_loss_accuracy.png")
    
except Exception as e:
    print(f"✗ Eroare Phase 2: {e}")

# ============================================================================
# CREATE AUC PLOT
# ============================================================================
print("[3/3] Generare AUC curves...")
try:
    if history_phase1 and history_phase2:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        epochs_p1 = range(1, len(history_phase1['auc']) + 1)
        epochs_p2 = range(len(history_phase1['auc']) + 1, len(history_phase1['auc']) + len(history_phase2['auc']) + 1)
        
        ax.plot(epochs_p1, history_phase1['auc'], 'b-', label='Phase 1 Train AUC', linewidth=2)
        ax.plot(epochs_p1, history_phase1['val_auc'], 'b--', label='Phase 1 Val AUC', linewidth=2)
        ax.plot(epochs_p2, history_phase2['auc'], 'g-', label='Phase 2 Train AUC', linewidth=2)
        ax.plot(epochs_p2, history_phase2['val_auc'], 'g--', label='Phase 2 Val AUC', linewidth=2)
        
        # Add vertical line between phases
        ax.axvline(x=len(history_phase1['auc']) + 0.5, color='red', linestyle=':', alpha=0.5, label='Phase Switch')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('AUC (ROC)', fontsize=12)
        ax.set_title('Training Progress - AUC Curves (Both Phases)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.6, 1.0])
        
        plt.tight_layout()
        plt.savefig(f'{CONFIG["docs_dir"]}/auc_curves_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ AUC curves: {CONFIG['docs_dir']}/auc_curves_combined.png")
        
except Exception as e:
    print(f"✗ Eroare AUC: {e}")

print("\n" + "=" * 80)
print("✓ VISUALIZARI GENERATE")
print("=" * 80)
print("\nFisiere generate:")
print(f"  1. {CONFIG['docs_dir']}/phase1_loss_accuracy.png")
print(f"  2. {CONFIG['docs_dir']}/phase2_loss_accuracy.png")
print(f"  3. {CONFIG['docs_dir']}/auc_curves_combined.png")
