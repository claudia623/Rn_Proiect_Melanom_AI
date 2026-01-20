"""
Generate optimization comparison plots for Etapa 6
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from optimization experiments
experiments = ['Baseline', 'LR Tuning', 'Augmentation', 'Dropout']
accuracy = [0.68, 0.69, 0.70, 0.695]
f1_scores = [0.70, 0.71, 0.72, 0.71]

# Colors
colors = ['#e74c3c', '#f39c12', '#27ae60', '#f39c12']

# 1. Accuracy Comparison
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(experiments, accuracy, color=colors, alpha=0.8, edgecolor='black')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, accuracy)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1%}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_xlabel('Experiment', fontsize=14, fontweight='bold')
ax.set_title('Accuracy Comparison - Optimization Experiments', fontsize=16, fontweight='bold')
ax.set_ylim(0.65, 0.73)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0.70, color='green', linestyle='--', linewidth=2, label='Target: 70%')
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. F1-Score Comparison
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(experiments, f1_scores, color=colors, alpha=0.8, edgecolor='black')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, f1_scores)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
ax.set_xlabel('Experiment', fontsize=14, fontweight='bold')
ax.set_title('F1-Score Comparison - Optimization Experiments', fontsize=16, fontweight='bold')
ax.set_ylim(0.68, 0.74)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0.72, color='green', linestyle='--', linewidth=2, label='Best: 0.72')
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('f1_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated:")
print("  - accuracy_comparison.png")
print("  - f1_comparison.png")
