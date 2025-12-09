"""
Script pentru generate diagram State Machine PNG
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Colors
color_idle = '#E8F4F8'
color_process = '#D4E6F1'
color_error = '#F5B7B1'
color_decision = '#F9E79F'
color_output = '#ABEBC6'

def add_box(ax, x, y, width, height, text, color, style='round'):
    """Add a state box"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle=f"round,pad=0.1", 
                         edgecolor='black', facecolor=color, 
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold', wrap=True)

def add_arrow(ax, x1, y1, x2, y2, label=''):
    """Add arrow between states"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='black')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.2, mid_y, label, fontsize=8, style='italic')

# Draw states
add_box(ax, 5, 13, 1, 0.6, 'START', '#FFFFFF')
add_arrow(ax, 5, 12.7, 5, 12.3)

add_box(ax, 5, 11.8, 1.5, 0.8, 'IDLE\n(awaiting input)', color_idle)
add_arrow(ax, 5, 11.4, 5, 10.7)

add_box(ax, 5, 10.2, 2, 0.8, 'VALIDATE_INPUT\n(format, size, blur)', color_process)
add_arrow(ax, 6.2, 10.2, 7.5, 10.2, '[Invalid] ✗')
add_arrow(ax, 5, 9.8, 5, 9.3)

add_box(ax, 8, 10.2, 1.8, 0.8, 'ERROR_INVALID\n→ IDLE', color_error)

add_box(ax, 5, 8.8, 2, 0.8, 'PREPROCESS\n(resize 224x224)', color_process)
add_arrow(ax, 5, 8.4, 5, 7.9)

add_box(ax, 5, 7.4, 2.2, 0.8, 'FEATURE_EXTRACTION\n(EfficientNetB0)', color_process)
add_arrow(ax, 5, 7, 5, 6.5)

add_box(ax, 5, 6, 2.2, 0.8, 'LOAD_REFERENCE_DB\n(30 images cached)', color_process)
add_arrow(ax, 5, 5.6, 5, 5.1)

add_box(ax, 5, 4.6, 2.4, 0.8, 'COMPUTE_SIMILARITY\n(vs 30 references)', color_process)
add_arrow(ax, 3.8, 4.6, 2.5, 4.6, '')
add_arrow(ax, 6.2, 4.6, 7.5, 4.6, '')

add_box(ax, 2, 4.6, 1.5, 0.8, 'CLASSIFY_BENIGN\n✅ Green', color_decision)
add_box(ax, 8, 4.6, 1.5, 0.8, 'CLASSIFY_MALIGNANT\n⚠️ Red', color_decision)

add_arrow(ax, 2, 4.2, 2, 3.7)
add_arrow(ax, 8, 4.2, 8, 3.7)
add_arrow(ax, 2, 3.4, 5, 3.4)
add_arrow(ax, 8, 3.4, 5, 3.4)

add_box(ax, 5, 2.9, 1.8, 0.8, 'DISPLAY_RESULT\n(UI output)', color_output)
add_arrow(ax, 5, 2.5, 5, 2.0)

add_box(ax, 5, 1.5, 1.5, 0.8, 'LOG_RESULT\n(CSV save)', color_output)
add_arrow(ax, 5, 1.1, 5, 0.6)

add_box(ax, 5, 0.1, 1.5, 0.8, 'RETURN_TO_IDLE', color_idle)

# Add error path on right
add_arrow(ax, 9.2, 8.8, 9.2, 7.4, '[Error]')
add_box(ax, 9.2, 6.9, 1.5, 0.8, 'ERROR_HANDLER\n→ Cleanup', color_error)
add_arrow(ax, 9.2, 6.5, 9.2, 0.1)

# Add title
ax.text(5, 13.8, 'MELANOM AI - STATE MACHINE ARCHITECTURE (Etapa 4)', 
        fontsize=14, weight='bold', ha='center')

# Add legend
legend_y = 12
ax.text(0.3, legend_y, 'Legend:', fontsize=10, weight='bold')
add_box(ax, 1, legend_y - 0.6, 0.8, 0.4, 'Input', color_idle)
add_box(ax, 2.2, legend_y - 0.6, 0.8, 0.4, 'Process', color_process)
add_box(ax, 3.4, legend_y - 0.6, 0.8, 0.4, 'Decision', color_decision)
add_box(ax, 4.6, legend_y - 0.6, 0.8, 0.4, 'Output', color_output)
add_box(ax, 5.8, legend_y - 0.6, 0.8, 0.4, 'Error', color_error)

# Save
plt.tight_layout()
plt.savefig('docs/state_machine_diagram.png', dpi=300, bbox_inches='tight')
print("✅ State machine diagram saved to docs/state_machine_diagram.png")
plt.close()

print("Done!")
