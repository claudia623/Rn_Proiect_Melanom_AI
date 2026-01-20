#!/usr/bin/env python3
"""Generează vizualizări finale în `docs/results/`:
- copy `results/confusion_matrix.png` -> `docs/results/confusion_matrix_optimized.png`
- generează `learning_curves_final.png` din history JSON
- generează `metrics_evolution.png` din valori Etapa4/5/6 (din fișier sau hardcodate)
- generează `example_predictions.png` grid cu 9 exemple annotate folosind `logs/predictions.csv` și ground-truth
"""

import os
import sys
import shutil
import json
import math
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def copy_confusion(results_dir, out_dir):
    src = os.path.join(results_dir, 'confusion_matrix.png')
    dst = os.path.join(out_dir, 'confusion_matrix_optimized.png')
    if os.path.exists(src):
        shutil.copy(src, dst)
        print('Copied', dst)
        return True
    print('WARN: confusion matrix not found at', src)
    return False


def plot_learning_curves_from_jsons(phase1, phase2, out_dir):
    # load histories (lists of metrics)
    h1 = {}
    h2 = {}
    if os.path.exists(phase1):
        with open(phase1) as f:
            h1 = json.load(f)
    if os.path.exists(phase2):
        with open(phase2) as f:
            h2 = json.load(f)

    # concatenate accuracy/loss over phases
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    if 'accuracy' in h1:
        acc += h1.get('accuracy', [])
        val_acc += h1.get('val_accuracy', [])
        loss += h1.get('loss', [])
        val_loss += h1.get('val_loss', [])
    if 'accuracy' in h2:
        acc += h2.get('accuracy', [])
        val_acc += h2.get('val_accuracy', [])
        loss += h2.get('loss', [])
        val_loss += h2.get('val_loss', [])

    if not acc:
        print('WARN: No history data found for learning curves.')
        return False

    epochs = list(range(1, len(acc)+1))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label='loss')
    if val_loss:
        # val may be shorter; pad if needed
        plt.plot(epochs[:len(val_loss)], val_loss, label='val_loss')
    plt.xlabel('Epoch')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, acc, label='accuracy')
    if val_acc:
        plt.plot(epochs[:len(val_acc)], val_acc, label='val_accuracy')
    plt.xlabel('Epoch')
    plt.title('Accuracy')
    plt.legend()

    out = os.path.join(out_dir, 'learning_curves_final.png')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print('Saved', out)
    return True


def plot_metrics_evolution(out_dir):
    # Use values from the document: Etapa4 ~0.20, Etapa5 0.72, Etapa6 0.78
    stages = ['Etapa 4', 'Etapa 5', 'Etapa 6']
    accuracy = [0.20, 0.72, 0.78]
    f1 = [0.15, 0.68, 0.75]

    x = range(len(stages))
    plt.figure(figsize=(6,4))
    plt.plot(x, accuracy, marker='o', label='Accuracy')
    plt.plot(x, f1, marker='o', label='F1-score')
    plt.xticks(x, stages)
    plt.ylim(0,1)
    plt.title('Evoluție metrici: Etapa 4 → 5 → 6')
    plt.legend()
    out = os.path.join(out_dir, 'metrics_evolution.png')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print('Saved', out)
    return True


def make_example_grid(pred_csv, gt_csv, test_dir, out_dir, n=9):
    if not os.path.exists(pred_csv) or not os.path.exists(gt_csv):
        print('WARN: predictions or ground-truth CSV missing.')
        return False
    preds = pd.read_csv(pred_csv)
    # normalize filename column
    preds['filename'] = preds['filename'].astype(str)
    gt = {}
    with open(gt_csv) as f:
        for line in f:
            if not line.strip():
                continue
            key, val = line.strip().split(',')
            gt[key + '.jpg'] = int(float(val))

    # Build prediction map
    pred_map = {str(r['filename']): r for _, r in preds.iterrows()}

    imgs = []
    # Walk test_dir and collect images; prefer those with predictions
    for root, _, files in os.walk(test_dir):
        for f in files:
            if not f.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            full = os.path.join(root, f)
            pred_row = pred_map.get(f)
            prob = pred_row.get('probability_malignant') if pred_row is not None else None
            pred_label = pred_row.get('classification') if pred_row is not None else 'N/A'
            true_label = gt.get(f) if f in gt else None
            imgs.append((full, true_label, pred_label, prob))
            if len(imgs) >= n:
                break
        if len(imgs) >= n:
            break

    if not imgs:
        print('WARN: No images found under test directory.')
        return False

    # Create grid
    cols = int(math.sqrt(n))
    rows = math.ceil(n / cols)
    thumb_w = 224
    thumb_h = 224
    pad = 8
    grid_w = cols * thumb_w + (cols+1)*pad
    grid_h = rows * thumb_h + (rows+1)*pad + 40
    grid = Image.new('RGB', (grid_w, grid_h), (255,255,255))

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, (path, true_label, pred_label, prob) in enumerate(imgs):
        im = Image.open(path).convert('RGB')
        im = im.resize((thumb_w, thumb_h))
        x = pad + (i % cols) * (thumb_w + pad)
        y = pad + (i // cols) * (thumb_h + pad)
        grid.paste(im, (x, y))
        # annotate
        draw = ImageDraw.Draw(grid)
        pred_text = f'P:{pred_label} {prob:.2f}' if pd.notna(prob) else f'P:{pred_label}'
        true_text = 'T:Melanom' if true_label == 1 else 'T:Benign'
        txt = f'{true_text} | {pred_text}'
        color = (0,128,0) if ((str(pred_label).upper().startswith('M') and true_label==1) or (str(pred_label).upper().startswith('B') and true_label==0)) else (200,20,20)
        draw.rectangle([x, y+thumb_h, x+thumb_w, y+thumb_h+30], fill=(255,255,255))
        draw.text((x+4, y+thumb_h+4), txt, fill=color, font=font)

    out = os.path.join(out_dir, 'example_predictions.png')
    grid.save(out)
    print('Saved', out)
    return True


def main():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(repo, 'results')
    docs_results = os.path.join(repo, 'docs', 'results')
    ensure_dir(docs_results)

    any_ok = False
    any_ok |= copy_confusion(results_dir, docs_results)
    any_ok |= plot_learning_curves_from_jsons(os.path.join(results_dir, 'melanom_efficientnetb0_phase1_history.json'),
                                              os.path.join(results_dir, 'melanom_efficientnetb0_phase2_history.json'),
                                              docs_results)
    any_ok |= plot_metrics_evolution(docs_results)
    any_ok |= make_example_grid(os.path.join(repo, 'logs', 'predictions.csv'),
                                os.path.join(repo, 'data', 'test', 'ISBI2016_ISIC_Part3_Test_GroundTruth.csv'),
                                os.path.join(repo, 'data', 'test'),
                                docs_results, n=9)

    if not any_ok:
        print('No visual artifacts generated; check warnings above.')
        sys.exit(1)
    print('All done.')


if __name__ == '__main__':
    main()
