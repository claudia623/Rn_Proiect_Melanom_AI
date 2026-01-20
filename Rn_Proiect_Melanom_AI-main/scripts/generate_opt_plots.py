#!/usr/bin/env python3
"""Generare grafice optimizare:
- `docs/optimization/accuracy_comparison.png`
- `docs/optimization/f1_comparison.png`
- `docs/optimization/learning_curves_best.png`

Script robust: funcționează chiar dacă fișiere lipsesc și raportează ce a generat.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_experiments(csv_path, out_dir):
    if not os.path.exists(csv_path):
        print(f"WARN: '{csv_path}' not found. Skipping experiments plots.")
        return False
    df = pd.read_csv(csv_path)
    # Try to find sensible columns
    # Common names: 'exp','accuracy','f1','f1_score','accuracy_val'
    cols = {c.lower(): c for c in df.columns}

    # Accuracy plot
    acc_col = cols.get('accuracy') or cols.get('test_accuracy') or cols.get('val_accuracy')
    f1_col = cols.get('f1') or cols.get('f1_score') or cols.get('f1_score_macro')

    x = df.index if 'exp' not in cols else df[cols['exp']]

    if acc_col is not None:
        plt.figure(figsize=(6,4))
        plt.plot(x, df[acc_col], marker='o')
        plt.title('Accuracy per experiment')
        plt.xlabel('Experiment')
        plt.ylabel('Accuracy')
        plt.grid(True)
        out = os.path.join(out_dir, 'accuracy_comparison.png')
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print(f"Saved {out}")
    else:
        print("WARN: Accuracy column not found in experiments CSV.")

    if f1_col is not None:
        plt.figure(figsize=(6,4))
        plt.plot(x, df[f1_col], marker='o', color='C1')
        plt.title('F1-score per experiment')
        plt.xlabel('Experiment')
        plt.ylabel('F1-score')
        plt.grid(True)
        out = os.path.join(out_dir, 'f1_comparison.png')
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print(f"Saved {out}")
    else:
        print("WARN: F1 column not found in experiments CSV.")

    return True


def plot_learning_curves(history_csv, out_dir):
    if not os.path.exists(history_csv):
        print(f"WARN: '{history_csv}' not found. Skipping learning curves.")
        return False
    df = pd.read_csv(history_csv)

    # Look for common columns
    cols = {c.lower(): c for c in df.columns}
    epoch = df.index if 'epoch' not in cols else df[cols['epoch']]

    loss_col = cols.get('loss')
    val_loss_col = cols.get('val_loss')
    acc_col = cols.get('accuracy') or cols.get('acc')
    val_acc_col = cols.get('val_accuracy') or cols.get('val_acc')

    plt.figure(figsize=(10,4))
    if loss_col or val_loss_col:
        plt.subplot(1,2,1)
        if loss_col:
            plt.plot(epoch, df[loss_col], label='loss')
        if val_loss_col:
            plt.plot(epoch, df[val_loss_col], label='val_loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)

    if acc_col or val_acc_col:
        plt.subplot(1,2,2)
        if acc_col:
            plt.plot(epoch, df[acc_col], label='accuracy')
        if val_acc_col:
            plt.plot(epoch, df[val_acc_col], label='val_accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)

    out = os.path.join(out_dir, 'learning_curves_best.png')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")
    return True


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    experiments_csv = os.path.join(repo_root, 'results', 'optimization_experiments.csv')
    history_csv = os.path.join(repo_root, 'results', 'training_history.csv')
    out_dir = os.path.join(repo_root, 'docs', 'optimization')
    ensure_dir(out_dir)

    any_ok = False
    try:
        any_ok |= plot_experiments(experiments_csv, out_dir)
        any_ok |= plot_learning_curves(history_csv, out_dir)
    except Exception as e:
        print('ERROR during plotting:', e)
        sys.exit(2)

    if not any_ok:
        print('No plots were generated. Check input CSV files.')
        sys.exit(1)

    print('Done.')


if __name__ == '__main__':
    main()
