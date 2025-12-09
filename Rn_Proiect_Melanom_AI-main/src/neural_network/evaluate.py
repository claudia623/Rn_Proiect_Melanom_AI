"""
Evaluation Module pentru Melanom AI
====================================
Script pentru evaluarea modelului antrenat
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
import json
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Încarcă configurația din fișierul YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model(model_path: str) -> keras.Model:
    """
    Încarcă modelul salvat
    
    Args:
        model_path: Calea către modelul salvat
    
    Returns:
        Modelul Keras încărcat
    """
    print(f"📂 Încărcare model din: {model_path}")
    model = keras.models.load_model(model_path)
    return model


def create_test_generator(test_dir: str,
                          image_size: Tuple[int, int] = (224, 224),
                          batch_size: int = 32) -> ImageDataGenerator:
    """
    Creează generatorul de date pentru testare
    
    Args:
        test_dir: Directorul cu date de test
        image_size: Dimensiunea imaginilor
        batch_size: Dimensiunea batch-ului
    
    Returns:
        Generatorul de date de test
    """
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return test_generator


def evaluate_model(model: keras.Model,
                   test_generator) -> dict:
    """
    Evaluează modelul pe setul de test
    
    Args:
        model: Modelul Keras
        test_generator: Generatorul de date de test
    
    Returns:
        Dicționar cu metrici de evaluare
    """
    print("\n🔍 Evaluare model pe setul de test...")
    
    # Evaluare cu metrici compilate
    results = model.evaluate(test_generator, verbose=1)
    
    metrics = {}
    for name, value in zip(model.metrics_names, results):
        metrics[name] = float(value)
        print(f"   {name}: {value:.4f}")
    
    return metrics


def get_predictions(model: keras.Model,
                    test_generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Obține predicțiile modelului
    
    Args:
        model: Modelul Keras
        test_generator: Generatorul de date de test
    
    Returns:
        Tuple cu (y_true, y_pred_proba, y_pred)
    """
    print("\n📊 Generare predicții...")
    
    # Predicții probabilități
    y_pred_proba = model.predict(test_generator, verbose=1).flatten()
    
    # Predicții clase
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Etichete reale
    y_true = test_generator.classes
    
    return y_true, y_pred_proba, y_pred


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: list = ['Benign', 'Malign'],
                          output_path: Optional[str] = None) -> None:
    """
    Plotează matricea de confuzie
    
    Args:
        y_true: Etichete reale
        y_pred: Etichete prezise
        class_names: Numele claselor
        output_path: Calea pentru salvare (opțional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confuzie')
    plt.ylabel('Etichetă Reală')
    plt.xlabel('Etichetă Prezisă')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Salvat: {output_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray,
                   y_pred_proba: np.ndarray,
                   output_path: Optional[str] = None) -> float:
    """
    Plotează curba ROC
    
    Args:
        y_true: Etichete reale
        y_pred_proba: Probabilități prezise
        output_path: Calea pentru salvare (opțional)
    
    Returns:
        Valoarea AUC
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Salvat: {output_path}")
    
    plt.show()
    
    return roc_auc


def plot_precision_recall_curve(y_true: np.ndarray,
                                 y_pred_proba: np.ndarray,
                                 output_path: Optional[str] = None) -> None:
    """
    Plotează curba Precision-Recall
    
    Args:
        y_true: Etichete reale
        y_pred_proba: Probabilități prezise
        output_path: Calea pentru salvare (opțional)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Salvat: {output_path}")
    
    plt.show()


def plot_training_history(history_path: str,
                          output_path: Optional[str] = None) -> None:
    """
    Plotează istoricul antrenării
    
    Args:
        history_path: Calea către fișierul JSON cu istoricul
        output_path: Calea pentru salvare (opțional)
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['loss'], label='Train')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['accuracy'], label='Train')
    if 'val_accuracy' in history:
        axes[0, 1].plot(history['val_accuracy'], label='Validation')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    if 'auc' in history:
        axes[1, 0].plot(history['auc'], label='Train')
        if 'val_auc' in history:
            axes[1, 0].plot(history['val_auc'], label='Validation')
        axes[1, 0].set_title('AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Precision & Recall
    if 'precision' in history:
        axes[1, 1].plot(history['precision'], label='Train Precision')
        axes[1, 1].plot(history['recall'], label='Train Recall')
        if 'val_precision' in history:
            axes[1, 1].plot(history['val_precision'], label='Val Precision')
            axes[1, 1].plot(history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Salvat: {output_path}")
    
    plt.show()


def generate_classification_report(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    class_names: list = ['Benign', 'Malign'],
                                    output_path: Optional[str] = None) -> str:
    """
    Generează raportul de clasificare
    
    Args:
        y_true: Etichete reale
        y_pred: Etichete prezise
        class_names: Numele claselor
        output_path: Calea pentru salvare (opțional)
    
    Returns:
        Raportul de clasificare ca string
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    print("\n📋 RAPORT CLASIFICARE")
    print("="*50)
    print(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"✓ Salvat: {output_path}")
    
    return report


def save_evaluation_results(metrics: dict,
                            output_path: str) -> None:
    """
    Salvează rezultatele evaluării
    
    Args:
        metrics: Dicționarul cu metrici
        output_path: Calea pentru salvare
    """
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Rezultate salvate: {output_path}")


def main():
    """Funcția principală de evaluare"""
    
    print("="*60)
    print("📊 MELANOM AI - EVALUARE MODEL")
    print("="*60)
    
    # Încarcă configurația
    try:
        config = load_config("config/config.yaml")
    except FileNotFoundError:
        print("⚠ Fișierul config/config.yaml nu a fost găsit!")
        config = {}
    
    # Parametri
    data_config = config.get('data', {})
    image_config = config.get('image', {})
    
    test_dir = data_config.get('test_path', 'data/test')
    image_size = (image_config.get('height', 224), image_config.get('width', 224))
    model_path = "models/melanom_efficientnetb0_best.keras"
    results_dir = "results"
    
    # Verifică existența modelului
    if not os.path.exists(model_path):
        print(f"\n❌ Eroare: Modelul nu a fost găsit la: {model_path}")
        print("   Rulează mai întâi src/neural_network/train.py")
        return
    
    # Verifică existența datelor de test
    if not os.path.exists(test_dir):
        print(f"\n❌ Eroare: Directorul de test nu există: {test_dir}")
        return
    
    # Crează directorul de rezultate
    os.makedirs(results_dir, exist_ok=True)
    
    # Încarcă modelul
    model = load_model(model_path)
    
    # Creează generatorul de test
    test_generator = create_test_generator(test_dir, image_size)
    
    # Evaluare
    metrics = evaluate_model(model, test_generator)
    
    # Obține predicții
    y_true, y_pred_proba, y_pred = get_predictions(model, test_generator)
    
    # Plotează rezultatele
    print("\n📈 Generare grafice...")
    
    # Matrice de confuzie
    plot_confusion_matrix(
        y_true, y_pred,
        output_path=os.path.join(results_dir, "confusion_matrix.png")
    )
    
    # Curba ROC
    roc_auc = plot_roc_curve(
        y_true, y_pred_proba,
        output_path=os.path.join(results_dir, "roc_curve.png")
    )
    metrics['roc_auc'] = roc_auc
    
    # Curba Precision-Recall
    plot_precision_recall_curve(
        y_true, y_pred_proba,
        output_path=os.path.join(results_dir, "precision_recall_curve.png")
    )
    
    # Raport clasificare
    generate_classification_report(
        y_true, y_pred,
        output_path=os.path.join(results_dir, "classification_report.txt")
    )
    
    # Salvare rezultate
    save_evaluation_results(
        metrics,
        os.path.join(results_dir, "evaluation_metrics.json")
    )
    
    # Rezumat final
    print("\n" + "="*60)
    print("✅ EVALUARE COMPLETĂ!")
    print("="*60)
    print(f"\n📊 Rezultate finale:")
    print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"   Precision: {metrics.get('precision', 0):.4f}")
    print(f"   Recall: {metrics.get('recall', 0):.4f}")
    print(f"   AUC-ROC: {roc_auc:.4f}")
    print(f"\n📁 Rezultate salvate în: {results_dir}/")


if __name__ == "__main__":
    main()
