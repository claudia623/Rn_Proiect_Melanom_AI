"""
Training Module pentru Melanom AI
==================================
Script pentru antrenarea modelului de clasificare a melanomului
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)
from datetime import datetime
from typing import Tuple, Optional, Dict
import yaml
import json

# Adaugă directorul părinte la path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_network.model import (
    create_melanom_classifier,
    compile_model,
    unfreeze_model,
    build_model_from_config
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Încarcă configurația din fișierul YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_data_generators(train_dir: str,
                           val_dir: str,
                           image_size: Tuple[int, int] = (224, 224),
                           batch_size: int = 32,
                           augmentation: bool = True) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    """
    Creează generatoare de date pentru antrenare și validare
    
    Args:
        train_dir: Directorul cu date de antrenare
        val_dir: Directorul cu date de validare
        image_size: Dimensiunea imaginilor
        batch_size: Dimensiunea batch-ului
        augmentation: Dacă să aplice augmentare pe date de antrenare
    
    Returns:
        Tuple cu (train_generator, val_generator)
    """
    # Generator pentru antrenare (cu augmentare)
    if augmentation:
        train_datagen = ImageDataGenerator(
            # rescale=1./255,  # EfficientNet include scalare
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            shear_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='reflect'
        )
    else:
        train_datagen = ImageDataGenerator() # rescale=1./255
    
    # Generator pentru validare (fără augmentare)
    val_datagen = ImageDataGenerator() # rescale=1./255
    
    # Creare generatoare
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator


def get_class_weights(train_generator) -> Dict[int, float]:
    """
    Calculează greutățile claselor pentru dataset dezechilibrat
    
    Args:
        train_generator: Generatorul de date de antrenare
    
    Returns:
        Dicționar cu greutățile claselor
    """
    # Numără instanțele per clasă
    class_counts = np.bincount(train_generator.classes)
    total = sum(class_counts)
    
    # Calculează greutățile inverse
    weights = {}
    for i, count in enumerate(class_counts):
        weights[i] = total / (len(class_counts) * count)
    
    print(f"📊 Distribuția claselor: {dict(enumerate(class_counts))}")
    print(f"⚖️ Greutăți clase: {weights}")
    
    return weights


def create_callbacks(model_name: str,
                     checkpoint_dir: str = "models",
                     log_dir: str = "logs",
                     patience: int = 10) -> list:
    """
    Creează callback-urile pentru antrenare
    
    Args:
        model_name: Numele modelului (pentru salvare)
        checkpoint_dir: Directorul pentru checkpoint-uri
        log_dir: Directorul pentru log-uri
        patience: Răbdarea pentru early stopping
    
    Returns:
        Lista de callback-uri
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Creare directoare
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        # Salvare cel mai bun model
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}_best.keras"),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Salvare ultimul model
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}_last.keras"),
            save_best_only=False,
            verbose=0
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reducere learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard
        TensorBoard(
            log_dir=os.path.join(log_dir, f"{model_name}_{timestamp}"),
            histogram_freq=1,
            write_graph=True
        ),
        
        # CSV Logger
        CSVLogger(
            os.path.join(log_dir, f"{model_name}_{timestamp}_history.csv"),
            separator=',',
            append=False
        )
    ]
    
    return callbacks


def train_model(model,
                train_generator,
                val_generator,
                epochs: int = 50,
                callbacks: list = None,
                class_weights: dict = None) -> keras.callbacks.History:
    """
    Antrenează modelul
    
    Args:
        model: Modelul Keras
        train_generator: Generatorul de date de antrenare
        val_generator: Generatorul de date de validare
        epochs: Numărul de epoci
        callbacks: Lista de callback-uri
        class_weights: Greutățile claselor
    
    Returns:
        Istoricul antrenării
    """
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return history


def fine_tune_model(model,
                    train_generator,
                    val_generator,
                    epochs: int = 20,
                    num_layers_to_unfreeze: int = 20,
                    learning_rate: float = 1e-5,
                    callbacks: list = None,
                    class_weights: dict = None) -> keras.callbacks.History:
    """
    Fine-tuning al modelului (dezgheață layer-uri)
    
    Args:
        model: Modelul Keras
        train_generator: Generatorul de date de antrenare
        val_generator: Generatorul de date de validare
        epochs: Numărul de epoci pentru fine-tuning
        num_layers_to_unfreeze: Numărul de layer-uri de dezghețat
        learning_rate: Rata de învățare pentru fine-tuning
        callbacks: Lista de callback-uri
        class_weights: Greutățile claselor
    
    Returns:
        Istoricul antrenării
    """
    print("\n🔧 Începere Fine-Tuning...")
    print(f"   Dezghețare ultimele {num_layers_to_unfreeze} layer-uri")
    print(f"   Learning rate: {learning_rate}")
    
    model = unfreeze_model(model, num_layers_to_unfreeze, learning_rate)
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return history


def save_training_results(history: keras.callbacks.History,
                          output_dir: str,
                          model_name: str) -> None:
    """
    Salvează rezultatele antrenării
    
    Args:
        history: Istoricul antrenării
        output_dir: Directorul de output
        model_name: Numele modelului
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvare istoric ca JSON
    history_dict = {key: [float(v) for v in values] 
                   for key, values in history.history.items()}
    
    with open(os.path.join(output_dir, f"{model_name}_history.json"), 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"✓ Rezultate salvate în: {output_dir}")


def main():
    """Funcția principală de antrenare"""
    
    print("="*60)
    print("🧠 MELANOM AI - ANTRENARE MODEL")
    print("="*60)
    
    # Încarcă configurația
    try:
        config = load_config("config/config.yaml")
    except FileNotFoundError:
        print("⚠ Fișierul config/config.yaml nu a fost găsit!")
        print("  Se folosesc valorile implicite.")
        config = {}
    
    # Extrage parametrii
    data_config = config.get('data', {})
    image_config = config.get('image', {})
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    
    # Parametri
    train_dir = data_config.get('train_path', 'data/train')
    val_dir = data_config.get('validation_path', 'data/validation')
    image_size = (image_config.get('height', 224), image_config.get('width', 224))
    batch_size = training_config.get('batch_size', 32)
    epochs = training_config.get('epochs', 50)
    patience = training_config.get('early_stopping_patience', 10)
    architecture = model_config.get('architecture', 'EfficientNetB0')
    
    print(f"\n📋 Configurație:")
    print(f"   Arhitectură: {architecture}")
    print(f"   Dimensiune imagine: {image_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epoci maxime: {epochs}")
    print(f"   Patience: {patience}")
    
    # Verifică existența directoarelor
    if not os.path.exists(train_dir):
        print(f"\n❌ Eroare: Directorul {train_dir} nu există!")
        print("   Rulează mai întâi src/data_acquisition/download_dataset.py")
        return
    
    # Creează generatoarele de date
    print("\n📁 Încărcare date...")
    train_generator, val_generator = create_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        image_size=image_size,
        batch_size=batch_size,
        augmentation=True
    )
    
    # Calculează greutățile claselor
    class_weights = get_class_weights(train_generator)
    
    # Creează modelul
    print(f"\n🏗️ Creare model {architecture}...")
    model = create_melanom_classifier(
        input_shape=(image_size[0], image_size[1], 3),
        architecture=architecture,
        dropout_rate=model_config.get('dropout_rate', 0.5),
        freeze_base=True
    )
    
    model = compile_model(
        model,
        learning_rate=training_config.get('learning_rate', 0.001)
    )
    
    print("\n📊 Sumar model:")
    model.summary()
    
    # Creează callback-uri
    model_name = f"melanom_{architecture.lower()}"
    callbacks = create_callbacks(model_name, patience=patience)
    
    # Faza 1: Antrenare cu baza înghețată
    print("\n" + "="*60)
    print("🚀 FAZA 1: ANTRENARE (bază înghețată)")
    print("="*60)
    
    history1 = train_model(
        model=model,
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=epochs // 2,
        callbacks=callbacks,
        class_weights=class_weights
    )
    
    # Faza 2: Fine-tuning
    print("\n" + "="*60)
    print("🔧 FAZA 2: FINE-TUNING")
    print("="*60)
    
    history2 = fine_tune_model(
        model=model,
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=epochs // 2,
        num_layers_to_unfreeze=30,
        learning_rate=1e-5,
        callbacks=callbacks,
        class_weights=class_weights
    )
    
    # Salvare rezultate
    print("\n💾 Salvare rezultate...")
    save_training_results(history1, "results", f"{model_name}_phase1")
    save_training_results(history2, "results", f"{model_name}_phase2")
    
    # Rezumat final
    print("\n" + "="*60)
    print("✅ ANTRENARE COMPLETĂ!")
    print("="*60)
    print(f"\n📁 Model salvat în: models/{model_name}_best.keras")
    print(f"📊 Log-uri TensorBoard în: logs/")
    print(f"📈 Rezultate în: results/")
    print("\nPentru vizualizare TensorBoard, rulează:")
    print("   tensorboard --logdir logs")


if __name__ == "__main__":
    main()
