"""
Model Definition pentru Melanom AI
===================================
Definirea arhitecturii rețelei neuronale pentru clasificarea melanomului
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    VGG16, 
    ResNet50, 
    EfficientNetB0, 
    EfficientNetB3,
    MobileNetV2
)
from typing import Tuple, Optional
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Încarcă configurația din fișierul YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_base_model(architecture: str = "EfficientNetB0",
                      input_shape: Tuple[int, int, int] = (224, 224, 3),
                      weights: str = "imagenet") -> Model:
    """
    Creează modelul de bază pentru transfer learning
    
    Args:
        architecture: Arhitectura de bază (VGG16, ResNet50, EfficientNetB0, etc.)
        input_shape: Dimensiunea imaginii de intrare
        weights: Greutățile pre-antrenate ('imagenet' sau None)
    
    Returns:
        Modelul de bază Keras
    """
    architectures = {
        "VGG16": VGG16,
        "ResNet50": ResNet50,
        "EfficientNetB0": EfficientNetB0,
        "EfficientNetB3": EfficientNetB3,
        "MobileNetV2": MobileNetV2
    }
    
    if architecture not in architectures:
        raise ValueError(f"Arhitectură necunoscută: {architecture}. "
                        f"Opțiuni disponibile: {list(architectures.keys())}")
    
    base = architectures[architecture](
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    return base


def create_melanom_classifier(input_shape: Tuple[int, int, int] = (224, 224, 3),
                               architecture: str = "EfficientNetB0",
                               num_classes: int = 1,
                               dropout_rate: float = 0.5,
                               freeze_base: bool = True) -> Model:
    """
    Creează modelul complet pentru clasificarea melanomului
    
    Args:
        input_shape: Dimensiunea imaginii de intrare (H, W, C)
        architecture: Arhitectura de bază pentru transfer learning
        num_classes: Numărul de clase (1 pentru clasificare binară cu sigmoid)
        dropout_rate: Rata de dropout pentru regularizare
        freeze_base: Dacă să înghețe layer-urile modelului de bază
    
    Returns:
        Modelul Keras compilat
    """
    # Modelul de bază pre-antrenat
    base_model = create_base_model(architecture, input_shape)
    
    # Înghețare layer-uri bază (opțional)
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    
    # Construire model complet
    inputs = keras.Input(shape=input_shape)
    
    # Preprocesare specifică arhitecturii
    if architecture.startswith("EfficientNet"):
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    elif architecture == "ResNet50":
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
    elif architecture == "VGG16":
        x = tf.keras.applications.vgg16.preprocess_input(inputs)
    elif architecture == "MobileNetV2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    else:
        x = inputs
    
    # Extragere caracteristici
    x = base_model(x, training=False)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully Connected Layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    
    # Output layer
    if num_classes == 1:
        # Clasificare binară
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        # Clasificare multi-clasă
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name=f"Melanom_{architecture}")
    
    return model


def create_custom_cnn(input_shape: Tuple[int, int, int] = (224, 224, 3),
                      num_classes: int = 1,
                      dropout_rate: float = 0.5) -> Model:
    """
    Creează un CNN personalizat de la zero (fără transfer learning)
    
    Args:
        input_shape: Dimensiunea imaginii de intrare
        num_classes: Numărul de clase
        dropout_rate: Rata de dropout
    
    Returns:
        Modelul Keras
    """
    model = keras.Sequential([
        # Input
        keras.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Clasificator
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate / 2),
        
        # Output
        layers.Dense(1 if num_classes == 1 else num_classes,
                    activation='sigmoid' if num_classes == 1 else 'softmax')
    ], name="Melanom_CustomCNN")
    
    return model


def compile_model(model: Model,
                  learning_rate: float = 0.001,
                  optimizer: str = "adam") -> Model:
    """
    Compilează modelul cu optimizer și funcție de pierdere
    
    Args:
        model: Modelul Keras de compilat
        learning_rate: Rata de învățare
        optimizer: Tipul de optimizer
    
    Returns:
        Modelul compilat
    """
    # Alegere optimizer
    if optimizer.lower() == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == "sgd":
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer.lower() == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compilare
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def unfreeze_model(model: Model, 
                   num_layers_to_unfreeze: int = 20,
                   learning_rate: float = 0.0001) -> Model:
    """
    Dezgheață ultimele layer-uri pentru fine-tuning
    
    Args:
        model: Modelul Keras
        num_layers_to_unfreeze: Numărul de layer-uri de dezghețat
        learning_rate: Rata de învățare (mai mică pentru fine-tuning)
    
    Returns:
        Modelul recompilat
    """
    # Dezgheață ultimele N layer-uri
    for layer in model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
    
    # Recompilare cu learning rate mai mic
    model = compile_model(model, learning_rate=learning_rate)
    
    return model


def get_model_summary(model: Model) -> str:
    """Returnează sumarul modelului ca string"""
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    return '\n'.join(summary_list)


def build_model_from_config(config_path: str = "config/config.yaml") -> Model:
    """
    Construiește modelul din fișierul de configurare
    
    Args:
        config_path: Calea către fișierul de configurare
    
    Returns:
        Modelul compilat
    """
    config = load_config(config_path)
    
    image_config = config.get('image', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    input_shape = (
        image_config.get('height', 224),
        image_config.get('width', 224),
        image_config.get('channels', 3)
    )
    
    model = create_melanom_classifier(
        input_shape=input_shape,
        architecture=model_config.get('architecture', 'EfficientNetB0'),
        dropout_rate=model_config.get('dropout_rate', 0.5)
    )
    
    model = compile_model(
        model,
        learning_rate=training_config.get('learning_rate', 0.001)
    )
    
    return model


if __name__ == "__main__":
    print("="*60)
    print("🧠 MELANOM AI - TEST ARHITECTURĂ MODEL")
    print("="*60)
    
    # Test creare model
    model = create_melanom_classifier(
        input_shape=(224, 224, 3),
        architecture="EfficientNetB0",
        dropout_rate=0.5
    )
    
    model = compile_model(model)
    
    print("\n📋 Sumar Model:")
    model.summary()
    
    print("\n✓ Model creat cu succes!")
    print(f"  Parametri totali: {model.count_params():,}")
