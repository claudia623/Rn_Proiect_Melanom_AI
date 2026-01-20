"""
Modul 2: Neural Network - Similarity-Based Model for Melanoma Classification
===============================================================================
Definirea arhitecturii rețelei neuronale pentru clasificarea melanomului
bazată pe similaritate imagini.

Arhitectura:
    INPUT (224x224x3 RGB image)
        ↓
    [EfficientNetB0 pretrained on ImageNet]  ← Transfer Learning
        ↓
    Global Average Pooling
        ↓
    Dense(256, ReLU) + Dropout(0.5)  ← Feature Extraction
        ↓
    OUTPUT: Feature Vector (256D) ← Pentru similarity computation

Similarity Matching:
    cosine_similarity(feature_test, feature_reference) ∈ [0, 1]
    Clasă: max_similarity(benign_refs) vs max_similarity(malignant_refs)
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, optimizers
from keras.applications import EfficientNetB0
from typing import Tuple, Optional
import numpy as np
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SIMILARITY-BASED FEATURE EXTRACTION MODEL
# ============================================================================

def create_similarity_model(input_shape: Tuple[int, int, int] = (224, 224, 3),
                          feature_dim: int = 256,
                          dropout_rate: float = 0.5,
                          pretrained: bool = True) -> Model:
    """
    Creează modelul Keras pentru extragere de features și similarity matching.
    
    Arhitectura:
        Input Layer (224x224x3)
            ↓
        EfficientNetB0 (pretrained ImageNet, frozen layers)
            ↓
        Global Average Pooling → 1280D vector
            ↓
        Dense(256, ReLU) + Dropout(0.5)
            ↓
        Output: Feature Vector (256D)
    
    Args:
        input_shape: Dimensiunea imaginii de intrare (H, W, C)
        feature_dim: Dimensiunea vector de features output (256D)
        dropout_rate: Rata dropout pentru regularizare (0.5)
        pretrained: Folosestem ImageNet pretraining (default: True)
    
    Returns:
        Modelul Keras compilat și gata de inferență
    
    Note:
        - Model NU e antrenat cu imagini medicale (doar feature extraction)
        - Weights sunt din ImageNet pre-training (transfer learning)
        - Similarity se computează cu cosine distance post-inference
    """
    
    # ========================================================================
    # 1. BASE MODEL - EfficientNetB0 pretrained
    # ========================================================================
    
    # Incarca EfficientNetB0 cu ImageNet weights
    # include_top=False → doar convolutional layers, fara classification head
    base_model = EfficientNetB0(
        weights='imagenet' if pretrained else None,
        include_top=False,
        input_shape=input_shape,
        pooling=None
    )
    
    # Congeleaza layerurile bazei (transfer learning)
    # NU vrem sa modificam ImageNet features in Etapa 4
    base_model.trainable = False
    
    logger.info(f"Base model layers: {len(base_model.layers)}")
    logger.info(f"Base model trainable: {base_model.trainable}")
    
    # ========================================================================
    # 2. BUILD FULL MODEL
    # ========================================================================
    
    inputs = keras.Input(shape=input_shape, name='image_input')
    
    # Preprocesare specific pentru EfficientNet
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    
    # Pass through base model
    x = base_model(x, training=False)  # training=False pentru frozen layers
    
    # Global Average Pooling: (7x7x1280) → (1280,)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    logger.info(f"After GlobalAveragePooling: shape = (1280,)")
    
    # Dense layer for feature extraction
    x = layers.Dense(feature_dim, activation='relu', 
                     kernel_initializer='he_normal',
                     name='feature_extraction')(x)
    
    # Dropout pentru regularizare
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    
    # Output: Feature vector (NU clasificare, dar features pentru similarity)
    # L2 normalizare opțională pentru cosine similarity
    features = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1),
                            name='feature_output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=features, name='SimilarityModel')
    
    logger.info(f"Model created successfully!")
    logger.info(f"Output shape: {model.output_shape}")
    
    return model


def compile_model(model: Model) -> Model:
    """
    Compilează modelul (pentru referință, NU se antrenează în Etapa 4).
    
    Note:
        - Loss function: None (inference only în Etapa 4)
        - Optimizer: None (NU se antrenează)
        - Metrics: None (NU evaluări)
    """
    
    # În Etapa 4, modelul e doar pentru feature extraction
    # Compilarea e doar pentru a salva/reîncărca modelul
    model.compile(
        optimizer=None,
        loss=None,
        metrics=None
    )
    
    logger.info("Model compiled (inference-only mode for Etapa 4)")
    return model


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_model(model_path: str) -> Model:
    """
    Încarcă modelul salvat dintr-un fișier.
    
    Args:
        model_path: Cale la fișier .h5 sau SavedModel
    
    Returns:
        Modelul Keras încărcat
    """
    try:
        if model_path.endswith('.h5'):
            model = keras.models.load_model(model_path)
        else:
            model = keras.models.load_model(model_path)
        
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def save_model(model: Model, output_path: str) -> str:
    """
    Salvează modelul în format .h5.
    
    Args:
        model: Modelul Keras de salvat
        output_path: Cale de ieșire
    
    Returns:
        Calea fișierului salvat
    """
    try:
        # Create parent directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save(output_path)
        logger.info(f"Model saved to {output_path}")
        
        return output_path
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def extract_features(model: Model, 
                     image: np.ndarray) -> np.ndarray:
    """
    Extrage features din imagine folosind modelul.
    
    Args:
        model: Modelul feature extraction
        image: Imagine preprocesat (224x224x3, normalized [0-1])
    
    Returns:
        Feature vector (256D) normalized L2
    """
    
    # Add batch dimension
    image_batch = np.expand_dims(image, axis=0)
    
    # Predict features
    features = model.predict(image_batch, verbose=0)
    
    # Return first (și unic) element din batch
    return features[0]


def compute_similarity(features_test: np.ndarray,
                      features_ref: np.ndarray,
                      metric: str = 'cosine') -> float:
    """
    Calculează similaritate între doi vectori de features.
    
    Args:
        features_test: Feature vector din imagine test (256D)
        features_ref: Feature vector din imagine referință (256D)
        metric: Tipul de distanță ('cosine', 'euclidean', etc.)
    
    Returns:
        Similarity score [0, 1] (1 = identic)
    """
    
    if metric == 'cosine':
        # Cosine similarity = 1 - cosine_distance
        # Vectorii sunt deja L2-normalized
        similarity = np.dot(features_test, features_ref)
    
    elif metric == 'euclidean':
        distance = np.linalg.norm(features_test - features_ref)
        similarity = 1 / (1 + distance)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return float(similarity)


def classify_melanoma(similarity_benign: list,
                     similarity_malignant: list,
                     threshold: float = 0.5) -> Tuple[str, float, dict]:
    """
    Clasifică imagine ca benign/malignant pe bază de similarități.
    
    Strategie:
        score_benign = mean(similarity_benign)
        score_malignant = mean(similarity_malignant)
        
        if score_benign > score_malignant:
            class = "BENIGN"
        else:
            class = "MALIGNANT"
        
        confidence = abs(score_benign - score_malignant)
    
    Args:
        similarity_benign: List de similarități cu imagini benigne
        similarity_malignant: List de similarități cu imagini maligne
        threshold: Prag de decizie (default: 0.5, nu se folosește în V1)
    
    Returns:
        (classification, confidence, scores_dict)
    """
    
    # Calculate mean similarities
    mean_benign = np.mean(similarity_benign) if similarity_benign else 0
    mean_malignant = np.mean(similarity_malignant) if similarity_malignant else 0
    
    # Classify
    if mean_benign > mean_malignant:
        classification = "BENIGN"
        confidence = mean_benign - mean_malignant
    else:
        classification = "MALIGNANT"
        confidence = mean_malignant - mean_benign
    
    # Additional statistics
    scores = {
        'benign_mean': float(mean_benign),
        'benign_std': float(np.std(similarity_benign)) if similarity_benign else 0,
        'benign_min': float(np.min(similarity_benign)) if similarity_benign else 0,
        'benign_max': float(np.max(similarity_benign)) if similarity_benign else 0,
        'malignant_mean': float(mean_malignant),
        'malignant_std': float(np.std(similarity_malignant)) if similarity_malignant else 0,
        'malignant_min': float(np.min(similarity_malignant)) if similarity_malignant else 0,
        'malignant_max': float(np.max(similarity_malignant)) if similarity_malignant else 0,
        'confidence': float(confidence),
    }
    
    return classification, confidence, scores


def get_model_summary_json(model: Model) -> dict:
    """
    Exportă arhitectura modelului în format JSON.
    
    Util pentru documentație.
    """
    
    summary = {
        'model_name': model.name,
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'total_params': model.count_params(),
        'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
        'non_trainable_params': sum([tf.size(w).numpy() for w in model.non_trainable_weights]),
        'layers': [
            {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': str(layer.output_shape),
                'params': layer.count_params(),
            }
            for layer in model.layers[:10]  # First 10 layers
        ]
    }
    
    return summary


# ============================================================================
# MAIN - TESTING
# ============================================================================

def main():
    """
    Test executie: creează și salvează modelul.
    """
    
    logger.info("="*70)
    logger.info("MODUL 2: NEURAL NETWORK - SIMILARITY MODEL")
    logger.info("="*70)
    
    # Create model
    logger.info("\n1. Creating similarity model...")
    model = create_similarity_model(
        input_shape=(224, 224, 3),
        feature_dim=256,
        dropout_rate=0.5,
        pretrained=True
    )
    
    # Print model summary
    logger.info("\n2. Model Summary:")
    model.summary()
    
    # Compile model
    logger.info("\n3. Compiling model...")
    model = compile_model(model)
    
    # Save model
    logger.info("\n4. Saving model...")
    model_path = 'models/similarity_model_untrained.h5'
    save_model(model, model_path)
    
    # Test load
    logger.info("\n5. Testing model load...")
    loaded_model = load_model(model_path)
    logger.info(f"✅ Model loaded successfully from {model_path}")
    
    # Test feature extraction (dummy image)
    logger.info("\n6. Testing feature extraction...")
    dummy_image = np.random.randn(224, 224, 3).astype(np.float32)
    dummy_image = np.clip(dummy_image / 255.0, 0, 1)  # Normalize to [0, 1]
    
    features = extract_features(loaded_model, dummy_image)
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Features norm (should be ~1.0 after L2): {np.linalg.norm(features):.4f}")
    
    # Test similarity computation
    logger.info("\n7. Testing similarity computation...")
    features_ref_benign = extract_features(loaded_model, dummy_image)
    similarity = compute_similarity(features, features_ref_benign)
    logger.info(f"Similarity (same image): {similarity:.4f}")
    
    # Test classification
    logger.info("\n8. Testing classification...")
    dummy_similarities_benign = [0.75, 0.70, 0.68]
    dummy_similarities_malignant = [0.30, 0.35, 0.25]
    
    classification, confidence, scores = classify_melanoma(
        dummy_similarities_benign,
        dummy_similarities_malignant
    )
    
    logger.info(f"Classification: {classification}")
    logger.info(f"Confidence: {confidence:.4f}")
    logger.info(f"Scores: {json.dumps(scores, indent=2)}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ MODUL 2 TEST COMPLETED SUCCESSFULLY")
    logger.info("="*70)


if __name__ == '__main__':
    main()
