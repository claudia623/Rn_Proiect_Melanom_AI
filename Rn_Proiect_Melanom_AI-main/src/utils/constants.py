#!/usr/bin/env python3
"""
Utility Constants and Helper Functions
Used across the melanoma detection pipeline
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_DIR = DATA_DIR / "train"
VALIDATION_DATA_DIR = DATA_DIR / "validation"
TEST_DATA_DIR = DATA_DIR / "test"
GENERATED_DATA_DIR = DATA_DIR / "generated"

# ============================================================================
# IMAGE PROCESSING CONSTANTS
# ============================================================================

TARGET_SIZE = (224, 224)  # EfficientNetB0 input size
IMAGE_CHANNELS = 3        # RGB channels

# Preprocessing parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8
BLUR_DETECTION_THRESHOLD = 100  # Laplacian variance threshold
KERNEL_SIZE = 5  # For various operations
MORPHOLOGICAL_KERNEL_SIZE = 5

# ============================================================================
# MODEL CONSTANTS
# ============================================================================

MODEL_NAME = "melanom_efficientnetb0"
MODEL_ARCHITECTURE = "EfficientNetB0"
TOTAL_PARAMETERS = 4_840_000

# Training configuration
BATCH_SIZE = 16
EPOCHS_PHASE1 = 30  # Fine-tune custom head
EPOCHS_PHASE2 = 50  # Fine-tune backbone
LEARNING_RATE_PHASE1 = 0.001
LEARNING_RATE_PHASE2 = 0.0001
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# ============================================================================
# DATA SPLIT RATIOS
# ============================================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================================
# CLASS LABELS
# ============================================================================

CLASS_NAMES = {
    0: "benign",
    1: "malignant"
}

BENIGN_CLASS = 0
MALIGNANT_CLASS = 1

# ============================================================================
# FILE NAMING PATTERNS
# ============================================================================

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
LOG_FILE_PATTERN = f"{MODEL_NAME}_{{timestamp}}_history.csv"
MODEL_FILE_PATTERN = f"{MODEL_NAME}_{{}}.keras"

# ============================================================================
# PERFORMANCE METRICS (Current Best)
# ============================================================================

BEST_MODEL_INFO = {
    "timestamp": "20260113_150021",
    "epoch": 25,
    "phase": 2,
    "val_auc": 0.8889,
    "val_accuracy": 0.8000,
    "val_loss": 0.4597,
    "train_auc": 0.9287,
    "train_accuracy": 0.8571,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        TRAIN_DATA_DIR, VALIDATION_DATA_DIR, TEST_DATA_DIR,
        GENERATED_DATA_DIR, SRC_DIR, CONFIG_DIR,
        MODELS_DIR, LOGS_DIR, RESULTS_DIR, DOCS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return all(d.exists() for d in directories)


def get_class_path(class_name: str, data_type: str = "train") -> Path:
    """
    Get path to class directory.
    
    Args:
        class_name: "benign" or "malignant"
        data_type: "train", "validation", "test", "processed", or "raw"
    
    Returns:
        Path to class directory
    """
    data_type_map = {
        "train": TRAIN_DATA_DIR,
        "validation": VALIDATION_DATA_DIR,
        "test": TEST_DATA_DIR,
        "processed": PROCESSED_DATA_DIR,
        "raw": RAW_DATA_DIR,
    }
    
    base_dir = data_type_map.get(data_type, TRAIN_DATA_DIR)
    return base_dir / class_name


def get_model_path(model_type: str = "best") -> Path:
    """
    Get path to saved model.
    
    Args:
        model_type: "best" or "last"
    
    Returns:
        Path to model file
    """
    if model_type == "best":
        return MODELS_DIR / f"{MODEL_NAME}_best.keras"
    else:
        return MODELS_DIR / f"{MODEL_NAME}_last.keras"


def get_config_path(config_type: str = "yaml") -> Path:
    """
    Get path to configuration file.
    
    Args:
        config_type: "yaml" or "metadata"
    
    Returns:
        Path to config file
    """
    if config_type == "yaml":
        return CONFIG_DIR / "config.yaml"
    elif config_type == "metadata":
        return CONFIG_DIR / "metadata.csv"
    else:
        return CONFIG_DIR / config_type


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(funcName)s(): %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "melanoma_detection.log"),
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": False,
        }
    },
}

# ============================================================================
# VALIDATION RULES
# ============================================================================

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
MIN_IMAGE_SIZE = (100, 100)
MAX_IMAGE_SIZE = (1024, 1024)
MINIMUM_IMAGES_PER_CLASS = 10

# ============================================================================
if __name__ == "__main__":
    """Test constants configuration"""
    print("🔧 Verifying project structure...")
    
    if ensure_directories():
        print("✅ All directories exist/created successfully")
    else:
        print("❌ Failed to create some directories")
    
    print(f"\n📁 Project Root: {PROJECT_ROOT}")
    print(f"📁 Data Dir: {DATA_DIR}")
    print(f"📁 Models Dir: {MODELS_DIR}")
    print(f"📁 Logs Dir: {LOGS_DIR}")
    
    print(f"\n🎯 Image Target Size: {TARGET_SIZE}")
    print(f"🎯 Batch Size: {BATCH_SIZE}")
    print(f"🎯 Model: {MODEL_ARCHITECTURE} ({TOTAL_PARAMETERS:,} params)")
    
    print(f"\n📊 Best Model Metrics:")
    for key, value in BEST_MODEL_INFO.items():
        print(f"   • {key}: {value}")
