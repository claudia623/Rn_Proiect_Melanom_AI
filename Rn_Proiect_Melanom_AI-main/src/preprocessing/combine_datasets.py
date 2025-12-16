"""
Combine Datasets Module
=======================
Combines original ISIC data (already in data/train, etc.) with generated synthetic data.
Splits generated data and adds it to the respective folders.
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def combine_datasets():
    base_dir = Path("data")
    generated_dir = base_dir / "generated" / "original"
    
    if not generated_dir.exists():
        logger.error(f"Generated data directory not found: {generated_dir}")
        return

    classes = ["benign", "malignant"]
    
    for class_name in classes:
        src_dir = generated_dir / class_name
        if not src_dir.exists():
            logger.warning(f"Source directory not found: {src_dir}")
            continue
            
        images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.png"))
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        n_test = n_total - n_train - n_val
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]
        
        logger.info(f"Class {class_name}: Total {n_total} generated images.")
        logger.info(f"  Adding {len(train_imgs)} to train")
        logger.info(f"  Adding {len(val_imgs)} to validation")
        logger.info(f"  Adding {len(test_imgs)} to test")
        
        # Copy to destinations
        for img in train_imgs:
            shutil.copy2(img, base_dir / "train" / class_name / img.name)
            
        for img in val_imgs:
            shutil.copy2(img, base_dir / "validation" / class_name / img.name)
            
        for img in test_imgs:
            shutil.copy2(img, base_dir / "test" / class_name / img.name)

    logger.info("Datasets combined successfully!")

if __name__ == "__main__":
    combine_datasets()
