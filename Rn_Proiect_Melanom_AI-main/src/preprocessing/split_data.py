
import os
import shutil
import random
from pathlib import Path

def split_data():
    base_raw = Path("data/raw")
    target_bases = {
        "train": Path("data/train"),
        "validation": Path("data/validation"),
        "test": Path("data/test")
    }

    classes = ["benign", "malignant"]
    ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}

    for cls in classes:
        src_dir = base_raw / cls
        if not src_dir.exists():
            print(f"Skipping {cls}, directory not found.")
            continue
            
        images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["validation"])
        
        splits = {
            "train": images[:n_train],
            "validation": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:]
        }
        
        for split_name, split_images in splits.items():
            dest_dir = target_bases[split_name] / cls
            os.makedirs(dest_dir, exist_ok=True)
            
            # Clear existing files in the split/class folder to ensure fresh start
            for existing in os.listdir(dest_dir):
                if existing != ".gitkeep":
                    os.remove(dest_dir / existing)
            
            for img_name in split_images:
                shutil.copy2(src_dir / img_name, dest_dir / img_name)
            
            print(f"✅ {split_name}/{cls}: {len(split_images)} images")

if __name__ == "__main__":
    split_data()
