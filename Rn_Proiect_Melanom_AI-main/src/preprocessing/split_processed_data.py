
import os
import shutil
import random
from pathlib import Path
import yaml

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def split_processed_data():
    config = load_config()
    # Folosim folderele procesate recent (unde imaginile sunt fixate ca dimensiune și claritate)
    base_processed = Path(config['data']['processed_path']) # "data/processed"
    target_bases = {
        "train": Path(config['data']['train_path']),        # "data/train"
        "validation": Path(config['data']['validation_path']), # "data/validation"
        "test": Path(config['data']['test_path'])           # "data/test"
    }

    classes = ["benign", "malignant"]
    ratios = {
        "train": config['split']['train_ratio'], 
        "validation": config['split']['validation_ratio'], 
        "test": config['split']['test_ratio']
    }

    print(f"--- Începere repartizare date din {base_processed} ---")

    for cls in classes:
        src_dir = base_processed / cls
        if not src_dir.exists():
            print(f"⚠️ Folderul {cls} nu a fost găsit în {base_processed}.")
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
            
            # Curățăm fișierele vechi (fără .gitkeep)
            for file in os.listdir(dest_dir):
                if file != ".gitkeep":
                    file_path = dest_dir / file
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            
            # Copiem noile imagini preprocesate
            for img_name in split_images:
                shutil.copy2(src_dir / img_name, dest_dir / img_name)
            
            print(f"✅ {split_name}/{cls}: {len(split_images)} imagini")

    print("\n--- Repartizare finalizată! ---")

if __name__ == "__main__":
    split_processed_data()
