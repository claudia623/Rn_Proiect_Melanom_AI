"""
Download Dataset Module pentru Melanom AI
==========================================
Script pentru descărcarea și organizarea dataset-ului de imagini dermatoscopice
"""

import os
import shutil
import requests
import zipfile
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
import random
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Încarcă configurația din fișierul YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def download_file(url: str, destination: str, chunk_size: int = 8192) -> None:
    """
    Descarcă un fișier de la URL
    
    Args:
        url: URL-ul fișierului
        destination: Calea de destinație
        chunk_size: Dimensiunea chunk-ului pentru descărcare
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Descărcare") as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extrage un arhivă ZIP
    
    Args:
        zip_path: Calea către arhiva ZIP
        extract_to: Directorul de extragere
    """
    print(f"Extragere arhivă: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extragere completă!")


def create_directory_structure(base_path: str = ".") -> None:
    """
    Creează structura de directoare pentru dataset
    
    Args:
        base_path: Calea de bază a proiectului
    """
    directories = [
        "data/raw/benign",
        "data/raw/malignant",
        "data/processed/benign",
        "data/processed/malignant",
        "data/train/benign",
        "data/train/malignant",
        "data/validation/benign",
        "data/validation/malignant",
        "data/test/benign",
        "data/test/malignant",
        "models",
        "logs",
        "results"
    ]
    
    for dir_path in directories:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"✓ Creat: {full_path}")
    
    print("\nStructura de directoare creată cu succes!")


def split_dataset(source_dir: str,
                  train_dir: str,
                  val_dir: str,
                  test_dir: str,
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  seed: int = 42) -> dict:
    """
    Împarte dataset-ul în train/validation/test
    
    Args:
        source_dir: Directorul sursă cu imaginile
        train_dir: Directorul pentru date de antrenare
        val_dir: Directorul pentru date de validare
        test_dir: Directorul pentru date de test
        train_ratio: Procentul pentru antrenare
        val_ratio: Procentul pentru validare
        test_ratio: Procentul pentru test
        seed: Seed pentru reproducibilitate
    
    Returns:
        Dicționar cu statisticile împărțirii
    """
    random.seed(seed)
    stats = {}
    
    for class_name in ['benign', 'malignant']:
        class_source = os.path.join(source_dir, class_name)
        
        if not os.path.exists(class_source):
            print(f"⚠ Directorul {class_source} nu există!")
            continue
        
        # Obține lista de fișiere
        files = [f for f in os.listdir(class_source) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not files:
            print(f"⚠ Nu s-au găsit imagini în {class_source}")
            continue
        
        # Amestecă fișierele
        random.shuffle(files)
        
        # Calculează indicii de împărțire
        n_files = len(files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        # Copiază fișierele
        for file_list, dest_base in [(train_files, train_dir),
                                      (val_files, val_dir),
                                      (test_files, test_dir)]:
            dest_dir = os.path.join(dest_base, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            for f in tqdm(file_list, desc=f"Copiere {class_name} -> {os.path.basename(dest_base)}"):
                src = os.path.join(class_source, f)
                dst = os.path.join(dest_dir, f)
                shutil.copy2(src, dst)
        
        stats[class_name] = {
            'total': n_files,
            'train': len(train_files),
            'validation': len(val_files),
            'test': len(test_files)
        }
    
    return stats


def print_dataset_stats(stats: dict) -> None:
    """
    Afișează statisticile dataset-ului
    
    Args:
        stats: Dicționarul cu statistici
    """
    print("\n" + "="*50)
    print("📊 STATISTICI DATASET")
    print("="*50)
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for class_name, class_stats in stats.items():
        print(f"\n📁 {class_name.upper()}")
        print(f"   Total: {class_stats['total']}")
        print(f"   Train: {class_stats['train']} ({class_stats['train']/class_stats['total']*100:.1f}%)")
        print(f"   Validation: {class_stats['validation']} ({class_stats['validation']/class_stats['total']*100:.1f}%)")
        print(f"   Test: {class_stats['test']} ({class_stats['test']/class_stats['total']*100:.1f}%)")
        
        total_train += class_stats['train']
        total_val += class_stats['validation']
        total_test += class_stats['test']
    
    total = total_train + total_val + total_test
    print(f"\n📈 TOTAL")
    print(f"   Total imagini: {total}")
    print(f"   Train: {total_train}")
    print(f"   Validation: {total_val}")
    print(f"   Test: {total_test}")


def verify_dataset_integrity(data_dir: str) -> bool:
    """
    Verifică integritatea dataset-ului
    
    Args:
        data_dir: Directorul de date
    
    Returns:
        True dacă dataset-ul este valid
    """
    print("\n🔍 Verificare integritate dataset...")
    
    required_dirs = [
        "train/benign", "train/malignant",
        "validation/benign", "validation/malignant",
        "test/benign", "test/malignant"
    ]
    
    all_valid = True
    
    for dir_path in required_dirs:
        full_path = os.path.join(data_dir, dir_path)
        if os.path.exists(full_path):
            n_files = len([f for f in os.listdir(full_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            status = "✓" if n_files > 0 else "⚠"
            print(f"   {status} {dir_path}: {n_files} imagini")
            if n_files == 0:
                all_valid = False
        else:
            print(f"   ✗ {dir_path}: LIPSEȘTE")
            all_valid = False
    
    return all_valid


def main():
    """Funcția principală pentru descărcarea și organizarea dataset-ului"""
    
    print("="*60)
    print("🔬 MELANOM AI - DESCĂRCARE ȘI ORGANIZARE DATASET")
    print("="*60)
    
    # Creează structura de directoare
    print("\n📁 Creare structură directoare...")
    create_directory_structure()
    
    print("\n" + "="*60)
    print("📝 INSTRUCȚIUNI PENTRU DESCĂRCAREA DATASET-ULUI")
    print("="*60)
    print("""
Pentru a descărca dataset-ul, urmează acești pași:

1. OPȚIUNEA 1 - ISIC Archive (Recomandat):
   - Accesează: https://www.isic-archive.com/
   - Creează un cont gratuit
   - Descarcă imaginile din secțiunea "Gallery"
   - Organizează-le în data/raw/benign și data/raw/malignant

2. OPȚIUNEA 2 - Kaggle:
   - Accesează: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
   - Descarcă dataset-ul HAM10000
   - Extrage și organizează imaginile

3. OPȚIUNEA 3 - Kaggle Competition:
   - Accesează: https://www.kaggle.com/c/siim-isic-melanoma-classification
   - Descarcă dataset-ul de competiție

După descărcare, plasează imaginile în:
   - data/raw/benign/     (pentru leziuni benigne)
   - data/raw/malignant/  (pentru melanom/maligne)

Apoi rulează din nou acest script pentru a împărți datele.
""")
    
    # Verifică dacă există date în directorul raw
    raw_benign = "data/raw/benign"
    raw_malignant = "data/raw/malignant"
    
    has_benign = os.path.exists(raw_benign) and len(os.listdir(raw_benign)) > 0
    has_malignant = os.path.exists(raw_malignant) and len(os.listdir(raw_malignant)) > 0
    
    if has_benign and has_malignant:
        print("\n✓ S-au detectat imagini în directorul raw!")
        print("  Se va realiza împărțirea în train/validation/test...")
        
        # Încarcă configurația
        try:
            config = load_config()
            split_config = config.get('split', {})
        except:
            split_config = {}
        
        train_ratio = split_config.get('train_ratio', 0.7)
        val_ratio = split_config.get('validation_ratio', 0.15)
        test_ratio = split_config.get('test_ratio', 0.15)
        
        stats = split_dataset(
            source_dir="data/raw",
            train_dir="data/train",
            val_dir="data/validation",
            test_dir="data/test",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        print_dataset_stats(stats)
        
        # Verifică integritatea
        verify_dataset_integrity("data")
        
    else:
        print("\n⚠ Nu s-au găsit imagini în directorul raw!")
        print("  Urmează instrucțiunile de mai sus pentru a descărca dataset-ul.")


if __name__ == "__main__":
    main()
