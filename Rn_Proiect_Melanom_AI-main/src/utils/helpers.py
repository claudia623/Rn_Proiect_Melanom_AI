"""
Helper Functions pentru Melanom AI
===================================
Funcții utilitare pentru proiect
"""

import os
import random
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional
from datetime import datetime


def set_seed(seed: int = 42) -> None:
    """
    Setează seed-ul pentru reproducibilitate
    
    Args:
        seed: Valoarea seed-ului
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Seed setat: {seed}")


def get_timestamp() -> str:
    """Returnează timestamp-ul curent formatat"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_available_gpus() -> List[str]:
    """
    Verifică GPU-urile disponibile
    
    Returns:
        Lista cu numele GPU-urilor disponibile
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ {len(gpus)} GPU(s) disponibil(e):")
        for gpu in gpus:
            print(f"   - {gpu.name}")
    else:
        print("⚠ Nu s-a detectat niciun GPU. Se va folosi CPU.")
    return [gpu.name for gpu in gpus]


def configure_gpu_memory_growth() -> None:
    """Configurează creșterea dinamică a memoriei GPU"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ Memory growth activat pentru GPU")
        except RuntimeError as e:
            print(f"⚠ Eroare configurare GPU: {e}")


def count_files_in_directory(directory: str, 
                             extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')) -> int:
    """
    Numără fișierele cu extensiile specificate dintr-un director
    
    Args:
        directory: Calea către director
        extensions: Tuple cu extensiile acceptate
    
    Returns:
        Numărul de fișiere
    """
    if not os.path.exists(directory):
        return 0
    
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                count += 1
    
    return count


def get_dataset_statistics(data_dir: str) -> dict:
    """
    Calculează statisticile dataset-ului
    
    Args:
        data_dir: Directorul de date
    
    Returns:
        Dicționar cu statistici
    """
    stats = {
        'train': {'benign': 0, 'malignant': 0},
        'validation': {'benign': 0, 'malignant': 0},
        'test': {'benign': 0, 'malignant': 0}
    }
    
    for split in ['train', 'validation', 'test']:
        for class_name in ['benign', 'malignant']:
            path = os.path.join(data_dir, split, class_name)
            stats[split][class_name] = count_files_in_directory(path)
    
    return stats


def print_dataset_statistics(data_dir: str) -> None:
    """
    Afișează statisticile dataset-ului
    
    Args:
        data_dir: Directorul de date
    """
    stats = get_dataset_statistics(data_dir)
    
    print("\n📊 STATISTICI DATASET")
    print("="*50)
    
    for split in ['train', 'validation', 'test']:
        total = stats[split]['benign'] + stats[split]['malignant']
        print(f"\n{split.upper()}:")
        print(f"   Benign: {stats[split]['benign']}")
        print(f"   Malign: {stats[split]['malignant']}")
        print(f"   Total: {total}")


def format_time(seconds: float) -> str:
    """
    Formatează timpul în format citibil
    
    Args:
        seconds: Timpul în secunde
    
    Returns:
        Șirul formatat
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def create_directory_if_not_exists(path: str) -> None:
    """
    Creează directorul dacă nu există
    
    Args:
        path: Calea către director
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"✓ Director creat: {path}")


def clean_empty_directories(base_path: str) -> None:
    """
    Șterge directoarele goale
    
    Args:
        base_path: Calea de bază
    """
    for root, dirs, files in os.walk(base_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                print(f"✓ Director gol șters: {dir_path}")


if __name__ == "__main__":
    print("🔧 Modul helpers încărcat cu succes!")
    
    # Test funcții
    set_seed(42)
    get_available_gpus()
    
    print(f"\nTimestamp: {get_timestamp()}")
    print(f"Format timp: {format_time(3725)}")
