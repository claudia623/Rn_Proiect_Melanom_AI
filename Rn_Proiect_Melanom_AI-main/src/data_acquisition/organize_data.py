"""
Organize Dataset Module pentru Melanom AI
==========================================
Script pentru organizarea dataset-ului ISIC în structura de proiect
"""

import os
import shutil
import pandas as pd
import zipfile
from tqdm import tqdm
from pathlib import Path
import sys

# Adaugă directorul părinte la path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_zip_if_needed(zip_path: str, extract_to: str) -> str:
    """
    Extrage arhiva ZIP dacă există și este validă
    
    Args:
        zip_path: Calea către arhiva ZIP
        extract_to: Directorul unde se extrage
        
    Returns:
        Calea către directorul cu imaginile extrase
    """
    if not os.path.exists(zip_path):
        print(f"⚠ Arhiva nu există: {zip_path}")
        return extract_to

    # Verifică dacă e .crdownload
    if zip_path.endswith('.crdownload'):
        print(f"❌ Eroare: Fișierul {os.path.basename(zip_path)} este o descărcare incompletă!")
        return extract_to

    try:
        print(f"📦 Extragere {os.path.basename(zip_path)}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✓ Extragere completă!")
        
        # Returnează directorul care conține imaginile (uneori e un subfolder în zip)
        # Căutăm primul subfolder care conține imagini
        for root, dirs, files in os.walk(extract_to):
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                return root
                
        return extract_to
        
    except zipfile.BadZipFile:
        print(f"❌ Eroare: Fișierul {os.path.basename(zip_path)} este corupt sau invalid!")
        return extract_to
    except Exception as e:
        print(f"❌ Eroare la extragere: {e}")
        return extract_to


def organize_isic_data(date_dir: str, output_dir: str = "data/raw"):
    """
    Organizează datele ISIC din folderul 'date' în structura proiectului
    
    Args:
        date_dir: Directorul sursă (unde sunt zip-urile și csv-urile)
        output_dir: Directorul destinație
    """
    print(f"🚀 Începere organizare date din: {date_dir}")
    
    # Căi fișiere
    train_csv = os.path.join(date_dir, "ISBI2016_ISIC_Part3_Training_GroundTruth.csv")
    test_csv = os.path.join(date_dir, "ISBI2016_ISIC_Part3_Test_GroundTruth.csv")
    
    train_zip = os.path.join(date_dir, "ISBI2016_ISIC_Part3_Training_Data.zip")
    test_zip = os.path.join(date_dir, "ISBI2016_ISIC_Part3_Test_Data.zip")
    
    # Verifică și gestionează extensia .crdownload
    if not os.path.exists(train_zip) and os.path.exists(train_zip + ".crdownload"):
        train_zip += ".crdownload"
    if not os.path.exists(test_zip) and os.path.exists(test_zip + ".crdownload"):
        test_zip += ".crdownload"

    # Directoare temporare pentru extragere
    temp_dir = os.path.join(date_dir, "temp_extracted")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. Procesare date de antrenare
    if os.path.exists(train_csv):
        print("\n📋 Procesare set antrenare...")
        
        # Extrage imaginile
        train_images_dir = extract_zip_if_needed(train_zip, os.path.join(temp_dir, "train"))
        
        # Citește CSV-ul
        try:
            df = pd.read_csv(train_csv, header=None, names=['image_id', 'label'])
            print(f"   Găsite {len(df)} intrări în CSV")
            
            # Contoare
            moved_count = 0
            missing_count = 0
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Organizare imagini"):
                image_id = row['image_id']
                label = row['label'] # benign / malignant
                
                # Construiește calea sursă și destinație
                src_path = os.path.join(train_images_dir, f"{image_id}.jpg")
                if not os.path.exists(src_path):
                    src_path = os.path.join(train_images_dir, f"{image_id}.jpeg")
                
                if os.path.exists(src_path):
                    dst_dir = os.path.join(output_dir, label)
                    os.makedirs(dst_dir, exist_ok=True)
                    
                    dst_path = os.path.join(dst_dir, f"{image_id}.jpg")
                    shutil.copy2(src_path, dst_path)
                    moved_count += 1
                else:
                    missing_count += 1
            
            print(f"✓ Mutate: {moved_count}")
            if missing_count > 0:
                print(f"⚠ Lipsă: {missing_count} (posibil din cauza arhivei incomplete)")
                
        except Exception as e:
            print(f"❌ Eroare la procesarea CSV-ului de antrenare: {e}")
    else:
        print(f"⚠ Nu s-a găsit CSV-ul de antrenare: {train_csv}")

    # 2. Procesare date de test (dacă există CSV cu ground truth)
    if os.path.exists(test_csv):
        print("\n📋 Procesare set testare...")
        
        # Extrage imaginile
        test_images_dir = extract_zip_if_needed(test_zip, os.path.join(temp_dir, "test"))
        
        try:
            df_test = pd.read_csv(test_csv, header=None, names=['image_id', 'label'])
            print(f"   Găsite {len(df_test)} intrări în CSV test")
            
            # Pentru test, le punem tot în raw/benign și raw/malignant momentan, 
            # sau putem să le punem direct în data/test dacă vrem să păstrăm split-ul original ISIC.
            # Dar scriptul download_dataset.py face split automat.
            # Pentru consistență, le punem în raw și lăsăm split-ul să decidă, 
            # SAU le punem separat. 
            # Având în vedere cerința "organizezi in folderul data dupa nume", 
            # și structura proiectului are data/raw, le voi pune în data/raw.
            
            moved_count = 0
            for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Organizare imagini test"):
                image_id = row['image_id']
                label = row['label']
                
                # Verifică dacă label-ul e valid (uneori e 0.0/1.0 în loc de string)
                if isinstance(label, (int, float)):
                    label = 'malignant' if label == 1 else 'benign'
                
                src_path = os.path.join(test_images_dir, f"{image_id}.jpg")
                if not os.path.exists(src_path):
                    src_path = os.path.join(test_images_dir, f"{image_id}.jpeg")
                
                if os.path.exists(src_path):
                    dst_dir = os.path.join(output_dir, label)
                    os.makedirs(dst_dir, exist_ok=True)
                    
                    dst_path = os.path.join(dst_dir, f"{image_id}.jpg")
                    shutil.copy2(src_path, dst_path)
                    moved_count += 1
            
            print(f"✓ Mutate (test): {moved_count}")
            
        except Exception as e:
            print(f"❌ Eroare la procesarea CSV-ului de test: {e}")

    # Curățenie
    # shutil.rmtree(temp_dir) # Comentat pentru debug
    print("\n✨ Organizare finalizată!")


if __name__ == "__main__":
    # Calea către folderul 'date' din rădăcina proiectului
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    date_folder = os.path.join(project_root, "date")
    
    organize_isic_data(date_folder)
