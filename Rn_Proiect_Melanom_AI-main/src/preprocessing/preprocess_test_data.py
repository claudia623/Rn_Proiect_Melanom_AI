import os
import cv2
import glob
import yaml
import numpy as np
from src.preprocessing.image_processing import preprocess_image, load_config

def calculate_blur_score(image):
    """Calculează scorul de blur dintr-o matrice de imagine (nu din cale)."""
    if image is None:
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_test_folder():
    # 1. Încarcă configurația
    print("--- Încărcare configurație ---")
    config = load_config("config/config.yaml")
    target_size = (config['image']['width'], config['image']['height'])
    test_path = config['data']['test_path'] # de obicei "data/test"
    
    print(f"Dimensiune țintă: {target_size}")
    print(f"Folder test vizat: {test_path}")

    # 2. Identifică toate imaginile din folderul de test
    extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(test_path, "**", ext), recursive=True))

    if not image_paths:
        print(f"❌ Nu s-au găsit imagini în folderul {test_path}!")
        return

    print(f"S-au găsit {len(image_paths)} imagini de test.")

    # 3. Procesează fiecare imagine
    processed_count = 0
    errors = 0

    print("\n--- Începere procesare folder TEST ---")
    total = len(image_paths)
    
    for i, img_path in enumerate(image_paths):
        # Afișăm progresul
        if (i + 1) % 5 == 0 or (i + 1) == total:
            print(f"Progres test: {i + 1}/{total} ({(i + 1)/total*100:.1f}%)")
            
        try:
            # Procesare: resize, sharpen, hair removal
            # Folosim aceiași parametri ca la dataset-ul de antrenare pentru consistență
            processed_img = preprocess_image(
                img_path, 
                target_size=target_size,
                remove_artifacts=True,
                enhance=True,
                sharpen=True
            )
            
            # Convertim înapoi la 0-255 și BGR pentru salvare (suprascriem imaginea originală cu cea procesată)
            output_img = (processed_img * 255).astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            
            # Suprascriem fișierul original cu versiunea îmbunătățită
            cv2.imwrite(img_path, output_img)
            processed_count += 1
            
        except Exception as e:
            print(f"⚠️ Eroare la procesarea {img_path}: {e}")
            errors += 1

    # 4. Rezumat
    print("\n--- Rezumat Procesare TEST ---")
    print(f"✅ Total imagini de test procesate și optimizate: {processed_count}")
    print(f"❌ Erori: {errors}")
    print(f"Imaginile din {test_path} sunt acum la link-ul corect și claritate optimă.")

if __name__ == "__main__":
    process_test_folder()
