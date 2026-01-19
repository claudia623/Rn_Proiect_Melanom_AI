import os
import cv2
import glob
import yaml
import numpy as np
from src.preprocessing.image_processing import preprocess_image, load_config

def calculate_blur_score(image_path):
    """Calculează scorul de blur folosind varianța Laplacianului."""
    image = cv2.imread(image_path)
    if image is None:
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_dataset():
    # 1. Încarcă configurația
    print("--- Încărcare configurație ---")
    config = load_config("config/config.yaml")
    target_size = (config['image']['width'], config['image']['height'])
    raw_path = config['data']['raw_path']
    processed_base_path = config['data']['processed_path']
    
    print(f"Dimensiune țintă: {target_size}")
    print(f"Sursă imagini: {raw_path}")
    print(f"Destinație: {processed_base_path}")

    # 2. Identifică toate imaginile din raw (benign și malignant)
    extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(raw_path, "**", ext), recursive=True))

    if not image_paths:
        print("❌ Nu s-au găsit imagini în folderul raw!")
        return

    print(f"S-au găsit {len(image_paths)} imagini.")

    # 3. Creează folderele de destinație
    os.makedirs(processed_base_path, exist_ok=True)
    os.makedirs(os.path.join(processed_base_path, "benign"), exist_ok=True)
    os.makedirs(os.path.join(processed_base_path, "malignant"), exist_ok=True)

    # 4. Procesează fiecare imagine
    processed_count = 0
    blurry_count = 0
    errors = 0

    print("\n--- Începere procesare ---")
    total = len(image_paths)
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"Progres: {i + 1}/{total} ({(i + 1)/total*100:.1f}%)")
            
        try:
            # Determinăm clasa (benign/malignant) pe baza folderului părinte
            parent_dir = os.path.basename(os.path.dirname(img_path))
            filename = os.path.basename(img_path)
            
            # Verificăm scorul de blur inițial
            initial_blur = calculate_blur_score(img_path)
            
            # Preprocesare: resize, deblur (sharpen), hair removal, etc.
            # Funcția preprocess_image returnează imaginea normalizată (float 0-1) în format RGB
            processed_img = preprocess_image(
                img_path, 
                target_size=target_size,
                remove_artifacts=True,
                enhance=True,
                sharpen=True
            )
            
            # Convertim înapoi la 0-255 și BGR pentru salvare cu OpenCV
            output_img = (processed_img * 255).astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            
            # Verificăm scorul de blur final
            final_gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
            final_blur = cv2.Laplacian(final_gray, cv2.CV_64F).var()
            
            # Salvare
            save_path = os.path.join(processed_base_path, parent_dir, filename)
            cv2.imwrite(save_path, output_img)
            
            if final_blur < 100:
                blurry_count += 1
                
            processed_count += 1
            
        except Exception as e:
            print(f"Eroare la {img_path}: {e}")
            errors += 1

    # 5. Sumar
    print("\n--- Rezumat Procesare ---")
    print(f"✅ Total imagini procesate: {processed_count}")
    print(f"⚠️ Imagini care încă par blurate (scor < 100): {blurry_count}")
    print(f"❌ Erori: {errors}")
    print(f"Imaginile au fost salvate în: {processed_base_path}")

if __name__ == "__main__":
    process_dataset()
