"""
Image Processing Module pentru Melanom AI
==========================================
Funcții pentru preprocesarea imaginilor dermatoscopice
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Încarcă configurația din fișierul YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resize_image(image: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Redimensionează imaginea la dimensiunea specificată
    
    Args:
        image: Imaginea de intrare (numpy array)
        size: Dimensiunea țintă (width, height)
    
    Returns:
        Imaginea redimensionată
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizează valorile pixelilor la intervalul [0, 1]
    
    Args:
        image: Imaginea de intrare
    
    Returns:
        Imaginea normalizată
    """
    return image.astype(np.float32) / 255.0


def standardize_image(image: np.ndarray, 
                      mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                      std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Standardizează imaginea folosind media și deviația standard ImageNet
    
    Args:
        image: Imaginea normalizată [0, 1]
        mean: Media pe canale RGB
        std: Deviația standard pe canale RGB
    
    Returns:
        Imaginea standardizată
    """
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    return (image - mean) / std


def remove_hair(image: np.ndarray, kernel_size: int = 17) -> np.ndarray:
    """
    Elimină artefactele de tip păr din imaginile dermatoscopice
    folosind black-hat morphological filtering
    
    Args:
        image: Imaginea BGR
        kernel_size: Dimensiunea kernel-ului morfologic
    
    Returns:
        Imaginea curățată
    """
    # Convertire la grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Black-hat transform pentru detectarea părului
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Thresholding pentru masca părului
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpainting pentru eliminarea părului
    result = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)
    
    return result


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Îmbunătățește contrastul imaginii folosind CLAHE
    (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        image: Imaginea BGR
    
    Returns:
        Imaginea cu contrast îmbunătățit
    """
    # Convertire la LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplicare CLAHE pe canalul L
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Recombinare și conversie înapoi la BGR
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def crop_center(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Decupează centrul imaginii
    
    Args:
        image: Imaginea de intrare
        crop_size: Dimensiunea decupării (width, height)
    
    Returns:
        Imaginea decupată
    """
    h, w = image.shape[:2]
    crop_w, crop_h = crop_size
    
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2
    
    return image[start_y:start_y+crop_h, start_x:start_x+crop_w]


def preprocess_image(image_path: str, 
                     target_size: Tuple[int, int] = (224, 224),
                     remove_artifacts: bool = True,
                     enhance: bool = True) -> np.ndarray:
    """
    Pipeline complet de preprocesare pentru o imagine
    
    Args:
        image_path: Calea către imagine
        target_size: Dimensiunea finală
        remove_artifacts: Dacă să elimine artefactele (păr)
        enhance: Dacă să îmbunătățească contrastul
    
    Returns:
        Imaginea preprocesată și normalizată
    """
    # Citire imagine
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Nu s-a putut citi imaginea: {image_path}")
    
    # Conversie BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Eliminare artefacte (păr, markere)
    if remove_artifacts:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = remove_hair(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Îmbunătățire contrast
    if enhance:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = enhance_contrast(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionare
    image = resize_image(image, target_size)
    
    # Normalizare
    image = normalize_image(image)
    
    return image


def preprocess_batch(image_paths: List[str], 
                     target_size: Tuple[int, int] = (224, 224),
                     **kwargs) -> np.ndarray:
    """
    Preprocesează un batch de imagini
    
    Args:
        image_paths: Lista de căi către imagini
        target_size: Dimensiunea finală
        **kwargs: Argumente adiționale pentru preprocess_image
    
    Returns:
        Array numpy cu imaginile preprocesate (N, H, W, C)
    """
    images = []
    for path in image_paths:
        try:
            img = preprocess_image(path, target_size, **kwargs)
            images.append(img)
        except Exception as e:
            print(f"Eroare la procesarea {path}: {e}")
    
    return np.array(images)


if __name__ == "__main__":
    # Test module
    print("Modul image_processing încărcat cu succes!")
    print("Funcții disponibile:")
    print("  - resize_image()")
    print("  - normalize_image()")
    print("  - standardize_image()")
    print("  - remove_hair()")
    print("  - enhance_contrast()")
    print("  - preprocess_image()")
    print("  - preprocess_batch()")
