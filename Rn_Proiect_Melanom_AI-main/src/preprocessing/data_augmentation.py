"""
Data Augmentation Module pentru Melanom AI
===========================================
Funcții pentru augmentarea datelor de antrenare
"""

import numpy as np
import cv2
from typing import Tuple, List, Callable
import random


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """Flip orizontal"""
    return np.fliplr(image)


def vertical_flip(image: np.ndarray) -> np.ndarray:
    """Flip vertical"""
    return np.flipud(image)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotește imaginea cu unghiul specificat
    
    Args:
        image: Imaginea de intrare
        angle: Unghiul de rotație în grade
    
    Returns:
        Imaginea rotită
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_rotation(image: np.ndarray, max_angle: float = 20) -> np.ndarray:
    """Rotație aleatorie în intervalul [-max_angle, max_angle]"""
    angle = random.uniform(-max_angle, max_angle)
    return rotate_image(image, angle)


def random_zoom(image: np.ndarray, zoom_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """
    Zoom aleator pe imagine
    
    Args:
        image: Imaginea de intrare
        zoom_range: Intervalul de zoom (min, max)
    
    Returns:
        Imaginea cu zoom aplicat
    """
    h, w = image.shape[:2]
    zoom_factor = random.uniform(*zoom_range)
    
    # Calculare noua dimensiune
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    
    # Redimensionare
    zoomed = cv2.resize(image, (new_w, new_h))
    
    # Crop sau padding pentru a reveni la dimensiunea originală
    if zoom_factor > 1:
        # Crop centru
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return zoomed[start_h:start_h+h, start_w:start_w+w]
    else:
        # Padding
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        result = np.zeros_like(image)
        result[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = zoomed
        return result


def random_shift(image: np.ndarray, 
                 max_shift: Tuple[float, float] = (0.1, 0.1)) -> np.ndarray:
    """
    Shift aleator pe imagine
    
    Args:
        image: Imaginea de intrare
        max_shift: Shift maxim ca fracție din dimensiune (x, y)
    
    Returns:
        Imaginea translatată
    """
    h, w = image.shape[:2]
    max_dx = int(w * max_shift[0])
    max_dy = int(h * max_shift[1])
    
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)
    
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Ajustează luminozitatea imaginii
    
    Args:
        image: Imaginea de intrare [0, 1]
        factor: Factor de luminozitate (>1 mai luminos, <1 mai întunecat)
    
    Returns:
        Imaginea cu luminozitate ajustată
    """
    return np.clip(image * factor, 0, 1)


def random_brightness(image: np.ndarray, 
                      range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """Ajustare aleatorie a luminozității"""
    factor = random.uniform(*range)
    return adjust_brightness(image, factor)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Ajustează contrastul imaginii
    
    Args:
        image: Imaginea de intrare [0, 1]
        factor: Factor de contrast
    
    Returns:
        Imaginea cu contrast ajustat
    """
    mean = np.mean(image)
    return np.clip((image - mean) * factor + mean, 0, 1)


def random_contrast(image: np.ndarray, 
                    range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """Ajustare aleatorie a contrastului"""
    factor = random.uniform(*range)
    return adjust_contrast(image, factor)


def add_gaussian_noise(image: np.ndarray, 
                       mean: float = 0, 
                       std: float = 0.01) -> np.ndarray:
    """
    Adaugă zgomot Gaussian
    
    Args:
        image: Imaginea de intrare [0, 1]
        mean: Media zgomotului
        std: Deviația standard
    
    Returns:
        Imaginea cu zgomot
    """
    noise = np.random.normal(mean, std, image.shape)
    return np.clip(image + noise, 0, 1)


def color_jitter(image: np.ndarray,
                 hue_range: float = 0.1,
                 saturation_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """
    Variație aleatorie a culorii (hue și saturație)
    
    Args:
        image: Imaginea RGB [0, 1]
        hue_range: Variație maximă hue
        saturation_range: Intervalul pentru saturație
    
    Returns:
        Imaginea cu culori modificate
    """
    # Conversie la HSV
    image_uint8 = (image * 255).astype(np.uint8)
    hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Modificare hue
    hsv[:, :, 0] += random.uniform(-hue_range, hue_range) * 180
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 180)
    
    # Modificare saturație
    sat_factor = random.uniform(*saturation_range)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
    
    # Conversie înapoi la RGB
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return result.astype(np.float32) / 255.0


class DataAugmentor:
    """
    Clasă pentru augmentarea datelor cu configurare flexibilă
    """
    
    def __init__(self, 
                 rotation_range: float = 20,
                 horizontal_flip: bool = True,
                 vertical_flip: bool = True,
                 zoom_range: Tuple[float, float] = (0.9, 1.1),
                 shift_range: Tuple[float, float] = (0.1, 0.1),
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 noise_std: float = 0.01,
                 color_jitter: bool = True):
        """
        Inițializare augmentor
        
        Args:
            rotation_range: Unghi maxim rotație
            horizontal_flip: Activare flip orizontal
            vertical_flip: Activare flip vertical
            zoom_range: Interval zoom
            shift_range: Interval shift (x, y)
            brightness_range: Interval luminozitate
            contrast_range: Interval contrast
            noise_std: Deviație standard zgomot
            color_jitter: Activare variație culori
        """
        self.rotation_range = rotation_range
        self.h_flip = horizontal_flip
        self.v_flip = vertical_flip
        self.zoom_range = zoom_range
        self.shift_range = shift_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.color_jitter_enabled = color_jitter
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Aplică augmentări aleatorii pe imagine
        
        Args:
            image: Imaginea de intrare [0, 1]
        
        Returns:
            Imaginea augmentată
        """
        result = image.copy()
        
        # Flip orizontal (50% șansă)
        if self.h_flip and random.random() > 0.5:
            result = horizontal_flip(result)
        
        # Flip vertical (50% șansă)
        if self.v_flip and random.random() > 0.5:
            result = vertical_flip(result)
        
        # Rotație
        if self.rotation_range > 0:
            result = random_rotation(result, self.rotation_range)
        
        # Zoom
        if self.zoom_range != (1.0, 1.0):
            result = random_zoom(result, self.zoom_range)
        
        # Shift
        if self.shift_range != (0, 0):
            result = random_shift(result, self.shift_range)
        
        # Luminozitate
        if self.brightness_range != (1.0, 1.0):
            result = random_brightness(result, self.brightness_range)
        
        # Contrast
        if self.contrast_range != (1.0, 1.0):
            result = random_contrast(result, self.contrast_range)
        
        # Zgomot
        if self.noise_std > 0:
            result = add_gaussian_noise(result, std=self.noise_std)
        
        # Color jitter
        if self.color_jitter_enabled:
            result = color_jitter(result)
        
        return result.astype(np.float32)
    
    def augment_batch(self, images: np.ndarray, 
                      augmentations_per_image: int = 1) -> np.ndarray:
        """
        Augmentează un batch de imagini
        
        Args:
            images: Array de imagini (N, H, W, C)
            augmentations_per_image: Numărul de augmentări per imagine
        
        Returns:
            Array cu imaginile augmentate
        """
        augmented = []
        for img in images:
            augmented.append(img)  # Păstrează originalul
            for _ in range(augmentations_per_image):
                augmented.append(self.augment(img))
        
        return np.array(augmented)


if __name__ == "__main__":
    print("Modul data_augmentation încărcat cu succes!")
    print("Funcții disponibile:")
    print("  - horizontal_flip()")
    print("  - vertical_flip()")
    print("  - rotate_image()")
    print("  - random_zoom()")
    print("  - random_shift()")
    print("  - adjust_brightness()")
    print("  - color_jitter()")
    print("  - DataAugmentor class")
