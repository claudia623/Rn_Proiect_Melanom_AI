"""
Utility functions for Web UI - Image processing and helpers
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def calculate_image_quality_metrics(image: np.ndarray) -> dict:
    """
    Calculează metrici de calitate imagine.
    
    Returns:
        {
            'laplacian_var': float,  # Blur score
            'contrast': float,       # Contrast score
            'brightness': float,     # Brightness score
        }
    """
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Laplacian variance (blur detection)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Contrast (standard deviation)
    contrast = np.std(gray)
    
    # Brightness (mean)
    brightness = np.mean(gray)
    
    return {
        'laplacian_var': laplacian_var,
        'contrast': contrast,
        'brightness': brightness,
    }


def enhance_image_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Îmbunătățește contrast imagine folosind CLAHE.
    
    Args:
        image: BGR image
        clip_limit: CLAHE clip limit
    
    Returns:
        Enhanced image
    """
    
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge and convert back to BGR
    lab_clahe = cv2.merge([l_clahe, a, b])
    image_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return image_enhanced


def get_image_statistics(image: np.ndarray) -> dict:
    """
    Calculează statistici per canal BGR.
    """
    
    stats = {}
    for i, channel_name in enumerate(['Blue', 'Green', 'Red']):
        channel = image[:, :, i]
        stats[channel_name] = {
            'mean': np.mean(channel),
            'std': np.std(channel),
            'min': np.min(channel),
            'max': np.max(channel),
        }
    
    return stats
