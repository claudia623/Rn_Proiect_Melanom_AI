"""
Script pentru organizare imagini ISIC din radacina in data/raw/benign/malignant
Pe baza metadata.csv
"""

import os
import shutil
import pandas as pd
from pathlib import Path

# Read metadata
metadata = pd.read_csv('metadata.csv')

# Create directories
os.makedirs('data/raw/benign', exist_ok=True)
os.makedirs('data/raw/malignant', exist_ok=True)

# Map diagnosis to class
malignant_keywords = ['malignant', 'melanoma', 'cancer', 'carcinoma']

# Get list of ISIC images in root
root_images = [f for f in os.listdir('.') if f.startswith('ISIC') and f.endswith('.jpg')]

print(f"Found {len(root_images)} ISIC images in root")

# Process each image
for image_file in root_images:
    isic_id = image_file.replace('.jpg', '')
    
    # Find in metadata
    if isic_id not in metadata['isic_id'].values:
        print(f"⚠️ {isic_id} not in metadata, skipping")
        continue
    
    row = metadata[metadata['isic_id'] == isic_id].iloc[0]
    diagnosis = str(row['diagnosis_1']).lower() if pd.notna(row['diagnosis_1']) else ""
    
    # Classify
    if any(keyword in diagnosis for keyword in malignant_keywords):
        dest = f'data/raw/malignant/{image_file}'
    else:
        dest = f'data/raw/benign/{image_file}'
    
    # Copy
    try:
        shutil.copy(image_file, dest)
        print(f"✅ {image_file} → {dest}")
    except Exception as e:
        print(f"❌ Error copying {image_file}: {e}")

# Statistics
benign_count = len(os.listdir('data/raw/benign'))
malignant_count = len(os.listdir('data/raw/malignant'))

print(f"\n✅ Done!")
print(f"Benign: {benign_count}")
print(f"Malignant: {malignant_count}")
print(f"Total: {benign_count + malignant_count}")
