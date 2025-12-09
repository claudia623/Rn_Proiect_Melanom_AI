"""
Modul 3: Web Service / UI - Streamlit Application
===================================================
Interfață web pentru clasificarea melanomului bazată pe similaritate imagini.

Pipeline:
1. User upload imagine
2. Validare imagine (format, dimensiuni, blur check)
3. Preproceșare
4. Feature extraction (Modul 2)
5. Similarity computation vs bază referință (Modul 1)
6. Clasificare (BENIGN/MALIGNANT)
7. Afișare rezultate interactive
8. Log predictions în CSV
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import os
from typing import Tuple, Optional, Dict

# Import Module 2 (Neural Network)
from src.neural_network.similarity_model import (
    create_similarity_model,
    load_model,
    extract_features,
    compute_similarity,
    classify_melanoma
)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'model_path': 'models/similarity_model_untrained.h5',
    'reference_dir': 'data/generated/original/',
    'output_dir': 'logs/',
    'log_file': 'logs/predictions.csv',
    'image_size': (224, 224),
    'blur_threshold': 100,  # Laplacian variance threshold
    'max_file_size_mb': 10,
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Melanom AI - Classification System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .diagnosis-benign {
        color: #00AA00;
        font-size: 24px;
        font-weight: bold;
    }
    .diagnosis-malignant {
        color: #FF0000;
        font-size: 24px;
        font-weight: bold;
    }
    .confidence-high {
        color: #00AA00;
    }
    .confidence-medium {
        color: #FFAA00;
    }
    .confidence-low {
        color: #FF0000;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_nn_model():
    """Încarcă modelul RN o singură dată (cached)."""
    try:
        if os.path.exists(CONFIG['model_path']):
            model = load_model(CONFIG['model_path'])
            logger.info(f"Model loaded from {CONFIG['model_path']}")
        else:
            logger.info("Creating new similarity model...")
            model = create_similarity_model()
            # Save for future use
            os.makedirs(Path(CONFIG['model_path']).parent, exist_ok=True)
            model.save(CONFIG['model_path'])
            logger.info(f"Model saved to {CONFIG['model_path']}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"❌ Error loading neural network model: {str(e)}")
        st.stop()


def validate_image(image_bytes) -> Tuple[bool, str, Optional[np.ndarray]]:
    """
    Validează imagine (format, size, blur).
    
    Returns:
        (is_valid, message, image_array)
    """
    
    try:
        # Check file size
        file_size_mb = len(image_bytes) / (1024 * 1024)
        if file_size_mb > CONFIG['max_file_size_mb']:
            return False, f"❌ File size too large: {file_size_mb:.1f}MB > {CONFIG['max_file_size_mb']}MB", None
        
        # Read image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return False, "❌ Invalid image format. Use JPG or PNG.", None
        
        # Check dimensions
        h, w = image.shape[:2]
        if h < 100 or w < 100:
            return False, f"❌ Image too small: {w}x{h}. Min 100x100 required.", None
        
        if h > 2048 or w > 2048:
            return False, f"❌ Image too large: {w}x{h}. Max 2048x2048 allowed.", None
        
        # Check blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < CONFIG['blur_threshold']:
            return False, f"❌ Image too blurry (score: {laplacian_var:.1f}). Please retake photo.", None
        
        return True, "✅ Image valid", image
    
    except Exception as e:
        return False, f"❌ Error validating image: {str(e)}", None


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocesează imagine pentru neural network.
    
    1. Resize la 224x224
    2. Normalize la [0, 1]
    3. Ensure RGB format
    """
    
    # Resize
    image_resized = cv2.resize(image, CONFIG['image_size'])
    
    # Convert BGR → RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image_normalized = image_rgb.astype(np.float32) / 255.0
    
    return image_normalized


@st.cache_data
def load_reference_images() -> Dict[str, list]:
    """
    Încarcă imagini referință din baza de date.
    
    Returns:
        {
            'benign': [(image_path, image_array), ...],
            'malignant': [(image_path, image_array), ...]
        }
    """
    
    reference_images = {'benign': [], 'malignant': []}
    
    for class_label in ['benign', 'malignant']:
        class_dir = os.path.join(CONFIG['reference_dir'], class_label)
        
        if not os.path.exists(class_dir):
            logger.warning(f"Reference directory not found: {class_dir}")
            continue
        
        image_files = sorted([f for f in os.listdir(class_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:10]  # Max 10 per class
        
        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    image_resized = cv2.resize(image, CONFIG['image_size'])
                    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                    image_normalized = image_rgb.astype(np.float32) / 255.0
                    reference_images[class_label].append((image_file, image_normalized))
            except Exception as e:
                logger.warning(f"Error loading reference image {image_path}: {str(e)}")
    
    logger.info(f"Loaded {len(reference_images['benign'])} benign reference images")
    logger.info(f"Loaded {len(reference_images['malignant'])} malignant reference images")
    
    return reference_images


def compute_similarities(model, image: np.ndarray, reference_images: Dict) -> Tuple[list, list]:
    """
    Calculează similarități cu toate imaginile referință.
    """
    
    try:
        # Extract features din imagine test
        features_test = extract_features(model, image)
        
        similarities_benign = []
        similarities_malignant = []
        
        # Compare with benign references
        for ref_name, ref_image in reference_images['benign']:
            features_ref = extract_features(model, ref_image)
            sim = compute_similarity(features_test, features_ref)
            similarities_benign.append(sim)
        
        # Compare with malignant references
        for ref_name, ref_image in reference_images['malignant']:
            features_ref = extract_features(model, ref_image)
            sim = compute_similarity(features_test, features_ref)
            similarities_malignant.append(sim)
        
        return similarities_benign, similarities_malignant
    
    except Exception as e:
        logger.error(f"Error computing similarities: {str(e)}")
        return [], []


def log_prediction(filename: str, classification: str, scores: Dict) -> None:
    """
    Salvează predicție în CSV pentru audit clinic.
    """
    
    try:
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'classification': classification,
            'benign_score': scores['benign_mean'],
            'benign_std': scores['benign_std'],
            'malignant_score': scores['malignant_mean'],
            'malignant_std': scores['malignant_std'],
            'confidence': scores['confidence'],
        }
        
        # Append to CSV
        df = pd.DataFrame([prediction_record])
        if os.path.exists(CONFIG['log_file']):
            df_existing = pd.read_csv(CONFIG['log_file'])
            df = pd.concat([df_existing, df], ignore_index=True)
        
        df.to_csv(CONFIG['log_file'], index=False)
        logger.info(f"Prediction logged to {CONFIG['log_file']}")
    
    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}")


# ============================================================================
# MAIN UI
# ============================================================================

def main():
    """
    Interfață Streamlit principală.
    """
    
    # Header
    st.title("🏥 Melanom AI - Similarity-Based Classification System")
    st.markdown("**Automatic skin lesion classification: Benign vs Malignant**")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ System Info")
        st.markdown("""
        **How it works:**
        1. Upload dermatoscopic image
        2. System validates image quality
        3. Extracts features using EfficientNetB0
        4. Compares with 20+ reference images
        5. Classifies as BENIGN or MALIGNANT
        
        **Classification Method:**
        - Similarity-based matching
        - Cosine distance metric
        - Confidence threshold: 0.50+
        
        **Reference Dataset:**
        - 15+ benign reference images
        - 15+ malignant reference images
        - ISIC archive source + synthetic augmentation
        """)
        
        st.divider()
        st.markdown("**Model Info:**")
        st.markdown(f"""
        - **Architecture:** EfficientNetB0 + Dense(256)
        - **Features:** 256D vectors (L2 normalized)
        - **Transfer Learning:** ImageNet pretraining
        - **Inference Time:** ~100ms per image
        - **Model Status:** Etapa 4 (Untrained)
        """)
    
    # Load model and references (cached)
    with st.spinner("Loading neural network model..."):
        model = load_nn_model()
    
    with st.spinner("Loading reference images..."):
        reference_images = load_reference_images()
    
    # Check if we have references
    if not reference_images['benign'] or not reference_images['malignant']:
        st.warning("⚠️ Not enough reference images loaded. Please ensure data/generated/original/ contains images.")
        st.info("Run: python src/data_acquisition/generate_synthetic_data.py")
        st.stop()
    
    # Main content - Two columns
    col1, col2 = st.columns([1, 1])
    
    # ========================================================================
    # LEFT COLUMN - INPUT
    # ========================================================================
    
    with col1:
        st.header("📸 Image Upload")
        
        uploaded_file = st.file_uploader(
            "Upload dermatoscopic image (JPG, PNG)",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            # Validate
            is_valid, validation_msg, image = validate_image(uploaded_file.read())
            st.info(validation_msg)
            
            if not is_valid:
                st.stop()
            
            # Display uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True, channels='RGB')
            
            # Preprocess
            image_preprocessed = preprocess_image(image)
            
            # ====================================================================
            # RIGHT COLUMN - ANALYSIS & RESULTS
            # ====================================================================
            
            with col2:
                st.header("🔍 Analysis Results")
                
                if st.button("🎯 Analyze Image", use_container_width=True):
                    
                    with st.spinner("Computing image features..."):
                        # Extract features
                        try:
                            similarities_benign, similarities_malignant = compute_similarities(
                                model, image_preprocessed, reference_images
                            )
                            
                            if not similarities_benign or not similarities_malignant:
                                st.error("❌ Error computing similarities")
                                st.stop()
                            
                            # Classify
                            classification, confidence, scores = classify_melanoma(
                                similarities_benign,
                                similarities_malignant
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
                            logger.error(f"Analysis error: {str(e)}")
                            st.stop()
                    
                    # ================================================================
                    # RESULTS DISPLAY
                    # ================================================================
                    
                    # Classification badge
                    st.divider()
                    st.markdown("### 📋 Classification Result")
                    
                    if classification == "BENIGN":
                        st.markdown(f"<div class='diagnosis-benign'>✅ BENIGN</div>", unsafe_allow_html=True)
                        color = "green"
                    else:
                        st.markdown(f"<div class='diagnosis-malignant'>⚠️ MALIGNANT</div>", unsafe_allow_html=True)
                        color = "red"
                    
                    st.divider()
                    
                    # Confidence score
                    st.markdown("### 📊 Confidence")
                    confidence_pct = confidence * 100
                    
                    if confidence > 0.7:
                        confidence_class = "confidence-high"
                        confidence_text = "HIGH"
                    elif confidence > 0.3:
                        confidence_class = "confidence-medium"
                        confidence_text = "MEDIUM"
                    else:
                        confidence_class = "confidence-low"
                        confidence_text = "LOW"
                    
                    st.markdown(f"<div class='{confidence_class}'>{confidence_pct:.1f}% ({confidence_text})</div>",
                              unsafe_allow_html=True)
                    
                    # Similarity scores
                    st.markdown("### 📈 Similarity Scores")
                    
                    col_ben, col_mal = st.columns(2)
                    
                    with col_ben:
                        st.metric(
                            "Benign Match",
                            f"{scores['benign_mean']:.1%}",
                            f"σ={scores['benign_std']:.1%}"
                        )
                    
                    with col_mal:
                        st.metric(
                            "Malignant Match",
                            f"{scores['malignant_mean']:.1%}",
                            f"σ={scores['malignant_std']:.1%}"
                        )
                    
                    # Detailed statistics
                    with st.expander("📊 Detailed Statistics"):
                        st.write("""
                        - **Benign Mean:** Average similarity with benign references
                        - **Benign Std:** Standard deviation (consistency)
                        - **Benign Min/Max:** Range of similarity scores
                        - Similar interpretation for Malignant scores
                        """)
                        
                        stats_df = pd.DataFrame({
                            'Class': ['Benign', 'Malignant'],
                            'Mean Score': [scores['benign_mean'], scores['malignant_mean']],
                            'Std Dev': [scores['benign_std'], scores['malignant_std']],
                            'Min': [scores['benign_min'], scores['malignant_min']],
                            'Max': [scores['benign_max'], scores['malignant_max']],
                        })
                        st.dataframe(stats_df, use_container_width=True)
                    
                    # Reference images grid
                    st.markdown("### 🖼️ Top Similar Reference Images")
                    
                    # Get top 3 similar benign
                    top_benign_idx = np.argsort(similarities_benign)[::-1][:3]
                    top_malignant_idx = np.argsort(similarities_malignant)[::-1][:3]
                    
                    st.markdown("**Similar Benign References:**")
                    cols = st.columns(3)
                    for i, idx in enumerate(top_benign_idx):
                        with cols[i]:
                            ref_name, ref_image = reference_images['benign'][idx]
                            st.image(ref_image, 
                                   caption=f"{ref_name}\n{similarities_benign[idx]:.1%}",
                                   use_column_width=True)
                    
                    st.markdown("**Similar Malignant References:**")
                    cols = st.columns(3)
                    for i, idx in enumerate(top_malignant_idx):
                        with cols[i]:
                            ref_name, ref_image = reference_images['malignant'][idx]
                            st.image(ref_image,
                                   caption=f"{ref_name}\n{similarities_malignant[idx]:.1%}",
                                   use_column_width=True)
                    
                    # Log prediction
                    log_prediction(
                        uploaded_file.name,
                        classification,
                        scores
                    )
                    
                    st.success("✅ Prediction saved to logs!")
        
        else:
            st.info("👆 Upload an image to start analysis")


if __name__ == '__main__':
    main()
