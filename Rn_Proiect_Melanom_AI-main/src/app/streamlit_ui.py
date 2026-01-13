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
import matplotlib.pyplot as plt
import json
from pathlib import Path
import logging
from datetime import datetime
import os
from typing import Tuple, Optional, Dict

# Import Module 2 (Neural Network)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'model_path': 'models/melanom_efficientnetb0_best.keras',
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
            return model
        else:
            current_dir = os.getcwd()
            st.error(f"❌ Model not found at {CONFIG['model_path']}")
            st.info(f"Current working directory: {current_dir}")
            st.info("Searching for models in local directory...")
            
            # Try to help the user find the file
            if os.path.exists('Rn_Proiect_Melanom_AI-main/' + CONFIG['model_path']):
                st.warning(f"💡 Found it in a subfolder! Please use the provided run_app.bat to launch.")
            
            st.stop()
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
    2. Convert to RGB
    3. Expand dims (1, 224, 224, 3)
    """
    
    # Resize
    image_resized = cv2.resize(image, CONFIG['image_size'])
    
    # Convert BGR → RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # Expand dims
    image_batch = np.expand_dims(image_rgb, axis=0)
    
    # Note: EfficientNet model includes preprocessing layer, so we pass raw pixels [0, 255]
    
    return image_batch





def log_prediction(filename: str, classification: str, probability: float) -> None:
    """
    Salvează predicție în CSV pentru audit clinic.
    """
    
    try:
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'classification': classification,
            'probability_malignant': probability,
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


def plot_training_history(history1_path: str, history2_path: str):
    """Afișează graficele de antrenare."""
    
    try:
        with open(history1_path, 'r') as f:
            h1 = json.load(f)
        
        has_h2 = False
        if os.path.exists(history2_path):
            with open(history2_path, 'r') as f:
                h2 = json.load(f)
            has_h2 = True
            
        # Merge metrics
        metrics = ['accuracy', 'loss', 'auc']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, metric in enumerate(metrics):
            if metric in h1:
                val_metric = f'val_{metric}'
                data = h1[metric]
                val_data = h1[val_metric]
                
                if has_h2 and metric in h2:
                    data = data + h2[metric]
                    val_data = val_data + h2[val_metric]
                
                axes[i].plot(data, label='Train')
                axes[i].plot(val_data, label='Val')
                if has_h2:
                    axes[i].axvline(x=len(h1[metric])-1, color='r', linestyle='--', label='Fine-tuning')
                axes[i].set_title(metric.capitalize())
                axes[i].set_xlabel('Epoch')
                axes[i].legend()
        
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Could not load history for plotting: {str(e)}")


# ============================================================================
# MAIN UI
# ============================================================================

def main():
    """
    Interfață Streamlit principală.
    """
    
    # Header
    st.title("🏥 Melanom AI - Classification System")
    st.markdown("**Automatic skin lesion classification: Benign vs Malignant**")
    
    # Tabs
    tab_diag, tab_metrics = st.tabs(["🩺 Diagnosis", "📊 Model Metrics"])
    
    with tab_diag:
        # Move previous main code here
        diag_ui()
    
    with tab_metrics:
        st.header("📈 Training Performance")
        st.info("Visualizing the learning curve from Phase 1 (Transfer Learning) and Phase 2 (Fine-tuning)")
        
        h1_path = 'results/melanom_efficientnetb0_phase1_history.json'
        h2_path = 'results/melanom_efficientnetb0_phase2_history.json'
        
        if os.path.exists(h1_path):
            plot_training_history(h1_path, h2_path)
            
            # Show stats
            with open(h1_path, 'r') as f:
                h1 = json.load(f)
            
            st.write(f"**Phase 1 Completed:** {len(h1['accuracy'])} epochs")
            if os.path.exists(h2_path):
                with open(h2_path, 'r') as f:
                    h2 = json.load(f)
                st.write(f"**Phase 2 Completed:** {len(h2['accuracy'])} epochs")
        else:
            st.warning("Training history not found. Finish training to see metrics.")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ System Info")
        st.markdown("""
        **How it works:**
        1. Upload dermatoscopic image
        2. System validates image quality
        3. Preprocesses image (EfficientNet standard)
        4. Classifies using trained Neural Network
        
        **Model Info:**
        - **Architecture:** EfficientNetB0
        - **Input:** 224x224 RGB
        - **Output:** Probability (0-1)
        - **Threshold:** 0.5
        """)
        
        st.divider()
        st.markdown("**Status:** Etapa 5 (Trained Model)")


def diag_ui():
    # Load model (cached)
    with st.spinner("Loading neural network model..."):
        model = load_nn_model()
    
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
            # Convert BGR (OpenCV) to RGB (Streamlit)
            image_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_display, caption="Uploaded Image", use_container_width=True)
            
            # Preprocess
            image_preprocessed = preprocess_image(image)
            
            # ====================================================================
            # RIGHT COLUMN - ANALYSIS & RESULTS
            # ====================================================================
            
            with col2:
                st.header("🔍 Analysis Results")
                
                if st.button("🎯 Analyze Image", use_container_width=True):
                    
                    with st.spinner("Analyzing image..."):
                        try:
                            # Predict
                            prediction = model.predict(image_preprocessed, verbose=0)
                            probability = float(prediction[0][0])
                            
                            # Classify
                            if probability > 0.5:
                                classification = "MALIGNANT"
                                confidence = probability
                            else:
                                classification = "BENIGN"
                                confidence = 1.0 - probability
                            
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
                    else:
                        st.markdown(f"<div class='diagnosis-malignant'>⚠️ MALIGNANT</div>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Confidence score
                    st.markdown("### 📊 Confidence")
                    confidence_pct = confidence * 100
                    
                    if confidence > 0.8:
                        confidence_class = "confidence-high"
                        confidence_text = "HIGH"
                    elif confidence > 0.6:
                        confidence_class = "confidence-medium"
                        confidence_text = "MEDIUM"
                    else:
                        confidence_class = "confidence-low"
                        confidence_text = "LOW"
                    
                    st.markdown(f"<div class='{confidence_class}'>{confidence_pct:.1f}% ({confidence_text})</div>",
                              unsafe_allow_html=True)
                    
                    st.progress(confidence)
                    
                    # Probability details
                    with st.expander("ℹ️ Technical Details"):
                        st.write(f"Raw Probability (Malignant): {probability:.4f}")
                        st.write(f"Threshold: 0.5")
                    
                    # Log prediction
                    log_prediction(
                        uploaded_file.name,
                        classification,
                        probability
                    )
                    
                    st.success("✅ Prediction saved to logs!")
        
        else:
            st.info("👆 Upload an image to start analysis")


if __name__ == '__main__':
    main()
