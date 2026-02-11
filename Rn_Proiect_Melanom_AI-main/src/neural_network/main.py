#!/usr/bin/env python3
"""
Main Entry Point - Melanom AI Application
==========================================

Aplicație principală pentru clasificarea melanomului folosind deep learning.
Lansează interfața web Streamlit pentru diagnostic asistă a melanomului.

Utilizare:
    python main.py                   # Lansează interfața Streamlit
    python main.py --debug           # Lansează în mod debug
    python main.py --train           # Lansează antrenamentul
    python main.py --check           # Verifică dependențele

Autor: Dumitru Claudia Ștefania
Data: 2026
"""

import sys
import os
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / 'src'
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'
LOGS_DIR = PROJECT_ROOT / 'logs'
CONFIG_DIR = PROJECT_ROOT / 'config'

# Add src directory to Python path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Configure logging with UTF-8 encoding
import io
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')),
        logging.FileHandler(LOGS_DIR / 'application.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# REQUIREMENTS CHECK
# ============================================================================

REQUIRED_PACKAGES = [
    'streamlit',
    'tensorflow',
    'keras',
    'pandas',
    'numpy',
    'opencv-python',
    'matplotlib',
    'pillow',
    'scikit-learn'
]

REQUIRED_FILES = [
    MODELS_DIR / 'melanom_efficientnetb0_best.keras',
    CONFIG_DIR / 'config.yaml',
    SRC_DIR / 'app' / 'streamlit_ui.py',
]

REQUIRED_DIRS = [
    MODELS_DIR,
    DATA_DIR,
    LOGS_DIR,
    CONFIG_DIR,
    SRC_DIR
]


def check_requirements():
    """Check if all required packages and files exist"""
    logger.info("[*] Checking requirements...")
    
    # Create required directories
    logger.info("  Creating required directories...")
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"    [OK] {directory}")
    
    # Check Python version
    logger.info(f"  Python version: {sys.version}")
    if sys.version_info < (3, 8):
        logger.error("  [ERROR] Python 3.8+ required!")
        return False
    logger.info("  [OK] Python version OK")
    
    # Check required packages
    logger.info("  Checking Python packages...")
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
            logger.info(f"    [OK] {package}")
        except ImportError:
            logger.warning(f"    [SKIP] {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"  Missing packages: {', '.join(missing_packages)}")
        logger.error("  Install with: pip install -r requirements.txt")
        return False
    
    # Check critical files
    logger.info("  Checking critical files...")
    missing_files = []
    for file_path in REQUIRED_FILES:
        if file_path.exists():
            logger.info(f"    [OK] {file_path.relative_to(PROJECT_ROOT)}")
        else:
            logger.warning(f"    [MISSING] {file_path.relative_to(PROJECT_ROOT)}")
            missing_files.append(str(file_path))
    
    if missing_files:
        logger.error(f"  Missing files: {missing_files}")
        return False
    
    logger.info("[SUCCESS] All requirements satisfied!\n")
    return True


def launch_streamlit_app(debug=False):
    """Launch the Streamlit web application"""
    
    logger.info("=" * 70)
    logger.info("[START] LAUNCHING MELANOM AI APPLICATION")
    logger.info("=" * 70)
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Change to project root
    os.chdir(PROJECT_ROOT)
    
    # Streamlit command
    streamlit_app = SRC_DIR / 'app' / 'streamlit_ui.py'
    
    cmd = [
        'streamlit',
        'run',
        str(streamlit_app),
        '--logger.level=info',
        '--client.showErrorDetails=true'
    ]
    
    if debug:
        logger.info("  Running in DEBUG mode")
        cmd.append('--logger.level=debug')
    
    logger.info(f"\nCommand: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        logger.error("[ERROR] Streamlit not found!")
        logger.error("   Install with: pip install streamlit")
        return False
    except KeyboardInterrupt:
        logger.info("\n[INTERRUPT] Application interrupted by user")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Error launching Streamlit: {e}")
        return False
    
    return True


def launch_training():
    """Launch model training"""
    
    logger.info("=" * 70)
    logger.info("[TRAIN] MELANOM AI TRAINING MODULE")
    logger.info("=" * 70)
    
    try:
        from neural_network.train import main as train_main
        logger.info("Launching training pipeline...")
        train_main()
        return True
    except ImportError as e:
        logger.error(f"[ERROR] Cannot import training module: {e}")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Training error: {e}")
        return False


def print_project_info():
    """Print project information"""
    
    info = f"""
╔═══════════════════════════════════════════════════════════════════╗
║          🔬 MELANOM AI - DIAGNOSTIC ASSISTANT v0.1              ║
╚═══════════════════════════════════════════════════════════════════╝

📋 PROJECT INFORMATION:
  ├─ Name: Rn_Proiect_Melanom_AI
  ├─ Author: Dumitru Claudia Ștefania
  ├─ Date: 2026
  ├─ Model: EfficientNetB0 Transfer Learning
  ├─ Dataset: ISIC + Personal Dermoscopic Images
  ├─ Framework: TensorFlow/Keras + Streamlit
  └─ Python: {sys.version.split()[0]}

📁 DIRECTORY STRUCTURE:
  ├─ models/           → Trained ML models
  ├─ data/             → Training/validation/test datasets
  ├─ logs/             → Training logs & predictions
  ├─ src/              → Source code (modules)
  │  ├─ app/           → Streamlit web UI
  │  ├─ neural_network/→ Model architecture & training
  │  ├─ preprocessing/ → Image preprocessing
  │  ├─ optimization/  → Model optimization
  │  └─ utils/         → Utility functions
  ├─ docs/             → Documentation & visualizations
  └─ config/           → Configuration files

🎯 QUICK START:
  1. Launch Web UI:        python main.py
  2. Check Requirements:   python main.py --check
  3. Launch Training:      python main.py --train
  4. Debug Mode:           python main.py --debug

📖 DOCUMENTATION:
  • README:              DUMITRU_Claudia-Stefania_631AB_README_Proiect_RN.md
  • Training:            docs/README_Etapa5_Antrenare_RN.md
  • Architecture:        docs/README_Etapa4_Arhitectura_SIA.md
  • Web UI:              docs/README_Module3_WebUI.md

🔗 MODEL METRICS:
  • Accuracy:    73.33%
  • Recall:      100%
  • Precision:   68.75%
  • AUC:         0.81
  • F1-Score:    0.79

═══════════════════════════════════════════════════════════════════
"""
    print(info)


def main():
    """Main entry point"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Melanom AI - Diagnostic Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --check              Check dependencies
  python main.py --debug              Run with debug logging
  python main.py --train              Launch training module
  python main.py                      Launch web application (default)
        """
    )
    
    parser.add_argument('--check', action='store_true',
                       help='Check all requirements')
    parser.add_argument('--debug', action='store_true',
                       help='Run with debug logging')
    parser.add_argument('--train', action='store_true',
                       help='Launch training module')
    parser.add_argument('--info', action='store_true',
                       help='Print project information')
    
    args = parser.parse_args()
    
    # Print project info
    if args.info:
        print_project_info()
        return 0
    
    # Check requirements
    if not check_requirements():
        if not args.check:
            logger.error("[ERROR] Requirements check failed!")
            logger.error("Run 'python main.py --check' for details")
        return 1
    
    if args.check:
        logger.info("[SUCCESS] All requirements are satisfied!")
        return 0
    
    # Launch appropriate module
    try:
        if args.train:
            success = launch_training()
        else:
            success = launch_streamlit_app(debug=args.debug)
        
        return 0 if success else 1
    
    except KeyboardInterrupt:
        logger.info("\n[INTERRUPT] Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
