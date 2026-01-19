.PHONY: help install setup preprocess split train evaluate clean visualize

# Project directories
PYTHON := python3
PIP := pip3
SRC_DIR := src
DATA_DIR := data
MODELS_DIR := models

# Help target
help:
	@echo "🧬 Melanoma Detection AI - Build System"
	@echo "========================================"
	@echo ""
	@echo "Available targets:"
	@echo "  make install       - Install Python dependencies"
	@echo "  make setup         - Setup project structure and environment"
	@echo "  make preprocess    - Preprocess raw images"
	@echo "  make split         - Split data into train/val/test"
	@echo "  make train         - Train the model"
	@echo "  make evaluate      - Evaluate model on test set"
	@echo "  make visualize     - Generate visualizations"
	@echo "  make clean         - Clean generated files"
	@echo "  make full-pipeline - Run complete pipeline"
	@echo "  make help          - Show this help message"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

# Setup project
setup:
	@echo "🔧 Setting up project structure..."
	@mkdir -p $(DATA_DIR)/{raw,processed,train,validation,test,generated}/benign
	@mkdir -p $(DATA_DIR)/{raw,processed,train,validation,test,generated}/malignant
	@mkdir -p $(MODELS_DIR)
	@mkdir -p logs results docs/error_analysis
	@echo "✅ Project structure created"

# Preprocess raw images
preprocess:
	@echo "🔄 Preprocessing raw images..."
	@$(PYTHON) $(SRC_DIR)/preprocessing/preprocess_dataset.py
	@echo "✅ Raw images preprocessed"

# Split data
split:
	@echo "✂️  Splitting data into train/validation/test..."
	@$(PYTHON) $(SRC_DIR)/preprocessing/split_processed_data.py
	@echo "✅ Data split completed"

# Preprocess test data
preprocess-test:
	@echo "🔄 Preprocessing test data..."
	@$(PYTHON) $(SRC_DIR)/preprocessing/preprocess_test_data.py
	@echo "✅ Test data preprocessed"

# Train model
train:
	@echo "🧠 Training model..."
	@$(PYTHON) $(SRC_DIR)/neural_network/train.py
	@echo "✅ Training completed"

# Evaluate model
evaluate:
	@echo "📊 Evaluating model on test set..."
	@$(PYTHON) $(SRC_DIR)/neural_network/evaluate.py --use-best
	@echo "✅ Evaluation completed"

# Generate visualizations
visualize:
	@echo "📈 Generating visualizations..."
	@$(PYTHON) docs/generate_etapa5_visualizations.py
	@echo "✅ Visualizations generated"

# Clean generated files
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf __pycache__ $(SRC_DIR)/__pycache__ $(SRC_DIR)/*/__pycache__
	@rm -rf .pytest_cache
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name ".DS_Store" -delete
	@echo "✅ Cleanup completed"

# Full pipeline
full-pipeline: setup preprocess split preprocess-test train evaluate visualize
	@echo ""
	@echo "✅ FULL PIPELINE COMPLETED!"
	@echo ""
	@echo "🎯 Results available in:"
	@echo "   - models/melanom_efficientnetb0_best.keras"
	@echo "   - results/"
	@echo "   - logs/"
	@echo ""

# Development targets
test:
	@echo "🧪 Running tests..."
	@$(PYTHON) -m pytest tests/ -v

format:
	@echo "🎨 Formatting code..."
	@$(PYTHON) -m black $(SRC_DIR)
	@$(PYTHON) -m isort $(SRC_DIR)

lint:
	@echo "🔍 Running linter..."
	@$(PYTHON) -m pylint $(SRC_DIR)

docs:
	@echo "📚 Building documentation..."
	@echo "Documentation available in docs/"

# Interactive shell
shell:
	@$(PYTHON) -i -c "from src.utils.constants import *; print('🚀 Project environment loaded')"
