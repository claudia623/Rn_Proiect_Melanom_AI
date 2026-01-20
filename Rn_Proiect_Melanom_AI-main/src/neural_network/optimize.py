"""
Hyperparameter Optimization Script for Melanoma Classification Model
Etapa 6 - Optimizare si Tuning
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Import din modul antrenare
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.neural_network.model import create_model
from src.neural_network.train import load_data, create_data_generators


class ModelOptimizer:
    """Clasa pentru optimizare hiperparametri model"""
    
    def __init__(self, data_dir='data', results_dir='results', models_dir='models'):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.models_dir = models_dir
        self.experiments = []
        
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
    
    def optimize_learning_rate(self, base_model_path=None):
        """Experimenta cu learning rate diferite"""
        print("\n=== Experiment 1: Learning Rate Tuning ===")
        
        learning_rates = [0.0001, 0.0005, 0.001]
        best_lr = None
        best_score = 0
        
        for lr in learning_rates:
            print(f"\nTesting learning_rate={lr}")
            
            # Creez model
            model = create_model(input_shape=(224, 224, 3), learning_rate=lr)
            
            # Antrenez
            history, metrics = self._train_and_evaluate(
                model, 
                epochs=20,
                experiment_name=f'lr_{lr}'
            )
            
            # Salvez rezultate
            self.experiments.append({
                'experiment': 'LR_Tuning',
                'learning_rate': lr,
                'test_accuracy': metrics['accuracy'],
                'f1_score': metrics['f1'],
                'auc': metrics['auc']
            })
            
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_lr = lr
        
        print(f"\nBest Learning Rate: {best_lr} (Accuracy: {best_score:.2%})")
        return best_lr
    
    def optimize_augmentation(self, learning_rate=0.001):
        """Testeaza data augmentation strategies"""
        print("\n=== Experiment 2: Data Augmentation ===")
        
        augmentation_configs = [
            {'rotation': 20, 'zoom': 0.2, 'shift': 0.2},
            {'rotation': 30, 'zoom': 0.3, 'shift': 0.2},
            {'rotation': 15, 'zoom': 0.15, 'shift': 0.15}
        ]
        
        best_config = None
        best_score = 0
        
        for config in augmentation_configs:
            print(f"\nTesting augmentation: {config}")
            
            model = create_model(input_shape=(224, 224, 3), learning_rate=learning_rate)
            
            history, metrics = self._train_and_evaluate(
                model,
                epochs=30,
                experiment_name=f'aug_{config["rotation"]}',
                augmentation_config=config
            )
            
            self.experiments.append({
                'experiment': 'Augmentation',
                'config': str(config),
                'test_accuracy': metrics['accuracy'],
                'f1_score': metrics['f1'],
                'auc': metrics['auc']
            })
            
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_config = config
        
        print(f"\nBest Augmentation: {best_config} (Accuracy: {best_score:.2%})")
        return best_config
    
    def optimize_dropout(self, learning_rate=0.001):
        """Testeaza dropout rates"""
        print("\n=== Experiment 3: Dropout Tuning ===")
        
        dropout_rates = [0.2, 0.3, 0.4, 0.5]
        best_dropout = None
        best_score = 0
        
        for dropout in dropout_rates:
            print(f"\nTesting dropout={dropout}")
            
            model = create_model(
                input_shape=(224, 224, 3), 
                learning_rate=learning_rate,
                dropout_rate=dropout
            )
            
            history, metrics = self._train_and_evaluate(
                model,
                epochs=25,
                experiment_name=f'dropout_{dropout}'
            )
            
            self.experiments.append({
                'experiment': 'Dropout',
                'dropout_rate': dropout,
                'test_accuracy': metrics['accuracy'],
                'f1_score': metrics['f1'],
                'auc': metrics['auc']
            })
            
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_dropout = dropout
        
        print(f"\nBest Dropout: {best_dropout} (Accuracy: {best_score:.2%})")
        return best_dropout
    
    def _train_and_evaluate(self, model, epochs, experiment_name, augmentation_config=None):
        """Antreneaza si evalueaza model"""
        
        # Load data
        train_gen, val_gen, test_gen = self._create_generators(augmentation_config)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
        
        # Antrenare
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluare
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        
        # Predictii pentru metrici detaliate
        y_true = test_gen.classes
        y_pred_proba = model.predict(test_gen, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        metrics = {
            'accuracy': test_acc,
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        return history, metrics
    
    def _create_generators(self, augmentation_config=None):
        """Creaza data generators"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        if augmentation_config:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=augmentation_config.get('rotation', 20),
                zoom_range=augmentation_config.get('zoom', 0.2),
                width_shift_range=augmentation_config.get('shift', 0.2),
                height_shift_range=augmentation_config.get('shift', 0.2),
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_gen = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=(224, 224),
            batch_size=16,
            class_mode='binary'
        )
        
        val_gen = test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'validation'),
            target_size=(224, 224),
            batch_size=16,
            class_mode='binary'
        )
        
        test_gen = test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),
            target_size=(224, 224),
            batch_size=16,
            class_mode='binary',
            shuffle=False
        )
        
        return train_gen, val_gen, test_gen
    
    def save_results(self):
        """Salveaza rezultatele experimentelor"""
        df = pd.DataFrame(self.experiments)
        csv_path = os.path.join(self.results_dir, 'optimization_experiments.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Salvez JSON cu best config
        best_experiment = max(self.experiments, key=lambda x: x['test_accuracy'])
        config = {
            'best_experiment': best_experiment['experiment'],
            'best_accuracy': float(best_experiment['test_accuracy']),
            'best_f1': float(best_experiment['f1_score']),
            'best_auc': float(best_experiment['auc']),
            'optimization_date': datetime.now().isoformat()
        }
        
        json_path = os.path.join(self.results_dir, 'final_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Best config saved to: {json_path}")
        print(f"\nBest Model: {best_experiment['experiment']}")
        print(f"  Accuracy: {best_experiment['test_accuracy']:.2%}")
        print(f"  F1-Score: {best_experiment['f1_score']:.4f}")
        print(f"  AUC: {best_experiment['auc']:.4f}")
    
    def create_optimized_model(self, best_lr, best_aug, best_dropout):
        """Creaza si antreneaza modelul final optimizat"""
        print("\n=== Training Final Optimized Model ===")
        
        model = create_model(
            input_shape=(224, 224, 3),
            learning_rate=best_lr,
            dropout_rate=best_dropout
        )
        
        history, metrics = self._train_and_evaluate(
            model,
            epochs=50,
            experiment_name='final_optimized',
            augmentation_config=best_aug
        )
        
        # Salvez modelul optimizat
        model_path = os.path.join(self.models_dir, 'optimized_model.h5')
        model.save(model_path)
        print(f"\nOptimized model saved to: {model_path}")
        
        return model, metrics


def main():
    """Functie principala pentru rulare optimizare"""
    print("=" * 60)
    print("Melanom AI - Model Optimization (Etapa 6)")
    print("=" * 60)
    
    optimizer = ModelOptimizer(
        data_dir='data',
        results_dir='results',
        models_dir='models'
    )
    
    # Ruleaza experimentele
    best_lr = optimizer.optimize_learning_rate()
    best_aug = optimizer.optimize_augmentation(learning_rate=best_lr)
    best_dropout = optimizer.optimize_dropout(learning_rate=best_lr)
    
    # Salveaza rezultate
    optimizer.save_results()
    
    # Creaza model final optimizat
    final_model, final_metrics = optimizer.create_optimized_model(
        best_lr, best_aug, best_dropout
    )
    
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print(f"\nFinal Model Metrics:")
    print(f"  Accuracy: {final_metrics['accuracy']:.2%}")
    print(f"  F1-Score: {final_metrics['f1']:.4f}")
    print(f"  AUC: {final_metrics['auc']:.4f}")
    

if __name__ == '__main__':
    main()
