#!/usr/bin/env python3
"""
Ensemble Model for DeepFake Detection
Combines multiple models for improved accuracy
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add Pretrained_Models to path
sys.path.append('Pretrained_Models')

from config import *
from data_preprocessing import data_preprocessing
from plot_loss_accuracy_graph import plot_history

def create_ensemble_models():
    """Create multiple models for ensemble"""
    print("🏗️  Creating ensemble models...")
    
    models = []
    
    # Model 1: Lightweight CNN
    model1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    models.append(("Lightweight_CNN", model1))
    
    # Model 2: Deeper CNN
    model2 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    models.append(("Deeper_CNN", model2))
    
    # Model 3: Wide CNN
    model3 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    models.append(("Wide_CNN", model3))
    
    # Model 4: Hybrid CNN
    model4 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    models.append(("Hybrid_CNN", model4))
    
    return models

def train_ensemble_models():
    """Train all ensemble models"""
    print("🚀 Starting Ensemble Training")
    print("=" * 50)
    
    # Check dataset
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Training directory not found: {TRAIN_DIR}")
        return False
    
    # Create model save directory
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Get data generators
    print("📊 Loading dataset...")
    train_generator, val_generator, test_generator = data_preprocessing()
    
    print(f"✅ Training samples: {train_generator.samples:,}")
    print(f"✅ Validation samples: {val_generator.samples:,}")
    print(f"✅ Test samples: {test_generator.samples:,}")
    
    # Create models
    models = create_ensemble_models()
    
    # Train each model
    trained_models = []
    model_results = []
    
    for i, (name, model) in enumerate(models):
        print(f"\n🏃 Training {name} ({i+1}/{len(models)})...")
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            validation_data=val_generator,
            validation_steps=val_generator.samples // BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"🎯 {name} - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Save model
        model_path = os.path.join(MODEL_SAVE_DIR, f"{name.lower()}_model.h5")
        model.save(model_path)
        print(f"💾 {name} saved to: {model_path}")
        
        trained_models.append((name, model))
        model_results.append((name, test_accuracy, history))
    
    return trained_models, model_results, test_generator

def create_ensemble_predictions(trained_models, test_generator):
    """Create ensemble predictions"""
    print("\n🔮 Creating ensemble predictions...")
    
    # Get test data
    test_generator.reset()
    predictions = []
    true_labels = []
    
    # Collect predictions from all models
    model_predictions = []
    
    for name, model in trained_models:
        print(f"📊 Getting predictions from {name}...")
        test_generator.reset()
        pred = model.predict(test_generator)
        model_predictions.append(pred)
    
    # Get true labels
    test_generator.reset()
    for i in range(len(test_generator)):
        batch_x, batch_y = test_generator[i]
        true_labels.extend(batch_y)
        if i >= len(test_generator) - 1:
            break
    
    # Ensemble predictions (weighted average)
    ensemble_pred = np.mean(model_predictions, axis=0)
    
    # Convert to binary predictions
    ensemble_binary = (ensemble_pred > 0.5).astype(int)
    true_labels = np.array(true_labels[:len(ensemble_binary)])
    
    # Calculate ensemble accuracy
    ensemble_accuracy = np.mean(ensemble_binary.flatten() == true_labels)
    
    print(f"🎯 Ensemble Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
    
    return ensemble_accuracy, model_predictions, true_labels

def main():
    print("🤖 Ensemble DeepFake Detection Training")
    print("=" * 60)
    print(f"📐 Image Size: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🔄 Epochs: {EPOCHS}")
    print(f"🧠 Learning Rate: {LEARNING_RATE}")
    print("=" * 60)
    
    # Set TensorFlow to use CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Train ensemble models
    trained_models, model_results, test_generator = train_ensemble_models()
    
    if trained_models:
        # Create ensemble predictions
        ensemble_accuracy, model_predictions, true_labels = create_ensemble_predictions(trained_models, test_generator)
        
        print("\n📊 Final Results:")
        print("=" * 40)
        for name, accuracy, history in model_results:
            print(f"{name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Ensemble: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
        
        print("\n🎉 Ensemble training completed successfully!")
        print("📁 Check the 'saved_models/' directory for your trained models")
    else:
        print("\n❌ Ensemble training failed. Check the error messages above.")

if __name__ == "__main__":
    main()
