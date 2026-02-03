#!/usr/bin/env python3
"""
Quick Training Script for CPU
Optimized for fast training with reduced parameters
"""

import os
import sys
import time
import tensorflow as tf
from pathlib import Path

# Add Pretrained_Models to path
sys.path.append('Pretrained_Models')

from config import *
from data_preprocessing import data_preprocessing
from plot_loss_accuracy_graph import plot_history

def create_lightweight_model():
    """Create a lightweight model optimized for CPU training"""
    print("🏗️  Creating lightweight model...")
    
    model = tf.keras.Sequential([
        # First block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Second block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Third block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Fourth block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def train_lightweight_model():
    """Train the lightweight model"""
    print("🚀 Starting Quick Training (CPU Optimized)")
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
    
    # Create model
    model = create_lightweight_model()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Train model
    print(f"\n🏃 Training for {EPOCHS} epochs...")
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"🎯 Test Loss: {test_loss:.4f}")
    
    # Plot results
    print("\n📈 Plotting training results...")
    plot_history(history, "Lightweight_CPU_Model")
    
    # Save model
    model_path = os.path.join(MODEL_SAVE_DIR, "lightweight_cpu_model.h5")
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}")
    
    return True

def main():
    print("🤖 Quick DeepFake Detection Training (CPU Optimized)")
    print("=" * 60)
    print(f"📐 Image Size: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🔄 Epochs: {EPOCHS}")
    print(f"🧠 Learning Rate: {LEARNING_RATE}")
    print("=" * 60)
    
    # Set TensorFlow to use CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Train model
    success = train_lightweight_model()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("\n📁 Check the 'saved_models/' directory for your trained model")
        print("📈 Training plots should be displayed")
    else:
        print("\n❌ Training failed. Check the error messages above.")

if __name__ == "__main__":
    main()
