#!/usr/bin/env python3
"""
Training script with checkpoint saving
Allows saving models at any point during training
"""

import os
import sys
import tensorflow as tf
from pathlib import Path

# Add Pretrained_Models to path
sys.path.append('Pretrained_Models')

from config import *
from data_preprocessing import data_preprocessing
from plot_loss_accuracy_graph import plot_history
from load_pre_trained_models import load_pretrained_models

def train_vgg16_with_checkpoints():
    """Train VGG16 with checkpoint saving"""
    print("🚀 Training VGG16 with Checkpoint Saving")
    print("=" * 50)
    
    # Set TensorFlow to use CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
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
    
    # Load VGG16 base model
    print("🏗️  Loading VGG16 base model...")
    vgg16_base = load_pretrained_models()[0]
    
    # Freeze initial layers, unfreeze last 5
    for layer in vgg16_base.layers[:-5]:
        layer.trainable = False
    
    for layer in vgg16_base.layers[-5:]:
        layer.trainable = True
    
    # Add custom layers
    x = vgg16_base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = tf.keras.Model(inputs=vgg16_base.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Create checkpoint callback
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, "vgg16_checkpoint_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=False,
        save_weights_only=False,
        verbose=1,
        save_freq='epoch'
    )
    
    # Create early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print(f"\n🏃 Training for {EPOCHS} epochs...")
    print("💾 Checkpoints will be saved after each epoch")
    print("⏹️  Press Ctrl+C to stop training and save current model")
    
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            validation_data=val_generator,
            validation_steps=val_generator.samples // BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[checkpoint_callback, early_stopping],
            verbose=1
        )
        
        # Evaluate model
        print("\n📊 Evaluating model...")
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"🎯 Test Loss: {test_loss:.4f}")
        
        # Save final model
        final_model_path = os.path.join(MODEL_SAVE_DIR, "vgg16_final_model.h5")
        model.save(final_model_path)
        print(f"💾 Final model saved to: {final_model_path}")
        
        # Plot results
        print("\n📈 Plotting training results...")
        plot_history(history, "VGG16_Checkpoint")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        
        # Save current model
        interrupted_model_path = os.path.join(MODEL_SAVE_DIR, "vgg16_interrupted_model.h5")
        model.save(interrupted_model_path)
        print(f"💾 Model saved at interruption: {interrupted_model_path}")
        
        return True

def main():
    print("🤖 VGG16 Training with Checkpoint Saving")
    print("=" * 60)
    print(f"📐 Image Size: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🔄 Epochs: {EPOCHS}")
    print(f"🧠 Learning Rate: {LEARNING_RATE}")
    print("=" * 60)
    
    success = train_vgg16_with_checkpoints()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("\n📁 Check the 'saved_models/' directory for your trained models")
    else:
        print("\n❌ Training failed. Check the error messages above.")

if __name__ == "__main__":
    main()