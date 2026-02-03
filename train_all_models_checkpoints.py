#!/usr/bin/env python3
"""
Train all models with checkpoint saving
VGG16, VGG19, ResNet50, InceptionV3, and Custom Model
"""

import os
import sys
import time
import tensorflow as tf
from pathlib import Path
from argparse import ArgumentParser

# Add Pretrained_Models to path
sys.path.append('Pretrained_Models')

from config import *
from data_preprocessing import data_preprocessing
from plot_loss_accuracy_graph import plot_history
from load_pre_trained_models import load_pretrained_models

def create_custom_model():
    """Create custom CNN model"""
    print("🏗️  Creating custom CNN model...")
    
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

def train_model_with_checkpoints(model_name, model, train_generator, val_generator, test_generator):
    """Train a model with checkpoint saving"""
    print(f"\n🚀 Training {model_name} with Checkpoints")
    print("=" * 50)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n📋 {model_name} Summary:")
    model.summary()
    
    # Create checkpoint callback
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"{model_name.lower()}_checkpoint_epoch_{{epoch:02d}}_val_acc_{{val_accuracy:.4f}}.h5")
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
    
    try:
        # Train model
        print(f"\n🏃 Training {model_name} for {EPOCHS} epochs...")
        print("💾 Checkpoints will be saved after each epoch")
        print("⏹️  Press Ctrl+C to stop training and save current model")
        
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
        print(f"\n📊 Evaluating {model_name}...")
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"🎯 {model_name} Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"🎯 {model_name} Test Loss: {test_loss:.4f}")
        
        # Save final model
        final_model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name.lower()}_final_model.h5")
        model.save(final_model_path)
        print(f"💾 {model_name} final model saved to: {final_model_path}")
        
        # Plot results
        print(f"\n📈 Plotting {model_name} training results...")
        plot_history(history, f"{model_name}_Checkpoint")
        
        return True, test_accuracy
        
    except KeyboardInterrupt:
        print(f"\n⏹️  {model_name} training interrupted by user")
        
        # Save current model
        interrupted_model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name.lower()}_interrupted_model.h5")
        model.save(interrupted_model_path)
        print(f"💾 {model_name} model saved at interruption: {interrupted_model_path}")
        
        return True, 0.0

def train_all_models():
    """Train all models with checkpoint saving"""
    print("🤖 Training All Models with Checkpoint Saving")
    print("=" * 60)
    print(f"📐 Image Size: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🔄 Epochs: {EPOCHS}")
    print(f"🧠 Learning Rate: {LEARNING_RATE}")
    print("=" * 60)
    
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
    
    # Load pretrained models
    print("🏗️  Loading pretrained models...")
    vgg16_base, vgg19_base, inception_base, resnet50_base = load_pretrained_models()
    
    # Define models to train
    models_to_train = []
    
    # VGG16
    for layer in vgg16_base.layers[:-5]:
        layer.trainable = False
    for layer in vgg16_base.layers[-5:]:
        layer.trainable = True
    
    x = vgg16_base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    vgg16_model = tf.keras.Model(inputs=vgg16_base.input, outputs=predictions)
    models_to_train.append(("VGG16", vgg16_model))
    
    # VGG19
    for layer in vgg19_base.layers[:-5]:
        layer.trainable = False
    for layer in vgg19_base.layers[-5:]:
        layer.trainable = True
    
    x = vgg19_base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    vgg19_model = tf.keras.Model(inputs=vgg19_base.input, outputs=predictions)
    models_to_train.append(("VGG19", vgg19_model))
    
    # ResNet50
    for layer in resnet50_base.layers[:-14]:
        layer.trainable = False
    for layer in resnet50_base.layers[-14:]:
        layer.trainable = True
    
    x = resnet50_base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    resnet50_model = tf.keras.Model(inputs=resnet50_base.input, outputs=predictions)
    models_to_train.append(("ResNet50", resnet50_model))
    
    # InceptionV3
    for layer in inception_base.layers[:-5]:
        layer.trainable = False
    for layer in inception_base.layers[-5:]:
        layer.trainable = True
    
    x = inception_base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    inception_model = tf.keras.Model(inputs=inception_base.input, outputs=predictions)
    models_to_train.append(("InceptionV3", inception_model))
    
    # Custom Model
    custom_model = create_custom_model()
    models_to_train.append(("Custom", custom_model))
    
    # Train each model
    results = {}
    successful_models = []
    
    for model_name, model in models_to_train:
        print(f"\n{'='*60}")
        print(f"🚀 Starting {model_name} Training")
        print(f"{'='*60}")
        
        try:
            success, accuracy = train_model_with_checkpoints(
                model_name, model, train_generator, val_generator, test_generator
            )
            
            if success:
                results[model_name] = accuracy
                successful_models.append(model_name)
                print(f"✅ {model_name} training completed successfully!")
            else:
                print(f"❌ {model_name} training failed!")
                
        except Exception as e:
            print(f"❌ {model_name} training error: {str(e)}")
        
        # Small delay between models
        time.sleep(5)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    
    if successful_models:
        print(f"✅ Successfully trained: {', '.join(successful_models)}")
        print(f"📁 Models saved in: {MODEL_SAVE_DIR}")
        
        # List saved models
        print("\n📋 Saved model files:")
        for model_name in successful_models:
            model_file = f"{MODEL_SAVE_DIR}{model_name.lower()}_final_model.h5"
            if os.path.exists(model_file):
                size_mb = os.path.getsize(model_file) / (1024 * 1024)
                print(f"   {model_file} ({size_mb:.1f} MB)")
    
    print(f"\n🎉 Training completed! {len(successful_models)}/{len(models_to_train)} models trained successfully")
    
    return len(successful_models) > 0

def main():
    print("🤖 DeepFake Detection - Training All Models with Checkpoints")
    print("=" * 70)
    print("Training: VGG16, VGG19, ResNet50, InceptionV3, Custom Model")
    print("Expected accuracies: 95.27%, 95.21%, 94.00%, 77.20%, 95.02%")
    print("=" * 70)
    
    parser = ArgumentParser(description="Train models with checkpoint saving")
    parser.add_argument("--only", nargs="*", default=[], help="Subset of models to train: vgg16 vgg19 resnet50 inception custom")
    args = parser.parse_args()

    # Normalize names
    only = set([name.lower() for name in args.only])

    success = train_all_models() if not only else _train_subset(only)
    
    if success:
        print("\n🎉 All training completed successfully!")
        print("\n📁 Check the 'saved_models/' directory for your trained models")
        print("📈 Training plots should be displayed")
    else:
        print("\n❌ Training failed. Check the error messages above.")


def _train_subset(only):
    # Set TensorFlow to use CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Training directory not found: {TRAIN_DIR}")
        return False

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    train_generator, val_generator, test_generator = data_preprocessing()

    print("🏗️  Loading pretrained backbones for subset...")
    vgg16_base, vgg19_base, inception_base, resnet50_base = load_pretrained_models()

    subset = []
    if 'inception' in only:
        for layer in inception_base.layers[:-5]:
            layer.trainable = False
        for layer in inception_base.layers[-5:]:
            layer.trainable = True
        x = inception_base.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        inception_model = tf.keras.Model(inputs=inception_base.input, outputs=predictions)
        subset.append(("InceptionV3", inception_model))

    if 'custom' in only:
        subset.append(("Custom", create_custom_model()))

    if not subset:
        print("⚠️  No valid models selected in --only")
        return False

    ok = True
    for name, model in subset:
        try:
            success, _ = train_model_with_checkpoints(name, model, train_generator, val_generator, test_generator)
            ok = ok and success
        except Exception as e:
            print(f"❌ {name} training error: {e}")
            ok = False
    return ok

if __name__ == "__main__":
    main()