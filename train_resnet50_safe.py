#!/usr/bin/env python3
"""
Safe ResNet50 Training Script - macOS Threading Safe
"""

import os
import sys

# Set threading safety BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Add path for imports
sys.path.append('Pretrained_Models')

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    
    # Configure TensorFlow threading
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    print("🚀 Starting ResNet50 training...")
    print("Expected accuracy: 94.00%")
    
    # Import training components
    from load_pre_trained_models import load_pretrained_models
    from data_preprocessing import data_preprocessing
    from plot_loss_accuracy_graph import plot_history
    from config import RESNET50_MODEL_PATH
    
    # Load pre-trained ResNet50
    resnet50_base = load_pretrained_models()[3]
    train_generator, val_generator, test_generator = data_preprocessing()
    
    # Freeze initial layers, unfreeze last 14
    for layer in resnet50_base.layers[:-14]:
        layer.trainable = False
    
    for layer in resnet50_base.layers[-14:]:
        layer.trainable = True
    
    print("✅ ResNet50 base model loaded and configured")
    
    # Add custom layers
    x = resnet50_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    # Create final model
    model = Model(inputs=resnet50_base.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model compiled, starting training...")
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        epochs=3,  # Reduced for faster training
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"✅ ResNet50 training completed!")
    print(f"📊 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"📊 Test Loss: {test_loss:.4f}")
    
    # Save model
    model.save(RESNET50_MODEL_PATH)
    print(f"💾 Model saved to: {RESNET50_MODEL_PATH}")
    
    # Plot training history
    plot_history(history, "ResNet50")
    
    print("🎉 ResNet50 training completed successfully!")
    
except Exception as e:
    print(f"❌ ResNet50 training failed: {str(e)}")
    sys.exit(1)
