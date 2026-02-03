#!/usr/bin/env python3
"""
Single Model Image Prediction Script
Select a specific model and predict if an image is real or fake
"""

import os
import sys
import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

# Add Pretrained_Models to path
sys.path.append('Pretrained_Models')

from config import *

def get_model_info():
    """Get model information including actual trained accuracy"""
    return {
        "VGG16": {"accuracy": 85.56, "description": "Deep CNN with 16 layers", "file": "saved_models/vgg16_final_model.h5"},
        "VGG19": {"accuracy": 85.58, "description": "Deep CNN with 19 layers", "file": "saved_models/vgg19_final_model.h5"},
        "ResNet50": {"accuracy": 50.00, "description": "Residual network with 50 layers", "file": "saved_models/resnet50_final_model.h5"},
        "InceptionV3": {"accuracy": 77.39, "description": "Inception architecture for efficient recognition", "file": "saved_models/inceptionv3_final_model.h5"},
        "Custom": {"accuracy": 95.02, "description": "Custom CNN with 9 layers", "file": "saved_models/custom_final_model.h5"}
    }

def preprocess_image(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """Preprocess image for prediction"""
    try:
        # Load and resize image
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def predict_with_model(model_path, model_name, image_array):
    """Predict using a single model"""
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Make prediction
        prediction = model.predict(image_array, verbose=0)
        confidence = float(prediction[0][0])
        
        # Convert to binary prediction
        is_fake = confidence > 0.5
        result = "FAKE" if is_fake else "REAL"
        
        return {
            'model': model_name,
            'prediction': result,
            'confidence': confidence,
            'fake_probability': confidence,
            'real_probability': 1 - confidence
        }
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")
        return None

def main():
    model_info = get_model_info()
    
    parser = argparse.ArgumentParser(description='Predict if an image is real or fake using a specific model')
    parser.add_argument('image_path', nargs='?', help='Path to the image file')
    parser.add_argument('--model', choices=list(model_info.keys()), default='VGG16', 
                       help='Model to use for prediction')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    args = parser.parse_args()
    
    if args.list_models:
        print("🤖 Available Models:")
        print("=" * 50)
        for model_name, info in model_info.items():
            print(f"• {model_name}: {info['accuracy']:.2f}% accuracy")
            print(f"  Description: {info['description']}")
            print(f"  File: {info['file']}")
            print()
        return
    
    if not args.image_path:
        print("❌ Please provide an image path or use --list-models to see available models")
        parser.print_help()
        return
    
    print("🔍 DeepFake Image Detection - Single Model")
    print("=" * 50)
    
    # Set TensorFlow to use CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"❌ Image not found: {args.image_path}")
        return
    
    # Get model info
    if args.model not in model_info:
        print(f"❌ Unknown model: {args.model}")
        print("Available models:", list(model_info.keys()))
        return
    
    info = model_info[args.model]
    model_path = info['file']
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Display model information
    print(f"🤖 Selected Model: {args.model}")
    print(f"📊 Model Accuracy: {info['accuracy']:.2f}%")
    print(f"📝 Description: {info['description']}")
    
    # Get model file size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"💾 Model Size: {size_mb:.1f} MB")
    print()
    
    # Preprocess image
    print(f"📸 Loading image: {args.image_path}")
    image_array = preprocess_image(args.image_path)
    if image_array is None:
        return
    
    print(f"✅ Image preprocessed: {IMG_WIDTH}x{IMG_HEIGHT}")
    
    # Make prediction
    print(f"\n🔍 Making prediction with {args.model}...")
    result = predict_with_model(model_path, args.model, image_array)
    
    if result is None:
        print("❌ Prediction failed")
        return
    
    # Display results
    print("\n" + "=" * 50)
    print("📊 PREDICTION RESULTS")
    print("=" * 50)
    
    print(f"Model: {result['model']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Fake Probability: {result['fake_probability']:.3f}")
    print(f"Real Probability: {result['real_probability']:.3f}")
    
    # Visual representation
    print("\n📊 Confidence Visualization:")
    fake_bars = int(result['fake_probability'] * 20)
    real_bars = int(result['real_probability'] * 20)
    
    print(f"Fake:  {'█' * fake_bars}{'░' * (20 - fake_bars)} {result['fake_probability']:.1%}")
    print(f"Real:  {'█' * real_bars}{'░' * (20 - real_bars)} {result['real_probability']:.1%}")
    
    # Final verdict
    print("\n" + "=" * 50)
    print("🎯 FINAL VERDICT")
    print("=" * 50)
    
    if result['prediction'] == 'FAKE':
        print("⚠️  This image appears to be a DEEPFAKE!")
        if result['confidence'] > 0.8:
            print("🔴 High confidence - very likely fake")
        elif result['confidence'] > 0.6:
            print("🟡 Medium confidence - probably fake")
        else:
            print("🟠 Low confidence - uncertain result")
    else:
        print("✅ This image appears to be REAL!")
        if result['confidence'] > 0.8:
            print("🟢 High confidence - very likely real")
        elif result['confidence'] > 0.6:
            print("🟡 Medium confidence - probably real")
        else:
            print("🟠 Low confidence - uncertain result")
    
    print(f"\n💡 Tip: Try different models for comparison: --model VGG19, ResNet50, etc.")

if __name__ == "__main__":
    main()