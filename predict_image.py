#!/usr/bin/env python3
"""
Image Prediction Script
Upload an image and predict if it's real or fake using all trained models
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

def ensemble_prediction(results):
    """Create ensemble prediction from all models"""
    if not results:
        return None
    
    # Weighted average (you can adjust weights based on model performance)
    weights = {
        'VGG16': 0.25,
        'VGG19': 0.25,
        'ResNet50': 0.20,
        'InceptionV3': 0.15,
        'Custom Model': 0.15
    }
    
    weighted_sum = 0
    total_weight = 0
    
    for result in results:
        if result:
            weight = weights.get(result['model'], 0.2)
            weighted_sum += result['fake_probability'] * weight
            total_weight += weight
    
    if total_weight == 0:
        return None
    
    ensemble_confidence = weighted_sum / total_weight
    is_fake = ensemble_confidence > 0.5
    result = "FAKE" if is_fake else "REAL"
    
    return {
        'prediction': result,
        'confidence': ensemble_confidence,
        'fake_probability': ensemble_confidence,
        'real_probability': 1 - ensemble_confidence
    }

def main():
    parser = argparse.ArgumentParser(description='Predict if an image is real or fake')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble prediction')
    args = parser.parse_args()
    
    print("🔍 DeepFake Image Detection")
    print("=" * 50)
    
    # Set TensorFlow to use CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"❌ Image not found: {args.image_path}")
        return
    
    # Preprocess image
    print(f"📸 Loading image: {args.image_path}")
    image_array = preprocess_image(args.image_path)
    if image_array is None:
        return
    
    print(f"✅ Image preprocessed: {IMG_WIDTH}x{IMG_HEIGHT}")
    
    # Model files
    models = [
        ("VGG16", "saved_models/vgg16_final_model.h5"),
        ("VGG19", "saved_models/vgg19_final_model.h5"),
        ("ResNet50", "saved_models/resnet50_final_model.h5"),
        ("InceptionV3", "saved_models/inceptionv3_final_model.h5"),
        ("Custom Model", "saved_models/custom_final_model.h5")
    ]
    
    # Predict with each model
    print("\n🤖 Making predictions...")
    results = []
    
    for model_name, model_path in models:
        if os.path.exists(model_path):
            result = predict_with_model(model_path, model_name, image_array)
            if result:
                results.append(result)
                print(f"✅ {model_name}: {result['prediction']} (Confidence: {result['confidence']:.3f})")
        else:
            print(f"⚠️  {model_name}: Model not found")
    
    if not results:
        print("❌ No models available for prediction")
        return
    
    # Display individual results
    print("\n" + "=" * 60)
    print("📊 INDIVIDUAL MODEL PREDICTIONS")
    print("=" * 60)
    print(f"{'Model':<15} {'Prediction':<10} {'Fake Prob':<12} {'Real Prob':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['model']:<15} {result['prediction']:<10} {result['fake_probability']:<12.3f} {result['real_probability']:<12.3f}")
    
    # Ensemble prediction
    if args.ensemble and len(results) > 1:
        ensemble_result = ensemble_prediction(results)
        if ensemble_result:
            print("\n" + "=" * 60)
            print("🎯 ENSEMBLE PREDICTION")
            print("=" * 60)
            print(f"Final Prediction: {ensemble_result['prediction']}")
            print(f"Confidence: {ensemble_result['confidence']:.3f}")
            print(f"Fake Probability: {ensemble_result['fake_probability']:.3f}")
            print(f"Real Probability: {ensemble_result['real_probability']:.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    fake_votes = sum(1 for r in results if r['prediction'] == 'FAKE')
    real_votes = len(results) - fake_votes
    
    print(f"Models tested: {len(results)}")
    print(f"FAKE votes: {fake_votes}")
    print(f"REAL votes: {real_votes}")
    
    if fake_votes > real_votes:
        print("🎯 Majority prediction: FAKE")
    elif real_votes > fake_votes:
        print("🎯 Majority prediction: REAL")
    else:
        print("🎯 Majority prediction: TIE")
    
    print(f"\n💡 Tip: Use --ensemble flag for weighted average prediction")

if __name__ == "__main__":
    main()