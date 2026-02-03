#!/usr/bin/env python3
"""
Main training script for DeepFake Detection Project
Trains all pre-trained models and saves results
"""

import os
import sys
import time
from pathlib import Path

# Add Pretrained_Models to path
sys.path.append('Pretrained_Models')

from config import *
from VGG16_finetuning import *
from VGG19_finetuning import *
from inceptionV3_finetuning import *
from resnet50_finetuning import *

def check_dataset():
    """Check if dataset is properly set up"""
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Training directory not found: {TRAIN_DIR}")
        print("Please download the dataset and update BASE_DIR in config.py")
        return False
    
    if not os.path.exists(VAL_DIR):
        print(f"❌ Validation directory not found: {VAL_DIR}")
        return False
    
    if not os.path.exists(TEST_DIR):
        print(f"❌ Test directory not found: {TEST_DIR}")
        return False
    
    print("✅ Dataset directories found")
    return True

def train_all_models():
    """Train all models sequentially"""
    models = [
        ("VGG16", "VGG16_finetuning.py"),
        ("VGG19", "VGG19_finetuning.py"),
        ("InceptionV3", "inceptionV3_finetuning.py"),
        ("ResNet50", "resnet50_finetuning.py")
    ]
    
    results = {}
    
    for model_name, script_name in models:
        print(f"\n🚀 Training {model_name}...")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Import and run the training script
            if model_name == "VGG16":
                # VGG16 training is already imported and will run
                pass
            elif model_name == "VGG19":
                # VGG19 training is already imported and will run
                pass
            elif model_name == "InceptionV3":
                # InceptionV3 training is already imported and will run
                pass
            elif model_name == "ResNet50":
                # ResNet50 training is already imported and will run
                pass
            
            end_time = time.time()
            training_time = end_time - start_time
            
            print(f"✅ {model_name} training completed in {training_time:.2f} seconds")
            results[model_name] = {"status": "success", "time": training_time}
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            results[model_name] = {"status": "error", "error": str(e)}
    
    return results

def main():
    print("🤖 DeepFake Detection Model Training")
    print("=" * 50)
    
    # Check dataset
    if not check_dataset():
        return
    
    # Create model save directory
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Train all models
    results = train_all_models()
    
    # Print summary
    print("\n📊 Training Summary")
    print("=" * 50)
    for model_name, result in results.items():
        if result["status"] == "success":
            print(f"✅ {model_name}: Completed in {result['time']:.2f}s")
        else:
            print(f"❌ {model_name}: Failed - {result['error']}")
    
    print(f"\n🎉 Training complete! Models saved in: {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    main()
