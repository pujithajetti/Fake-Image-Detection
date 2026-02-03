#!/usr/bin/env python3
"""
Safe Training Script for All Models - macOS Threading Safe
Trains VGG16, VGG19, InceptionV3, ResNet50 with proper threading configuration
"""

import os
import sys
import subprocess
import time

# Set threading safety BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

def train_model_safely(model_name, script_path):
    """Train a model using external process to avoid threading issues"""
    print(f"\n🚀 Starting training for {model_name}...")
    print("=" * 60)
    
    try:
        # Create a safe training script
        safe_script = f'''
import os
import sys
import tensorflow as tf

# Set threading safety
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Configure TensorFlow threading
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Add path for imports
sys.path.append('Pretrained_Models')

# Import and run the training script
exec(open('{script_path}').read())
'''
        
        # Write safe script to temporary file
        with open('temp_train.py', 'w') as f:
            f.write(safe_script)
        
        # Run the training
        start_time = time.time()
        result = subprocess.run([sys.executable, 'temp_train.py'], 
                              capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        training_time = time.time() - start_time
        
        # Clean up
        if os.path.exists('temp_train.py'):
            os.remove('temp_train.py')
        
        if result.returncode == 0:
            print(f"✅ {model_name} training completed successfully!")
            print(f"⏱️  Training time: {training_time/60:.1f} minutes")
            return True
        else:
            print(f"❌ {model_name} training failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} training timed out (1 hour limit)")
        return False
    except Exception as e:
        print(f"❌ {model_name} training error: {str(e)}")
        return False

def check_dataset():
    """Check if dataset is available"""
    print("🔍 Checking dataset availability...")
    
    dataset_paths = [
        "archive/real_vs_fake/real-vs-fake/train",
        "archive/real_vs_fake/real-vs-fake/valid", 
        "archive/real_vs_fake/real-vs-fake/test"
    ]
    
    for path in dataset_paths:
        if not os.path.exists(path):
            print(f"❌ Dataset path not found: {path}")
            return False
        else:
            # Count files
            real_files = len([f for f in os.listdir(os.path.join(path, 'real')) if f.endswith(('.jpg', '.jpeg', '.png'))])
            fake_files = len([f for f in os.listdir(os.path.join(path, 'fake')) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"✅ {path}: {real_files} real, {fake_files} fake images")
    
    return True

def main():
    print("🕵️‍♂️ DeepFake Detection - Training All Models")
    print("=" * 60)
    print("Training VGG16, VGG19, InceptionV3, ResNet50")
    print("Expected accuracies: 95.27%, 95.21%, 77.20%, 94.00%")
    print("=" * 60)
    
    # Check dataset
    if not check_dataset():
        print("\n❌ Dataset not found! Please ensure the dataset is in the correct location.")
        print("Expected structure:")
        print("archive/real_vs_fake/real-vs-fake/")
        print("├── train/real/")
        print("├── train/fake/")
        print("├── valid/real/")
        print("├── valid/fake/")
        print("├── test/real/")
        print("└── test/fake/")
        return
    
    # Training scripts and models
    models_to_train = [
        ("VGG16", "Pretrained_Models/VGG16_finetuning.py"),
        ("VGG19", "Pretrained_Models/VGG19_finetuning.py"),
        ("InceptionV3", "Pretrained_Models/inceptionV3_finetuning.py"),
        ("ResNet50", "Pretrained_Models/resnet50_finetuning.py")
    ]
    
    # Check if training scripts exist
    for model_name, script_path in models_to_train:
        if not os.path.exists(script_path):
            print(f"❌ Training script not found: {script_path}")
            return
    
    print(f"\n📊 Found {len(models_to_train)} models to train")
    print("⚠️  Note: Training may take 2-4 hours on CPU")
    print("💡 For faster training, consider using GPU or cloud computing")
    
    # Confirm training
    response = input("\n🤔 Do you want to start training? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Training cancelled by user")
        return
    
    # Train each model
    successful_models = []
    failed_models = []
    
    for model_name, script_path in models_to_train:
        success = train_model_safely(model_name, script_path)
        if success:
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
        
        # Small delay between models
        time.sleep(5)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    
    if successful_models:
        print(f"✅ Successfully trained: {', '.join(successful_models)}")
        print(f"📁 Models saved in: saved_models/")
        
        # List saved models
        print("\n📋 Saved model files:")
        for model_name in successful_models:
            model_file = f"saved_models/fine_tuned_{model_name.lower()}_last5.h5"
            if os.path.exists(model_file):
                size_mb = os.path.getsize(model_file) / (1024 * 1024)
                print(f"   {model_file} ({size_mb:.1f} MB)")
    
    if failed_models:
        print(f"❌ Failed to train: {', '.join(failed_models)}")
        print("💡 This is likely due to macOS threading issues")
        print("🔧 Try training on Linux/Docker for better results")
    
    # Next steps
    print("\n🚀 NEXT STEPS:")
    print("1. Test the trained models")
    print("2. Create ensemble model for 96.80% accuracy")
    print("3. Update UI to include all models")
    print("4. Deploy for production use")
    
    print(f"\n🎉 Training completed! {len(successful_models)}/{len(models_to_train)} models trained successfully")

if __name__ == "__main__":
    main()
