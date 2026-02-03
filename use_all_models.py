#!/usr/bin/env python3
"""
Use All Models - Programmatic Example
Shows how to use all available models for deepfake detection
"""

import os
import sys
import numpy as np
from PIL import Image
import time

# Add path for config
sys.path.append('Pretrained_Models')

def setup_threading_safety():
    """Setup threading safety for macOS"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

def load_model_safely(model_path):
    """Load model with threading safety"""
    try:
        import tensorflow as tf
        
        # Configure TensorFlow threading
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        
        # Load model with CPU device
        with tf.device('/CPU:0'):
            model = tf.keras.models.load_model(model_path)
        
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess_image(image_path):
    """Preprocess image for model input"""
    try:
        image = Image.open(image_path)
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)
        return image_batch, None
    except Exception as e:
        return None, str(e)

def predict_with_model(model, image_batch):
    """Make prediction with a model"""
    try:
        import tensorflow as tf
        with tf.device('/CPU:0'):
            prediction = model.predict(image_batch, verbose=0)
        
        score = float(prediction[0][0])
        
        if score > 0.5:
            result = "FAKE"
            confidence = score * 100
        else:
            result = "REAL"
            confidence = (1 - score) * 100
        
        return result, confidence, score, None
    except Exception as e:
        return None, None, None, str(e)

def demonstrate_all_models():
    """Demonstrate all available models"""
    print("🕵️‍♂️ DeepFake Detection - Using All Models")
    print("=" * 60)
    
    # Setup threading safety
    setup_threading_safety()
    
    # Model configurations
    model_configs = {
        "lightweight_cpu_model.h5": {
            "name": "Lightweight CPU Model",
            "description": "Fast, efficient model"
        },
        "lightweight_cnn_model.h5": {
            "name": "Lightweight CNN Model", 
            "description": "Balanced model"
        },
        "deeper_cnn_model.h5": {
            "name": "Deeper CNN Model",
            "description": "More layers for better features"
        },
        "wide_cnn_model.h5": {
            "name": "Wide CNN Model",
            "description": "More filters for comprehensive representation"
        }
    }
    
    # Check available models
    model_dir = "saved_models"
    available_models = []
    
    print("\n📊 Checking Available Models:")
    print("-" * 40)
    
    for model_file, config in model_configs.items():
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✅ {config['name']} ({size_mb:.1f} MB)")
            available_models.append((model_file, config, model_path))
        else:
            print(f"❌ {config['name']} - Not found")
    
    if not available_models:
        print("\n❌ No models available!")
        return
    
    # Find sample images
    print("\n📸 Looking for Sample Images:")
    print("-" * 40)
    
    sample_images = []
    uploads_dir = "uploads"
    
    if os.path.exists(uploads_dir):
        for file in os.listdir(uploads_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample_images.append(os.path.join(uploads_dir, file))
    
    if sample_images:
        print(f"✅ Found {len(sample_images)} sample images")
        test_image = sample_images[0]
        print(f"📷 Using: {os.path.basename(test_image)}")
    else:
        print("⚠️  No sample images found in uploads/ directory")
        print("💡 You can add test images to the uploads/ folder")
        return
    
    # Test each model
    print(f"\n🔮 Testing All Models on: {os.path.basename(test_image)}")
    print("=" * 60)
    
    # Preprocess image once
    image_batch, error = preprocess_image(test_image)
    if error:
        print(f"❌ Image preprocessing failed: {error}")
        return
    
    results = []
    
    for model_file, config, model_path in available_models:
        print(f"\n🧪 Testing {config['name']}...")
        
        # Load model
        model, error = load_model_safely(model_path)
        if error:
            print(f"❌ Failed to load: {error}")
            continue
        
        # Make prediction
        start_time = time.time()
        result, confidence, score, error = predict_with_model(model, image_batch)
        prediction_time = time.time() - start_time
        
        if error:
            print(f"❌ Prediction failed: {error}")
            continue
        
        print(f"   Result: {result}")
        print(f"   Confidence: {confidence:.2f}%")
        print(f"   Raw Score: {score:.4f}")
        print(f"   Time: {prediction_time:.3f}s")
        
        results.append({
            'model': config['name'],
            'result': result,
            'confidence': confidence,
            'score': score,
            'time': prediction_time
        })
    
    # Show ensemble result
    if len(results) > 1:
        print(f"\n🤝 Ensemble Analysis:")
        print("-" * 40)
        
        # Calculate ensemble score
        scores = [r['score'] for r in results]
        ensemble_score = np.mean(scores)
        
        if ensemble_score > 0.5:
            ensemble_result = "FAKE"
            ensemble_confidence = ensemble_score * 100
        else:
            ensemble_result = "REAL"
            ensemble_confidence = (1 - ensemble_score) * 100
        
        print(f"Ensemble Result: {ensemble_result}")
        print(f"Ensemble Confidence: {ensemble_confidence:.2f}%")
        print(f"Ensemble Score: {ensemble_score:.4f}")
        
        # Show individual model results
        print(f"\nIndividual Model Results:")
        for result in results:
            print(f"   {result['model']}: {result['result']} ({result['confidence']:.1f}%)")
        
        # Calculate agreement
        real_count = sum(1 for r in results if r['result'] == 'REAL')
        fake_count = sum(1 for r in results if r['result'] == 'FAKE')
        
        print(f"\nModel Agreement:")
        print(f"   REAL: {real_count} models")
        print(f"   FAKE: {fake_count} models")
        
        if real_count > fake_count:
            majority = "REAL"
        elif fake_count > real_count:
            majority = "FAKE"
        else:
            majority = "TIE"
        
        print(f"   Majority Vote: {majority}")
    
    # Show summary
    print(f"\n📊 Summary:")
    print("-" * 40)
    print(f"✅ {len(results)} models tested successfully")
    print(f"📷 Test image: {os.path.basename(test_image)}")
    print(f"🎯 Final result: {ensemble_result if len(results) > 1 else results[0]['result']}")
    print(f"📈 Confidence: {ensemble_confidence if len(results) > 1 else results[0]['confidence']:.2f}%")
    
    # Show usage instructions
    print(f"\n💻 Usage Instructions:")
    print("-" * 40)
    print("1. Add your test images to the 'uploads/' directory")
    print("2. Run this script: python use_all_models.py")
    print("3. Compare results from all models")
    print("4. Use ensemble result for best accuracy")
    
    print(f"\n🚀 For Web Interface:")
    print("-" * 40)
    print("1. Run: streamlit run working_ui.py")
    print("2. Open: http://localhost:8501")
    print("3. Navigate through different sections")

def show_code_example():
    """Show code example for using all models"""
    print(f"\n💻 Code Example - Using All Models:")
    print("=" * 60)
    
    code_example = '''
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Setup threading safety
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

def predict_with_all_models(image_path):
    """Predict using all available models"""
    models = [
        'lightweight_cpu_model.h5',
        'lightweight_cnn_model.h5', 
        'deeper_cnn_model.h5',
        'wide_cnn_model.h5'
    ]
    
    # Preprocess image
    image = Image.open(image_path)
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    
    predictions = []
    
    for model_file in models:
        model_path = f'saved_models/{model_file}'
        if os.path.exists(model_path):
            # Load model
            with tf.device('/CPU:0'):
                model = tf.keras.models.load_model(model_path)
            
            # Make prediction
            with tf.device('/CPU:0'):
                prediction = model.predict(image_batch, verbose=0)
            
            score = float(prediction[0][0])
            predictions.append(score)
    
    # Ensemble result
    ensemble_score = np.mean(predictions)
    result = 'FAKE' if ensemble_score > 0.5 else 'REAL'
    confidence = max(ensemble_score, 1-ensemble_score) * 100
    
    return result, confidence, ensemble_score

# Usage
result, confidence, score = predict_with_all_models('your_image.jpg')
print(f"Result: {result}")
print(f"Confidence: {confidence:.2f}%")
print(f"Score: {score:.4f}")
'''
    
    print(code_example)

if __name__ == "__main__":
    demonstrate_all_models()
    show_code_example()
