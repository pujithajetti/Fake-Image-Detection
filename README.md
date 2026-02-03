# DeepFake Image Detection Project

A comprehensive deep learning project for detecting fake images using multiple pre-trained models and a custom CNN architecture.

## 🎯 Project Overview

This project implements a multi-model approach to detect deepfake images using:
- **Pre-trained Models**: VGG16, VGG19, InceptionV3, ResNet50
- **Custom CNN**: Lightweight custom architecture
- **Ensemble Learning**: Weighted combination of all models

## 📊 Model Performance

| Model | Accuracy | Description | Size |
|-------|----------|-------------|------|
| VGG16 | 85.56% | Deep CNN with 16 layers | ~116 MB |
| VGG19 | 85.58% | Deep CNN with 19 layers | ~155 MB |
| ResNet50 | 50.00% | Residual network with 50 layers | ~157 MB |
| InceptionV3 | 77.39% | Inception architecture for efficient recognition | ~108 MB |
| Custom Model | 95.02% | Custom CNN with 9 layers | ~30 MB |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment (Linux/Mac)
source venv/bin/activate.fish  # For fish shell
# or
source venv/bin/activate       # For bash/zsh

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Dataset

```bash
python verify_dataset.py
```

### 3. Run Web Interface

```bash
streamlit run web_predictor.py --server.port 8501
```

Open your browser to `http://localhost:8501`

## 🎮 Usage Commands

### Web Interface (Recommended)
```bash
streamlit run web_predictor.py --server.port 8501
```
- Upload images through the web interface
- Select individual models for prediction
- View model accuracy and descriptions

### Command Line - Single Model Prediction
```bash
# List available models
python predict_single_model.py --list-models

# Predict with specific model
python predict_single_model.py --model VGG16 --image path/to/image.jpg
python predict_single_model.py --model Custom --image path/to/image.jpg
```

### Command Line - All Models Prediction
```bash
python predict_image.py path/to/image.jpg
```

## 🏋️ Training Commands

### Train All Models with Checkpoints
```bash
# Train all models (VGG16, VGG19, InceptionV3, ResNet50, Custom)
python train_all_models_checkpoints.py

# Train specific models only
python train_all_models_checkpoints.py --only vgg16 vgg19
python train_all_models_checkpoints.py --only inception custom
```

### Individual Model Training
```bash
# Train with checkpoint saving (recommended)
python train_with_checkpoints.py --model vgg16
python train_with_checkpoints.py --model vgg19
python train_with_checkpoints.py --model inception
python train_with_checkpoints.py --model resnet50
python train_with_checkpoints.py --model custom
```

### Quick CPU Training
```bash
# Lightweight model for quick testing
python quick_train_cpu.py
```

### Custom Model Training
```bash
# Train ensemble custom models
python ensemble_model.py
```

## 📁 Project Structure

```
deep/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── web_predictor.py                   # Main web interface
├── predict_single_model.py            # CLI single model predictor
├── predict_image.py                   # CLI all models predictor
├── train_all_models_checkpoints.py    # Main training script
├── train_with_checkpoints.py          # Individual model training
├── ensemble_model.py                  # Custom model training
├── quick_train_cpu.py                 # Quick CPU training
├── verify_dataset.py                  # Dataset verification
├── saved_models/                      # Trained model files
│   ├── vgg16_final_model.h5
│   ├── vgg19_final_model.h5
│   ├── resnet50_final_model.h5
│   ├── inceptionv3_final_model.h5
│   └── custom_final_model.h5
├── Pretrained_Models/                 # Training configurations
│   ├── config.py                      # Model configurations
│   ├── data_preprocessing.py          # Data loading
│   ├── data_augmentation.py           # Data augmentation
│   ├── load_pre_trained_models.py     # Model loading
│   ├── VGG16_finetuning.py           # VGG16 training
│   ├── VGG19_finetuning.py           # VGG19 training
│   ├── inceptionV3_finetuning.py     # InceptionV3 training
│   └── resnet50_finetuning.py        # ResNet50 training
└── archive/                           # Dataset directory
    └── real_vs_fake/
        └── real-vs-fake/
            ├── train/
            │   ├── real/
            │   └── fake/
            ├── valid/
            │   ├── real/
            │   └── fake/
            └── test/
                ├── real/
                └── fake/
```

## 🔧 Configuration

### Model Parameters (Pretrained_Models/config.py)
- **Image Size**: 128x128 pixels
- **Batch Size**: 16
- **Epochs**: 10 (with early stopping)
- **Learning Rate**: 1e-4
- **Optimizer**: Adam

### Data Augmentation
- Rotation: ±20 degrees
- Width/Height Shift: 0.2
- Shear: 0.2
- Zoom: 0.2
- Horizontal Flip: True

## 📈 Training Process

1. **Data Preprocessing**: Images resized to 128x128, normalized
2. **Data Augmentation**: Applied to training set only
3. **Model Fine-tuning**: Pre-trained models fine-tuned on dataset
4. **Checkpoint Saving**: Models saved after each epoch
5. **Early Stopping**: Training stops if no improvement for 3 epochs

## 🎯 Prediction Features

### Web Interface Features
- **Model Selection**: Choose specific models for prediction
- **Real-time Results**: Instant prediction with confidence scores
- **Model Information**: Display accuracy, description, and file size
- **Image Preview**: Visual confirmation of uploaded images

### Command Line Features
- **Batch Processing**: Process multiple images
- **Model Comparison**: Compare results across all models
- **Confidence Scores**: Detailed probability outputs

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Models run on CPU by default

2. **Memory Issues**: Use smaller batch sizes or individual model training
3. **Dataset Issues**: Run `python verify_dataset.py` to check dataset structure

### Performance Tips

1. **CPU Training**: Set `CUDA_VISIBLE_DEVICES='-1'` for CPU-only training
2. **Memory Optimization**: Use checkpoint training for large models
3. **Quick Testing**: Use `quick_train_cpu.py` for rapid prototyping

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Scikit-learn

## 🎉 Results Summary

The project successfully implements a multi-model deepfake detection system with:
- **5 trained models** with varying performance characteristics
- **Web interface** for easy image upload and prediction
- **Command-line tools** for batch processing
- **Comprehensive training pipeline** with checkpoint saving
- **Real-time prediction** with confidence scores

The **Custom Model** achieves the highest accuracy at 95.02%, making it the recommended choice for production use.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify dataset structure with `verify_dataset.py`
3. Ensure all dependencies are installed correctly
4. Check model files exist in `saved_models/` directory