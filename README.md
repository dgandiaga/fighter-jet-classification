# Computer Vision Image Classification Project

This project implements a computer vision model using PyTorch to classify images of different fighter jets into 5 classes.

## Project Structure

- `download_images.py` - Script to download images of fighter jets
- `requirements.txt` - Python dependencies
- `setup_env.py` - Script to create virtual environment and install packages
- `train_model.py` - Main training script with MPS support
- `dataset/` - Directory for storing the dataset (will be created automatically)

## Features

- Uses PyTorch with MPS (Metal Performance Shaders) backend for Apple Silicon Macs
- Implements ResNet50 as the base model for image classification
- Data augmentation techniques for better generalization
- Training and validation pipeline
- Evaluation with classification report and confusion matrix

## Setup Instructions

1. **Install dependencies:**
   ```bash
   python setup_env.py
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Download dataset:**
   ```bash
   # Download 100 images per class (default)
   python download_images.py
   
   # Download custom number of images per class
   python download_images.py --count 50
   ```

4. **Train the model:**
   ```bash
   python train_model.py
   ```

## Model Architecture

The model uses ResNet50 as a pre-trained base model with:
- Feature extraction from pre-trained ResNet50
- Frozen base layers to preserve learned features
- Custom final layer for 5-class classification
- Data augmentation for improved generalization

## Training Process

- Uses Adam optimizer with learning rate scheduling
- Cross-entropy loss function
- 25 epochs of training with validation
- Batch size of 32
- Data augmentation including:
  - Random horizontal flips
  - Random rotations
  - Color jittering
  - Normalization

## Evaluation

The model provides:
- Training and validation metrics
- Classification report
- Confusion matrix visualization
- Model saving capability

## Requirements

- Python 3.8+
- PyTorch 2.1.0
- TorchVision 0.16.0
- NumPy 1.24.3
- Matplotlib 3.7.2
- Scikit-learn 1.3.0

## MPS Support

The model automatically detects and uses MPS (Metal Performance Shaders) on Apple Silicon Macs for accelerated computation.