# Fighter Jet Classification Project

This project implements a complete computer vision pipeline for classifying images of different fighter jets into 5 classes. The system performs web scraping, image processing with YOLO object detection, duplicate removal, and training of a ResNet model with MPS support.

## Project Structure

```
.
├── README.md                 # Project documentation
├── download_images_v2.py     # Web scraping script for downloading fighter jet images
├── process_dataset.py          # Script to extract crops using YOLO model
├── train_model.py            # Main training script with MPS support
├── run_experiments.py         # Script to run multiple training experiments
├── test_dataset.py           # Script to test dataset functionality
├── test_duplicate_detection.py # Script to test duplicate detection
├── test_duplicate_threshold.py   # Script to test duplicate threshold
├── utils.py                 # Utility functions for file operations and duplicate detection
├── requirements.txt         # Python dependencies
├── config/
│   ├── experiments.json     # Configuration for training experiments
│   ├── queries.json          # Search queries for web scraping
├── dataset/                 # Directory for storing raw downloaded images (created automatically)
├── dataset_curated/           # Directory for storing processed and curated images (created automatically)
├── train/                   # Directory for storing training results (created automatically)
└── models/                  # Directory for storing trained models (created automatically)
```

## Project Workflow

This project follows a complete pipeline from data collection to model training:

1. **Web Scraping**: Uses Bing Image Crawler to download images of fighter jets from the web
2. **Image Processing**: Applies YOLO object detection to extract airplane images from downloaded images
3. **Duplicate Removal**: Removes duplicate images using image hash comparison
4. **Model Training**: Trains a ResNet50 model on the refined dataset

## Dataset Creation Process

### 1. Web Scraping
- Downloads images of 5 fighter jet classes: F-35, J-20, Eurofighter Typhoon, Rafale, and Gripen
- Uses Bing Image Crawler with stealth headers to avoid blocking
- Implements persistent download to ensure target number of images per class
- Removes duplicates after each download batch

### 2. Image Processing with YOLO
- Uses YOLOv8 model to detect and crop airplane images from downloaded images
- Extracts only airplane images from the downloaded dataset
- Preserves original image structure while focusing on aircraft

### 3. Duplicate Removal
- Compares images using imagehash library
- Removes duplicate images based on hash comparison
- Maintains only unique images in the dataset

## Model Training

### Architecture
- Uses ResNet50 as the base model for image classification
- Implements feature extraction from pre-trained ResNet50
- Uses frozen base layers to preserve learned features
- Custom final layer for 5-class classification

### Training Process
- Uses Adam optimizer with learning rate scheduling
- Cross-entropy loss function
- 25 epochs of training with validation
- Batch size of 32
- Data augmentation including:
  - Random horizontal flips
  - Random rotations
  - Color jittering
  - Normalization

### Experimentation
- Multiple training experiments with different configurations
- Configured in `config/experiments.json`
- Runs all experiments using `python run_experiments.py`

## Features

- **MPS Support**: Automatically detects and uses MPS (Metal Performance Shaders) on Apple Silicon Macs for accelerated computation
- **Data Augmentation**: Improves generalization through various data augmentation techniques
- **Duplicate Detection**: Removes duplicate images based on hash comparison
- **YOLO Integration**: Uses YOLO for accurate airplane detection and cropping
- **Experimentation Framework**: Supports multiple training experiments with different configurations

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download dataset:**
   ```bash
   # Download 100 images per class (default)
   python download_images_v2.py
   
   # Download custom number of images per class
   python download_images_v2.py --count 50
   ```

3. **Process dataset:**
   ```bash
   python process_dataset.py
   ```

4. **Run experiments:**
   ```bash
   python run_experiments.py
   ```

## Requirements

- Python 3.8+
- PyTorch 2.1.0
- TorchVision 0.16.0
- NumPy 1.24.3
- Pillow 10.0.1
- ImageHash 4.3.1
- scikit-learn 1.3.0
- tqdm 4.65.0
- Ultralytics 8.x (for YOLO)

## Training Process

The training process includes:
1. Data loading and preprocessing
2. Model creation with ResNet50 architecture
3. Training with warmup and fine-tuning phases
4. Validation and early stopping
5. Evaluation with classification report and confusion matrix
6. Model saving capability

## Evaluation

The model provides:
- Training and validation metrics
- Classification report
- Confusion matrix visualization
- Model saving capability

## Configuration

### Search Queries
The `config/queries.json` file contains search queries for each fighter jet class to improve image quality and variety.

### Experiments
The `config/experiments.json` file contains configurations for multiple training experiments with different parameters.

## MPS Support

The model automatically detects and uses MPS (Metal Performance Shaders) on Apple Silicon Macs for accelerated computation.

## 📊 Results and Analysis
The project evolved through several experimental phases, transitioning from raw web-scraped data to a highly curated dataset using YOLOv8 crops and advanced regularization techniques.

### Training Dynamics and Overfitting

Initial experiments showed a classic overfitting pattern. As seen in the training curves, the model quickly achieved near 100% training accuracy, while validation accuracy lagged significantly behind, plateauing around 71%.

This gap indicated that the ResNet50 backbone, when fully unfrozen, was memorizing specific image artifacts (noise, backgrounds, or watermarks) rather than generalizing the geometric features of the aircraft. By implementing Differential Learning Rates (e.g., 1×10 
−5
  for the backbone and 1×10 
−3
  for the head) and a 5-epoch Warmup phase, we stabilized the training process, forcing the model to rely more on pre-trained ImageNet features before fine-tuning.

### Dataset Curation and YOLOv8 Impact

The most significant leap in performance came from dataset curation. Raw scrapes contained significant "noise," including non-plane images or aircraft obscured in the background. By utilizing YOLOv8 to isolate tight crops, the model's focus was restricted to the aircraft itself.

However, this curation introduced a "data leakage" risk. Because multiple crops often originated from the same high-resolution source or burst-photo gallery, structural hashes (pHash) initially flagged up to 95% of the data as duplicates. To combat this while maintaining a viable dataset size, we implemented Source-based Splitting, ensuring that all crops from a single original image remain within the same Train or Validation fold.

### Model Performance and Classification

The final model achieves strong class separation, though some confusion remains between aerodynamically similar delta-wing fighters. The Confusion Matrix reveals that the Dassault Rafale (45 correct) and Eurofighter Typhoon (36 correct) are the strongest performers, while the Saab Gripen and J-20 occasionally show inter-class confusion due to similar planform profiles in certain orientations.

Despite these challenges, the qualitative results show the model is highly capable of identifying key features even in high-aspect-ratio or tilted shots. The integration of Test-Time Augmentation (TTA) and Label Smoothing (0.1) helped bridge the final validation gap, resulting in a more robust and generalizable classifier.