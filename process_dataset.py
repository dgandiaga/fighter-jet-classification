#!/usr/bin/env python3
"""
Script to process dataset and extract airplane images using YOLO model.
"""

import os
import shutil
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from tqdm import tqdm


def create_dataset_curated_structure():
    """Create the dataset_curated folder with the same structure as dataset."""
    # Remove existing dataset_curated folder if it exists
    if os.path.exists('dataset_curated'):
        shutil.rmtree('dataset_curated')
    
    # Create dataset_curated folder
    os.makedirs('dataset_curated', exist_ok=True)
    
    # Copy the folder structure from dataset to dataset_curated
    for root, dirs, files in os.walk('dataset'):
        if root == 'dataset':
            continue
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            relative_path = os.path.relpath(dir_path, 'dataset')
            new_dir_path = os.path.join('dataset_curated', relative_path)
            os.makedirs(new_dir_path, exist_ok=True)


def process_images_with_yolo():
    """Process images with YOLO to detect and crop airplanes."""
    # Load the pretrained YOLO model
    model = YOLO('models/yolov8l.pt')
    
    # Check if MPS is available for Apple Silicon Macs
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Process each image in dataset
    for root, dirs, files in os.walk('dataset'):
        if root == 'dataset':
            continue
        for file in tqdm(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, 'dataset')
                output_dir = os.path.join('dataset_curated', relative_path)
                
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Run YOLO detection
                results = model(image_path, conf=0.3, iou=0.45, verbose=False)
                
                # Process detections
                image = Image.open(image_path)
                image_array = np.array(image)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get class
                            cls = int(box.cls)
                            
                            # Check for aircraft classes (COCO dataset)
                            # Airplane is class 4 in COCO dataset
                            if cls == 4:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                # Crop the bounding box
                                cropped = image_array[y1:y2, x1:x2]
                                
                                # Save the cropped image
                                output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_crop_{x1}_{y1}_{x2}_{y2}.jpg")
                                cropped_image = Image.fromarray(cropped)
                                cropped_image.convert("RGB").save(output_path)
                                # print(f"Saved cropped image: {output_path}")


def main():
    """Main function to process dataset."""
    print("Creating dataset_curated structure...")
    create_dataset_curated_structure()
    
    print("Processing images with YOLO...")
    process_images_with_yolo()
    
    print("Processing complete!")
    for folder in os.listdir('dataset'):
        print(f'{folder}: {len(os.listdir(os.path.join("dataset", folder)))}')


if __name__ == "__main__":
    main()