#!/usr/bin/env python3
"""
Training script for computer vision model with PyTorch and MPS backend.
This script trains a model to classify 5 different fighter jets.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import argparse
from tqdm import tqdm

# Check for MPS support
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def create_data_loaders(data_dir, img_size=224, batch_size=32, val_split=0.2, test_split=0.1):
    """
    Create data loaders for training, validation, and testing
    """
    # Define transforms for training and validation/test
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    
    # Get labels for stratified splitting
    labels = [label for _, label in dataset.samples]
    
    # Calculate sizes for splits
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    train_size = total_size - val_size - test_size
    
    # Split into train and (val + test) first
    train_indices, temp_indices = train_test_split(
        range(len(dataset)), 
        test_size=(val_size + test_size),
        train_size=train_size,
        stratify=labels,
        random_state=42
    )
    
    # Split temp_indices into validation and test
    if val_size > 0 and test_size > 0:
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=test_size,
            train_size=val_size,
            stratify=[labels[i] for i in temp_indices],
            random_state=42
        )
    elif val_size > 0:
        # Only validation split
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=test_size,
            train_size=val_size,
            stratify=[labels[i] for i in temp_indices],
            random_state=42
        )
    else:
        # Only test split
        val_indices = []
        test_indices = temp_indices
    
    # Create datasets using indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Set transforms for validation and test datasets
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, dataset.classes

def create_model(num_classes):
    """
    Create a pre-trained model for image classification
    """
    # Use ResNet50 as the base model - it's a well-established architecture
    # that provides good performance for image classification tasks
    # with good feature extraction capabilities
    model = models.resnet50(weights='DEFAULT')
    
    # Freeze the base model layers to preserve learned features
    #for param in model.parameters():
    #    param.requires_grad = False
    
    # Replace the final layer for our 5-class problem
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, num_epochs=25, learning_rate=0.001):
    """
    Train the model with checkpoint saving
    """
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Track best validation accuracy
    best_val_acc = 0.0
    best_model_path = 'train/best_model.pth'
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Add tqdm progress bar for training
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.data.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels.data).sum().item()
            
            # Update progress bar
            train_loader_tqdm.set_postfix({
                'Loss': f'{train_loss/(len(train_loader_tqdm)):.4f}',
                'Acc': f'{100. * correct_train/total_train:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        # Add tqdm progress bar for validation
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.data.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels.data).sum().item()
                
                # Update progress bar
                val_loader_tqdm.set_postfix({
                    'Loss': f'{val_loss/(len(val_loader_tqdm)):.4f}',
                    'Acc': f'{100. * correct_val/total_val:.2f}%'
                })
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average losses and accuracies
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * correct_train / total_train
        val_acc = 100. * correct_val / total_val
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Save checkpoint if this is the best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print()
    
    # Load the best model
    best_model = create_model(model.fc.out_features)  # This is a bit hacky, but we'll load the weights
    best_model.load_state_dict(torch.load(best_model_path))
    best_model = best_model.to(device)
    
    print(f"Best model loaded with validation accuracy: {best_val_acc:.2f}%")
    
    return best_model, train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, class_names):
     """
     Evaluate the model and generate metrics
     """
     model.eval()
     
     all_preds = []
     all_labels = []
     
     # Add tqdm progress bar for evaluation
     test_loader_tqdm = tqdm(test_loader, desc="Evaluating")
     with torch.no_grad():
         for inputs, labels in test_loader_tqdm:
             inputs, labels = inputs.to(device), labels.to(device)
             outputs = model(inputs)
             _, predicted = outputs.max(1)
             
             all_preds.extend(predicted.cpu().numpy())
             all_labels.extend(labels.cpu().numpy())
     
     # Generate classification report
     report = classification_report(all_labels, all_preds, target_names=class_names)
     print("Classification Report:")
     print(report)
     
     # Generate confusion matrix
     cm = confusion_matrix(all_labels, all_preds)
     
     # Plot confusion matrix
     plt.figure(figsize=(10, 8))
     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=class_names, yticklabels=class_names)
     plt.title('Confusion Matrix')
     plt.xlabel('Predicted')
     plt.ylabel('Actual')
     plt.tight_layout()
     plt.savefig('train/confusion_matrix.png')
     plt.close()
     
     return all_preds, all_labels

def visualize_test_predictions(model, test_loader, class_names, num_samples=10):
     """
     Visualize predictions on test samples
     """
     model.eval()
     
     # Get all test samples
     all_images = []
     all_labels = []
     all_predictions = []
     
     # Add tqdm progress bar for prediction visualization
     test_loader_tqdm = tqdm(test_loader, desc="Visualizing predictions")
     with torch.no_grad():
         for images, labels in test_loader_tqdm:
             images, labels = images.to(device), labels.to(device)
             
             outputs = model(images)
             _, predicted = outputs.max(1)
             
             all_images.extend(images.cpu())
             all_labels.extend(labels.cpu())
             all_predictions.extend(predicted.cpu())
     
     # Calculate number of plots needed
     total_samples = len(all_images)
     num_plots = (total_samples + num_samples - 1) // num_samples
     
     # Create plots with num_samples images each
     for plot_idx in range(num_plots):
         start_idx = plot_idx * num_samples
         end_idx = min((plot_idx + 1) * num_samples, total_samples)
         
         plt.figure(figsize=(15, 10))
         
         for i in range(start_idx, end_idx):
             plt.subplot(2, 5, i - start_idx + 1)
             img = all_images[i].numpy().transpose(1, 2, 0)
             # Denormalize image
             img = (img * 0.225 + 0.456).clip(0, 1)
             plt.imshow(img)
             plt.title(f'True: {class_names[all_labels[i]]}\nPred: {class_names[all_predictions[i]]}')
             plt.axis('off')
         
         plt.tight_layout()
         plt.savefig(f'train/test_predictions_{plot_idx + 1}.png')
         plt.close()

def main():
    """
    Main function to run the training pipeline
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train a computer vision model for fighter jet classification')
    parser.add_argument('--data-dir', type=str, default='./dataset',
                          help='Path to dataset directory (default: ./dataset)')
    parser.add_argument('--batch-size', type=int, default=32,
                          help='Batch size for training (default: 32)')
    parser.add_argument('--val-split', type=float, default=0.2,
                          help='Validation split ratio (default: 0.2)')
    parser.add_argument('--test-split', type=float, default=0.1,
                          help='Test split ratio (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=25,
                          help='Number of training epochs (default: 25)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                          help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    # Set data directory
    data_dir = args.data_dir
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"Dataset directory {data_dir} does not exist.")
        print("Please run the download_images.py script first to download the dataset.")
        return
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split)
    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    
    # Create model
    print("Creating model...")
    model = create_model(len(class_names))
    
    # Train model
    print("Starting training...")
    trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), 'train/trained_model.pth')
    print("Model saved as 'trained_model.pth'")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('train/training_history.png')
    plt.close()
    
    print("Training completed and results saved.")
    evaluate_model(trained_model, test_loader, class_names)
    visualize_test_predictions(trained_model, test_loader, class_names)

if __name__ == "__main__":
    main()