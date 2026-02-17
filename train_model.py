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
import pandas as pd
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
    
    # Freeze params
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final layer for our 5-class problem
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, num_epochs=25, warmup_epochs=5, warmup_lr=1e-3, unfreeze_lr=1e-5, scheduler_step_size=7, scheduler_gamma=0.1, patience=10, experiment_folder='train', label_smoothing=0.0):
    """
    Train the model with checkpoint saving and early stopping
    """
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    # Initial optimizer during warmup
    optimizer = optim.Adam(model.fc.parameters(), lr=warmup_lr)
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    # Lists to store learning rates
    fc_lr_history = []
    layer4_lr_history = []
    
    # Track best validation accuracy and early stopping
    best_val_acc = 0.0
    best_model_path = f'{experiment_folder}/best_model.pth'
    patience_counter = 0
    
    print("Starting training...")
    
    # Initialize scheduler variable
    scheduler = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        if epoch == warmup_epochs:
            print("Warm-up complete. Unfreezing backbone for fine-tuning...")
            
            # Unfreeze the backbone (or just Layer 4)
            for param in model.layer4.parameters():
                param.requires_grad = True
                
            # IMPORTANT: Re-initialize optimizer to include new parameters
            # Use your previous successful warmup_lr for the head, but unfreeze_lr for the backbone
            optimizer = optim.Adam([
                {'params': model.layer4.parameters(), 'lr': unfreeze_lr},
                {'params': model.fc.parameters(), 'lr': warmup_lr}
            ])
            
            # Switch to your preferred StepLR scheduler for Phase 2
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        
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
        #scheduler.step()
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()
        
        # Track learning rates
        # Get current learning rates from optimizer
        current_fc_lr = 0
        current_layer4_lr = 0
        if len(optimizer.param_groups) > 1:
            # First group is layer4 (frozen during warmup, then decays)
            # Second group is fc (fixed during warmup, then decays)
            current_fc_lr = optimizer.param_groups[1]['lr']
            current_layer4_lr = optimizer.param_groups[0]['lr']
        elif len(optimizer.param_groups) > 0:
            # If only one group, it's likely the fc layer
            current_fc_lr = optimizer.param_groups[0]['lr']
        fc_lr_history.append(current_fc_lr)
        layer4_lr_history.append(current_layer4_lr)
        
        # Calculate average losses and accuracies
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * correct_train / total_train
        val_acc = 100. * correct_val / total_val
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Check for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs. No improvement in validation accuracy for {patience} epochs.")
                break
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print()
    
    # Load the best model
    best_model = create_model(model.fc.out_features)  # This is a bit hacky, but we'll load the weights
    best_model.load_state_dict(torch.load(best_model_path))
    best_model = best_model.to(device)
    
    print(f"Best model loaded with validation accuracy: {best_val_acc:.2f}%")
    
    # Create learning rate plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fc_lr_history)), fc_lr_history, label='Fully Connected Layer LR', marker='o')
    plt.plot(range(len(layer4_lr_history)), layer4_lr_history, label='Layer 4 LR', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rates During Training')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.savefig(f'{experiment_folder}/learning_rates.png')
    plt.close()

    return best_model, train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, class_names, experiment_folder="train", use_tta=False):
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
            
            # Apply test time augmentation if enabled
            if use_tta:
                # Apply multiple augmentations and average predictions
                predictions = []
                # Original prediction
                outputs = model(inputs)
                predictions.append(outputs)
                
                # Add some augmentations for TTA
                # We'll apply horizontal flip augmentation
                flipped_inputs = transforms.functional.hflip(inputs)
                flipped_outputs = model(flipped_inputs)
                predictions.append(flipped_outputs)
                
                # Average the predictions
                avg_outputs = torch.stack(predictions).mean(dim=0)
                _, predicted = avg_outputs.max(1)
            else:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # 2. Load it into a Pandas DataFrame
    df = pd.DataFrame(report).transpose()

    # 3. Save it
    df.to_csv(f'{experiment_folder}/classification_report.csv', index=True)

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
    plt.savefig(f'{experiment_folder}/confusion_matrix.png')
    plt.close()
    
    return all_preds, all_labels

def visualize_test_predictions(model, test_loader, class_names, num_samples=10, experiment_folder="train", use_tta=False):
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
            
            # Apply test time augmentation if enabled
            if use_tta:
                # Apply multiple augmentations and average predictions
                predictions = []
                # Original prediction
                outputs = model(images)
                predictions.append(outputs)
                
                # Add some augmentations for TTA
                # We'll apply horizontal flip augmentation
                flipped_inputs = transforms.functional.hflip(images)
                flipped_outputs = model(flipped_inputs)
                predictions.append(flipped_outputs)
                
                # Average the predictions
                avg_outputs = torch.stack(predictions).mean(dim=0)
                _, predicted = avg_outputs.max(1)
            else:
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
        plt.savefig(f'{experiment_folder}/test_predictions_{plot_idx + 1}.png')
        plt.close()

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, experiment_folder='train'):
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
    plt.savefig(f'{experiment_folder}/training_history.png')
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
    parser.add_argument('--epochs', type=int, default=100,
                          help='Number of training epochs (default: 25)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                          help='Learning rate (default: 0.001)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                          help='Number of warmup epochs (default: 5)')
    parser.add_argument('--warmup-lr', type=float, default=1e-3,
                          help='Learning rate during warmup (default: 1e-3)')
    parser.add_argument('--unfreeze-lr', type=float, default=1e-5,
                          help='Learning rate after unfreezing (default: 1e-5)')
    parser.add_argument('--scheduler-step-size', type=int, default=7,
                          help='Step size for scheduler (default: 7)')
    parser.add_argument('--scheduler-gamma', type=float, default=0.1,
                          help='Gamma for scheduler (default: 0.1)')
    parser.add_argument('--img-size', type=int, default=224,
                          help='Input image size (default: 224)')
    parser.add_argument('--patience', type=int, default=10,
                          help='Number of epochs with no improvement to wait before stopping (default: 10)')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                          help='Label smoothing factor (default: 0.0, set to 0.1 for smoothing)')
    parser.add_argument('--experiment-name', type=str, default=None,
                          help='Name of the experiment (default: None)')
    parser.add_argument('--test-time-augmentation', action='store_true',
                          help='Enable test time augmentation (default: disabled)')
    
    args = parser.parse_args()
    
    # Set data directory
    data_dir = args.data_dir
    
    # Create experiment folder if experiment_name is provided
    experiment_folder = None
    if args.experiment_name:
        experiment_folder = f"train/{args.experiment_name}"
        os.makedirs(experiment_folder, exist_ok=True)
    else:
        experiment_folder = "train"
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"Dataset directory {data_dir} does not exist.")
        print("Please run the download_images.py script first to download the dataset.")
        return
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir,
        img_size=args.img_size,
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
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
        unfreeze_lr=args.unfreeze_lr,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        patience=args.patience,
        experiment_folder=experiment_folder,
        label_smoothing=args.label_smoothing)
    
    print("Training completed and results saved.")
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, experiment_folder)
    evaluate_model(trained_model, test_loader, class_names, experiment_folder, use_tta=args.test_time_augmentation)
    visualize_test_predictions(trained_model, test_loader, class_names, experiment_folder=experiment_folder, use_tta=args.test_time_augmentation)

if __name__ == "__main__":
    main()