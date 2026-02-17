#!/usr/bin/env python3
"""
Grad-CAM (Gradient-weighted Class Activation Mapping) visualization script.
Generates saliency maps for a given image using the model trained with
train_model.py.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from typing import Optional

# Check for MPS support
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module,
                 num_classes: int):
        """
        Initialize Grad-CAM.

        Args:
            model: The trained model
            target_layer: The layer to compute Grad-CAM for
            num_classes: Number of classes in the model
        """
        self.model = model
        self.target_layer = target_layer
        self.num_classes = num_classes
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and
        gradients."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def compute_gradcam(self, input_tensor: torch.Tensor,
                        target_class: Optional[int] = None) -> np.ndarray:
        """
        Compute Grad-CAM for the input tensor.

        Args:
            input_tensor: Input image tensor (1, 3, H, W)
            target_class: Target class index. If None, uses the predicted
            class.

        Returns:
            Grad-CAM heatmap (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # If target_class is not specified, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero out all gradients
        self.model.zero_grad()
        
        # Backward pass for the target class
        output[0, target_class].backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Compute weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # ReLU to remove negative values
        
        # Normalize CAM
        cam = cam.squeeze().cpu().numpy()  # (H, W)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def load_model(
    model_path: str,
    num_classes: int,
    unfreeze_backbone: bool = True,
) -> nn.Module:
    """
    Load the trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        num_classes: Number of classes in the model
        unfreeze_backbone: Whether to unfreeze backbone for Grad-CAM
    
    Returns:
        Loaded model
    """
    # Create a new model with the same architecture
    model = models.resnet50(weights='DEFAULT')
    
    # Freeze params unless specified
    if not unfreeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path: str, img_size: int = 224) -> torch.Tensor:
    """
    Preprocess an image for model input.
    
    Args:
        image_path: Path to the image
        img_size: Target image size
    
    Returns:
        Preprocessed image tensor
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor.to(device), image


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize an image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor (1, 3, H, W)
    
    Returns:
        Denormalized image as numpy array (H, W, 3)
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    return image


def overlay_gradcam_on_image(
    image: np.ndarray,
    gradcam: np.ndarray,
    alpha: float = 0.4,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        image: Original image (H, W, 3)
        gradcam: Grad-CAM heatmap (H, W)
        alpha: Transparency of the heatmap overlay
        colormap: Colormap name for the heatmap
    
    Returns:
        Overlayed image (H, W, 3)
    """
    # Create a colored heatmap
    heatmap = plt.get_cmap(colormap)(gradcam)
    heatmap = np.delete(heatmap, 3, axis=-1)  # Remove alpha channel
    
    # Resize heatmap to match image size
    heatmap = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]), Image.BILINEAR))
    
    # Overlay heatmap on image
    overlayed = (1 - alpha) * image + alpha * heatmap
    
    return np.clip(overlayed, 0, 1)


def visualize_gradcam(
    model: nn.Module,
    image_path: str,
    target_layer: nn.Module,
    num_classes: int,
    output_path: str = "gradcam_output.png",
    target_class: Optional[int] = None,
    img_size: int = 224,
    alpha: float = 0.4,
    colormap: str = 'jet'
):
    """
    Generate and visualize Grad-CAM for an image.
    
    Args:
        model: Trained model
        image_path: Path to the input image
        target_layer: Target layer for Grad-CAM
        num_classes: Number of classes
        output_path: Path to save the output image
        target_class: Target class index (None for predicted class)
        img_size: Input image size
        alpha: Transparency of heatmap overlay
        colormap: Colormap name
    """
    # Preprocess image
    input_tensor, original_image = preprocess_image(image_path, img_size)
    
    # Create Grad-CAM instance
    gradcam = GradCAM(model, target_layer, num_classes)
    
    # Compute Grad-CAM
    print("Computing Grad-CAM...")
    cam = gradcam.compute_gradcam(input_tensor, target_class)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[predicted_class].item()
    
    # Denormalize image
    image_array = denormalize_image(input_tensor)
    
    # Overlay Grad-CAM
    overlayed_image = overlay_gradcam_on_image(
        image_array, cam, alpha, colormap)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(image_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Grad-CAM overlay
    axes[1].imshow(overlayed_image)
    axes[1].set_title(
        f'Grad-CAM\nClass: {predicted_class} '
        f'(Confidence: {confidence:.2%})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Grad-CAM visualization saved to: {output_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    # Print top 3 predictions
    top3_probs, top3_classes = torch.topk(probabilities, 3)
    print("\nTop 3 Predictions:")
    for i, (class_idx, prob) in enumerate(zip(top3_classes, top3_probs)):
        print(f"  {i+1}. Class {class_idx.item()}: {prob.item():.2%}")


def main():
    """
    Main function to run Grad-CAM visualization.
    """
    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM saliency maps for a given image'
    )
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument(
        '--model-path', type=str, default='train/best_model.pth',
        help='Path to the trained model checkpoint')
    parser.add_argument('--num-classes', type=int, default=5,
                        help='Number of classes in the model (default: 5)')
    parser.add_argument(
        '--output-path', type=str, default='gradcam_output.png',
        help='Path to save the output image')
    parser.add_argument('--target-class', type=int, default=None,
                        help='Target class index for Grad-CAM')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Transparency of heatmap overlay (default: 0.4)')
    parser.add_argument('--colormap', type=str, default='jet',
                        help='Colormap for heatmap (default: jet)')
    parser.add_argument(
        '--target-layer', type=str, default='layer4',
        help='Target layer for Grad-CAM (default: layer4)')
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print("Please train the model first using train_model.py")
        return
    
    # Check if image path exists
    if not os.path.exists(args.image_path):
        print(f"Image file not found: {args.image_path}")
        return
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, args.num_classes)
    
    # Get target layer
    if args.target_layer == 'layer4':
        target_layer = model.layer4[-1]  # Use the last conv block in layer4
    elif args.target_layer == 'layer3':
        target_layer = model.layer3[-1]
    else:
        print(f"Unknown target layer: {args.target_layer}")
        print("Using layer4 as default")
        target_layer = model.layer4[-1]
    
    # Generate Grad-CAM visualization
    visualize_gradcam(
        model=model,
        image_path=args.image_path,
        target_layer=target_layer,
        num_classes=args.num_classes,
        output_path=args.output_path,
        target_class=args.target_class,
        img_size=args.img_size,
        alpha=args.alpha,
        colormap=args.colormap
    )


if __name__ == "__main__":
    main()