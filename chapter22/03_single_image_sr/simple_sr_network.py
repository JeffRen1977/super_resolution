"""
Chapter 22 - Section 3: Single Image Super Resolution (SISR)
=============================================================

This module demonstrates basic deep learning approach for super resolution
using a simple convolutional neural network.

Learning Objectives:
- Understand how neural networks learn to upscale
- See the difference from traditional interpolation
- Learn basic SR network architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple


class SimpleSRNetwork(nn.Module):
    """
    Simple Super Resolution Network
    
    This is a basic CNN that learns to upscale images.
    Architecture:
    1. Feature extraction (learns image features)
    2. Upsampling (increases resolution)
    3. Reconstruction (generates HR image)
    
    This demonstrates the fundamental idea: neural networks can learn
    to predict high-frequency details that interpolation cannot.
    """
    
    def __init__(self, scale_factor: int = 4):
        super(SimpleSRNetwork, self).__init__()
        self.scale_factor = scale_factor
        
        # Feature extraction: Learn useful image features
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),  # Extract features
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Refine features
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # More refinement
            nn.ReLU(inplace=True)
        )
        
        # Upsampling: Increase resolution
        # Method 1: Pixel shuffle (efficient sub-pixel convolution)
        out_channels = 3 * (scale_factor ** 2)
        self.upsampler = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)  # Rearranges channels to spatial dimensions
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Low-resolution image (B, C, H, W)
        
        Returns:
            High-resolution image (B, C, H×scale, W×scale)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Upsample and reconstruct
        hr_image = self.upsampler(features)
        
        return hr_image


def prepare_image_for_network(image: np.ndarray) -> torch.Tensor:
    """
    Convert NumPy image to PyTorch tensor
    
    Args:
        image: Image as NumPy array (H, W, C), values 0-255
    
    Returns:
        Tensor (1, C, H, W), values 0-1
    """
    # Normalize to [0, 1]
    image_tensor = torch.from_numpy(image).float() / 255.0
    
    # Convert HWC to CHW
    image_tensor = image_tensor.permute(2, 0, 1)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor back to NumPy image
    
    Args:
        tensor: Image tensor (1, C, H, W), values 0-1
    
    Returns:
        NumPy array (H, W, C), values 0-255
    """
    # Remove batch dimension and convert CHW to HWC
    image = tensor.squeeze(0).permute(1, 2, 0)
    
    # Denormalize and clip
    image = torch.clamp(image * 255.0, 0, 255).numpy().astype(np.uint8)
    
    return image


def apply_sr_network(lr_image: np.ndarray, model: nn.Module, 
                    device: str = 'cpu') -> np.ndarray:
    """
    Apply super resolution network to an image
    
    Args:
        lr_image: Low-resolution image
        model: Trained SR network
        device: 'cpu' or 'cuda'
    
    Returns:
        Super-resolved high-resolution image
    """
    model.eval()
    
    # Prepare input
    input_tensor = prepare_image_for_network(lr_image).to(device)
    
    # Forward pass
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Convert back to image
    sr_image = tensor_to_image(output_tensor.cpu())
    
    return sr_image


def demonstrate_sisr():
    """
    Demonstrate single image super resolution
    
    Note: This uses an untrained model for demonstration.
    In practice, the model would be trained on HR-LR pairs.
    """
    print("=" * 60)
    print("Chapter 22 - Section 3: Single Image Super Resolution")
    print("=" * 60)
    print()
    
    print("Key Concept:")
    print("- Traditional interpolation: Mathematical upsampling")
    print("- Deep learning SR: Learns to predict missing details")
    print("- Advantage: Can recover high-frequency information")
    print()
    
    # Create test image
    print("Creating test image...")
    hr_image = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(hr_image, (20, 20), (60, 60), (255, 255, 255), -1)
    cv2.circle(hr_image, (90, 40), 15, (0, 255, 0), -1)
    cv2.putText(hr_image, 'SISR', (30, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Create low-resolution version
    lr_image = cv2.resize(hr_image, (32, 32), interpolation=cv2.INTER_CUBIC)
    
    # Create model (untrained for demo)
    print("Creating SR network...")
    model = SimpleSRNetwork(scale_factor=4)
    
    # Apply SR
    print("Applying super resolution (untrained model)...")
    sr_image = apply_sr_network(lr_image, model, device='cpu')
    
    # Compare with bicubic
    bicubic = cv2.resize(lr_image, (128, 128), interpolation=cv2.INTER_CUBIC)
    
    print()
    print("Note: This model is untrained (random weights).")
    print("A trained model would show significant improvement over bicubic.")
    print()
    print("✅ Single Image SR demonstration complete!")
    print()


if __name__ == "__main__":
    demonstrate_sisr()

