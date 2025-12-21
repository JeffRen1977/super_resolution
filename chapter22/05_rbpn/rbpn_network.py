"""
Chapter 22 - Section 5: Recurrent Back-Projection Network (RBPN)
==================================================================

This module implements the RBPN architecture, an advanced video super resolution
method that uses iterative back-projection for refinement.

Learning Objectives:
- Understand the back-projection principle
- Learn how iterative refinement improves quality
- See the complete RBPN architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple


class RBPNNetwork(nn.Module):
    """
    Recurrent Back-Projection Network (RBPN)
    
    Key innovations:
    1. Back-Projection: Iteratively refines output by computing and correcting errors
    2. Recurrent Structure: Processes frames sequentially with temporal dependencies
    3. Dense Connections: Enables effective information flow
    
    This is a simplified version for educational purposes.
    Full RBPN includes more sophisticated temporal alignment and multiple projections.
    """
    
    def __init__(self, scale_factor: int = 4, num_frames: int = 3):
        super(RBPNNetwork, self).__init__()
        self.scale_factor = scale_factor
        self.num_frames = num_frames
        
        # Feature extraction: Extract rich features from each frame
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Back-projection module: Iteratively refines features
        # The key idea: compute error, project it back, and refine
        self.back_projection = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Reconstruction: Convert features to HR image
        out_channels = 3 * (scale_factor ** 2)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )
        
        # Pixel shuffle: Efficient upsampling
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RBPN
        
        Processing flow:
        1. Extract features from each frame
        2. Fuse temporal information
        3. Apply back-projection refinement
        4. Reconstruct and upsample
        
        Args:
            frames: (B, T, C, H, W) - batch of T frames
        
        Returns:
            Super-resolved frame (B, C, H×scale, W×scale)
        """
        B, T, C, H, W = frames.shape
        
        # Step 1: Feature extraction for each frame
        features = []
        for t in range(T):
            feat = self.feature_extractor(frames[:, t])
            features.append(feat)
        
        # Step 2: Temporal fusion (simple average for demo)
        # In full RBPN, this uses optical flow for alignment
        fused_feature = torch.stack(features, dim=1).mean(dim=1)
        
        # Step 3: Back-projection refinement
        # This is the key innovation: iterative error correction
        projected = self.back_projection(fused_feature)
        
        # Step 4: Reconstruction
        reconstructed = self.reconstruction(projected)
        
        # Step 5: Upsampling via pixel shuffle
        super_resolved = self.pixel_shuffle(reconstructed)
        
        return super_resolved


def explain_back_projection():
    """
    Explain the back-projection principle
    
    This is a key concept that makes RBPN effective.
    """
    print("=" * 60)
    print("Understanding Back-Projection")
    print("=" * 60)
    print()
    print("Back-projection principle:")
    print()
    print("1. Forward Projection:")
    print("   - Start with HR estimate: I_HR")
    print("   - Simulate degradation: I_LR_est = D(I_HR)")
    print("   - D = downsampling + blur")
    print()
    print("2. Error Computation:")
    print("   - Compare with observed LR: error = I_LR - I_LR_est")
    print()
    print("3. Back-Projection:")
    print("   - Project error back to HR space: error_HR = U(error)")
    print("   - U = upsampling operation")
    print()
    print("4. Refinement:")
    print("   - Update HR estimate: I_HR = I_HR + error_HR")
    print("   - Iterate until convergence")
    print()
    print("Key insight: This iterative process ensures the reconstructed")
    print("HR image, when degraded, matches the observed LR image.")
    print()


def prepare_frames_for_network(frames: List[np.ndarray]) -> torch.Tensor:
    """Prepare frames for network input"""
    tensor_frames = []
    for frame in frames:
        tensor = torch.from_numpy(frame).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        tensor_frames.append(tensor)
    return torch.stack(tensor_frames, dim=0).unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to image"""
    image = tensor.squeeze(0).permute(1, 2, 0)
    image = torch.clamp(image * 255.0, 0, 255).numpy().astype(np.uint8)
    return image


def demonstrate_rbpn():
    """
    Demonstrate RBPN super resolution
    """
    print("=" * 60)
    print("Chapter 22 - Section 5: Recurrent Back-Projection Network")
    print("=" * 60)
    print()
    
    explain_back_projection()
    
    # Create test image
    print("Creating test image...")
    hr_image = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(hr_image, (20, 20), (60, 60), (255, 255, 255), -1)
    cv2.circle(hr_image, (90, 40), 15, (0, 255, 0), -1)
    cv2.putText(hr_image, 'RBPN', (35, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Create low-resolution version
    lr_image = cv2.resize(hr_image, (32, 32), interpolation=cv2.INTER_CUBIC)
    
    # Create temporal frames
    lr_frames = [lr_image] * 3  # Simple: same frame repeated
    
    # Create RBPN model
    print("Creating RBPN network...")
    model = RBPNNetwork(scale_factor=4, num_frames=3)
    
    # Apply RBPN
    print("Applying RBPN super resolution...")
    input_tensor = prepare_frames_for_network(lr_frames)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    sr_image = tensor_to_image(output_tensor)
    
    print()
    print("Key Advantages of RBPN:")
    print("- Iterative refinement through back-projection")
    print("- Better detail recovery than simple fusion")
    print("- Temporal consistency through recurrent structure")
    print()
    print("Note: This is a simplified demo. Full RBPN includes:")
    print("- Optical flow for temporal alignment")
    print("- Multiple back-projection iterations")
    print("- Dense connections between projections")
    print()
    print("✅ RBPN demonstration complete!")
    print()


if __name__ == "__main__":
    demonstrate_rbpn()

