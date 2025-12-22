"""
Chapter 22 - Section 4: Video Super Resolution (VSR)
=====================================================

This module demonstrates how to leverage temporal information from multiple
video frames to improve super resolution quality.

Learning Objectives:
- Understand why video SR is better than frame-by-frame SISR
- Learn temporal fusion strategies
- See the importance of temporal consistency
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Tuple


class TemporalFusionSR(nn.Module):
    """
    Simple Video Super Resolution Network
    
    Key idea: Use information from multiple frames (past, current, future)
    to better reconstruct the current frame.
    
    Advantages over SISR:
    1. Sub-pixel motion provides different sampling positions
    2. Temporal redundancy helps disambiguate uncertain reconstructions
    3. Better temporal consistency (reduces flickering)
    """
    
    def __init__(self, scale_factor: int = 4, num_frames: int = 3):
        super(TemporalFusionSR, self).__init__()
        self.scale_factor = scale_factor
        self.num_frames = num_frames
        
        # Feature extraction for each frame
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Temporal fusion: Combine features from multiple frames
        # Simple approach: Average features (more advanced methods use
        # optical flow or attention mechanisms)
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Reconstruction and upsampling
        out_channels = 3 * (scale_factor ** 2)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temporal fusion
        
        Args:
            frames: (B, T, C, H, W) - batch of T frames
        
        Returns:
            Super-resolved frame (B, C, H×scale, W×scale)
        """
        B, T, C, H, W = frames.shape
        
        # Extract features from each frame
        features = []
        for t in range(T):
            feat = self.feature_extractor(frames[:, t])
            features.append(feat)
        
        # Temporal fusion: Combine information across frames
        # Simple average (in practice, use optical flow or attention)
        fused_features = torch.stack(features, dim=1).mean(dim=1)
        
        # Refine fused features
        refined = self.temporal_fusion(fused_features)
        
        # Reconstruct HR image
        hr_image = self.reconstruction(refined)
        
        return hr_image


def prepare_frames_for_network(frames: List[np.ndarray]) -> torch.Tensor:
    """
    Prepare multiple frames for network input
    
    Args:
        frames: List of images (H, W, C), values 0-255
    
    Returns:
        Tensor (1, T, C, H, W), values 0-1
    """
    tensor_frames = []
    for frame in frames:
        # Normalize and convert HWC to CHW
        tensor = torch.from_numpy(frame).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        tensor_frames.append(tensor)
    
    # Stack along temporal dimension and add batch dimension
    stacked = torch.stack(tensor_frames, dim=0).unsqueeze(0)
    
    return stacked


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to image"""
    image = tensor.squeeze(0).permute(1, 2, 0)
    image = torch.clamp(image * 255.0, 0, 255).numpy().astype(np.uint8)
    return image


def create_temporal_frames(base_image: np.ndarray, num_frames: int = 3) -> List[np.ndarray]:
    """
    Create temporal frames with slight variations
    
    In real videos, frames have small motions. This simulates that.
    """
    frames = []
    for i in range(num_frames):
        # Add slight translation to simulate motion
        M = np.float32([[1, 0, i*0.5], [0, 1, i*0.5]])
        frame = cv2.warpAffine(base_image, M, 
                              (base_image.shape[1], base_image.shape[0]))
        frames.append(frame)
    return frames


def demonstrate_video_sr():
    """
    Demonstrate video super resolution with temporal fusion
    """
    print("=" * 60)
    print("Chapter 22 - Section 4: Video Super Resolution")
    print("=" * 60)
    print()
    
    print("Key Concept:")
    print("- Single Image SR: Uses only one frame")
    print("- Video SR: Uses multiple frames (temporal information)")
    print("- Advantage: Can recover details from sub-pixel motion")
    print()
    
    # Create test image
    print("Creating test image...")
    hr_image = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(hr_image, (20, 20), (60, 60), (255, 255, 255), -1)
    cv2.circle(hr_image, (90, 40), 15, (0, 255, 0), -1)
    cv2.putText(hr_image, 'VSR', (40, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Create low-resolution version
    lr_image = cv2.resize(hr_image, (32, 32), interpolation=cv2.INTER_CUBIC)
    
    # Create temporal frames (simulate video)
    print("Creating temporal frames...")
    lr_frames = create_temporal_frames(lr_image, num_frames=3)
    
    # Create model
    print("Creating VSR network...")
    model = TemporalFusionSR(scale_factor=4, num_frames=3)
    
    # Apply VSR
    print("Applying video super resolution...")
    input_tensor = prepare_frames_for_network(lr_frames)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    sr_image = tensor_to_image(output_tensor)
    
    print()
    print("Key Insight:")
    print("- Used 3 frames instead of 1")
    print("- Temporal fusion combines information from all frames")
    print("- Better quality than frame-by-frame SISR")
    print()
    print("✅ Video SR demonstration complete!")
    print()


if __name__ == "__main__":
    demonstrate_video_sr()


