#!/usr/bin/env python3
"""
Super Resolution Demo - RBPN Style Implementation
Based on the reference document for video super resolution

This demo shows:
1. How to implement a simplified RBPN-style super resolution
2. Comparison with bicubic baseline
3. Evaluation using PSNR and SSIM metrics
4. Practical usage examples
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import color
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import argparse


class SimpleRBPN(nn.Module):
    """
    Simplified RBPN (Recurrent Back-Projection Network) implementation
    This is a minimal version for demonstration purposes
    """
    def __init__(self, scale_factor: int = 4, num_frames: int = 3):
        super(SimpleRBPN, self).__init__()
        self.scale_factor = scale_factor
        self.num_frames = num_frames
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Back-projection blocks (simplified)
        self.back_projection = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Reconstruction
        self.reconstruction = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3 * scale_factor * scale_factor, 3, padding=1)
        )
        
        # Pixel shuffle for upsampling
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RBPN
        Args:
            frames: (B, T, C, H, W) - batch of video frames
        Returns:
            super_resolved: (B, C, H*scale, W*scale)
        """
        B, T, C, H, W = frames.shape
        
        # Process each frame
        features = []
        for t in range(T):
            feat = self.feature_extractor(frames[:, t])  # (B, 64, H, W)
            features.append(feat)
        
        # Simple temporal fusion (average for demo)
        fused_feature = torch.stack(features, dim=1).mean(dim=1)  # (B, 64, H, W)
        
        # Back-projection
        projected = self.back_projection(fused_feature)
        
        # Reconstruction and upsampling
        reconstructed = self.reconstruction(projected)
        super_resolved = self.pixel_shuffle(reconstructed)
        
        return super_resolved


class SuperResolutionDemo:
    """
    Main demo class for super resolution
    """
    def __init__(self, scale_factor: int = 4, device: str = 'cpu'):
        self.scale_factor = scale_factor
        self.device = device
        self.model = SimpleRBPN(scale_factor=scale_factor).to(device)
        
    def bicubic_upscale(self, image: np.ndarray) -> np.ndarray:
        """
        Bicubic interpolation baseline
        """
        h, w = image.shape[:2]
        new_h, new_w = h * self.scale_factor, w * self.scale_factor
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    def prepare_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Prepare frames for model input
        """
        # Convert to tensor and normalize
        tensor_frames = []
        for frame in frames:
            frame_tensor = torch.from_numpy(frame).float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
            tensor_frames.append(frame_tensor)
        
        # Stack frames: (T, C, H, W) -> (1, T, C, H, W)
        stacked = torch.stack(tensor_frames, dim=0).unsqueeze(0)
        return stacked.to(self.device)
    
    def tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor back to image
        """
        # Move to CPU and denormalize
        image = tensor.squeeze(0).cpu().detach()
        image = image.permute(1, 2, 0)  # CHW -> HWC
        image = torch.clamp(image * 255.0, 0, 255).numpy().astype(np.uint8)
        return image
    
    def super_resolve(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Perform super resolution on frames
        """
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.prepare_frames(frames)
            output_tensor = self.model(input_tensor)
            return self.tensor_to_image(output_tensor)
    
    def evaluate_metrics(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        """
        Calculate PSNR and SSIM metrics
        """
        # Convert to Y channel for evaluation
        pred_y = color.rgb2ycbcr(pred)[:, :, 0]
        gt_y = color.rgb2ycbcr(gt)[:, :, 0]
        
        psnr_value = psnr(gt_y, pred_y, data_range=255)
        ssim_value = ssim(gt_y, pred_y, data_range=255)
        
        return {
            'PSNR': psnr_value,
            'SSIM': ssim_value
        }
    
    def create_test_image(self, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """
        Create a test image with patterns for demonstration
        """
        h, w = size
        image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create some patterns
        # Checkerboard pattern
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                if (i // 8 + j // 8) % 2 == 0:
                    image[i:i+8, j:j+8] = [255, 255, 255]
        
        # Add some circles
        cv2.circle(image, (w//4, h//4), 10, (255, 0, 0), -1)
        cv2.circle(image, (3*w//4, 3*h//4), 15, (0, 255, 0), -1)
        
        # Add text
        cv2.putText(image, 'SR Demo', (w//2-30, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return image
    
    def create_lr_frames(self, hr_image: np.ndarray, num_frames: int = 3) -> List[np.ndarray]:
        """
        Create low-resolution frames by downsampling
        """
        # Downsample by scale factor
        h, w = hr_image.shape[:2]
        lr_h, lr_w = h // self.scale_factor, w // self.scale_factor
        lr_image = cv2.resize(hr_image, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        
        # Create multiple frames (slightly different for temporal consistency demo)
        frames = []
        for i in range(num_frames):
            # Add slight noise/variation to simulate temporal frames
            noise = np.random.normal(0, 2, lr_image.shape).astype(np.uint8)
            frame = np.clip(lr_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            frames.append(frame)
        
        return frames
    
    def run_demo(self, save_results: bool = True) -> dict:
        """
        Run the complete super resolution demo
        """
        print("🎬 Starting Super Resolution Demo...")
        print(f"Scale factor: {self.scale_factor}x")
        print(f"Device: {self.device}")
        
        # Create test image
        print("\n📸 Creating test image...")
        hr_image = self.create_test_image((128, 128))
        
        # Create low-resolution frames
        print("📉 Creating low-resolution frames...")
        lr_frames = self.create_lr_frames(hr_image, num_frames=3)
        
        # Bicubic baseline
        print("🔄 Running bicubic baseline...")
        bicubic_result = self.bicubic_upscale(lr_frames[0])
        
        # Super resolution
        print("🚀 Running RBPN-style super resolution...")
        sr_result = self.super_resolve(lr_frames)
        
        # Evaluate metrics
        print("📊 Evaluating results...")
        bicubic_metrics = self.evaluate_metrics(bicubic_result, hr_image)
        sr_metrics = self.evaluate_metrics(sr_result, hr_image)
        
        # Print results
        print("\n" + "="*50)
        print("📈 RESULTS COMPARISON")
        print("="*50)
        print(f"Bicubic PSNR: {bicubic_metrics['PSNR']:.2f} dB")
        print(f"Bicubic SSIM: {bicubic_metrics['SSIM']:.4f}")
        print(f"RBPN PSNR:    {sr_metrics['PSNR']:.2f} dB")
        print(f"RBPN SSIM:    {sr_metrics['SSIM']:.4f}")
        print(f"PSNR Improvement: {sr_metrics['PSNR'] - bicubic_metrics['PSNR']:.2f} dB")
        print(f"SSIM Improvement: {sr_metrics['SSIM'] - bicubic_metrics['SSIM']:.4f}")
        
        # Save results if requested
        if save_results:
            self.save_comparison_images(hr_image, lr_frames[0], bicubic_result, sr_result)
        
        return {
            'bicubic_metrics': bicubic_metrics,
            'sr_metrics': sr_metrics,
            'hr_image': hr_image,
            'lr_image': lr_frames[0],
            'bicubic_result': bicubic_result,
            'sr_result': sr_result
        }
    
    def save_comparison_images(self, hr: np.ndarray, lr: np.ndarray, 
                             bicubic: np.ndarray, sr: np.ndarray):
        """
        Save comparison images
        """
        os.makedirs('results', exist_ok=True)
        
        # Save individual images
        cv2.imwrite('results/01_ground_truth.png', hr)
        cv2.imwrite('results/02_low_resolution.png', lr)
        cv2.imwrite('results/03_bicubic.png', bicubic)
        cv2.imwrite('results/04_super_resolution.png', sr)
        
        # Create comparison grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].imshow(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Ground Truth (HR)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Low Resolution')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Bicubic Interpolation')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(sr, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('RBPN Super Resolution')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("💾 Results saved to 'results/' directory")


def main():
    """
    Main function to run the demo
    """
    parser = argparse.ArgumentParser(description='Super Resolution Demo')
    parser.add_argument('--scale', type=int, default=4, help='Scale factor (default: 4)')
    parser.add_argument('--device', type=str, default='cpu', 
                       choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--save', action='store_true', help='Save results to files')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Run demo
    demo = SuperResolutionDemo(scale_factor=args.scale, device=args.device)
    results = demo.run_demo(save_results=args.save)
    
    print("\n✅ Demo completed successfully!")
    print("Check the 'results/' directory for saved images and comparison plots.")


if __name__ == "__main__":
    main()


