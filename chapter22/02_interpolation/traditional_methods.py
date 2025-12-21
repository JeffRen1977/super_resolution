"""
Chapter 22 - Section 2: Traditional Interpolation Methods
==========================================================

This module demonstrates traditional upsampling methods (baseline approaches)
that super resolution aims to improve upon.

Learning Objectives:
- Understand interpolation methods (nearest, bilinear, bicubic)
- See their limitations
- Establish baseline for comparison
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple


def nearest_neighbor_upscale(lr_image: np.ndarray, scale_factor: int = 4) -> np.ndarray:
    """
    Nearest neighbor interpolation - simplest method
    
    Each pixel in HR is copied from nearest LR pixel.
    Results in blocky, pixelated appearance.
    
    Args:
        lr_image: Low-resolution image
        scale_factor: Upscaling factor
    
    Returns:
        Upscaled image
    """
    h, w = lr_image.shape[:2]
    new_h, new_w = h * scale_factor, w * scale_factor
    return cv2.resize(lr_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def bilinear_upscale(lr_image: np.ndarray, scale_factor: int = 4) -> np.ndarray:
    """
    Bilinear interpolation - smooth but blurry
    
    Uses linear interpolation in both directions.
    Better than nearest neighbor but still loses sharpness.
    
    Args:
        lr_image: Low-resolution image
        scale_factor: Upscaling factor
    
    Returns:
        Upscaled image
    """
    h, w = lr_image.shape[:2]
    new_h, new_w = h * scale_factor, w * scale_factor
    return cv2.resize(lr_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def bicubic_upscale(lr_image: np.ndarray, scale_factor: int = 4) -> np.ndarray:
    """
    Bicubic interpolation - best traditional method
    
    Uses cubic interpolation for smoother results.
    This is the standard baseline for super resolution.
    
    Args:
        lr_image: Low-resolution image
        scale_factor: Upscaling factor
    
    Returns:
        Upscaled image
    """
    h, w = lr_image.shape[:2]
    new_h, new_w = h * scale_factor, w * scale_factor
    return cv2.resize(lr_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def compare_interpolation_methods(lr_image: np.ndarray, hr_ground_truth: np.ndarray,
                                 scale_factor: int = 4, save_path: str = None):
    """
    Compare all interpolation methods side-by-side
    
    Args:
        lr_image: Low-resolution input
        hr_ground_truth: High-resolution ground truth (for comparison)
        scale_factor: Upscaling factor
        save_path: Path to save comparison image
    """
    # Apply all methods
    nearest = nearest_neighbor_upscale(lr_image, scale_factor)
    bilinear = bilinear_upscale(lr_image, scale_factor)
    bicubic = bicubic_upscale(lr_image, scale_factor)
    
    # Create comparison grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Ground truth
    axes[0, 0].imshow(cv2.cvtColor(hr_ground_truth, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Ground Truth (HR)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Nearest neighbor
    axes[0, 1].imshow(cv2.cvtColor(nearest, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Nearest Neighbor\n(Blocky, Pixelated)', fontsize=12)
    axes[0, 1].axis('off')
    
    # Bilinear
    axes[1, 0].imshow(cv2.cvtColor(bilinear, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Bilinear\n(Smooth but Blurry)', fontsize=12)
    axes[1, 0].axis('off')
    
    # Bicubic
    axes[1, 1].imshow(cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Bicubic (Baseline)\n(Best Traditional Method)', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'nearest': nearest,
        'bilinear': bilinear,
        'bicubic': bicubic
    }


def demonstrate_interpolation():
    """
    Demonstrate traditional interpolation methods
    """
    print("=" * 60)
    print("Chapter 22 - Section 2: Traditional Interpolation Methods")
    print("=" * 60)
    print()
    
    # Create test image
    print("Creating test image...")
    hr_image = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(hr_image, (20, 20), (60, 60), (255, 255, 255), -1)
    cv2.circle(hr_image, (90, 40), 15, (0, 255, 0), -1)
    cv2.putText(hr_image, 'TEST', (30, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Create low-resolution version
    print("Creating low-resolution version...")
    lr_image = cv2.resize(hr_image, (32, 32), interpolation=cv2.INTER_CUBIC)
    
    # Compare methods
    print("Comparing interpolation methods...")
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, '02_interpolation_comparison.png')
    results = compare_interpolation_methods(lr_image, hr_image, scale_factor=4,
                                          save_path=save_path)
    
    print()
    print("Key Observations:")
    print("- Nearest Neighbor: Fast but blocky")
    print("- Bilinear: Smooth but loses sharpness")
    print("- Bicubic: Best traditional method (used as baseline)")
    print("- All methods: Cannot recover lost high-frequency details")
    print()
    print("✅ Interpolation demonstration complete!")
    print()


if __name__ == "__main__":
    demonstrate_interpolation()

