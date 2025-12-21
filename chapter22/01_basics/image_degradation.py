"""
Chapter 22 - Section 1: Understanding Image Degradation
========================================================

This module demonstrates how high-resolution images are degraded to low-resolution,
which is the fundamental problem that super resolution aims to solve.

Learning Objectives:
- Understand the degradation process (downsampling + blur)
- Visualize the information loss
- Prepare for super resolution reconstruction
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple


def downsample_image(image: np.ndarray, scale_factor: int = 4, 
                    method: str = 'bicubic') -> np.ndarray:
    """
    Simulate the degradation process: downsample a high-resolution image
    
    This represents what happens when we capture or store an image at lower resolution.
    
    Args:
        image: High-resolution image (H, W, C)
        scale_factor: Downsampling factor (e.g., 4 means 4× smaller)
        method: Interpolation method ('bicubic', 'bilinear', 'nearest')
    
    Returns:
        Low-resolution image
    """
    h, w = image.shape[:2]
    new_h, new_w = h // scale_factor, w // scale_factor
    
    if method == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    elif method == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_NEAREST
    
    # Downsample (this simulates what happens in real cameras/sensors)
    lr_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    return lr_image


def add_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Add blur to simulate optical degradation
    
    Real cameras have optical blur due to:
    - Lens imperfections
    - Motion blur
    - Focus issues
    
    Args:
        image: Input image
        kernel_size: Size of blur kernel
    
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_noise(image: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """
    Add noise to simulate sensor noise
    
    Args:
        image: Input image (0-255)
        noise_level: Standard deviation of noise
    
    Returns:
        Noisy image
    """
    noise = np.random.normal(0, noise_level * 255, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def create_degraded_image(hr_image: np.ndarray, scale_factor: int = 4,
                         add_blur_flag: bool = True,
                         add_noise_flag: bool = False) -> np.ndarray:
    """
    Complete degradation pipeline
    
    This simulates the real-world process:
    1. Optical blur (optional)
    2. Downsampling (sensor resolution)
    3. Noise (sensor noise, optional)
    
    Args:
        hr_image: High-resolution image
        scale_factor: Downsampling factor
        add_blur_flag: Whether to add blur
        add_noise_flag: Whether to add noise
    
    Returns:
        Degraded low-resolution image
    """
    degraded = hr_image.copy()
    
    # Step 1: Optical blur (happens before downsampling)
    if add_blur_flag:
        degraded = add_blur(degraded, kernel_size=5)
    
    # Step 2: Downsampling (sensor captures at lower resolution)
    degraded = downsample_image(degraded, scale_factor=scale_factor)
    
    # Step 3: Sensor noise (optional)
    if add_noise_flag:
        degraded = add_noise(degraded, noise_level=0.01)
    
    return degraded


def visualize_degradation(hr_image: np.ndarray, lr_image: np.ndarray,
                         save_path: str = None):
    """
    Visualize the degradation process
    
    Shows side-by-side comparison of HR and LR images
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'High-Resolution\n{hr_image.shape[1]}×{hr_image.shape[0]}', 
                     fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Low-Resolution (Degraded)\n{lr_image.shape[1]}×{lr_image.shape[0]}', 
                     fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def demonstrate_degradation():
    """
    Complete demonstration of image degradation
    
    This function shows the fundamental problem super resolution solves
    """
    print("=" * 60)
    print("Chapter 22 - Section 1: Understanding Image Degradation")
    print("=" * 60)
    print()
    
    # Create a test image
    print("Creating test image...")
    test_image = np.zeros((128, 128, 3), dtype=np.uint8)
    
    # Add patterns to see degradation effects
    cv2.rectangle(test_image, (20, 20), (60, 60), (255, 255, 255), -1)
    cv2.circle(test_image, (90, 40), 15, (0, 255, 0), -1)
    cv2.putText(test_image, 'HR', (40, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Degrade the image
    print("Degrading image (4× downsampling)...")
    degraded = create_degraded_image(test_image, scale_factor=4, 
                                    add_blur_flag=True, add_noise_flag=False)
    
    # Visualize
    print("Visualizing degradation...")
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, '01_degradation_demo.png')
    visualize_degradation(test_image, degraded, save_path=save_path)
    
    print()
    print("Key Insight:")
    print("- High-resolution image: 128×128 pixels")
    print("- Low-resolution image: 32×32 pixels")
    print("- Information loss: 16× fewer pixels!")
    print("- Super resolution goal: Recover lost details")
    print()
    print("✅ Degradation demonstration complete!")
    print()


if __name__ == "__main__":
    demonstrate_degradation()

