"""
Chapter 22 - Section 6: Evaluation Metrics
===========================================

This module provides evaluation metrics for super resolution.

Learning Objectives:
- Understand PSNR and SSIM metrics
- Learn how to evaluate SR quality
- Understand metric interpretation
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import color
import cv2


def calculate_psnr(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    PSNR measures pixel-level accuracy between prediction and ground truth.
    Higher values indicate better quality.
    
    Formula: PSNR = 10 × log₁₀(MAX²/MSE)
    where MAX = 255 for 8-bit images, MSE = mean squared error
    
    Args:
        predicted: Predicted HR image
        ground_truth: Ground truth HR image
    
    Returns:
        PSNR value in dB
    """
    # Convert to Y channel (luminance) for fair comparison
    pred_y = color.rgb2ycbcr(predicted.astype(np.uint8))[:, :, 0]
    gt_y = color.rgb2ycbcr(ground_truth.astype(np.uint8))[:, :, 0]
    
    return psnr(gt_y, pred_y, data_range=255)


def calculate_ssim(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (SSIM)
    
    SSIM measures perceptual similarity including:
    - Luminance (brightness)
    - Contrast (variation)
    - Structure (patterns)
    
    Values range from 0 to 1, higher is better.
    
    Args:
        predicted: Predicted HR image
        ground_truth: Ground truth HR image
    
    Returns:
        SSIM value (0-1)
    """
    pred_y = color.rgb2ycbcr(predicted.astype(np.uint8))[:, :, 0]
    gt_y = color.rgb2ycbcr(ground_truth.astype(np.uint8))[:, :, 0]
    
    return ssim(gt_y, pred_y, data_range=255)


def evaluate_sr_result(predicted: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Comprehensive evaluation of super resolution result
    
    Args:
        predicted: Predicted HR image
        ground_truth: Ground truth HR image
    
    Returns:
        Dictionary with all metrics
    """
    # Ensure same dimensions
    if predicted.shape != ground_truth.shape:
        predicted = cv2.resize(predicted, 
                              (ground_truth.shape[1], ground_truth.shape[0]))
    
    metrics = {
        'PSNR': calculate_psnr(predicted, ground_truth),
        'SSIM': calculate_ssim(predicted, ground_truth)
    }
    
    return metrics


def interpret_psnr(psnr_value: float) -> str:
    """
    Interpret PSNR value
    
    Returns:
        Quality description
    """
    if psnr_value < 25:
        return "Poor quality"
    elif psnr_value < 30:
        return "Acceptable (bicubic level)"
    elif psnr_value < 35:
        return "Good (trained SR)"
    elif psnr_value < 40:
        return "Excellent"
    else:
        return "Near-perfect"


def interpret_ssim(ssim_value: float) -> str:
    """
    Interpret SSIM value
    
    Returns:
        Quality description
    """
    if ssim_value < 0.5:
        return "Very poor"
    elif ssim_value < 0.7:
        return "Poor to fair"
    elif ssim_value < 0.85:
        return "Good (bicubic level)"
    elif ssim_value < 0.95:
        return "Very good"
    else:
        return "Excellent"


def print_evaluation_results(metrics: dict, method_name: str = "Method"):
    """
    Print evaluation results in a formatted way
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results - {method_name}")
    print(f"{'='*60}")
    print(f"PSNR: {metrics['PSNR']:.2f} dB - {interpret_psnr(metrics['PSNR'])}")
    print(f"SSIM: {metrics['SSIM']:.4f} - {interpret_ssim(metrics['SSIM'])}")
    print(f"{'='*60}\n")

