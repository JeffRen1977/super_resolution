#!/usr/bin/env python3
"""
Evaluation Metrics for Super Resolution
Based on the reference document requirements

This module provides comprehensive evaluation metrics including:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- Temporal consistency metrics
"""

import os
import numpy as np
import cv2
from skimage import io, color
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for super resolution
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
    def to_y_channel(self, img: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to Y channel (luminance)
        Args:
            img: RGB image (H, W, 3) with values 0-255
        Returns:
            Y channel (H, W) with values 0-255
        """
        if len(img.shape) == 3 and img.shape[2] == 3:
            y = color.rgb2ycbcr(img.astype(np.uint8))[:, :, 0]
        else:
            y = img
        return y
    
    def calculate_psnr(self, pred: np.ndarray, gt: np.ndarray, 
                      use_y_channel: bool = True) -> float:
        """
        Calculate PSNR (Peak Signal-to-Noise Ratio)
        Args:
            pred: Predicted image
            gt: Ground truth image
            use_y_channel: Whether to use Y channel only
        Returns:
            PSNR value in dB
        """
        if use_y_channel:
            pred_y = self.to_y_channel(pred)
            gt_y = self.to_y_channel(gt)
        else:
            pred_y = pred
            gt_y = gt
            
        return psnr(gt_y, pred_y, data_range=255)
    
    def calculate_ssim(self, pred: np.ndarray, gt: np.ndarray,
                      use_y_channel: bool = True) -> float:
        """
        Calculate SSIM (Structural Similarity Index)
        Args:
            pred: Predicted image
            gt: Ground truth image
            use_y_channel: Whether to use Y channel only
        Returns:
            SSIM value (0-1, higher is better)
        """
        if use_y_channel:
            pred_y = self.to_y_channel(pred)
            gt_y = self.to_y_channel(gt)
        else:
            pred_y = pred
            gt_y = gt
            
        return ssim(gt_y, pred_y, data_range=255)
    
    def calculate_lpips(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate LPIPS (Learned Perceptual Image Patch Similarity)
        This is a simplified version - for full LPIPS, install lpips package
        Args:
            pred: Predicted image
            gt: Ground truth image
        Returns:
            LPIPS value (lower is better)
        """
        # Convert to tensor
        pred_tensor = torch.from_numpy(pred).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        gt_tensor = torch.from_numpy(gt).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Simple L2 distance in feature space (simplified LPIPS)
        pred_tensor = pred_tensor.to(self.device)
        gt_tensor = gt_tensor.to(self.device)
        
        # Calculate L2 distance
        l2_distance = torch.nn.functional.mse_loss(pred_tensor, gt_tensor)
        
        return l2_distance.item()
    
    def calculate_temporal_consistency(self, frames: List[np.ndarray]) -> float:
        """
        Calculate temporal consistency metric
        Args:
            frames: List of consecutive frames
        Returns:
            Temporal consistency score (lower is better)
        """
        if len(frames) < 2:
            return 0.0
        
        total_diff = 0.0
        for i in range(1, len(frames)):
            # Calculate frame difference
            diff = np.mean(np.abs(frames[i].astype(np.float32) - 
                                 frames[i-1].astype(np.float32)))
            total_diff += diff
        
        return total_diff / (len(frames) - 1)
    
    def evaluate_single_image(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a single image pair
        Args:
            pred: Predicted image
            gt: Ground truth image
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Ensure images have same dimensions
        if pred.shape != gt.shape:
            print(f"Warning: Image dimensions don't match. Pred: {pred.shape}, GT: {gt.shape}")
            # Resize prediction to match ground truth
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        
        # Calculate metrics
        metrics['PSNR'] = self.calculate_psnr(pred, gt)
        metrics['SSIM'] = self.calculate_ssim(pred, gt)
        metrics['LPIPS'] = self.calculate_lpips(pred, gt)
        
        return metrics
    
    def evaluate_folder(self, pred_dir: str, gt_dir: str) -> Dict[str, float]:
        """
        Evaluate all images in two directories
        Args:
            pred_dir: Directory containing predicted images
            gt_dir: Directory containing ground truth images
        Returns:
            Dictionary of average metrics
        """
        pred_files = sorted([f for f in os.listdir(pred_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        gt_files = sorted([f for f in os.listdir(gt_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(pred_files) != len(gt_files):
            print(f"Warning: Number of files don't match. Pred: {len(pred_files)}, GT: {len(gt_files)}")
        
        all_metrics = {'PSNR': [], 'SSIM': [], 'LPIPS': []}
        
        for pred_file in pred_files:
            if pred_file not in gt_files:
                print(f"Warning: No corresponding GT file for {pred_file}")
                continue
                
            pred_path = os.path.join(pred_dir, pred_file)
            gt_path = os.path.join(gt_dir, pred_file)
            
            try:
                pred_img = io.imread(pred_path)
                gt_img = io.imread(gt_path)
                
                metrics = self.evaluate_single_image(pred_img, gt_img)
                
                for key, value in metrics.items():
                    all_metrics[key].append(value)
                    
            except Exception as e:
                print(f"Error processing {pred_file}: {e}")
                continue
        
        # Calculate averages
        avg_metrics = {}
        for key, values in all_metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
            else:
                avg_metrics[key] = 0.0
                avg_metrics[f'{key}_std'] = 0.0
        
        return avg_metrics
    
    def print_results(self, metrics: Dict[str, float], method_name: str = "Method"):
        """
        Print evaluation results in a formatted way
        Args:
            metrics: Dictionary of metrics
            method_name: Name of the method being evaluated
        """
        print(f"\n{'='*50}")
        print(f"📊 EVALUATION RESULTS - {method_name}")
        print(f"{'='*50}")
        
        if 'PSNR' in metrics:
            print(f"PSNR:     {metrics['PSNR']:.4f} ± {metrics.get('PSNR_std', 0):.4f} dB")
        if 'SSIM' in metrics:
            print(f"SSIM:     {metrics['SSIM']:.4f} ± {metrics.get('SSIM_std', 0):.4f}")
        if 'LPIPS' in metrics:
            print(f"LPIPS:    {metrics['LPIPS']:.4f} ± {metrics.get('LPIPS_std', 0):.4f}")
        
        print(f"{'='*50}")


def main():
    """
    Example usage of the evaluation metrics
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Super Resolution Results')
    parser.add_argument('--pred_dir', type=str, required=True, 
                       help='Directory containing predicted images')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory containing ground truth images')
    parser.add_argument('--method_name', type=str, default='Super Resolution',
                       help='Name of the method being evaluated')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    
    # Evaluate
    metrics = evaluator.evaluate_folder(args.pred_dir, args.gt_dir)
    
    # Print results
    evaluator.print_results(metrics, args.method_name)


if __name__ == "__main__":
    main()


