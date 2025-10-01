#!/usr/bin/env python3
"""
Example script showing how to use the super resolution demo
This demonstrates various ways to use the super resolution functionality
"""

import cv2
import numpy as np
import os
from super_resolution_demo import SuperResolutionDemo
from eval_metrics import EvaluationMetrics


def example_1_basic_demo():
    """
    Example 1: Basic demo with default settings
    """
    print("🎬 Example 1: Basic Demo")
    print("-" * 40)
    
    # Initialize demo
    demo = SuperResolutionDemo(scale_factor=4, device='cpu')
    
    # Run the complete demo
    results = demo.run_demo(save_results=True)
    
    print("✅ Basic demo completed!")
    return results


def example_2_custom_image():
    """
    Example 2: Using custom images
    """
    print("\n🎨 Example 2: Custom Image Processing")
    print("-" * 40)
    
    # Initialize demo
    demo = SuperResolutionDemo(scale_factor=4, device='cpu')
    
    # Create a more complex test image
    hr_image = create_complex_test_image()
    
    # Create low-resolution frames
    lr_frames = demo.create_lr_frames(hr_image, num_frames=3)
    
    # Apply super resolution
    sr_result = demo.super_resolve(lr_frames)
    
    # Apply bicubic for comparison
    bicubic_result = demo.bicubic_upscale(lr_frames[0])
    
    # Evaluate
    evaluator = EvaluationMetrics()
    bicubic_metrics = evaluator.evaluate_single_image(bicubic_result, hr_image)
    sr_metrics = evaluator.evaluate_single_image(sr_result, hr_image)
    
    # Print results
    print(f"Bicubic PSNR: {bicubic_metrics['PSNR']:.2f} dB")
    print(f"RBPN PSNR:    {sr_metrics['PSNR']:.2f} dB")
    print(f"Improvement:  {sr_metrics['PSNR'] - bicubic_metrics['PSNR']:.2f} dB")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    cv2.imwrite('results/custom_hr.png', hr_image)
    cv2.imwrite('results/custom_lr.png', lr_frames[0])
    cv2.imwrite('results/custom_bicubic.png', bicubic_result)
    cv2.imwrite('results/custom_sr.png', sr_result)
    
    print("✅ Custom image processing completed!")
    return {
        'hr_image': hr_image,
        'lr_frames': lr_frames,
        'bicubic_result': bicubic_result,
        'sr_result': sr_result,
        'metrics': {'bicubic': bicubic_metrics, 'sr': sr_metrics}
    }


def example_3_different_scales():
    """
    Example 3: Testing different scale factors
    """
    print("\n📏 Example 3: Different Scale Factors")
    print("-" * 40)
    
    # Test different scale factors
    scales = [2, 4, 8]
    results = {}
    
    for scale in scales:
        print(f"\nTesting scale factor: {scale}x")
        
        # Initialize demo with current scale
        demo = SuperResolutionDemo(scale_factor=scale, device='cpu')
        
        # Create test image
        hr_image = demo.create_test_image((64, 64))
        lr_frames = demo.create_lr_frames(hr_image, num_frames=3)
        
        # Apply methods
        bicubic_result = demo.bicubic_upscale(lr_frames[0])
        sr_result = demo.super_resolve(lr_frames)
        
        # Evaluate
        evaluator = EvaluationMetrics()
        bicubic_metrics = evaluator.evaluate_single_image(bicubic_result, hr_image)
        sr_metrics = evaluator.evaluate_single_image(sr_result, hr_image)
        
        results[scale] = {
            'bicubic_psnr': bicubic_metrics['PSNR'],
            'sr_psnr': sr_metrics['PSNR'],
            'improvement': sr_metrics['PSNR'] - bicubic_metrics['PSNR']
        }
        
        print(f"  Bicubic PSNR: {bicubic_metrics['PSNR']:.2f} dB")
        print(f"  RBPN PSNR:    {sr_metrics['PSNR']:.2f} dB")
        print(f"  Improvement:  {sr_metrics['PSNR'] - bicubic_metrics['PSNR']:.2f} dB")
    
    # Print summary
    print(f"\n📊 Scale Factor Comparison:")
    print(f"{'Scale':<8} {'Bicubic':<10} {'RBPN':<10} {'Improvement':<12}")
    print("-" * 45)
    for scale, metrics in results.items():
        print(f"{scale}x{'':<5} {metrics['bicubic_psnr']:<10.2f} {metrics['sr_psnr']:<10.2f} {metrics['improvement']:<12.2f}")
    
    print("✅ Scale factor testing completed!")
    return results


def example_4_temporal_consistency():
    """
    Example 4: Testing temporal consistency
    """
    print("\n🎞️ Example 4: Temporal Consistency")
    print("-" * 40)
    
    # Initialize demo
    demo = SuperResolutionDemo(scale_factor=4, device='cpu')
    evaluator = EvaluationMetrics()
    
    # Create a sequence of frames
    hr_frames = []
    for i in range(5):
        # Create slightly different frames to simulate motion
        hr_image = demo.create_test_image((64, 64))
        # Add slight translation
        M = np.float32([[1, 0, i*2], [0, 1, i*2]])
        hr_image = cv2.warpAffine(hr_image, M, (64, 64))
        hr_frames.append(hr_image)
    
    # Create low-resolution frames
    lr_frames = [demo.create_lr_frames(hr_frame, num_frames=1)[0] for hr_frame in hr_frames]
    
    # Apply super resolution to each frame
    sr_frames = []
    for i in range(len(lr_frames)):
        # Use context frames for better temporal consistency
        context_frames = lr_frames[max(0, i-1):min(len(lr_frames), i+2)]
        if len(context_frames) < 3:
            # Pad with the same frame if needed
            while len(context_frames) < 3:
                context_frames.append(context_frames[-1])
        
        sr_result = demo.super_resolve(context_frames)
        sr_frames.append(sr_result)
    
    # Calculate temporal consistency
    hr_consistency = evaluator.calculate_temporal_consistency(hr_frames)
    sr_consistency = evaluator.calculate_temporal_consistency(sr_frames)
    
    print(f"Ground truth temporal consistency: {hr_consistency:.4f}")
    print(f"Super resolution temporal consistency: {sr_consistency:.4f}")
    print(f"Consistency ratio: {sr_consistency/hr_consistency:.4f}")
    
    # Save frame sequence
    os.makedirs('results/temporal', exist_ok=True)
    for i, (hr, sr) in enumerate(zip(hr_frames, sr_frames)):
        cv2.imwrite(f'results/temporal/frame_{i:02d}_hr.png', hr)
        cv2.imwrite(f'results/temporal/frame_{i:02d}_sr.png', sr)
    
    print("✅ Temporal consistency testing completed!")
    return {
        'hr_consistency': hr_consistency,
        'sr_consistency': sr_consistency,
        'consistency_ratio': sr_consistency/hr_consistency
    }


def create_complex_test_image(size=(128, 128)):
    """
    Create a more complex test image with various patterns
    """
    h, w = size
    image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Background gradient
    for i in range(h):
        for j in range(w):
            image[i, j] = [int(255 * i / h), int(255 * j / w), 128]
    
    # Add geometric shapes
    cv2.circle(image, (w//4, h//4), 20, (255, 255, 255), -1)
    cv2.rectangle(image, (3*w//4-20, 3*h//4-20), (3*w//4+20, 3*h//4+20), (0, 255, 0), -1)
    
    # Add text
    cv2.putText(image, 'SUPER', (w//2-40, h//2-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(image, 'RESOLUTION', (w//2-50, h//2+10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Add noise pattern
    noise = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image


def main():
    """
    Run all examples
    """
    print("🚀 Super Resolution Examples")
    print("=" * 50)
    
    # Run all examples
    example_1_basic_demo()
    example_2_custom_image()
    example_3_different_scales()
    example_4_temporal_consistency()
    
    print("\n🎉 All examples completed successfully!")
    print("Check the 'results/' directory for all output files.")


if __name__ == "__main__":
    main()

