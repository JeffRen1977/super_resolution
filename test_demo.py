#!/usr/bin/env python3
"""
Quick test script to verify super resolution demo works
This is a minimal test to ensure everything is functioning correctly
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        print("✅ scikit-image imported successfully")
    except ImportError as e:
        print(f"❌ scikit-image import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    return True


def test_super_resolution():
    """Test the super resolution functionality"""
    print("\nTesting super resolution functionality...")
    
    try:
        from super_resolution_demo import SuperResolutionDemo
        
        # Initialize demo
        demo = SuperResolutionDemo(scale_factor=2, device='cpu')  # Use smaller scale for faster test
        print("✅ SuperResolutionDemo initialized")
        
        # Create test image
        test_image = demo.create_test_image((32, 32))  # Small image for quick test
        print("✅ Test image created")
        
        # Create LR frames
        lr_frames = demo.create_lr_frames(test_image, num_frames=3)
        print("✅ Low-resolution frames created")
        
        # Test bicubic upscaling
        bicubic_result = demo.bicubic_upscale(lr_frames[0])
        print("✅ Bicubic upscaling completed")
        
        # Test super resolution
        sr_result = demo.super_resolve(lr_frames)
        print("✅ Super resolution completed")
        
        # Test evaluation
        from eval_metrics import EvaluationMetrics
        evaluator = EvaluationMetrics()
        metrics = evaluator.evaluate_single_image(sr_result, test_image)
        print(f"✅ Evaluation completed - PSNR: {metrics['PSNR']:.2f} dB, SSIM: {metrics['SSIM']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Super resolution test failed: {e}")
        return False


def test_evaluation_metrics():
    """Test the evaluation metrics"""
    print("\nTesting evaluation metrics...")
    
    try:
        from eval_metrics import EvaluationMetrics
        import numpy as np
        
        evaluator = EvaluationMetrics()
        
        # Create test images
        img1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        noise = np.random.randint(-10, 10, (64, 64, 3), dtype=np.int16)
        img2 = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Test metrics
        psnr_val = evaluator.calculate_psnr(img1, img2)
        ssim_val = evaluator.calculate_ssim(img1, img2)
        lpips_val = evaluator.calculate_lpips(img1, img2)
        
        print(f"✅ PSNR calculation: {psnr_val:.2f} dB")
        print(f"✅ SSIM calculation: {ssim_val:.4f}")
        print(f"✅ LPIPS calculation: {lpips_val:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation metrics test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Super Resolution Demo Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install required dependencies:")
        print("pip install -r requirements.txt")
        return False
    
    # Test evaluation metrics
    if not test_evaluation_metrics():
        print("\n❌ Evaluation metrics test failed.")
        return False
    
    # Test super resolution
    if not test_super_resolution():
        print("\n❌ Super resolution test failed.")
        return False
    
    print("\n🎉 All tests passed! The super resolution demo is working correctly.")
    print("\nTo run the full demo:")
    print("python super_resolution_demo.py --save")
    print("\nTo run examples:")
    print("python run_example.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
