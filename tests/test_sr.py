"""
Pytest tests for the super_resolution module (Chapter 22).

Tests the SimpleRBPN model architecture and utility functions.
Skips tests that require scikit-image if it is not installed.
"""

import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import skimage  # noqa: F401
    _has_skimage = True
except ImportError:
    _has_skimage = False

requires_skimage = pytest.mark.skipif(not _has_skimage, reason="scikit-image not installed")


class TestSimpleRBPN:
    """Test the SimpleRBPN neural network architecture."""

    def _make_model(self, scale=4):
        from super_resolution_demo import SimpleRBPN
        return SimpleRBPN(scale_factor=scale, num_frames=3)

    def test_model_creation(self):
        model = self._make_model(scale=4)
        assert model.scale_factor == 4
        assert model.num_frames == 3

    def test_forward_shape_4x(self):
        model = self._make_model(scale=4)
        model.eval()
        inp = torch.randn(1, 3, 3, 16, 16)
        with torch.no_grad():
            out = model(inp)
        assert out.shape == (1, 3, 64, 64)

    def test_forward_shape_2x(self):
        model = self._make_model(scale=2)
        model.eval()
        inp = torch.randn(1, 3, 3, 16, 16)
        with torch.no_grad():
            out = model(inp)
        assert out.shape == (1, 3, 32, 32)

    def test_output_range(self):
        model = self._make_model(scale=2)
        model.eval()
        inp = torch.rand(1, 3, 3, 8, 8)
        with torch.no_grad():
            out = model(inp)
        assert out.dtype == torch.float32

    def test_batch_size(self):
        model = self._make_model(scale=2)
        model.eval()
        inp = torch.randn(2, 3, 3, 8, 8)
        with torch.no_grad():
            out = model(inp)
        assert out.shape[0] == 2


@requires_skimage
class TestSuperResolutionDemo:
    """Test the full demo pipeline (requires scikit-image)."""

    def _make_demo(self, scale=2):
        from super_resolution_demo import SuperResolutionDemo
        return SuperResolutionDemo(scale_factor=scale, device="cpu")

    def test_create_test_image(self):
        demo = self._make_demo()
        img = demo.create_test_image((32, 32))
        assert img.shape == (32, 32, 3)
        assert img.dtype == np.uint8

    def test_bicubic_upscale(self):
        demo = self._make_demo(scale=2)
        lr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        hr = demo.bicubic_upscale(lr)
        assert hr.shape == (32, 32, 3)

    def test_create_lr_frames(self):
        demo = self._make_demo(scale=2)
        hr = demo.create_test_image((32, 32))
        frames = demo.create_lr_frames(hr, num_frames=3)
        assert len(frames) == 3
        assert frames[0].shape == (16, 16, 3)

    def test_super_resolve(self):
        demo = self._make_demo(scale=2)
        lr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        result = demo.super_resolve([lr, lr, lr])
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_evaluate_metrics(self):
        demo = self._make_demo(scale=2)
        img1 = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
        img2 = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
        metrics = demo.evaluate_metrics(img1, img2)
        assert "PSNR" in metrics
        assert "SSIM" in metrics
        assert metrics["PSNR"] > 0


@requires_skimage
class TestEvaluationMetrics:
    """Test the evaluation metrics module (requires scikit-image)."""

    def _make_evaluator(self):
        from eval_metrics import EvaluationMetrics
        return EvaluationMetrics(device="cpu")

    def test_psnr_identical(self):
        evaluator = self._make_evaluator()
        img = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
        val = evaluator.calculate_psnr(img, img)
        assert val > 50

    def test_ssim_identical(self):
        evaluator = self._make_evaluator()
        img = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
        val = evaluator.calculate_ssim(img, img)
        assert val > 0.99

    def test_lpips_identical(self):
        evaluator = self._make_evaluator()
        img = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
        val = evaluator.calculate_lpips(img, img)
        assert val < 0.01

    def test_temporal_consistency_single(self):
        evaluator = self._make_evaluator()
        frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        val = evaluator.calculate_temporal_consistency([frame])
        assert val == 0.0

    def test_temporal_consistency_identical_frames(self):
        evaluator = self._make_evaluator()
        frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        val = evaluator.calculate_temporal_consistency([frame, frame, frame])
        assert val == 0.0
