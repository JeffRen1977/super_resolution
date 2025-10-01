# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Setup Environment

```bash
# Make scripts executable
chmod +x *.sh

# Run setup (only needed once)
./setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies (PyTorch, OpenCV, etc.)
- Verify installation

**Expected time:** 2-5 minutes (depending on internet speed)

---

### Step 2: Run Tests

```bash
# Verify everything works
./run_tests.sh
```

This will:
- Test all imports
- Verify evaluation metrics
- Test super resolution functionality

**Expected output:** All tests should pass ✅

---

### Step 3: Run Demo

```bash
# Run basic demo
./run_demo.sh
```

This will:
- Create a test image
- Apply super resolution
- Compare with bicubic baseline
- Save results to `./results/` directory

**Expected time:** 5-10 seconds

---

## 📋 Available Scripts

### Basic Usage

| Script | Purpose | Time | Output |
|--------|---------|------|--------|
| `./setup.sh` | Initial setup (one-time) | 2-5 min | Virtual environment + dependencies |
| `./run_tests.sh` | Verify installation | 10-15 sec | Test results |
| `./run_demo.sh` | Basic demo | 5-10 sec | `./results/` directory |

### Advanced Usage

| Script | Purpose | Time | Output |
|--------|---------|------|--------|
| `./run_examples.sh` | All 4 examples | 30-60 sec | Multiple result sets |
| `./run_gpu.sh` | GPU-accelerated | 2-5 sec | Faster processing |
| `./run_scales.sh` | Test 2x, 4x, 8x | 15-30 sec | Scale comparisons |

---

## 🎯 What Each Script Does

### `setup.sh` - Environment Setup
```bash
./setup.sh
```
- Creates Python virtual environment
- Installs PyTorch, OpenCV, NumPy, etc.
- Verifies Python version
- **Run this first, only once**

### `run_tests.sh` - Test Suite
```bash
./run_tests.sh
```
- Tests imports (PyTorch, OpenCV, etc.)
- Verifies PSNR/SSIM calculations
- Tests super resolution model
- **Run after setup to verify**

### `run_demo.sh` - Basic Demo
```bash
./run_demo.sh
```
- Runs 4x super resolution
- Uses CPU (works on all machines)
- Saves all results
- **Good starting point**

### `run_examples.sh` - All Examples
```bash
./run_examples.sh
```
Runs four comprehensive examples:
1. Basic demo workflow
2. Custom image processing
3. Different scale factors (2x, 4x, 8x)
4. Temporal consistency testing

### `run_gpu.sh` - GPU Acceleration
```bash
./run_gpu.sh
```
- Uses CUDA GPU if available
- Falls back to CPU if no GPU
- 5-10× faster than CPU
- **Use if you have NVIDIA GPU**

### `run_scales.sh` - Scale Comparison
```bash
./run_scales.sh
```
- Tests 2x upscaling (easy)
- Tests 4x upscaling (standard)
- Tests 8x upscaling (hard)
- Saves comparisons

---

## 📊 Expected Results

After running the demo, you'll find in `./results/`:

```
results/
├── 01_ground_truth.png       # Original high-res image
├── 02_low_resolution.png     # Downsampled low-res
├── 03_bicubic.png            # Bicubic interpolation
├── 04_super_resolution.png   # RBPN result
└── comparison.png            # Side-by-side comparison
```

### View Results

```bash
# Open results folder
open results/

# View comparison image
open results/comparison.png
```

---

## 🐍 Direct Python Usage

If you prefer running Python directly:

### Activate Environment
```bash
source venv/bin/activate
```

### Run Demo
```bash
# Basic demo
python super_resolution_demo.py --save

# Custom scale
python super_resolution_demo.py --scale 2 --save

# GPU acceleration
python super_resolution_demo.py --scale 4 --device cuda --save

# All examples
python run_example.py

# Tests
python test_demo.py
```

---

## 🔧 Command-Line Options

### `super_resolution_demo.py`

```bash
python super_resolution_demo.py [OPTIONS]

Options:
  --scale     Scale factor (2, 4, or 8)     Default: 4
  --device    Device (cpu or cuda)          Default: cpu
  --save      Save results to files         Default: False (flag)
```

### Examples

```bash
# 2x upscaling with CPU
python super_resolution_demo.py --scale 2 --device cpu --save

# 4x upscaling with GPU
python super_resolution_demo.py --scale 4 --device cuda --save

# 8x upscaling, don't save
python super_resolution_demo.py --scale 8 --device cpu
```

---

## 📈 Performance Expectations

### Processing Time (per image)

| Device | Scale | Time |
|--------|-------|------|
| CPU (Intel i7) | 2x | ~2 sec |
| CPU (Intel i7) | 4x | ~5 sec |
| CPU (Intel i7) | 8x | ~10 sec |
| GPU (RTX 3080) | 2x | ~0.1 sec |
| GPU (RTX 3080) | 4x | ~0.3 sec |
| GPU (RTX 3080) | 8x | ~0.8 sec |

### Memory Usage

| Scale | Input Size | Output Size | RAM Usage |
|-------|-----------|-------------|-----------|
| 2x | 64×64 | 128×128 | ~500 MB |
| 4x | 32×32 | 128×128 | ~800 MB |
| 8x | 16×16 | 128×128 | ~1.2 GB |

---

## 🎓 Step-by-Step Tutorial

### Complete Workflow

```bash
# 1. First time setup
chmod +x *.sh
./setup.sh

# 2. Verify installation
./run_tests.sh

# 3. Run basic demo
./run_demo.sh

# 4. View results
open results/comparison.png

# 5. Try different scales
./run_scales.sh

# 6. Run all examples
./run_examples.sh

# 7. (Optional) Try GPU
./run_gpu.sh
```

---

## ❓ Troubleshooting

### Issue: "Permission denied"
```bash
# Solution: Make scripts executable
chmod +x *.sh
```

### Issue: "Python not found"
```bash
# Solution: Install Python 3.8+
# macOS: brew install python3
# Linux: sudo apt install python3
# Windows: Use WSL or download from python.org
```

### Issue: "Virtual environment not found"
```bash
# Solution: Run setup first
./setup.sh
```

### Issue: "CUDA out of memory"
```bash
# Solution: Use CPU or smaller scale
./run_demo.sh  # Uses CPU by default
# or
python super_resolution_demo.py --scale 2 --device cuda --save
```

### Issue: "Import errors"
```bash
# Solution: Reinstall dependencies
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🎯 Next Steps

After running the basic demo:

1. **Read the README.md** for detailed explanations
2. **Examine the code** in `super_resolution_demo.py`
3. **Modify parameters** to experiment
4. **Process your own images** (see README Section 6.1)
5. **Explore evaluation metrics** (see README Section 4)

---

## 📚 Additional Resources

- **Full Documentation**: See README.md
- **Code Examples**: See `run_example.py`
- **Evaluation Metrics**: See `eval_metrics.py`
- **Tests**: See `test_demo.py`

---

## 💡 Tips

### For Best Results:
1. Use GPU if available (`./run_gpu.sh`)
2. Start with 4x scale (standard)
3. Run tests first to verify setup
4. Check `results/` folder after each run

### For Learning:
1. Read the code in `super_resolution_demo.py`
2. Understand the network in `SimpleRBPN` class
3. Study evaluation metrics in `eval_metrics.py`
4. Experiment with different parameters

### For Development:
1. Modify `SimpleRBPN` architecture
2. Add new evaluation metrics
3. Create custom test images
4. Implement new training loops

---

**Need Help?** Check the full README.md for comprehensive documentation!

**Ready to Start?** Run `./setup.sh` now! 🚀
