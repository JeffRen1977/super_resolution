# How to Run the Super Resolution Code

## 📋 Table of Contents
1. [Quick Start (3 Steps)](#quick-start-3-steps)
2. [Detailed Instructions](#detailed-instructions)
3. [All Available Scripts](#all-available-scripts)
4. [Python Command Reference](#python-command-reference)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start (3 Steps)

### Step 1: Setup
```bash
./setup.sh
```
**What it does:** Creates virtual environment and installs dependencies  
**Time:** 2-5 minutes  
**Run:** Only once

### Step 2: Test
```bash
./run_tests.sh
```
**What it does:** Verifies everything is working correctly  
**Time:** 10-15 seconds  
**Expected:** All tests pass ✅

### Step 3: Demo
```bash
./run_demo.sh
```
**What it does:** Runs super resolution demo and saves results  
**Time:** 5-10 seconds  
**Output:** `./results/` directory with images

**View Results:**
```bash
open results/comparison.png
```

---

## Detailed Instructions

### Initial Setup (First Time Only)

1. **Navigate to the project directory:**
   ```bash
   cd /path/to/super_resolution
   ```

2. **Make scripts executable:**
   ```bash
   chmod +x *.sh
   ```

3. **Run setup:**
   ```bash
   ./setup.sh
   ```
   
   This will:
   - Check Python 3 is installed
   - Create virtual environment in `./venv/`
   - Install all required packages:
     - PyTorch (deep learning framework)
     - OpenCV (image processing)
     - NumPy (numerical computing)
     - scikit-image (metrics)
     - Matplotlib (visualization)

4. **Verify installation:**
   ```bash
   ./run_tests.sh
   ```
   
   You should see:
   ```
   ✅ PyTorch imported successfully
   ✅ OpenCV imported successfully
   ✅ NumPy imported successfully
   ✅ scikit-image imported successfully
   ✅ Matplotlib imported successfully
   ✅ PSNR calculation: XX.XX dB
   ✅ SSIM calculation: 0.XXXX
   🎉 All tests passed!
   ```

### Running the Demo

**Basic Demo (Recommended for first run):**
```bash
./run_demo.sh
```

This runs:
- 4× upscaling (standard scale)
- CPU processing (works on all computers)
- Saves all results

**Output:**
```
🎬 Starting Super Resolution Demo...
Scale factor: 4x
Device: cpu

📸 Creating test image...
📉 Creating low-resolution frames...
🔄 Running bicubic baseline...
🚀 Running RBPN-style super resolution...
📊 Evaluating results...

==================================================
📈 RESULTS COMPARISON
==================================================
Bicubic PSNR: XX.XX dB
Bicubic SSIM: 0.XXXX
RBPN PSNR:    XX.XX dB
RBPN SSIM:    0.XXXX
💾 Results saved to 'results/' directory

✅ Demo completed successfully!
```

### Viewing Results

The demo creates several files in `./results/`:

```
results/
├── 01_ground_truth.png       # Original high-resolution image
├── 02_low_resolution.png     # Downsampled low-resolution input
├── 03_bicubic.png            # Bicubic interpolation (baseline)
├── 04_super_resolution.png   # RBPN super resolution result
└── comparison.png            # 2×2 grid showing all four images
```

**View results:**
```bash
# Open results folder
open results/

# Or view specific files
open results/comparison.png
open results/04_super_resolution.png
```

---

## All Available Scripts

### 1. `setup.sh` - Initial Setup
```bash
./setup.sh
```
- **Purpose:** Set up development environment
- **When:** Run once, before anything else
- **Time:** 2-5 minutes
- **Creates:** Virtual environment + installs dependencies

### 2. `run_tests.sh` - Test Suite
```bash
./run_tests.sh
```
- **Purpose:** Verify installation and functionality
- **When:** After setup, or when debugging issues
- **Time:** 10-15 seconds
- **Output:** Test results (should all pass)

### 3. `run_demo.sh` - Basic Demo
```bash
./run_demo.sh
```
- **Purpose:** Run standard demo (4× upscaling, CPU)
- **When:** First run, or quick testing
- **Time:** 5-10 seconds
- **Output:** Results in `./results/`

### 4. `run_examples.sh` - Comprehensive Examples
```bash
./run_examples.sh
```
- **Purpose:** Run all four example scenarios
- **When:** Want to see full capabilities
- **Time:** 30-60 seconds
- **Output:** Multiple result sets

**Runs:**
- Example 1: Basic demo workflow
- Example 2: Custom image processing
- Example 3: Scale factor comparison (2×, 4×, 8×)
- Example 4: Temporal consistency testing

### 5. `run_gpu.sh` - GPU Acceleration
```bash
./run_gpu.sh
```
- **Purpose:** Use GPU for faster processing
- **When:** Have NVIDIA GPU with CUDA
- **Time:** 2-5 seconds (5-10× faster)
- **Falls back:** To CPU if no GPU available

### 6. `run_scales.sh` - Scale Comparison
```bash
./run_scales.sh
```
- **Purpose:** Test different upscaling factors
- **When:** Want to compare 2×, 4×, and 8× results
- **Time:** 15-30 seconds
- **Output:** Three result images in `./results/scale_comparison/`

**Creates:**
```
results/scale_comparison/
├── sr_2x.png    # 2× upscaling (easiest)
├── sr_4x.png    # 4× upscaling (standard)
└── sr_8x.png    # 8× upscaling (hardest)
```

---

## Python Command Reference

If you prefer running Python commands directly:

### Activate Environment First
```bash
source venv/bin/activate
```

### Basic Commands

**1. Run basic demo:**
```bash
python super_resolution_demo.py --save
```

**2. Specify scale factor:**
```bash
python super_resolution_demo.py --scale 2 --save  # 2× upscaling
python super_resolution_demo.py --scale 4 --save  # 4× upscaling (default)
python super_resolution_demo.py --scale 8 --save  # 8× upscaling
```

**3. Use GPU:**
```bash
python super_resolution_demo.py --scale 4 --device cuda --save
```

**4. Run without saving:**
```bash
python super_resolution_demo.py --scale 4 --device cpu
```

**5. Run all examples:**
```bash
python run_example.py
```

**6. Run tests:**
```bash
python test_demo.py
```

**7. Evaluate results:**
```bash
python eval_metrics.py --pred_dir ./results/ --gt_dir ./ground_truth/
```

### Command-Line Options

#### `super_resolution_demo.py`

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--scale` | 2, 4, 8 | 4 | Upscaling factor |
| `--device` | cpu, cuda | cpu | Processing device |
| `--save` | flag | False | Save results to files |

**Examples:**

```bash
# Minimal (4× CPU, no save)
python super_resolution_demo.py

# Standard (4× CPU, save results)
python super_resolution_demo.py --save

# Easy task (2× CPU, save)
python super_resolution_demo.py --scale 2 --save

# Hard task (8× CPU, save)
python super_resolution_demo.py --scale 8 --save

# Fast processing (4× GPU, save)
python super_resolution_demo.py --scale 4 --device cuda --save

# Fast + hard (8× GPU, save)
python super_resolution_demo.py --scale 8 --device cuda --save
```

---

## Workflow Examples

### Example 1: Quick Test Run
```bash
# Setup (first time only)
./setup.sh

# Verify
./run_tests.sh

# Run demo
./run_demo.sh

# View results
open results/comparison.png
```

### Example 2: Complete Exploration
```bash
# Setup
./setup.sh

# Basic demo
./run_demo.sh

# All examples
./run_examples.sh

# Scale comparison
./run_scales.sh

# View everything
open results/
```

### Example 3: GPU-Accelerated
```bash
# Setup
./setup.sh

# Test GPU
./run_gpu.sh

# If GPU works, use it for examples
source venv/bin/activate
python super_resolution_demo.py --scale 4 --device cuda --save
python run_example.py
```

### Example 4: Development Workflow
```bash
# Activate environment
source venv/bin/activate

# Make changes to code
nano super_resolution_demo.py

# Test changes
python test_demo.py

# Run demo
python super_resolution_demo.py --save

# Check results
open results/
```

---

## Troubleshooting

### Issue 1: "Permission denied" when running scripts

**Error:**
```
bash: ./setup.sh: Permission denied
```

**Solution:**
```bash
chmod +x *.sh
```

### Issue 2: "Python not found"

**Error:**
```
python3: command not found
```

**Solution:**
```bash
# macOS
brew install python3

# Ubuntu/Debian
sudo apt install python3 python3-pip

# Windows
# Download from python.org or use WSL
```

### Issue 3: "Virtual environment not found"

**Error:**
```
❌ Virtual environment not found!
```

**Solution:**
```bash
# Run setup first
./setup.sh
```

### Issue 4: CUDA/GPU Issues

**Error:**
```
CUDA out of memory
```

**Solution:**
```bash
# Use CPU instead
python super_resolution_demo.py --scale 4 --device cpu --save

# Or use smaller scale
python super_resolution_demo.py --scale 2 --device cuda --save
```

**Error:**
```
CUDA not available
```

**Solution:**
```bash
# GPU not detected, use CPU
./run_demo.sh

# Or explicitly use CPU
python super_resolution_demo.py --device cpu --save
```

### Issue 5: Import Errors

**Error:**
```
ImportError: No module named 'torch'
```

**Solution:**
```bash
# Activate virtual environment first
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Then run again
python super_resolution_demo.py --save
```

### Issue 6: Slow Processing

**Problem:** Demo takes too long

**Solutions:**
```bash
# 1. Use GPU (if available)
./run_gpu.sh

# 2. Use smaller scale
python super_resolution_demo.py --scale 2 --save

# 3. Process smaller images (modify test image size in code)
```

### Issue 7: Out of Memory

**Error:**
```
RuntimeError: out of memory
```

**Solutions:**
```bash
# 1. Use CPU instead of GPU
python super_resolution_demo.py --device cpu --save

# 2. Use smaller scale
python super_resolution_demo.py --scale 2 --save

# 3. Close other applications
```

---

## Performance Expectations

### Processing Time

| Configuration | Time per Image | Use Case |
|--------------|----------------|----------|
| CPU, 2× scale | ~2 seconds | Quick testing |
| CPU, 4× scale | ~5 seconds | Standard demo |
| CPU, 8× scale | ~10 seconds | High resolution |
| GPU, 2× scale | ~0.1 seconds | Real-time |
| GPU, 4× scale | ~0.3 seconds | Fast processing |
| GPU, 8× scale | ~0.8 seconds | High-res fast |

### Memory Usage

| Scale | Input | Output | RAM | GPU RAM |
|-------|-------|--------|-----|---------|
| 2× | 64×64 | 128×128 | ~500 MB | ~200 MB |
| 4× | 32×32 | 128×128 | ~800 MB | ~400 MB |
| 8× | 16×16 | 128×128 | ~1.2 GB | ~600 MB |

---

## Tips and Best Practices

### Getting Started
1. ✅ Run `./setup.sh` first
2. ✅ Run `./run_tests.sh` to verify
3. ✅ Start with `./run_demo.sh`
4. ✅ View results in `./results/`

### For Best Performance
1. Use GPU if available (`./run_gpu.sh`)
2. Close unnecessary applications
3. Use appropriate scale (4× is standard)
4. Process images in batches

### For Learning
1. Read the code in `super_resolution_demo.py`
2. Modify parameters and observe changes
3. Compare different scales
4. Study evaluation metrics

### For Development
1. Keep virtual environment activated
2. Test changes with `python test_demo.py`
3. Use version control (git)
4. Document modifications

---

## Quick Reference

### Most Common Commands

```bash
# Setup (first time)
./setup.sh

# Run basic demo
./run_demo.sh

# Run all examples
./run_examples.sh

# Run tests
./run_tests.sh

# View results
open results/
```

### Most Useful Python Commands

```bash
# Activate environment
source venv/bin/activate

# Basic demo
python super_resolution_demo.py --save

# GPU demo
python super_resolution_demo.py --device cuda --save

# All examples
python run_example.py

# Tests
python test_demo.py
```

---

## Next Steps

After running the demos successfully:

1. **Understand the Code:** Read `super_resolution_demo.py`
2. **Read Documentation:** See README.md for theory
3. **Experiment:** Modify parameters and observe results
4. **Process Your Own Images:** See README Section 6
5. **Explore Advanced Topics:** Training, optimization, applications

---

## Getting Help

- **Full Documentation:** README.md
- **Quick Start:** QUICKSTART.md
- **Code Reference:** Comments in `.py` files
- **Examples:** `run_example.py`

**Have questions?** Check the comprehensive README.md for detailed explanations of theory, implementation, and advanced usage!

---

**Ready to start? Run `./setup.sh` now!** 🚀
