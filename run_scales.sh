#!/bin/bash
# Run Super Resolution with different scale factors
# This script tests 2x, 4x, and 8x upscaling

echo "📏 Testing Different Scale Factors"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "../virtual_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first to set up the environment."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ../virtual_env/bin/activate

echo ""
echo "Testing three different scale factors:"
echo "  - 2x upscaling (easier)"
echo "  - 4x upscaling (standard)"
echo "  - 8x upscaling (challenging)"
echo ""

# Create directory for scale comparisons
mkdir -p results/scale_comparison

# Test 2x scale
echo "🚀 Running 2x upscaling..."
python super_resolution_demo.py --scale 2 --device cpu --save
mv results/04_super_resolution.png results/scale_comparison/sr_2x.png 2>/dev/null
echo "✅ 2x complete"
echo ""

# Test 4x scale
echo "🚀 Running 4x upscaling..."
python super_resolution_demo.py --scale 4 --device cpu --save
mv results/04_super_resolution.png results/scale_comparison/sr_4x.png 2>/dev/null
echo "✅ 4x complete"
echo ""

# Test 8x scale
echo "🚀 Running 8x upscaling..."
python super_resolution_demo.py --scale 8 --device cpu --save
mv results/04_super_resolution.png results/scale_comparison/sr_8x.png 2>/dev/null
echo "✅ 8x complete"
echo ""

echo "✅ All scale tests completed!"
echo "📁 Results saved in:"
echo "    - ./results/scale_comparison/sr_2x.png"
echo "    - ./results/scale_comparison/sr_4x.png"
echo "    - ./results/scale_comparison/sr_8x.png"
echo ""
echo "Compare different scales:"
echo "    open results/scale_comparison/"
echo ""


