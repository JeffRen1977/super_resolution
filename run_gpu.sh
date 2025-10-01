#!/bin/bash
# Run Super Resolution Demo with GPU acceleration
# This script runs the demo using CUDA GPU for faster processing

echo "🎮 Running Super Resolution Demo with GPU"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first to set up the environment."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo ""
echo "🔍 Checking for CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "🚀 Running demo with GPU acceleration (4x scale)..."
echo ""

python super_resolution_demo.py --scale 4 --device cuda --save

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ GPU demo completed successfully!"
    echo "📁 Results saved in ./results/ directory"
    echo ""
else
    echo ""
    echo "⚠️  GPU demo failed. Trying CPU fallback..."
    echo ""
    python super_resolution_demo.py --scale 4 --device cpu --save
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Demo completed successfully on CPU!"
        echo "Note: GPU was not available, used CPU instead."
    else
        echo ""
        echo "❌ Demo failed on both GPU and CPU."
        exit 1
    fi
fi

