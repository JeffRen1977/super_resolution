#!/bin/bash
# Run Chapter 22 Educational Learning Path
# This script runs the progressive educational modules for Chapter 22

echo "📚 Running Chapter 22: Super Resolution Learning Path"
echo "====================================================="
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

# Check if chapter22 directory exists
if [ ! -d "chapter22" ]; then
    echo "❌ Chapter 22 directory not found!"
    echo "Please make sure you're in the super_resolution directory."
    exit 1
fi

echo ""
echo "This will run all 6 educational sections:"
echo "  1. Understanding Image Degradation"
echo "  2. Traditional Interpolation Methods"
echo "  3. Single Image Super Resolution"
echo "  4. Video Super Resolution"
echo "  5. Recurrent Back-Projection Network"
echo "  6. Evaluation Metrics"
echo ""
echo "Press Ctrl+C to cancel, or wait 3 seconds to continue..."
sleep 3

echo ""
echo "🚀 Starting Chapter 22 learning path..."
echo ""

cd chapter22
python run_chapter22.py --all

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Chapter 22 learning path completed successfully!"
    echo "📁 Results saved in:"
    echo "    - ./chapter22/results/ (educational outputs)"
    echo ""
    echo "You have learned:"
    echo "  ✓ How images are degraded"
    echo "  ✓ Traditional upsampling methods"
    echo "  ✓ Deep learning for single image SR"
    echo "  ✓ Using temporal information in video SR"
    echo "  ✓ Advanced RBPN architecture"
    echo "  ✓ How to evaluate SR quality"
    echo ""
    echo "Next steps:"
    echo "  - Review the code in each section"
    echo "  - Experiment with parameters"
    echo "  - Try the main demo: ../run_demo.sh"
    echo ""
else
    echo ""
    echo "❌ Chapter 22 learning path failed."
    echo "Please check the error messages above."
    exit 1
fi

