#!/bin/bash
# Run all comprehensive examples
# This script runs all four example scenarios

echo "🎨 Running All Super Resolution Examples"
echo "========================================"
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
echo "🚀 Running comprehensive examples..."
echo "This will run:"
echo "  - Example 1: Basic Demo"
echo "  - Example 2: Custom Image Processing"
echo "  - Example 3: Different Scale Factors (2x, 4x, 8x)"
echo "  - Example 4: Temporal Consistency Testing"
echo ""

python run_example.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All examples completed successfully!"
    echo "📁 Results saved in:"
    echo "    - ./results/ (main results)"
    echo "    - ./results/temporal/ (temporal sequence)"
    echo ""
    echo "To view results:"
    echo "    open results/comparison.png"
    echo "    open results/temporal/"
    echo ""
else
    echo ""
    echo "❌ Examples failed. Please check the error messages above."
    exit 1
fi

