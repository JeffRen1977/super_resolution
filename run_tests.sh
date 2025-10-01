#!/bin/bash
# Run test suite to verify installation
# This script runs all tests to ensure everything is working correctly

echo "🧪 Running Super Resolution Test Suite"
echo "======================================"
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
echo "🚀 Running tests..."
echo ""

python test_demo.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
    echo "The super resolution demo is working correctly."
    echo ""
    echo "Next steps:"
    echo "  - Run basic demo:        ./run_demo.sh"
    echo "  - Run all examples:      ./run_examples.sh"
    echo "  - Run with GPU:          ./run_gpu.sh"
    echo ""
else
    echo ""
    echo "❌ Tests failed. Please check the error messages above."
    echo "Make sure all dependencies are installed correctly."
    exit 1
fi

