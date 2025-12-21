#!/bin/bash
# Setup script for Super Resolution Demo
# This script sets up the virtual environment and installs all dependencies

echo "🚀 Super Resolution Demo - Setup Script"
echo "========================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo ""
echo "📦 Creating virtual environment in parent directory..."
python3 -m venv ../virtual_env

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment."
    exit 1
fi

echo "✅ Virtual environment created successfully in ../virtual_env!"
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ../virtual_env/bin/activate

echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

echo ""
echo "📥 Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies."
    exit 1
fi

echo ""
echo "✅ All dependencies installed successfully!"
echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "    source ../virtual_env/bin/activate"
echo ""
echo "To run the demo, use:"
echo "    ./run_demo.sh"
echo ""


