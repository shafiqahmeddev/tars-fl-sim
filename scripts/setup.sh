#!/bin/bash

# TARS Project Setup Script
# This script sets up the development environment for the TARS federated learning project

set -e  # Exit on any error

echo "🚀 Setting up TARS Federated Learning environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv is installed"

# Create virtual environment and install dependencies
echo "📦 Installing dependencies with uv..."
uv sync

echo "🔧 Virtual environment created in .venv/"

# Verify installation by checking if torch is available
echo "🧪 Verifying installation..."
uv run python -c "import torch; import torchvision; import numpy; import pandas; print('✅ All dependencies installed successfully')"

echo ""
echo "🎉 Setup complete! You can now run the TARS simulation with:"
echo "   uv run python main.py"
echo "   or"
echo "   ./scripts/run.sh"
echo ""
echo "💡 To activate the virtual environment manually:"
echo "   source .venv/bin/activate"