#!/bin/bash

# TARS Simulation Runner Script
# This script runs the TARS federated learning simulation

set -e  # Exit on any error

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   ./scripts/setup.sh"
    exit 1
fi

echo "🔥 Running TARS Federated Learning Simulation..."
echo "================================================"

# Run the simulation using uv
uv run python main.py

echo ""
echo "✅ Simulation completed!"
echo "📊 Results saved to simulation_results.csv"