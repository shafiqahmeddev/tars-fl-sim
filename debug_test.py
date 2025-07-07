#!/usr/bin/env python3
"""Debug test script to check basic functionality."""

import torch
import sys
import os

print("PyTorch version:", torch.__version__)
print("Python version:", sys.version)
print("Working directory:", os.getcwd())

# Test basic imports
try:
    from app.shared.data_utils import load_datasets
    print("✅ Data utils import successful")
except Exception as e:
    print("❌ Data utils import failed:", e)
    sys.exit(1)

try:
    from app.shared.models import MNIST_CNN
    print("✅ Models import successful")
except Exception as e:
    print("❌ Models import failed:", e)
    sys.exit(1)

# Test data loading
try:
    print("Loading datasets...")
    train_mnist, test_mnist, train_cifar, test_cifar = load_datasets()
    print(f"✅ Datasets loaded - MNIST train: {len(train_mnist)}, test: {len(test_mnist)}")
except Exception as e:
    print("❌ Dataset loading failed:", e)
    sys.exit(1)

# Test model creation
try:
    print("Creating model...")
    model = MNIST_CNN()
    print("✅ Model created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
except Exception as e:
    print("❌ Model creation failed:", e)
    sys.exit(1)

# Test basic forward pass
try:
    print("Testing forward pass...")
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"✅ Forward pass successful - output shape: {output.shape}")
except Exception as e:
    print("❌ Forward pass failed:", e)
    sys.exit(1)

print("🎉 All basic tests passed!")