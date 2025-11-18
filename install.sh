#!/bin/bash
# Installation script for nano-PEARL
# This script ensures dependencies are installed in the correct order

set -e

echo "Installing nano-PEARL dependencies..."

# Step 1: Install torch first (CRITICAL for flash-attn)
echo "Step 1: Installing torch..."
pip install torch>=2.4.0 triton>=3.0.0

# Step 2: Install other dependencies (except flash-attn)
echo "Step 2: Installing other dependencies..."
pip install transformers>=4.51.0 xxhash rich>=14.1.0 numpy>=1.24.0 tqdm>=4.65.0

# Step 3: Install flash-attn (requires torch to be already installed)
echo "Step 3: Installing flash-attn (this may take a while)..."
pip install flash-attn || {
    echo "Warning: flash-attn installation failed. Trying with --no-build-isolation..."
    pip install flash-attn --no-build-isolation || {
        echo "Error: flash-attn installation failed. Please install manually."
        echo "Try: pip install flash-attn --no-build-isolation"
        exit 1
    }
}

# Step 4: Install optional GPU power monitoring (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Step 4: Installing GPU power monitoring (Linux)..."
    pip install nvidia-ml-py3>=11.0.0 || {
        echo "Warning: nvidia-ml-py3 installation failed. GPU power monitoring will be disabled."
    }
else
    echo "Step 4: Skipping GPU power monitoring (not on Linux)"
fi

echo "Installation complete!"

