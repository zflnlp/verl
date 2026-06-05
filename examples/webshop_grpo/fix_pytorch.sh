#!/bin/bash
# Fix PyTorch installation for WebShop conda environment
#
# The issue: conda env uses Python 3.8, but system PyTorch is compiled for Python 3.10.
# This script installs PyTorch matching Python 3.8 + CUDA 12.4.
#
# Usage:
#   conda activate webshop
#   bash examples/webshop_grpo/fix_pytorch.sh

set -e

echo "=========================================="
echo "Fixing PyTorch for Python 3.8 environment"
echo "=========================================="

# Check current Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "Current Python: $PYTHON_VERSION"

# Uninstall existing torch (if any)
echo ""
echo "Removing existing torch..."
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Install PyTorch for Python 3.8 with CUDA 12.4
# Using the official PyTorch wheel index
echo ""
echo "Installing PyTorch for Python 3.8 + CUDA 12.4..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify installation
echo ""
echo "Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=========================================="
echo "PyTorch fix complete!"
echo "=========================================="
echo ""
echo "Now test WebShop environment:"
echo "  python -c \"import gym; from web_agent_site.envs import WebAgentTextEnv; print('OK')\""
